"""Accuracy (WER) and latency of local STT engines on public English sets (SPEC §4.2).

Data: the Hugging Face Open ASR Leaderboard test sets (``hf-audio/open-asr-leaderboard``),
first parquet shard of each, sampled with a fixed stride so runs are reproducible:

* ``clean``  LibriSpeech test-clean (read audiobooks)
* ``other``  LibriSpeech test-other (harder read speech)
* ``ami``    AMI meeting corpus (spontaneous, overlapping, far-ish field: the closest public
             proxy for real dictation mistakes)
* ``earnings22``  earnings calls (accents, numbers, company names)

Engines are named ``<engine>[:<model>][@<device>]`` (device: cpu | cuda | auto):

* ``parakeet[:parakeet-tdt-0.6b-v3|parakeet-ultra|parakeet-tdt-0.6b-v2]``  onnx-asr int8
* ``whisper[:large-v3-turbo|distil-large-v3.5|small.en]``  faster-whisper
* ``sherpa:cohere`` / ``sherpa:qwen3``  sherpa-onnx int8 packages, extracted under
  ``models_dir()/sherpa`` (see docs/benchmarks.md for the archive URLs)
* ``tcpp:<file.gguf or absolute path>``  transcribe.cpp GGUF; needs ``TRANSCRIBE_LIBRARY``
  pointing at a shared build and its ``bindings/python/src`` on ``PYTHONPATH``
* ``onnxasr:<folder>``  any onnx-asr TDT folder under ``models_dir()`` (fp32 exports)

Each clip is transcribed whole (<= 30 s), timing only the transcribe call after a warm-up.
``--latency`` also times one fixed 10 s utterance 15 times (the SPEC's release-to-text
reference) and reports the median; use ``-n 0`` for a latency-only run.

    uv run python scripts/eval_stt.py --engines parakeet@cpu whisper:large-v3-turbo@cpu \
        --sets clean other ami -n 150 --out results.json
    uv run python scripts/eval_stt.py --engines parakeet:parakeet-ultra@cpu -n 0 --latency
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from openwhisprflow.config import SttConfig, data_dir, models_dir
from openwhisprflow.models import ensure_file

SETS = {
    "clean": ("librispeech", "test.clean"),
    "other": ("librispeech", "test.other"),
    "ami": ("ami", "test"),
    "earnings22": ("earnings22", "test"),
}
PARQUET = "https://huggingface.co/api/datasets/hf-audio/open-asr-leaderboard/parquet/{cfg}/{split}/0.parquet"


@dataclass
class Clip:
    id: str
    audio: np.ndarray
    reference: str

    @property
    def seconds(self) -> float:
        return len(self.audio) / 16000


@dataclass
class Run:
    engine: str
    dataset: str
    clips: int = 0
    wer: float = 0.0
    ref_words: int = 0
    audio_s: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    rtf: float = 0.0
    ms_per_10s: float = 0.0
    load_s: float = 0.0
    device: str = ""
    fwer: float | None = None      # formatted WER (case + . , ? ! as tokens); cased references only
    errors: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    hyps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- data
def load_set(name: str, n: int, cache: Path, max_s: float = 30.0) -> list[Clip]:
    import pyarrow.parquet as pq
    import soundfile as sf

    cfg, split = SETS[name]
    path = ensure_file(PARQUET.format(cfg=cfg, split=split), cache / f"{cfg}-{split}-0.parquet")
    table = pq.read_table(path, columns=["id", "text", "audio_length_s"])
    lengths = table.column("audio_length_s").to_pylist()
    texts = table.column("text").to_pylist()
    ok = [i for i, (s, t) in enumerate(zip(lengths, texts, strict=True)) if 1.0 <= s <= max_s and t and t.strip()]
    stride = max(1, len(ok) // n)
    rows = ok[::stride][:n]
    audio_col = pq.read_table(path, columns=["audio"]).column("audio")
    ids = table.column("id").to_pylist()
    clips = []
    for i in rows:
        data, rate = sf.read(io.BytesIO(audio_col[i].as_py()["bytes"]), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        if rate != 16000:
            from openwhisprflow.audio.resample import resample

            data = resample(data, rate, 16000)
        clips.append(Clip(ids[i], np.ascontiguousarray(data, dtype=np.float32), texts[i]))
    return clips


# ---------------------------------------------------------------------------- scoring
def normalizer():
    """Whisper's English normalizer (the leaderboard standard) when installed, else a basic one."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        n = EnglishTextNormalizer()
        return lambda s: " ".join(n(s).split())
    except ImportError:
        import re

        def basic(s: str) -> str:
            s = s.lower().replace("’", "'").replace("-", " ")
            return " ".join(re.sub(r"[^\w\s']", " ", s).split())
        return basic


def score(refs: list[str], hyps: list[str]) -> tuple[float, int]:
    import jiwer

    norm = normalizer()
    pairs = [(norm(r), norm(h)) for r, h in zip(refs, hyps, strict=True)]
    pairs = [(r, h) for r, h in pairs if r]
    out = jiwer.process_words([r for r, _ in pairs], [h for _, h in pairs])
    return out.wer, sum(len(r.split()) for r, _ in pairs)


_FMT_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*|[.,?!]")


def formatted_tokens(text: str) -> str:
    """Words with their case plus sentence punctuation as separate tokens: what a dictation
    user actually sees, which normalized WER deliberately ignores. Filler words are dropped on
    both sides: references keep them, and leaving them out is desirable in dictation."""
    return " ".join(t for t in _FMT_TOKEN.findall(text.replace("’", "'")) if t.lower() not in _FILLERS)


_FILLERS = {"um", "uh", "er", "ah", "eh", "hmm", "mm", "mhm", "uhm", "erm"}


def formatted_wer(refs: list[str], hyps: list[str]) -> float | None:
    import jiwer

    if not any(any(ch.isupper() for ch in r) and any(ch in ".,?!" for ch in r) for r in refs):
        return None  # LibriSpeech references are lowercase and unpunctuated
    pairs = [(formatted_tokens(r), formatted_tokens(h)) for r, h in zip(refs, hyps, strict=True)]
    pairs = [(r, h) for r, h in pairs if r]
    return jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


def summarize(run: Run, clips: list[Clip], hyps: list[str], times_ms: list[float]) -> Run:
    run.clips = len(clips)
    run.wer, run.ref_words = score([c.reference for c in clips], hyps)
    run.audio_s = sum(c.seconds for c in clips)
    run.median_ms = statistics.median(times_ms)
    run.p95_ms = sorted(times_ms)[min(len(times_ms) - 1, int(len(times_ms) * 0.95))]
    run.rtf = sum(times_ms) / 1000 / run.audio_s
    run.ms_per_10s = run.rtf * 10_000
    run.fwer = formatted_wer([c.reference for c in clips], hyps)
    run.samples = [{"id": c.id, "ref": c.reference, "hyp": h} for c, h in list(zip(clips, hyps, strict=True))[:5]]
    run.hyps = list(hyps)
    return run


# ---------------------------------------------------------------------------- engines
def parse(spec: str) -> tuple[str, str, str]:
    name, _, device = spec.partition("@")
    engine, _, model = name.partition(":")
    return engine, model, device or "auto"


class SherpaBackend:
    """Eval-only: sherpa-onnx offline recognizers for models onnx-asr cannot run (the same
    files a Rust build would load through sherpa-onnx's C API)."""

    PACKAGES = {
        "qwen3": "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        "cohere": "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
    }

    def __init__(self, model: str, device: str) -> None:
        import sherpa_onnx

        d = models_dir() / "sherpa" / self.PACKAGES[model]
        provider = "cuda" if device == "cuda" else "cpu"
        threads = max(1, (os.cpu_count() or 8) // 2)
        if model == "qwen3":
            self.rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=str(d / "conv_frontend.onnx"), encoder=str(d / "encoder.int8.onnx"),
                decoder=str(d / "decoder.int8.onnx"), tokenizer=str(d / "tokenizer"),
                num_threads=threads, provider=provider, max_new_tokens=256)
        else:
            self.rec = sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
                encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                tokens=str(d / "tokens.txt"), num_threads=threads, language="en", provider=provider)
        self.providers = [provider]

    def transcribe(self, audio: np.ndarray) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(16000, audio)
        self.rec.decode_stream(s)
        return s.result.text.strip()


class TcppBackend:
    """Eval-only: GGUF models through transcribe.cpp (the runtime behind the Rust
    ``transcribe-cpp`` crate that Handy ships). Needs the ``transcribe_cpp`` Python binding on
    the path and ``TRANSCRIBE_LIBRARY`` pointing at a shared build. ``model`` is a file name
    under ``models_dir()/gguf`` or an absolute path."""

    def __init__(self, model: str, device: str) -> None:
        import transcribe_cpp

        path = Path(model) if Path(model).is_absolute() else models_dir() / "gguf" / model
        backend = "cuda" if device == "cuda" else "cpu"
        self.model = transcribe_cpp.Model(path, backend=backend)
        self.session = self.model.session(n_threads=max(1, (os.cpu_count() or 8) // 2))
        self.providers = [f"transcribe.cpp/{backend}"]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.session.run(np.ascontiguousarray(audio, dtype=np.float32), timestamps="none").text.strip()


class OnnxAsrDirBackend:
    """Eval-only: any onnx-asr TDT folder (e.g. fp32 exports for GPU comparisons). ``model`` is a
    folder name under ``models_dir()`` or an absolute path; int8 is used when present."""

    def __init__(self, model: str, device: str) -> None:
        import onnx_asr
        import onnxruntime as ort

        from openwhisprflow.stt.parakeet import select_providers

        folder = Path(model) if Path(model).is_absolute() else models_dir() / model
        quant = "int8" if (folder / "encoder-model.int8.onnx").exists() else None
        providers = select_providers(device)
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        self.model = onnx_asr.load_model("nemo-conformer-tdt", folder, quantization=quant, providers=providers)
        self.providers = [f"{providers[0]} ({quant or 'fp32'})"]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.model.recognize(audio, sample_rate=16000).strip()


class RegistryBackend:
    def __init__(self, engine: str, model: str, device: str) -> None:
        from openwhisprflow.stt import registry

        self.stt = registry.create(SttConfig(engine=engine, model=model, device=device))
        self.stt.load()
        self.providers = getattr(self.stt, "providers", None) or [getattr(self.stt, "device", device)]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.stt.transcribe(audio, language="en").text


def run_engine(spec: str, datasets: dict[str, list[Clip]], lat_clip: np.ndarray | None) -> list[Run]:
    engine, model, device = parse(spec)
    t0 = time.perf_counter()
    backends = {"sherpa": SherpaBackend, "tcpp": TcppBackend, "onnxasr": OnnxAsrDirBackend}
    backend = backends[engine](model, device) if engine in backends else RegistryBackend(engine, model, device)
    load_s = round(time.perf_counter() - t0, 1)
    dev = str(backend.providers[0])
    warm = next(iter(datasets.values()))[0].audio if datasets else lat_clip
    for _ in range(2):
        backend.transcribe(warm)
    runs = []
    if lat_clip is not None:
        times = []
        for _ in range(LAT_REPEATS):
            t = time.perf_counter()
            backend.transcribe(lat_clip)
            times.append((time.perf_counter() - t) * 1000)
        runs.append(latency_run(spec, dev, load_s, times))
    for name, clips in datasets.items():
        hyps, times = [], []
        for c in clips:
            t = time.perf_counter()
            hyps.append(backend.transcribe(c.audio))
            times.append((time.perf_counter() - t) * 1000)
        run = summarize(Run(spec, name, load_s=load_s, device=dev), clips, hyps, times)
        print(fmt(run), flush=True)
        runs.append(run)
    return runs


LAT_REPEATS = 15


def latency_clip(cache: Path) -> np.ndarray:
    """Exactly 10 s of real speech (consecutive LibriSpeech test-clean clips), the SPEC's
    release-to-text reference utterance."""
    clips = load_set("clean", 40, cache)
    audio = np.concatenate([c.audio for c in clips])
    return np.ascontiguousarray(audio[:160_000])


def latency_run(spec: str, dev: str, load_s: float, times: list[float]) -> Run:
    times = sorted(times)
    run = Run(spec, "10s", clips=len(times), audio_s=10.0 * len(times), load_s=load_s, device=dev,
              median_ms=statistics.median(times), p95_ms=times[min(len(times) - 1, int(len(times) * 0.95))])
    run.rtf = run.median_ms / 10_000
    run.ms_per_10s = run.median_ms
    print(f"{spec:34s} 10s     median={run.median_ms:6.0f}ms p95={run.p95_ms:6.0f}ms  load={load_s}s [{dev}]",
          flush=True)
    return run


def fmt(r: Run) -> str:
    return (f"{r.engine:34s} {r.dataset:7s} n={r.clips:3d} WER={r.wer * 100:5.2f}%  "
            f"median={r.median_ms:6.0f}ms p95={r.p95_ms:6.0f}ms  RTF={r.rtf:.3f} "
            f"(~{r.ms_per_10s:.0f} ms per 10 s)  "
            + (f"fWER={r.fwer * 100:5.2f}%  " if r.fwer is not None else "") + f"[{r.device}]")


def machine() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "cpu": platform.processor(), "cores": os.cpu_count()}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        info["gpu"] = gpu
    except Exception:
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", nargs="+", default=["parakeet@cpu"])
    ap.add_argument("--sets", nargs="+", default=["clean", "other", "ami"], choices=list(SETS))
    ap.add_argument("-n", type=int, default=150, help="clips per set")
    ap.add_argument("--cache", type=Path, default=data_dir() / "eval-cache")
    ap.add_argument("--out", type=Path, help="append JSON results here")
    ap.add_argument("--latency", action="store_true", help="also time a fixed 10 s utterance (15 runs)")
    args = ap.parse_args()

    datasets = {s: load_set(s, args.n, args.cache) for s in args.sets} if args.n > 0 else {}
    lat = latency_clip(args.cache) if args.latency else None
    for s, clips in datasets.items():
        print(f"{s}: {len(clips)} clips, {sum(c.seconds for c in clips) / 60:.1f} min", flush=True)
    for spec in args.engines:
        try:
            runs = run_engine(spec, datasets, lat)
        except Exception as e:
            print(f"{spec}: FAILED {type(e).__name__}: {e}", flush=True)
            runs = [Run(spec, "-", errors=[f"{type(e).__name__}: {e}"])]
        if args.out:  # saved per engine so a long or interrupted run keeps what it measured
            prev = json.loads(args.out.read_text()) if args.out.exists() else {"runs": []}
            prev["machine"] = machine()
            prev["runs"] += [asdict(r) for r in runs]
            args.out.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()

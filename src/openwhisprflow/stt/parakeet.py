"""NVIDIA Parakeet TDT 0.6B through onnx-asr: the default local engine (SPEC §4.2).

Why Parakeet: a 600M transducer with built-in punctuation and casing, with better formatted
output than a Granite 470M CTC + separate punctuation model at a similar CPU cost (see docs/benchmarks.md).

Three int8 checkpoints share the architecture, tokenizer family and runtime, so they are
models of one engine rather than separate engines (``SttConfig.model``):

* ``parakeet-tdt-0.6b-v3``  NVIDIA v3, 25 European languages (istupakov's onnx-asr export)
* ``parakeet-ultra``        moondream's post-train of v3, same languages (Olicorne's export)
* ``parakeet-tdt-0.6b-v2``  NVIDIA v2, English only

Files are downloaded by us, not by huggingface_hub, so the UI gets byte-level progress and
every file is pinned to a repo revision and SHA-256. onnx-asr is then pointed at the local
folder (offline) through its generic ``nemo-conformer-tdt`` type.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openwhisprflow.config import SttConfig, models_dir
from openwhisprflow.models import ensure_file, hf_url
from openwhisprflow.stt.base import SAMPLE_RATE, EngineInfo, LoadProgress, SttError, SttResult

log = logging.getLogger("openwhisprflow.stt.parakeet")

ONNX_ASR_TYPE = "nemo-conformer-tdt"
QUANTIZATION = "int8"
_CONFIG = (97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466")
_VOCAB_V3 = (93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d")


@dataclass(frozen=True)
class Variant:
    repo: str
    revision: str
    # local name -> (path in repo, bytes, sha256). Pinned so a changed upstream file is never loaded.
    files: dict[str, tuple[str, int, str]]
    label: str
    languages: str

    @property
    def download_bytes(self) -> int:
        return sum(size for _, size, _ in self.files.values())


VARIANTS: dict[str, Variant] = {
    "parakeet-tdt-0.6b-v3": Variant(
        "istupakov/parakeet-tdt-0.6b-v3-onnx", "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce", {
            "config.json": ("config.json", *_CONFIG),
            "vocab.txt": ("vocab.txt", *_VOCAB_V3),
            "decoder_joint-model.int8.onnx": ("decoder_joint-model.int8.onnx", 18202004,
                                              "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70"),
            "encoder-model.int8.onnx": ("encoder-model.int8.onnx", 652183999,
                                        "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09"),
        }, "Parakeet TDT 0.6B v3", "en + 24 European languages (auto)"),
    "parakeet-ultra": Variant(
        "Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx", "3fd3b4d9772b2e595a9162f91f929f16bb4ab4cd", {
            "config.json": ("config.json", *_CONFIG),
            "vocab.txt": ("vocab.txt", *_VOCAB_V3),
            "decoder_joint-model.int8.onnx": ("int8/decoder_joint-model.int8.onnx", 18203490,
                                              "f7e2db395a3b738cb2893cfb853d25863cebcc5282583a2e86b5762559e0bd32"),
            "encoder-model.int8.onnx": ("int8/encoder-model.int8.onnx", 649537325,
                                        "8a2b47169cf3f1b114010e12c0221bc6f07203289477a65a44847b9e3f1e01ed"),
        }, "Parakeet Ultra (v3 post-train)", "en + 24 European languages (auto)"),
    "parakeet-tdt-0.6b-v2": Variant(
        "istupakov/parakeet-tdt-0.6b-v2-onnx", "0bbb45a3365852604aef28b538a8f066f4ccaa85", {
            "config.json": ("config.json", *_CONFIG),
            "vocab.txt": ("vocab.txt", 9384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
            "decoder_joint-model.int8.onnx": ("decoder_joint-model.int8.onnx", 8998286,
                                              "a449f49acd68979d418651dd2dcb737cc0f1bf0225e009e29ee326354edbf7d3"),
            "encoder-model.int8.onnx": ("encoder-model.int8.onnx", 652184014,
                                        "3e0581fda6ab843888b51e56d7ee78b6d5bc3237ec113af1f732d1d5286aa155"),
        }, "Parakeet TDT 0.6B v2 (English)", "en"),
}
DEFAULT_MODEL = "parakeet-ultra"  # measured best int8 (docs/benchmarks.md)

# Encoder attention cost grows with length; the segmenter never sends more than 30 s, and
# direct callers (file transcription) are split at pauses to stay under this.
MAX_SEGMENT_S = 30.0

INFO = EngineInfo(
    name="parakeet", kind="local", label="Parakeet TDT 0.6B (local)",
    models=list(VARIANTS), default_model=DEFAULT_MODEL,
    languages="en + 24 European languages (v2: English only)",
)

_PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def variant(model: str = "") -> Variant:
    try:
        return VARIANTS[model or DEFAULT_MODEL]
    except KeyError:
        raise SttError(f"unknown Parakeet model {model!r}; choose one of {', '.join(VARIANTS)}",
                       code="bad_model") from None


def model_dir(model: str = "") -> Path:
    return models_dir() / f"{model or DEFAULT_MODEL}-int8"


def is_downloaded(model: str = "") -> bool:
    d = model_dir(model)
    return all((d / name).is_file() and (d / name).stat().st_size == size
               for name, (_, size, _) in variant(model).files.items())


def select_providers(device: str = "auto", available: list[str] | None = None) -> list[str]:
    """ORT providers for ``device`` ("auto" picks the first accelerator present), CPU always last."""
    if available is None:
        import onnxruntime as ort

        available = ort.get_available_providers()
    wanted = ["cuda", "directml", "coreml"] if device == "auto" else [device]
    chosen = [_PROVIDERS[d] for d in wanted if d in _PROVIDERS and _PROVIDERS[d] in available]
    if device not in ("auto", "cpu") and not chosen:
        log.warning("device %r requested but %s is not available; using CPU", device, _PROVIDERS.get(device))
    return [p for p in chosen if p != "CPUExecutionProvider"] + ["CPUExecutionProvider"]


def is_speech_like(audio: np.ndarray) -> bool:
    """Cheap guard so pure silence never reaches the decoder."""
    return (len(audio) >= SAMPLE_RATE // 10 and float(np.max(np.abs(audio))) > 0.002
            and float(np.sqrt(np.mean(audio * audio))) > 0.0003)


class ParakeetEngine:
    name = "parakeet"
    kind = "local"
    sample_rate = SAMPLE_RATE

    def __init__(self, cfg: SttConfig) -> None:
        self.cfg = cfg
        self.model_name = cfg.model or DEFAULT_MODEL
        self.variant = variant(self.model_name)
        self.providers: list[str] = []
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self, progress: Callable[[LoadProgress], None] | None = None,
             cancel: Callable[[], bool] | None = None) -> None:
        with self._lock:
            if self._model is not None:
                return
            v, folder = self.variant, model_dir(self.model_name)
            done_before = 0
            for name, (remote, size, sha) in v.files.items():
                def report(item: str, done: int, total: int, base: int = done_before) -> None:
                    if progress:
                        progress(LoadProgress(file=item, done=base + done, total=v.download_bytes))
                try:
                    ensure_file(hf_url(v.repo, remote, v.revision), folder / name, sha256=sha, size=size,
                                progress=report, cancel=cancel)
                except Exception as e:
                    raise SttError(f"Parakeet model download failed: {e}", code="model_download",
                                   retryable=True) from e
                done_before += size
            self._model = self._open(folder)

    def _open(self, folder: Path) -> Any:
        import onnx_asr
        import onnxruntime as ort

        providers = select_providers(self.cfg.device)
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            try:  # onnxruntime-gpu >= 1.21: load CUDA/cuDNN from the nvidia-* wheels if present
                ort.preload_dlls()
            except Exception:
                log.debug("ort.preload_dlls failed", exc_info=True)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        last: Exception | None = None
        for attempt in (providers, ["CPUExecutionProvider"]):
            try:
                t0 = time.perf_counter()
                model = onnx_asr.load_model(ONNX_ASR_TYPE, folder, quantization=QUANTIZATION,
                                            sess_options=opts, providers=attempt)
                self.providers = attempt
                log.info("%s loaded in %.1fs on %s", self.model_name, time.perf_counter() - t0, attempt[0])
                return model
            except Exception as e:  # a broken GPU runtime must not take dictation down
                last = e
                log.warning("parakeet failed to load on %s: %s", attempt[0], e)
                if attempt == ["CPUExecutionProvider"]:
                    break
        raise SttError(f"Parakeet failed to load: {last}", code="model_load") from last

    def transcribe(self, audio: np.ndarray, *, language: str | None = None,
                   prompt: str | None = None) -> SttResult:
        """``language``/``prompt`` are accepted for the contract; Parakeet auto-detects language
        and has no prompt input (the dictionary applies through text/corrections)."""
        if self._model is None:
            self.load()
        t0 = time.perf_counter()
        pcm = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(pcm).all():
            raise SttError("audio contains NaN/inf", code="bad_audio")
        duration_ms = round(len(pcm) * 1000 / SAMPLE_RATE)
        parts: list[str] = []
        if is_speech_like(pcm):
            from openwhisprflow.audio.segmenter import split_at_pauses

            for chunk in split_at_pauses(pcm, SAMPLE_RATE, MAX_SEGMENT_S):
                if is_speech_like(chunk):
                    text = self._model.recognize(chunk, sample_rate=SAMPLE_RATE).strip()
                    if text:
                        parts.append(text)
        return SttResult(text=" ".join(parts), duration_ms=duration_ms,
                         processing_ms=round((time.perf_counter() - t0) * 1000), engine=self.name)

    def close(self) -> None:
        self._model = None


def create(cfg: SttConfig) -> ParakeetEngine:
    return ParakeetEngine(cfg)

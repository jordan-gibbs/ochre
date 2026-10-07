"""Render script `spoken` text to 16 kHz speech with many voices (dry: no room, no noise).

    .venv-data/Scripts/python tools/refine-data/tts.py --backend piper|kokoro|qwen|qwen_clone|sapi
        [--files W1 W2] [--limit N] [--per-file N] [--shard i/n] [--threads 4]

Writes ``training/refine/data/audio-v2/tts/<id>.flac`` and ``<id>.json`` (render log). Skips
clips already rendered. The plan (which backend/voice a script gets) is fixed by its id
(common.plan_for), so shards and re-runs are deterministic.

How `spoken` markers become audio:
* ``,``   a real pause: 150-450 ms of silence (scaled by the speaker's pause habit).
* ``...`` trailing off: the last 300 ms fade to ~35% and 0.5-1.1 s of silence follows.
* ``th-`` an abrupt cut: the word it starts (the next word that begins with the fragment, e.g.
  "think") is synthesized and chopped after len(fragment)/len(word) of its duration with a 4 ms
  edge, then 80-250 ms of silence. Without a completion, the fragment itself is spoken.
* um / uh / er / hmm: spoken slower as their own segment with short gaps around them (Piper,
  Kokoro, SAPI). Qwen3-TTS reads them in-line inside a phrase chunk, which sounds most natural.

Backends:
* piper      Piper VITS LibriTTS-R "high", 904 speakers (+30% SLERP blends of two), on CUDA.
             Checkpoint read in place from the livekit-wakeword data. Phonemes via espeak-ng.
* kokoro     Kokoro-82M v1.0 ONNX (CPU): 27 US/UK voices (+35% two-voice blends), plus 14
             non-English voices reading English (accented speech).
* qwen       Qwen3-TTS 12Hz 0.6B CustomVoice, 9 preset speakers (several non-native accents).
* qwen_clone Qwen3-TTS 0.6B Base cloning a reference voice ("clone-a", read in place, never
             copied). NOTE: that reference is a synthetic assistant voice, not the user's.
* sapi       Windows SAPI 5 (David, Zira): robotic, a small share for diversity.
"""

from __future__ import annotations

import nowmi  # noqa: F401  (must precede numpy/scipy)

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (CLONE_REF, CLONE_REF_TXT, DRY, LOGS, PIPER_CKPT, SR, TTS_MODELS,  # noqa: E402
                    TTS_STRIP_HESITATIONS, load_scripts, plan_for, rng_for, strip_spoken_hesitations)

HESITATION_WORDS = {"um", "umm", "uh", "uhh", "er", "erm", "hmm", "hm", "mm", "mmm", "ah", "ahh", "uhm", "mhm"}
# How each hesitation is written for the segmental TTS engines (they read it as a word). Chosen
# empirically: each spelling was synthesized between "so" and "we should go" with 4 voices and
# recognized by Parakeet; these are the spellings it most often heard as a hesitation (and so
# strip_hesitations removes), e.g. Kokoro "uhh" was heard as "ooh"/"though", Piper "hmm" as "him".
HES_KOKORO = {"um": "um", "umm": "um", "uhm": "um", "uh": "uh", "uhh": "uh", "ah": "uh", "ahh": "uh",
              "er": "er", "erm": "erm", "hmm": "hmm", "hm": "hmm", "mm": "hmm", "mmm": "hmm", "mhm": "hmm"}
HES_PIPER = {"um": "umm", "umm": "umm", "uhm": "umm", "uh": "uh", "uhh": "uh", "ah": "ah", "ahh": "ah",
             "er": "er", "erm": "erm", "hmm": "um", "hm": "um", "mm": "um", "mmm": "um", "mhm": "um"}
FRAG_FALLBACK = {"th": "the", "wh": "what", "sh": "should", "ch": "check", "st": "start", "tr": "try",
                 "pr": "probably", "gr": "great", "br": "bring", "cl": "close", "fr": "from", "pl": "please",
                 "w": "we", "t": "to", "s": "so", "b": "but", "m": "maybe", "d": "do", "c": "can", "y": "you",
                 "n": "no", "h": "how", "f": "for", "g": "go", "l": "let", "p": "put", "k": "keep"}


# ----------------------------------------------------------------------------- parsing


def parse_spoken(spoken: str) -> list[dict]:
    """-> units: {"kind": "words"|"hes"|"frag", "text", "after": None|"pause"|"trail"|"cut", ...}"""
    toks = spoken.replace("\u2026", "...").split()
    units: list[dict] = []
    words: list[str] = []

    def flush(after: str | None) -> None:
        if words:
            units.append({"kind": "words", "text": " ".join(words), "after": after})
            words.clear()
        elif after and units:
            units[-1]["after"] = units[-1]["after"] or after

    all_words = [re.sub(r"[^\w'@.-]", "", t).strip(".,-").lower() for t in toks]
    for i, tok in enumerate(toks):
        marker = None
        core = tok
        if core.endswith("..."):
            marker, core = "trail", core[:-3]
        while core.endswith((",", ";", ":")):
            marker, core = marker or "pause", core[:-1]
        if core.endswith("-") and len(core) > 1:
            frag = core.rstrip("-")
            flush(None)
            comp = next((w for w in all_words[i + 1:i + 6] if w.startswith(frag.lower()) and len(w) > len(frag)), None)
            units.append({"kind": "frag", "text": frag, "complete": comp, "after": marker or "cut"})
            continue
        if core in ("-", "--", ""):
            flush(marker or "pause")
            continue
        if core.lower().strip(".") in HESITATION_WORDS:
            flush(None)
            units.append({"kind": "hes", "text": core.lower().strip("."), "after": marker})
            continue
        words.append(core)
        if marker:
            flush(marker)
    flush(None)
    return units


def tts_text(s: str) -> str:
    """Light cosmetic casing for TTS input (pronunciation only): "i" -> "I"."""
    return re.sub(r"\bi\b", "I", re.sub(r"\bi'", "I'", s))


# ----------------------------------------------------------------------------- audio helpers


def to16k(a: np.ndarray, sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if sr == SR:
        return a
    g = gcd(SR, sr)
    return resample_poly(a, SR // g, sr // g).astype(np.float32)


def trim(a: np.ndarray, keep_s: float = 0.015) -> np.ndarray:
    if not len(a):
        return a
    env = np.abs(a)
    thr = 0.02 * float(env.max() or 1.0)
    hit = np.flatnonzero(env > thr)
    if not len(hit):
        return a[:0]
    k = int(keep_s * SR)
    return a[max(0, hit[0] - k): min(len(a), hit[-1] + k)]


def silence(s: float) -> np.ndarray:
    return np.zeros(max(0, int(s * SR)), dtype=np.float32)


def fade_tail(a: np.ndarray, s: float, to: float) -> np.ndarray:
    n = min(len(a), int(s * SR))
    if n:
        a = a.copy()
        a[-n:] *= np.linspace(1.0, to, n, dtype=np.float32)
    return a


def hard_edge(a: np.ndarray, ms: float = 4.0) -> np.ndarray:
    return fade_tail(a, ms / 1000, 0.0)


# ----------------------------------------------------------------------------- backends


class Piper:
    def __init__(self) -> None:
        import torch
        from livekit.wakeword.data.piper.synthesis import _load_vits_model

        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dev.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.2)  # shared GPU: stay small
        self.model = _load_vits_model(PIPER_CKPT, self.dev)
        cfg = json.loads(PIPER_CKPT.with_suffix(".json").read_text(encoding="utf-8"))
        self.idmap: dict[str, list[int]] = cfg["phoneme_id_map"]
        self.sr = int(cfg.get("audio", {}).get("sample_rate", 22050))
        self.espeak = shutil.which("espeak-ng") or r"C:\Program Files\eSpeak NG\espeak-ng.exe"
        self.cache: dict[str, str] = {}

    def ipa(self, text: str) -> str:
        if text not in self.cache:
            r = subprocess.run([self.espeak, "--ipa", "-q", "-v", "en-us", text], capture_output=True,
                               encoding="utf-8", check=True)
            self.cache[text] = " ".join(x.strip() for x in r.stdout.splitlines() if x.strip())
        return self.cache[text]

    def synth(self, text: str, voice: dict, slow: float = 1.0) -> np.ndarray:
        torch = self.torch
        from livekit.wakeword.data.piper.vits_utils import generate_path, sequence_mask, slerp

        ids = list(self.idmap["^"])
        for ch in unicodedata.normalize("NFD", self.ipa(text)):
            if ch in self.idmap:
                ids += self.idmap[ch] + self.idmap["_"]
        ids += self.idmap["$"]
        m = self.model
        with torch.no_grad():
            x = torch.LongTensor([ids]).to(self.dev)
            xl = torch.LongTensor([len(ids)]).to(self.dev)
            x_enc, m_p0, logs_p0, x_mask = m.enc_p(x, xl)
            e1 = m.emb_g(torch.LongTensor([voice["speaker"]]).to(self.dev))
            if "speaker2" in voice:
                e2 = m.emb_g(torch.LongTensor([voice["speaker2"]]).to(self.dev))
                g = slerp(e1, e2, float(voice["mix"])).unsqueeze(-1)
            else:
                g = e1.unsqueeze(-1)
            logw = m.dp(x_enc, x_mask, g=g, reverse=True, noise_scale=voice["noise_w"]) if m.use_sdp \
                else m.dp(x_enc, x_mask, g=g)
            w = torch.clamp(torch.ceil(torch.exp(logw) * x_mask * voice["length_scale"] * slow), max=40)
            y_len = torch.clamp_min(torch.sum(w, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(sequence_mask(y_len, int(y_len.max().item())), 1).type_as(x_mask)
            attn = generate_path(w, torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1))
            m_p = torch.matmul(attn.squeeze(1), m_p0.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p0.transpose(1, 2)).transpose(1, 2)
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * voice["noise"]
            z = m.flow(z_p, y_mask, g=g, reverse=True)
            audio = m.dec(z * y_mask, g=g).float().cpu().numpy().reshape(-1)
        return to16k(audio, self.sr)


class KokoroTTS:
    def __init__(self, threads: int = 4) -> None:
        try:  # .venv-kgpu: torch first, so ORT's CUDA EP finds cuBLAS/cuDNN from the torch wheel
            import torch  # noqa: F401
        except ImportError:
            pass
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        # HEURISTIC: every clip/segment has a new input length, and the default EXHAUSTIVE cuDNN
        # algorithm search re-runs per shape (measured: ~25 s per clip instead of < 1 s).
        providers = ([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
                     if gpu else ["CPUExecutionProvider"])
        sess = ort.InferenceSession(str(TTS_MODELS / "kokoro-v1.0.onnx"), so, providers=providers)
        print(f"[tts kokoro] providers {sess.get_providers()}", flush=True)
        self.k = Kokoro.from_session(sess, str(TTS_MODELS / "voices-v1.0.bin"))
        self.styles: dict[str, np.ndarray] = {}

    def style(self, voice: dict) -> np.ndarray:
        key = voice["id"]
        if key not in self.styles:
            a = self.k.get_voice_style(voice["voice"])
            if "voice2" in voice:
                b = self.k.get_voice_style(voice["voice2"])
                a = a * float(voice["mix"]) + b * (1 - float(voice["mix"]))
            self.styles[key] = a.astype(np.float32)
        return self.styles[key]

    def synth(self, text: str, voice: dict, slow: float = 1.0) -> np.ndarray:
        speed = max(0.5, min(2.0, voice["speed"] / slow))
        a, sr = self.k.create(text, voice=self.style(voice), speed=speed, lang=voice["lang"], trim=False)
        return to16k(a, sr)


class Qwen:
    """Qwen3-TTS; whole phrase chunks in one call (natural prosody for fillers and commas)."""

    chunked = True

    def __init__(self, clone: bool) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" if clone else "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        self.clone = clone
        self.model = FasterQwen3TTS.from_pretrained(name, device="cuda", dtype=torch.bfloat16, local_files_only=True)
        if clone:
            if not (CLONE_REF.is_file() and CLONE_REF_TXT.is_file()):
                raise SystemExit(f"clone reference missing: {CLONE_REF}")
            self.ref_text = " ".join(CLONE_REF_TXT.read_text(encoding="utf-8").split())

    def synth(self, text: str, voice: dict, slow: float = 1.0) -> np.ndarray:
        n_words = len(text.split())
        max_tokens = int(min(2048, 60 + n_words * 12))  # 12 Hz codec: ~4 tokens per word + slack
        if self.clone:
            # str path: faster-qwen3-tts reads the file in place (never copied anywhere)
            out, sr = self.model.generate_voice_clone(text=text, language="English", ref_audio=str(CLONE_REF),
                                                      ref_text=self.ref_text, max_new_tokens=max_tokens)
        else:
            out, sr = self.model.generate_custom_voice(text=text, speaker=voice["speaker"], language="English",
                                                       max_new_tokens=max_tokens)
        return to16k(np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in out]), sr)


class Sapi:
    def __init__(self) -> None:
        import win32com.client as w

        self.w = w
        self.voice = w.Dispatch("SAPI.SpVoice")
        self.tokens = {t.GetDescription(): t for t in self.voice.GetVoices()}
        self.tmp = LOGS / f"sapi-{os.getpid()}.wav"
        self.tmp.parent.mkdir(parents=True, exist_ok=True)

    def synth(self, text: str, voice: dict, slow: float = 1.0) -> np.ndarray:
        import soundfile as sf

        tok = next(t for d, t in self.tokens.items() if d.startswith(voice["voice"]))
        stream = self.w.Dispatch("SAPI.SpFileStream")
        fmt = self.w.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = 18  # SAFT16kHz16BitMono
        stream.Format = fmt
        stream.Open(str(self.tmp), 3)
        self.voice.AudioOutputStream = stream
        self.voice.Voice = tok
        self.voice.Rate = int(max(-10, min(10, voice["rate"] - (3 if slow > 1.2 else 0))))
        self.voice.Speak(text)
        stream.Close()
        a, sr = sf.read(str(self.tmp), dtype="float32")
        return to16k(a, sr)


def load_backend(name: str, threads: int):
    if name == "piper":
        return Piper()
    if name == "kokoro":
        return KokoroTTS(threads)
    if name in ("qwen", "qwen_clone"):
        return Qwen(clone=name == "qwen_clone")
    if name == "sapi":
        return Sapi()
    raise SystemExit(f"unknown backend {name}")


# ----------------------------------------------------------------------------- rendering


def render(engine, sid: str, spoken: str, voice: dict) -> tuple[np.ndarray, dict]:
    rng = rng_for(sid, "render")
    ps = float(voice.get("pause_scale", 1.0))
    units = parse_spoken(spoken)
    pieces: list[np.ndarray] = []
    log: list[dict] = []
    chunked = getattr(engine, "chunked", False)

    def gap(after: str | None, last: bool) -> np.ndarray:
        if after == "pause":
            return silence(rng.uniform(0.15, 0.45) * ps)
        if after == "trail":
            return silence(0.0 if last else rng.uniform(0.5, 1.1) * ps)
        if after == "cut":
            return silence(rng.uniform(0.08, 0.25) * ps)
        return silence(rng.uniform(0.02, 0.09))

    # Group units into synthesis calls.
    calls: list[dict] = []
    if chunked:
        buf: list[dict] = []

        def close_chunk() -> None:
            if not buf:
                return
            parts = []
            for u in buf:
                t = u["text"]
                if u["after"] == "pause" and u is not buf[-1]:
                    t += ","
                elif u["after"] == "trail" and u is not buf[-1]:
                    t += "..."
                parts.append(t)
            text = " ".join(parts)
            last = buf[-1]["after"]
            calls.append({"kind": "chunk", "text": text, "after": last,
                          "trail": last == "trail"})
            buf.clear()

        for u in units:
            if u["kind"] == "frag":
                close_chunk()
                calls.append(u)
                continue
            buf.append(u)
            n = sum(len(x["text"].split()) for x in buf)
            if (u["after"] in ("pause", "trail") and n >= 10) or n >= 28:
                close_chunk()
        close_chunk()
    else:
        calls = units

    for i, u in enumerate(calls):
        last = i == len(calls) - 1
        kind = u["kind"]
        if kind == "frag":
            word = u.get("complete")
            frag = u["text"]
            if word:
                full = trim(engine.synth(tts_text(word), voice))
                frac = min(0.75, max(0.25, len(frag) / len(word)))
                a = hard_edge(full[: int(len(full) * frac)])
                how = f"'{word}' chopped at {frac:.2f}"
            else:
                say = frag if (len(frag) >= 2 and re.search(r"[aeiouy]", frag)) else FRAG_FALLBACK.get(frag.lower(), frag + "uh")
                full = trim(engine.synth(tts_text(say), voice))
                a = hard_edge(full[: int(len(full) * (0.85 if say == frag else 0.5))])
                how = f"'{say}' chopped"
            pieces += [a, gap("cut" if u["after"] == "cut" else u["after"], last)]
            log.append({"frag": frag, "how": how})
            continue
        if kind == "hes":
            if rng.random() < 0.5:
                pieces.append(silence(rng.uniform(0.05, 0.25) * ps))
            hes_map = HES_KOKORO if isinstance(engine, KokoroTTS) else HES_PIPER
            a = trim(engine.synth(hes_map.get(u["text"], u["text"]), voice, slow=rng.uniform(1.0, 1.3)))
            pieces += [a, gap(u["after"], last) if u["after"] else silence(rng.uniform(0.1, 0.35) * ps)]
            log.append({"hes": u["text"]})
            continue
        text = tts_text(u["text"])
        a = trim(engine.synth(text, voice))
        if u["after"] == "trail":
            a = fade_tail(a, 0.3, rng.uniform(0.25, 0.45))
        pieces += [a, gap(u["after"], last)]
        log.append({"say": text, "after": u["after"], "s": round(len(a) / SR, 2)})
    audio = np.concatenate(pieces) if pieces else np.zeros(SR // 10, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak * 0.9).astype(np.float32)  # level is set later by augmentation
    return audio, {"units": log, "seconds": round(len(audio) / SR, 2)}


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-file", type=int)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reverse", action="store_true", help="work from the end (a helper worker on the same shard)")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    rows = load_scripts(a.files, a.limit, a.per_file)
    todo = []
    for r in rows:
        p = plan_for(r)
        if p["voice"]["backend"] != a.backend:
            continue
        if int(p["id"].encode().hex(), 16) % n != k:
            continue
        if (DRY / f"{r['id']}.json").exists():  # json marks a finished clip
            continue
        todo.append((r, p))
    print(f"[tts {a.backend} {a.shard}] {len(todo)} clips to render", flush=True)
    if not todo:
        return
    DRY.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = load_backend(a.backend, a.threads)
    print(f"[tts {a.backend}] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    t0, audio_s = time.perf_counter(), 0.0
    if a.reverse:
        todo.reverse()
    for i, (r, p) in enumerate(todo, 1):
        if (DRY / f"{r['id']}.json").exists():  # another worker got there first
            continue
        t1 = time.perf_counter()
        try:
            text = r.get("tts_text") or (strip_spoken_hesitations(r["spoken"]) if TTS_STRIP_HESITATIONS else r["spoken"])
            audio, meta = render(engine, r["id"], text, p["voice"])
            meta["tts_text"] = text
        except Exception as e:  # noqa: BLE001 - one bad script must not stop the run
            print(f"[tts {a.backend}] {r['id']} FAILED: {e!r}", flush=True)
            continue
        meta.update({"id": r["id"], "voice": p["voice"], "render_s": round(time.perf_counter() - t1, 2)})
        sf.write(str(DRY / f"{r['id']}.flac"), audio, SR, subtype="PCM_16")
        (DRY / f"{r['id']}.json").write_text(json.dumps(meta), encoding="utf-8")
        audio_s += meta["seconds"]
        if i % 20 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            print(f"[tts {a.backend} {a.shard}] {i}/{len(todo)} clips, {audio_s / 60:.1f} min audio in "
                  f"{el / 60:.1f} min ({audio_s / max(el, 1e-6):.1f}x realtime)", flush=True)


if __name__ == "__main__":
    main()

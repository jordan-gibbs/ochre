"""Recognition the way the app does it.

* Phrase cutting: a port of the Rust ``PhraseCutter`` (crates/ochre-audio/src/segmenter.rs) with
  ``CutConfig::default()`` (2 s + 200 ms pause, 8 s + 100 ms, hard cap 20 s), energy gate only
  (the app wires no Silero gate into the segmenter). Each phrase is decoded on its own and the
  texts are joined with one space, so Parakeet's per-phrase casing and full stops ("Hi all.
  Quick update...") show up exactly as they do in the app.
* Parakeet TDT 0.6B v3 int8 via onnx-asr on CPU: the same int8 export, revision and files the
  app downloads (``%LOCALAPPDATA%/openwhisprflow/models/parakeet-tdt-0.6b-v3-int8``).
* Whisper large-v3-turbo via faster-whisper on CUDA (float16), greedy, no context, no
  timestamps, language auto, dictionary as ``initial_prompt`` "w1, w2." (the Rust engine's
  settings, crates/ochre-stt/src/whisper.rs; the app runs whisper.cpp q8_0, we run CTranslate2).
* Then ``owf_text.app_prepass`` (strip_hesitations + dictionary corrections), byte-identical to
  the Rust pre-pass (see parity.py).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from common import SR
from owf_text import app_prepass

# ----------------------------------------------------------------------------- phrase cutter

FRAME_S = 0.02
QUIET_CAP = 0.002
QUIET_RELATIVE = 0.03
MIN_SPEECH_FRAMES = 3
CUT = dict(min_phrase_s=2.0, pause_s=0.2, soft_phrase_s=8.0, soft_pause_s=0.10, max_phrase_s=20.0,
           cut_window_s=3.0, release_window_s=1.0)


class PhraseCutter:
    """Line-for-line port of the Rust cutter (energy gate, f32 math as closely as numpy allows)."""

    def __init__(self, rate: int = SR, cfg: dict | None = None) -> None:
        self.c = dict(cfg or CUT)
        self.rate = rate
        self.width = max(1, round(rate * FRAME_S))
        self.buf = np.zeros(0, dtype=np.float32)
        self.pos = 0
        self.levels: list[float] = []
        self.quiet_flags: list[bool] = []
        self.quiet = 0
        self.peak_rms = np.float32(0.0)

    def samples(self, s: float) -> int:
        return int(np.float32(self.rate) * np.float32(s))

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        self.buf = np.concatenate([self.buf, np.asarray(block, dtype=np.float32)])
        out: list[np.ndarray] = []
        c, w = self.c, self.width
        n_full = (len(self.buf) - self.pos) // w
        if n_full <= 0:
            return out
        frames = self.buf[self.pos:self.pos + n_full * w].reshape(n_full, w)
        rms_all = np.sqrt((frames * frames).sum(axis=1, dtype=np.float32) / np.float32(w)).astype(np.float32)
        peak_all = np.abs(frames).max(axis=1)
        for k in range(n_full):
            rms, peak = rms_all[k], peak_all[k]
            self.pos += w
            self.levels.append(float(rms))
            self.peak_rms = max(self.peak_rms, rms)
            threshold = min(np.float32(QUIET_CAP), self.peak_rms * np.float32(QUIET_RELATIVE))
            quiet = bool(rms <= threshold and peak <= max(threshold * np.float32(4.0), np.float32(1e-6)))
            self.quiet_flags.append(quiet)
            self.quiet = self.quiet + w if quiet else 0
            length = self.pos
            if ((length >= self.samples(c["min_phrase_s"]) and self.quiet >= self.samples(c["pause_s"]))
                    or (length >= self.samples(c["soft_phrase_s"]) and self.quiet >= self.samples(c["soft_pause_s"]))):
                p = self._take_speech(len(self.levels))
                if p is not None:
                    out.append(p)
            elif length >= self.samples(c["max_phrase_s"]):
                window = min(len(self.levels), max(1, round(c["cut_window_s"] / FRAME_S)))
                p = self._take_speech(self._quietest_in_last(window) + 1)
                if p is not None:
                    out.append(p)
        return out

    def _quietest_in_last(self, window: int) -> int:
        start = len(self.levels) - min(window, len(self.levels))
        best = start
        for i in range(start, len(self.levels)):
            if self.levels[i] < self.levels[best]:
                best = i
        return best

    def _take_speech(self, count: int) -> np.ndarray | None:
        loud = sum(1 for q in self.quiet_flags[:count] if not q)
        p = self._take(count)
        return p if loud >= MIN_SPEECH_FRAMES else None

    def _take(self, count: int) -> np.ndarray:
        n = count * self.width
        phrase = self.buf[:n].copy()
        self.buf = self.buf[n:]
        self.pos -= n
        del self.levels[:count]
        del self.quiet_flags[:count]
        self.quiet = 0
        return phrase

    def pending_has_speech(self) -> bool:
        loud = sum(1 for q in self.quiet_flags if not q)
        if loud >= MIN_SPEECH_FRAMES:
            return True
        carry = self.buf[self.pos:]
        threshold = max(min(np.float32(QUIET_CAP), self.peak_rms * np.float32(QUIET_RELATIVE)), np.float32(1e-6))
        return loud + int(bool(np.any(np.abs(carry) > threshold * 4.0))) >= MIN_SPEECH_FRAMES

    def flush(self) -> np.ndarray | None:
        self.levels, self.quiet_flags, self.pos, self.quiet = [], [], 0, 0
        if not len(self.buf):
            return None
        b, self.buf = self.buf, np.zeros(0, dtype=np.float32)
        return b

    def flush_speech(self) -> np.ndarray | None:
        speech = self.pending_has_speech()
        p = self.flush()
        return p if speech else None

    def release_cut(self) -> np.ndarray | None:
        if not self.levels:
            return self.flush_speech()
        window = min(len(self.levels), max(1, round(self.c["release_window_s"] / FRAME_S)))
        start = len(self.levels) - window
        last_quiet = next((i for i in range(len(self.levels) - 1, start - 1, -1) if self.quiet_flags[i]), None)
        if last_quiet is not None and last_quiet + 1 == len(self.levels):
            return self.flush_speech()
        cut = last_quiet if last_quiet is not None else self._quietest_in_last(window)
        return self._take_speech(cut + 1)


def phrases_like_app(audio: np.ndarray, block_s: float = 0.02) -> list[np.ndarray]:
    """Feed the clip as the mic would, release at the end, finish (flush what has speech)."""
    c = PhraseCutter(SR)
    out = c.push(audio)
    p = c.release_cut()
    if p is not None:
        out.append(p)
    if c.pending_has_speech():
        rest = c.flush()
        if rest is not None:
            out.append(rest)
    return out


def is_speech_like(audio: np.ndarray) -> bool:
    return (len(audio) >= SR // 10 and float(np.max(np.abs(audio))) > 0.002
            and float(np.sqrt(np.mean(audio * audio))) > 0.0003)


# ----------------------------------------------------------------------------- engines

PARAKEET_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "openwhisprflow" / "models" / "parakeet-tdt-0.6b-v3-int8"
WHISPER_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "openwhisprflow" / "models" / "whisper" / "large-v3-turbo"
MAX_SEGMENT_S = 30.0


class Parakeet:
    name = "parakeet-tdt-0.6b-v3-int8"

    def __init__(self, threads: int = 4) -> None:
        import onnx_asr
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.model = onnx_asr.load_model("nemo-conformer-tdt", str(PARAKEET_DIR), quantization="int8",
                                         sess_options=opts, providers=["CPUExecutionProvider"])

    def phrase(self, pcm: np.ndarray, dictionary: list[str]) -> str:
        if not is_speech_like(pcm):
            return ""
        from openwhisprflow.audio.segmenter import split_at_pauses  # engine-level 30 s guard

        parts = []
        for chunk in split_at_pauses(pcm, SR, MAX_SEGMENT_S):
            if is_speech_like(chunk):
                t = self.model.recognize(chunk, sample_rate=SR).strip()
                if t:
                    parts.append(t)
        return " ".join(parts)


class Whisper:
    name = "whisper-large-v3-turbo"

    def __init__(self) -> None:
        import torch  # noqa: F401  (loads cuBLAS/cuDNN DLLs from the torch wheel for CTranslate2)

        lib = Path(torch.__file__).parent / "lib"
        if hasattr(os, "add_dll_directory") and lib.is_dir():
            os.add_dll_directory(str(lib))
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(WHISPER_DIR), device="cuda", compute_type="float16")

    def phrase(self, pcm: np.ndarray, dictionary: list[str]) -> str:
        if not is_speech_like(pcm):
            return ""
        from openwhisprflow.audio.segmenter import split_at_pauses

        prompt = f"{', '.join(dictionary)}." if dictionary else None
        parts = []
        for chunk in split_at_pauses(pcm, SR, MAX_SEGMENT_S):
            if len(chunk) < SR + SR // 10:  # whisper.cpp pads short phrases to 1.1 s
                chunk = np.concatenate([chunk, np.zeros(SR + SR // 10 - len(chunk), dtype=np.float32)])
            segs, _ = self.model.transcribe(chunk, language=None, initial_prompt=prompt, beam_size=1,
                                            best_of=1, temperature=0.0, condition_on_previous_text=False,
                                            without_timestamps=True, vad_filter=False)
            t = "".join(s.text for s in segs).strip()
            if t:
                parts.append(t)
        return " ".join(parts)


def recognize(engine, audio: np.ndarray, dictionary: list[str]) -> dict:
    t0 = time.perf_counter()
    phrases = phrases_like_app(audio)
    texts = [engine.phrase(p, dictionary) for p in phrases]
    joined = " ".join(t.strip() for t in texts if t.strip())
    return {"raw_asr": joined, "raw": app_prepass(joined, dictionary), "phrases": len(phrases),
            "phrase_texts": texts, "asr_ms": round((time.perf_counter() - t0) * 1000)}

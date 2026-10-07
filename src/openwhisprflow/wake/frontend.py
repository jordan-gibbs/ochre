"""Streaming openWakeWord feature front end (mel spectrogram + speech embedding) in plain onnxruntime.

Every wake head we ship or train sits on the same frozen front end: ``melspectrogram.onnx`` turns
audio into 32-bin log-mel frames, and ``embedding_model.onnx`` turns each 76-frame mel window into
one 96-d embedding every 80 ms. A head then scores the last 16 embeddings (~1.3 s of context).

Why not the ``openwakeword`` package at runtime: its streaming buffer copies a 10 s deque on every
frame, it drags in extra deps (scipy, sklearn, tflite on some platforms), and livekit-style heads
with dynamic output shapes are not guaranteed to load in ``openwakeword.Model``. The math here
mirrors ``openwakeword.utils.AudioFeatures._streaming_features`` exactly for 1280-sample frames
(see tests/test_wake_frontend.py).

Audio contract: 16 kHz mono at **int16 scale** (float values in [-32768, 32767] are fine). Training
(``training/wakeword``) extracts features the same way; feeding [-1, 1] floats shifts the log-mel by
~9 units and silently ruins detection, so :func:`to_int16_scale` is applied to every input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("openwhisprflow.wake.frontend")

SAMPLE_RATE = 16_000
FRAME = 1280            # 80 ms @ 16 kHz: one embedding per frame
MEL_CONTEXT = 480       # 3 extra 10 ms hops of left context, as openWakeWord does
MEL_WINDOW = 76         # mel frames per embedding window
MEL_BINS = 32
EMBED_DIM = 96
_KEEP_EMBEDDINGS = 64   # heads look at 16 frames; keep a little headroom for wider heads


@dataclass(frozen=True)
class FrontendFile:
    name: str
    url: str
    sha256: str
    size: int


# openWakeWord v0.5.1 release assets (Apache-2.0). Byte-identical to the copies inside the
# openwakeword / livekit-wakeword packages that training uses (hashes checked 2026-10-04).
_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
MEL_MODEL = FrontendFile(
    "melspectrogram.onnx", f"{_RELEASE}/melspectrogram.onnx",
    "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f", 1_087_958)
EMBED_MODEL = FrontendFile(
    "embedding_model.onnx", f"{_RELEASE}/embedding_model.onnx",
    "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f", 1_326_578)


def frontend_dir() -> Path:
    from openwhisprflow.config import models_dir

    return models_dir() / "openwakeword"


def ensure_frontend_models(dest: Path | None = None, progress: Any = None) -> tuple[Path, Path]:
    """Download (once, SHA-256 verified) the two front-end models; returns (mel, embedding) paths."""
    from openwhisprflow.models import ensure_file

    d = Path(dest) if dest else frontend_dir()
    return tuple(  # type: ignore[return-value]
        ensure_file(f.url, d / f.name, sha256=f.sha256, size=f.size, progress=progress)
        for f in (MEL_MODEL, EMBED_MODEL))


def make_session(path: str | Path, provider: str = "cpu") -> Any:
    """Single-threaded, non-spinning ORT session.

    These models are tiny; ORT's default spinning thread pool would burn a core per session even
    while idle, which matters for something that listens all day.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def to_int16_scale(x: np.ndarray) -> np.ndarray:
    """int16 -> as is; float in [-1, 1] -> scaled by 32767. Returns float32 at int16 scale.

    A float block whose peak is already > 1.5 is assumed to be int16-scale floats.
    """
    if x.dtype == np.int16:
        return x.astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.size and float(np.max(np.abs(x))) > 1.5:
        return x
    return x * 32767.0


class WakeFrontend:
    """Streaming features: ``push(audio)`` -> number of new embeddings; ``window(n)`` -> (1, n, 96)."""

    def __init__(self, mel_path: str | Path | None = None, embed_path: str | Path | None = None,
                 provider: str = "cpu") -> None:
        if mel_path is None or embed_path is None:
            mel_path, embed_path = ensure_frontend_models()
        self.mel = make_session(mel_path, provider)
        self.emb = make_session(embed_path, provider)
        self.frames = 0  # embeddings produced since construction (not reset), for profiling
        self.reset()

    def reset(self) -> None:
        """Back to the state openWakeWord starts from (ones mel buffer, embeddings of weak noise)."""
        self._raw = np.zeros(FRAME + MEL_CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._mel_buf = np.ones((MEL_WINDOW, MEL_BINS), dtype=np.float32)
        rng = np.random.default_rng(0)
        warm = rng.integers(-1000, 1000, SAMPLE_RATE * 4).astype(np.float32)
        self.features = self.embed_clip(warm)[-_KEEP_EMBEDDINGS:]

    # ------------------------------------------------------------------ model calls
    def _melspec(self, x: np.ndarray) -> np.ndarray:
        out = self.mel.run(None, {"input": x.astype(np.float32, copy=False)[None, :]})[0]
        return np.squeeze(out).reshape(-1, MEL_BINS) / 10.0 + 2.0

    def _embed(self, windows: np.ndarray) -> np.ndarray:
        out = self.emb.run(None, {"input_1": windows.astype(np.float32, copy=False)})[0]
        return out.reshape(-1, EMBED_DIM)

    def embed_clip(self, audio: np.ndarray) -> np.ndarray:
        """Whole clip (int16 scale) -> (n, 96) embeddings, one per 80 ms (batch path, for tests/tools)."""
        spec = self._melspec(to_int16_scale(audio))
        wins = [spec[i:i + MEL_WINDOW] for i in range(0, spec.shape[0] - MEL_WINDOW + 1, 8)]
        if not wins:
            return np.zeros((0, EMBED_DIM), np.float32)
        return self._embed(np.stack(wins)[..., None])

    # ------------------------------------------------------------------ streaming
    def push(self, audio: np.ndarray) -> int:
        """Add audio of any length; returns how many new embeddings were produced."""
        x = np.concatenate([self._pending, to_int16_scale(audio)])
        n = len(x) // FRAME
        self._pending = x[n * FRAME:]
        for k in range(n):
            chunk = x[k * FRAME:(k + 1) * FRAME]
            self._raw = np.concatenate([self._raw[FRAME:], chunk])  # keeps FRAME + MEL_CONTEXT samples
            mel = self._melspec(self._raw)                          # 8 new mel rows
            self._mel_buf = np.concatenate([self._mel_buf, mel])[-MEL_WINDOW:]
            emb = self._embed(self._mel_buf[None, :, :, None])
            self.features = np.concatenate([self.features, emb])[-_KEEP_EMBEDDINGS:]
            self.frames += 1
        return n

    def window(self, n: int) -> np.ndarray:
        """The last ``n`` embeddings as a (1, n, 96) float32 head input."""
        return self.features[-n:][None, :, :].astype(np.float32, copy=False)

"""Silero VAD (v6.2, ONNX, MIT) for speech gating: hands-free idle listening and
"is there any speech in this phrase?" checks before decoding.

Approach: a single-threaded, non-spinning ORT session (spinning
thread pools burn idle CPU) and an RMS floor that skips inference on near-silent frames, so the
VAD costs almost nothing while the room is quiet. The model is the v5-style graph (``state``
input, 512-sample chunks at 16 kHz with 64 samples of context), fetched from
``istupakov/silero-vad-onnx`` pinned by revision and SHA-256, instead of depending on the
openWakeWord package.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from openwhisprflow.config import models_dir
from openwhisprflow.models import ensure_file, hf_url

log = logging.getLogger("openwhisprflow.audio.vad")

REPO = "istupakov/silero-vad-onnx"
REVISION = "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6"
FILE, SIZE, SHA256 = "silero_vad.onnx", 2327524, "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SAMPLE_RATE = 16000
CHUNK, CONTEXT = 512, 64


def model_path(progress: Callable[[str, int, int], None] | None = None) -> Path:
    return ensure_file(hf_url(REPO, FILE, REVISION), models_dir() / "silero-vad" / FILE,
                       sha256=SHA256, size=SIZE, progress=progress)


def _session(path: Path):  # -> onnxruntime.InferenceSession
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class SileroVad:
    """Streaming speech probability for 16 kHz float32 audio of any block size.

    ``prob(block)`` returns the max probability over the 32 ms chunks completed by ``block``
    (leftover samples carry to the next call). ``speech_ratio(audio)`` scores a whole clip.
    """

    def __init__(self, path: str | Path | None = None, *, energy_floor: float = 0.0) -> None:
        self.session = _session(Path(path) if path else model_path())
        self.energy_floor = float(energy_floor)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.inferences = self.skipped = 0
        self._gap = 0
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._rem = np.zeros(0, dtype=np.float32)

    def _chunk(self, c: np.ndarray) -> float:
        if self.energy_floor > 0.0 and float(np.sqrt(np.mean(c * c))) < self.energy_floor:
            self.skipped += 1
            self._gap += 1
            self._context = c[-CONTEXT:]
            return 0.0
        if self._gap > 30:  # ~1 s skipped: the recurrent state is stale
            self._state[:] = 0.0
        self._gap = 0
        self.inferences += 1
        inp = np.concatenate([self._context, c])[None, :]
        out, self._state = self.session.run(None, {"input": inp, "state": self._state, "sr": self._sr})
        self._context = c[-CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])

    def probs(self, block: np.ndarray) -> list[float]:
        buf = np.concatenate([self._rem, np.asarray(block, dtype=np.float32).reshape(-1)])
        n = len(buf) // CHUNK
        out = [self._chunk(buf[i * CHUNK:(i + 1) * CHUNK]) for i in range(n)]
        self._rem = buf[n * CHUNK:]
        return out

    def prob(self, block: np.ndarray) -> float:
        p = self.probs(block)
        return max(p) if p else 0.0

    def speech_ratio(self, audio: np.ndarray, threshold: float = 0.5) -> float:
        """Fraction of 32 ms chunks above ``threshold`` in a whole clip (state is reset)."""
        self.reset()
        p = self.probs(audio)
        self.reset()
        return float(np.mean(np.asarray(p) >= threshold)) if p else 0.0

    def has_speech(self, audio: np.ndarray, threshold: float = 0.5, min_chunks: int = 3) -> bool:
        """At least ``min_chunks`` (~100 ms) of speech. Suitable as ``Segmenter(speech_check=...)``
        to keep Whisper from hallucinating on silent phrases."""
        self.reset()
        hits = sum(p >= threshold for p in self.probs(audio))
        self.reset()
        return hits >= min_chunks

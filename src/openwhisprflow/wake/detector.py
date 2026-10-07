"""Streaming wake word detector: front end + ONNX head + gate, VAD-gated so silence costs ~nothing.

Pieces:

* :class:`WakeHead` runs one classifier head on the last N embeddings (N = 16 for openWakeWord and
  livekit conv-attention heads). Heads output a sigmoid score; a logit head is squashed.
* :class:`WakeGate` debounces the score stream: threshold, N consecutive frames, then a cooldown.
* :class:`WakeDetector` ties them together for a live mic stream. The front end only runs while the
  VAD heard voice within ``hangover_s``; when voice starts again, the frames that were skipped
  (kept in a small ring, up to ``catchup_s``) are fed first, so the start of the word is never lost
  to the VAD's own onset delay. The VAD is injected as ``is_speech(frame) -> bool`` (int16 frame of
  1280 samples), so this module has no hard dependency on any particular VAD.

Thread-safety: one detector per audio thread; nothing here locks.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openwhisprflow.wake.frontend import (
    EMBED_DIM,
    FRAME,
    SAMPLE_RATE,
    WakeFrontend,
    make_session,
    to_int16_scale,
)

log = logging.getLogger("openwhisprflow.wake.detector")

FRAME_S = FRAME / SAMPLE_RATE  # 0.08
MODELS_DIR = Path(__file__).resolve().parents[3] / "assets" / "wake"  # repo checkout (reference runtime)

SpeechFn = Callable[[np.ndarray], bool]


def bundled_model_path(phrase: str) -> Path:
    """``assets/wake/<phrase>.onnx`` for a phrase like "transcribe" or "hey computer"."""
    return MODELS_DIR / f"{phrase.strip().lower().replace(' ', '_')}.onnx"


def resolve_model(phrase: str, model: str = "") -> Path:
    """Config ``handsfree.model`` (a path) wins; else the bundled model for ``handsfree.phrase``."""
    p = Path(model).expanduser() if model else bundled_model_path(phrase)
    if not p.exists():
        raise FileNotFoundError(f"no wake word model for {phrase!r}: {p} (train one: openwhisprflow train-wake)")
    return p


def load_meta(model_path: Path) -> dict[str, Any]:
    """The ``<model>.json`` written by training (threshold table, stats); {} if absent or unreadable."""
    meta = Path(model_path).with_suffix(".json")
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class WakeHead:
    """One ONNX classifier head over (1, n_frames, 96) embeddings."""

    def __init__(self, path: str | Path, provider: str = "cpu") -> None:
        self.path = Path(path)
        self.name = self.path.stem
        self.session = make_session(self.path, provider)
        inp = self.session.get_inputs()[0]
        shape = list(inp.shape)
        if len(shape) != 3 or (isinstance(shape[2], int) and shape[2] != EMBED_DIM):
            raise ValueError(f"{self.path.name}: expected (batch, frames, 96) embeddings input, got {shape}")
        self.input_name = inp.name
        self.n_frames = shape[1] if isinstance(shape[1], int) else 16

    def score(self, frontend: WakeFrontend) -> float:
        out = self.session.run(None, {self.input_name: frontend.window(self.n_frames)})[0]
        v = float(np.max(np.asarray(out, dtype=np.float32)))
        if not 0.0 <= v <= 1.0:  # logit head: squash
            v = 1.0 / (1.0 + math.exp(-v))
        return v


class WakeGate:
    """Threshold + N consecutive frames + cooldown on a per-frame score stream.

    Two frames (160 ms) above threshold rejects single-frame spikes from clicks and keyboard
    noise, which dominate false wakes on general audio; the cooldown stops one utterance firing
    twice. ``training/wakeword/pipeline.py:gate_count`` mirrors this exactly for FP/h numbers.
    """

    def __init__(self, threshold: float = 0.5, consecutive: int = 2, cooldown_frames: int = 19) -> None:
        self.threshold = threshold
        self.consecutive = max(1, consecutive)
        self.cooldown_frames = cooldown_frames
        self._run = 0
        self._last_fire: int | None = None
        self.peak = 0.0

    def reset_run(self) -> None:
        self._run = 0
        self.peak = 0.0

    def hold(self, frame_idx: int) -> None:
        """Start a cooldown now (e.g. right after a session ends, so its last words can't re-wake)."""
        self._last_fire = frame_idx
        self.reset_run()

    def update(self, score: float, frame_idx: int) -> bool:
        if score < self.threshold:
            self.reset_run()
            return False
        self._run += 1
        self.peak = max(self.peak, score)
        if self._run < self.consecutive:
            return False
        if self._last_fire is not None and frame_idx - self._last_fire < self.cooldown_frames:
            return False
        self._last_fire = frame_idx
        self._run = 0
        return True


@dataclass
class WakeHit:
    """A gate firing. Frame indices count 80 ms frames since the detector was created."""

    frame: int           # frame on which the gate fired (end of the word + ~0-400 ms)
    rise_frame: int      # first frame of the above-``rise`` run that fired: roughly the end of the word
    score: float         # peak score of the run
    model: str = ""


@dataclass
class DetectorStats:
    frames: int = 0          # 80 ms frames seen
    inferred: int = 0        # frames that ran the front end + head
    speech_frames: int = 0
    hits: int = 0
    scores: deque[float] = field(default_factory=lambda: deque(maxlen=64))  # recent head scores (debug UI)


class WakeDetector:
    """``process(block) -> list[WakeHit]`` for a live 16 kHz mono stream of any block size."""

    def __init__(self, model_path: str | Path, *, threshold: float | None = None, consecutive: int = 2,
                 cooldown_s: float = 1.5, is_speech: SpeechFn | None = None, hangover_s: float = 1.0,
                 catchup_s: float = 2.0, provider: str = "cpu", frontend: WakeFrontend | None = None,
                 head: WakeHead | None = None, rise_threshold: float | None = None) -> None:
        """``threshold=None`` uses the model's recommended threshold from ``<model>.json``.
        ``frontend`` / ``head`` can be injected (shared front end, tests)."""
        self.meta = load_meta(Path(model_path))
        rec = self.meta.get("recommended") or {}
        if threshold is None:
            threshold = float(rec.get("threshold", 0.5))
        self.frontend = frontend or WakeFrontend(provider=provider)
        self.head = head or WakeHead(model_path, provider)
        self.gate = WakeGate(threshold, consecutive, max(1, round(cooldown_s / FRAME_S)))
        self.rise_threshold = rise_threshold if rise_threshold is not None else threshold * 0.6
        self.is_speech = is_speech
        self.hangover = max(0, round(hangover_s / FRAME_S))
        self.stats = DetectorStats()
        self._pending = np.zeros(0, dtype=np.float32)
        self._idx = 0                       # last frame index seen
        self._fed = 0                       # last frame index pushed through the front end
        self._last_voice = -10**9
        self._rise: int | None = None
        self._backlog: deque[tuple[int, np.ndarray]] = deque(maxlen=max(1, round(catchup_s / FRAME_S)))
        self.last_score = 0.0
        self.last_voiced = True              # VAD decision for the most recent frame (True without a VAD)

    @property
    def has_vad(self) -> bool:
        return self.is_speech is not None

    @property
    def threshold(self) -> float:
        return self.gate.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.gate.threshold = float(value)

    @property
    def frame_index(self) -> int:
        return self._idx

    def reset(self) -> None:
        """Forget audio context (front end, gate run, backlog). Keeps the cooldown and counters."""
        self.frontend.reset()
        self.gate.reset_run()
        self._pending = np.zeros(0, dtype=np.float32)
        self._backlog.clear()
        self._rise = None
        self._fed = self._idx
        self.last_score = 0.0

    def hold(self) -> None:
        """Cooldown from now (see :meth:`WakeGate.hold`)."""
        self.gate.hold(self._idx)

    def process(self, block: np.ndarray) -> list[WakeHit]:
        """Feed audio (int16, or float [-1, 1]); returns the hits that fired inside it (usually none)."""
        x = np.concatenate([self._pending, to_int16_scale(np.asarray(block).reshape(-1))])
        n = len(x) // FRAME
        self._pending = x[n * FRAME:]
        hits: list[WakeHit] = []
        for k in range(n):
            hit = self._frame(x[k * FRAME:(k + 1) * FRAME])
            if hit is not None:
                hits.append(hit)
        return hits

    # ------------------------------------------------------------------ internals
    def _frame(self, frame: np.ndarray) -> WakeHit | None:
        self._idx += 1
        idx = self._idx
        self.stats.frames += 1
        if self.is_speech is None:
            voiced = True
        else:
            voiced = bool(self.is_speech(frame.astype(np.int16)))
        self.last_voiced = voiced
        if voiced:
            self.stats.speech_frames += 1
            self._last_voice = idx
        if idx - self._last_voice > self.hangover:
            # Silence: no inference. Remember the frame in case speech starts right after it.
            self._backlog.append((idx, frame))
            return None
        todo = [(i, f) for i, f in self._backlog if i > self._fed]
        self._backlog.clear()
        todo.append((idx, frame))
        hit = None
        for i, f in todo:
            h = self._infer(i, f)
            hit = hit or h
        return hit

    def _infer(self, i: int, frame: np.ndarray) -> WakeHit | None:
        if i != self._fed + 1:  # a gap since the last inference: the score run is not continuous
            self.gate.reset_run()
            self._rise = None
        self._fed = i
        self.frontend.push(frame)
        s = self.head.score(self.frontend)
        self.last_score = s
        self.stats.inferred += 1
        self.stats.scores.append(s)
        if s >= self.rise_threshold:
            if self._rise is None:
                self._rise = i
        else:
            self._rise = None
        if self.gate.update(s, i):
            self.stats.hits += 1
            return WakeHit(frame=i, rise_frame=self._rise if self._rise is not None else i,
                           score=max(self.gate.peak, s), model=self.head.name)
        return None


def detector_from_config(hf: Any, *, is_speech: SpeechFn | None = None, provider: str = "cpu",
                         **kwargs: Any) -> WakeDetector:
    """Build from ``Config.handsfree`` (phrase, model, threshold)."""
    path = resolve_model(hf.phrase, hf.model)
    return WakeDetector(path, threshold=float(hf.threshold), is_speech=is_speech, provider=provider, **kwargs)


def score_clip(detector: WakeDetector, audio: np.ndarray, *, lead_s: float = 1.0, tail_s: float = 1.0,
               block: int = FRAME) -> tuple[float, list[WakeHit]]:
    """Stream one clip (with silence around it) through a fresh-context detector: (peak score, hits).

    Used by tests and ``training/wakeword/verify_runtime.py``; mirrors live use (blocks of 80 ms).
    """
    detector.reset()
    x = to_int16_scale(np.asarray(audio).reshape(-1))
    pad = lambda s: np.zeros(int(s * SAMPLE_RATE), np.float32)  # noqa: E731
    x = np.concatenate([pad(lead_s), x, pad(tail_s)])
    peak, hits = 0.0, []
    for i in range(0, len(x), block):
        hits += detector.process(x[i:i + block])
        peak = max(peak, detector.last_score)
    return peak, hits

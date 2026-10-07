"""Decode completed phrases while the user is still talking.

Why: release-to-text latency should cost one phrase, not the whole utterance. Audio is cut
into phrases only at real pauses, so no word is split and no samples are dropped:

* after 5 s, the first pause of >= 320 ms ends a phrase;
* after 20 s, a pause of >= 120 ms is enough;
* a phrase never exceeds 30 s: someone who talks without stopping is cut at the quietest
  20 ms frame of the last 5 s.

"Quiet" is relative to the utterance's peak (3 %) and capped at -54 dBFS, which also keeps
softly spoken words. Decodes run on one worker thread (recognizers are not re-entrant and a
single worker keeps CPU use predictable); results come back in order.

``PhraseCutter`` is the pure, synchronous cutting logic; ``Segmenter`` drives it, resamples
each phrase to the engine rate and runs the decodes. ``split_at_pauses`` is the offline
variant for files longer than an engine can take at once.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from openwhisprflow.audio.resample import resample
from openwhisprflow.stt.base import SAMPLE_RATE, SttResult

log = logging.getLogger("openwhisprflow.audio.segmenter")

FRAME_S = 0.02
MIN_PHRASE_S, SOFT_PHRASE_S, MAX_PHRASE_S, CUT_WINDOW_S = 5.0, 20.0, 30.0, 5.0
LONG_PAUSE_S, SHORT_PAUSE_S = 0.32, 0.12
QUIET_CAP = 0.002          # -54 dBFS
QUIET_RELATIVE = 0.03      # of the utterance's peak frame RMS


class PhraseCutter:
    """Splits a stream of mono float32 blocks into phrases at pauses. No threads, no I/O."""

    def __init__(self, rate: int) -> None:
        self.rate = int(rate)
        self.width = max(1, round(self.rate * FRAME_S))
        self.parts: list[np.ndarray] = []
        self.levels: list[float] = []
        self.frames = 0
        self.quiet = 0
        self.peak_rms = 0.0
        self._carry = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        """Add audio; return any phrases completed by it (each at ``self.rate``)."""
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        if self._carry.size:
            data = np.concatenate([self._carry, data])
        whole = len(data) - len(data) % self.width
        self._carry = data[whole:].copy()
        out: list[np.ndarray] = []
        rate = self.rate
        for start in range(0, whole, self.width):
            frame = data[start:start + self.width]
            self.parts.append(frame)
            self.frames += len(frame)
            rms = float(np.sqrt(np.mean(frame * frame)))
            self.levels.append(rms)
            self.peak_rms = max(self.peak_rms, rms)
            threshold = min(QUIET_CAP, self.peak_rms * QUIET_RELATIVE)
            quiet = rms <= threshold and float(np.max(np.abs(frame))) <= max(threshold * 4, 1e-6)
            self.quiet = self.quiet + len(frame) if quiet else 0
            if self.frames >= rate * MIN_PHRASE_S and self.quiet >= rate * LONG_PAUSE_S:
                out.append(self._take())
            elif self.frames >= rate * SOFT_PHRASE_S and self.quiet >= rate * SHORT_PAUSE_S:
                out.append(self._take())
            elif self.frames >= rate * MAX_PHRASE_S:
                window = min(len(self.levels), max(1, round(CUT_WINDOW_S / FRAME_S)))
                tail = self.levels[-window:]
                out.append(self._take(len(self.levels) - window + tail.index(min(tail)) + 1))
        return out

    def _take(self, count: int | None = None) -> np.ndarray:
        count = len(self.parts) if count is None else count
        audio = np.concatenate(self.parts[:count]) if count else np.zeros(0, dtype=np.float32)
        self.parts, self.levels = self.parts[count:], self.levels[count:]
        self.frames = sum(len(p) for p in self.parts)
        self.quiet = 0
        return audio

    def pending(self) -> np.ndarray:
        """Copy of the open (not yet cut) phrase, for live partials."""
        chunks = [*self.parts, self._carry] if self._carry.size else self.parts
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    @property
    def pending_seconds(self) -> float:
        return (self.frames + len(self._carry)) / self.rate

    def flush(self) -> np.ndarray | None:
        """Everything not yet cut (end of capture), or None if empty."""
        if self._carry.size:
            self.parts.append(self._carry)
            self._carry = np.zeros(0, dtype=np.float32)
        audio = self._take()
        return audio if audio.size else None


def split_at_pauses(audio: np.ndarray, rate: int, max_s: float = MAX_PHRASE_S,
                    window_s: float = CUT_WINDOW_S) -> list[np.ndarray]:
    """Split a long recording into chunks <= ``max_s``, each cut at the quietest 20 ms frame of
    its last ``window_s``. Audio that already fits is returned whole (best for accuracy)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    limit = int(max_s * rate)
    width = max(1, round(rate * FRAME_S))
    chunks: list[np.ndarray] = []
    pos = 0
    while len(audio) - pos > limit:
        lo = pos + max(width, int((max_s - window_s) * rate))
        seg = audio[lo:pos + limit]
        n = len(seg) // width
        rms = np.sqrt(np.mean(seg[:n * width].reshape(n, width) ** 2, axis=1))
        cut = lo + (int(np.argmin(rms)) + 1) * width
        chunks.append(audio[pos:cut])
        pos = cut
    chunks.append(audio[pos:])
    return chunks


_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def decode_pool() -> ThreadPoolExecutor:
    """Process-wide single decode worker, shared across sessions so a finishing session and a
    new one never run the recognizer concurrently."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owf-decode")
        return _POOL


Transcribe = Callable[[np.ndarray], SttResult]
PartialCallback = Callable[[str, int], None]        # (text, stable_chars)
SegmentCallback = Callable[[SttResult], None]


class Segmenter:
    """One dictation session: ``feed()`` capture blocks, then ``finish()`` (or ``cancel()``).

    ``transcribe`` receives mono float32 at ``engine_rate`` (bind language/prompt with a
    lambda). ``on_segment`` fires on the worker thread as each phrase is decoded (hands-free
    uses it to spot trailing control phrases). ``on_partial(text, stable_chars)`` updates the
    HUD bubble: decoded phrases are stable; with ``partial_interval_s`` set, the open phrase is
    also decoded opportunistically (only while the worker is idle) and shown as unstable text.
    """

    def __init__(self, transcribe: Transcribe, input_rate: int, *, engine_rate: int = SAMPLE_RATE,
                 executor: ThreadPoolExecutor | None = None,
                 on_segment: SegmentCallback | None = None,
                 on_partial: PartialCallback | None = None,
                 partial_interval_s: float | None = None,
                 speech_check: Callable[[np.ndarray], bool] | None = None) -> None:
        self._transcribe = transcribe
        self.input_rate = int(input_rate)
        self.engine_rate = int(engine_rate)
        self._pool = executor or decode_pool()
        self._cutter = PhraseCutter(self.input_rate)
        self._on_segment = on_segment
        self._on_partial = on_partial
        self._partial_interval = partial_interval_s
        self._speech_check = speech_check
        self._jobs: list[Future[SttResult | None]] = []
        self._committed: list[str] = []
        self._lock = threading.Lock()
        self._busy = 0
        self._last_partial = 0.0
        self._canceled = threading.Event()
        self._finished = False

    # -- capture side (call from one thread) ---------------------------------------------
    def feed(self, block: np.ndarray) -> None:
        if self._finished or self._canceled.is_set():
            return
        for phrase in self._cutter.push(block):
            self._submit(phrase)
        self._maybe_partial()

    def finish(self, timeout: float | None = None) -> list[SttResult]:
        """Decode the tail and wait for every phrase. Raises the first decode error."""
        if self._canceled.is_set():
            return []
        if self._finished:
            raise RuntimeError("Segmenter.finish() called twice")
        self._finished = True
        tail = self._cutter.flush()
        if tail is not None:
            self._submit(tail)
        deadline = None if timeout is None else time.monotonic() + timeout
        results: list[SttResult] = []
        for job in self._jobs:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            res = job.result(timeout=left)
            if res is not None:
                results.append(res)
        return [] if self._canceled.is_set() else results

    def cancel(self) -> None:
        """Drop everything; queued decodes become no-ops. Returns immediately."""
        self._canceled.set()
        self._finished = True
        self._cutter = PhraseCutter(self.input_rate)

    @property
    def seconds(self) -> float:
        return self._cutter.pending_seconds

    @staticmethod
    def join(results: list[SttResult], engine: str = "") -> SttResult:
        """Combine per-phrase results into one utterance result."""
        return SttResult(
            text=" ".join(r.text.strip() for r in results if r.text.strip()),
            duration_ms=sum(r.duration_ms for r in results),
            processing_ms=sum(r.processing_ms for r in results),
            language=next((r.language for r in results if r.language), None),
            engine=engine or next((r.engine for r in results if r.engine), ""),
        )

    # -- worker side ---------------------------------------------------------------------
    def _to_engine_rate(self, audio: np.ndarray) -> np.ndarray:
        if self.input_rate == self.engine_rate:
            return audio
        return resample(audio, self.input_rate, self.engine_rate)

    def _submit(self, phrase: np.ndarray) -> None:
        with self._lock:
            self._busy += 1
        self._jobs.append(self._pool.submit(self._decode, phrase))

    def _decode(self, phrase: np.ndarray) -> SttResult | None:
        try:
            if self._canceled.is_set():
                return None
            audio = self._to_engine_rate(phrase)
            if self._speech_check is not None and not self._speech_check(audio):
                return SttResult(text="", duration_ms=round(len(audio) * 1000 / self.engine_rate),
                                 processing_ms=0)
            result = self._transcribe(audio)
            if self._canceled.is_set():
                return None
            if result.text.strip():
                self._committed.append(result.text.strip())
            if self._on_segment:
                self._safe(self._on_segment, result)
            if self._on_partial:
                text = " ".join(self._committed)
                self._safe(self._on_partial, text, len(text))
            return result
        finally:
            with self._lock:
                self._busy -= 1

    def _maybe_partial(self) -> None:
        if self._on_partial is None or self._partial_interval is None:
            return
        now = time.monotonic()
        if now - self._last_partial < self._partial_interval or self._cutter.pending_seconds < 1.0:
            return
        with self._lock:
            if self._busy:
                return  # never delay a real phrase for a preview
            self._busy += 1
        self._last_partial = now
        self._pool.submit(self._decode_partial, self._cutter.pending())

    def _decode_partial(self, audio: np.ndarray) -> None:
        try:
            if self._canceled.is_set() or self._finished:
                return
            interim = self._transcribe(self._to_engine_rate(audio)).text.strip()
            if self._canceled.is_set() or self._finished or not self._on_partial:
                return
            stable = " ".join(self._committed)
            text = f"{stable} {interim}".strip() if interim else stable
            self._safe(self._on_partial, text, len(stable))
        except Exception:
            log.debug("partial decode failed", exc_info=True)
        finally:
            with self._lock:
                self._busy -= 1

    @staticmethod
    def _safe(fn: Callable[..., None], *args: object) -> None:
        try:
            fn(*args)
        except Exception:  # a UI callback must never break transcription
            log.exception("segmenter callback failed")

"""Phrase cutting and the decode worker, on synthetic audio (no models)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from openwhisprflow.audio.segmenter import PhraseCutter, Segmenter, split_at_pauses
from openwhisprflow.stt.base import SttResult

RATE = 48000


def speech(seconds: float, rate: int = RATE, amp: float = 0.2, seed: int = 0) -> np.ndarray:
    """Noise modulated at a syllable rate: loud enough never to count as quiet."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3 * t))
    return (amp * env * rng.standard_normal(len(t))).astype(np.float32)


def silence(seconds: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


def blocks(audio: np.ndarray, size: int = 480 * 2 + 37):
    """Odd block size so frames straddle callback blocks, as with real devices."""
    for i in range(0, len(audio), size):
        yield audio[i:i + size]


def cut(audio: np.ndarray, rate: int = RATE) -> list[np.ndarray]:
    c = PhraseCutter(rate)
    out = [p for b in blocks(audio) for p in c.push(b)]
    tail = c.flush()
    return out + ([tail] if tail is not None else [])


def test_short_utterance_is_one_phrase_and_lossless() -> None:
    audio = np.concatenate([speech(2), silence(0.5), speech(1.5)])
    phrases = cut(audio)
    assert len(phrases) == 1
    np.testing.assert_array_equal(np.concatenate(phrases), audio)


def test_pause_after_five_seconds_cuts() -> None:
    audio = np.concatenate([speech(6), silence(0.5), speech(3)])
    phrases = cut(audio)
    assert len(phrases) == 2
    assert 6.3 <= len(phrases[0]) / RATE <= 6.4   # cut 320 ms into the pause
    np.testing.assert_array_equal(np.concatenate(phrases), audio)


def test_short_pause_before_five_seconds_does_not_cut() -> None:
    audio = np.concatenate([speech(3), silence(1.0), speech(3)])
    assert len(cut(audio)) == 1


def test_brief_pause_cuts_only_after_twenty_seconds() -> None:
    gap = silence(0.15)
    audio = np.concatenate([speech(10), gap, speech(11), gap, speech(3)])
    phrases = cut(audio)
    assert len(phrases) == 2
    assert 21.1 < len(phrases[0]) / RATE < 21.3


def test_continuous_speech_is_cut_at_quietest_point_before_thirty_seconds() -> None:
    audio = speech(40)
    dip = int(27.0 * RATE)
    audio[dip:dip + 960] *= 0.05          # one soft 20 ms frame, not silence
    phrases = cut(audio)
    assert len(phrases) == 2
    assert all(len(p) <= 30 * RATE for p in phrases)
    assert abs(len(phrases[0]) / RATE - 27.02) < 0.03
    np.testing.assert_array_equal(np.concatenate(phrases), audio)


def test_soft_speech_is_not_quiet() -> None:
    loud, soft = speech(6, amp=0.3), speech(3, amp=0.01, seed=1)   # soft = 3 % of loud
    assert len(cut(np.concatenate([loud, soft]))) == 1


def test_split_at_pauses_offline() -> None:
    short = speech(12, rate=16000)
    assert len(split_at_pauses(short, 16000)) == 1
    long = speech(70, rate=16000)
    for at in (28.0, 55.5):
        i = int(at * 16000)
        long[i:i + 320] = 0
    chunks = split_at_pauses(long, 16000)
    assert [round(len(c) / 16000, 1) for c in chunks] == [28.0, 27.5, 14.5]
    np.testing.assert_array_equal(np.concatenate(chunks), long)


class FakeEngine:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[int] = []
        self.lock = threading.Lock()

    def __call__(self, audio: np.ndarray) -> SttResult:
        time.sleep(self.delay)
        with self.lock:
            self.calls.append(len(audio))
            n = len(self.calls)
        return SttResult(text=f"p{n}", duration_ms=round(len(audio) / 16), processing_ms=1, engine="fake")


@pytest.fixture
def pool():
    p = ThreadPoolExecutor(max_workers=1)
    yield p
    p.shutdown(wait=True)


def test_segmenter_decodes_during_capture_in_order(pool: ThreadPoolExecutor) -> None:
    engine = FakeEngine(delay=0.01)
    segs = []
    seg = Segmenter(engine, RATE, executor=pool, on_segment=segs.append)
    audio = np.concatenate([speech(6), silence(0.5), speech(6), silence(0.5), speech(2)])
    for b in blocks(audio):
        seg.feed(b)
    pool.submit(lambda: None).result(timeout=5)   # barrier: decodes queued so far are done
    assert len(engine.calls) == 2          # two phrases decoded before release
    results = seg.finish(timeout=5)
    assert [r.text for r in results] == ["p1", "p2", "p3"]
    assert len(segs) == 3
    # resampled to 16 kHz, nothing lost (+-1 sample per phrase from ceil)
    assert abs(sum(engine.calls) - len(audio) / 3) <= 3
    joined = Segmenter.join(results)
    assert joined.text == "p1 p2 p3" and joined.engine == "fake"


def test_segmenter_partials(pool: ThreadPoolExecutor) -> None:
    engine = FakeEngine()
    partials: list[tuple[str, int]] = []
    seg = Segmenter(engine, RATE, executor=pool, on_partial=lambda t, s: partials.append((t, s)),
                    partial_interval_s=0.0)
    for b in blocks(speech(2)):
        seg.feed(b)
    pool.submit(lambda: None).result(timeout=2)   # let the opportunistic preview run
    for b in blocks(np.concatenate([speech(4), silence(0.5), speech(2)])):
        seg.feed(b)
    seg.finish(timeout=5)
    assert partials, "expected partial updates"
    final_text, stable = partials[-1]
    assert stable == len(final_text) and final_text.startswith("p")
    assert any(s < len(t) for t, s in partials), "expected an unstable interim partial"


def test_cancel_drops_queued_work(pool: ThreadPoolExecutor) -> None:
    engine = FakeEngine(delay=0.2)
    seg = Segmenter(engine, RATE, executor=pool)
    for b in blocks(np.concatenate([speech(6), silence(0.5), speech(6), silence(0.5)])):
        seg.feed(b)
    seg.cancel()
    assert seg.finish() == []
    pool.submit(lambda: None).result(timeout=2)
    assert len(engine.calls) <= 1          # the in-flight decode may finish; queued ones never run


def test_speech_check_skips_decode(pool: ThreadPoolExecutor) -> None:
    engine = FakeEngine()
    seg = Segmenter(engine, RATE, executor=pool, speech_check=lambda a: False)
    seg.feed(speech(1))
    results = seg.finish(timeout=2)
    assert engine.calls == [] and results[0].text == ""


def test_decode_errors_surface_on_finish(pool: ThreadPoolExecutor) -> None:
    def boom(audio: np.ndarray) -> SttResult:
        raise RuntimeError("model exploded")

    seg = Segmenter(boom, 16000, executor=pool)
    seg.feed(speech(1, rate=16000))
    with pytest.raises(RuntimeError, match="exploded"):
        seg.finish(timeout=2)

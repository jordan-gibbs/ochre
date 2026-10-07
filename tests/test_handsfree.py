"""wake/handsfree.py: the hands-free session state machine, driven by fake audio, hits and transcripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from openwhisprflow.config import HandsFreeConfig
from openwhisprflow.text.commands import Action
from openwhisprflow.wake.frontend import FRAME
from openwhisprflow.wake.handsfree import EndReason, HandsFreeController, HFState, SessionEnd, WakeStart

LOUD = np.full(FRAME, 3000, dtype=np.int16)
QUIET = np.zeros(FRAME, dtype=np.int16)


@dataclass
class Hit:
    score: float
    frame: int
    rise_frame: int


class FakeDetector:
    """Voiced = loud frame (when it has a VAD); fires on scripted frame numbers."""

    def __init__(self, vad: bool = True) -> None:
        self.vad = vad
        self.idx = 0
        self.fire_at: set[int] = set()
        self.last_voiced = True
        self.processed = 0
        self.resets = 0
        self.holds = 0

    @property
    def has_vad(self) -> bool:
        return self.vad

    def process(self, block: np.ndarray) -> list[Hit]:
        self.idx += 1
        self.processed += 1
        self.last_voiced = bool(np.abs(block.astype(np.float32)).mean() > 100) if self.vad else True
        return [Hit(0.9, self.idx, self.idx - 3)] if self.idx in self.fire_at else []

    def reset(self) -> None:
        self.resets += 1

    def hold(self) -> None:
        self.holds += 1


class FakeMonitor:
    def __init__(self) -> None:
        self.apps: list[str] = []

    def in_use(self) -> list[str]:
        return list(self.apps)


class Rig:
    def __init__(self, vad: bool = True, monitor: FakeMonitor | None = None, **kw: object) -> None:
        self.t = 1000.0
        self.cfg = HandsFreeConfig(enabled=True, preroll_s=1.5, idle_timeout_s=45.0)
        self.det = FakeDetector(vad)
        self.wakes: list[WakeStart] = []
        self.ends: list[SessionEnd] = []
        self.pauses: list[tuple[bool, list[str]]] = []
        self.states: list[HFState] = []
        self.hf = HandsFreeController(
            self.cfg, self.det, on_wake=self.wakes.append, on_end=self.ends.append,
            on_pause=lambda p, a: self.pauses.append((p, a)), on_state=self.states.append,
            mic_monitor=monitor, clock=lambda: self.t, **kw)  # type: ignore[arg-type]

    def frames(self, n: int, frame: np.ndarray = LOUD) -> None:
        for _ in range(n):
            self.hf.feed(frame)
            self.t += 0.08

    def wake(self, fire_after: int = 10, lead_quiet: int = 20) -> WakeStart:
        """Silence, then speech that fires the detector after ``fire_after`` voiced frames."""
        self.frames(lead_quiet, QUIET)
        self.det.fire_at = {self.det.idx + fire_after}
        self.frames(fire_after)
        assert self.wakes, "wake did not fire"
        return self.wakes[-1]


def test_wake_starts_session_with_clean_preroll() -> None:
    r = Rig()
    w = r.wake(fire_after=10, lead_quiet=30)
    assert r.hf.state is HFState.SESSION and r.hf.in_session
    assert w.session_id == 1 and w.score == pytest.approx(0.9)
    assert w.clean_start
    # 10 voiced frames + 2 frames of quiet lead-in, not the whole 1.5 s ring
    assert len(w.preroll) == 12 * FRAME
    assert w.preroll.dtype == np.int16
    assert r.states == [HFState.SESSION]


def test_preroll_without_vad_is_the_whole_window() -> None:
    r = Rig(vad=False)
    w = r.wake(fire_after=10, lead_quiet=40)
    assert not w.clean_start
    assert len(w.preroll) == 19 * FRAME  # ceil(1.5 / 0.08)


def test_preroll_mid_speech_is_not_clean() -> None:
    r = Rig()
    r.frames(60)  # continuous talking, no gap before the word
    r.det.fire_at = {r.det.idx + 3}
    r.frames(3)
    w = r.wakes[-1]
    assert not w.clean_start
    # capped at max(pre-roll, 1.2 s before the end of the word)
    assert 19 * FRAME <= len(w.preroll) <= 25 * FRAME


def test_full_session_send() -> None:
    r = Rig()
    r.wake()
    a = r.hf.on_phrase("Transcribe, hey Sarah, just checking in.")
    assert a.action is Action.NONE and a.kept == "Hey Sarah, just checking in."
    b = r.hf.on_phrase("Does five work for you? Transcribe send.")
    assert b.action is Action.SEND and b.ended is not None
    assert len(r.ends) == 1
    end = r.ends[0]
    assert end.reason is EndReason.SEND
    assert end.text == "Hey Sarah, just checking in. Does five work for you?"
    assert end.insert and end.press_enter
    assert r.hf.state is HFState.LISTENING and not r.hf.in_session
    assert r.det.holds == 1 and r.det.resets >= 1


def test_stop_in_first_phrase() -> None:
    r = Rig()
    r.wake()
    res = r.hf.on_phrase("Transcribe, buy milk and eggs. Transcribe stop.")
    assert res.action is Action.FINISH
    assert r.ends[0].text == "Buy milk and eggs."
    assert r.ends[0].insert and not r.ends[0].press_enter


def test_wake_word_then_immediate_stop_inserts_nothing() -> None:
    r = Rig()
    r.wake()
    r.hf.on_phrase("Transcribe stop.")
    assert r.ends[0].reason is EndReason.STOP
    assert r.ends[0].text == "" and not r.ends[0].insert


def test_false_wake_is_dropped() -> None:
    r = Rig()
    r.wake()
    res = r.hf.on_phrase("I need to transcribe this video by Friday.")
    assert res.ended is not None
    assert r.ends[0].reason is EndReason.FALSE_WAKE
    assert not r.ends[0].insert
    assert r.ends[0].detail == "I need to transcribe this video by Friday."


def test_verify_off_keeps_text() -> None:
    r = Rig(verify=False)
    r.wake()
    r.hf.on_phrase("Something without the wake word.")
    r.hf.on_phrase("transcribe done")
    assert r.ends[0].text == "Something without the wake word."


def test_cancel_discards() -> None:
    r = Rig()
    r.wake()
    r.hf.on_phrase("Transcribe. Dear landlord, I am furious.")
    r.hf.on_phrase("Transcribe cancel.")
    assert r.ends[0].reason is EndReason.CANCEL
    assert not r.ends[0].insert


def test_scratch_that() -> None:
    r = Rig()
    r.wake()
    r.hf.on_phrase("Transcribe. First line.")
    r.hf.on_phrase("Second line.")
    res = r.hf.on_phrase("Transcribe scratch that.")
    assert res.action is Action.SCRATCH and res.session_text == "First line."
    res = r.hf.on_phrase("Wrong words here, transcribe scratch that.")
    assert res.session_text == "First line."
    r.hf.on_phrase("Third line. Transcribe stop.")
    assert r.ends[0].text == "First line. Third line."


def test_segmenter_split_between_phrase_and_command() -> None:
    r = Rig()
    r.wake()
    r.hf.on_phrase("Transcribe, notes for Monday.")
    r.hf.on_phrase("That's all. Transcribe.")
    res = r.hf.on_phrase("Send.")
    assert res.action is Action.SEND
    assert r.ends[0].text == "Notes for Monday. That's all."
    assert r.ends[0].press_enter


def test_dangling_wake_word_kept_when_no_command_follows() -> None:
    r = Rig()
    r.wake()
    r.hf.on_phrase("Transcribe. We need someone to transcribe")
    r.hf.on_phrase("the interviews by Friday.")
    r.hf.on_phrase("transcribe done")
    assert r.ends[0].text == "We need someone to transcribe the interviews by Friday."


def test_wake_hit_during_session_arms_bare_command() -> None:
    r = Rig()
    r.wake()
    r.hf.on_phrase("Transcribe, hello.")
    assert r.hf.on_phrase("Stop.").action is Action.NONE  # not armed: literal dictation
    r.det.fire_at = {r.det.idx + 2}
    r.frames(3)
    assert r.hf.in_session  # a hit during a session never starts another one
    res = r.hf.on_phrase("Stop.")
    assert res.action is Action.FINISH
    assert r.ends[0].text == "Hello. Stop."
    assert len(r.wakes) == 1


def test_arming_expires() -> None:
    r = Rig(arm_s=2.0)
    r.wake()
    r.hf.on_phrase("Transcribe, hello.")
    r.det.fire_at = {r.det.idx + 1}
    r.frames(1)
    r.frames(40)  # 3.2 s later
    assert r.hf.on_phrase("Stop.").action is Action.NONE


def test_idle_timeout_inserts() -> None:
    r = Rig()
    r.wake()
    r.hf.on_phrase("Transcribe, remember the dentist.")
    r.frames(int(44 / 0.08), QUIET)
    assert r.hf.in_session
    r.frames(int(2 / 0.08), QUIET)
    assert r.ends[0].reason is EndReason.IDLE_TIMEOUT
    assert r.ends[0].insert and r.ends[0].text == "Remember the dentist."


def test_speech_keeps_session_alive() -> None:
    r = Rig()
    r.wake()
    for _ in range(3):
        r.frames(int(30 / 0.08), QUIET)
        r.frames(5)  # talking (phrase not finished yet)
    assert r.hf.in_session


def test_max_duration() -> None:
    r = Rig(max_session_s=10.0)
    r.wake()
    r.hf.on_phrase("Transcribe, long story.")
    r.frames(int(11 / 0.08))
    assert r.ends[0].reason is EndReason.MAX_DURATION
    assert r.ends[0].text == "Long story."


def test_cooldown_after_session_blocks_rewake() -> None:
    r = Rig(post_session_cooldown_s=1.5)
    r.wake()
    r.hf.on_phrase("Transcribe stop.")
    r.det.fire_at = {r.det.idx + 2}
    r.frames(4)
    assert len(r.wakes) == 1
    r.frames(30, QUIET)
    r.det.fire_at = {r.det.idx + 5}
    r.frames(6)
    assert len(r.wakes) == 2 and r.wakes[1].session_id == 2


def test_late_and_stale_phrases_are_ignored() -> None:
    r = Rig()
    w = r.wake()
    r.hf.on_phrase("Transcribe stop.")
    assert r.hf.on_phrase("late words").ignored
    r.frames(30, QUIET)
    r.det.fire_at = {r.det.idx + 5}
    r.frames(6)
    assert r.hf.on_phrase("Transcribe, x", session_id=w.session_id).ignored
    assert not r.hf.on_phrase("Transcribe, x", session_id=r.wakes[-1].session_id).ignored


def test_external_finish_and_cancel() -> None:
    r = Rig()
    r.wake()
    r.hf.on_phrase("Transcribe, note to self.")
    end = r.hf.finish(send=True)
    assert end is not None and end.reason is EndReason.SEND and end.press_enter
    assert r.hf.finish() is None and r.hf.cancel() is None
    r.frames(30, QUIET)
    r.det.fire_at = {r.det.idx + 5}
    r.frames(6)
    assert r.hf.cancel().reason is EndReason.CANCEL  # type: ignore[union-attr]


def test_disable_cancels_session_and_stops_listening() -> None:
    r = Rig()
    r.wake()
    r.hf.set_enabled(False)
    assert r.ends[0].reason is EndReason.CANCEL
    assert r.hf.state is HFState.OFF
    n = r.det.processed
    r.frames(10)
    assert r.det.processed == n
    r.hf.set_enabled(True)
    assert r.hf.state is HFState.LISTENING


def test_pause_while_mic_in_use() -> None:
    mon = FakeMonitor()
    r = Rig(monitor=mon, call_poll_s=1.0)
    r.frames(5, QUIET)
    mon.apps = ["Zoom"]
    r.frames(15, QUIET)
    assert r.hf.state is HFState.PAUSED
    assert r.pauses == [(True, ["Zoom"])]
    n = r.det.processed
    r.det.fire_at = {r.det.idx + 1}
    r.frames(10)
    assert r.det.processed == n and not r.wakes  # no inference while paused
    r.det.fire_at = set()
    mon.apps = []
    r.frames(15, QUIET)
    assert r.hf.state is HFState.LISTENING
    assert r.pauses[-1] == (False, [])


def test_pause_on_calls_off() -> None:
    mon = FakeMonitor()
    mon.apps = ["Teams"]
    r = Rig(monitor=mon)
    r.cfg.pause_on_calls = False
    r.frames(50, QUIET)
    assert r.hf.state is HFState.LISTENING


def test_callbacks_may_reenter() -> None:
    r = Rig()
    seen: list[str] = []
    r.hf.on_end_cb = lambda e: seen.append(r.hf.session_text + "|" + e.text)
    r.hf.on_wake_cb = lambda w: seen.append(str(r.hf.in_session))
    r.frames(20, QUIET)
    r.det.fire_at = {r.det.idx + 5}
    r.frames(6)
    r.hf.on_phrase("Transcribe, hi. Transcribe stop.")
    assert seen == ["True", "|Hi."]


def test_failing_callback_does_not_break_feed() -> None:
    r = Rig()

    def boom(_: object) -> None:
        raise RuntimeError("orchestrator bug")

    r.hf.on_wake_cb = boom
    r.frames(20, QUIET)
    r.det.fire_at = {r.det.idx + 5}
    r.frames(10)
    assert r.hf.in_session


def test_float_blocks_of_any_size() -> None:
    r = Rig()
    r.frames(20, QUIET)
    r.det.fire_at = {r.det.idx + 4}
    audio = np.full(FRAME * 5, 0.1, dtype=np.float32)
    for chunk in np.array_split(audio, 7):  # odd block sizes
        r.hf.feed(chunk)
    assert r.det.idx == 25 and r.wakes

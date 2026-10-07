"""Hands-free session controller: wake word -> dictation -> "transcribe stop / send / cancel" (SPEC §6.3).

The controller only decides; it never records, transcribes or types. The orchestrator wires it up:

* ``feed(block)`` with every mic block (any size, int16 or float [-1, 1]) while hands-free is on.
  While listening, the wake detector runs on it (VAD-gated inside the detector) and a pre-roll ring
  keeps the last ``handsfree.preroll_s`` seconds.
* On a wake hit the controller calls ``on_wake(WakeStart)``. The orchestrator plays the earcon,
  starts a dictation capture seeded with ``WakeStart.preroll`` (so the first words are not
  clipped), and from then on passes each finished phrase transcript to ``on_phrase(text)``.
* ``on_phrase`` checks the first phrase for the wake word (a false wake ends the session silently),
  strips it, and looks for a trailing control phrase (text/commands.py). When the session ends
  -- by command, ``idle_timeout_s`` of silence, the session cap, or ``finish()/cancel()`` -- the
  controller calls ``on_end(SessionEnd)`` exactly once. ``SessionEnd.insert`` / ``press_enter`` say
  what to do with ``SessionEnd.text``.
* Wake hits during a session "arm" control detection for a few seconds: the audio said the wake
  word, so a rougher transcript ("transcript stop", or a bare "Stop." when the segmenter cut
  between the two words) still counts.
* While another app holds the microphone (a call), listening pauses (wake/calls.py; Windows only
  for now) and ``on_pause(True, apps)`` fires so the UI can say so.

Callbacks run on the calling thread (the audio thread for ``feed``, the STT worker for
``on_phrase``) after the internal lock is released, so they may call back into the controller.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from openwhisprflow.text.commands import (
    Action,
    ends_with_phrase,
    parse_control,
    starts_with_phrase,
    strip_leading_phrase,
)
from openwhisprflow.wake.frontend import FRAME, SAMPLE_RATE, to_int16_scale

log = logging.getLogger("openwhisprflow.wake.handsfree")

FRAME_S = FRAME / SAMPLE_RATE


class DetectorLike(Protocol):
    """What the controller needs from wake/detector.py's WakeDetector (fakes in tests)."""

    last_voiced: bool

    @property
    def has_vad(self) -> bool: ...
    def process(self, block: np.ndarray) -> list[Any]: ...  # hits with .score, .rise_frame, .frame
    def reset(self) -> None: ...
    def hold(self) -> None: ...


class MicMonitorLike(Protocol):
    def in_use(self) -> list[Any]: ...  # entries with .short (display name)


class HFState(StrEnum):
    OFF = "off"
    LISTENING = "listening"  # wake word armed
    PAUSED = "paused"        # another app is using the mic (a call)
    SESSION = "session"      # dictating


class EndReason(StrEnum):
    STOP = "stop"                  # "<phrase> stop" / "<phrase> done"
    SEND = "send"                  # "<phrase> send"
    CANCEL = "cancel"              # "<phrase> cancel", or cancel() (Escape / UI)
    IDLE_TIMEOUT = "idle_timeout"  # handsfree.idle_timeout_s of silence: insert what was said
    MAX_DURATION = "max_duration"  # audio.max_session_s reached: insert
    FALSE_WAKE = "false_wake"      # first phrase did not start with the wake word: drop silently
    EXTERNAL = "external"          # finish() from the UI / hotkey: insert


@dataclass
class WakeStart:
    """Passed to ``on_wake``: everything the orchestrator needs to start capturing."""

    session_id: int
    preroll: np.ndarray            # int16 mono 16 kHz, from the start of the wake word's speech run
    score: float
    clean_start: bool              # pre-roll starts in silence (VAD found the gap before the word)
    at: float                      # clock() at the hit


@dataclass
class SessionEnd:
    session_id: int
    reason: EndReason
    text: str                      # joined phrases (control phrases and the wake word stripped)
    phrases: list[str]
    duration_s: float
    wake_score: float
    detail: str = ""               # the false-wake transcript, the command span, ...

    @property
    def insert(self) -> bool:
        return self.reason not in (EndReason.CANCEL, EndReason.FALSE_WAKE) and bool(self.text.strip())

    @property
    def press_enter(self) -> bool:
        return self.reason is EndReason.SEND and self.insert


@dataclass
class PhraseResult:
    """What ``on_phrase`` did with one transcript (the end itself is also delivered to ``on_end``)."""

    action: Action
    kept: str                      # text added to the session from this phrase ("" if none)
    session_text: str
    ended: SessionEnd | None = None
    ignored: bool = False          # no session (late phrase after the end, or stale session id)


@dataclass
class _Session:
    id: int
    started: float
    score: float
    clean_start: bool
    phrases: list[str] = field(default_factory=list)
    verified: bool = False
    last_activity: float = 0.0
    armed_until: float = 0.0
    dangling: tuple[int, str] | None = None   # (phrase index, text without the trailing wake word)


class HandsFreeController:
    def __init__(self, cfg: Any, detector: DetectorLike, *,
                 on_wake: Callable[[WakeStart], None],
                 on_end: Callable[[SessionEnd], None],
                 on_pause: Callable[[bool, list[str]], None] | None = None,
                 on_state: Callable[[HFState], None] | None = None,
                 mic_monitor: MicMonitorLike | None = None,
                 max_session_s: float = 600.0,
                 aliases: Sequence[str] = (),
                 verify: bool = True,
                 arm_s: float = 5.0,
                 post_session_cooldown_s: float = 1.5,
                 call_poll_s: float = 3.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """``cfg``: ``Config.handsfree`` (phrase, preroll_s, idle_timeout_s, pause_on_calls, enabled)."""
        self.cfg = cfg
        self.detector = detector
        self.on_wake_cb = on_wake
        self.on_end_cb = on_end
        self.on_pause_cb = on_pause
        self.on_state_cb = on_state
        self.mic_monitor = mic_monitor
        self.max_session_s = float(max_session_s)
        self.aliases = tuple(aliases)
        self.verify = verify
        self.arm_s = arm_s
        self.post_session_cooldown_s = post_session_cooldown_s
        self.call_poll_s = call_poll_s
        self.clock = clock
        self._lock = threading.RLock()
        self._pending = np.zeros(0, dtype=np.int16)
        self._idx = 0
        self._preroll_frames = max(1, math.ceil(float(cfg.preroll_s) / FRAME_S))
        # extra room: the pre-roll may reach ~1.2 s before the word's end when the gate fired late
        self._ring: deque[tuple[int, np.ndarray, bool]] = deque(maxlen=self._preroll_frames + 25)
        self._session: _Session | None = None
        self._next_id = 1
        self._quiet_until = 0.0
        self._next_call_poll = 0.0
        self._paused_apps: list[str] = []
        self._last_end: SessionEnd | None = None
        self.state = HFState.LISTENING if getattr(cfg, "enabled", True) else HFState.OFF

    # ------------------------------------------------------------------ public API
    @property
    def phrase(self) -> str:
        return str(self.cfg.phrase)

    @property
    def in_session(self) -> bool:
        return self._session is not None

    @property
    def session_text(self) -> str:
        s = self._session
        return _join(s.phrases) if s else ""

    def set_enabled(self, enabled: bool) -> None:
        """Turn listening on/off. Turning it off during a session cancels the session."""
        calls: list[Callable[[], None]] = []
        with self._lock:
            if not enabled:
                if self._session is not None:
                    calls += self._end(EndReason.CANCEL, "hands-free disabled")
                calls += self._set_state(HFState.OFF)
            elif self.state is HFState.OFF:
                self.detector.reset()
                self._ring.clear()
                calls += self._set_state(HFState.LISTENING)
        _run(calls)

    def feed(self, block: np.ndarray) -> None:
        """One mic block (any length). Runs the detector, the pre-roll ring and the session timers."""
        calls: list[Callable[[], None]] = []
        with self._lock:
            if self.state is HFState.OFF:
                return
            x = np.concatenate([self._pending, to_int16_scale(np.asarray(block).reshape(-1))
                                .clip(-32768, 32767).astype(np.int16)])
            n = len(x) // FRAME
            self._pending = x[n * FRAME:]
            for k in range(n):
                calls += self._frame(x[k * FRAME:(k + 1) * FRAME])
            calls += self._timers()
        _run(calls)

    def on_phrase(self, text: str, session_id: int | None = None) -> PhraseResult:
        """A finished phrase transcript from the current session. Returns what was done with it."""
        calls: list[Callable[[], None]] = []
        with self._lock:
            res = self._phrase(text, session_id, calls)
        _run(calls)
        return res

    def finish(self, send: bool = False) -> SessionEnd | None:
        """End the session from outside (UI button, hotkey) and insert what was said."""
        return self._external(EndReason.SEND if send else EndReason.EXTERNAL)

    def cancel(self) -> SessionEnd | None:
        """Discard the session (Escape / UI ×)."""
        return self._external(EndReason.CANCEL)

    def tick(self) -> None:
        """Check timers without audio (the orchestrator may call this from a UI timer)."""
        with self._lock:
            calls = self._timers()
        _run(calls)

    # ------------------------------------------------------------------ audio
    def _frame(self, frame: np.ndarray) -> list[Callable[[], None]]:
        self._idx += 1
        now = self.clock()
        calls: list[Callable[[], None]] = []
        if self.state is HFState.PAUSED:
            return calls  # no inference, no pre-roll while a call holds the mic
        hits = self.detector.process(frame)
        voiced = bool(getattr(self.detector, "last_voiced", True))
        self._ring.append((self._idx, frame, voiced))
        s = self._session
        if s is not None:
            if voiced and self.detector.has_vad:
                s.last_activity = now
            if hits:
                s.armed_until = now + self.arm_s
                log.debug("wake word during session (score %.2f): control phrases armed", hits[-1].score)
            return calls
        if hits and now >= self._quiet_until:
            hit = hits[-1]
            start, clean = self._preroll_start(getattr(hit, "rise_frame", self._idx))
            pre = [f for i, f, _ in self._ring if i >= start]
            preroll = np.concatenate(pre) if pre else np.zeros(0, np.int16)
            sid = self._next_id
            self._next_id += 1
            self._session = _Session(sid, now, float(hit.score), clean, last_activity=now)
            log.info("wake word (score %.2f): session %d, pre-roll %.2f s%s", hit.score, sid,
                     len(preroll) / SAMPLE_RATE, "" if clean else " (starts mid-speech)")
            start_info = WakeStart(sid, preroll, float(hit.score), clean, now)
            calls += self._set_state(HFState.SESSION)
            calls.append(lambda: self.on_wake_cb(start_info))
        return calls

    def _preroll_start(self, anchor: int) -> tuple[int, bool]:
        """First frame of the wake word's speech run, capped to the pre-roll (a backward scan).

        Without a VAD the whole pre-roll is used and ``clean`` is False. With one, the scan walks back
        from now; once it has passed voiced audio at/before ``anchor`` (the score's rise, ~the end of
        the word), ~0.4 s of silence marks the start of the run.
        """
        idx = self._idx
        floor = idx - self._preroll_frames + 1
        if not self.detector.has_vad:
            return floor, False
        anchor = min(anchor, idx)
        word_end = next((i for i, _f, v in reversed(self._ring) if i <= anchor and v), anchor)
        floor = min(floor, word_end - math.ceil(1.2 / FRAME_S))
        quiet, heard = 0, False
        for i, _f, v in reversed(self._ring):
            if i < floor:
                break
            if not v:
                quiet += 1
                if quiet >= 5 and heard:
                    return max(floor, i + quiet - 2), True  # keep 2 quiet frames (160 ms) of lead-in
            else:
                quiet = 0
                if i <= anchor:
                    heard = True
        return max(floor, self._ring[0][0] if self._ring else floor), False

    # ------------------------------------------------------------------ timers / calls
    def _timers(self) -> list[Callable[[], None]]:
        now = self.clock()
        s = self._session
        if s is not None:
            if now - s.started >= self.max_session_s:
                return self._end(EndReason.MAX_DURATION, f"{self.max_session_s:.0f} s cap")
            if now - s.last_activity >= float(self.cfg.idle_timeout_s):
                return self._end(EndReason.IDLE_TIMEOUT, f"{self.cfg.idle_timeout_s} s without speech")
            return []
        return self._poll_calls(now)

    def _poll_calls(self, now: float) -> list[Callable[[], None]]:
        if self.mic_monitor is None or not getattr(self.cfg, "pause_on_calls", True) or now < self._next_call_poll:
            return []
        self._next_call_poll = now + self.call_poll_s
        apps = [str(getattr(e, "short", e)) for e in self.mic_monitor.in_use()]
        calls: list[Callable[[], None]] = []
        if apps and self.state is HFState.LISTENING:
            log.info("pausing hands-free: microphone in use by %s", ", ".join(apps))
            self._paused_apps = apps
            self.detector.reset()
            self._ring.clear()
            calls += self._set_state(HFState.PAUSED)
            if self.on_pause_cb:
                calls.append(lambda: self.on_pause_cb(True, apps))  # type: ignore[misc]
        elif not apps and self.state is HFState.PAUSED:
            log.info("resuming hands-free: microphone free")
            self._paused_apps = []
            self.detector.reset()
            calls += self._set_state(HFState.LISTENING)
            if self.on_pause_cb:
                calls.append(lambda: self.on_pause_cb(False, []))  # type: ignore[misc]
        return calls

    # ------------------------------------------------------------------ phrases
    def _phrase(self, text: str, session_id: int | None, calls: list[Callable[[], None]]) -> PhraseResult:
        s = self._session
        if s is None or (session_id is not None and session_id != s.id):
            return PhraseResult(Action.NONE, "", "", ignored=True)
        now = self.clock()
        s.last_activity = now
        raw = (text or "").strip()
        if not raw:
            return PhraseResult(Action.NONE, "", _join(s.phrases))
        armed = now < s.armed_until
        kw = {"aliases": self.aliases}
        frags = 0 if s.clean_start else 1

        first = not s.verified
        if first:
            if self.verify and not starts_with_phrase(raw, self.phrase, max_fragments=frags, **kw):
                log.info("false wake (score %.2f), transcript: %r", s.score, raw)
                end = self._end(EndReason.FALSE_WAKE, raw)
                calls += end
                return PhraseResult(Action.CANCEL, "", "", ended=self._last_end)
            s.verified = True

        # "... transcribe." | "Stop." : the segmenter cut between the wake word and the command
        if s.dangling is not None:
            d_idx, d_text = s.dangling
            s.dangling = None
            bare = parse_control(raw, self.phrase, armed=True, **kw)
            if bare.is_command and not bare.text:
                if d_text:
                    s.phrases[d_idx] = d_text
                else:
                    del s.phrases[d_idx]
                return self._apply(s, bare.action, "", bare.matched, calls)

        ctl = parse_control(raw, self.phrase, armed=armed, **kw)
        body = ctl.text if ctl.is_command else raw
        if first:
            body = strip_leading_phrase(body, self.phrase, max_fragments=frags, **kw)
        return self._apply(s, ctl.action, body, ctl.matched, calls)

    def _apply(self, s: _Session, action: Action, body: str, matched: str,
               calls: list[Callable[[], None]]) -> PhraseResult:
        kept = ""
        if action is Action.SCRATCH:
            if not body and s.phrases:
                dropped = s.phrases.pop()
                log.info("scratch that: dropped %r", dropped)
            # with body: the words said just before "scratch that" are the phrase being scratched
            return PhraseResult(action, "", _join(s.phrases))
        if action is Action.CANCEL:
            calls += self._end(EndReason.CANCEL, matched)
            return PhraseResult(action, "", "", ended=self._last_end)
        if body:
            s.phrases.append(body)
            kept = body
        if action in (Action.FINISH, Action.SEND):
            reason = EndReason.SEND if action is Action.SEND else EndReason.STOP
            calls += self._end(reason, matched)
            return PhraseResult(action, kept, self._last_end.text if self._last_end else "", ended=self._last_end)
        if body:
            dangling, rest = ends_with_phrase(body, self.phrase, aliases=self.aliases)
            if dangling:
                s.dangling = (len(s.phrases) - 1, rest)
        return PhraseResult(Action.NONE, kept, _join(s.phrases))

    # ------------------------------------------------------------------ ending
    def _external(self, reason: EndReason) -> SessionEnd | None:
        with self._lock:
            if self._session is None:
                return None
            calls = self._end(reason, "external")
            end = self._last_end
        _run(calls)
        return end

    def _end(self, reason: EndReason, detail: str = "") -> list[Callable[[], None]]:
        s = self._session
        if s is None:
            return []
        self._session = None
        now = self.clock()
        phrases = list(s.phrases)
        end = SessionEnd(s.id, reason, _join(phrases), phrases, now - s.started, s.score, detail)
        self._last_end = end
        log.info("session %d ended: %s (%d phrases, %.1f s)", s.id, reason.value, len(phrases), end.duration_s)
        # don't let the session's own last words ("transcribe stop") wake it again
        self.detector.reset()
        self.detector.hold()
        self._quiet_until = now + self.post_session_cooldown_s
        calls = self._set_state(HFState.LISTENING if self.state is not HFState.OFF else HFState.OFF)
        calls.append(lambda: self.on_end_cb(end))
        return calls

    def _set_state(self, state: HFState) -> list[Callable[[], None]]:
        if state is self.state:
            return []
        self.state = state
        if self.on_state_cb is None:
            return []
        cb = self.on_state_cb
        return [lambda: cb(state)]


def _join(phrases: list[str]) -> str:
    return " ".join(p.strip() for p in phrases if p.strip())


def _run(calls: list[Callable[[], None]]) -> None:
    for c in calls:
        try:
            c()
        except Exception:  # a broken callback must never kill the audio thread
            log.exception("hands-free callback failed")


def controller_from_config(cfg: Any, *, on_wake: Callable[[WakeStart], None], on_end: Callable[[SessionEnd], None],
                           is_speech: Callable[[np.ndarray], bool] | None = None, provider: str = "cpu",
                           **kwargs: Any) -> HandsFreeController:
    """Build detector + controller from a full ``Config`` (handsfree section + audio.max_session_s)."""
    from openwhisprflow.wake.calls import MicUsageMonitor
    from openwhisprflow.wake.detector import detector_from_config

    hf = cfg.handsfree
    det = detector_from_config(hf, is_speech=is_speech, provider=provider)
    kwargs.setdefault("mic_monitor", MicUsageMonitor() if hf.pause_on_calls else None)
    kwargs.setdefault("max_session_s", float(cfg.audio.max_session_s))
    return HandsFreeController(hf, det, on_wake=on_wake, on_end=on_end, **kwargs)

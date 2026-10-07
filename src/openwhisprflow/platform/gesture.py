"""The Voice key gesture as a pure, clock-injected state machine (SPEC §6.1).

A port of earlier native gesture code (Windows and macOS), reshaped for
the :class:`~openwhisprflow.platform.base.Gesture` contract. It only decides what a key transition
*means*; the OS adapters own the hooks, the clock and the replays. Nothing here touches the OS,
so every edge case is unit-tested (``tests/test_gesture.py``).

Gestures emitted, in the order the app sees them:

* hold:        ``PRESS`` on key down (recording starts at once so no audio is lost), then
               ``RELEASE`` when a press of at least ``hold_min_ms`` ends.
* double tap:  ``PRESS`` on the first down; a short first press leaves recording running and
               waits; a second down within ``double_tap_ms`` emits ``LOCK``. The next full tap
               (its release) emits ``FINISH``.
* lone tap:    ``PRESS``, then ``CANCEL`` once the double-tap window expires (the take is
               discarded) and the tap is *replayed* to the system, so a quick tap of Caps Lock
               or Right Alt still does what it always did.
* ``CANCEL``:  Escape (the cancel key) while a take is running.
* ``RAW``:     emitted *instead of* ``RELEASE``/``FINISH`` when the raw modifier is held at the
               moment the take finishes (finish without refinement).

Swallowing rules (the "never jam the keyboard" hardening):

* A press is swallowed only when this machine takes ownership of it; its repeats and its
  release are then swallowed too. A press that is passed through (ineligible, or part of a
  chord) has its release passed through as well, so the OS always sees matched pairs.
* When another (non-modifier) key goes down while the Voice key is held before the hold
  threshold, the Voice key was being used as a modifier (AltGr+e, Right Ctrl+C, ...): the take
  is cancelled and the swallowed press is replayed so the chord works.
* A release we never saw pressed (hook installed mid-press) passes through untouched.
* Escape is swallowed only while a take is running, and its release only if its press was.
* A press that arrives while the key is believed down, after more than ``repeat_gap_ms`` of
  silence, means a release was missed (hook dropped by the OS, session switch...). It is
  treated as release + new press instead of an endless "repeat" that would wedge the gesture.

Times are integer milliseconds on any monotonic clock the caller chooses; adapters should pass
the *event's* timestamp, not the time the hook ran (a late hook must not turn a double tap into
two single taps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from openwhisprflow.platform.base import Gesture


class Phase(str, Enum):
    IDLE = "idle"
    PRESSED = "pressed"      # key down, not yet long enough to be a hold
    HELD = "held"            # key down past hold_min_ms: release finishes
    WAITING = "waiting"      # a short first tap ended; recording runs while we wait for a second
    LOCKED = "locked"        # double-tapped (or started externally): a tap finishes


@dataclass
class Decision:
    """What one input means.

    ``replay`` lists Voice key events the adapter must send to the OS *before* letting the
    current event continue (``True`` = key down, ``False`` = key up).
    """

    swallow: bool = False
    gestures: list[Gesture] = field(default_factory=list)
    replay: list[bool] = field(default_factory=list)

    def merge(self, other: "Decision") -> "Decision":
        return Decision(self.swallow or other.swallow, self.gestures + other.gestures,
                        self.replay + other.replay)


def safe_swallow(swallow: bool, down: bool, system_down: bool) -> bool:
    """Whether an event the machine wants hidden may really be hidden ("safe swallow").

    A press may always be hidden. A release may only be hidden when the system never saw the
    press: if the press leaked through (Windows skips a hook that answers too slowly), hiding
    the release would leave the key held down for every app until it is pressed again.
    ``system_down`` is the OS's view of the key from inside the hook, i.e. before this event.
    """
    return swallow and (down or not system_down)


def event_time(now64: int, now32: int, event32: int) -> int:
    """Map a 32-bit event tick (Windows ``KBDLLHOOKSTRUCT.time``) onto the 64-bit clock.

    Gesture timing must use when the key moved, not when the hook got to run. Implausibly old
    stamps (over 60 s, or before the clock began) fall back to ``now64``.
    """
    age = (now32 - event32) & 0xFFFFFFFF  # wraps correctly
    return now64 - age if age <= 60000 and age <= now64 else now64


class GestureMachine:
    """Turns Voice key / other key / Escape transitions into :class:`Gesture` events.

    Not thread-safe: the driver serializes calls under its lock.
    """

    def __init__(self, double_tap_ms: int = 350, hold_min_ms: int = 250, *,
                 replay_taps: bool = True, repeat_gap_ms: int = 1500) -> None:
        self.double_tap_ms = double_tap_ms
        self.hold_min_ms = hold_min_ms
        self.replay_taps = replay_taps
        self.repeat_gap_ms = repeat_gap_ms
        self.phase = Phase.IDLE
        self.recording = False        # set by the app: a take (or its processing) is running
        self.from_key = False         # the app's current take was started by this machine
        self.external = False         # a take started elsewhere (UI, tray): a tap finishes it
        self.key_down = False         # the Voice key's last seen state
        self.owned = False            # we swallowed the current press; its release is ours too
        self.forwarding = False       # the current press went to the OS; so must its release
        self.latch_release = False    # the release of the locking second tap is ignored
        self.escape_owned = False     # we swallowed Escape's press; swallow its release too
        self.at = 0                   # when the current phase started
        self.last_key_at = 0          # last Voice key event (press, repeat or release)

    # ------------------------------------------------------------------ queries

    @property
    def idle(self) -> bool:
        """No gesture in progress and the key is not held (safe to change the key)."""
        return self.phase == Phase.IDLE and not self.key_down

    @property
    def involved(self) -> bool:
        """Whether this machine has a hand in the key's OS state right now."""
        return self.key_down and (self.owned or self.forwarding)

    def next_deadline(self) -> int | None:
        """When :meth:`tick` next has something to do, or None."""
        if self.phase == Phase.PRESSED:
            return self.at + self.hold_min_ms
        if self.phase == Phase.WAITING:
            return self.at + self.double_tap_ms + 1
        return None

    # ------------------------------------------------------------------ inputs

    def tick(self, now: int) -> Decision:
        if self.phase == Phase.PRESSED and now - self.at >= self.hold_min_ms:
            self.phase = Phase.HELD
            return Decision()
        if self.phase == Phase.WAITING and now - self.at > self.double_tap_ms:
            return self._expire()
        return Decision()

    def key(self, down: bool, now: int, *, eligible: bool = True, raw: bool = False) -> Decision:
        """One Voice key transition. ``eligible``: no conflicting modifier is held (so this is
        dictation, not a shortcut). ``raw``: the raw modifier is held right now."""
        if down and self.key_down:
            if now - self.last_key_at <= self.repeat_gap_ms:
                self.last_key_at = now
                return Decision(swallow=self.owned and not self.forwarding)  # auto-repeat
            # A press while "down" after a long silence: the release was missed. Recover.
            return self.key(False, now, eligible=eligible, raw=raw).merge(
                self.key(True, now, eligible=eligible, raw=raw))
        if not down and not self.key_down:
            return Decision()  # a release whose press we never saw: not ours
        self.key_down = down
        self.last_key_at = now
        if self.forwarding:
            if not down:
                self.forwarding = False
            return Decision()
        return self._press(now, eligible) if down else self._release(now, raw)

    def other(self, down: bool, now: int, *, escape: bool = False) -> Decision:
        """Any other non-modifier key (modifiers must not be fed here: the raw modifier and
        AltGr's synthesized Ctrl would otherwise break gestures)."""
        if escape:
            return self._escape(down)
        if not down:
            return Decision()
        if self.phase == Phase.WAITING:
            return self._expire()  # typing after a lone tap: it was just a tap
        if not self.key_down or self.forwarding:
            return Decision()
        # The Voice key is held and another key went down: it was a modifier chord.
        self.forwarding = True
        self.owned = False
        if self.phase == Phase.LOCKED:
            return Decision(replay=[True])  # the finishing tap became a chord; stay locked
        self.phase = Phase.IDLE
        return Decision(gestures=[Gesture.CANCEL], replay=[True])

    def set_recording(self, recording: bool) -> None:
        """The app's view: True while a take runs or is being processed, so Escape cancels it.

        A take the app started itself (UI, tray, wake word) becomes *external*: one Voice key
        tap finishes it. This never changes the gesture phase: a late "not recording" for the
        previous take must not reset the take the user is holding now, and a first tap waiting
        for its second belongs to the key (resetting it made double taps miss).
        """
        self.recording = recording
        if not recording:
            self.from_key = False
            self.external = False
        elif self.phase == Phase.IDLE and not self.from_key:
            self.external = True

    def reset(self) -> None:
        """Forget everything (listener stopped or restarted)."""
        self.__init__(self.double_tap_ms, self.hold_min_ms, replay_taps=self.replay_taps,  # type: ignore[misc]
                      repeat_gap_ms=self.repeat_gap_ms)

    # ------------------------------------------------------------------ transitions

    def _press(self, now: int, eligible: bool) -> Decision:
        prefix = Decision()
        if self.phase == Phase.WAITING:
            if now - self.at <= self.double_tap_ms and eligible:
                self.phase = Phase.LOCKED
                self.latch_release = True
                self.owned = True
                return Decision(swallow=True, gestures=[Gesture.LOCK])
            prefix = self._expire()  # too late for a double tap: the first was a lone tap
        if self.phase == Phase.LOCKED or (self.phase == Phase.IDLE and self.external):
            # The tap that will finish a locked (or externally started) take.
            self.phase = Phase.LOCKED
            self.latch_release = False
            self.owned = True
            return prefix.merge(Decision(swallow=True))
        if not eligible:
            self.forwarding = True
            self.owned = False
            return prefix
        self.phase = Phase.PRESSED
        self.at = now
        self.owned = True
        self.from_key = True
        return prefix.merge(Decision(swallow=True, gestures=[Gesture.PRESS]))

    def _release(self, now: int, raw: bool) -> Decision:
        owned, self.owned = self.owned, False
        def finish(normal: Gesture) -> list[Gesture]:
            return [Gesture.RAW if raw else normal]

        if self.phase == Phase.PRESSED:
            if now - self.at >= self.hold_min_ms:
                self.phase = Phase.IDLE
                return Decision(swallow=True, gestures=finish(Gesture.RELEASE))
            self.phase = Phase.WAITING
            self.at = now
            return Decision(swallow=True)
        if self.phase == Phase.HELD:
            self.phase = Phase.IDLE
            return Decision(swallow=True, gestures=finish(Gesture.RELEASE))
        if self.phase == Phase.LOCKED:
            if self.latch_release:
                self.latch_release = False
                return Decision(swallow=True)
            self.phase = Phase.IDLE
            self.external = False
            return Decision(swallow=True, gestures=finish(Gesture.FINISH))
        return Decision(swallow=owned)  # e.g. cancelled by Escape mid-hold: release stays hidden

    def _escape(self, down: bool) -> Decision:
        if not down:
            owned, self.escape_owned = self.escape_owned, False
            return Decision(swallow=owned)
        if self.escape_owned:
            return Decision(swallow=True)  # Escape auto-repeat after we cancelled
        if self.phase != Phase.IDLE or self.recording:
            self.phase = Phase.IDLE
            self.latch_release = False
            self.external = False
            self.escape_owned = True
            return Decision(swallow=True, gestures=[Gesture.CANCEL])
        return Decision()  # nothing running: Escape belongs to the app

    def _expire(self) -> Decision:
        self.phase = Phase.IDLE
        return Decision(gestures=[Gesture.CANCEL], replay=[True, False] if self.replay_taps else [])

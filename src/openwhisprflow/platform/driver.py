"""Shared plumbing behind every OS hotkey listener.

The OS adapters only translate native key events into sided key names (see :mod:`keys`) and
apply the swallow answer. Everything else lives here, once:

* **Routing**: is this the Voice key, the cancel key, a modifier or some other key; is the
  press eligible (no conflicting modifier = dictation, not a shortcut); is the raw modifier
  held.
* **Locking**: hook threads call :meth:`GestureDriver.feed` and must get an answer in
  microseconds, so the lock is held only around the pure state machine.
* **Timers**: a ticker thread wakes at the machine's next deadline (hold threshold, double-tap
  window) instead of polling.
* **Dispatch**: gestures go through a queue to a dispatcher thread that calls ``on_gesture``.
  The hook callback never runs app code, because Windows (and macOS) silently drop a hook that
  answers too slowly, which can leave a key (e.g. Right Alt) stuck for every app.
* **Replays**: Voice key presses the machine swallowed but turned out to belong to the OS
  (a lone tap, a chord) are handed to the adapter's ``replay`` callback, outside the lock.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

from openwhisprflow.config import HotkeyConfig
from openwhisprflow.platform.base import Gesture, GestureFn
from openwhisprflow.platform.gesture import Decision, GestureMachine
from openwhisprflow.platform.keys import KeySpec, modifier_class, normalize, parse_key, parse_modifier

log = logging.getLogger("openwhisprflow.hotkeys")

ReplayFn = Callable[[str, bool], None]  # (key name, down)


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


class GestureDriver:
    """Thread-safe owner of a :class:`GestureMachine` for one OS adapter."""

    def __init__(self, cfg: HotkeyConfig, replay: ReplayFn | None = None,
                 clock_ms: Callable[[], int] = monotonic_ms, *, replay_taps: bool = True,
                 validate: Callable[[KeySpec], None] | None = None) -> None:
        self._validate = validate or (lambda spec: None)
        self.spec = parse_key(cfg.key)
        self._validate(self.spec)
        self._pending: KeySpec | None = None
        self.raw_mod = parse_modifier(cfg.raw_modifier)
        self.cancel_key = normalize(cfg.cancel_key or "escape")
        self.machine = GestureMachine(cfg.double_tap_ms, cfg.hold_min_ms, replay_taps=replay_taps)
        self.held: set[str] = set()       # sided modifier names physically down
        self.clock_ms = clock_ms
        self._replay = replay or (lambda name, down: None)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue: queue.SimpleQueue[Gesture | None] = queue.SimpleQueue()
        self._on_gesture: GestureFn | None = None
        self._threads: list[threading.Thread] = []
        self._running = False

    # ------------------------------------------------------------------ lifecycle

    def start(self, on_gesture: GestureFn) -> None:
        self._on_gesture = on_gesture
        self._running = True
        self._threads = [threading.Thread(target=self._dispatch_loop, name="owf-gestures", daemon=True),
                         threading.Thread(target=self._tick_loop, name="owf-gesture-timer", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        with self._wake:
            self._running = False
            self._wake.notify_all()
        self._queue.put(None)
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(1.0)
        self._threads = []

    # ------------------------------------------------------------------ app-facing

    def set_key(self, key: str) -> None:
        """Validate now (raises ValueError); takes effect between gestures only, so a take in
        progress keeps the key it started with."""
        spec = parse_key(key)
        self._validate(spec)
        with self._lock:
            self._pending = spec
            self._apply_pending()

    def set_recording(self, recording: bool) -> None:
        with self._wake:
            self.machine.set_recording(recording)
            self._wake.notify_all()

    @property
    def key_name(self) -> str:
        return self.spec.key

    # ------------------------------------------------------------------ adapter-facing

    def feed(self, name: str, down: bool, now: int | None = None) -> bool:
        """One native key event, already translated to a key name. Returns swallow.

        Called on the hook thread: never blocks beyond the lock, never raises.
        """
        try:
            with self._wake:
                if now is None:
                    now = self.clock_ms()
                key = self.spec.key
                decision = self._route(name, down, now)
                self._post(decision)
                self._apply_pending()
                self._wake.notify_all()
            self._send_replays(key, decision)
            return decision.swallow
        except Exception:  # never break the user's keyboard
            log.exception("hotkey routing failed")
            return False

    def involved(self) -> bool:
        with self._lock:
            return self.machine.involved

    def reset(self) -> None:
        with self._lock:
            self.machine.reset()
            self.held.clear()

    # ------------------------------------------------------------------ internals

    def _route(self, name: str, down: bool, now: int) -> Decision:
        cls = modifier_class(name)
        if cls is not None:
            if down:
                self.held.add(name)
            else:
                self.held.discard(name)
        spec = self.spec
        if name == spec.key:
            return self.machine.key(down, now, eligible=self._eligible(spec), raw=self._raw(spec))
        if name == self.cancel_key:
            return self.machine.other(down, now, escape=True)
        if cls is not None:
            return Decision()  # modifiers never break a gesture (raw modifier, AltGr's Ctrl)
        return self.machine.other(down, now)

    def _held_classes(self, spec: KeySpec) -> set[str]:
        return {modifier_class(n) for n in self.held if n != spec.key}  # type: ignore[misc]

    def _eligible(self, spec: KeySpec) -> bool:
        """A single key: no other modifier held. A chord: exactly its modifiers held."""
        return self._held_classes(spec) == set(spec.modifiers)

    def _raw(self, spec: KeySpec) -> bool:
        mod = self.raw_mod
        if mod is None or mod in spec.modifiers or modifier_class(spec.key) == mod:
            return False
        return mod in self._held_classes(spec)

    def _apply_pending(self) -> None:
        if self._pending is not None and self.machine.idle:
            self.spec, self._pending = self._pending, None

    def _post(self, decision: Decision) -> None:
        for g in decision.gestures:
            self._queue.put(g)

    def _send_replays(self, key: str, decision: Decision) -> None:
        for down in decision.replay:
            try:
                self._replay(key, down)
            except Exception:
                log.exception("replaying %s failed", key)

    def _tick_loop(self) -> None:
        while True:
            with self._wake:
                if not self._running:
                    return
                deadline = self.machine.next_deadline()
                now = self.clock_ms()
                if deadline is None or deadline > now:
                    wait = 0.5 if deadline is None else (deadline - now) / 1000
                    self._wake.wait(timeout=min(wait, 0.5))
                    continue
                key = self.spec.key
                decision = self.machine.tick(now)
                self._post(decision)
                self._apply_pending()
            self._send_replays(key, decision)

    def _dispatch_loop(self) -> None:
        while True:
            g = self._queue.get()
            if g is None:
                return
            fn = self._on_gesture
            if fn is None:
                continue
            try:
                fn(g)
            except Exception:
                log.exception("on_gesture(%s) failed", g)

"""macOS Voice key listener: a CGEventTap through pyobjc on its own CFRunLoop thread.

Default key: Right Option (kVK_RightOption, 61),
told apart from Left Option by the NX device-specific flag bits, the only way macOS exposes
sides. Only the Voice key's own events (and Escape while a take runs) are ever swallowed, and
only while a gesture owns them: a key pressed with Right Option held still reaches the app
with the Option flag, so Option-character entry (Option+e, ...) keeps working and the
gesture bows out (the chord rule in :mod:`gesture`).

Requirements and limitations (shown by :mod:`permissions` during onboarding):

* The tap needs **Accessibility** (to swallow) and **Input Monitoring** (to listen). Without
  them ``CGEventTapCreate`` returns NULL and :meth:`start` raises ``PermissionError``. The grant
  belongs to the binary that runs us (the Terminal, or the packaged app).
* **Fn / Globe** ("fn") is detected from its flagsChanged event, but macOS acts on Globe below
  the event tap: set System Settings > Keyboard > "Press 🌐 key to" > "Do Nothing", or the
  emoji picker / input switch fires as well.
* **Caps Lock** cannot be a hold key (macOS reports its toggle, never its release).
* When the system disables the tap for being slow, it is re-enabled at once.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from openwhisprflow.config import HotkeyConfig
from openwhisprflow.platform.base import GestureFn
from openwhisprflow.platform.driver import GestureDriver
from openwhisprflow.platform.keys import MAC_KEYCODE, MAC_UNSUPPORTED, KeySpec, mac_name

log = logging.getLogger("openwhisprflow.hotkeys.macos")

OWN_EVENT_TAG = 0x4F574652  # eventSourceUserData on our replays; the tap passes them through

KEY_DOWN, KEY_UP, FLAGS_CHANGED = 10, 11, 12
TAP_DISABLED_BY_TIMEOUT, TAP_DISABLED_BY_USER_INPUT = 0xFFFFFFFE, 0xFFFFFFFF

# NX device-dependent modifier bits (IOLLEvent.h), plus the device-independent one for Fn.
DEVICE_BITS: dict[str, int] = {
    "left_ctrl": 0x1, "left_shift": 0x2, "right_shift": 0x4, "left_win": 0x8, "right_win": 0x10,
    "left_alt": 0x20, "right_alt": 0x40, "right_ctrl": 0x2000,
    "fn": 0x800000,
}
# Device-independent masks for replayed modifier presses.
CLASS_MASK: dict[str, int] = {"shift": 0x20000, "ctrl": 0x40000, "alt": 0x80000, "win": 0x100000}


def modifier_down(name: str, flags: int) -> bool:
    """Whether a modifier's own flagsChanged event is a press or a release."""
    bit = DEVICE_BITS.get(name)
    return bool(bit and flags & bit)


def event_down(event_type: int, name: str, flags: int) -> bool | None:
    """Press (True), release (False), or None for events we do not route."""
    if event_type == KEY_DOWN:
        return True
    if event_type == KEY_UP:
        return False
    if event_type == FLAGS_CHANGED:
        return modifier_down(name, flags) if name in DEVICE_BITS else None
    return None


def _validate(spec: KeySpec) -> None:
    if spec.key in MAC_UNSUPPORTED:
        raise ValueError(f"{spec.key!r} cannot be the Voice key on macOS: {MAC_UNSUPPORTED[spec.key]}")
    if spec.key not in MAC_KEYCODE:
        raise ValueError(f"{spec.key!r} cannot be used as the Voice key on macOS")


class MacHotkeyListener:
    """:class:`~openwhisprflow.platform.base.HotkeyListener` on a CGEventTap."""

    def __init__(self, cfg: HotkeyConfig) -> None:
        self._driver = GestureDriver(cfg, self._replay, validate=_validate)
        self._tap: Any = None
        self._loop: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def start(self, on_gesture: GestureFn) -> None:
        if self._thread is not None:
            return
        self._driver.start(on_gesture)
        self._thread = threading.Thread(target=self._run, name="owf-eventtap", daemon=True)
        self._thread.start()
        if not self._ready.wait(3.0) or self._error:
            err = self._error or OSError("event tap thread did not start")
            self.stop()
            raise err

    def set_key(self, key: str) -> None:
        self._driver.set_key(key)

    def set_recording(self, recording: bool) -> None:
        self._driver.set_recording(recording)

    def stop(self) -> None:
        import Quartz

        if self._loop is not None:
            Quartz.CFRunLoopStop(self._loop)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(2.0)
        self._thread = None
        self._loop = None
        name = self._driver.key_name
        code = MAC_KEYCODE.get(name)
        if code is not None and self._driver.involved() and Quartz.CGEventSourceKeyState(
                Quartz.kCGEventSourceStateCombinedSessionState, code):
            self._replay(name, False)
        self._driver.stop()
        self._driver.reset()

    # ------------------------------------------------------------------ tap thread

    def _run(self) -> None:
        import Quartz

        from openwhisprflow.platform import permissions

        mask = (1 << KEY_DOWN) | (1 << KEY_UP) | (1 << FLAGS_CHANGED)
        tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                      Quartz.kCGEventTapOptionDefault, mask, self._callback, None)
        if tap is None:
            missing = permissions.check()
            self._error = PermissionError(
                "macOS refused the keyboard event tap. " + " ".join(missing.values())
                if missing else "macOS refused the keyboard event tap (Accessibility / Input Monitoring).")
            self._ready.set()
            return
        self._tap = tap
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        self._ready.set()
        try:
            Quartz.CFRunLoopRun()
        finally:
            Quartz.CGEventTapEnable(tap, False)
            Quartz.CFRunLoopRemoveSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
            Quartz.CFMachPortInvalidate(tap)
            self._tap = None

    def _callback(self, proxy: Any, event_type: int, event: Any, refcon: Any) -> Any:
        import Quartz

        try:
            if event_type in (TAP_DISABLED_BY_TIMEOUT, TAP_DISABLED_BY_USER_INPUT):
                if self._tap is not None:
                    Quartz.CGEventTapEnable(self._tap, True)
                return event
            if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == OWN_EVENT_TAG:
                return event
            code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            flags = int(Quartz.CGEventGetFlags(event))
            name = mac_name(int(code))
            down = event_down(event_type, name, flags)
            self._recover_missed_release(name, flags)
            if down is None:
                return event
            return None if self._driver.feed(name, down) else event
        except Exception:  # never break the user's keyboard
            log.exception("event tap callback failed")
            return event

    def _recover_missed_release(self, name: str, flags: int) -> None:
        """Every event carries the full modifier state: if the Voice key is a modifier we
        believe is down but the flags say it is up, its release was lost (tap disabled for a
        moment). Feed the release so the gesture never wedges."""
        main = self._driver.key_name
        bit = DEVICE_BITS.get(main)
        if bit and name != main and self._driver.machine.key_down and not flags & bit:
            self._driver.feed(main, False)

    # ------------------------------------------------------------------ replays

    def _replay(self, name: str, down: bool) -> None:
        import Quartz

        code = MAC_KEYCODE.get(name)
        if code is None:
            return
        event = Quartz.CGEventCreateKeyboardEvent(None, code, down)
        flags = 0
        if down and name in DEVICE_BITS:
            from openwhisprflow.platform.keys import modifier_class

            cls = modifier_class(name)
            flags = DEVICE_BITS[name] | (CLASS_MASK.get(cls, 0) if cls else 0)
        Quartz.CGEventSetFlags(event, flags)
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, OWN_EVENT_TAG)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

"""Hotkey listener factory (SPEC §6.1).

``create(cfg)`` returns the :class:`~openwhisprflow.platform.base.HotkeyListener` for this OS.
``on_gesture`` is always called on a listener-owned thread (never the OS hook thread) and
should return quickly. Gesture semantics are documented in :mod:`openwhisprflow.platform.gesture`;
the app contract in short:

* ``PRESS`` start a take; ``LOCK`` mark it locked; ``RELEASE``/``FINISH`` finish it; ``RAW``
  finish it without refinement; ``CANCEL`` discard it (Escape, or a lone short tap of the key).
* Call ``set_recording(True)`` while a take runs or is being processed (so Escape cancels it,
  and a key tap finishes a take started from the UI), ``set_recording(False)`` once it is over.
* Ignore ``RELEASE``/``FINISH``/``RAW``/``CANCEL`` when nothing is running.
"""

from __future__ import annotations

import logging
import sys

from openwhisprflow.config import HotkeyConfig
from openwhisprflow.platform.base import GestureFn, HotkeyListener
from openwhisprflow.platform.keys import KeySpec, parse_key

log = logging.getLogger("openwhisprflow.hotkeys")

__all__ = ["create", "CompositeListener", "parse_key", "KeySpec", "toggle"]


class CompositeListener:
    """Several listeners behind one interface (Linux: the toggle socket plus a key backend).

    Listeners after the first are optional: if one cannot start, the reason is kept in
    :attr:`problems` (the UI shows it) and the rest keep working.
    """

    def __init__(self, required: HotkeyListener, *optional: HotkeyListener) -> None:
        self.listeners = [required, *optional]
        self.running: list[HotkeyListener] = []
        self.problems: list[str] = []

    def start(self, on_gesture: GestureFn) -> None:
        self.listeners[0].start(on_gesture)
        self.running = [self.listeners[0]]
        for listener in self.listeners[1:]:
            try:
                listener.start(on_gesture)
                self.running.append(listener)
            except Exception as e:
                log.warning("hotkey backend %s unavailable: %s", type(listener).__name__, e)
                self.problems.append(str(e))

    def set_key(self, key: str) -> None:
        for listener in self.listeners:
            listener.set_key(key)

    def set_recording(self, recording: bool) -> None:
        for listener in self.running:
            listener.set_recording(recording)

    def stop(self) -> None:
        for listener in reversed(self.running):
            try:
                listener.stop()
            except Exception:
                log.exception("stopping %s failed", type(listener).__name__)
        self.running = []


def create(cfg: HotkeyConfig | None = None, *, backend: str | None = None) -> HotkeyListener:
    """The listener for this OS. Raises ValueError for a key this OS cannot use."""
    cfg = cfg or HotkeyConfig()
    if sys.platform == "win32":
        from openwhisprflow.platform.hotkeys_windows import WindowsHotkeyListener

        return WindowsHotkeyListener(cfg)
    if sys.platform == "darwin":
        from openwhisprflow.platform.hotkeys_macos import MacHotkeyListener

        return MacHotkeyListener(cfg)
    from openwhisprflow.platform import hotkeys_linux as lx

    if backend == "toggle":
        return lx.ToggleListener()
    return CompositeListener(lx.ToggleListener(), lx.key_listener(cfg, backend))


def toggle(action: str = "toggle") -> None:
    """For ``openwhisprflow toggle [start|stop|cancel]``: drive the running app (Linux, macOS).

    Raises ConnectionError when the app is not running.
    """
    from openwhisprflow.platform.hotkeys_linux import send_toggle

    send_toggle(action)

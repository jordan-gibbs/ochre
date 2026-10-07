"""macOS text injection: Unicode key events, with an NSPasteboard + Cmd+V fallback.

* ``CGEventKeyboardSetUnicodeString`` carries at most 20 UTF-16 units per event (longer strings
  are silently truncated by many apps), so text goes out in chunks of <= 20 units that never
  split a surrogate pair. Newline and tab are real Return / Tab
  key presses. Posting requires the Accessibility permission.
* **Secure input**: while a password field (or an app such as a terminal with "Secure Keyboard
  Entry") holds secure input, macOS drops synthetic keystrokes. ``IsSecureEventInputEnabled``
  detects it and we raise InjectError instead of typing into nothing.
* The paste fallback saves every item and type on the general pasteboard, pastes with Cmd+V
  (key code 9, the ANSI "V" position: layouts that move V, like Dvorak, need the
  "Dvorak - QWERTY Cmd" input source for the shortcut to land), and restores the pasteboard
  unless something else wrote to it meanwhile.
"""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Any

from openwhisprflow.platform.base import FocusInfo, InjectError
from openwhisprflow.platform.focus_macos import get_focus
from openwhisprflow.platform.textunits import chunks, plan

log = logging.getLogger("openwhisprflow.inject.macos")

OWN_EVENT_TAG = 0x4F574654
KC_RETURN, KC_TAB, KC_V = 36, 48, 9
MASK_SHIFT, MASK_COMMAND = 0x20000, 0x100000
MAX_UNITS = 20

SECURE = ("macOS secure input is on (a password field, or an app with Secure Keyboard Entry), "
          "so typing is blocked. The text is in History.")


def secure_input_enabled() -> bool:
    try:
        carbon = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Carbon.framework/Carbon")
        fn = carbon.IsSecureEventInputEnabled
        fn.restype = ctypes.c_bool
        return bool(fn())
    except (OSError, AttributeError):
        return False


class MacInjector:
    """:class:`~openwhisprflow.platform.base.Injector` for macOS."""

    def __init__(self, *, chunk_gap_s: float = 0.003, paste_settle_s: float = 0.3,
                 newline: str = "enter") -> None:
        self.chunk_gap_s = chunk_gap_s
        self.paste_settle_s = paste_settle_s
        self.newline = newline

    def focus(self) -> FocusInfo:
        return get_focus()

    def type_text(self, text: str) -> None:
        if not text:
            return
        self._check()
        import Quartz

        source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        for kind, value in plan(text):
            if kind == "key":
                code = KC_TAB if value == "tab" else KC_RETURN
                flags = MASK_SHIFT if value == "enter" and self.newline == "shift_enter" else 0
                self._tap(source, code, flags)
            else:
                for chunk in chunks(value, MAX_UNITS):
                    for down in (True, False):
                        event = Quartz.CGEventCreateKeyboardEvent(source, 0, down)
                        Quartz.CGEventSetFlags(event, 0)  # held modifiers must not turn text into shortcuts
                        units = len(chunk.encode("utf-16-le")) // 2
                        Quartz.CGEventKeyboardSetUnicodeString(event, units, chunk)
                        self._post(event)
            time.sleep(self.chunk_gap_s)

    def paste_text(self, text: str) -> None:
        if not text:
            return
        self._check()
        import Quartz
        from AppKit import NSPasteboard, NSPasteboardTypeString

        board = NSPasteboard.generalPasteboard()
        saved = _save(board)
        board.clearContents()
        board.setString_forType_(text, NSPasteboardTypeString)
        change = board.changeCount()
        source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        self._tap(source, KC_V, MASK_COMMAND)
        time.sleep(self.paste_settle_s)
        if board.changeCount() == change:
            _restore(board, saved)

    def press_enter(self) -> None:
        self._check()
        import Quartz

        self._tap(Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState), KC_RETURN, 0)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _check() -> None:
        if secure_input_enabled():
            raise InjectError(SECURE)

    def _tap(self, source: Any, code: int, flags: int) -> None:
        import Quartz

        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(source, code, down)
            Quartz.CGEventSetFlags(event, flags)
            self._post(event)

    @staticmethod
    def _post(event: Any) -> None:
        import Quartz

        if event is None:
            raise InjectError("macOS refused to create a key event (is Accessibility granted?)")
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, OWN_EVENT_TAG)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def _save(board: Any) -> list[dict[str, Any]]:
    items = []
    for item in board.pasteboardItems() or []:
        data = {}
        for t in item.types() or []:
            d = item.dataForType_(t)
            if d is not None:
                data[str(t)] = d
        items.append(data)
    return items


def _restore(board: Any, items: list[dict[str, Any]]) -> None:
    from AppKit import NSPasteboardItem

    board.clearContents()
    if not items:
        return
    out = []
    for data in items:
        item = NSPasteboardItem.alloc().init()
        for t, d in data.items():
            item.setData_forType_(d, t)
        out.append(item)
    board.writeObjects_(out)

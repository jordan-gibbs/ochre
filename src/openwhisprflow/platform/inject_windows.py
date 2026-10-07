"""Windows text injection: ``SendInput`` with ``KEYEVENTF_UNICODE``, no clipboard (SPEC §6.2).

Lessons carried over from earlier native injection code:

* Text goes out in batches of UTF-16 units that never split a surrogate pair. Between batches
  we check that the same window still has focus and that the user has not started typing;
  either stops the rest instead of spraying it into the wrong place (InjectError).
* Win11 Notepad and Windows Terminal (XAML input) translate injected characters lazily and
  corrupt a burst into repeats of the last character (verified: 5 ms per character
  fails, 8 ms works), so they get one character per batch with a short gap.
* Held modifiers: we first wait briefly for the user to let go (they are usually still
  releasing the Voice key chord). Ones still down are released with a menu-mask key first (so
  a lone Alt/Win release opens no menu), and given back afterwards **only** if our hook saw
  the user is still physically holding them: re-pressing a key the user has let go would strand
  it down across the desktop, which is far worse than having to press it again.
  Ctrl+V borrows a Ctrl the user is holding instead of releasing it.
* An elevated foreground window (UIPI) cannot receive our input: we refuse up front with a
  clear message instead of typing into the void.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from openwhisprflow.platform import _win32 as w
from openwhisprflow.platform import clipboard_windows as clip
from openwhisprflow.platform.base import FocusInfo, InjectError
from openwhisprflow.platform.focus_windows import get_focus
from openwhisprflow.platform.hotkeys_windows import PHYSICAL
from openwhisprflow.platform.textunits import next_batch, plan, utf16_units

log = logging.getLogger("openwhisprflow.inject.windows")

VK_LSHIFT, VK_RSHIFT, VK_LCONTROL, VK_RCONTROL = 0xA0, 0xA1, 0xA2, 0xA3
VK_LMENU, VK_RMENU, VK_LWIN, VK_RWIN = 0xA4, 0xA5, 0x5B, 0x5C
VK_RETURN, VK_TAB, VK_V, VK_MASK = 0x0D, 0x09, 0x56, 0xE8  # 0xE8: unassigned, the "menu mask"
MODIFIER_VKS = (VK_LSHIFT, VK_RSHIFT, VK_LCONTROL, VK_RCONTROL, VK_LMENU, VK_RMENU, VK_LWIN, VK_RWIN)
_MASK_NEEDED = {VK_LMENU, VK_RMENU, VK_LWIN, VK_RWIN}

SLOW_APPS = frozenset({"notepad", "windowsterminal"})

ELEVATED = ("{app} is running as administrator, and Windows does not let a normal app type into "
            "it. The text is in History. To dictate into admin windows, run Open Whisperflow as "
            "administrator too.")
BLOCKED = ("Windows blocked the keystrokes (an admin window, the lock screen or a UAC prompt has "
           "focus). The text is in History.")
MOVED = "Focus moved to another window while typing, so the rest was not typed. The text is in History."
TYPED = "You started typing while dictated text was going out, so the rest was not typed. The text is in History."


class WindowsInjector:
    """:class:`~openwhisprflow.platform.base.Injector` for Windows."""

    def __init__(self, *, batch_units: int = 32, batch_gap_s: float = 0.004,
                 slow_apps: frozenset[str] = SLOW_APPS, slow_gap_s: float = 0.008,
                 modifier_wait_s: float = 0.6, paste_settle_s: float = 0.3,
                 newline: str = "enter") -> None:
        self.batch_units = batch_units
        self.batch_gap_s = batch_gap_s
        self.slow_apps = slow_apps
        self.slow_gap_s = slow_gap_s
        self.modifier_wait_s = modifier_wait_s
        self.paste_settle_s = paste_settle_s
        self.newline = newline  # "enter" or "shift_enter" (a line break in chat apps)

    # ------------------------------------------------------------------ Injector

    def focus(self) -> FocusInfo:
        return get_focus()

    def type_text(self, text: str) -> None:
        if not text:
            return
        target = self._check_target()
        hwnd = w.user32.GetForegroundWindow()
        slow = target.app_name in self.slow_apps
        max_units = 1 if slow else self.batch_units
        gap = self.slow_gap_s if slow else self.batch_gap_s
        with self._modifiers_released():
            presses = PHYSICAL.presses
            first = True
            for kind, value in plan(text):
                if kind == "key":
                    self._guard(first, hwnd, presses)
                    self._send(self._key_tap(value))
                    first = False
                    time.sleep(gap)
                    continue
                units = utf16_units(value)
                start = 0
                while start < len(units):
                    self._guard(first, hwnd, presses)
                    n = next_batch(units, start, max_units)
                    batch = []
                    for u in units[start:start + n]:
                        batch += [w.unicode_input(u, False), w.unicode_input(u, True)]
                    self._send(batch)
                    start += n
                    first = False
                    time.sleep(gap)

    def paste_text(self, text: str) -> None:
        if not text:
            return
        self._check_target()
        try:
            snapshot: clip.Snapshot | None = clip.save()
        except Exception:  # never lose the paste because the save failed
            log.warning("could not save the clipboard; it will not be restored", exc_info=True)
            snapshot = None
        try:
            seq = clip.set_text(text)
        except Exception as e:
            raise InjectError(f"Could not use the clipboard ({e}). The text is in History.") from e
        with self._modifiers_released(keep={VK_LCONTROL, VK_RCONTROL}):
            # Sampled after the wait, right before injecting: a Ctrl the user still holds is
            # borrowed (neither pressed nor released), so their key state is handed back intact.
            user_ctrl = w.key_down(VK_LCONTROL) or w.key_down(VK_RCONTROL)
            seq_in = [] if user_ctrl else [w.key_input(VK_LCONTROL, False)]
            seq_in += [w.key_input(VK_V, False), w.key_input(VK_V, True)]
            seq_in += [] if user_ctrl else [w.key_input(VK_LCONTROL, True)]
            self._send(seq_in)
        time.sleep(self.paste_settle_s)  # apps read the clipboard asynchronously after Ctrl+V
        if snapshot is not None and clip.sequence() == seq:  # nobody copied anything meanwhile
            try:
                clip.restore(snapshot)
            except Exception:
                log.warning("could not restore the clipboard", exc_info=True)

    def press_enter(self) -> None:
        self._check_target()
        with self._modifiers_released():
            self._send([w.key_input(VK_RETURN, False), w.key_input(VK_RETURN, True)])

    # ------------------------------------------------------------------ helpers

    def _check_target(self) -> FocusInfo:
        target = get_focus()
        if target.elevated:
            raise InjectError(ELEVATED.format(app=target.app_name or "The focused app"))
        return target

    def _guard(self, first: bool, hwnd: int, presses: int) -> None:
        if first:
            return
        if w.user32.GetForegroundWindow() != hwnd:
            raise InjectError(MOVED)
        if PHYSICAL.active and PHYSICAL.presses != presses:
            raise InjectError(TYPED)

    @staticmethod
    def _send(inputs: list[w.INPUT]) -> None:
        if w.send_inputs(inputs) != len(inputs):
            raise InjectError(BLOCKED)

    def _key_tap(self, key: str) -> list[w.INPUT]:
        if key == "tab":
            return [w.key_input(VK_TAB, False), w.key_input(VK_TAB, True)]
        tap = [w.key_input(VK_RETURN, False), w.key_input(VK_RETURN, True)]
        if self.newline == "shift_enter":
            return [w.key_input(VK_LSHIFT, False), *tap, w.key_input(VK_LSHIFT, True)]
        return tap

    @contextmanager
    def _modifiers_released(self, keep: frozenset[int] | set[int] = frozenset()) -> Iterator[None]:
        deadline = time.monotonic() + self.modifier_wait_s
        while time.monotonic() < deadline and any(w.key_down(vk) for vk in MODIFIER_VKS if vk not in keep):
            time.sleep(0.015)
        held = [vk for vk in MODIFIER_VKS if vk not in keep and w.key_down(vk)]
        if held:
            ups = [w.key_input(VK_MASK, False), w.key_input(VK_MASK, True)] if _MASK_NEEDED & set(held) else []
            ups += [w.key_input(vk, True) for vk in held]
            w.send_inputs(ups)
        try:
            yield
        finally:
            # Give back only what the user is still physically holding (our hook knows); without
            # a hook we cannot tell, and a stranded modifier is worse than a re-press.
            back = [vk for vk in held if PHYSICAL.active and PHYSICAL.is_down(vk)]
            if back:
                downs = [w.key_input(vk, False) for vk in back]
                if _MASK_NEEDED & set(back):  # so the user's own release opens no menu
                    downs += [w.key_input(VK_MASK, False), w.key_input(VK_MASK, True)]
                w.send_inputs(downs)

"""Windows Voice key listener: a ``WH_KEYBOARD_LL`` hook on its own thread (SPEC §6.1).

Hardening ("Voice never jams the keyboard"):

* The hook lives on a dedicated thread that does nothing but pump messages. Windows holds
  every keystroke on the machine until a low-level hook answers and silently lets the key
  through when it answers late, so the callback only translates the event and asks the
  :class:`~openwhisprflow.platform.driver.GestureDriver` (a lock held for microseconds);
  gestures reach the app through the driver's dispatcher thread.
* A release is never hidden while Windows thinks the key is down (:func:`safe_swallow`): if a
  press leaked through a late hook, hiding its release would leave the key stuck everywhere.
* Gesture timing uses the event's own timestamp, not when the hook ran.
* A watchdog reinstalls the hook when Windows drops it (it does so without telling anyone
  after repeated timeouts): input the system saw that the hook did not means the hook is gone.
* Stopping releases a Voice key the gesture had a hand in, so nothing stays held.
* AltGr layouts send a synthesized Left Ctrl right before Right Alt (same timestamp, scan code
  0x21D). It passes through untouched and never counts as a held modifier or as "another key".
* Injected input (ours and everyone else's) always passes through untouched.

The hook also keeps a record of which keys are *physically* down (non-injected events), which
the injector uses to give back modifiers the user is still holding after typing.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes

from openwhisprflow.config import HotkeyConfig
from openwhisprflow.platform import _win32 as w
from openwhisprflow.platform.base import GestureFn
from openwhisprflow.platform.driver import GestureDriver
from openwhisprflow.platform.gesture import event_time, safe_swallow
from openwhisprflow.platform.keys import WIN_VK, KeySpec, win_name

log = logging.getLogger("openwhisprflow.hotkeys.windows")

VK_LCONTROL, VK_RMENU = 0xA2, 0xA5
_WATCHDOG_MS = 1000
_REINSTALL_GAP_MS = 10000


class PhysicalKeys:
    """Which keys the user is physically holding, as seen by our hook (thread-safe enough:
    single writer, readers take a snapshot)."""

    def __init__(self) -> None:
        self.down: set[int] = set()
        self.presses = 0          # physical key presses seen; the injector watches for typing
        self.active = False       # a hook is running and feeding this

    def update(self, vk: int, down: bool) -> None:
        if down:
            if vk not in self.down:
                self.presses += 1
            self.down.add(vk)
        else:
            self.down.discard(vk)

    def is_down(self, vk: int) -> bool:
        return vk in self.down


PHYSICAL = PhysicalKeys()


def _validate(spec: KeySpec) -> None:
    for name in (spec.key,):
        if name not in WIN_VK:
            raise ValueError(f"{name!r} cannot be used as the Voice key on Windows")


class WindowsHotkeyListener:
    """:class:`~openwhisprflow.platform.base.HotkeyListener` on a low-level keyboard hook."""

    def __init__(self, cfg: HotkeyConfig, *, accept_test_input: bool = False) -> None:
        self._driver = GestureDriver(cfg, self._replay, clock_ms=w.kernel32.GetTickCount64,
                                     validate=_validate)
        self._accept_test = accept_test_input
        self._proc = w.HOOKPROC(self._callback)  # keep a reference: Windows calls it
        self._hook: int | None = None
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._ready = threading.Event()
        self._error = ""
        self._last_hook_tick = 0
        self._lctrl_time = -1
        self._lctrl_fake = False
        self.reinstalls = 0

    # ------------------------------------------------------------------ HotkeyListener

    def start(self, on_gesture: GestureFn) -> None:
        if self._thread is not None:
            return
        self._driver.start(on_gesture)
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="owf-keyhook", daemon=True)
        self._thread.start()
        if not self._ready.wait(3.0) or self._error:
            self.stop()
            raise OSError(self._error or "keyboard hook thread did not start")

    def set_key(self, key: str) -> None:
        self._driver.set_key(key)

    def set_recording(self, recording: bool) -> None:
        self._driver.set_recording(recording)

    def stop(self) -> None:
        if self._tid:
            w.user32.PostThreadMessageW(self._tid, w.WM_QUIT, 0, 0)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(2.0)
        self._thread = None
        self._tid = 0
        PHYSICAL.active = False
        # Never leave the Voice key held for other apps once the hook is gone. If a finger
        # really is on it, its own release passes straight through and does no harm.
        vk = WIN_VK.get(self._driver.key_name)
        if vk is not None and self._driver.involved() and w.key_down(vk):
            self._replay(self._driver.key_name, False)
        self._driver.stop()
        self._driver.reset()

    # ------------------------------------------------------------------ hook thread

    def _install(self) -> bool:
        if self._hook:
            w.user32.UnhookWindowsHookEx(self._hook)
        self._hook = w.user32.SetWindowsHookExW(w.WH_KEYBOARD_LL, self._proc,
                                                w.kernel32.GetModuleHandleW(None), 0)
        return bool(self._hook)

    def _run(self) -> None:
        self._tid = w.kernel32.GetCurrentThreadId()
        msg = wintypes.MSG()
        w.user32.PeekMessageW(ctypes.byref(msg), None, w.WM_USER, w.WM_USER, 0)  # create the queue
        w.kernel32.SetThreadPriority(w.kernel32.GetCurrentThread(), 15)  # TIME_CRITICAL
        self._last_hook_tick = w.kernel32.GetTickCount()
        if not self._install():
            self._error = f"SetWindowsHookExW failed (error {ctypes.get_last_error()})"
            self._ready.set()
            return
        PHYSICAL.active = True
        self._ready.set()
        timer = w.user32.SetTimer(None, 0, _WATCHDOG_MS, None)
        last_reinstall = 0
        try:
            while w.user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message != w.WM_TIMER:
                    continue
                now = w.kernel32.GetTickCount64()
                info = w.LASTINPUTINFO(ctypes.sizeof(w.LASTINPUTINFO), 0)
                if not w.user32.GetLastInputInfo(ctypes.byref(info)):
                    continue
                silent = ctypes.c_int32((info.dwTime - self._last_hook_tick) & 0xFFFFFFFF).value
                if silent > 2000 and now - last_reinstall >= _REINSTALL_GAP_MS:
                    last_reinstall = now
                    if self._install():
                        self._last_hook_tick = w.kernel32.GetTickCount()
                        self.reinstalls += 1
                        log.warning("keyboard hook was dropped by Windows; reinstalled")
        finally:
            if timer:
                w.user32.KillTimer(None, timer)
            if self._hook:
                w.user32.UnhookWindowsHookEx(self._hook)
                self._hook = None
            PHYSICAL.active = False

    def _callback(self, code: int, wparam: int, lparam: int) -> int:
        if code == w.HC_ACTION:
            try:
                if self._handle(wparam, lparam):
                    return 1
            except Exception:  # never break the user's keyboard
                pass
        return w.user32.CallNextHookEx(None, code, wparam, lparam)

    def _handle(self, wparam: int, lparam: int) -> bool:
        kb = w.KBDLLHOOKSTRUCT.from_address(lparam)
        self._last_hook_tick = w.kernel32.GetTickCount()
        injected = bool(kb.flags & w.LLKHF_INJECTED)
        if injected and not (self._accept_test and kb.dwExtraInfo == w.TAG_TEST):
            return False
        down = wparam in (w.WM_KEYDOWN, w.WM_SYSKEYDOWN)
        vk = int(kb.vkCode)
        if not injected:
            PHYSICAL.update(vk, down)
        if vk == VK_LCONTROL:
            if down and not self._lctrl_fake:
                self._lctrl_time = int(kb.time)
                self._lctrl_fake = bool(kb.scanCode & 0x200)  # AltGr's synthesized Ctrl (0x21D)
            if self._lctrl_fake:
                if not down:
                    self._lctrl_fake = False
                return False  # AltGr's Ctrl: not a modifier the user pressed, not another key
        elif vk == VK_RMENU and down and int(kb.time) == self._lctrl_time:
            # AltGr whose synthesized Ctrl had a plain scan code: same timestamp gives it away.
            self._lctrl_fake = True
            self._driver.held.discard("left_ctrl")
        now = event_time(w.kernel32.GetTickCount64(), w.kernel32.GetTickCount(), int(kb.time))
        swallow = self._driver.feed(win_name(vk), down, now)
        return safe_swallow(swallow, down, w.key_down(vk))

    # ------------------------------------------------------------------ replays

    def _replay(self, name: str, down: bool) -> None:
        vk = WIN_VK.get(name)
        if vk is not None:
            w.send_inputs([w.key_input(vk, not down, w.TAG_REPLAY)])

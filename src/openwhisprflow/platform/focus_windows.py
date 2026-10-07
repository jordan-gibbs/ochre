"""The focused app on Windows: executable name, window title, window id, elevation."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

from openwhisprflow.platform import _win32 as w
from openwhisprflow.platform.base import FocusInfo

_SELF_ELEVATED: bool | None = None


def _pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    w.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _title(hwnd: int) -> str:
    n = w.user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    w.user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def focused_control(hwnd: int) -> int:
    """The control with keyboard focus inside the foreground window's thread (0 if unknown)."""
    tid = w.user32.GetWindowThreadProcessId(hwnd, None)
    info = w.GUITHREADINFO(cbSize=ctypes.sizeof(w.GUITHREADINFO))
    if tid and w.user32.GetGUIThreadInfo(tid, ctypes.byref(info)):
        return info.hwndFocus or 0
    return 0


def _uwp_child_pid(hwnd: int, host_pid: int) -> int:
    """UWP apps live inside ApplicationFrameHost; the real app owns a child CoreWindow."""
    found = [0]

    def visit(child: int, _: int) -> bool:
        pid = _pid(child)
        if pid and pid != host_pid:
            found[0] = pid
            return False
        return True

    w.user32.EnumChildWindows(hwnd, w.WNDENUMPROC(visit), 0)
    return found[0]


def app_name_from_path(path: str) -> str:
    """"C:\\...\\Slack.exe" -> "slack" (matches the keys in RefineConfig.app_styles)."""
    name = os.path.basename(path).lower()
    return name[:-4] if name.endswith(".exe") else name


def self_elevated() -> bool:
    global _SELF_ELEVATED
    if _SELF_ELEVATED is None:
        _SELF_ELEVATED = w.self_elevated()
    return _SELF_ELEVATED


def get_focus() -> FocusInfo:
    hwnd = w.user32.GetForegroundWindow() or 0
    if not hwnd:
        return FocusInfo()
    pid = _pid(hwnd)
    path = w.process_path(pid)
    if app_name_from_path(path) == "applicationframehost":
        child = _uwp_child_pid(hwnd, pid)
        if child:
            pid, path = child, w.process_path(child) or path
    elevated = bool(pid) and not self_elevated() and w.process_elevated(pid)
    return FocusInfo(app_name=app_name_from_path(path), window_title=_title(hwnd),
                     window_id=f"{hwnd:x}:{focused_control(hwnd):x}", elevated=elevated)

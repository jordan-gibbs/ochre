"""Minimal ctypes bindings for the Windows platform modules (Windows only).

Private ``WinDLL`` handles with explicit argtypes, so these prototypes never clash with other
libraries that configure ``ctypes.windll.user32`` differently, and 64-bit handles are never
truncated to ``int``.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

ULONG_PTR = ctypes.c_size_t
LRESULT = ctypes.c_ssize_t

# Tags in dwExtraInfo so our own hook (and anyone else's) can tell our input apart.
TAG_REPLAY = 0x4F574652   # "OWFR": Voice key replays
TAG_TEXT = 0x4F574654     # "OWFT": dictated text and chords
TAG_TEST = 0x4F574658     # "OWFX": hardware tests; accepted by a listener built with accept_test_input

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, KEYEVENTF_SCANCODE = 0x1, 0x2, 0x4, 0x8
WH_KEYBOARD_LL, HC_ACTION = 13, 0
WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0100, 0x0101, 0x0104, 0x0105
WM_TIMER, WM_QUIT, WM_USER = 0x0113, 0x0012, 0x0400
LLKHF_EXTENDED, LLKHF_INJECTED = 0x01, 0x10
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TokenElevation = 20
ERROR_ACCESS_DENIED = 5
GMEM_MOVEABLE = 0x0002


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND), ("hwndCapture", wintypes.HWND),
                ("hwndMenuOwner", wintypes.HWND), ("hwndMoveSize", wintypes.HWND),
                ("hwndCaret", wintypes.HWND), ("rcCaret", wintypes.RECT)]


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _proto(fn, argtypes, restype) -> None:  # noqa: ANN001
    fn.argtypes = argtypes
    fn.restype = restype


_proto(user32.SetWindowsHookExW, [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD], ctypes.c_void_p)
_proto(user32.CallNextHookEx, [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM], LRESULT)
_proto(user32.UnhookWindowsHookEx, [ctypes.c_void_p], wintypes.BOOL)
_proto(user32.GetMessageW, [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT],
       wintypes.BOOL)
_proto(user32.PeekMessageW, [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT,
                             wintypes.UINT], wintypes.BOOL)
_proto(user32.PostThreadMessageW, [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM],
       wintypes.BOOL)
_proto(user32.SetTimer, [wintypes.HWND, ULONG_PTR, wintypes.UINT, ctypes.c_void_p], ULONG_PTR)
_proto(user32.KillTimer, [wintypes.HWND, ULONG_PTR], wintypes.BOOL)
_proto(user32.GetAsyncKeyState, [ctypes.c_int], ctypes.c_short)
_proto(user32.SendInput, [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int], wintypes.UINT)
_proto(user32.GetForegroundWindow, [], wintypes.HWND)
_proto(user32.GetWindowThreadProcessId, [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD)
_proto(user32.GetWindowTextLengthW, [wintypes.HWND], ctypes.c_int)
_proto(user32.GetWindowTextW, [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int)
_proto(user32.GetClassNameW, [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int)
_proto(user32.GetGUIThreadInfo, [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)], wintypes.BOOL)
_proto(user32.GetLastInputInfo, [ctypes.POINTER(LASTINPUTINFO)], wintypes.BOOL)
_proto(user32.EnumChildWindows, [wintypes.HWND, WNDENUMPROC, wintypes.LPARAM], wintypes.BOOL)
_proto(user32.MapVirtualKeyW, [wintypes.UINT, wintypes.UINT], wintypes.UINT)
_proto(user32.OpenClipboard, [wintypes.HWND], wintypes.BOOL)
_proto(user32.CloseClipboard, [], wintypes.BOOL)
_proto(user32.EmptyClipboard, [], wintypes.BOOL)
_proto(user32.GetClipboardData, [wintypes.UINT], wintypes.HANDLE)
_proto(user32.SetClipboardData, [wintypes.UINT, wintypes.HANDLE], wintypes.HANDLE)
_proto(user32.EnumClipboardFormats, [wintypes.UINT], wintypes.UINT)
_proto(user32.RegisterClipboardFormatW, [wintypes.LPCWSTR], wintypes.UINT)
_proto(user32.GetClipboardSequenceNumber, [], wintypes.DWORD)
_proto(user32.IsWindow, [wintypes.HWND], wintypes.BOOL)

_proto(kernel32.GetModuleHandleW, [wintypes.LPCWSTR], wintypes.HMODULE)
_proto(kernel32.GetCurrentThreadId, [], wintypes.DWORD)
_proto(kernel32.GetCurrentProcessId, [], wintypes.DWORD)
_proto(kernel32.GetCurrentProcess, [], wintypes.HANDLE)
_proto(kernel32.GetCurrentThread, [], wintypes.HANDLE)
_proto(kernel32.SetThreadPriority, [wintypes.HANDLE, ctypes.c_int], wintypes.BOOL)
_proto(kernel32.GetTickCount, [], wintypes.DWORD)
_proto(kernel32.GetTickCount64, [], ctypes.c_uint64)
_proto(kernel32.OpenProcess, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE)
_proto(kernel32.CloseHandle, [wintypes.HANDLE], wintypes.BOOL)
_proto(kernel32.QueryFullProcessImageNameW, [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                             ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL)
_proto(kernel32.GlobalAlloc, [wintypes.UINT, ctypes.c_size_t], wintypes.HGLOBAL)
_proto(kernel32.GlobalLock, [wintypes.HGLOBAL], ctypes.c_void_p)
_proto(kernel32.GlobalUnlock, [wintypes.HGLOBAL], wintypes.BOOL)
_proto(kernel32.GlobalSize, [wintypes.HGLOBAL], ctypes.c_size_t)
_proto(kernel32.GlobalFree, [wintypes.HGLOBAL], wintypes.HGLOBAL)

_proto(advapi32.OpenProcessToken, [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)],
       wintypes.BOOL)
_proto(advapi32.GetTokenInformation, [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                      ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL)


def key_down(vk: int) -> bool:
    """The OS's (logical) view of a key; swallowed keys never show up here."""
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def key_input(vk: int, up: bool, tag: int = TAG_TEXT, extended: bool | None = None) -> INPUT:
    from openwhisprflow.platform.keys import WIN_EXTENDED

    inp = INPUT(type=INPUT_KEYBOARD)
    inp.ki.wVk = vk
    inp.ki.wScan = user32.MapVirtualKeyW(vk, 0) & 0xFF
    ext = vk in WIN_EXTENDED if extended is None else extended
    inp.ki.dwFlags = (KEYEVENTF_EXTENDEDKEY if ext else 0) | (KEYEVENTF_KEYUP if up else 0)
    inp.ki.dwExtraInfo = tag
    return inp


def unicode_input(unit: int, up: bool, tag: int = TAG_TEXT) -> INPUT:
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.ki.wVk = 0
    inp.ki.wScan = unit
    inp.ki.dwFlags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
    inp.ki.dwExtraInfo = tag
    return inp


def send_inputs(inputs: list[INPUT]) -> int:
    """SendInput one batch atomically; returns how many events Windows accepted."""
    if not inputs:
        return 0
    arr = (INPUT * len(inputs))(*inputs)
    return user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def process_path(pid: int) -> str:
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(32768)
        return buf.value if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)) else ""
    finally:
        kernel32.CloseHandle(h)


def _token_elevated(process: int) -> bool | None:
    """True/False from the token, or None when the token cannot be opened."""
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        value = wintypes.DWORD()
        size = wintypes.DWORD()
        if not advapi32.GetTokenInformation(token, TokenElevation, ctypes.byref(value),
                                            ctypes.sizeof(value), ctypes.byref(size)):
            return None
        return bool(value.value)
    finally:
        kernel32.CloseHandle(token)


def self_elevated() -> bool:
    return bool(_token_elevated(kernel32.GetCurrentProcess()))


def process_elevated(pid: int) -> bool:
    """Whether ``pid`` runs elevated. A non-elevated caller usually cannot open an elevated
    process's token at all, which is itself the answer (access denied = elevated)."""
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        result = _token_elevated(h)
        if result is None:
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        return result
    finally:
        kernel32.CloseHandle(h)

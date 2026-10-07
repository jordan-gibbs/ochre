"""Windows clipboard snapshot / restore for the paste fallback (raw Win32, memory formats).

Every format on the clipboard that is backed by global memory is saved (text, RTF, HTML, DIB
images, file drops, app-private formats...). GDI-handle formats (bitmaps, metafiles,
palettes) and owner-display formats cannot be copied as bytes and are skipped; Windows
re-synthesizes CF_BITMAP from the DIB we do keep. Text we put there for a paste is marked to
stay out of clipboard history and cloud sync.
"""

from __future__ import annotations

import ctypes
import time

from openwhisprflow.platform import _win32 as w

CF_TEXT, CF_BITMAP, CF_METAFILEPICT, CF_OEMTEXT, CF_PALETTE = 1, 2, 3, 7, 9
CF_UNICODETEXT, CF_ENHMETAFILE = 13, 14
_HANDLE_FORMATS = {CF_BITMAP, CF_METAFILEPICT, CF_PALETTE, CF_ENHMETAFILE, 0x80, 0x82, 0x83, 0x8E}
# Synthesized by Windows from CF_UNICODETEXT; restoring them too would be redundant.
_SYNTHESIZED = {CF_TEXT, CF_OEMTEXT}
_MAX_TOTAL = 64 * 1024 * 1024
_NO_HISTORY = ("ExcludeClipboardContentFromMonitorProcessing", "CanIncludeInClipboardHistory",
               "CanUploadToCloudClipboard")

Snapshot = list[tuple[int, bytes]]


class ClipboardBusy(RuntimeError):
    pass


class _Open:
    """``with _Open():`` opens the clipboard, retrying while another app holds it."""

    def __init__(self, timeout_s: float = 1.0) -> None:
        self.timeout_s = timeout_s

    def __enter__(self) -> "_Open":
        deadline = time.monotonic() + self.timeout_s
        while not w.user32.OpenClipboard(None):
            if time.monotonic() > deadline:
                raise ClipboardBusy("the clipboard is held open by another app")
            time.sleep(0.02)
        return self

    def __exit__(self, *_: object) -> None:
        w.user32.CloseClipboard()


def _read(handle: int, budget: int) -> bytes | None:
    size = w.kernel32.GlobalSize(handle)
    if not size or size > budget:
        return None
    p = w.kernel32.GlobalLock(handle)
    if not p:
        return None
    try:
        return ctypes.string_at(p, size)
    finally:
        w.kernel32.GlobalUnlock(handle)


def _put(fmt: int, data: bytes) -> bool:
    h = w.kernel32.GlobalAlloc(w.GMEM_MOVEABLE, max(len(data), 1))
    if not h:
        return False
    p = w.kernel32.GlobalLock(h)
    if not p:
        w.kernel32.GlobalFree(h)
        return False
    ctypes.memmove(p, data, len(data))
    w.kernel32.GlobalUnlock(h)
    if not w.user32.SetClipboardData(fmt, h):
        w.kernel32.GlobalFree(h)  # ownership only passes on success
        return False
    return True


def _mark_private() -> None:
    zero = (0).to_bytes(4, "little")
    for name in _NO_HISTORY:
        fmt = w.user32.RegisterClipboardFormatW(name)
        if fmt:
            _put(fmt, zero)


def save() -> Snapshot:
    out: Snapshot = []
    budget = _MAX_TOTAL
    with _Open():
        fmt = w.user32.EnumClipboardFormats(0)
        while fmt:
            if fmt not in _HANDLE_FORMATS and fmt not in _SYNTHESIZED:
                h = w.user32.GetClipboardData(fmt)
                data = _read(h, budget) if h else None
                if data is not None:
                    out.append((fmt, data))
                    budget -= len(data)
            fmt = w.user32.EnumClipboardFormats(fmt)
    return out


def restore(snapshot: Snapshot) -> None:
    with _Open():
        w.user32.EmptyClipboard()
        for fmt, data in snapshot:
            _put(fmt, data)
        if snapshot:
            _mark_private()  # the restore itself should not show up as a new history entry


def set_text(text: str) -> int:
    """Put ``text`` on the clipboard (kept out of history); returns the new sequence number."""
    with _Open():
        w.user32.EmptyClipboard()
        if not _put(CF_UNICODETEXT, (text + "\0").encode("utf-16-le", "surrogatepass")):
            raise OSError("SetClipboardData(CF_UNICODETEXT) failed")
        _mark_private()
    return w.user32.GetClipboardSequenceNumber()


def get_text() -> str | None:
    with _Open():
        h = w.user32.GetClipboardData(CF_UNICODETEXT)
        data = _read(h, _MAX_TOTAL) if h else None
    if data is None:
        return None
    return data.decode("utf-16-le", "surrogatepass").split("\0", 1)[0]


def sequence() -> int:
    return w.user32.GetClipboardSequenceNumber()

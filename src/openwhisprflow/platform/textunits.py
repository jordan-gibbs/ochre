"""Pure text helpers for the injectors: UTF-16 batching and newline/tab planning.

Windows (``KEYEVENTF_UNICODE``) and macOS (``CGEventKeyboardSetUnicodeString``) both take
UTF-16 code units, and a batch boundary must never split a surrogate pair or the emoji arrives
as two replacement characters.
"""

from __future__ import annotations

from typing import Literal

Segment = tuple[Literal["text", "key"], str]


def utf16_units(text: str) -> list[int]:
    data = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]


def is_high_surrogate(unit: int) -> bool:
    return 0xD800 <= unit <= 0xDBFF


def next_batch(units: list[int], start: int, max_units: int) -> int:
    """Length of the next batch from ``start``: at most ``max_units`` UTF-16 units, extended by
    one rather than splitting a surrogate pair."""
    if start >= len(units):
        return 0
    n = min(len(units) - start, max_units)
    if is_high_surrogate(units[start + n - 1]) and start + n < len(units):
        n += 1
    return n


def chunks(text: str, max_units: int) -> list[str]:
    """``text`` cut into strings of at most ``max_units`` UTF-16 units (pairs kept whole;
    a pair longer than the limit allows, i.e. ``max_units == 1``, still goes out whole)."""
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for ch in text:
        n = 2 if ord(ch) > 0xFFFF else 1
        if cur and size + n > max_units:
            out.append("".join(cur))
            cur, size = [], 0
        cur.append(ch)
        size += n
    if cur:
        out.append("".join(cur))
    return out


def plan(text: str) -> list[Segment]:
    """Split text into typed runs and key presses: newline -> Enter, tab -> Tab.

    CRLF and lone CR count as one newline, so Windows line endings never press Enter twice.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[Segment] = []
    run: list[str] = []
    for ch in text:
        if ch in "\n\t":
            if run:
                out.append(("text", "".join(run)))
                run = []
            out.append(("key", "enter" if ch == "\n" else "tab"))
        else:
            run.append(ch)
    if run:
        out.append(("text", "".join(run)))
    return out

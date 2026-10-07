"""The leading-space join rule (SPEC §6.2), pure apart from a small last-insertion memory.

Dictating twice in a row into the same field must not glue the sentences together ("Hello
there.How are you"). We do not read the caret (it is unreliable across apps): one space
is prepended when the previous insertion went to the same app and window, ended with a
non-space, and was less than ``join_window_s`` ago. With ``trailing_space`` on (the default)
every insertion already ends with a space, so the leading space only matters when it is off.

No space is added before closing punctuation (",", ".", ")"...) or after an opening bracket or
quote, nor between CJK characters, which are written without spaces.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass
from typing import Callable

from openwhisprflow.platform.base import FocusInfo

NO_SPACE_BEFORE = set(",.;:!?)]}%…»”’、。，．！？；：）」』】")
NO_SPACE_AFTER = set("([{«“‘¿¡/-—–「『【")


def is_cjk(ch: str) -> bool:
    if not ch:
        return False
    name = unicodedata.name(ch, "")
    return name.startswith(("CJK ", "HIRAGANA", "KATAKANA", "HANGUL", "IDEOGRAPHIC", "FULLWIDTH"))


def needs_leading_space(prev_end: str, text: str) -> bool:
    """Whether ``text`` typed right after a string ending in ``prev_end`` needs a space."""
    if not prev_end or not text:
        return False
    first = text[0]
    if prev_end.isspace() or first.isspace():
        return False
    if first in NO_SPACE_BEFORE or prev_end in NO_SPACE_AFTER:
        return False
    if is_cjk(prev_end) or is_cjk(first):
        return False
    return True


def with_trailing_space(text: str, enabled: bool) -> str:
    if not enabled or not text or text[-1].isspace() or is_cjk(text[-1]):
        return text
    return text + " "


@dataclass
class _Last:
    target: tuple[str, str]
    end: str
    at: float


class Joiner:
    """Remembers the last insertion so the next one can join it."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.last: _Last | None = None

    def join_text(self, text: str, focus: FocusInfo, *, trailing_space: bool = True,
                  join_window_s: float = 20.0) -> str:
        """The exact payload to type for ``text`` into ``focus``; remembers it as the last
        insertion (call :meth:`forget` if typing it then fails)."""
        if not text:
            return text
        now = self.clock()
        target = (focus.app_name, focus.window_id)
        payload = text
        last = self.last
        if (last is not None and any(target) and last.target == target
                and now - last.at < join_window_s and needs_leading_space(last.end, text)):
            payload = " " + payload
        payload = with_trailing_space(payload, trailing_space)
        self.last = _Last(target, payload[-1], now)
        return payload

    def forget(self) -> None:
        self.last = None


_default = Joiner()


def join_text(text: str, focus: FocusInfo, *, trailing_space: bool = True, join_window_s: float = 20.0) -> str:
    """Module-level :meth:`Joiner.join_text` on a shared default instance."""
    return _default.join_text(text, focus, trailing_space=trailing_space, join_window_s=join_window_s)


def forget() -> None:
    _default.forget()

"""Personal dictionary applied after every STT engine.

Two explicit rule kinds, no fuzzy or acoustic guessing (a wrong "smart" fix is worse than a
visible miss):

* **replacements** ``{"said": "written"}`` - whole-phrase, case-insensitive, word-bounded,
  whitespace-tolerant ("open whisper flow" matches "Open  Whisper flow").
* **preferred spellings** ``words`` - a case-insensitive match is rewritten to the listed
  casing ("github" -> "GitHub").

Matching rules: longest match wins at a position, earlier rules break ties, matches never
overlap. Replacement text is protected from the later spelling pass, so a replacement's own
spelling wins. An all-lowercase replacement that lands on a capitalised word (sentence start)
is capitalised to keep the sentence well-formed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

MAX_RULES = 500
MAX_LEN = 160
_BOUND_L = r"(?<![\w'’])"
_BOUND_R = r"(?![\w'’])"
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def words(text: str) -> list[str]:
    """Normalized word list (NFKC, casefold, straight apostrophes): for comparisons in tests/evals."""
    return [unicodedata.normalize("NFKC", m.group()).casefold().replace("’", "'")
            for m in WORD.finditer(text)]


def _pattern(heard: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(p) for p in heard.split())
    return re.compile(_BOUND_L + body + _BOUND_R, re.IGNORECASE)


def _match_case(matched: str, replacement: str) -> str:
    if replacement and replacement == replacement.lower() and matched[:1].isupper() and not matched.isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_rules(text: str, rules: Iterable[tuple[str, str]], *, keep_case: bool = True,
                protected: Iterable[tuple[int, int]] = ()) -> tuple[str, list[tuple[int, int]]]:
    """Apply ``(heard, written)`` rules. Returns the new text and the spans of inserted text.

    Matches overlapping a ``protected`` span (in the input) are skipped.
    """
    compiled = []
    for heard, written in list(rules)[:MAX_RULES]:
        heard, written = (heard or "").strip(), (written or "").strip()
        if heard and written and len(heard) <= MAX_LEN and len(written) <= MAX_LEN:
            compiled.append((_pattern(heard), written))
    guard = sorted(protected)
    candidates = []
    for order, (pattern, written) in enumerate(compiled):
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in guard):
                continue
            candidates.append((m.start(), -(m.end() - m.start()), order, m.end(), written))
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = length = 0
    for start, _, _, end, written in sorted(candidates):
        if start < cursor:
            continue
        piece = _match_case(text[start:end], written) if keep_case else written
        before = text[cursor:start]
        out += [before, piece]
        length += len(before)
        spans.append((length, length + len(piece)))
        length += len(piece)
        cursor = end
    out.append(text[cursor:])
    return "".join(out), spans


def apply(text: str, dictionary: Any | None = None) -> str:
    """Apply a ``config.DictionaryConfig`` (or any object/dict with ``words`` and
    ``replacements``) to ``text``. ``None`` or an empty dictionary returns ``text`` unchanged."""
    if not text or dictionary is None:
        return text
    get = dictionary.get if isinstance(dictionary, dict) else lambda k, d=None: getattr(dictionary, k, d)
    replacements = get("replacements", None) or {}
    vocabulary = [w for w in (get("words", None) or []) if isinstance(w, str)][:MAX_RULES]
    text, spans = apply_rules(text, replacements.items())
    text, _ = apply_rules(text, [(w, w) for w in vocabulary], keep_case=False, protected=spans)
    return text


def vocabulary_prompt(dictionary: Any | None, limit_chars: int = 600) -> str | None:
    """Dictionary words (and replacement targets) as a Whisper ``initial_prompt``: Whisper is
    biased toward spellings it has just "seen". Truncated because Whisper keeps only ~224
    prompt tokens."""
    if dictionary is None:
        return None
    get = dictionary.get if isinstance(dictionary, dict) else lambda k, d=None: getattr(dictionary, k, d)
    terms = [w for w in (get("words", None) or []) if isinstance(w, str) and w.strip()]
    terms += [w for w in (get("replacements", None) or {}).values() if isinstance(w, str) and w.strip()]
    seen: list[str] = []
    for t in terms:
        if t.strip() not in seen:
            seen.append(t.strip())
    prompt = ""
    for t in seen:
        nxt = f"{prompt}, {t}" if prompt else t
        if len(nxt) > limit_chars:
            break
        prompt = nxt
    return f"Vocabulary: {prompt}." if prompt else None

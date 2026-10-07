"""Deterministic spoken-form -> written-form scaffold (SPEC §5.2).

Quill 0.8B is verbatim-only by design: it removes fillers and fixes punctuation, but it is not
trusted to guess that "john at gmail dot com" is an email. This module does those conversions
with rules instead, after the model has run:

- times:    "at three thirty pm" -> "at 3:30 PM", "five oh five am" -> "5:05 AM"
- numbers:  "twenty five" -> "25", "three point five" -> "3.5", "ten percent" -> "10%",
            "fifty dollars" -> "$50"
- emails:   "john dot smith at gmail dot com" -> "john.smith@gmail.com"
- domains:  "example dot com slash pricing" -> "example.com/pricing"
- symbols:  "at sign" -> "@", "ampersand" -> "&", ...

The rules are deliberately conservative, because a wrong rewrite of plain prose is worse than a
missed conversion. A single number word stays a word ("I have two cats", "twenty minutes") unless a
unit makes it unambiguous ("five percent"). Clock times need a cue ("at/by/until...", or am/pm),
so "two twenty dollar bills" is not turned into a time. A run of number words that does not
parse as one number ("twenty twenty six") is left untouched rather than half-converted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------- tokens

_SPLIT = re.compile(r"(\s*)(\S+)")
_EDGE = re.compile(r"^([^\w@#$%&]*)(.*?)([^\w%]*)$", re.DOTALL)


@dataclass
class _Tok:
    pre: str     # whitespace before
    lead: str    # leading punctuation, e.g. "(" or '"'
    core: str
    trail: str   # trailing punctuation, e.g. "," or "?"

    @property
    def low(self) -> str:
        return self.core.lower()

    def text(self) -> str:
        return self.pre + self.lead + self.core + self.trail


def _tokens(text: str) -> list[_Tok]:
    toks = []
    for m in _SPLIT.finditer(text):
        em = _EDGE.match(m.group(2))
        assert em is not None
        lead, core, trail = em.groups()
        if not core:  # pure punctuation token
            lead, core, trail = "", m.group(2), ""
        toks.append(_Tok(m.group(1), lead, core, trail))
    return toks


def _join(toks: list[_Tok], tail: str) -> str:
    return "".join(t.text() for t in toks) + tail


def _clean_span(toks: list[_Tok], i: int, j: int) -> bool:
    """True if tokens i..j-1 can be merged: no punctuation between them."""
    return all(not toks[k].trail for k in range(i, j - 1)) and all(not toks[k].lead for k in range(i + 1, j))


def _replace(toks: list[_Tok], i: int, j: int, core: str) -> None:
    toks[i:j] = [_Tok(toks[i].pre, toks[i].lead, core, toks[j - 1].trail)]


# ---------------------------------------------------------------- numbers

UNITS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * n for n, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMWORDS = set(UNITS) | set(TENS) | set(SCALES)


def _words_of(tok: _Tok) -> list[str]:
    return tok.low.split("-") if "-" in tok.low else [tok.low]


def _is_numword(tok: _Tok) -> bool:
    return all(w in _NUMWORDS for w in _words_of(tok)) and bool(tok.low)


def parse_number(words: list[str]) -> int | None:
    """Value of a whole run of English number words, or None if the run is not one number."""
    if not words or words[0] in SCALES:
        return None
    total, current = 0, 0
    prev: str | None = None   # "unit" | "teen" | "tens" | "hundred" | "scale"
    for i, w in enumerate(words):
        if w == "and":
            if prev != "hundred" or i == len(words) - 1:
                return None
            continue
        if w in UNITS:
            n = UNITS[w]
            kind = "teen" if n >= 10 else "unit"
            if prev in ("unit", "teen") or (prev == "tens" and kind == "teen") or (prev == "tens" and n == 0):
                return None
            current += n
            prev = kind
        elif w in TENS:
            if prev in ("unit", "teen", "tens"):
                return None
            current += TENS[w]
            prev = "tens"
        elif w == "hundred":
            if prev not in ("unit", "teen", "tens") or current == 0 or current % 100 >= 100:
                return None
            current *= 100
            prev = "hundred"
        elif w in SCALES:
            if prev is None or prev == "scale" or current == 0:
                return None
            total += current * SCALES[w]
            current = 0
            prev = "scale"
        else:
            return None
    return total + current


def _number_run(toks: list[_Tok], i: int) -> int:
    """End index of the maximal run of number words (and inner "and") starting at i."""
    j = i
    while j < len(toks) and (_is_numword(toks[j]) or (toks[j].low == "and" and j > i and j + 1 < len(toks)
                                                     and _is_numword(toks[j + 1]) and toks[j - 1].low == "hundred")):
        if j > i and not _clean_span(toks, j - 1, j + 1):
            break
        j += 1
    return j


def _run_words(toks: list[_Tok], i: int, j: int) -> list[str]:
    return [w for k in range(i, j) for w in _words_of(toks[k])]


# ---------------------------------------------------------------- times

_TIME_CUES = {"at", "by", "until", "till", "til", "around", "before", "after", "since"}
_AMPM = {"am": "AM", "a.m": "AM", "pm": "PM", "p.m": "PM"}


def _ampm(toks: list[_Tok], j: int) -> tuple[str | None, int]:
    """Recognize "am"/"pm"/"a.m."/"a m" at j; return (label, tokens consumed)."""
    if j >= len(toks):
        return None, 0
    low = toks[j].low.rstrip(".")
    if low in _AMPM:
        return _AMPM[low], 1
    if low in ("a", "p") and j + 1 < len(toks) and toks[j + 1].low.rstrip(".") == "m" and not toks[j].trail.strip("."):
        return ("AM" if low == "a" else "PM"), 2
    return None, 0


def _minutes(toks: list[_Tok], j: int) -> tuple[int | None, int]:
    """Minute words at j: "oh five", "fifteen", "thirty", "forty five" -> (minutes, tokens used)."""
    if j >= len(toks):
        return None, 0
    w = _words_of(toks[j])
    if w[0] == "oh" and len(w) == 1 and j + 1 < len(toks) and toks[j + 1].low in UNITS \
            and 1 <= UNITS[toks[j + 1].low] <= 9:
        return UNITS[toks[j + 1].low], 2
    if len(w) == 1 and w[0] in UNITS and 10 <= UNITS[w[0]] <= 19:
        return UNITS[w[0]], 1
    if w[0] in TENS and TENS[w[0]] <= 50:
        if len(w) == 2 and w[1] in UNITS and 1 <= UNITS[w[1]] <= 9:
            return TENS[w[0]] + UNITS[w[1]], 1
        if len(w) == 1 and j + 1 < len(toks) and toks[j + 1].low in UNITS and 1 <= UNITS[toks[j + 1].low] <= 9 \
                and not toks[j].trail:
            return TENS[w[0]] + UNITS[toks[j + 1].low], 2
        if len(w) == 1:
            return TENS[w[0]], 1
    return None, 0


def _times(toks: list[_Tok]) -> None:
    i = 0
    while i < len(toks):
        hour = UNITS.get(toks[i].low)
        if hour is None or not 1 <= hour <= 12:
            i += 1
            continue
        cue = i > 0 and toks[i - 1].low in _TIME_CUES and not toks[i - 1].trail
        mins, used = (None, 0) if toks[i].trail else _minutes(toks, i + 1)
        end = i + 1 + used
        if mins is not None and not _clean_span(toks, i, end):
            mins, end = None, i + 1
        label, n = _ampm(toks, end) if not toks[end - 1].trail else (None, 0)
        if label:   # with am/pm the span is always converted below
            _drop_abbrev_period(toks, end + n - 1)
        if mins is not None and (cue or label):
            core = f"{hour}:{mins:02d}" + (f" {label}" if label else "")
            _replace(toks, i, end + n, core)
        elif mins is None and label:
            _replace(toks, i, end + n, f"{hour} {label}")
        i += 1


def _drop_abbrev_period(toks: list[_Tok], k: int) -> None:
    """"a.m." becomes "AM": its period belonged to the abbreviation, unless it also ends the
    sentence (it is the last token, or the next word is capitalized)."""
    t = toks[k]
    if t.trail.startswith(".") and k + 1 < len(toks) and not toks[k + 1].core[:1].isupper():
        t.trail = t.trail[1:]


# ---------------------------------------------------------------- plain numbers, %, $

def _numbers(toks: list[_Tok]) -> None:
    i = 0
    while i < len(toks):
        if not _is_numword(toks[i]) or toks[i].low in SCALES:
            i += 1
            continue
        j = _number_run(toks, i)
        words = _run_words(toks, i, j)
        value = parse_number(words)
        if value is None:
            i = j
            continue
        text = str(value)
        # decimal: "three point five", "zero point two five"
        if j + 1 < len(toks) and toks[j].low == "point" and not toks[j - 1].trail and toks[j + 1].low in UNITS:
            k = j + 1
            digits = []
            while k < len(toks) and toks[k].low in UNITS and UNITS[toks[k].low] <= 9:
                digits.append(str(UNITS[toks[k].low]))
                if toks[k].trail:
                    k += 1
                    break
                k += 1
            if digits:
                text, j = f"{text}.{''.join(digits)}", k
        unit = toks[j].low if j < len(toks) and not toks[j - 1].trail else ""
        multiword = len(words) > 1 or "." in text
        if unit == "percent":
            _replace(toks, i, j + 1, f"{text}%")
        elif unit in ("dollars", "dollar") and (unit == "dollars") != (text == "1"):
            _replace(toks, i, j + 1, f"${text}")
        elif multiword and not (j < len(toks) and toks[j].low in ("o'clock",)):
            _replace(toks, i, j, text)
        i += 1


# ---------------------------------------------------------------- emails, domains, symbols

TLDS = {"com", "org", "net", "io", "ai", "dev", "app", "co", "edu", "gov", "uk", "ca", "de", "fr", "xyz",
        "info", "tv", "gg", "ly", "fm", "eu", "au", "jp", "nl", "ch", "se", "es", "it", "in", "us", "me"}
# These are also everyday words; they count as a TLD only after "dot" and at the end of a domain,
# and never as the *first* label.
_NOT_LABEL = {"the", "a", "an", "this", "that", "my", "your", "our", "their", "his", "her", "its", "at", "and",
              "or", "to", "of", "in", "on", "is"}
_NOT_LOCAL = {"me", "us", "you", "him", "her", "them", "it", "out", "back", "home", "work", "school", "or",
              "and", "here", "there", "look", "now", "least", "all", "first", "last"}
_WORDLIKE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_LOCALLIKE = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")   # email local part ("jane_doe", "j.smith")
_JOINERS = {"dot": ".", "underscore": "_", "dash": "-", "hyphen": "-"}


def _domain(toks: list[_Tok], i: int) -> tuple[str | None, int]:
    """Spoken domain starting at i ("example dot co dot uk", or "example.com" already written)."""
    if i >= len(toks) or toks[i].lead:
        return None, i
    low = toks[i].low
    if "." in low and low.rsplit(".", 1)[-1] in TLDS and re.fullmatch(r"[a-z0-9.-]+", low):
        return low, i + 1
    if not _WORDLIKE.match(low) or low in _NOT_LABEL:
        return None, i
    labels, j = [low], i + 1
    while j + 1 < len(toks) and toks[j].low == "dot" and _WORDLIKE.match(toks[j + 1].low) \
            and _clean_span(toks, j - 1, j + 2):
        labels.append(toks[j + 1].low)
        j += 2
    while len(labels) > 1 and labels[-1] not in TLDS:   # trim to the last TLD
        labels.pop()
        j -= 2
    if len(labels) < 2:
        return None, i
    return ".".join(labels), j


def _emails_and_domains(toks: list[_Tok]) -> None:
    i = 0
    while i < len(toks):
        # https colon slash slash
        scheme = ""
        if toks[i].low in ("https", "http") and i + 3 < len(toks) and [t.low for t in toks[i + 1:i + 4]] == \
                ["colon", "slash", "slash"] and _clean_span(toks, i, i + 4):
            scheme, start = toks[i].low + "://", i + 4
        else:
            start = i
        dom, j = _domain(toks, start)
        if dom is None:
            i += 1
            continue
        # email: <local> at <domain>
        if not scheme and i >= 2 and toks[i - 1].low == "at" and _clean_span(toks, i - 2, j):
            k = i - 2
            local = toks[k].low
            while k >= 2 and toks[k - 1].low in _JOINERS and _LOCALLIKE.match(toks[k - 2].low) \
                    and _clean_span(toks, k - 2, k + 1):
                local = toks[k - 2].low + _JOINERS[toks[k - 1].low] + local
                k -= 2
            word = toks[i - 2].low
            if _LOCALLIKE.match(word) and word not in _NOT_LOCAL and word not in _NUMWORDS:
                _replace(toks, k, j, f"{local}@{dom}")
                i = k + 1
                continue
        # URL path: "slash pricing slash team"
        path = ""
        while j + 1 < len(toks) and toks[j].low == "slash" and _WORDLIKE.match(toks[j + 1].low) \
                and _clean_span(toks, j - 1, j + 2):
            path += "/" + toks[j + 1].low
            j += 2
        if scheme or path or dom != toks[start].low:   # something spoken was actually converted
            _replace(toks, i, j, scheme + dom + path)
        i += 1


_SYMBOLS = [
    (re.compile(r"\bat sign\b", re.I), "@"),
    (re.compile(r"\bampersand\b", re.I), "&"),
    (re.compile(r"\bpercent sign\b", re.I), "%"),
    (re.compile(r"\bdollar sign\b", re.I), "$"),
    (re.compile(r"\b(?:hash|pound) sign\b", re.I), "#"),
    (re.compile(r"\bplus sign\b", re.I), "+"),
    (re.compile(r"\bequals sign\b", re.I), "="),
]


# ---------------------------------------------------------------- entry point

def normalize(text: str) -> str:
    """Apply every rule. Idempotent: running it twice gives the same result."""
    if not text or not text.strip():
        return text
    tail = text[len(text.rstrip()):]
    toks = _tokens(text.rstrip())
    _emails_and_domains(toks)
    _times(toks)
    _numbers(toks)
    out = _join(toks, tail)
    for pattern, sym in _SYMBOLS:
        out = pattern.sub(sym, out)
    return out

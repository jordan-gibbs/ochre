"""Python port of the app's deterministic text pre-pass (crates/ochre/src/text.rs).

The app does ``apply_corrections(collapse_repeats(strip_hesitations(joined)), dictionary)``
(``text::prepass``) on the recognizer output before the refinement model sees it
(crates/ochre/src/app.rs). Training `raw` must be built
the same way, byte for byte. ``parity.py`` proves equality against the Rust code on a test table
(it runs ``cargo run -p owf --example text_parity``).

Rust/Python character-class mapping used here:
  char::is_whitespace  -> _WS (Unicode White_Space; Python's str.isspace also matches \\x1c-\\x1f)
  char::is_alphanumeric -> isalpha() or isnumeric()
  char::is_uppercase   -> isupper() on one char
"""

from __future__ import annotations

import re
import string

HESITATIONS = frozenset([
    "um", "umm", "ummm", "uh", "uhh", "uhhh", "uhm", "ah", "ahh", "er", "erm", "hm", "hmm", "hmmm",
    "mm", "mmm", "mhm",
])

# Unicode White_Space (what Rust's split_whitespace uses).
_WS = ("\t\n\x0b\x0c\r \x85\xa0           "
       "     　")
_WS_RE = re.compile("[" + re.escape(_WS) + "]+")


def split_whitespace(text: str) -> list[str]:
    return [t for t in _WS_RE.split(text) if t]


def _alnum(c: str) -> bool:
    return c.isalpha() or c.isnumeric()


def _trim_non_alnum(token: str) -> str:
    i, j = 0, len(token)
    while i < j and not _alnum(token[i]):
        i += 1
    while j > i and not _alnum(token[j - 1]):
        j -= 1
    return token[i:j]


def _is_upper(c: str) -> bool:
    return c.isupper()


def capitalize(s: str) -> str:
    return s[:1].upper() + s[1:] if s else ""


def is_hesitation(token: str) -> bool:
    raw_core = _trim_non_alnum(token)
    # A two-letter all-caps token is an acronym, not a sound: "the ER", "UM football". Longer
    # all-caps forms ("UMM", "HMM") are recognizers shouting a hesitation (text.rs, 2026-10-04).
    if len(raw_core) == 2 and all(_is_upper(c) for c in raw_core):
        return False
    core = raw_core.lower()
    return bool(core) and core in HESITATIONS


def strip_hesitations(text: str) -> str:
    out: list[str] = []
    capitalize_next = False
    for token in split_whitespace(text):
        if not is_hesitation(token):
            t = token
            if capitalize_next:
                t = capitalize(t)
                capitalize_next = False
            out.append(t)
            continue
        at_sentence_start = not out or out[-1].endswith((".", "!", "?"))
        stripped = token.rstrip("\"')")
        terminal = stripped[-1] if stripped and stripped[-1] in ".!?" else None
        if out:
            prev = out[-1]
            if terminal is not None:
                trimmed = prev.rstrip(",;")
                out[-1] = trimmed if trimmed.endswith((".", "!", "?")) else trimmed + terminal
            elif token.endswith(",") and prev.endswith(","):
                out[-1] = prev[:-1]
                if not out[-1]:
                    out.pop()
        if at_sentence_start and token[:1] and _is_upper(token[0]):
            capitalize_next = True
        elif at_sentence_start and not out:
            first_alpha = next((c for c in text if c.isalpha()), None)
            capitalize_next = first_alpha is not None and _is_upper(first_alpha)
    return " ".join(out)


# --------------------------------------------------------------------------- corrections


def _is_word_char(c: str) -> bool:
    return _alnum(c) or c in "'_"


def _replace_whole(hay: str, needle: str, with_fn) -> str:
    """Byte-level mirror of text.rs `replace_whole` (indices are UTF-8 byte offsets there)."""
    hay_b, lower_b = hay.encode(), hay.lower().encode()
    needle_b, needle_lower_b = needle.encode(), needle.lower().encode()
    if len(lower_b) != len(hay_b) or len(needle_lower_b) != len(needle_b):
        return hay
    out = bytearray()
    i = 0
    while True:
        pos = lower_b.find(needle_lower_b, i)
        if pos < 0 or not needle_lower_b:
            if not needle_lower_b:  # Rust's find("") matches at every position; callers filter empties
                return hay
            break
        start, end = pos, pos + len(needle_lower_b)
        before = hay_b[:start].decode("utf-8", errors="strict")
        after = hay_b[end:].decode("utf-8", errors="strict")
        before_ok = not before or not _is_word_char(before[-1])
        after_ok = not after or not _is_word_char(after[0])
        out += hay_b[i:start]
        found = hay_b[start:end].decode()
        out += (with_fn(found) if before_ok and after_ok else found).encode()
        i = end
    out += hay_b[i:]
    return out.decode()


def apply_corrections(text: str, words: list[str] | None = None,
                      replacements: dict[str, str] | None = None) -> str:
    out = text
    reps = [(k, v) for k, v in (replacements or {}).items() if k.strip()]
    # Rust: sort_by_key(Reverse(k.len())) is stable over a BTreeMap (sorted keys); len is bytes.
    reps.sort(key=lambda kv: kv[0])
    reps.sort(key=lambda kv: -len(kv[0].encode()))
    for said, written in reps:
        out = _replace_whole(out, said, lambda _f, w=written: w)
    for word in [w for w in (words or []) if w.strip()]:
        def fix(found: str, word: str = word) -> str:
            starts_upper = bool(found) and _is_upper(found[0])
            fixed = word
            if starts_upper and not (fixed and _is_upper(fixed[0])):
                fixed = capitalize(fixed)
            return fixed
        out = _replace_whole(out, word, fix)
    return out


# --------------------------------------------------------------------------- repeats

MIN_REPEAT_WORDS = 3        # three or more copies
MIN_PAIR_REPEAT_WORDS = 5   # exactly two copies
MAX_REPEAT_WORDS = 40
COMMAND_WORDS = frozenset(["bullet", "comma", "period", "colon", "newline"])
_ASCII_PUNCT = frozenset(string.punctuation)   # == Rust char::is_ascii_punctuation
NUMBER_WORDS = frozenset("""zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand
million point dot dash""".split())


def _repeat_key(token: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in token if c not in _ASCII_PUNCT)


def _collapsible(keys: list[str]) -> bool:
    return (all(keys) and any(k != keys[0] for k in keys)
            and not all((k and all("0" <= c <= "9" for c in k)) or k in NUMBER_WORDS for k in keys)
            and not any(k in COMMAND_WORDS for k in keys)
            and not any(a == "new" and b in ("line", "paragraph") for a, b in zip(keys, keys[1:])))


def collapse_repeats(text: str) -> str:
    """Mirror of text.rs `collapse_repeats`: a phrase said 3+ times in a row (3+ words) or exactly
    twice (5+ words), case and ASCII punctuation ignored, becomes one copy (the first, with the
    last copy's final token)."""
    toks = split_whitespace(text)
    keys = [_repeat_key(t) for t in toks]
    n, out, changed, i = len(toks), [], False, 0

    def copies_at(i: int, ln: int) -> int:
        c = 1
        while i + (c + 1) * ln <= n and keys[i + c * ln:i + (c + 1) * ln] == keys[i:i + ln]:
            c += 1
        return c

    while i < n:
        longest = min(MAX_REPEAT_WORDS, (n - i) // 2)
        hit = None
        for ln in range(longest, MIN_REPEAT_WORDS - 1, -1):
            c = copies_at(i, ln)
            if (c >= 3 or (c == 2 and ln >= MIN_PAIR_REPEAT_WORDS)) and _collapsible(keys[i:i + ln]):
                hit = (ln, c)
                break
        if hit is None:
            out.append(toks[i])
            i += 1
            continue
        ln, copies = hit
        out.extend(toks[i:i + ln - 1])
        out.append(toks[i + copies * ln - 1])
        i += copies * ln
        changed = True
    return " ".join(out) if changed else text


def app_prepass(joined: str, dictionary: list[str] | None = None) -> str:
    """Exactly what crates/ochre/src/app.rs does to the joined recognizer text (text::prepass)."""
    return apply_corrections(collapse_repeats(strip_hesitations(joined)), dictionary or [])

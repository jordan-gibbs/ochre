"""Chunked refinement for long dictations: byte-identical port of ``crates/ochre-refine/src/chunk.rs``.

The eval harness (``tools/eval/eval_refine.py``, chunk mode) uses this to split exactly as the
app does; ``tools/eval/chunk_parity.py`` runs both on the same inputs and fails on any
difference. Everything is ASCII-only on purpose, so the two cannot drift on Unicode classes.
See the Rust module for the boundary rules and docs/refine-chunking.md for the numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_WORDS = 80
TARGET_WORDS = 40
MIN_SENTENCE_WORDS = 5

SEP_NONE, SEP_SPACE, SEP_PARAGRAPH = "", " ", "\n\n"

WEAK_CUES = frozenset(["no", "nope", "wait", "actually", "sorry", "oops", "rather", "correction", "scratch",
                       "strike", "nevermind", "pardon", "rephrase", "instead"])
STRONG_BIGRAMS = (("i", "mean"), ("make", "that"), ("scratch", "that"), ("strike", "that"), ("or", "rather"),
                  ("no", "wait"), ("never", "mind"), ("let", "me"))
STRONG_WORDS = frozenset(["sorry", "correction", "rephrase", "nevermind"])
CONTINUATION = frozenset("""and but or nor so because cause which who whom whose that where when while whereas
although though unless until till if for to with without by from of in on at into like including especially
than as not plus then usually mostly except instead otherwise only even rather both either neither after before
since about via per etc""".split())
SUBORDINATORS = frozenset("if when whenever once unless although though because since while whereas until before "
                          "after as".split())
FUNCTION_END = frozenset("""the a an to of and or but with for in on at from by my your our their his her its this
that these those is are was were be been will would can could should if so because than as about into like i we
you""".split())
ABBREVIATIONS = frozenset("mr mrs ms dr prof st sr jr vs etc e.g i.e a.m p.m inc ltd co corp approx dept est fig no "
                          "vol".split())
LIST_STRONG = frozenset(["bullet", "bullets", "newline", "colon"])
LIST_ORDINALS = frozenset("first firstly second secondly third thirdly fourth fifth lastly finally".split())
SMALL_NUMBERS = frozenset("one two three four five six seven eight nine ten 1 2 3 4 5 6 7 8 9 10".split())

_ASCII_ALNUM = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_CLOSERS = "\"')]”’"
_OPENERS = "\"'([“‘"
# Unicode White_Space, what Rust's split_whitespace uses (same table as tools/refine-data/owf_text.py).
_WS = ("\t\n\x0b\x0c\r \x85\xa0            "
       "    　")
_WS_RE = re.compile("[" + re.escape(_WS) + "]+")


@dataclass(frozen=True)
class Piece:
    sep: str
    text: str


def split_whitespace(text: str) -> list[str]:
    return [t for t in _WS_RE.split(text) if t]


def _ascii_lower(s: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


def core(token: str) -> str:
    i, j = 0, len(token)
    while i < j and token[i] not in _ASCII_ALNUM:
        i += 1
    while j > i and token[j - 1] not in _ASCII_ALNUM:
        j -= 1
    return _ascii_lower(token[i:j])


def ends_sentence(token: str) -> bool:
    t = token.rstrip(_CLOSERS)
    if t.endswith("...") or t.endswith("…"):
        return False
    if not t.endswith((".", "!", "?")):
        return False
    c = core(token)
    if t.endswith(".") and (c in ABBREVIATIONS or len(c) == 1):
        return False
    return True


def starts_upper(token: str) -> bool:
    t = token.lstrip(_OPENERS)
    return bool(t) and "A" <= t[0] <= "Z"


def _paragraph_command_at(tokens: list[str], i: int) -> bool:
    if i == 0 or i + 2 >= len(tokens):
        return False
    new, para = tokens[i], tokens[i + 1]
    if new not in ("New", "new"):
        return False
    if _ascii_lower(para.rstrip(".,!?:;")) != "paragraph":
        return False
    prev = tokens[i - 1].rstrip(_CLOSERS)
    return prev.endswith((".", "!", "?")) or (new == "New" and prev.endswith((",", ";", ":")))


def _paragraphs(tokens: list[str]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    i = 0
    while i < len(tokens):
        if _paragraph_command_at(tokens, i):
            out.append([])
            i += 2
            continue
        out[-1].append(tokens[i])
        i += 1
    return [p for p in out if p]


def _sentences(tokens: list[str]) -> list[list[str]]:
    out, cur = [], []
    for t in tokens:
        cur.append(t)
        if ends_sentence(t):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _has_bigram(words: list[str], pairs) -> bool:
    return any((words[k], words[k + 1]) == p for k in range(len(words) - 1) for p in pairs)


def _list_cue(words: list[str]) -> tuple[bool, bool]:
    strong = any(w in LIST_STRONG for w in words) or any(
        (a == "new" and b == "line") or (a == "next" and b == "line")
        or (a in ("number", "step", "item", "point") and b in SMALL_NUMBERS)
        for a, b in zip(words, words[1:]))
    ordinal = any(w in LIST_ORDINALS for w in words[:3])
    return strong, ordinal


def _boundary_ok(s: list[str], t: list[str]) -> bool:
    if len(s) < MIN_SENTENCE_WORDS or len(t) < MIN_SENTENCE_WORDS:
        return False
    if not starts_upper(t[0]) or not starts_upper(s[0]):
        return False
    sw = [core(x) for x in s]
    tw = [core(x) for x in t]
    if tw[0] in CONTINUATION:
        return False
    if sw[-1] in FUNCTION_END or sw[0] in SUBORDINATORS:
        return False
    if any(w in WEAK_CUES for w in tw[:4]) or any(w in WEAK_CUES for w in sw[max(0, len(sw) - 4):]):
        return False
    tail = sw[max(0, len(sw) - 4):]
    if _has_bigram(tw, STRONG_BIGRAMS) or any(w in STRONG_WORDS for w in tw) or _has_bigram(tail, STRONG_BIGRAMS):
        return False
    return True


def _boundaries(sents: list[list[str]]) -> list[bool]:
    n = len(sents)
    ok = [_boundary_ok(sents[j], sents[j + 1]) for j in range(max(0, n - 1))]
    cues = [_list_cue([core(x) for x in s]) for s in sents]
    ordinal_count = sum(1 for c in cues if c[1])
    flagged = [j for j in range(n) if cues[j][0] or (cues[j][1] and ordinal_count >= 2)]
    if flagged:
        lo = max(0, flagged[0] - 1)
        hi = min(flagged[-1] + 1, n - 1)
        for j in range(len(ok)):
            if lo <= j < hi:
                ok[j] = False
    return ok


def _div_ceil(a: int, b: int) -> int:
    return -(-a // b)


def _pack(sents: list[list[str]], target: int) -> list[str]:
    ok = _boundaries(sents)
    units: list[list[str]] = [[]]
    for j, s in enumerate(sents):
        units[-1].extend(s)
        if j < len(ok) and ok[j]:
            units.append([])
    total = sum(len(u) for u in units)
    if total <= target + target // 2 or len(units) < 2:
        return [" ".join(t for u in units for t in u)]
    k = _div_ceil(total, target)
    goal = _div_ceil(total, k)
    pieces: list[list[str]] = []
    cur: list[str] = []
    for u in units:
        if cur and len(cur) + len(u) > goal and len(cur) * 2 >= goal:
            pieces.append(cur)
            cur = []
        cur = cur + u
    if cur:
        pieces.append(cur)
    if len(pieces) >= 2 and len(pieces[-1]) < target // 3:
        last = pieces.pop()
        pieces[-1] = pieces[-1] + last
    return [" ".join(p) for p in pieces]


def split(text: str, min_words: int = MIN_WORDS, target: int = TARGET_WORDS) -> list[Piece]:
    """Pieces for chunked refinement; a short text (or one with no usable boundary) is one piece."""
    whole = [Piece(SEP_NONE, text)]
    tokens = split_whitespace(text)
    if len(tokens) < max(min_words, 1) or target == 0:
        return whole
    out: list[Piece] = []
    for pi, para in enumerate(_paragraphs(tokens)):
        para = list(para)
        if para:
            para[-1] = para[-1].rstrip(",;:")
        para = [t for t in para if t]
        if not para:
            continue
        for k, p in enumerate(_pack(_sentences(para), target)):
            sep = SEP_NONE if not out else (SEP_PARAGRAPH if k == 0 and pi > 0 else SEP_SPACE)
            out.append(Piece(sep, p))
    return out if len(out) >= 2 else whole


def join(pieces) -> str:
    """``(sep, text)`` pairs back into one string."""
    return "".join(sep + text for sep, text in pieces)

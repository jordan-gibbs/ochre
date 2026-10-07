"""Automated audit pass 1 (cheap, deterministic). Writes ``audit.auto`` per row.

Checks
* ``wer``: recognizer WER of ``raw`` against ``spoken`` (both through Whisper's English
  normalizer after apostrophes are dropped and times are joined; hesitations removed on both
  sides; cut-off fragments like "th-" removed from the reference). This measures the
  recognizer + TTS, not the target.
* ``missing``: key tokens of ``intended`` (names, numbers, dictionary terms, content words) that
  the speaker said (found in ``spoken``) but that are absent from ``raw`` in every recoverable
  form (exact, split/joined words, near spelling). These are recoverability suspects: the target
  may contain something the model cannot know from ``raw``.
* ``not_in_spoken``: key tokens of ``intended`` that are not in ``spoken`` either (script-level:
  added content, or a number/format the normalizer could not match).
* ``garbage``: empty recognition, U+FFFD or non-Latin script, a repeated n-gram loop, or a raw
  far shorter/longer than what was said.
* ``length``: intended much longer/shorter than spoken, implausible speaking rate.

``severity``: ``suspect`` (an auditor must look closely), ``check``, or ``ok``.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
import warnings
from functools import lru_cache

from owf_text import HESITATIONS

STOP = set("""a about above after again against all am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll
she's should shouldn't so some such than that that's the their theirs them themselves then there there's these
they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves also just like yeah okay ok oh well really
get got go going gonna wanna kind sort thing things know mean think want need make one two three four five six
seven eight nine ten will shall may might must said say says lot bit""".split())


@lru_cache(maxsize=1)
def _normalizer():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer()


def _pre(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"(\d):(\d\d)", r"\1\2", s)               # 3:30 -> 330 (the normalizer reads "three thirty" as 330)
    s = re.sub(r"(\w)'(\w)", r"\1\2", s)                  # it's -> its (scripts are written without apostrophes)
    s = re.sub(r"(?<=\w)@(?=\w)", " at ", s)
    s = re.sub(r"(?<=[a-zA-Z])\.(?=[a-zA-Z]{2,}\b)", " dot ", s)  # example.com -> example dot com
    s = re.sub(r"(?<=\w)/(?=\w)", " slash ", s)
    return s


def norm_tokens(s: str) -> list[str]:
    toks = _normalizer()(_pre(s)).split()
    return [t for t in toks if t not in HESITATIONS]


def spoken_reference(spoken: str) -> str:
    """Spoken script minus cut-off fragments ("th-") and marker punctuation."""
    toks = [t for t in spoken.replace("...", " ").split() if not re.fullmatch(r"[\w']+-[,.]*", t)]
    return " ".join(toks).replace(",", " ")


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """(distance, substitutions, deletions, insertions) by Levenshtein with backtrace."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, s, de, ins = n, m, 0, 0, 0
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            de, i = de + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return d[n][m], s, de, ins


def _forms(tokens: list[str]) -> set[str]:
    """Tokens plus adjacent-pair joins ("git hub" -> "github") and de-hyphenated forms."""
    out = set(tokens)
    out |= {a + b for a, b in zip(tokens, tokens[1:])}
    out |= {a + b + c for a, b, c in zip(tokens, tokens[1:], tokens[2:])}
    out |= {t.replace("-", "") for t in tokens}
    return out


def _recoverable(tok: str, forms: set[str], kind: str) -> bool:
    if tok in forms or tok.replace("-", "") in forms:
        return True
    if any(c.isdigit() for c in tok):
        digits = re.sub(r"\D", "", tok)
        return bool(digits) and any(re.sub(r"\D", "", f) == digits for f in forms if any(c.isdigit() for c in f))
    if len(tok) < 4:
        return False
    cut = 0.72 if kind in ("name", "dictionary") else 0.8
    for f in forms:
        if abs(len(f) - len(tok)) <= 3 and difflib.SequenceMatcher(None, tok, f).ratio() >= cut:
            return True
        if kind != "content" and f[:1] == tok[:1] and len(f) >= 4 and (tok.startswith(f) or f.startswith(tok)):
            return True
    return False


_SENT_START = re.compile(r"(?:^|[.!?:]\s+|\n\s*(?:[-*]|\d+[.)])?\s*)([A-Z][\w'-]*)")
_CAP = re.compile(r"\b([A-Z][\w'-]*|[a-z]+[A-Z][\w'-]*)")


def key_tokens(intended: str, dictionary: list[str]) -> dict[str, str]:
    """normalized token -> kind (dictionary / name / number / content)."""
    starts = {m.group(1) for m in _SENT_START.finditer(intended)}
    names = set()
    for m in _CAP.finditer(intended):
        w = m.group(1)
        if w in ("I", "I'm", "I've", "I'll", "I'd") or (w in starts and w.lower() in STOP):
            continue
        if w in starts and not any(c.isupper() for c in w[1:]):
            continue  # sentence-initial capital: not evidence of a name
        names.update(norm_tokens(w))
    dict_toks = {t for d in dictionary for t in norm_tokens(d)}
    out: dict[str, str] = {}
    for t in norm_tokens(intended):
        if t in out:
            continue
        if t in dict_toks:
            out[t] = "dictionary"
        elif any(c.isdigit() for c in t):
            out[t] = "number"
        elif t in names:
            out[t] = "name"
        elif t not in STOP and len(t) >= 4 and t.isalpha():
            out[t] = "content"
    return out


def garbage_flags(raw: str, raw_asr: str, spoken_n: list[str], raw_n: list[str]) -> list[str]:
    flags = []
    if not raw.strip():
        flags.append("empty")
        return flags
    if "�" in raw_asr:
        flags.append("replacement_char")
    letters = [c for c in raw if c.isalpha()]
    if letters:
        non_latin = sum(1 for c in letters if not unicodedata.name(c, "").startswith("LATIN"))
        if non_latin / len(letters) > 0.05:
            flags.append("non_latin_script")
    words = raw.lower().split()
    for n in (1, 2, 3):
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        run = best = 1
        for i in range(n, len(grams)):
            run = run + 1 if grams[i] == grams[i - n] else 1
            best = max(best, run)
        if best >= (5 if n == 1 else 4):
            flags.append(f"repeat_loop_{n}gram")
            break
    if spoken_n:
        ratio = len(raw_n) / len(spoken_n)
        if ratio < 0.6:
            flags.append("raw_too_short")
        elif ratio > 1.5:
            flags.append("raw_too_long")
    return flags


def audit_row(row: dict) -> dict:
    spoken, raw, intended = row["spoken"], row.get("raw") or "", row.get("clean") or row["intended"]
    ref = norm_tokens(spoken_reference(spoken))
    hyp = norm_tokens(raw)
    dist, s, de, ins = edit_ops(ref, hyp)
    wer = round(dist / max(1, len(ref)), 3)
    raw_forms, spk_forms = _forms(hyp), _forms(ref)
    missing, extra = [], []
    for tok, kind in key_tokens(intended, row.get("dictionary") or []).items():
        if _recoverable(tok, raw_forms, kind):
            continue
        if _recoverable(tok, spk_forms, kind):
            missing.append({"token": tok, "kind": kind})
        else:
            extra.append({"token": tok, "kind": kind})
    flags = garbage_flags(raw, row.get("raw_asr") or raw, ref, hyp)
    n_int = len(norm_tokens(intended))
    length = {"spoken_words": len(ref), "raw_words": len(hyp), "intended_words": n_int}
    if ref and n_int / len(ref) > 1.25 and n_int - len(ref) >= 3:
        flags.append("intended_longer_than_spoken")
    if ref and n_int / len(ref) < 0.4 and len(ref) - n_int >= 4:
        flags.append("intended_much_shorter")
    sec = row.get("audio_s") or 0
    if sec and len(ref) >= 5:
        wps = len(ref) / max(0.1, sec)
        length["words_per_s"] = round(wps, 2)
        if wps > 4.5 or wps < 0.8:
            flags.append("odd_speaking_rate")
    hard = [m for m in missing if m["kind"] in ("name", "number", "dictionary")]
    if (hard or len(missing) >= 2 or wer > 0.35
            or any(f in flags for f in ("empty", "non_latin_script", "replacement_char", "raw_too_short",
                                        "raw_too_long") or f.startswith("repeat_loop"))):
        sev = "suspect"
    elif missing or extra or wer > 0.15 or flags:
        sev = "check"
    else:
        sev = "ok"
    return {"version": 1, "severity": sev, "wer": wer,
            "wer_ops": {"sub": s, "del": de, "ins": ins, "ref_words": len(ref)},
            "missing": missing, "not_in_spoken": extra, "flags": flags, "length": length}

"""ASR-noise simulator: spoken script text -> simulated Parakeet `raw_asr` -> app pre-pass -> `raw`.

Round 4 / Phase D (text-only distillation data). Learned from the real-v2 rows recognized by
Parakeet TDT 0.6B v3 (TTS -> room/noise augmentation -> recognizer). Pure Python, light compute.

  fit                  align tts_text <-> raw_asr on the 85% fit split, learn tables -> asr_sim/model.json
  validate             simulate the held-out 15% (5 seeds), compare statistics -> asr_sim/validation.json
  make-validation-set  real-vs-sim eval pairs -> training/refine/datasets/asr-sim/heldout-pairs.jsonl
  apply --in X --out Y [--seed N] [--keep-hesitations]
                       script rows {id, spoken, intended, dictionary, ...} -> rows with simulated raw

Model (all tables in model.json, see docs/asr-noise-sim.md):
  * word errors: per-word substitution/deletion tables with additive backoff to a word-class rate
    (hes, frag, rep, num, func, short, proper, common, rare); novel substitutions are drawn from a
    vocabulary of recognizer output words by spelling similarity; learned merges ("stand up" ->
    "standup") and splits; insertions; rare long deletion bursts (dropped phrases);
  * recognizer conventions that are not noise ("dont" -> "don't", "ok" -> "okay"): per-word, not
    scaled by the noise level; repeat collapse ("to to" -> "to"); spelled letters merged ("a w s"
    -> "AWS"); "x dot y" -> "x.y"; edge truncation (onset / tail of the clip lost) and end-of-clip
    hallucinations ("... Yeah.");
  * an utterance-level noise multiplier m (empirical distribution of observed/expected errors per
    length bucket; 11-65% of clips have m = 0) that scales every word's error probability;
  * punctuation after each output token: P(none | comma | stop) from the spoken marker after the
    word (pause ",", trail "...", cut "-", none), words since the last sentence break, the next
    and the current spoken word, with hierarchical backoff; stop -> "." / "?" by the first
    question word among the sentence's first three words;
  * casing: sentence-initial capitals, per-word casing (proper nouns, acronyms, "I"), and spurious
    mid-sentence capitals by context;
  * numbers: maximal spoken number spans, typed (pct, money, ordinal, decimal, multi-group/time,
    small, mid, large, million) -> P(digits | type) and the observed formats ("12%", "$15" glued to
    the previous word, "10,000", "1030" vs "10:30", "19th", "1.8.0");
  * spoken hesitations (only when kept; the current TTS strips them): recognized as "Um"/"UMM"
    (then stripped by the pre-pass), dropped, or misheard as a word.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from owf_text import HESITATIONS, app_prepass  # noqa: E402  (byte-identical port of the Rust pre-pass)
from common import TTS_HES, strip_spoken_hesitations  # noqa: E402

REAL = ROOT / "training" / "refine" / "datasets" / "real-v2"
SIM_DIR = HERE / "asr_sim"
MODEL = SIM_DIR / "model.json"
HELDOUT = SIM_DIR / "heldout_ids.json"
VALIDATION = SIM_DIR / "validation.json"
PAIRS = ROOT / "training" / "refine" / "datasets" / "asr-sim" / "heldout-pairs.jsonl"
PAIRS_MAP = SIM_DIR / "heldout_pairs_map.json"  # pair n -> source row id, tts_text, real/sim raw_asr

HES = set(HESITATIONS) | set(TTS_HES)

FUNC = set("""a an the and or but so if then than that this these those there here of to in on at
by for with from up down out over into onto about as is are was were be been being am do does did
have has had i you he she it we they me him her us them my your his its our their mine yours what
which who whom whose when where why how not no yes just can could will would shall should may might
must also too very really like well oh okay ok all any some each every one more most other such only
own same few both own let me get got go i'm i'll i'd i've you're it's that's don't dont can't cant
im ill id ive youre thats its isnt wont""".split())

ONES = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
       "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
       "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
       "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
       "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90, "hundredth": 100,
       "thousandth": 1000}
NUM_CORE = set(ONES) | set(TENS) | set(SCALES) | set(ORD)
NUM_SUFFIX = {"percent": "pct", "dollars": "money", "dollar": "money", "bucks": "money"}
SYMBOLS = {"dot": ".", "dash": "-", "slash": "/", "underscore": "_", "colon": ":"}
STOPS = frozenset({".", "?", "!"})
LEN_BUCKETS = [6, 12, 20, 35, 60]  # utterance length (spoken words) buckets for the noise multiplier
SINCE_BUCKETS = [1, 3, 6, 10, 15, 22]  # upper bounds; last bucket is "23+"


# ============================================================================ tokens


def key(tok: str) -> str:
    """Comparison key: lowercase letters/digits/inner apostrophes."""
    k = re.sub(r"[^a-z0-9']", "", tok.lower().replace("’", "'"))
    return k.strip("'")


def core(tok: str) -> str:
    """Token without surrounding punctuation (keeps inner '.', '-', ''', '$', '%', ':')."""
    return re.sub(r"^[^\w$]+|[^\w%]+$", "", tok)


def hyp_tokens(text: str) -> list[str]:
    """Recognizer text -> tokens; Parakeet glues "$" amounts to the previous word ("to$15")."""
    return re.sub(r"(\w)\$(\d)", r"\1 $\2", text).split()


def is_hes(k: str) -> bool:
    return k in HES


class Ref:
    __slots__ = ("k", "after", "frag", "hes", "cls", "num")

    def __init__(self, k: str, after: str | None, frag: bool, hes: bool):
        self.k, self.after, self.frag, self.hes = k, after, frag, hes
        self.cls = ""
        self.num = -1  # index of the number span this token belongs to


def parse_spoken(text: str) -> list[Ref]:
    """Spoken/tts text -> tokens with the marker after each word (pause / trail / cut / None)."""
    out: list[Ref] = []
    for tok in text.replace("…", "...").split():
        marker = None
        c = tok
        if c.endswith("..."):
            marker, c = "trail", c[:-3]
        while c.endswith((",", ";", ":", ".", "!", "?")):
            marker, c = marker or "pause", c[:-1]
        frag = False
        if c in ("-", "--", ""):
            if out:
                out[-1].after = out[-1].after or "pause"
            continue
        if c.endswith("-") and len(c) > 1:
            c, frag, marker = c.rstrip("-"), True, marker or "cut"
        k = key(c)
        if not k:
            continue
        out.append(Ref(k, marker, frag, is_hes(k) and not frag))
    return out


def number_spans(refs: list[Ref]) -> list[tuple[int, int, str]]:
    """Maximal spoken-number spans -> (start, end_exclusive, type); sets Ref.num."""
    spans = []
    i, n = 0, len(refs)
    while i < n:
        k = refs[i].k
        starts = k in NUM_CORE and k != "oh" or (k == "a" and i + 1 < n and refs[i + 1].k in ("hundred", "thousand", "million"))
        if not starts:
            i += 1
            continue
        j = i + 1
        while j < n:
            kj = refs[j].k
            if refs[j - 1].after in ("pause", "trail", "cut"):
                break
            if kj in NUM_CORE or kj == "oh":
                j += 1
            elif kj in ("and", "point") and j + 1 < n and refs[j + 1].k in NUM_CORE | {"oh"}:
                j += 1
            else:
                break
        words = [r.k for r in refs[i:j]]
        typ = None
        if j < n and refs[j].k in NUM_SUFFIX and refs[j - 1].after is None:
            typ = NUM_SUFFIX[refs[j].k]
            j += 1
        if typ is None:
            typ = span_type(words, refs[i - 1].k if i else "")
        if words == ["one"] or words == ["a"]:
            i = j
            continue  # "one" is mostly a pronoun / article: not a number span
        for t in range(i, j):
            refs[t].num = len(spans)
        spans.append((i, j, typ))
        i = j
    return spans


def parse_groups(words: list[str]) -> tuple[list[int], str | None, bool]:
    """Number words -> (groups, decimal_tail, ordinal). "ten thirty" -> [10, 30];
    "one hundred and twenty" -> [120]; "four oh four" -> [4, 0, 4]; "one point eight" -> [1], "8"."""
    groups: list[int] = []
    total = cur = 0
    active = False
    last = None  # kind of last word: unit / teen / tens / scale
    dec = None
    ordinal = False
    i = 0
    while i < len(words):
        w = words[i]
        if w == "point":
            if active:
                groups.append(total + cur)
                total = cur = 0
                active = False
            if not groups:
                groups.append(0)
            dec = "".join(str(ONES.get(x, 0)) if x in ONES else "." if x == "point" else "" for x in words[i + 1:])
            break
        if w == "and" or w == "a":
            i += 1
            continue
        if w in ORD:
            ordinal = True
            v = ORD[w]
            kind = "scale" if v >= 100 else ("tens" if v >= 20 and v % 10 == 0 else ("teen" if v >= 10 else "unit"))
            w_val = v
        elif w in ONES:
            w_val = ONES[w]
            kind = "teen" if w_val >= 10 else "unit"
            if w == "oh":
                kind = "oh"
        elif w in TENS:
            w_val, kind = TENS[w], "tens"
        elif w in SCALES:
            w_val, kind = SCALES[w], "scale"
        else:
            i += 1
            continue
        if kind == "scale":
            if not active:
                cur = 1
            if w_val == 100:
                cur = cur * 100
            else:
                total += cur * w_val
                cur = 0
            active, last = True, "scale"
        else:
            new_group = False
            if active:
                if kind == "oh":
                    new_group = True
                elif kind == "unit" and last == "tens":
                    pass
                elif last in ("scale",) and (cur % 100 == 0 or cur == 0):
                    pass
                else:
                    new_group = True
            if new_group:
                groups.append(total + cur)
                total = cur = 0
            cur += w_val
            active, last = True, kind
        i += 1
    if active:
        groups.append(total + cur)
    return groups, dec, ordinal


def span_type(words: list[str], prev: str) -> str:
    groups, dec, ordinal = parse_groups(words)
    if dec is not None:
        return "decimal"
    if ordinal:
        # a lone "first" / "second" / "third" is mostly a list or discourse marker, not a date
        return "ord_small" if len(words) == 1 and groups and groups[0] <= 3 else "ordinal"
    if len(groups) >= 2:
        return "multi"
    v = groups[0] if groups else 0
    if v >= 1_000_000:
        return "million"
    if v >= 100:
        return "large"
    if v >= 10:
        return "mid"
    return "small"


def digits_of(words: list[str]) -> str:
    groups, dec, _ = parse_groups(words)
    if not groups:
        return ""
    if len(groups) == 1:
        s = str(groups[0])
    else:
        s = "".join(str(g) if (g >= 10 or idx == 0) else str(g) for idx, g in enumerate(groups))
    if dec:
        s += "." + dec
    return s


def ordinal_suffix(v: int) -> str:
    if 10 <= v % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th")


# ============================================================================ alignment


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - lev(a, b) / max(len(a), len(b))


def align(refs: list[Ref], hyp: list[str]):
    """Weighted Levenshtein with merges/splits and many-to-one number/symbol spans.
    -> list of ops (type, r0, r1, h0, h1); type in match/sub/del/ins/merge/split/num/sym/skip."""
    hk = [key(h) for h in hyp]
    hdig = [bool(re.search(r"\d", h)) for h in hyp]
    hsym = [bool(re.search(r"[./_:@-]", core(h))) for h in hyp]
    n, m = len(refs), len(hk)
    INF = 1e9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    # number span starts for each ref index
    span_start = {}
    for idx, r in enumerate(refs):
        if r.num >= 0:
            s = idx
            while s > 0 and refs[s - 1].num == r.num:
                s -= 1
            span_start[idx] = s
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, bp = INF, None
            if j > 0:
                c = D[i][j - 1] + (0.01 if not hk[j - 1] else 1.0)
                if c < best:
                    best, bp = c, ("skip" if not hk[j - 1] else "ins", i, j - 1)
            if i > 0:
                c = D[i - 1][j] + 1.0
                if c < best:
                    best, bp = c, ("del", i - 1, j)
            if i > 0 and j > 0:
                r, h = refs[i - 1], hk[j - 1]
                if r.k == h or (r.hes and is_hes(h)):
                    c, t = 0.0, "match"
                elif r.frag and h.startswith(r.k):
                    c, t = 0.3, "sub"
                else:
                    c, t = 1.0 - 0.5 * sim(r.k, h), "sub"
                c += D[i - 1][j - 1]
                if c < best:
                    best, bp = c, (t, i - 1, j - 1)
                if i > 1 and h and refs[i - 2].k + r.k == h:
                    c = D[i - 2][j - 1] + 0.1
                    if c < best:
                        best, bp = c, ("merge", i - 2, j - 1)
                if j > 1 and hk[j - 2] and hk[j - 2] + h == r.k:
                    c = D[i - 1][j - 2] + 0.1
                    if c < best:
                        best, bp = c, ("split", i - 1, j - 2)
                if hdig[j - 1] and (i - 1) in span_start:
                    s0 = span_start[i - 1]
                    hd = re.sub(r"\D", "", hyp[j - 1])
                    for s in range(s0, i):
                        words = [x.k for x in refs[s:i]]
                        k = i - s
                        ok = digits_of([w for w in words if w not in NUM_SUFFIX]).replace(".", "") == hd
                        c = D[s][j - 1] + (0.1 if ok else 0.5 + 0.5 * k)
                        if c < best:
                            best, bp = c, ("num", s, j - 1)
                if hsym[j - 1] and h:
                    for k in range(2, 8):
                        s = i - k
                        if s < 0:
                            break
                        words = [x.k for x in refs[s:i]]
                        if not any(w in SYMBOLS for w in words):
                            continue
                        joined = "".join(w for w in words if w not in SYMBOLS)
                        if joined == h:
                            c = D[s][j - 1] + 0.1
                            if c < best:
                                best, bp = c, ("sym", s, j - 1)
                if h and i >= 3 and len(r.k) == 1:
                    for k in range(3, 7):
                        s = i - k
                        if s < 0 or len(refs[s].k) != 1:
                            break
                        if "".join(x.k for x in refs[s:i]) == h:
                            c = D[s][j - 1] + 0.1
                            if c < best:
                                best, bp = c, ("letters", s, j - 1)
            D[i][j], B[i][j] = best, bp
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        t, pi, pj = B[i][j]
        ops.append((t, pi, i, pj, j))
        i, j = pi, pj
    ops.reverse()
    return ops


# ============================================================================ data


# The round-4 fit set: W1..W11 + R2 (4,611 Parakeet rows). Later batches (W12+) are opt-in
# (--all-batches) so the split and model stay reproducible while new batches land.
FIT_BATCHES = r"(W([1-9]|1[01])|R2)"
ALL_BATCHES = r"(W\d+|R2)"
# --fresh: round-4 batches W12-W51 (augmented audio; W52 is clean audio). None of these rows is in
# the fit or in any v3 training set, so a v3 model can be evaluated on them without leakage.
FRESH_BATCHES = r"W(1[2-9]|[2-4]\d|5[01])"
BATCHES = FIT_BATCHES
FRESH = False


def load_real() -> list[dict]:
    rows = []
    for f in sorted(REAL.glob("*.jsonl")):
        if not re.fullmatch(BATCHES, f.stem):
            continue
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                d = json.loads(line)
                if str(d.get("raw_engine", "")).startswith("parakeet") and d.get("tts_text") and d.get("raw_asr") is not None:
                    rows.append(d)
    return rows


def is_heldout(rid: str) -> bool:
    if FRESH:
        return True
    return int(hashlib.sha256(f"asr-sim|{rid}".encode()).hexdigest()[:8], 16) % 100 < 15


def proper_words(intended: str | None, dictionary: list[str] | None) -> dict[str, str]:
    """Lowercase key -> cased form for words written capitalized mid-sentence in `intended`."""
    out = {}
    for w in dictionary or []:
        for p in w.split():
            out[key(p)] = core(p)
    if intended:
        toks = intended.split()
        for i, t in enumerate(toks):
            c = core(t)
            if not c or not c[0].isupper():
                continue
            if i == 0 or toks[i - 1].endswith((".", "!", "?", ":")):
                if not (len(c) >= 2 and c.isupper()):
                    continue
            if key(c) == "i" or key(c).startswith("i'"):
                continue
            out[key(c)] = c
    return out


def repeats(refs: list[Ref]) -> list[tuple[list[int], list[int]]]:
    """Immediately repeated 1-3 word groups, ignoring hesitations ("the uh the", "has anyone, has
    anyone") -> [(first copy ref indices, second copy ref indices)]."""
    idx = [i for i, r in enumerate(refs) if not r.hes]
    keys = [refs[i].k for i in idx]
    out = []
    used = set()
    n = len(keys)
    i = 0
    while i < n:
        for L in (3, 2, 1):
            if i + 2 * L <= n and keys[i:i + L] == keys[i + L:i + 2 * L] and not used & set(range(i, i + 2 * L)):
                out.append((idx[i:i + L], idx[i + L:i + 2 * L]))
                used.update(range(i, i + 2 * L))
                i += L - 1
                break
        i += 1
    return out


def letter_runs(refs: list[Ref]) -> list[tuple[int, int]]:
    """Runs of >= 2 spelled single letters with no pause inside ("a w s", "q a") -> [(start, end)]."""
    out = []
    i, n = 0, len(refs)
    while i < n:
        j = i
        while j < n and len(refs[j].k) == 1 and refs[j].k.isalpha() and not refs[j].hes and refs[j].num < 0:
            j += 1
            if refs[j - 1].after:
                break
        if j - i >= 2 and any(refs[t].k not in ("a", "i") for t in range(i, j)):
            out.append((i, j))
            i = j
        else:
            i = max(i + 1, j if j > i + 1 else i + 1)
    return out


def classify(refs: list[Ref], common: set[str], proper: dict[str, str]) -> None:
    rep = set()
    for a, b in repeats(refs):
        rep.update(a)
        rep.update(b)
    for i, r in enumerate(refs):
        if r.hes:
            r.cls = "hes"
        elif r.frag:
            r.cls = "frag"
        elif i in rep:
            r.cls = "rep"
        elif r.num >= 0:
            r.cls = "num"
        elif r.k in FUNC:
            r.cls = "func"
        elif r.k in proper:
            r.cls = "proper"
        elif len(r.k) <= 3:
            r.cls = "short"
        elif r.k in common:
            r.cls = "common"
        else:
            r.cls = "rare"


def case_of(tok: str) -> str:
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return "none"
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        return "upper"
    if letters[0].isupper():
        return "cap"
    return "lower"


def punct_of(tok: str) -> str:
    t = tok.rstrip("\"')")
    if t.endswith("?"):
        return "?"
    if t.endswith("!"):
        return "!"
    if t.endswith(".") and not t.endswith("..."):
        return "."
    if t.endswith("..."):
        return "."
    if t.endswith((",", ";", ":")):
        return ","
    return ""


def since_bucket(n: int) -> int:
    return bisect.bisect_left(SINCE_BUCKETS, n)


def len_bucket(n: int) -> int:
    return bisect.bisect_left(LEN_BUCKETS, n)


def marker(r: Ref | None) -> str:
    return (r.after or "none") if r else "none"


# ============================================================================ fit


def fit(rows: list[dict]) -> dict:
    train = [r for r in rows if not is_heldout(r["id"])]
    held = sorted(r["id"] for r in rows if is_heldout(r["id"]))
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    HELDOUT.write_text(json.dumps({"rule": "sha256('asr-sim|'+id)[:8] % 100 < 15", "ids": held}, indent=0),
                       encoding="utf-8")
    print(f"fit rows {len(train)}, held-out {len(held)}")

    ref_counts = Counter()
    for row in train:
        for r in parse_spoken(row["tts_text"]):
            ref_counts[r.k] += 1
    common = {w for w, c in ref_counts.items() if c >= 5}

    aligned = []
    for n_done, row in enumerate(train):
        refs = parse_spoken(row["tts_text"])
        spans = number_spans(refs)
        classify(refs, common, proper_words(row.get("intended"), row.get("dictionary")))
        hyp = hyp_tokens(row["raw_asr"])
        ops = align(refs, hyp)
        aligned.append((row, refs, spans, hyp, ops))
        if n_done % 500 == 0:
            print(f"  aligned {n_done}/{len(train)}", flush=True)

    cls_stats = defaultdict(lambda: Counter())
    word_stats = defaultdict(lambda: {"n": 0, "del": 0, "sub": Counter()})
    hes_out = Counter()
    hes_stats = Counter()
    hes_sub = Counter()
    ins_words = Counter()
    n_ins = n_ref_words = 0
    merges = defaultdict(Counter)
    pair_counts = Counter()
    splits = defaultdict(Counter)
    bursts = []
    edge_n = Counter()
    edges = {"lead": Counter(), "trail": Counter()}
    edge_frac = {"lead": defaultdict(list), "trail": defaultdict(list)}
    end_ins = Counter()
    rep_stats = Counter()
    let_stats = Counter()
    burst_words = 0
    sym_stats = defaultdict(Counter)
    num_stats = defaultdict(Counter)
    num_fmt = Counter()
    vocab = Counter()
    per_utt = []

    for row, refs, spans, hyp, ops in aligned:
        for h in hyp:
            k = key(core(h))
            if k and not re.search(r"\d", k) and not is_hes(k):
                vocab[k] += 1
        deleted = {op[1] for op in ops if op[0] == "del"}
        # edge truncation: >= 2 leading / trailing spoken words lost (clip onset/tail not decoded)
        burst_idx = set()
        idx = [i for i, r in enumerate(refs) if not r.hes]
        nb = len_bucket(len(idx))
        edge_n[nb] += 1
        for side, seq in (("lead", idx), ("trail", idx[::-1])):
            L = 0
            for i in seq:
                if i not in deleted:
                    break
                L += 1
            if L >= 2:
                edges[side][nb] += 1
                edge_frac[side][nb].append(round(L / len(idx), 3))
                burst_idx.update(seq[:L])
        # end-of-clip hallucinations ("... Yeah.")
        if hyp and ops[-1][0] == "ins" and key(hyp[-1]) and not is_hes(key(hyp[-1])):
            end_ins[key(hyp[-1])] += 1
            end_ins_skip = len(hyp) - 1
        else:
            end_ins_skip = -1
        # deletion bursts: runs of >= 4 deleted non-hesitation ref words
        run = []
        for op in ops + [("end", 0, 0, 0, 0)]:
            if op[0] == "del":
                if not refs[op[1]].hes and op[1] not in burst_idx:
                    run.append(op[1])
                continue
            if op[0] in ("skip",):
                continue
            if len(run) >= 4:
                bursts.append(len(run))
                burst_idx.update(run)
            run = []
        for a, b in repeats(refs):
            rep_stats["n"] += 1
            for copy in (a, b):
                if all(t in deleted for t in copy):
                    rep_stats["collapsed"] += 1
                    burst_idx.update(copy)  # explained by the collapse, not by word deletions
                    break
        for a, b in letter_runs(refs):
            let_stats["n"] += 1
            if any(op[0] in ("letters", "merge") and op[1] >= a and op[2] <= b for op in ops):
                let_stats["merged"] += 1
        for i in range(len(refs) - 1):
            pair_counts[refs[i].k + " " + refs[i + 1].k] += 1
        for r in refs:
            if r.k in SYMBOLS:
                sym_stats[r.k]["n"] += 1
        obs = exp_dummy = 0
        nonhes = sum(1 for r in refs if not r.hes)
        n_ref_words += nonhes
        burst_words += nonhes
        utt_err = Counter()
        for t, r0, r1, h0, h1 in ops:
            if t == "ins":
                k = key(hyp[h0])
                if h0 == end_ins_skip:
                    continue
                if is_hes(k):
                    hes_out[core(hyp[h0])] += 1  # recognizer hesitation with nothing spoken
                    hes_stats["ins"] += 1
                else:
                    n_ins += 1
                    ins_words[k] += 1
                    utt_err["ins"] += 1
                continue
            if t == "skip":
                continue
            if t == "sym":
                for r in refs[r0:r1]:
                    if r.k in SYMBOLS:
                        sym_stats[r.k]["joined"] += 1
                continue
            if t in ("num", "letters"):
                continue
            if t == "merge":
                a, b = refs[r0], refs[r0 + 1]
                merges[a.k + " " + b.k][core(hyp[h0]).lower()] += 1
                for r in (a, b):
                    word_stats[r.k]["n"] += 1
                    cls_stats[r.cls]["n"] += 1
                    cls_stats[r.cls]["merge"] += 1
                continue
            if t == "split":
                r = refs[r0]
                splits[r.k][" ".join(core(h).lower() for h in hyp[h0:h1])] += 1
                word_stats[r.k]["n"] += 1
                cls_stats[r.cls]["n"] += 1
                cls_stats[r.cls]["split"] += 1
                utt_err["split"] += 1
                continue
            r = refs[r0]
            if r.hes:
                hes_stats["n"] += 1
                if t == "match":
                    hes_stats["hes"] += 1
                    hes_out[core(hyp[h0])] += 1
                elif t == "del":
                    hes_stats["del"] += 1
                else:
                    hes_stats["sub"] += 1
                    hes_sub[key(hyp[h0])] += 1
                continue
            if r0 in burst_idx:
                continue
            ws = word_stats[r.k]
            ws["n"] += 1
            cls_stats[r.cls]["n"] += 1
            if t == "del":
                ws["del"] += 1
                cls_stats[r.cls]["del"] += 1
                utt_err["del"] += 1
            elif t == "sub":
                hk = key(hyp[h0])
                ws["sub"][hk] += 1
                cls_stats[r.cls]["sub"] += 1
                utt_err["sub:" + r.k] += 1
        # number spans: outcome per span
        for si, (s0, s1, typ) in enumerate(spans):
            span_ops = [op for op in ops if op[0] not in ("ins", "skip") and op[1] < s1 and op[2] > s0]
            nums = [op for op in span_ops if op[0] == "num"]
            if nums:
                num_stats[typ]["digits"] += 1
                tok = hyp[nums[-1][3]]
                c = core(tok)
                has_suffix = refs[s1 - 1].k in NUM_SUFFIX
                if typ == "pct":
                    num_fmt["pct_sign" if "%" in c else "pct_word"] += 1
                if typ == "money":
                    if "$" in c:
                        num_fmt["money_sign"] += 1
                    else:
                        num_fmt["money_word"] += 1
                if typ == "multi":
                    num_fmt["multi_colon" if ":" in c else "multi_plain"] += 1
                d = re.sub(r"\D", "", c)
                if len(d) == 4 and "." not in c:
                    num_fmt["four_comma" if "," in c else "four_plain"] += 1
            elif all(op[0] == "match" for op in span_ops):
                num_stats[typ]["words"] += 1
            else:
                num_stats[typ]["other"] += 1
        per_utt.append((refs, burst_idx, utt_err))
        num_fmt["money_glued"] += len(re.findall(r"\w\$\d", row["raw_asr"]))
        num_fmt["money_spaced"] += len(re.findall(r"(?:^|\s)\$\d", row["raw_asr"]))

    # ---- class and word rates
    classes = {}
    for c, s in cls_stats.items():
        n = max(1, s["n"])
        classes[c] = {"n": s["n"], "sub": s["sub"] / n, "del": s["del"] / n,
                      "merge": s["merge"] / n, "split": s["split"] / n}
    words = {}
    for w, s in word_stats.items():
        nerr = s["del"] + sum(s["sub"].values())
        if s["n"] >= 3 or nerr:
            words[w] = {"n": s["n"], "del": s["del"], "sub": dict(s["sub"].most_common(12))}
            top = s["sub"].most_common(1)
            # a recognizer convention ("dont" -> "don't", "ok" -> "okay"), not noise: its rate is
            # not scaled by the utterance noise level
            if s["n"] >= 3 and top and top[0][1] / s["n"] >= 0.3:
                words[w]["conv"] = 1
    model = {
        "version": 1,
        "fit_rows": len(train),
        "common": sorted(common),
        "classes": classes,
        "words": words,
        "smooth_k": 4.0,
        "novel_alpha": 1.5,
        "ins": {"rate": n_ins / max(1, n_ref_words), "words": dict(ins_words.most_common(80))},
        "merges": {p: {"n": pair_counts[p], "out": dict(o)} for p, o in merges.items()},
        "splits": {w: dict(o) for w, o in splits.items()},
        "hes": {"n": hes_stats["n"], "p_hes": hes_stats["hes"] / max(1, hes_stats["n"]),
                "p_del": hes_stats["del"] / max(1, hes_stats["n"]),
                "p_sub": hes_stats["sub"] / max(1, hes_stats["n"]),
                "surface": dict(hes_out.most_common(20)), "sub": dict(hes_sub.most_common(40))},
        "burst": {"rate": len(bursts) / max(1, burst_words), "lengths": dict(Counter(bursts))},
        "edges": {side: {"rate_by_length": {str(b): edges[side][b] / max(1, edge_n[b]) for b in edge_n},
                         "fractions": {str(b): sorted(v) for b, v in edge_frac[side].items()}} for side in edges},
        "end_ins": {"rate": sum(end_ins.values()) / max(1, len(train)), "words": dict(end_ins.most_common(30))},
        "symbols": {s: c["joined"] / c["n"] for s, c in sym_stats.items() if c["n"]},
        "repeats": dict(rep_stats),
        "letters": dict(let_stats),
        "numbers": {"types": {t: dict(c) for t, c in num_stats.items()}, "format": dict(num_fmt)},
        "vocab": dict(vocab.most_common(6000)),
    }

    # ---- utterance noise multiplier: observed / expected errors
    sim_model = Sim(model)
    ratios = []
    for refs, burst_idx, err in per_utt:
        exp = 0.0
        for i, r in enumerate(refs):
            if r.hes or i in burst_idx or (r.num >= 0):
                continue
            ps, pd, conv = sim_model.word_rates(r)
            exp += pd + (0 if conv else ps)
        exp += model["ins"]["rate"] * sum(1 for r in refs if not r.hes)
        obs = err["del"] + err["ins"] + sum(v for e, v in err.items()
                                            if e.startswith("sub:") and "conv" not in words.get(e[4:], {}))
        nw = sum(1 for r in refs if not r.hes)
        ratios.append((len_bucket(nw), obs / exp if exp > 0 else 0.0))
    mult = {}
    for b in range(len(LEN_BUCKETS) + 1):
        rs = sorted(x for bb, x in ratios if bb == b)
        q = [round(x, 3) for x in rs]  # the full empirical distribution (heavy tail matters)
        mult[str(b)] = {"quantiles": q, "n": len(rs), "share_zero": sum(1 for r in rs if r == 0) / len(rs),
                        "mean": sum(rs) / len(rs)}
    model["multiplier"] = {"by_length": mult, "buckets": LEN_BUCKETS,
                           "mean": sum(x for _, x in ratios) / len(ratios)}

    # ---- punctuation + casing, observed on hyp tokens anchored to ref tokens
    punct = defaultdict(Counter)
    final = defaultdict(Counter)
    stop = defaultdict(Counter)
    case_tab = defaultdict(Counter)
    spur = defaultdict(Counter)
    after_stop = Counter()
    first_cap = Counter()
    proper_case = Counter()
    nx_counts = Counter()
    for row, refs, spans, hyp, ops in aligned:
        for i in range(len(refs) - 1):
            nx_counts[refs[i + 1].k] += 1
    top_next = {w for w, _ in nx_counts.most_common(80)}
    model["top_next"] = sorted(top_next)

    for row, refs, spans, hyp, ops in aligned:
        prop = proper_words(row.get("intended"), row.get("dictionary"))
        since = 0
        sent: list[str] = []
        prev_p = "."
        n_ref = len(refs)
        for t, r0, r1, h0, h1 in ops:
            if t in ("skip", "del"):
                continue
            for j in range(h0, h1):
                tok = hyp[j]
                p = punct_of(tok)
                cs = case_of(core(tok))
                k = key(core(tok))
                last_tok = j == h1 - 1
                anchor = refs[r1 - 1] if (t != "ins" and last_tok) else None
                aidx = r1 - 1 if anchor is not None else None
                # casing
                if k and not re.search(r"\d", k):
                    if prev_p in STOPS:
                        if j == 0:
                            first_cap[cs == "cap" or cs == "upper"] += 1
                        else:
                            after_stop[cs != "lower"] += 1
                    else:
                        case_tab[k][cs] += 1
                        pm = "none"
                        if t != "ins" and r0 > 0:
                            pm = marker(refs[r0 - 1])
                        spur[(prev_p or "_") + "|" + pm][k + "\t" + cs] += 1
                        if t == "match" and refs[r0].k in prop:
                            proper_case[case_of(core(tok)) == case_of(prop[refs[r0].k])] += 1
                if len(sent) < 3:
                    sent.append((refs[r0].k if t != "ins" and r0 < n_ref else k) or "_")
                since += 1
                if anchor is not None:
                    fw = fw_class(sent, top_next)
                    if aidx == n_ref - 1:
                        final[fw][p or "none"] += 1
                    else:
                        nxt = refs[aidx + 1]
                        feats = feat_keys(anchor, since, nxt, top_next)
                        b = "stop" if p in STOPS else ("comma" if p == "," else "none")
                        for f in feats:
                            punct[f][b] += 1
                        if b == "stop":
                            stop[fw][p] += 1
                if p in STOPS:
                    since = 0
                    sent = []
                prev_p = p if p else ("" if k else prev_p)

    model["punct"] = {f: dict(c) for f, c in punct.items() if sum(c.values()) >= 3}  # sparse leaves add nothing
    model["final"] = {f: dict(c) for f, c in final.items()}
    model["stop"] = {f: dict(c) for f, c in stop.items()}
    model["case"] = {"first_cap": first_cap[True] / max(1, sum(first_cap.values())),
                     "after_stop_cap": after_stop[True] / max(1, sum(after_stop.values())),
                     "proper_case_match": proper_case[True] / max(1, sum(proper_case.values()))}
    words_case = {}
    for k, c in case_tab.items():
        n = sum(c.values())
        nl = n - c["lower"] - c["none"]
        if (n >= 2 and nl / n >= 0.5) or (n == 1 and nl == 1 and k not in common):
            words_case[k] = {cs: v for cs, v in c.items() if cs != "none"}
    model["case"]["words"] = words_case
    spur_rates = {}
    for ctx, c in spur.items():
        tot = cap = 0
        for kc, v in c.items():
            k, cs = kc.split("\t")
            if k in words_case:
                continue
            tot += v
            cap += v if cs in ("cap", "upper") else 0
        spur_rates[ctx] = [cap, tot]
    model["case"]["spurious"] = spur_rates
    return model


def fw_class(sent: list[str], top: set[str]) -> str:
    """Sentence-type key for '?' vs '.': the first question word among the sentence's first three
    spoken words, with its position ("are@2" for "okay so are you ...")."""
    for pos, w in enumerate(sent[:3]):
        if w in QWORDS:
            return f"{w}@{pos}"
    return "_other"


QWORDS = set("can could would will is are was were do does did what why how where when who which should "
             "shall have has may might isnt arent doesnt dont didnt wont".split())


def feat_keys(anchor: Ref, since: int, nxt: Ref, top: set[str]) -> list[str]:
    m = anchor.after or "none"
    if anchor.hes:
        m = "hes_" + m
    sb = since_bucket(since)
    if nxt.hes:
        nx, nc = "<hes>", "hes"
    else:
        nx = nxt.k if nxt.k in top else "_"
        nc = nxt.cls or "x"
    cw = anchor.k if anchor.k in top else "_"
    # fine -> coarse; the current word catches lexical commas ("hi, ...", "okay, ...", "wait, ...")
    return [f"{m}|{sb}|{nx}|{cw}", f"{m}|{sb}|{nx}", f"{m}|{sb}|c:{nc}", f"{m}|{sb}", f"{m}"]


# ============================================================================ simulate


class Sim:
    def __init__(self, model: dict):
        self.m = model
        self.common = set(model.get("common", []))
        self.k = model["smooth_k"]
        self.vocab = model.get("vocab", {})
        self._by_first: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for w, c in self.vocab.items():
            if len(w) >= 2:
                self._by_first[w[0]].append((w, c))
        self.top_next = set(model.get("top_next", []))

    # ---- word error rates
    def word_rates(self, r: Ref) -> tuple[float, float, bool]:
        cl = self.m["classes"].get(r.cls) or self.m["classes"].get("rare") or {"sub": 0.05, "del": 0.02}
        ws = self.m["words"].get(r.k)
        k = self.k
        if ws:
            n = ws["n"]
            ps = (sum(ws["sub"].values()) + k * cl["sub"]) / (n + k)
            pd = (ws["del"] + k * cl["del"]) / (n + k)
        else:
            ps, pd = cl["sub"], cl["del"]
        return ps, pd, bool(ws and ws.get("conv"))

    def substitute(self, w: str, rng: random.Random) -> str | None:
        ws = self.m["words"].get(w)
        subs = ws["sub"] if ws else {}
        tot = sum(subs.values())
        alpha = self.m["novel_alpha"]
        if tot and rng.random() < tot / (tot + alpha):
            return weighted(subs, rng)
        return self.novel(w, rng)

    def novel(self, w: str, rng: random.Random) -> str | None:
        """A plausible recognizer word for `w`: similar spelling, frequency-weighted."""
        cands = []
        firsts = {w[0]} if w else set()
        for f in firsts:
            for v, c in self._by_first.get(f, []):
                if v == w or abs(len(v) - len(w)) > max(2, len(w) // 3):
                    continue
                s = sim(v, w)
                if s >= 0.5:
                    cands.append((v, (s ** 4) * math.log(2 + c)))
        if not cands and len(w) >= 6:
            for cut in range(3, len(w) - 2):  # split into two known words
                a, b = w[:cut], w[cut:]
                if a in self.vocab and b in self.vocab:
                    return a + " " + b
        if not cands:
            if w.endswith("s") and len(w) > 3:
                return w[:-1]
            return w + "s" if len(w) > 3 else None
        tot = sum(c for _, c in cands)
        x = rng.random() * tot
        for v, c in cands:
            x -= c
            if x <= 0:
                return v
        return cands[-1][0]

    def multiplier(self, rng: random.Random, n_words: int) -> float:
        q = self.m["multiplier"]["by_length"][str(len_bucket(n_words))]["quantiles"]
        return q[rng.randrange(len(q))]

    # ---- numbers
    def render_number(self, words: list[str], typ: str, rng: random.Random) -> list[tuple[str, bool]] | None:
        """-> list of (token, glue_to_previous) or None to keep the words."""
        st = self.m["numbers"]["types"].get(typ, {})
        # spans with recognition errors ("other") were not rendered as digits: count them as words
        d, wd = st.get("digits", 0), st.get("words", 0) + st.get("other", 0)
        if rng.random() >= (d + 0.5) / (d + wd + 1.0):
            return None
        fmt = self.m["numbers"]["format"]
        core_words = [w for w in words if w not in NUM_SUFFIX]
        groups, dec, ordinal = parse_groups(core_words)
        if not groups:
            return None

        def p(a: str, b: str) -> float:
            x, y = fmt.get(a, 0), fmt.get(b, 0)
            return (x + 0.5) / (x + y + 1.0)

        def fmt_int(v: int) -> str:
            s = str(v)
            if v >= 10000 or (v >= 1000 and rng.random() < p("four_comma", "four_plain")):
                s = f"{v:,}"
            return s

        if typ == "million":
            v = groups[0]
            if v % 1_000_000 == 0:
                return [(str(v // 1_000_000), False), ("million", False)]
            return [(fmt_int(v), False)]
        if dec is not None:
            s = str(groups[0]) + "." + dec
        elif len(groups) >= 2:
            if (len(groups) == 2 and 1 <= groups[0] <= 12 and 0 <= groups[1] < 60
                    and groups[1] >= 10 and rng.random() < p("multi_colon", "multi_plain")):
                s = f"{groups[0]}:{groups[1]:02d}"
            else:
                s = "".join(str(g) for g in groups)
        else:
            s = fmt_int(groups[0])
        if ordinal and dec is None and len(groups) == 1:
            s = str(groups[0]) + ordinal_suffix(groups[0])
        if typ == "pct":
            return [(s + "%", False)] if rng.random() < p("pct_sign", "pct_word") else [(s, False), ("percent", False)]
        if typ == "money":
            if rng.random() < p("money_sign", "money_word"):
                return [("$" + s, rng.random() < p("money_glued", "money_spaced"))]
            return [(s, False), ("dollars", False)]
        return [(s, False)]

    # ---- main
    def simulate(self, text: str, seed: int, keep_hesitations: bool = False, intended: str | None = None,
                 dictionary: list[str] | None = None, mult: float | None = None) -> str:
        """Spoken script text -> simulated raw_asr (before the app pre-pass)."""
        rng = random.Random(seed)
        if not keep_hesitations:
            text = strip_spoken_hesitations(text)
        refs = parse_spoken(text)
        if not refs:
            return ""
        spans = number_spans(refs)
        classify(refs, self.common, proper_words(intended, dictionary))
        prop = proper_words(intended, dictionary)
        m = self.multiplier(rng, sum(1 for r in refs if not r.hes)) if mult is None else mult
        M = self.m
        out: list[dict] = []  # {"w": text, "a": anchor ref idx or None, "fixed": bool, "glue": bool}
        span_at = {s0: (s0, s1, typ) for s0, s1, typ in spans}
        skip = set()
        rs = M.get("repeats", {})
        p_col = (rs.get("collapsed", 0) + 0.5) / (rs.get("n", 0) + 1.0)
        for a, b in repeats(refs):
            if rng.random() < p_col:
                skip.update(a)
        ls = M.get("letters", {})
        p_let = (ls.get("merged", 0) + 0.5) / (ls.get("n", 0) + 1.0)
        let_at = {a: b for a, b in letter_runs(refs) if rng.random() < p_let}
        i = 0
        n = len(refs)
        burst_rate = M["burst"]["rate"]
        blens = M["burst"]["lengths"]
        idx = [t for t, x in enumerate(refs) if not x.hes]
        nb = str(len_bucket(len(idx)))
        for side in ("lead", "trail"):
            e = M["edges"][side]
            fr = e["fractions"].get(nb)
            if idx and fr and rng.random() < e["rate_by_length"].get(nb, 0.0):
                L = max(2, round(fr[rng.randrange(len(fr))] * len(idx)))
                skip.update(idx[:L] if side == "lead" else idx[::-1][:L])
        while i < n:
            r = refs[i]
            if i in skip:
                i += 1
                continue
            if i in let_at:
                b = let_at[i]
                out.append({"w": "".join(x.k for x in refs[i:b]).upper(), "a": b - 1, "fixed": True, "glue": False})
                i = b
                continue
            # dropped phrase
            if not r.hes and rng.random() < burst_rate * min(m, 3.0) / max(M["multiplier"]["mean"], 1e-6):
                L = int(weighted(blens, rng))
                i += L
                continue
            if i in span_at:
                s0, s1, typ = span_at[i]
                rend = self.render_number([x.k for x in refs[s0:s1]], typ, rng)
                if rend is not None:
                    for t_i, (tok, glue) in enumerate(rend):
                        out.append({"w": tok, "a": s1 - 1 if t_i == len(rend) - 1 else None, "fixed": True,
                                    "glue": glue})
                    i = s1
                    continue
            if r.hes:
                h = M["hes"]
                x = rng.random()
                if x < h["p_hes"]:
                    out.append({"w": weighted(h["surface"], rng), "a": i, "fixed": True, "glue": False})
                elif x < h["p_hes"] + h["p_sub"]:
                    out.append({"w": weighted(h["sub"], rng), "a": i, "fixed": False, "glue": False})
                i += 1
                continue
            # merges with the next word
            if i + 1 < n and not refs[i + 1].hes:
                pair = r.k + " " + refs[i + 1].k
                mg = M["merges"].get(pair)
                if mg:
                    tot = sum(mg["out"].values())
                    if rng.random() < tot / (mg["n"] + 1.0):
                        out.append({"w": weighted(mg["out"], rng), "a": i + 1, "fixed": False, "glue": False})
                        i += 2
                        continue
            # symbol spans: "staging dot example dot com" -> "staging.example.com"
            if (i + 2 < n and refs[i + 1].k in SYMBOLS and refs[i].after is None and refs[i + 1].after is None
                    and rng.random() < M["symbols"].get(refs[i + 1].k, 0.0)):
                j = i
                parts = [r.k]
                while j + 2 < n and refs[j + 1].k in SYMBOLS and refs[j + 1].after is None and refs[j].after is None:
                    parts += [SYMBOLS[refs[j + 1].k], refs[j + 2].k]
                    j += 2
                out.append({"w": "".join(parts), "a": j, "fixed": True, "glue": False})
                i = j + 1
                continue
            ps, pd, conv = self.word_rates(r)
            sp = M["splits"].get(r.k)
            x = rng.random()
            w = r.k
            if x < pd * m:
                pass  # deleted
            elif x < pd * m + (ps if conv else ps * m):
                s = self.substitute(r.k, rng)
                if s:
                    parts = s.split()
                    for t_i, p_ in enumerate(parts):
                        out.append({"w": p_, "a": i if t_i == len(parts) - 1 else None, "fixed": False, "glue": False})
                else:
                    out.append({"w": w, "a": i, "fixed": False, "glue": False})
            elif sp and rng.random() < sum(sp.values()) / (M["words"].get(r.k, {}).get("n", 0) + 1.0):
                parts = weighted(sp, rng).split()
                for t_i, p_ in enumerate(parts):
                    out.append({"w": p_, "a": i if t_i == len(parts) - 1 else None, "fixed": False, "glue": False})
            else:
                out.append({"w": w, "a": i, "fixed": False, "glue": False, "proper": prop.get(r.k)})
            if rng.random() < M["ins"]["rate"] * m:
                out.append({"w": weighted(M["ins"]["words"], rng), "a": None, "fixed": False, "glue": False})
            i += 1
        if rng.random() < M["end_ins"]["rate"]:
            out.append({"w": weighted(M["end_ins"]["words"], rng), "a": None, "fixed": False, "glue": False})
        return self.finish(out, refs, rng)

    def punct_probs(self, anchor: Ref, since: int, nxt: Ref) -> dict[str, float]:
        feats = feat_keys(anchor, since, nxt, self.top_next)
        tabs = self.m["punct"]
        probs = {"none": 1 / 3, "comma": 1 / 3, "stop": 1 / 3}
        for f in reversed(feats):  # coarse -> fine, each level smoothed toward its parent
            c = tabs.get(f)
            if not c:
                continue
            tot = sum(c.values())
            kk = 6.0
            probs = {b: (c.get(b, 0) + kk * probs[b]) / (tot + kk) for b in probs}
        return probs

    def finish(self, out: list[dict], refs: list[Ref], rng: random.Random) -> str:
        M = self.m
        case = M["case"]
        toks: list[str] = []
        since = 0
        sent: list[str] = []
        prev_p = "."
        n_ref = len(refs)
        for idx, o in enumerate(out):
            w = o["w"]
            k = key(w)
            # casing
            if o["fixed"] and (re.search(r"[A-Z]", w) or re.search(r"\d", w)):
                pass
            elif prev_p in STOPS:
                cap_p = case["first_cap"] if not toks else case["after_stop_cap"]
                w = self.case_word(w, k, rng, o.get("proper"))
                if rng.random() < cap_p and w[:1].islower():
                    w = w[:1].upper() + w[1:]
            else:
                w = self.case_word(w, k, rng, o.get("proper"), ctx=(prev_p or "_") + "|" + self.prev_marker(o, refs))
            if len(sent) < 3:
                sent.append(refs[o["a"]].k if o["a"] is not None else k)
            since += 1
            p = ""
            a = o["a"]
            if a is not None:
                anchor = refs[a]
                fw = fw_class(sent, self.top_next)
                if a == n_ref - 1 or idx == len(out) - 1:
                    p = self.pick_final(fw, rng)
                elif anchor.num >= 0 and refs[a + 1].num == anchor.num:
                    p = ""  # no break inside a spoken number ("four hundred and ninety nine")
                else:
                    probs = self.punct_probs(anchor, since, refs[a + 1])
                    b = weighted(probs, rng)
                    if b == "comma":
                        p = ","
                    elif b == "stop":
                        st = M["stop"].get(fw) or M["stop"].get("_other") or {".": 1}
                        p = weighted(st, rng)
            elif idx == len(out) - 1:
                p = self.pick_final(fw_class(sent, self.top_next), rng)
            if o["glue"] and toks and toks[-1][-1:].isalnum():
                toks[-1] += w + p
            else:
                toks.append(w + p)
            if p in (".", "?", "!"):
                since, sent = 0, []
            prev_p = p
        return " ".join(toks)

    def pick_final(self, fw: str, rng: random.Random) -> str:
        fin = self.m["final"].get(fw) or self.m["final"].get("_other") or {".": 1}
        p = weighted(fin, rng)
        return "" if p == "none" else p

    def prev_marker(self, o: dict, refs: list[Ref]) -> str:
        a = o["a"]
        if a is None or a == 0:
            return "none"
        return marker(refs[a - 1])

    def case_word(self, w: str, k: str, rng: random.Random, proper: str | None, ctx: str | None = None) -> str:
        case = self.m["case"]
        if not k or re.search(r"\d", w):
            return w
        tab = case["words"].get(k)
        if tab:
            # one lowercase pseudo-count: a word seen capitalized once is not always capitalized
            cs = weighted({**tab, "lower": tab.get("lower", 0) + 1.0}, rng)
        elif proper and rng.random() < case["proper_case_match"]:
            return proper if key(proper) == k else w
        else:
            cs = "lower"
            if ctx is not None:
                cap, tot = case["spurious"].get(ctx, case["spurious"].get("_|none", [0, 1]))
                if rng.random() < (cap + 0.5) / (tot + 1.0):
                    cs = "cap"
        if cs == "upper":
            return w.upper()
        if cs == "cap":
            return w[:1].upper() + w[1:]
        return w

    def raw(self, text: str, seed: int, dictionary: list[str] | None = None, **kw) -> tuple[str, str]:
        asr = self.simulate(text, seed, dictionary=dictionary, **kw)
        return asr, app_prepass(asr, dictionary or [])


def weighted(d: dict, rng: random.Random):
    tot = sum(d.values())
    if tot <= 0:
        return next(iter(d))
    x = rng.random() * tot
    for k, v in d.items():
        x -= v
        if x <= 0:
            return k
    return k


_SIM: Sim | None = None


def load_sim() -> Sim:
    global _SIM
    if _SIM is None:
        _SIM = Sim(json.loads(MODEL.read_text(encoding="utf-8-sig")))
    return _SIM


def apply(text: str, seed: int, dictionary: list[str] | None = None, intended: str | None = None,
          keep_hesitations: bool = False) -> str:
    """Spoken script text -> simulated `raw` (after the app pre-pass). Deterministic given seed."""
    return load_sim().raw(text, seed, dictionary=dictionary, intended=intended,
                          keep_hesitations=keep_hesitations)[1]


# ============================================================================ validation


def wer_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    n, m = len(ref), len(hyp)
    D = list(range(m + 1))
    Bk = [[(0, 0, j)] for j in range(m + 1)]  # not tracked per cell to stay light: recompute below
    # standard DP with op counts
    prev = [(j, 0, 0, j) for j in range(m + 1)]  # (cost, sub, del, ins)
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            s = prev[j - 1]
            if ref[i - 1] == hyp[j - 1]:
                c1 = s
            else:
                c1 = (s[0] + 1, s[1] + 1, s[2], s[3])
            d = prev[j]
            c2 = (d[0] + 1, d[1], d[2] + 1, d[3])
            ii = cur[j - 1]
            c3 = (ii[0] + 1, ii[1], ii[2], ii[3] + 1)
            cur.append(min(c1, c2, c3))
        prev = cur
    _, s, d, ins = prev[m]
    return s, d, ins


def ref_words(spoken: str) -> list[str]:
    return [r.k for r in parse_spoken(spoken) if not r.hes]


def hyp_words(raw: str) -> list[str]:
    return [k for k in (key(t) for t in raw.split()) if k and not is_hes(k)]


CLAUSE_STARTERS = set("and but so because then also or if when which that i we you it please".split())


def row_stats(spoken_ref: str, tts_text: str, raw: str) -> dict:
    rw, hw = ref_words(spoken_ref), hyp_words(raw)
    s, d, i = wer_ops(rw, hw)
    toks = hyp_tokens(raw)
    puncts = [punct_of(t) for t in toks]
    commas = sum(p == "," for p in puncts)
    periods = sum(p in (".", "!") for p in puncts)
    qs = sum(p == "?" for p in puncts)
    breaks = sum(p in STOPS for p in puncts[:-1])
    caps = midcaps = 0
    for j in range(1, len(toks)):
        c = core(toks[j])
        if not c or not c[0].isalpha() or key(c) in ("i",) or key(c).startswith("i'"):
            continue
        if puncts[j - 1] in STOPS:
            continue
        midcaps += 1
        caps += c[0].isupper()
    # mid-clause breaks: sentence break in raw where the spoken text has no marker and the next spoken
    # word does not start a clause (via alignment of tts tokens to raw tokens)
    refs = parse_spoken(tts_text)
    spans = number_spans(refs)
    ops = align(refs, toks)
    mid = 0
    for t, r0, r1, h0, h1 in ops:
        if t in ("ins", "skip", "del") or h1 - 1 >= len(toks) - 1:
            continue
        if punct_of(toks[h1 - 1]) in STOPS:
            a = refs[r1 - 1]
            if a.after is None and r1 < len(refs) and refs[r1].k not in CLAUSE_STARTERS and not refs[r1].hes:
                mid += 1
    nd = nw = 0
    for s0, s1, typ in spans:
        span_ops = [op for op in ops if op[0] not in ("ins", "skip") and op[1] < s1 and op[2] > s0]
        if any(op[0] == "num" for op in span_ops):
            nd += 1
        elif span_ops and all(op[0] == "match" for op in span_ops):
            nw += 1
    return {"n": len(rw), "sub": s, "del": d, "ins": i, "wer": (s + d + i) / max(1, len(rw)),
            "commas": commas, "periods": periods, "qs": qs, "breaks": breaks, "mid": mid,
            "hw": len(hw), "caps": caps, "midcaps": midcaps, "num_digits": nd, "num_words": nw}


def aggregate(stats: list[dict]) -> dict:
    W = sum(s["n"] for s in stats)
    H = sum(s["hw"] for s in stats) or 1
    wers = sorted(s["wer"] for s in stats)

    def q(p):
        return wers[min(len(wers) - 1, int(p * len(wers)))]
    nd = sum(s["num_digits"] for s in stats)
    nw = sum(s["num_words"] for s in stats)
    return {
        "rows": len(stats),
        "wer_mean": sum(wers) / len(wers),
        "wer_q25": q(0.25), "wer_q50": q(0.5), "wer_q75": q(0.75), "wer_q90": q(0.9),
        "wer_corpus": sum(s["sub"] + s["del"] + s["ins"] for s in stats) / W,
        "share_wer0": sum(1 for w in wers if w == 0) / len(wers),
        "sub_rate": sum(s["sub"] for s in stats) / W,
        "del_rate": sum(s["del"] for s in stats) / W,
        "ins_rate": sum(s["ins"] for s in stats) / W,
        "commas_per100": 100 * sum(s["commas"] for s in stats) / H,
        "periods_per100": 100 * sum(s["periods"] for s in stats) / H,
        "q_per100": 100 * sum(s["qs"] for s in stats) / H,
        "breaks_per100": 100 * sum(s["breaks"] for s in stats) / H,
        "midclause_breaks_per100": 100 * sum(s["mid"] for s in stats) / H,
        "midsentence_cap_rate": sum(s["caps"] for s in stats) / max(1, sum(s["midcaps"] for s in stats)),
        "number_digit_rate": nd / max(1, nd + nw),
    }


def has_hes(text: str) -> bool:
    return any(r.hes for r in parse_spoken(text))


def validate(rows: list[dict], seeds=(1, 2, 3, 4, 5)) -> dict:
    sim_ = load_sim()
    held = [r for r in rows if is_heldout(r["id"])]
    result = {}
    fit_sample = [r for r in rows if not is_heldout(r["id"])
                  and int(hashlib.sha256(f"fit-sample|{r['id']}".encode()).hexdigest()[:4], 16) % 100 < 18]
    for subset, sel in (("all", held), ("no_hes_tts", [r for r in held if not has_hes(r["tts_text"])]),
                        ("fit_sample_in_sample", fit_sample)):
        if not sel:
            continue
        real = [row_stats(r["spoken"], r["tts_text"], r["raw"]) for r in sel]
        simd = []
        for seed in seeds:
            for r in sel:
                kh = has_hes(r["tts_text"])
                raw = sim_.raw(r["tts_text"], seed * 1_000_003 + int(hashlib.md5(r["id"].encode()).hexdigest()[:6], 16),
                               dictionary=r.get("dictionary"), intended=r.get("intended"), keep_hesitations=kh)[1]
                simd.append(row_stats(r["spoken"], r["tts_text"], raw))
        a, b = aggregate(real), aggregate(simd)
        result[subset] = {"real": a, "sim": b,
                          "rel_diff": {k: (b[k] - a[k]) / a[k] if a[k] else None for k in a if k != "rows"}}
    return result


def print_table(res: dict) -> None:
    for subset, r in res.items():
        print(f"\n== held-out subset: {subset} (real rows {r['real']['rows']}, sim rows {r['sim']['rows']})")
        print(f"{'statistic':28s} {'real':>9s} {'sim':>9s} {'rel':>7s}")
        for k, v in r["real"].items():
            if k == "rows":
                continue
            s = r["sim"][k]
            rel = r["rel_diff"][k]
            print(f"{k:28s} {v:9.4f} {s:9.4f} {(f'{rel:+.0%}' if rel is not None else '-'):>7s}")


# ============================================================================ outputs


def audited_ok(row: dict) -> bool:
    a = row.get("audit") or {}
    if not ("claude" in a or "adjudicator" in a):
        return False
    for k in ("claude", "adjudicator"):
        if k in a and (a[k] or {}).get("decision") == "drop":
            return False
    return True


def make_validation_set(rows: list[dict], cap: int = 400) -> int:
    sim_ = load_sim()
    held = [r for r in rows if is_heldout(r["id"]) and audited_ok(r) and r.get("clean")]
    held.sort(key=lambda r: r["id"])
    if not FRESH:
        held = held[:cap]
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pair_map = {}
    if FRESH:   # spread the cap over all fresh batches instead of the first ids
        held.sort(key=lambda r: hashlib.sha256(f"fresh|{r['id']}".encode()).hexdigest())
        held = held[:cap]
    with PAIRS.open("w", encoding="utf-8", newline="\n") as f:
        for n, r in enumerate(held, 1):
            # The real side passed the audit (unusable clips were dropped), so a simulated clip that
            # came out empty (whole-clip loss) is re-drawn with the next seed.
            for attempt in range(5):
                seed = int(hashlib.sha256(f"simval|{r['id']}|{attempt}".encode()).hexdigest()[:8], 16)
                asr, raw = sim_.raw(r["tts_text"], seed, dictionary=r.get("dictionary"),
                                    intended=r.get("intended"), keep_hesitations=has_hes(r["tts_text"]))
                if raw.strip():
                    break
            base = {"mode": "clean", "style": r.get("style", ""), "dictionary": r.get("dictionary", []),
                    "context": r.get("context", ""), "clean": r["clean"], "tags": r.get("tags", [])}
            f.write(json.dumps({"id": f"simval-real-{n}", "source": "real-v2", **base, "raw": r["raw"]},
                               ensure_ascii=False) + "\n")
            f.write(json.dumps({"id": f"simval-sim-{n}", "source": "sim-v4", **base, "raw": raw},
                               ensure_ascii=False) + "\n")
            pair_map[str(n)] = {"orig_id": r["id"], "tts_text": r["tts_text"], "raw_asr_real": r["raw_asr"],
                                "raw_asr_sim": asr, "hesitations_spoken": has_hes(r["tts_text"])}
    PAIRS_MAP.write_text(json.dumps(pair_map, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(held)


def apply_file(inp: Path, outp: Path, seed: int, keep_hes: bool) -> int:
    sim_ = load_sim()
    rows = [json.loads(l) for l in inp.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            s = int(hashlib.sha256(f"sim-v4|{r.get('id')}|{seed}".encode()).hexdigest()[:8], 16)
            asr, raw = sim_.raw(r["spoken"], s, dictionary=r.get("dictionary"), intended=r.get("intended"),
                                keep_hesitations=keep_hes)
            o = dict(r)
            o.update({"raw": raw, "raw_asr_sim": asr, "clean": r.get("intended", r.get("clean")),
                      "source": "sim-v4", "sim_seed": seed})
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "validate", "make-validation-set", "apply"])
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-hesitations", action="store_true",
                    help="simulate a TTS that speaks um/uh (the current TTS strips them)")
    ap.add_argument("--fresh", action="store_true",
                    help="validate / make-validation-set on W12-W51 (out of the fit and of v3's training)")
    ap.add_argument("--all-batches", action="store_true", help="fit/validate on every W*/R2 batch, not W1-W11+R2")
    a = ap.parse_args()
    global BATCHES, FRESH, PAIRS, PAIRS_MAP, VALIDATION
    if a.all_batches:
        BATCHES = ALL_BATCHES
    if a.fresh:
        if a.cmd in ("fit", "apply"):
            ap.error("--fresh is for validate / make-validation-set")
        BATCHES, FRESH = FRESH_BATCHES, True
        PAIRS = PAIRS.with_name("fresh-pairs.jsonl")
        PAIRS_MAP = SIM_DIR / "fresh_pairs_map.json"
        VALIDATION = SIM_DIR / "validation-fresh.json"
    if a.cmd == "fit":
        model = fit(load_real())
        MODEL.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {MODEL} ({MODEL.stat().st_size // 1024} KB)")
    elif a.cmd == "validate":
        res = validate(load_real())
        print_table(res)
        VALIDATION.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {VALIDATION}")
    elif a.cmd == "make-validation-set":
        n = make_validation_set(load_real())
        print(f"wrote {2 * n} rows ({n} held-out rows) to {PAIRS}")
    elif a.cmd == "apply":
        if not a.inp or not a.out:
            ap.error("apply needs --in and --out")
        n = apply_file(Path(a.inp), Path(a.out), a.seed, a.keep_hesitations)
        print(f"wrote {n} rows to {a.out}")


if __name__ == "__main__":
    main()

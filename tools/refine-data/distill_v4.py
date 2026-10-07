"""Round-4 text-only distillation pairs (`source: sim-v4`): scripts -> ASR-noise simulator -> raw.

    python tools/refine-data/distill_v4.py build [--seeds 2] [--out training/refine/datasets/distill-v4/sim-v4.jsonl]
    python tools/refine-data/distill_v4.py filter --agree <eval output jsonl> ...   # after v3-4b ran on them
    python tools/refine-data/distill_v4.py sample --n 1000                           # rows for the 2-auditor audit

Sources: the round-4 distillation scripts (`distill-v4/scripts/D*.jsonl` by Claude writers,
`C*.jsonl` by Codex writers with `writer: codex/<model>`; never rendered) and the
round-4 real-audio scripts W12-W64 (simulated again; their audited real-audio rows stay the real
data). W rows whose real-audio audit found a script problem (`guide_violation`, `script_defect`)
are skipped: their `intended` is wrong.

**Targets must stay recoverable from the simulated raw (GUIDE rule 13).** The simulator makes
real recognition errors, so every simulated sample is aligned word by word against what was said:
* casing, punctuation, sentence breaks, number rendering, merges/splits: target unchanged;
* a substitution by a near spelling (similarity >= 0.6) of a common word: inferable, target
  unchanged; of a capitalized name not in the dictionary: the target takes raw's spelling;
* any other substitution: the sample is rejected (a first version propagated raw's word into the
  target, which produced targets nobody would write; lexical-error handling is left to the
  audited real-audio data) and another seed is tried;
* a dropped function word: removed from the target; a dropped content word, an inserted content
  word, or a changed digit string: the sample is rejected (and another seed is tried).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
DS = ROOT / "training" / "refine" / "datasets"
import asr_noise_sim as sim  # noqa: E402
from common import read_jsonl, strip_spoken_hesitations  # noqa: E402

FUNC = set("""a an the and or but so to of in on at for with from by as is are was were be been it its this that
these those i you he she we they me him her us them my your his our their do does did not no yes if then than
just up out about into over off there here what when where who how which will would can could should shall
may might must have has had am im i'm it's that's don't can't won't""".split())
NUMW = set("""zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion
first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth
sixteenth seventeenth eighteenth nineteenth twentieth thirtieth point percent dollars dollar cents cent""".split())


def words(s: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9']+", s.lower().replace("’", "'"))]


def spoken_words(spoken: str) -> list[str]:
    return [w for w in words(strip_spoken_hesitations(spoken).replace("-", " "))]


def align(a: list[str], b: list[str]) -> list[tuple[str, str | None, str | None]]:
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            ops.append(("eq" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    return ops[::-1]


def similar(a: str, b: str) -> float:
    return sim.sim(a, b)


def _uk_us(w: str) -> str:
    for a, b in (("ogue", "og"), ("our", "or"), ("ise", "ize"), ("ising", "izing"), ("ised", "ized"), ("yse", "yze"),
                 ("mme", "m"), ("ll", "l"), ("tre", "ter")):
        w = w.replace(a, b)
    return w


def variant(x: str, y: str) -> bool:
    """Spelling variant or inflection of the same word (not a different word)."""
    if min(len(x), len(y)) < 3 or x == y:
        return False
    if _uk_us(x) == _uk_us(y):
        return True
    short, long_ = sorted((x, y), key=len)
    return long_.startswith(short) and len(long_) - len(short) <= 3 and len(short) >= 4 or \
        (x.rstrip("s") == y.rstrip("s") and len(x.rstrip("s")) >= 3)


def numberish(w: str | None) -> bool:
    return bool(w) and (w in NUMW or bool(re.search(r"\d", w)))


def replace_once(text: str, old: str, new: str) -> str | None:
    pat = re.compile(r"(?<![A-Za-z0-9'])" + re.escape(old) + r"(?![A-Za-z0-9'])", re.I)
    hits = list(pat.finditer(text))
    if len(hits) != 1:
        return None
    h = hits[0]
    src = h.group(0)
    rep = new
    if src[:1].isupper():
        rep = new[:1].upper() + new[1:]
    if src.isupper() and len(src) > 1:
        rep = new.upper()
    out = text[:h.start()] + rep + text[h.end():]
    return re.sub(r"  +", " ", out)


def patch_target(spoken: str, raw: str, intended: str, dictionary: list[str]) -> tuple[str | None, dict]:
    a, b = spoken_words(spoken), words(raw)
    ops = align(a, b)
    info = {"sub": 0, "del": 0, "ins": 0, "kept": 0, "propagated": 0}
    target = intended
    dict_l = {d.lower() for d in dictionary or []}
    # merged / split words show up as sub+ins/del pairs; collapse "stand up" <-> "standup"
    for k, (op, x, y) in enumerate(ops):
        if op == "eq":
            continue
        if numberish(x) or numberish(y):
            continue   # number rendering; values are checked separately below
        if op == "sub":
            info["sub"] += 1
            nxt = ops[k + 1] if k + 1 < len(ops) else None
            prv = ops[k - 1] if k else None
            if (nxt and nxt[0] == "del" and (x + nxt[1]) == y) or (nxt and nxt[0] == "ins" and x == y + nxt[2]) \
                    or (prv and prv[0] == "ins" and x == prv[2] + y) or (prv and prv[0] == "del" and prv[1] + x == y):
                info["kept"] += 1
                continue
            name = re.search(r"(?<![A-Za-z])" + re.escape(x) + r"(?![A-Za-z])", target, re.I)
            is_name = bool(name) and name.group(0)[:1].isupper() and x not in FUNC and \
                not target.lstrip().lower().startswith(x) and x not in dict_l
            if variant(x, y) and not is_name:
                # a spelling variant (colour/color) or inflection (total/totals, short/shorter) in raw is
                # not something the model can undo (GUIDE rules 10, 13): the target takes raw's word
                new = replace_once(target, x, y)
                if new is None:
                    return None, info
                target = new
                info["propagated"] += 1
                continue
            if similar(x, y) >= 0.6 and not is_name:
                info["kept"] += 1          # inferable near spelling: the model should fix it
                continue
            if not (is_name and similar(x, y) >= 0.5):
                return None, info      # a real lexical error: the audited real-audio data teaches these
            new = replace_once(target, x, y)
            if new is None:
                return None, info
            target = new
            info["propagated"] += 1
        elif op == "del":
            info["del"] += 1
            if k + 1 < len(ops) and ops[k + 1][0] == "sub" and (x + ops[k + 1][1]) == ops[k + 1][2]:
                continue
            if k and ops[k - 1][0] == "sub" and (ops[k - 1][1] + x) == ops[k - 1][2]:
                continue
            if x in FUNC:
                new = replace_once(target, x, "")
                if new is None:
                    return None, info
                target = re.sub(r"\s+([,.?!])", r"\1", new).strip()
            else:
                return None, info
        else:
            info["ins"] += 1
            if y not in FUNC:
                return None, info
    # number values: every digit string in raw must appear in the target
    for dgt in re.findall(r"\d+", raw.replace(",", "")):
        if dgt not in re.sub(r"[,]", "", target):
            return None, info
    if target and target[0].islower() and intended[:1].isupper():
        target = target[0].upper() + target[1:]
    return target, info


def load_sources() -> list[dict]:
    rows = []
    # A few Codex jobs wrote through a non-UTF-8 console: line breaks, "°" and "£" became "?".
    mangled = re.compile(r"\?\?|\w\?\w|\d\?[A-Z]|\?\d")
    for f in sorted((DS / "distill-v4" / "scripts").glob("[DC]*.jsonl")):   # D = Claude, C = Codex
        rows += [dict(r, _src=f.stem) for r in read_jsonl(f) if not mangled.search(r["intended"])]
    bad = set()
    for f in (DS / "real-v2").glob("W*.jsonl"):
        for r in read_jsonl(f):
            au = r.get("audit", {})
            reasons = []
            for k in ("claude", "adjudicator"):
                v = au.get(k) or {}
                reasons += v.get("reasons") or [v.get("reason")]
            if any(x in ("guide_violation", "script_defect") for x in reasons if x):
                bad.add(r["id"])
    for n in range(12, 65):
        f = DS / "scripts-v2" / f"W{n}.jsonl"
        if f.exists():
            rows += [dict(r, _src=f.stem) for r in read_jsonl(f) if r["id"] not in bad]
    return rows


def cmd_build(a: argparse.Namespace) -> None:
    src = load_sources()
    S = sim.load_sim()
    out, stats = [], {"scripts": len(src), "samples": 0, "rejected": 0, "propagated_rows": 0}
    seen = set()   # the same script written twice (across writers / files): keep the first
    stats["duplicate_scripts"] = 0
    for r in src:
        key = " ".join(words(r["spoken"]))
        if key in seen:
            stats["duplicate_scripts"] += 1
            continue
        seen.add(key)
        if "\\n" in r["intended"]:   # a few Claude-written scripts typed the escape instead of a line break
            r = dict(r, intended=r["intended"].replace("\\n", "\n"))
            stats["literal_newline_fixed"] = stats.get("literal_newline_fixed", 0) + 1
        got = 0
        for t in range(a.seeds * 4):
            if got >= a.seeds:
                break
            seed = int(hashlib.sha256(f"sim-v4|{r['id']}|{t}".encode()).hexdigest()[:8], 16)
            asr, raw = S.raw(r["spoken"], seed, dictionary=r.get("dictionary"), intended=r.get("intended"))
            if not raw.strip():
                continue
            tgt, info = patch_target(r["spoken"], raw, r["intended"], r.get("dictionary") or [])
            if tgt is None:
                stats["rejected"] += 1
                continue
            got += 1
            stats["samples"] += 1
            stats["propagated_rows"] += bool(info["propagated"])
            out.append({"id": f"sim4-{r['id']}-{t}", "source": "sim-v4", "mode": "clean", "style": r.get("style") or "",
                        "dictionary": r.get("dictionary") or [], "context": r.get("context") or "notes",
                        "raw": raw, "clean": tgt, "tags": r.get("tags") or [], "cat": r.get("cat") or "",
                        "script_id": r["id"], "script_src": r["_src"], "sim_seed": seed,
                        "target_patched": tgt != r["intended"],
                        "writer": r.get("writer") or ("claude/sonnet" if r["_src"].startswith(("D", "W")) else "")})
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print(json.dumps(stats), "->", dest)


def cmd_sample(a: argparse.Namespace) -> None:
    rows = read_jsonl(Path(a.inp))
    rng = random.Random(4)
    pick = rng.sample(rows, min(a.n, len(rows)))
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pick), encoding="utf-8")
    print(len(pick), "->", a.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seeds", type=int, default=2)
    b.add_argument("--out", default=str(DS / "distill-v4" / "sim-v4.jsonl"))
    s = sub.add_parser("sample")
    s.add_argument("--in", dest="inp", default=str(DS / "distill-v4" / "sim-v4.jsonl"))
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--out", default=str(DS / "distill-v4" / "audit-sample.jsonl"))
    a = ap.parse_args()
    {"build": cmd_build, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Near-duplicate leakage check between training rows/scripts and held-out eval rows (round 4).

    python tools/refine-data/leakcheck.py --train <jsonl/glob> ... --eval <jsonl/glob> ... [--out report.json]
           [--threshold 0.5] [--field spoken|raw|clean|intended ...]

For every train row and every eval row it compares normalized text (lowercase, letters/digits only,
hesitations removed) of the given fields (default: raw, clean, spoken, intended, whichever exist):
* exact match of any normalized field -> leak;
* word 4-gram containment (|A & B| / min(|A|, |B|)) >= threshold on any field pair -> near-dup
  (only when both sides have >= 6 four-grams, i.e. >= 9 words; short commands share phrases naturally);
* rare-name overlap is reported separately (capitalized tokens of clean/intended that occur in
  both sets and in fewer than 3 eval rows), as a signal that writers reused the reserved pools.
Prints a summary; --out writes the flagged pairs; --drop-list writes train ids to drop.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

HES = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "hm", "mm", "mhm"}


def rows(patterns: list[str]) -> list[dict]:
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for x in Path(f).read_text(encoding="utf-8-sig").splitlines():
                if x.strip():
                    try:
                        r = json.loads(x)
                    except json.JSONDecodeError:
                        continue
                    if r.get("kind") in ("header",):
                        continue
                    r["_file"] = Path(f).name
                    out.append(r)
    return out


def norm(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in HES]


def grams(ws: list[str], n: int = 4) -> set:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)} if len(ws) >= n else {tuple(ws)} if ws else set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--field", nargs="*", default=["raw", "clean", "spoken", "intended"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--drop-list", type=Path)
    a = ap.parse_args()
    tr, ev = rows(a.train), rows(a.eval)
    idx: dict[tuple, set] = collections.defaultdict(set)
    ev_exact: dict[str, str] = {}
    ev_grams = []
    for j, r in enumerate(ev):
        gs = set()
        for f in a.field:
            if isinstance(r.get(f), str):
                w = norm(r[f])
                if len(w) >= 3:
                    ev_exact[" ".join(w)] = r["id"]
                g = grams(w)
                gs |= g
        ev_grams.append(gs)
        for g in gs:
            idx[g].add(j)
    flagged, drop = [], set()
    for r in tr:
        best, best_j = 0.0, None
        exact = None
        for f in a.field:
            if not isinstance(r.get(f), str):
                continue
            w = norm(r[f])
            k = " ".join(w)
            if len(w) >= 3 and k in ev_exact:
                exact = ev_exact[k]
            g = grams(w)
            cand = collections.Counter(j for x in g for j in idx.get(x, ()))
            for j, c in cand.most_common(5):
                if min(len(g), len(ev_grams[j])) < 6:   # short rows: only exact matches count
                    continue
                score = c / max(1, min(len(g), len(ev_grams[j])))
                if score > best:
                    best, best_j = score, j
        if exact or best >= a.threshold:
            flagged.append({"train_id": r["id"], "train_file": r["_file"], "eval_id": exact or ev[best_j]["id"],
                            "kind": "exact" if exact else "near", "score": round(best, 3),
                            "train_text": (r.get("clean") or r.get("intended") or "")[:200],
                            "eval_text": ((ev[best_j].get("clean") or ev[best_j].get("intended") or "") if best_j is not None else "")[:200]})
            drop.add(r["id"])
    ev_names = collections.Counter(n for r in ev for n in set(re.findall(r"\b[A-Z][a-z]{3,}\b", r.get("clean") or r.get("intended") or "")))
    tr_names = collections.Counter(n for r in tr for n in set(re.findall(r"\b[A-Z][a-z]{3,}\b", r.get("clean") or r.get("intended") or "")))
    rare_shared = sorted(n for n, c in ev_names.items() if c < 3 and n in tr_names and tr_names[n] < 5)
    print(json.dumps({"train_rows": len(tr), "eval_rows": len(ev), "flagged": len(flagged),
                      "exact": sum(f["kind"] == "exact" for f in flagged),
                      "near": sum(f["kind"] == "near" for f in flagged),
                      "rare_capitalized_tokens_shared": len(rare_shared)}, indent=1))
    for f in flagged[:15]:
        print(f"  {f['kind']} {f['score']} {f['train_id']} ~ {f['eval_id']}\n    T: {f['train_text'][:120]}\n    E: {f['eval_text'][:120]}")
    if a.out:
        a.out.write_text(json.dumps({"flagged": flagged, "rare_shared": rare_shared}, indent=1), encoding="utf-8")
    if a.drop_list:
        a.drop_list.write_text("\n".join(sorted(drop)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

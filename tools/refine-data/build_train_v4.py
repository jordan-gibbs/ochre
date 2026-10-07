"""Assemble the round-4 training sets (training/refine/datasets/real-v4/).

    python tools/refine-data/merge_audits.py apply && python tools/refine-data/merge_audits.py export
    python tools/refine-data/build_train_v4.py [--sim training/refine/datasets/distill-v4/sim-v4-filtered.jsonl]

Writes:
* real-v4/train-real.jsonl: round-3 train (real-v3/final/train.jsonl, 2,486) + round-4 audited
  real-audio rows (ids s4-*, from real-v2/final/real-v2.jsonl), minus anything near-duplicating
  an eval row (eval-v4, round-3 held-out / new / casual; tools/refine-data/leakcheck.py rules);
* real-v4/train-real-x2.jsonl: the same, every row twice (real upweighting for the distill mix);
* real-v4/sim.jsonl: the filtered distillation rows (if --sim), same leak filter;
* real-v4/README.md: counts by source and category.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "training" / "refine" / "datasets"
OUT = DS / "real-v4"
EVALS = [DS / "eval-v4" / "eval.jsonl", DS / "real-v3" / "final" / "heldout.jsonl",
         DS / "real-v3" / "final" / "heldout-new.jsonl", DS / "real-v3" / "final" / "casual-probe.jsonl"]
KEEP = ["id", "source", "mode", "style", "dictionary", "context", "raw", "clean", "tags", "raw_engine", "voice"]


def read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def leak_ids(train: Path) -> set[str]:
    drop = OUT / "_leak_drop.txt"
    subprocess.run([sys.executable, str(ROOT / "tools/refine-data/leakcheck.py"), "--train", str(train),
                    "--eval", *map(str, EVALS), "--threshold", "0.6", "--drop-list", str(drop),
                    "--out", str(OUT / f"_leak_{train.stem}.json")], check=True)
    return {x.strip() for x in drop.read_text(encoding="utf-8-sig").splitlines() if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = {}
    for f in (DS / "scripts-v2").glob("W*.jsonl"):
        for r in read(f):
            cats[r["id"]] = r.get("cat") or ""
    v3 = read(DS / "real-v3" / "final" / "train.jsonl")
    r4 = [r for r in read(DS / "real-v2" / "final" / "real-v2.jsonl") if r["id"].startswith("s4-")]
    for r in v3:
        r["round"] = 3
    for r in r4:
        r["source"], r["round"], r["cat"] = "real-v4", 4, cats.get(r["id"], "")
    rows = v3 + r4
    tmp = OUT / "_candidates.jsonl"
    write(tmp, rows)
    bad = leak_ids(tmp)
    rows = [r for r in rows if r["id"] not in bad]
    write(OUT / "train-real.jsonl", rows)
    write(OUT / "train-real-x2.jsonl", rows + rows)
    lines = ["# real-v4 (round-4 training data)", "",
             f"- round-3 audited real-audio rows: {sum(r['round'] == 3 for r in rows)}",
             f"- round-4 audited real-audio rows: {sum(r['round'] == 4 for r in rows)}",
             f"- removed as near-duplicates of an eval row: {len(bad)}", "", "Round-4 rows by category:", ""]
    c = collections.Counter(r.get("cat") for r in rows if r["round"] == 4)
    lines += [f"- {k or '?'}: {v}" for k, v in c.most_common()]
    if a.sim:
        sim = read(Path(a.sim))
        write(tmp, sim)
        bad_s = leak_ids(tmp)
        sim = [r for r in sim if r["id"] not in bad_s]
        write(OUT / "sim.jsonl", sim)
        cs = collections.Counter(r.get("cat") for r in sim)
        w = collections.Counter(r.get("writer") or "?" for r in sim)
        lines += ["", f"sim-v4 distillation rows: {len(sim)} (removed {len(bad_s)} near-duplicates); writers {dict(w)}",
                  ""] + [f"- {k or '?'}: {v}" for k, v in cs.most_common()]
    tmp.unlink(missing_ok=True)
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

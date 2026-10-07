"""Build blind, side-by-side judge packets for the v3 sweep.

Each packet row shows one dictation (raw + reference) and every model's output under a shuffled
letter, so one judge grades all candidates for a row consistently. The letter -> run mapping
goes to KEY.json, which judges never see.

    .venv/Scripts/python tools/eval/build_judge_v3.py v2-ref v3-2b v3-2b-e3 ...
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools" / "eval" / "out" / "judge-v3"
SETS = ["heldout", "new", "casual"]
CHUNK = 20


def load(run: str, s: str) -> dict[str, dict]:
    p = ROOT / "training" / "refine" / "output" / run / "eval" / f"refine-{run}-{s}.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def main() -> None:
    runs = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261005)
    letters = "ABCDEFGH"[: len(runs)]
    packets, key = [], {}
    for s in SETS:
        per = {run: load(run, s) for run in runs}
        ids = sorted(set.intersection(*(set(d) for d in per.values())))
        for i in ids:
            ref = per[runs[0]][i]
            order = runs[:]
            rng.shuffle(order)
            k = f"{s}:{i}"
            key[k] = dict(zip(letters, order))
            packets.append({
                "key": k, "context": ref["context"], "style": ref["style"], "tags": ref.get("tags", []),
                "raw": ref["raw"], "expected": ref["clean"],
                "candidates": {L: per[run][i]["output"] for L, run in zip(letters, order)},
            })
    rng.shuffle(packets)
    names = []
    for n in range(0, len(packets), CHUNK):
        name = f"chunk-{n // CHUNK + 1:02d}"
        names.append(name)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as o:
            for r in packets[n : n + CHUNK]:
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "KEY.json").write_text(json.dumps({"runs": runs, "rows": key}, indent=1), encoding="utf-8")
    print(json.dumps({"rows": len(packets), "chunks": names}))


if __name__ == "__main__":
    main()

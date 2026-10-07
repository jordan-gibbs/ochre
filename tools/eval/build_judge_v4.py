"""Blind side-by-side judge packets, protocol v4 (tools/eval/JUDGE_V4.md): 3 independent judges.

Each packet row shows one dictation (raw + reference) and every distinct model output under a
letter. Identical outputs are merged under one letter (so they always get one verdict); letters
are shuffled independently per row **and per judge**, so position bias does not line up across
judges. The letter -> runs mapping is written to KEY.json, which judges never see.

    python tools/eval/build_judge_v4.py --manifest tools/eval/out/judge-v4/<name>/manifest.json

manifest.json:
    {"name": "baseline", "judges": 3, "chunk": 15, "seed": 4,
     "sets": {"eval4": "training/refine/datasets/eval-v4/eval.jsonl", ...},     # reference rows
     "outputs": {"v3-4b": {"eval4": "tools/eval/out/eval-runs/<run>/refine-v3-4b-eval4.jsonl"}, ...}}

Rows are keyed "<set>:<id>". Only rows every run has an output for are packed. Category per row:
the reference row's `cat` (eval-v4) or the set name (legacy sets).

Optional manifest keys: "field" (the output field graded, default "output"; "typed" grades what
the app would insert, guard fallback included, as docs/refine-chunking.md does) and "keys" (a
file of row keys, one per line, to pack only those rows).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    a = ap.parse_args()
    m = json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    out = a.manifest.parent
    judges, chunk = int(m.get("judges", 3)), int(m.get("chunk", 15))
    rng = random.Random(int(m.get("seed", 4)))
    runs = list(m["outputs"])
    refs: dict[str, dict] = {}
    for s, p in m["sets"].items():
        for r in read_jsonl(ROOT / p):
            refs[f"{s}:{r['id']}"] = {**r, "_set": s}
    field = m.get("field", "output")
    outs: dict[str, dict[str, str]] = {run: {} for run in runs}
    for run, per in m["outputs"].items():
        for s, p in per.items():
            for r in read_jsonl(ROOT / p):
                outs[run][f"{s}:{r['id']}"] = r[field]
    only = set((ROOT / m["keys"]).read_text(encoding="utf-8").split()) if m.get("keys") else None
    keys = sorted(k for k in refs if all(k in outs[run] for run in runs) and (only is None or k in only))
    missing = {run: sum(1 for k in refs if k not in outs[run]) for run in runs}
    rng.shuffle(keys)
    key_map: dict[str, dict] = {}
    packets: dict[int, list[dict]] = {j: [] for j in range(1, judges + 1)}
    for k in keys:
        ref = refs[k]
        groups: dict[str, list[str]] = {}
        for run in runs:
            groups.setdefault(outs[run][k], []).append(run)
        texts = list(groups)
        key_map[k] = {"cat": ref.get("cat") or ref["_set"], "set": ref["_set"], "judges": {}}
        for j in range(1, judges + 1):
            order = texts[:]
            rng.shuffle(order)
            letters = [chr(65 + i) for i in range(len(order))]
            key_map[k]["judges"][str(j)] = {L: groups[t] for L, t in zip(letters, order)}
            packets[j].append({"key": k, "context": ref.get("context", ""), "style": ref.get("style", ""),
                               "dictionary": ref.get("dictionary") or [], "raw": ref["raw"], "expected": ref["clean"],
                               "candidates": dict(zip(letters, order))})
    names = []
    (out / "packets").mkdir(parents=True, exist_ok=True)
    (out / "verdicts").mkdir(parents=True, exist_ok=True)
    for j in range(1, judges + 1):
        rows = packets[j]
        if j > 1:   # a different row order per judge too
            rows = rows[:]
            random.Random(int(m.get("seed", 4)) * 100 + j).shuffle(rows)
        for n in range(0, len(rows), chunk):
            name = f"j{j}-chunk-{n // chunk + 1:03d}"
            names.append(name)
            with open(out / "packets" / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as o:
                for r in rows[n:n + chunk]:
                    o.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "KEY.json").write_text(json.dumps({"runs": runs, "judges": judges, "rows": key_map}, indent=1),
                                  encoding="utf-8")
    print(json.dumps({"rows": len(keys), "chunks": len(names), "per_judge": len(names) // judges,
                      "missing_outputs": missing,
                      "mean_distinct_outputs": round(sum(len(v["judges"]["1"]) for v in key_map.values())
                                                     / max(1, len(keys)), 2)}))


if __name__ == "__main__":
    main()

"""Chunked vs unchunked refinement: checks, latency, judge manifest and full-set CIs
(docs/refine-chunking.md).

    python tools/eval/chunk_compare.py prepare <eval-run dir> <judge dir>   # checks + latency + manifest
    python tools/eval/chunk_compare.py full <judge dir>                     # after tally_judge_v4.py

`prepare` expects `refine-<model>-<set>.jsonl` and `refine-<model>-chunk-<set>.jsonl` from one
`owf_daytona.py eval --chunk-modes off on` run. It checks that every row the chunker leaves whole
got byte-identical text in both modes (same code path), reports wall-time and token cost on the
rows it splits, and writes a protocol-v4 manifest over those rows only, grading `typed` (what the
app inserts, per-chunk guard fallback included).

`full` folds the judged split rows back into whole sets: a row the chunker leaves whole has the
same output in both modes, so its paired difference is exactly 0. It prints paired pass and
harmful differences with bootstrap 95% CIs per set and overall.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from openwhisprflow.refine import chunk  # noqa: E402

SETS = {"eval4": "training/refine/datasets/eval-v4/eval.jsonl",
        "heldout": "training/refine/datasets/real-v3/final/heldout.jsonl",
        "new": "training/refine/datasets/real-v3/final/heldout-new.jsonl",
        "casual": "training/refine/datasets/real-v3/final/casual-probe.jsonl"}
MODELS = ["v4-4b", "v4-2b", "v4-0.8b"]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT)).replace("\\", "/")


def all_rows() -> dict[str, dict]:
    return {f"{s}:{r['id']}": {**r, "_set": s} for s, p in SETS.items() for r in read_jsonl(ROOT / p)}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def prepare(run: Path, out: Path) -> None:
    rows = all_rows()
    split = sorted(k for k, r in rows.items() if len(chunk.split(r["raw"])) >= 2)
    out.mkdir(parents=True, exist_ok=True)
    (out / "keys.txt").write_text("\n".join(split) + "\n", encoding="utf-8")
    print(f"rows {len(rows)}, split by the chunker {len(split)} "
          f"(eval4 {sum(k.startswith('eval4:') for k in split)}, long {sum(rows[k].get('cat') == 'long' for k in split)})")
    outputs, lat = {}, {}
    for m in MODELS:
        for mode in ("", "-chunk"):
            outputs[m + mode] = {}
            for s in SETS:
                f = run / f"refine-{m}{mode}-{s}.jsonl"
                if f.exists():
                    outputs[m + mode][s] = rel(f)
        recs = {mode: {f"{s}:{r['id']}": r for s in SETS for r in read_jsonl(ROOT / outputs[m + mode][s])}
                for mode in ("", "-chunk") if outputs[m + mode]}
        if len(recs) < 2:
            continue
        a, b = recs[""], recs["-chunk"]
        same = sum(a[k]["typed"] == b[k]["typed"] for k in a if k not in split)
        n_whole = sum(1 for k in a if k not in split)
        wa = [a[k]["wall_ms"] for k in split]
        wb = [b[k]["wall_ms"] for k in split]
        extra = [y - x for x, y in zip(wa, wb)]
        ntok = lambda r, f: (r.get("timings") or {}).get(f) or 0  # noqa: E731
        lat[m] = {"rows": len(split), "whole_rows_identical": f"{same}/{n_whole}",
                  "pieces_mean": round(statistics.mean(len(b[k].get("chunks") or [0]) for k in split), 2),
                  "unchunked_ms_median": round(statistics.median(wa)), "chunked_ms_median": round(statistics.median(wb)),
                  "unchunked_ms_p95": round(pct(wa, 0.95)), "chunked_ms_p95": round(pct(wb, 0.95)),
                  "extra_ms_median": round(statistics.median(extra)), "extra_ms_mean": round(statistics.mean(extra)),
                  "extra_ms_p95": round(pct(extra, 0.95)),
                  "gen_tokens_unchunked": sum(ntok(a[k], "predicted_n") for k in split),
                  "gen_tokens_chunked": sum(ntok(b[k], "predicted_n") for k in split),
                  "prompt_tokens_unchunked": sum(ntok(a[k], "prompt_n") for k in split),
                  "prompt_tokens_chunked": sum(ntok(b[k], "prompt_n") for k in split),
                  "chunk_guard_rejects": sum(p["guard"] != "ok" for k in split for p in b[k].get("chunks") or []),
                  "unchunked_guard_rejects": sum(a[k]["guard"] != "ok" for k in split)}
    print(json.dumps(lat, indent=1))
    (out / "latency.json").write_text(json.dumps(lat, indent=1), encoding="utf-8")
    manifest = {"name": out.name, "judges": 3, "chunk": 11, "seed": 1006, "field": "typed",
                "keys": rel(out / "keys.txt"), "sets": SETS, "outputs": {k: v for k, v in outputs.items() if v}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {out / 'manifest.json'}")


def boot(d: list[int], B: int = 10000, seed: int = 4) -> tuple[float, float, float]:
    rng, n = random.Random(seed), len(d)
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def full(judge: Path) -> None:
    rows = all_rows()
    maj = json.loads((judge / "majority.json").read_text(encoding="utf-8"))
    res = {}
    for m in MODELS:
        A, B = maj.get(m + "-chunk"), maj.get(m)
        if not A or not B:
            continue
        res[m] = {}
        for scope in ("long", "eval4", "all"):
            ks = [k for k in rows if scope == "all" or rows[k]["_set"] == scope or rows[k].get("cat") == scope]
            dp = [int(A[k]["pass"]) - int(B[k]["pass"]) if k in A else 0 for k in ks]
            dh = [int(A[k]["harmful"]) - int(B[k]["harmful"]) if k in A else 0 for k in ks]
            judged = [k for k in ks if k in A]
            p, plo, phi = boot(dp)
            h, hlo, hhi = boot(dh, seed=5)
            res[m][scope] = {"n": len(ks), "judged": len(judged),
                             "chunk_pass_judged": sum(A[k]["pass"] for k in judged),
                             "plain_pass_judged": sum(B[k]["pass"] for k in judged),
                             "only_chunk": sum(d > 0 for d in dp), "only_plain": sum(d < 0 for d in dp),
                             "pass_diff": round(100 * p, 1), "pass_ci": [round(100 * plo, 1), round(100 * phi, 1)],
                             "harm_diff": round(100 * h, 1), "harm_ci": [round(100 * hlo, 1), round(100 * hhi, 1)]}
            r = res[m][scope]
            print(f"{m:8s} {scope:5s} n={r['n']:3d} judged={r['judged']:2d} pass chunk {r['chunk_pass_judged']} vs "
                  f"{r['plain_pass_judged']} | only chunk {r['only_chunk']} / only plain {r['only_plain']} | "
                  f"pass {r['pass_diff']:+.1f} {r['pass_ci']} | harmful {r['harm_diff']:+.1f} {r['harm_ci']}")
    (judge / "full.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    f = sub.add_parser("full")
    f.add_argument("judge", type=Path)
    a = ap.parse_args()
    prepare(a.run, a.out) if a.cmd == "prepare" else full(a.judge)


if __name__ == "__main__":
    main()

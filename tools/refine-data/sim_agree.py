"""v3-4b agreement check on the sim-v4 distillation set (round 4, Phase D).

    python tools/refine-data/sim_agree.py --outputs tools/eval/out/eval-runs/<run>/refine-v3-4b-simagree.jsonl

v3-4b (the shipped 4B, trained on real-audio data only) cleaned a 6,026-row subset of
`real-v4/sim.jsonl`: every audited row plus 3,600 random ones. Reports how often it agrees with
the simulated target (word level, punctuation and casing ignored), how word error is
distributed, and, on the audited rows, whether disagreement predicts an audit defect (fix or
drop). Writes `distill-v4/audit/agreement.json`.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "training" / "refine" / "datasets"


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))


def wer(ref: str, hyp: str) -> float:
    a, b = words(ref), words(hyp)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)] / max(1, len(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    a = ap.parse_args()
    sim = {}
    for line in (DS / "real-v4" / "sim.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            r = json.loads(line)
            sim[r["id"]] = r
    final = {}
    out_dir = DS / "distill-v4" / "audit"
    for who in ("A", "B"):
        for f in glob.glob(str(out_dir / "out" / f"batch-*.{who}.jsonl")):
            for line in Path(f).read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    r = json.loads(line)
                    final.setdefault(r["id"], {})[who] = r["decision"]
    for f in glob.glob(str(out_dir / "adjudicated" / "batch-*.jsonl")):
        for line in Path(f).read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                r = json.loads(line)
                final.setdefault(r["id"], {})["adj"] = r["decision"]

    def verdict(i: str) -> str | None:
        d = final.get(i)
        if not d:
            return None
        if "adj" in d:
            return d["adj"]
        return d.get("A") if d.get("A") == d.get("B") else None

    w_all, by_writer, by_cat = [], collections.defaultdict(list), collections.defaultdict(list)
    audited = collections.defaultdict(list)
    for line in Path(a.outputs).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        r = sim.get(o["id"])
        if r is None:
            continue
        e = wer(r["clean"], o.get("output") or "")
        w_all.append(e)
        by_writer[r.get("writer") or "?"].append(e)
        by_cat[r.get("cat") or "?"].append(e)
        v = verdict(o["id"])
        if v:
            audited["accept" if v == "accept" else "defect"].append(e)

    def summ(x: list[float]) -> dict:
        return {"n": len(x), "agree_exact_words": round(sum(e == 0 for e in x) / len(x), 3),
                "mean_wer": round(st.mean(x), 4), "share_wer_gt_0.2": round(sum(e > 0.2 for e in x) / len(x), 3)}

    rep = {"all": summ(w_all), "by_writer": {k: summ(v) for k, v in sorted(by_writer.items())},
           "by_cat": {k: summ(v) for k, v in sorted(by_cat.items()) if len(v) >= 30},
           "audited": {k: summ(v) for k, v in audited.items()}}
    acc, dfc = audited.get("accept", []), audited.get("defect", [])
    if acc and dfc:
        rep["defect_rate_by_disagreement"] = {
            t: {"rows": sum(e > t for e in acc + dfc),
                "defect_share": round(sum(e > t for e in dfc) / max(1, sum(e > t for e in acc + dfc)), 3)}
            for t in (0.0, 0.1, 0.2, 0.3)}
        rep["defect_share_overall"] = round(len(dfc) / (len(acc) + len(dfc)), 3)
    (out_dir / "agreement.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

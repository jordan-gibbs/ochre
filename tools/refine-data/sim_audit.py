"""Sampled two-auditor audit of the sim-v4 distillation set (round 4, Phase D).

    python tools/refine-data/sim_audit.py pack [--frac 0.10] [--size 100]   # -> distill-v4/audit/batch-NNN.jsonl
    python tools/refine-data/sim_audit.py conflicts                         # A vs B -> audit/conflicts/
    python tools/refine-data/sim_audit.py apply                             # -> real-v4/sim-audited.jsonl + stats

The sample is stratified by writer x category (seeded). Each packet row holds what was said
(`spoken`, from the script), the simulated recognizer output after the pre-pass (`raw`) and the
target (`clean`). Two auditors (A Opus, B Sonnet) decide accept / fix_clean / drop
independently; an Opus adjudicator settles disagreements. `apply` writes the training file:
the audited rows take their final decision (fixed target or dropped), the unaudited rows are
kept, and every writer x category slice whose audited drop+fix rate exceeds `--max-bad` (0.25)
is removed entirely.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "training" / "refine" / "datasets"
SIM = DS / "real-v4" / "sim.jsonl"
AUD = DS / "distill-v4" / "audit"


def read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def spoken_by_id() -> dict[str, str]:
    out = {}
    for d in (DS / "distill-v4" / "scripts", DS / "scripts-v2"):
        for f in d.glob("*.jsonl"):
            for r in read(f):
                if "id" in r and "spoken" in r:
                    out[r["id"]] = r["spoken"]
    return out


def slice_key(r: dict) -> str:
    return f"{r.get('writer') or '?'}|{r.get('cat') or '?'}"


def cmd_pack(a) -> None:
    rows = read(SIM)
    spoken = spoken_by_id()
    rng = random.Random(41)
    by = collections.defaultdict(list)
    for r in rows:
        by[slice_key(r)].append(r)
    pick = []
    for k in sorted(by):
        g = by[k]
        n = max(3, round(len(g) * a.frac)) if len(g) >= 10 else len(g) if len(g) <= 3 else 3
        pick += rng.sample(g, min(n, len(g)))
    rng.shuffle(pick)
    keep = ("id", "context", "style", "dictionary", "raw", "clean", "cat", "tags")
    packet = [dict({k: r.get(k) for k in keep}, spoken=spoken.get(r["script_id"], "")) for r in pick]
    for i in range(0, len(packet), a.size):
        write(AUD / f"batch-{i // a.size + 1:03d}.jsonl", packet[i:i + a.size])
    print(f"{len(packet)} rows ({len(packet) / len(rows):.1%}) in {(len(packet) + a.size - 1) // a.size} batches -> {AUD}")


def load_out(who: str) -> dict[str, dict]:
    out = {}
    for f in sorted((AUD / "out").glob(f"batch-*.{who}.jsonl")):
        for r in read(f):
            out[r["id"]] = r
    return out


def cmd_conflicts(a) -> None:
    A, B = load_out("A"), load_out("B")
    n_conf = 0
    for f in sorted(AUD.glob("batch-*.jsonl")):
        rows = read(f)
        conf = []
        for r in rows:
            x, y = A.get(r["id"]), B.get(r["id"])
            if x is None or y is None:
                raise SystemExit(f"{f.name}: {r['id']} missing an auditor output")
            same = x["decision"] == y["decision"] and (
                x["decision"] != "fix_clean" or x.get("clean", "").strip() == y.get("clean", "").strip())
            if not same:
                conf.append({"row": r, "a": x, "b": y})
        if conf:
            write(AUD / "conflicts" / f.name, conf)
        n_conf += len(conf)
    print(f"{n_conf} conflicts of {len(A)} rows")


def final_decisions() -> dict[str, dict]:
    A, B = load_out("A"), load_out("B")
    adj = {}
    for f in sorted((AUD / "adjudicated").glob("batch-*.jsonl")):
        for r in read(f):
            adj[r["id"]] = r
    out = {}
    for i, x in A.items():
        out[i] = adj.get(i) or x
    return out


def cmd_apply(a) -> None:
    rows = read(SIM)
    dec = final_decisions()
    A, B = load_out("A"), load_out("B")
    agree = sum(1 for i in A if i in B and A[i]["decision"] == B[i]["decision"])
    sl = collections.defaultdict(collections.Counter)
    for r in rows:
        d = dec.get(r["id"])
        if d:
            sl[slice_key(r)][d["decision"]] += 1
    bad_slices = {k for k, c in sl.items() if sum(c.values()) >= 5 and
                  (c["drop"] + c["fix_clean"]) / sum(c.values()) > a.max_bad}
    out, st = [], collections.Counter()
    for r in rows:
        d = dec.get(r["id"])
        if slice_key(r) in bad_slices:
            st["slice_removed"] += 1
            continue
        if d is None:
            out.append(r)
            st["unaudited_kept"] += 1
        elif d["decision"] == "drop":
            st["audited_drop"] += 1
        elif d["decision"] == "fix_clean":
            out.append(dict(r, clean=d["clean"], audit={"final": "fix_clean"}))
            st["audited_fix"] += 1
        else:
            out.append(dict(r, audit={"final": "accept"}))
            st["audited_accept"] += 1
    write(DS / "real-v4" / "sim-audited.jsonl", out)
    tot = collections.Counter()
    for c in sl.values():
        tot.update(c)
    rep = {"audited": len(dec), "auditor_agreement": round(agree / max(1, len(A)), 3), "final": dict(tot),
           "bad_slices": sorted(bad_slices), "counts": dict(st), "rows_out": len(out),
           "by_writer": {w: dict(sum((c for k, c in sl.items() if k.startswith(w + "|")), collections.Counter()))
                         for w in sorted({k.split("|")[0] for k in sl})}}
    (AUD / "summary.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--frac", type=float, default=0.10)
    p.add_argument("--size", type=int, default=100)
    sub.add_parser("conflicts")
    q = sub.add_parser("apply")
    q.add_argument("--max-bad", type=float, default=0.25)
    a = ap.parse_args()
    {"pack": cmd_pack, "conflicts": cmd_conflicts, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()

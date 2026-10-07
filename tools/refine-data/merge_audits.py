"""Merge two independent Claude audits of the same batch; apply decisions to real-v2.

    # one batch, two auditor outputs
    .venv-data/Scripts/python tools/refine-data/merge_audits.py merge \
        --batch training/refine/datasets/real-v2/audit/batch-001.jsonl \
        --a training/refine/datasets/real-v2/audit/out/batch-001.A.jsonl \
        --b training/refine/datasets/real-v2/audit/out/batch-001.B.jsonl
    # every batch that has exactly two outputs in audit/out/
    .venv-data/Scripts/python tools/refine-data/merge_audits.py merge --all
    # write agreed decisions (and human decisions from review.html) into real-v2/<W>.jsonl
    .venv-data/Scripts/python tools/refine-data/merge_audits.py apply [--human decisions.jsonl]
    # training-ready file: audited, not dropped
    .venv-data/Scripts/python tools/refine-data/merge_audits.py export

Agreement: same decision; for ``fix_clean`` the same text after light normalization (whitespace,
curly quotes). A ``fix_clean`` whose text equals the current clean counts as ``accept``. Two
``drop``s agree whatever their reasons. Everything else is a conflict.

Outputs: ``audit/merged/batch-XXX.jsonl`` (agreed rows), ``audit/conflicts/batch-XXX.jsonl``
(both decisions + the row, for a human or a third auditor), ``audit/merge_summary.json``.
"""

from __future__ import annotations

import nowmi  # noqa: F401

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AUDIT_DIR, OUT, read_jsonl, write_jsonl  # noqa: E402

DECISIONS = {"accept", "fix_clean", "drop"}
_Q = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "…": "..."})


def light(s: str) -> str:
    s = (s or "").translate(_Q)
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in s.strip().split("\n"))


def load_batch(path: Path) -> tuple[dict, list[dict]]:
    lines = read_jsonl(path)
    header = next((x for x in lines if x.get("kind") == "header"), {})
    return header, [x for x in lines if x.get("kind") == "row"]


def validate(out: list[dict], rows: list[dict], who: str) -> tuple[dict[str, dict], list[str]]:
    errs, by_id = [], {}
    ids = {r["id"] for r in rows}
    for o in out:
        i = o.get("id")
        if i not in ids:
            errs.append(f"{who}: unknown id {i!r}")
            continue
        if i in by_id:
            errs.append(f"{who}: duplicate id {i}")
        d = o.get("decision")
        if d not in DECISIONS:
            errs.append(f"{who}: {i}: bad decision {d!r}")
            continue
        if d == "fix_clean" and not (o.get("clean") or "").strip():
            errs.append(f"{who}: {i}: fix_clean without clean")
            continue
        by_id[i] = o
    for r in rows:
        if r["id"] not in by_id:
            errs.append(f"{who}: missing id {r['id']}")
    return by_id, errs


def effective(o: dict, row: dict) -> tuple[str, str]:
    d = o["decision"]
    if d == "fix_clean":
        if light(o["clean"]) == light(row["clean"]):
            return "accept", ""
        return "fix_clean", light(o["clean"])
    return d, ""


def merge_one(batch: Path, fa: Path, fb: Path) -> dict:
    header, rows = load_batch(batch)
    a_id, ea = validate(read_jsonl(fa), rows, fa.name)
    b_id, eb = validate(read_jsonl(fb), rows, fb.name)
    agreed, conflicts = [], []
    for r in rows:
        oa, ob = a_id.get(r["id"]), b_id.get(r["id"])
        if not oa or not ob:
            conflicts.append({"id": r["id"], "why": "missing_output", "a": oa, "b": ob, "row": r})
            continue
        (da, ca), (db, cb) = effective(oa, r), effective(ob, r)
        if da == db and ca == cb:
            rec = {"id": r["id"], "decision": da, "reasons": [oa.get("reason"), ob.get("reason")],
                   "notes": [x for x in (oa.get("note"), ob.get("note")) if x],
                   "auditors": [fa.stem.split(".")[-1], fb.stem.split(".")[-1]]}
            if da == "fix_clean":
                rec["clean"] = oa["clean"].strip()
            agreed.append(rec)
        else:
            why = "decision" if da != db else "fix_text"
            conflicts.append({"id": r["id"], "why": why, "a": oa, "b": ob, "row": r})
    name = batch.stem
    write_jsonl(AUDIT_DIR / "merged" / f"{name}.jsonl", agreed)
    write_jsonl(AUDIT_DIR / "conflicts" / f"{name}.jsonl", conflicts)
    n = len(rows)
    # Cohen's kappa over the three decisions (rows both auditors labelled)
    pairs = [(effective(a_id[i], r)[0], effective(b_id[i], r)[0]) for r in rows
             for i in [r["id"]] if i in a_id and i in b_id]
    po = sum(x == y for x, y in pairs) / max(1, len(pairs))
    pe = sum((sum(x == k for x, _ in pairs) / max(1, len(pairs))) * (sum(y == k for _, y in pairs) / max(1, len(pairs)))
             for k in DECISIONS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    s = {"batch": name, "rows": n, "agreed": len(agreed), "conflicts": len(conflicts),
         "decision_agreement": round(po, 3), "kappa": round(kappa, 3),
         "agreed_decisions": {d: sum(x["decision"] == d for x in agreed) for d in DECISIONS},
         "errors": ea + eb}
    print(json.dumps({k: v for k, v in s.items() if k != "errors"}), *(f"\n  ! {e}" for e in s["errors"][:10]))
    return s


def cmd_merge(a: argparse.Namespace) -> None:
    summaries = []
    if a.all:
        for batch in sorted(AUDIT_DIR.glob("batch-*.jsonl")):
            outs = sorted((AUDIT_DIR / "out").glob(f"{batch.stem}.*.jsonl"))
            if len(outs) == 2:
                summaries.append(merge_one(batch, outs[0], outs[1]))
            elif outs:
                print(f"{batch.stem}: {len(outs)} outputs (need exactly 2), skipped")
    else:
        summaries.append(merge_one(Path(a.batch), Path(a.a), Path(a.b)))
    f = AUDIT_DIR / "merge_summary.json"
    old = {s["batch"]: s for s in json.loads(f.read_text(encoding="utf-8-sig"))["batches"]} if f.exists() else {}
    old.update({s["batch"]: s for s in summaries})
    tot = {k: sum(s[k] for s in old.values()) for k in ("rows", "agreed", "conflicts")}
    f.write_text(json.dumps({"total": tot, "batches": sorted(old.values(), key=lambda s: s["batch"])}, indent=1),
                 encoding="utf-8")
    print(f"total: {tot}")


def cmd_apply(a: argparse.Namespace) -> None:
    # Later files win (batch-rx-* re-audits sort after batch-0xx). Track which file each decision
    # came from, so a stale adjudication can't override a newer re-audit of the same row.
    claude, claude_src = {}, {}
    for f in sorted((AUDIT_DIR / "merged").glob("*.jsonl")):
        for x in read_jsonl(f):
            claude[x["id"]] = x; claude_src[x["id"]] = f.stem
    human = {x["id"]: x for x in read_jsonl(Path(a.human))} if a.human else {}
    adj = {}
    for f in sorted((AUDIT_DIR / "adjudicated").glob("*.jsonl")):
        for x in read_jsonl(f):
            if claude_src.get(x["id"], "") > f.stem:
                continue  # a newer batch (re-audit) agreed on this row; its decision stands
            adj[x["id"]] = x
            claude.pop(x["id"], None) if claude_src.get(x["id"], "") < f.stem else None
    changed = 0
    for f in sorted(OUT.glob("W*.jsonl")) + sorted(OUT.glob("R2.jsonl")) + sorted(OUT.glob("E*.jsonl")):
        rows = read_jsonl(f)
        for r in rows:
            hit = False
            if r["id"] in claude:  # relative to the script's intended text
                c = claude[r["id"]]
                r["audit"]["claude"] = {k: c[k] for k in ("decision", "reasons", "notes", "auditors", "clean") if k in c}
                r.pop("drop", None)
                r["clean"] = c["clean"] if c["decision"] == "fix_clean" else r["intended"]
                if c["decision"] == "drop":
                    r["drop"] = next((x for x in c.get("reasons", []) if x), "drop")
                hit = True
            if r["id"] in adj:  # a third (Opus) auditor settled an A/B conflict
                d = adj[r["id"]]
                r["audit"]["adjudicator"] = {k: d[k] for k in ("decision", "clean", "reason", "note") if k in d}
                r.pop("drop", None)
                if d["decision"] == "fix_clean" and (d.get("clean") or "").strip():
                    r["clean"] = d["clean"]
                elif d["decision"] == "accept":
                    r["clean"] = r["intended"]
                else:
                    r["drop"] = "adjudicator: " + (d.get("reason") or "drop")
                hit = True
            if r["id"] in human:  # relative to what review.html showed (clean after the Claude audit)
                h = human[r["id"]]
                r["audit"]["human"] = {k: h[k] for k in ("decision", "clean", "note") if k in h}
                r.pop("drop", None)
                if h["decision"] == "fix_clean" and (h.get("clean") or "").strip():
                    r["clean"] = h["clean"]
                elif h["decision"] == "drop":
                    r["drop"] = "human: " + (h.get("note") or "drop")
                hit = True
            changed += hit
        write_jsonl(f, rows)
    print(f"applied decisions to {changed} rows (claude {len(claude)}, adjudicated {len(adj)}, human {len(human)})")


def cmd_export(a: argparse.Namespace) -> None:
    keep = ["id", "source", "mode", "style", "dictionary", "context", "raw", "clean", "tags", "raw_engine", "voice"]
    if a.eval:   # round 4: eval-only E files -> final/eval-v4-real.jsonl (never training data)
        keep += ["cat", "spoken"]
        files, dest = sorted(OUT.glob("E*.jsonl"), key=lambda p: (len(p.stem), p.stem)), OUT / "final" / "eval-v4-real.jsonl"
    else:
        files, dest = sorted(OUT.glob("W*.jsonl")) + sorted(OUT.glob("R2.jsonl")), OUT / "final" / "real-v2.jsonl"
    out, stats = [], {"audited": 0, "dropped": 0, "unaudited": 0}
    for f in files:
        for r in read_jsonl(f):
            au = r.get("audit", {})
            if not ({"claude", "adjudicator", "human"} & au.keys()):
                stats["unaudited"] += 1
                continue
            if r.get("drop"):
                stats["dropped"] += 1
                continue
            stats["audited"] += 1
            out.append({k: r.get(k) for k in keep})
    write_jsonl(dest, out)
    print(f"exported {len(out)} rows -> {dest}; {stats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--batch")
    m.add_argument("--a")
    m.add_argument("--b")
    m.add_argument("--all", action="store_true")
    p = sub.add_parser("apply")
    p.add_argument("--human", help="JSONL exported from review.html")
    x = sub.add_parser("export")
    x.add_argument("--eval", action="store_true", help="export the eval-only E files instead (round 4)")
    a = ap.parse_args()
    if a.cmd == "merge":
        if not a.all and not (a.batch and a.a and a.b):
            ap.error("merge needs --all or --batch/--a/--b")
        cmd_merge(a)
    elif a.cmd == "apply":
        cmd_apply(a)
    else:
        cmd_export(a)


if __name__ == "__main__":
    main()

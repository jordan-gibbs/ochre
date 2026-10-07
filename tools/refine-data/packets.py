"""Audit packets for the Claude auditors: ``training/refine/datasets/real-v2/audit/batch-XXX.jsonl``.

    .venv-data/Scripts/python tools/refine-data/packets.py [--size 50] [--unaudited-only]

Each batch file is self-contained JSON Lines:
* line 1: ``{"kind": "header", ...}`` with the batch id, row count, the full rubric
  (AUDIT_RUBRIC.md), the GUIDE's target rules, and the output contract (where to write, format).
* then one ``{"kind": "row", ...}`` per row with everything an auditor needs (no audio needed).

Batch membership is fixed by script order (W1..W8, file order): batch k holds scripts
50(k-1)+1..50k, so numbering never shifts while the pipeline is still running. A batch file is
written only once all of its rows are assembled; existing batch files are left alone (they may
already be out for audit) unless ``--rewrite``. ``index.json`` lists every batch with its ids.
"""

from __future__ import annotations

import nowmi  # noqa: F401

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from augment import summary  # noqa: E402
from common import AUDIT_DIR, OUT, ROOT, is_eval_file, load_scripts, read_jsonl, write_jsonl  # noqa: E402

RUBRIC = OUT / "AUDIT_RUBRIC.md"
GUIDE = ROOT / "training" / "refine" / "datasets" / "GUIDE.md"


def guide_rules() -> str:
    g = GUIDE.read_text(encoding="utf-8-sig")
    start = g.index("## The hard filler rule")
    end = g.index("## v2: real-audio pairs")
    rec = g[g.index("### Recoverability rule"):]
    return g[start:end].strip() + "\n\n" + rec.strip()


def row_packet(r: dict) -> dict:
    a = r["audit"]["auto"]
    return {"kind": "row", "id": r["id"], "context": r["context"], "style": r["style"],
            "dictionary": r["dictionary"], "tags": r.get("tags", []), "spoken": r["spoken"],
            "raw_asr": r["raw_asr"], "raw": r["raw"], "clean": r["clean"], "raw_engine": r["raw_engine"],
            "voice": r["voice"], "audio_conditions": summary(r["audio_conditions"]),
            "audit_auto": {k: a[k] for k in ("severity", "wer", "missing", "not_in_spoken", "flags")},
            **({"tts_text": r["tts_text"]} if r.get("tts_text") and r["tts_text"] != r["spoken"] else {}),
            **({"orig_id": r["orig_id"]} if r.get("orig_id") else {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--rewrite", action="store_true", help="rewrite batch files that already exist")
    ap.add_argument("--set", default="main", choices=["main", "r2", "rx", "eval", "ids"],
                    help="main: W1..W8 (batch-XXX); r2: re-rendered rows from R2.jsonl (batch-r2-XXX); "
                         "rx: rows whose raw changed after their batch was written (batch-rx-XXX, ids from "
                         "audit/raw_changed.jsonl); eval: eval-only E files (batch-e-XXX, round 4); ids: a re-audit of the ids in --ids-file "
                         "(one per line), written as <--prefix>XXX")
    ap.add_argument("--ids-file", type=Path)
    ap.add_argument("--prefix", default="batch-rx2-")
    a = ap.parse_args()
    have = {r["id"]: r for f in OUT.glob("W*.jsonl") for r in read_jsonl(f)}
    if a.set == "main":
        order, prefix, index_name = [r["id"] for r in load_scripts()], "batch-", "index.json"
    elif a.set == "r2":
        order, prefix, index_name = [r["id"] for r in load_scripts(["R2"])], "batch-r2-", "index-r2.json"
        have = {r["id"]: r for r in read_jsonl(OUT / "R2.jsonl")}
    elif a.set == "eval":
        efiles = sorted({f.stem for f in OUT.glob("E*.jsonl") if is_eval_file(f.stem)}, key=lambda s: (len(s), s))
        order, prefix, index_name = [r["id"] for r in load_scripts(efiles)], "batch-e-", "index-eval.json"
        have = {r["id"]: r for f in efiles for r in read_jsonl(OUT / f"{f}.jsonl")}
    elif a.set == "ids":
        order = [x.strip() for x in a.ids_file.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        prefix, index_name = a.prefix, f"index-{a.prefix.strip('-')}.json"
        have = {r["id"]: r for f in list(OUT.glob("W*.jsonl")) + list(OUT.glob("E*.jsonl")) for r in read_jsonl(f)}
    else:
        order = [r["id"] for r in read_jsonl(AUDIT_DIR / "raw_changed.jsonl")]
        prefix, index_name = "batch-rx-", "index-rx.json"
    rubric = RUBRIC.read_text(encoding="utf-8-sig")
    rubric_sha = hashlib.sha256(rubric.encode()).hexdigest()[:12]
    version = (re.search(r"\(v(\d+)", rubric.splitlines()[0]) or [None, "1"])[1]
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rules = guide_rules()
    index, written, pending = [], 0, 0
    for n, i in enumerate(range(0, len(order), a.size), 1):
        ids = order[i:i + a.size]
        name = f"{prefix}{n:03d}"
        path = AUDIT_DIR / f"{name}.jsonl"
        ready = all(x in have for x in ids)
        index.append({"batch": name, "rows": len(ids), "first": ids[0], "last": ids[-1], "ids": ids,
                      "ready": ready or path.exists()})
        if not ready:
            pending += 1
            continue
        if path.exists() and not a.rewrite:
            continue
        b = [have[x] for x in ids]
        header = {
            "kind": "header", "batch": name, "rows": len(b), "rubric_version": f"v{version}",
            "rubric_sha256": rubric_sha,
            "task": ("Audit every row of this batch with the rubric below. For each row decide accept, fix_clean "
                     "(with the full corrected clean text) or drop (with a reason code). Work row by row, in order; "
                     "rows are independent. Output exactly one JSON object per row, same order, as JSON Lines."),
            "output": {"path": f"training/refine/datasets/real-v2/audit/out/{name}.<auditor>.jsonl",
                       "auditor": "a short id that differs between the two independent passes, e.g. A or B",
                       "line": {"id": "row id", "decision": "accept|fix_clean|drop",
                                "clean": "fix_clean only: the entire new target",
                                "reason": "ok | unrecoverable_word | dropped_by_recognizer | number_value | "
                                          "spelling_from_raw | guide_violation | trailing_off | garbage | non_english | "
                                          "hallucination | hesitation_misheard_as_word | unrecoverable_meaning | script_defect",
                                "rules": ["R1..R13 or RECOVER (optional)"], "note": "optional, one line"}},
            "rubric": rubric,
            "guide_rules": rules,
        }
        write_jsonl(path, [header] + [row_packet(r) for r in b])
        written += 1
    (AUDIT_DIR / index_name).write_text(json.dumps({"rubric_version": f"v{version}", "rubric_sha256": rubric_sha,
                                                      "batch_size": a.size, "batches": index}, indent=1),
                                          encoding="utf-8")
    (AUDIT_DIR / "out").mkdir(exist_ok=True)
    ready = sum(x["ready"] for x in index)
    print(f"{ready}/{len(index)} batches ready ({written} written now, {pending} waiting for rows) -> {AUDIT_DIR}")


if __name__ == "__main__":
    main()

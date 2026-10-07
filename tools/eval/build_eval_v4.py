"""Assemble eval-v4 (round 4): the audited eval-only rows -> training/refine/datasets/eval-v4/eval.jsonl.

    python tools/refine-data/merge_audits.py export --eval        # real-v2/final/eval-v4-real.jsonl
    python tools/eval/build_eval_v4.py

Every row keeps its category (`cat`, from the E-file writer brief) for per-category reporting.
The legacy sets (real-v3/final/heldout.jsonl 150, heldout-new.jsonl 60, casual-probe.jsonl 12)
are judged next to it for continuity; they are read in place.
Also writes eval-v4/README.md with the counts.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "training" / "refine" / "datasets"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def main() -> None:
    cats = {}
    for f in (DS / "scripts-v2").glob("E*.jsonl"):
        for r in read_jsonl(f):
            cats[r["id"]] = r.get("cat") or f.stem
    rows = read_jsonl(DS / "real-v2" / "final" / "eval-v4-real.jsonl")
    out = []
    for r in rows:
        r["cat"] = cats.get(r["id"], "?")
        r["source"] = "eval-v4"
        r["clean"] = r["clean"].rstrip() if r["clean"].strip() else r["clean"]
        out.append({k: r.get(k) for k in ("id", "source", "mode", "style", "dictionary", "context", "raw", "clean",
                                           "tags", "cat", "raw_engine", "voice")})
    dest = DS / "eval-v4" / "eval.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    c = collections.Counter(r["cat"] for r in out)
    scripted = collections.Counter(cats.values())
    lines = ["# eval-v4 (held-out, never trained on)", "",
             "Audited real-audio rows from the eval-only writers (scripts-v2/E1..E13, reserved names and",
             "topics), rendered TTS -> augmentation -> Parakeet -> strip_hesitations, two auditors + an",
             "adjudicator (rubric v5/v6). Built by tools/eval/build_eval_v4.py.", "",
             "| category | scripted | kept |", "|---|---:|---:|"]
    for k in sorted(scripted):
        lines.append(f"| {k} | {scripted[k]} | {c.get(k, 0)} |")
    lines.append(f"| **total** | {sum(scripted.values())} | {len(out)} |")
    (DS / "eval-v4" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

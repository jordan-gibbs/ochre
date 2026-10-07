"""Prove src/openwhisprflow/refine/chunk.py (the eval harness's port) splits exactly like
crates/ochre-refine/src/chunk.rs.

Runs the same inputs through `cargo run -p ochre-refine --example chunk_parity` and through the
Python port and compares the pieces (separator + text) per case. Inputs: a hand-written table, a
seeded fuzz set (sentences, correction cues, "new paragraph" commands, list cues, abbreviations,
Unicode, odd whitespace) at several thresholds, and every `raw` in the given JSONL files.

    python tools/eval/chunk_parity.py [--fuzz 3000] [--from training/refine/datasets/eval-v4/eval.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from openwhisprflow.refine import chunk  # noqa: E402

TABLE = [
    "",
    "Short one.",
    "We can have it done by Thursday. No sorry. By Wednesday afternoon, if you let us know today, New paragraph "
    "also the chain is stretched past point seven five.",
    "Hi Dr. Smith, thanks... I mean it. e.g. this a.m. we said no. 5 is fine.",
    "New paragraph at the start. And new paragraph at the end new paragraph",
    "First we ship it. Second we tell support. Third we watch errors. Bullet point one. Number two is here.",
    "Ünïcödé Wörds. Ärger über Öl. “Quoted Start.” (Paren start.) 'Single.'",
    "tabs\tand\nnewlines　ideographic nbsp thin",
]

WORDS = ("we ship the build today and then tell support about it because the rim is worn which "
         "means a new hub for handling cold chain deliveries usually on tuesday mostly vaccines "
         "Torsten Beatrice Lisbon GitHub I'd it's don't o'neil 2.8B 0.75 $140 riverside.org").split()
CUES = ["no wait", "actually", "sorry", "scratch that", "I mean", "or rather", "make that", "No sorry.",
        "Sorry.", "Actually,", "let me rephrase", "never mind"]
LISTS = ["first", "second", "third", "finally", "bullet point", "number one", "number two", "step 3",
         "new line", "colon", "Next line"]
ABBR = ["Dr.", "Mr.", "e.g.", "i.e.", "a.m.", "p.m.", "St.", "No.", "J.", "etc.", "U.S."]
PUNCT = [".", ".", ".", "?", "!", ",", "...", "…", ".\"", ".)", ".”", ";", ":"]
LEAD = ["", "", "", "", "\"", "(", "“", "'"]
SEPS = [" ", " ", " ", "  ", "\t", "\n", " ", " "]


def fuzz_text(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 30)):
        r = rng.random()
        if r < 0.08:
            parts.append(rng.choice(["New paragraph", "new paragraph", "New Paragraph.", "New paragraph,",
                                     "new paragraph."]))
            continue
        if r < 0.16:
            parts.append(rng.choice(CUES))
        elif r < 0.22:
            parts.append(rng.choice(LISTS))
        elif r < 0.27:
            parts.append(rng.choice(ABBR))
        n = rng.randint(1, 16)
        ws = [rng.choice(WORDS) for _ in range(n)]
        if rng.random() < 0.7:
            ws[0] = ws[0][:1].upper() + ws[0][1:]
        s = rng.choice(LEAD) + " ".join(ws)
        if rng.random() < 0.85:
            s += rng.choice(PUNCT)
        parts.append(s)
    sep = rng.choice(SEPS)
    return sep.join(parts)


def cases(fuzz: int, files: list[Path]) -> list[dict]:
    out = [{"text": t, "min": m, "target": g} for t in TABLE for m, g in ((1, 10), (5, 8), (80, 40))]
    rng = random.Random(11)
    for _ in range(fuzz):
        out.append({"text": fuzz_text(rng), "min": rng.choice([1, 20, 40, 80]), "target": rng.choice([0, 10, 25, 40])})
    for f in files:
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("raw"):
                    out.append({"text": r["raw"], "min": 80, "target": 40})
                    out.append({"text": r["raw"], "min": 30, "target": 20})
    return out


def run_rust(cs: list[dict]) -> list[list]:
    payload = "".join(json.dumps(c) + "\n" for c in cs)
    env = dict(os.environ)
    env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target" / "refine-data"))
    p = subprocess.run(["cargo", "run", "-q", "-j", "4", "-p", "ochre-refine", "--example", "chunk_parity"], cwd=ROOT,
                       input=payload.encode("utf-8"), capture_output=True, env=env)
    if p.returncode:
        sys.stderr.write(p.stderr.decode("utf-8", "replace"))
        raise SystemExit("cargo run failed")
    return [json.loads(x)["pieces"] for x in p.stdout.decode("utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=3000)
    ap.add_argument("--from", dest="files", nargs="*", type=Path, default=[])
    a = ap.parse_args()
    cs = cases(a.fuzz, a.files)
    rust = run_rust(cs)
    assert len(rust) == len(cs), (len(rust), len(cs))
    bad = split_cases = 0
    for c, r in zip(cs, rust):
        py = [[p.sep, p.text] for p in chunk.split(c["text"], c["min"], c["target"])]
        split_cases += len(py) > 1
        if py != r:
            bad += 1
            if bad <= 10:
                print("MISMATCH", json.dumps({"case": c, "rust": r, "py": py}, ensure_ascii=False))
    print(f"chunk parity: {len(cs) - bad}/{len(cs)} identical (table {len(TABLE) * 3}, fuzz {a.fuzz}, "
          f"files {len(a.files)}); {split_cases} cases split into 2+ pieces")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

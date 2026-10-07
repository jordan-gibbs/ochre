"""Prove owf_text.py is byte-identical to crates/ochre/src/text.rs.

Runs the same inputs through `cargo run -p ochre --example text_parity` and through the Python
port, and compares `strip_hesitations` and the full pre-pass (with dictionary words) per case.

Inputs: a hand-written table (every Rust unit test plus edge cases), a seeded fuzz set built from
hesitations / punctuation / acronyms / Unicode, and (with --from) every raw recognizer string in
real-v2 output files (field `raw_asr`).

    .venv-data/Scripts/python tools/refine-data/parity.py [--fuzz 5000] [--from training/refine/datasets/real-v2]
"""

from __future__ import annotations

import nowmi  # noqa: F401  (must precede numpy/scipy)

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from owf_text import app_prepass, collapse_repeats, strip_hesitations  # noqa: E402

TABLE: list[tuple[str, list[str]]] = [
    ("Um, so the, uh, build failed. Uh.", []),
    ("um so the build failed", []),
    ("we're done, uh.", []),
    ("I think uh we should hmm ship it", []),
    ("Yes. Umm, let's go.", []),
    ("The UM department", []),
    ("Ahmed and Erma", []),
    ("summer humming", []),
    ("I like it, you know.", []),
    ("uhhuh is a word", []),
    ("", []),
    ("Uh", []),
    ("Uh.", []),
    ("Um, uh, hmm.", []),
    ("uh, uh, okay", []),
    ("So, uh, the thing is, um, we should go?", []),
    ("Is it ready, uh?", []),
    ("He said \"uh.\" Then left.", []),
    ("(um) okay", []),
    ("Okay (uh) fine", []),
    ("Done; uh.", []),
    ("Done, um!", []),
    ("It's fine.   Um,  next   item", []),
    ("tabs\tum\tand\nnewlines\nuh here", []),
    ("UH oh", []),
    ("UMM, I think so.", []),
    ("So HMM. Okay", []),
    ("UHH the build", []),
    ("UM football and ER visits", []),
    ("Uh oh", []),
    ("U.M. is fine", []),
    ("The ER was busy, er, very busy.", []),
    ("Er, Ah, MM, mhm, Mhm.", []),
    ("ummm uhhh hmmm mmm", []),
    ("über um äh öffnen", []),
    ("Um, élan vital.", []),
    ("um, ñandú", []),
    ("1. Um, first item", []),
    ("Wait... um... what?", []),
    ("hmm? no", []),
    ("Uh-huh, sure.", []),
    ("um-hum", []),
    ("'um' is filler", []),
    ("deploy to kubernetes now", ["Kubernetes"]),
    ("Iphone and iphones", ["Kubernetes", "iPhone"]),
    ("my iphone.", ["iPhone"]),
    ("um, my iphone, uh, broke", ["iPhone"]),
    ("Postgres and postgresql and PostgreSQL", ["PostgreSQL"]),
    ("talk to priya about the jira ticket", ["Priya", "Jira"]),
    ("Github github.com git hub", ["GitHub"]),
    ("Héllo, wörld!", []),
    ("the o'neil report", ["O'Neil"]),
    ("snake_case and case", ["Case"]),
    (" um hello　uh", []),
    ("STRASSE straße um", ["Straße"]),
    ("ǆungla um", []),
    ("İstanbul um", ["istanbul"]),
    ("The 2.8B long. The 2.8B long. The 2.8B long.", []),
    ("So the plan is, um, the plan is to ship Friday.", []),
    ("we should ship it, We should ship it. Then test.", []),
    ("very, very good. no no no no no no. one two three one two three", []),
    ("ça va bien ça va bien. Über alles über alles über alles", []),
    ("send it to github send it to github now", ["GitHub"]),
    ("a - b c a - b c", []),
    ("C D build new line, make new line, make new line, make install.", []),
    ("we should ship it today, We should ship it today. Then test.", []),
    ("Wait for it, wait for it. Wait for it! Silence.", []),
]

WORDS = ("so the build failed and we should ship it today i think that okay yes no wait actually "
         "send me the deck by friday kubernetes github priya meeting at three thirty "
         "ER UM NASA I'm it's don't o'neil e-mail").split()
HES = ["um", "uh", "er", "erm", "hmm", "mm", "mhm", "ah", "umm", "uhh", "hm", "uhm", "mmm"]
PUNCT = ["", "", "", ",", ".", "?", "!", ";", "...", ")", "\"", ".\"", "?)"]
LEAD = ["", "", "", "", "(", "\"", "'"]


def fuzz(n: int, seed: int = 7) -> list[tuple[str, list[str]]]:
    rng = random.Random(seed)
    rep = random.Random(seed + 1)   # separate stream: the original fuzz cases stay the same
    cases = []
    for _ in range(n):
        toks = []
        for _ in range(rng.randint(1, 14)):
            w = rng.choice(HES) if rng.random() < 0.35 else rng.choice(WORDS)
            r = rng.random()
            if r < 0.2:
                w = w.capitalize()
            elif r < 0.25:
                w = w.upper()
            toks.append(rng.choice(LEAD) + w + rng.choice(PUNCT))
        if len(toks) >= 3 and rep.random() < 0.3:   # an immediate repeat, case/punctuation varied
            a, ln = rep.randrange(len(toks)), rep.randint(1, 6)
            seg, more = toks[a:a + ln], []
            for _ in range(rep.randint(1, 2)):
                more += [w.upper() if rep.random() < 0.1 else w.rstrip(".,?!") + rep.choice(PUNCT) for w in seg]
            toks[a + ln:a + ln] = more
        sep = rng.choice([" ", " ", " ", "  ", "\t", "\n"])
        words = rng.sample(["Kubernetes", "GitHub", "Priya", "iPhone", "NASA", "ship"], rng.randint(0, 2))
        cases.append((sep.join(toks), words))
    return cases


def from_outputs(d: Path) -> list[tuple[str, list[str]]]:
    cases = []
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("raw_asr") is not None:
                    cases.append((r["raw_asr"], r.get("dictionary") or []))
    return cases


def run_rust(cases: list[tuple[str, list[str]]]) -> list[dict]:
    payload = "".join(json.dumps({"text": t, "words": w}) + "\n" for t, w in cases)
    env = dict(os.environ)
    env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target" / "refine-data"))
    p = subprocess.run(["cargo", "run", "-q", "-p", "ochre", "--example", "text_parity"], cwd=ROOT,
                       input=payload.encode("utf-8"), capture_output=True, env=env)
    if p.returncode:
        sys.stderr.write(p.stderr.decode("utf-8", "replace"))
        raise SystemExit("cargo run failed")
    return [json.loads(x) for x in p.stdout.decode("utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=5000)
    ap.add_argument("--from", dest="src", type=Path, default=None,
                    help="also check every raw_asr string in this real-v2 directory")
    a = ap.parse_args()
    cases = TABLE + fuzz(a.fuzz)
    n_real = 0
    if a.src:
        real = from_outputs(a.src)
        n_real = len(real)
        cases += real
    rust = run_rust(cases)
    assert len(rust) == len(cases), (len(rust), len(cases))
    bad = 0
    for (text, words), r in zip(cases, rust):
        ps, pc, pf = strip_hesitations(text), collapse_repeats(text), app_prepass(text, words)
        if ps != r["strip"] or pc != r["collapse"] or pf != r["full"]:
            bad += 1
            if bad <= 20:
                print("MISMATCH", json.dumps({"text": text, "words": words, "rust": r,
                                              "py": {"strip": ps, "collapse": pc, "full": pf}}, ensure_ascii=False))
    changed = sum(strip_hesitations(t) != t for t, _ in cases)
    collapsed = sum(collapse_repeats(t) != t for t, _ in cases)
    print(f"parity: {len(cases) - bad}/{len(cases)} identical (table {len(TABLE)}, fuzz {a.fuzz}, "
          f"real recognizer outputs {n_real}); strip changed the text in {changed} cases, "
          f"collapse in {collapsed}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

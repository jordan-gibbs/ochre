"""Bulk script generation with the OpenAI Codex CLI (round 4; owner direction: Codex for bulk generation,
Claude for auditing/judging). Runs `codex exec` jobs in parallel, one JSONL file per job.

    python tools/refine-data/codex_gen.py run --files C001 C002 ... [--parallel 10] [--rows 300] [--model gpt-6-astra]
    python tools/refine-data/codex_gen.py validate [--files ...]

Assignments: training/refine/datasets/scripts-v2/briefs/codex-assign.json (names, topics, emphasis);
brief: briefs/codex-distill.md. Output: training/refine/datasets/distill-v4/scripts/<FILE>.jsonl.
Logs: training/refine/data/codex-logs/<FILE>.log (gitignored).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "training" / "refine" / "datasets"
OUT = DS / "distill-v4" / "scripts"
LOGS = ROOT / "training" / "refine" / "data" / "codex-logs"
ASSIGN = DS / "scripts-v2" / "briefs" / "codex-assign.json"
KEYS = ["id", "context", "style", "dictionary", "spoken", "intended", "tags", "cat", "writer"]
CATS = {"self_correction", "filler_contrast", "long_dictation", "numbers", "voice_command", "trailing_off",
        "ai_prompt", "casual", "technical_terminal", "minimal_edit"}
CTX = {"slack", "email", "ai_chat", "docs", "code_comment", "terminal", "notes", "sms", "issue_tracker", "search"}
HES = re.compile(r"\b(um+|uh+|uhm|er|erm|hmm+|mm+|mhm)\b", re.I)


def prompt_for(a: dict, rows: int, model: str) -> str:
    return (f"Follow training/refine/datasets/scripts-v2/briefs/codex-distill.md exactly. Your file: {a['file']}. "
            f"Write exactly {rows} rows to training/refine/datasets/distill-v4/scripts/{a['file']}.jsonl with ids "
            f"d4-{a['file']}-0001..{rows:04d} and writer \"codex/{model}\". Names to use (and similar invented ones): "
            f"{', '.join(a['names'])}. Topic seeds: {'; '.join(a['topics'])}. Emphasis: {a['emphasis']}. "
            "Write each row yourself (no template generators); validate with python at the end. Write the file as UTF-8 without a BOM.")


def run_one(a: dict, rows: int, model: str, effort: str) -> str:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{a['file']}.log"
    t0 = time.time()
    cmd = ["codex", "exec", "-m", model, "-c", f"model_reasoning_effort=\"{effort}\"", "-s", "workspace-write",
           "-C", str(ROOT), "--skip-git-repo-check", "--ephemeral", "-o", str(LOGS / f"{a['file']}.last.txt"),
           prompt_for(a, rows, model)]
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, shell=sys.platform == "win32")
    return f"{a['file']}: exit {p.returncode} in {time.time() - t0:.0f}s; {validate_file(OUT / (a['file'] + '.jsonl'), rows)}"


def validate_file(p: Path, rows: int | None = None) -> str:
    if not p.exists():
        return "MISSING"
    data = p.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):   # some Codex jobs write through PowerShell, which adds a BOM
        p.write_bytes(data[3:])
    bad, ids, n, cats = [], set(), 0, {}
    for k, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad.append(f"L{k} json")
            continue
        n += 1
        if set(r) - set(KEYS) - {"pair"} or not set(KEYS) - {"writer"} <= set(r):
            bad.append(f"L{k} keys")
        if r.get("id") in ids:
            bad.append(f"L{k} dup")
        ids.add(r.get("id"))
        if r.get("spoken", "") != r.get("spoken", "").lower():
            bad.append(f"L{k} case")
        if HES.search(r.get("intended", "")):
            bad.append(f"L{k} hes")
        if r.get("cat") not in CATS:
            bad.append(f"L{k} cat")
        if r.get("context") not in CTX:
            bad.append(f"L{k} ctx")
        cats[r.get("cat")] = cats.get(r.get("cat"), 0) + 1
    ok = not bad and (rows is None or n == rows)
    return f"{'OK' if ok else 'PROBLEMS'} rows={n} bad={len(bad)} {bad[:5]} cats={cats}"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--files", nargs="+", required=True)
    r.add_argument("--parallel", type=int, default=10)
    r.add_argument("--rows", type=int, default=300)
    r.add_argument("--model", default="gpt-6-astra")
    r.add_argument("--effort", default="medium")
    v = sub.add_parser("validate")
    v.add_argument("--files", nargs="*")
    a = ap.parse_args()
    assign = {x["file"]: x for x in json.loads(ASSIGN.read_text(encoding="utf-8-sig"))}
    if a.cmd == "run":
        with ThreadPoolExecutor(a.parallel) as ex:
            for res in ex.map(lambda f: run_one(assign[f], a.rows, a.model, a.effort), a.files):
                print(res, flush=True)
    else:
        files = a.files or sorted(p.stem for p in OUT.glob("C*.jsonl"))
        for f in files:
            print(f, validate_file(OUT / f"{f}.jsonl"))


if __name__ == "__main__":
    main()

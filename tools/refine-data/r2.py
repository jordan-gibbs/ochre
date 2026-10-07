"""Second pass for rows the audit dropped as TTS artefacts, and re-audit of rows whose raw changed.

    # 1. re-render: every id that either auditor dropped as hesitation_misheard_as_word
    .venv-data/Scripts/python tools/refine-data/r2.py build      # -> training/refine/data/scripts-r2/R2.jsonl
    .venv-data/Scripts/python tools/refine-data/r2.py run        # TTS + Parakeet + assemble + batch-r2-XXX
    # 2. rows already sent out in batches whose raw changed later (pre-pass fix): batch-rx-XXX
    .venv-data/Scripts/python tools/refine-data/r2.py rx

R2 rows: id = original id + "-r2", same script (spoken / intended / context ...), ``orig_id``,
and ``tts_text`` = spoken with the hesitation tokens removed (um umm uh uhh uhm er erm ah hmm hm
mm mhm); repeats, cut-offs, false starts, corrections, pauses and "..." are kept. A new voice is
drawn for the new id (Piper, Kokoro or SAPI). Output: real-v2/R2.jsonl and audit/batch-r2-XXX.jsonl.
"""

from __future__ import annotations

import nowmi  # noqa: F401

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AUDIT_DIR, OUT, ROOT, SCRIPTS_R2, load_scripts, read_jsonl, write_jsonl  # noqa: E402
from common import strip_spoken_hesitations  # noqa: E402
from owf_text import app_prepass  # noqa: E402

KEGPU = ROOT / ".venv-kgpu" / "Scripts" / "python.exe"


def dropped_ids() -> set[str]:
    ids = set()
    for f in sorted((AUDIT_DIR / "out").glob("batch-[0-9]*.*.jsonl")):
        for o in read_jsonl(f):
            if o.get("decision") == "drop" and o.get("reason") == "hesitation_misheard_as_word":
                ids.add(o["id"])
    return ids


def cmd_build(a: argparse.Namespace) -> None:
    """Append-only, so batch-r2 numbering stays stable as more audits come in."""
    ids = dropped_ids()
    scripts = {r["id"]: r for r in load_scripts()}
    order = list(scripts)
    rows = read_jsonl(SCRIPTS_R2 / "R2.jsonl")
    have = {r["orig_id"] for r in rows}
    new = []
    for i in sorted(ids - have, key=lambda x: order.index(x) if x in scripts else 10**9):
        s = scripts.get(i)
        if not s:
            continue
        r = {k: v for k, v in s.items() if not k.startswith("_")}
        r["orig_id"], r["id"] = i, i + "-r2"
        r["tts_text"] = strip_spoken_hesitations(s["spoken"])
        new.append(r)
    write_jsonl(SCRIPTS_R2 / "R2.jsonl", rows + new)
    print(f"R2.jsonl: {len(rows) + len(new)} rows ({len(new)} new) from {len(ids)} dropped ids")


def cmd_run(a: argparse.Namespace) -> None:
    py = sys.executable
    env_args = ["--files", "R2"]
    procs = [subprocess.Popen([py, str(HERE / "tts.py"), "--backend", b] + env_args, cwd=ROOT) for b in ("piper", "sapi")]
    procs.append(subprocess.Popen([str(KEGPU) if KEGPU.exists() else py, str(HERE / "tts.py"), "--backend", "kokoro"]
                                  + env_args, cwd=ROOT))
    procs += [subprocess.Popen([py, str(HERE / "recognize.py"), "--engine", "parakeet", "--wait", "--idle-timeout", "300",
                                "--shard", f"{i}/{a.workers}", "--threads", "4"] + env_args, cwd=ROOT)
              for i in range(a.workers)]
    for p in procs:
        p.wait()
    subprocess.run([py, str(HERE / "assemble.py")] + env_args, cwd=ROOT, check=True)
    subprocess.run([py, str(HERE / "packets.py"), "--set", "r2"], cwd=ROOT, check=True)


def cmd_rx(a: argparse.Namespace) -> None:
    """Rows inside already-written main batches whose raw differs from the current pre-pass."""
    out, seen = [], {r["id"] for r in read_jsonl(AUDIT_DIR / "raw_changed.jsonl")}
    for f in sorted(AUDIT_DIR.glob("batch-[0-9]*.jsonl")):
        for r in read_jsonl(f):
            if r.get("kind") != "row":
                continue
            new = app_prepass(r["raw_asr"], r["dictionary"])
            if new != r["raw"] and r["id"] not in seen:
                out.append({"id": r["id"], "batch": f.stem, "raw_in_batch": r["raw"], "raw_now": new})
    write_jsonl(AUDIT_DIR / "raw_changed.jsonl", read_jsonl(AUDIT_DIR / "raw_changed.jsonl") + out)
    print(f"{len(out)} newly changed rows (total {len(seen) + len(out)}) -> audit/raw_changed.jsonl")
    subprocess.run([sys.executable, str(HERE / "packets.py"), "--set", "rx"], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=3)
    sub.add_parser("rx")
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "rx": cmd_rx}[a.cmd](a)


if __name__ == "__main__":
    main()

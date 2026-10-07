"""Stage 3: assemble ``training/refine/datasets/real-v2/<W>.jsonl`` and run audit pass 1.

    .venv-data/Scripts/python tools/refine-data/assemble.py [--files W1 ..]

Row = the script fields + ``raw`` (recognizer output after the app's pre-pass), ``raw_asr``
(before it), ``raw_engine``, ``voice``, ``voice_detail``, ``audio_conditions``, ``audio_s``,
``phrases`` (phrase cuts), ``audio_file`` (cache path, gitignored), ``clean`` (= ``intended``
until an audit changes it) and ``audit`` (``auto`` from pass 1; Claude decisions are merged in
later by merge_audits.py apply and kept across re-assembly). Only scripts with a finished
recognition are written. Re-running is safe: existing ``audit`` keys other than ``auto`` and an
audited ``clean`` are preserved.
"""

from __future__ import annotations

import nowmi  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_auto import audit_row  # noqa: E402
from owf_text import app_prepass  # noqa: E402
from common import DATA, DRY, HEARD, OUT, ROOT, load_scripts, plan_for, read_jsonl, write_jsonl  # noqa: E402

ASR_OUT = DATA / "asr-v2"
SCRIPT_KEYS = ["id", "context", "style", "dictionary", "spoken", "intended", "tags"]


def build_row(script: dict, prev: dict | None) -> dict | None:
    f = ASR_OUT / f"{script['id']}.json"
    if not f.exists():
        return None
    res = json.loads(f.read_text(encoding="utf-8-sig"))
    plan = plan_for(script)
    meta_f = DRY / f"{script['id']}.json"
    tts_meta = json.loads(meta_f.read_text(encoding="utf-8-sig")) if meta_f.exists() else {}
    row = {k: script.get(k) for k in SCRIPT_KEYS}
    row["dictionary"] = row["dictionary"] or []
    row["style"] = row["style"] or ""
    row.update({
        "source": "real-v2", "mode": "clean",
        # recomputed from raw_asr so a pre-pass fix (e.g. 2026-10-04 "UMM") reaches every row
        "raw": app_prepass(res["raw_asr"], script.get("dictionary") or []), "clean": script["intended"],
        "raw_asr": res["raw_asr"], "raw_engine": res["engine"],
        "voice": plan["voice"]["id"], "voice_detail": plan["voice"],
        "audio_conditions": res["audio_conditions"], "audio_s": res["audio_s"],
        "phrases": res["phrases"],
        "audio_file": str((HEARD / f"{script['id']}.flac").relative_to(ROOT)).replace("\\", "/"),
        "tts_text": tts_meta.get("tts_text", script["spoken"]),
        "tts_units": tts_meta.get("units", []),
        "audit": {},
    })
    if prev:
        kept = {k: v for k, v in (prev.get("audit") or {}).items() if k != "auto"}
        row["audit"].update(kept)
        if prev.get("intended") == script["intended"] and prev.get("clean") != prev.get("intended"):
            row["clean"] = prev["clean"]  # an audit fix survives re-assembly
        if prev.get("drop"):
            row["drop"] = prev["drop"]
    if script.get("orig_id"):
        row["orig_id"] = script["orig_id"]
    row["audit"]["auto"] = audit_row(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*")
    a = ap.parse_args()
    scripts = load_scripts(a.files)
    by_file: dict[str, list[dict]] = {}
    for s in scripts:
        by_file.setdefault(s["_file"], []).append(s)
    total = 0
    for w, rows in by_file.items():
        out_f = OUT / f"{w}.jsonl"
        prev = {r["id"]: r for r in read_jsonl(out_f)}
        built = [r for r in (build_row(s, prev.get(s["id"])) for s in rows) if r]
        if built:
            write_jsonl(out_f, built)
        total += len(built)
        sev = {}
        for r in built:
            k = r["audit"]["auto"]["severity"]
            sev[k] = sev.get(k, 0) + 1
        print(f"{w}: {len(built)}/{len(rows)} rows assembled; auto audit {sev}")
    print(f"total {total} rows -> {OUT}")


if __name__ == "__main__":
    main()

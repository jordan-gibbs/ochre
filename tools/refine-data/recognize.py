"""Stage 2: augment each dry clip, save what the mic "heard", recognize it like the app does.

    .venv-data/Scripts/python tools/refine-data/recognize.py --engine parakeet|whisper
        [--files W1 ..] [--limit N] [--per-file N] [--shard i/n] [--threads 4]

Per clip writes ``training/refine/data/audio-v2/heard/<id>.flac`` (16 kHz, what the recognizer
got) and ``training/refine/data/asr-v2/<id>.json``. Skips finished clips; resumable.
"""

from __future__ import annotations

import nowmi  # noqa: F401  (must precede numpy/scipy)

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, DRY, HEARD, SR, load_scripts, plan_for  # noqa: E402

ASR_OUT = DATA / "asr-v2"


def main() -> None:
    import soundfile as sf

    import asr
    from augment import apply

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["parakeet", "whisper"])
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-file", type=int)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--wait", action="store_true", help="keep polling for dry clips until all planned ones exist")
    ap.add_argument("--idle-timeout", type=float, default=1800, help="with --wait: stop after this long without new audio")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    want = "whisper-large-v3-turbo" if a.engine == "whisper" else "parakeet-tdt-0.6b-v3-int8"
    rows = load_scripts(a.files, a.limit, a.per_file)
    todo = []
    for r in rows:
        p = plan_for(r)
        if p["engine"] != want or int(r["id"].encode().hex(), 16) % n != k:
            continue
        if (ASR_OUT / f"{r['id']}.json").exists():
            continue
        todo.append((r, p))
    print(f"[asr {a.engine} {a.shard}] {len(todo)} clips", flush=True)
    if not todo:
        return
    ASR_OUT.mkdir(parents=True, exist_ok=True)
    HEARD.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = asr.Whisper() if a.engine == "whisper" else asr.Parakeet(a.threads)
    print(f"[asr {a.engine}] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    t0, audio_s, done = time.perf_counter(), 0.0, 0
    pending = list(todo)
    last_progress = time.perf_counter()
    while pending:
        progressed = False
        rest = []
        for r, p in pending:
            dry_f = DRY / f"{r['id']}.flac"
            if not (DRY / f"{r['id']}.json").exists():  # json is written after the flac is complete
                rest.append((r, p))
                continue
            dry, sr = sf.read(str(dry_f), dtype="float32")
            heard, cond = apply(r["id"], dry, p["clean_audio"])
            sf.write(str(HEARD / f"{r['id']}.flac"), heard, SR, subtype="PCM_16")
            res = asr.recognize(engine, heard, r.get("dictionary") or [])
            res.update({"id": r["id"], "engine": want, "audio_conditions": cond,
                        "audio_s": round(len(heard) / SR, 2)})
            (ASR_OUT / f"{r['id']}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            audio_s += res["audio_s"]
            done += 1
            progressed = True
            if done % 25 == 0:
                el = time.perf_counter() - t0
                print(f"[asr {a.engine} {a.shard}] {done}/{len(todo)}, {audio_s / max(el, 1e-6):.1f}x realtime, "
                      f"{done / el * 60:.1f} clips/min", flush=True)
        pending = rest
        if pending:
            if not a.wait:
                print(f"[asr {a.engine} {a.shard}] {len(pending)} clips have no TTS audio yet (use --wait)", flush=True)
                break
            if progressed:
                last_progress = time.perf_counter()
            elif time.perf_counter() - last_progress > a.idle_timeout:
                print(f"[asr {a.engine} {a.shard}] giving up on {len(pending)} clips with no TTS audio", flush=True)
                break
            else:
                time.sleep(10)
    el = time.perf_counter() - t0
    print(f"[asr {a.engine} {a.shard}] done {done} clips, {audio_s / 60:.1f} min audio in {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

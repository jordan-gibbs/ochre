"""One command for the whole real-v2 pipeline: TTS (all backends in parallel) -> augment +
recognize (starts as soon as clips exist) -> assemble + audit pass 1 -> parity check on every
recognizer output -> audit packets -> report.

    # pilot: first 13 scripts of every W file
    .venv-data/Scripts/python tools/refine-data/run.py --per-file 13
    # everything (resumes; finished clips are skipped)
    .venv-data/Scripts/python tools/refine-data/run.py

Logs: training/refine/data/logs-v2/<job>.log. Worker counts are flags; defaults suit a 16-thread
CPU + one shared GPU (GPU jobs: Piper ~0.5 GB, Qwen3-TTS ~3 GB, Whisper ~2 GB of VRAM).
"""

from __future__ import annotations

import nowmi  # noqa: F401

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LOGS, ROOT  # noqa: E402

PY = sys.executable
# Kokoro runs on the GPU from its own venv when present (onnxruntime-gpu 1.23 for CUDA 12 can't
# share .venv-data with onnx-asr's CPU onnxruntime): see the setup notes in common.py.
KGPU = ROOT / ".venv-kgpu" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def sel_args(a: argparse.Namespace) -> list[str]:
    out = []
    if a.files:
        out += ["--files", *a.files]
    if a.per_file:
        out += ["--per-file", str(a.per_file)]
    if a.limit:
        out += ["--limit", str(a.limit)]
    return out


def chain(name: str, cmds: list[list[str]], results: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(LOGS / f"{name}.log", "a", encoding="utf-8") as log:
        for c in cmds:
            log.write(f"\n$ {' '.join(c)}\n")
            log.flush()
            p = subprocess.run(c, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1",
                                    "PYTHONIOENCODING": "utf-8"})
            if p.returncode:
                results[name] = f"exit {p.returncode}"
                return
    results[name] = f"ok {time.perf_counter() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--per-file", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--piper-workers", type=int, default=2)
    ap.add_argument("--kokoro-workers", type=int, default=3)
    ap.add_argument("--kokoro-threads", type=int, default=3)
    ap.add_argument("--qwen-workers", type=int, default=1)
    ap.add_argument("--sapi-workers", type=int, default=1)
    ap.add_argument("--parakeet-workers", type=int, default=2)
    ap.add_argument("--parakeet-threads", type=int, default=5)
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--post", choices=["all", "assemble"], default="all",
                    help="assemble: only assemble rows after recognition (the cloud job, tools/cloud/: no cargo for "
                         "parity, and packets need every W file, so they run where the dataset lives)")
    a = ap.parse_args()
    s = sel_args(a)
    tts = [PY, str(HERE / "tts.py")]
    rec = [PY, str(HERE / "recognize.py")]
    jobs: dict[str, list[list[str]]] = {}
    if not a.skip_tts:
        for i in range(a.piper_workers):
            jobs[f"tts-piper-{i}"] = [tts + ["--backend", "piper", "--shard", f"{i}/{a.piper_workers}"] + s]
        for i in range(a.kokoro_workers):
            kok = [str(KGPU) if KGPU.exists() else PY, str(HERE / "tts.py")]
            jobs[f"tts-kokoro-{i}"] = [kok + ["--backend", "kokoro", "--shard", f"{i}/{a.kokoro_workers}",
                                              "--threads", str(a.kokoro_threads)] + s]
        for i in range(a.qwen_workers):  # clone voice after the presets, so at most one Qwen model per worker
            jobs[f"tts-qwen-{i}"] = [tts + ["--backend", "qwen", "--shard", f"{i}/{a.qwen_workers}"] + s,
                                     tts + ["--backend", "qwen_clone", "--shard", f"{i}/{a.qwen_workers}"] + s]
        for i in range(a.sapi_workers):
            jobs[f"tts-sapi-{i}"] = [tts + ["--backend", "sapi", "--shard", f"{i}/{a.sapi_workers}"] + s]
    for i in range(a.parakeet_workers):
        jobs[f"asr-parakeet-{i}"] = [rec + ["--engine", "parakeet", "--wait", "--shard", f"{i}/{a.parakeet_workers}",
                                           "--threads", str(a.parakeet_threads)] + s]
    jobs["asr-whisper"] = [rec + ["--engine", "whisper", "--wait"] + s]
    results: dict[str, str] = {}
    t0 = time.perf_counter()
    threads = [threading.Thread(target=chain, args=(n, c, results), daemon=True) for n, c in jobs.items()]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        done = sorted(results)
        print(f"[run] {time.perf_counter() - t0:.0f}s elapsed; finished: {', '.join(f'{d}={results[d]}' for d in done) or '-'}",
              flush=True)
    wall = time.perf_counter() - t0
    print(f"[run] TTS + recognition wall time {wall / 60:.1f} min", flush=True)
    files = ["--files", *a.files] if a.files else []
    post = [[PY, str(HERE / "assemble.py")] + files,
            [PY, str(HERE / "parity.py"), "--fuzz", "0", "--from", str(ROOT / "training/refine/datasets/real-v2")],
            [PY, str(HERE / "packets.py")],
            [PY, str(HERE / "report.py"), "--wall-s", f"{wall:.0f}"] + s]
    for c in post[:1] if a.post == "assemble" else post:
        print(f"$ {' '.join(c[1:])}", flush=True)
        subprocess.run(c, cwd=ROOT)
    bad = {k: v for k, v in results.items() if not v.startswith("ok")}
    if bad:
        print(f"[run] FAILED jobs (see {LOGS}): {bad}")


if __name__ == "__main__":
    main()

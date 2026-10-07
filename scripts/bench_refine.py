"""Latency + output benchmark for local refinement (Quill on llama-server). SPEC §5.2.

Target: < 1 s for a ~100-word paragraph on CPU, well under on a GPU.

For each (model, accelerator, threads, prompt style) it starts a managed llama-server exactly as
the app does, warms it, then runs every sample ``--reps`` times and reports the median wall time
plus llama.cpp's own split: prompt processing (``prompt_n`` tokens in ``prompt_ms``; ``cache_n`` tokens
reused from the KV prompt cache) vs generation (``predicted_n`` tokens in ``predicted_ms``).

    uv run python scripts/bench_refine.py                                    # 0.8b + 2b, cuda + cpu
    uv run python scripts/bench_refine.py --models quill-0.8b --accels cpu --threads 4,6,8
    uv run python scripts/bench_refine.py --styles quill,shared --show-output

Downloads the models / llama.cpp builds on first run (0.5-1.3 GB per model).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openwhisprflow.refine import local_server as ls  # noqa: E402
from openwhisprflow.refine import prompts  # noqa: E402
from openwhisprflow.refine.base import RefineContext  # noqa: E402
from openwhisprflow.refine.guard import check  # noqa: E402
from openwhisprflow.refine.local import clean_output  # noqa: E402
from openwhisprflow.text.normalize import normalize  # noqa: E402

# Realistic raw ASR output: lower-case, no punctuation, fillers, repeats, self-corrections.
SAMPLES: dict[str, str] = {
    "short": "um can you send me the the file when you get a chance",
    "paragraph": (
        "so um i wanted to give everyone a quick update on the project uh we finished the first round of "
        "user interviews last week and the main thing we heard was that people find the onboarding way too "
        "long like they drop off before they even see the dashboard so uh what we're thinking is we cut it "
        "down to three steps instead of seven and we move the integrations stuff to later you know after "
        "they've actually seen some value um i'll share the full notes in the doc by thursday and if anyone "
        "has concerns just let me know before then"
    ),
    "self_correction": "let's meet on monday no wait make that tuesday at three thirty in the small conference room",
    "list": "okay the grocery list is first eggs second uh milk third bread and fourth some some coffee beans",
    "question": "what time does the pharmacy on main street close today",
    "email": "my email is john dot smith at gmail dot com and the site is example dot com slash pricing",
}


@dataclass
class Row:
    model: str
    accel: str
    threads: int
    style: str
    sample: str
    words: int
    wall_ms: float
    prompt_n: int
    cache_n: int
    prompt_ms: float
    predicted_n: int
    predicted_ms: float
    guard: str
    output: str


def system_for(style: str) -> str:
    return prompts.QUILL_SYSTEM if style == "quill" else prompts.system_prompt(RefineContext())


def build_prompt(style: str, text: str) -> str:
    if style == "quill":
        return prompts.chatml(prompts.QUILL_SYSTEM, text)
    return prompts.chatml(prompts.system_prompt(RefineContext()), prompts.user_message(text))


def run_config(model: str, accel: str, threads: int, styles: list[str], reps: int, ctx: int,
               flash_attn: str, prio: int = 0, server_args: tuple[str, ...] = (), prime: bool = False) -> list[Row]:
    exe, got = ls.ensure_binary(accel)
    path = ls.ensure_model(model)
    extra = (("--prio", str(prio), "--prio-batch", str(prio)) if prio else ()) + server_args
    opts = ls.ServerOptions(ctx=ctx, threads=threads, flash_attn=flash_attn, extra=extra)
    server = ls.LlamaServer(exe, path, accel=got, options=opts)
    t0 = time.perf_counter()
    server.start()
    startup = (time.perf_counter() - t0) * 1000
    print(f"\n## {model} / {got} / threads={threads} / ctx={ctx} / fa={flash_attn} / prio={prio}  "
          f"(cold start, spawn -> /health ok: {startup:.0f} ms) prime={prime} args={' '.join(server_args)}", flush=True)
    rows: list[Row] = []
    try:
        for style in styles:
            # Round-robin over the samples so consecutive requests always differ, as in real use:
            # only the shared system-prompt prefix is reused from the KV cache, never the dictation.
            runs: dict[str, list[tuple[float, dict]]] = {name: [] for name in SAMPLES}
            prefix = prompts.chatml_prefix(system_for(style))
            if prime:
                server.completion(prefix, n_predict=0, timeout=60)
            for rep in range(reps + 1):          # rep 0 is a warm-up pass (not counted)
                for name, text in SAMPLES.items():
                    n_predict = max(32, int(len(text) / 3.2 * 2) + 16)
                    t = time.perf_counter()
                    data = server.completion(build_prompt(style, text), n_predict=n_predict, timeout=120,
                                             stop=prompts.STOP)
                    wall = (time.perf_counter() - t) * 1000
                    if rep == 0 and name == next(iter(SAMPLES)):
                        print(f"   first request after start ({style}): {wall:.0f} ms", flush=True)
                    if rep:
                        runs[name].append((wall, data))
                    if prime:  # what local.py does after each refinement, off the latency path
                        server.completion(prefix, n_predict=0, timeout=60)
            for name, text in SAMPLES.items():
                wall, data = sorted(runs[name], key=lambda r: r[0])[len(runs[name]) // 2]
                tm = data.get("timings", {})
                out = normalize(clean_output(data.get("content", "")))
                rows.append(Row(model, got, threads, style, name, len(text.split()), round(wall, 1),
                                int(tm.get("prompt_n", 0)), int(tm.get("cache_n", 0)),
                                round(tm.get("prompt_ms", 0.0), 1), int(tm.get("predicted_n", 0)),
                                round(tm.get("predicted_ms", 0.0), 1), check(text, out) or "ok", out))
    finally:
        server.stop()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="quill-0.8b,quill-2b")
    ap.add_argument("--accels", default="cuda,cpu", help="cuda | vulkan | metal | cpu | auto")
    ap.add_argument("--threads", default=str(ls.default_threads()), help="comma list; CPU thread counts to sweep")
    ap.add_argument("--styles", default="quill", help="quill (Quill's own prompt) and/or shared (prompts.py)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--flash-attn", default="auto")
    ap.add_argument("--prio", type=int, default=0, help="llama-server --prio/--prio-batch (0 normal, 2 high)")
    ap.add_argument("--server-args", default="", help='extra llama-server flags, e.g. "--cache-ram 0"')
    ap.add_argument("--prime", action="store_true", help="re-prime the system-prompt prefix after each request")
    ap.add_argument("--show-output", action="store_true")
    ap.add_argument("--json", type=Path, help="also write all rows to this file")
    a = ap.parse_args()

    rows: list[Row] = []
    for model in a.models.split(","):
        for accel in a.accels.split(","):
            thread_list = [int(t) for t in a.threads.split(",")] if accel == "cpu" else [ls.default_threads()]
            for threads in thread_list:
                try:
                    got = run_config(model, accel, threads, a.styles.split(","), a.reps, a.ctx, a.flash_attn, a.prio,
                                     tuple(a.server_args.split()), a.prime)
                except Exception as e:
                    print(f"   FAILED: {e}", flush=True)
                    continue
                rows += got
                print("| sample | words | wall ms | prompt tok (cached) | prompt ms | gen tok | gen ms | gen tok/s "
                      "| http+other ms | guard |")
                print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
                for r in got:
                    tps = r.predicted_n / r.predicted_ms * 1000 if r.predicted_ms else 0
                    other = r.wall_ms - r.prompt_ms - r.predicted_ms
                    print(f"| {r.style}/{r.sample} | {r.words} | {r.wall_ms:.0f} | {r.prompt_n} ({r.cache_n}) | "
                          f"{r.prompt_ms:.0f} | {r.predicted_n} | {r.predicted_ms:.0f} | {tps:.0f} | {other:.0f} "
                          f"| {r.guard} |")
                if a.show_output:
                    for r in got:
                        print(f"   [{r.style}/{r.sample}] {r.output}")
                para = [r.wall_ms for r in got if r.sample == "paragraph"]
                if para:
                    verdict = "PASS" if max(para) < 1000 else "FAIL"
                    print(f"   paragraph (~100 words): {statistics.median(para):.0f} ms  -> {verdict} (< 1000 ms)")
    if a.json:
        a.json.write_text(json.dumps([asdict(r) for r in rows], indent=1))


if __name__ == "__main__":
    main()

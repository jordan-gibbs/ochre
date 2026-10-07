"""CPU-only refinement latency of one GGUF, run inside the Daytona CPU sandbox
(tools/cloud/owf_daytona.py bench-cpu; docs/refine-cpu-latency.md). Never run on the dev box.

Measures what the app does on a CPU-only machine: llama.cpp b11398 (ubuntu x64 CPU release),
llama-server with the app's CPU flags (openwhisprflow.refine.local_server: --prio 2, --cache-ram 0,
--no-jinja, ngram-simple speculation n=3 m=16, -t = all usable vCPUs), the production prompt and
n_predict from tools/eval/eval_refine.py (checked byte-identical to prompts.rs), and the app's
steady state: the system-prompt prefix primed (off the clock) before each timed /completion.

    python bench_cpu.py --model /workspace/models/x.gguf --eval heldout.jsonl --limit 60 \
        --label 8vcpu --out /workspace/owf/bench-out [--cpus 4] [--threads 4]

--cpus N pins this process (and so llama-server) to N CPUs, hyperthread siblings together
(N/2 cores x 2 threads), to stand in for an N-vCPU machine. --threads defaults to the app's
choice: every usable CPU (Rust's available_parallelism: affinity and cgroup quota).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/eval"))

import eval_refine as ev  # noqa: E402  (production prompt path + parity check)
from openwhisprflow.refine import local_server as ls  # noqa: E402
from openwhisprflow.refine.local import clean_output  # noqa: E402

# crates/ochre-refine/examples/bench.rs "paragraph": the ~100-word sample behind the 1.26 s
# CPU figure in docs/go-checklist.md.
PARAGRAPH = (
    "so um i wanted to give everyone a quick update on the project uh we finished the first round of user "
    "interviews last week and the main thing we heard was that people find the onboarding way too long like they drop off "
    "before they even see the dashboard so uh what we're thinking is we cut it down to three steps instead of seven and we "
    "move the integrations stuff to later you know after they've actually seen some value um i'll share the full notes in "
    "the doc by thursday and if anyone has concerns just let me know before then")


def quota_cpus() -> int | None:
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if q == "max" else max(1, int(int(q) / int(p)))
    except Exception:  # noqa: BLE001
        return None


def usable_cpus() -> int:
    n = len(os.sched_getaffinity(0))
    q = quota_cpus()
    return min(n, q) if q else n


def sibling_order() -> list[int]:
    """Allowed CPUs ordered core by core (siblings adjacent), from lscpu -p=CPU,CORE."""
    allowed = os.sched_getaffinity(0)
    out = subprocess.run(["lscpu", "-p=CPU,CORE"], capture_output=True, text=True).stdout
    pairs = [tuple(map(int, ln.split(","))) for ln in out.splitlines() if ln and not ln.startswith("#")]
    pairs = [(core, cpu) for cpu, core in pairs if cpu in allowed]
    return [cpu for _, cpu in sorted(pairs)]


def proc_mem(pid: int) -> dict:
    out = {}
    for ln in Path(f"/proc/{pid}/status").read_text().splitlines():
        k, _, v = ln.partition(":")
        if k in ("VmRSS", "VmHWM"):
            out[k] = round(int(v.split()[0]) / 1024)   # MiB
    return out


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cpus", type=int, help="pin to N CPUs (siblings together)")
    ap.add_argument("--threads", type=int, help="default: every usable CPU, as the app does")
    ap.add_argument("--para-reps", type=int, default=5)
    a = ap.parse_args()

    if a.cpus:
        os.sched_setaffinity(0, set(sibling_order()[: a.cpus]))
    threads = a.threads or usable_cpus()
    exe = Path(os.environ["OWF_LLAMA_BIN_DIR"]) / "llama-server"
    rows = ev.load_eval(a.eval, a.limit)
    ev.check_prompt_parity(rows)
    sysm = ev.System("bench", "local", a.model.name, "shared", "")
    name = f"{a.model.stem}-{a.label}"
    a.out.mkdir(parents=True, exist_ok=True)

    dropped = subprocess.run("sync && echo 3 > /proc/sys/vm/drop_caches", shell=True,
                             capture_output=True).returncode == 0
    server = ls.LlamaServer(exe, a.model, accel="cpu", options=ls.ServerOptions(threads=threads),
                            log_path=a.out / f"llama-server-{name}.log")
    meta = {"name": name, "model": a.model.name, "model_mb": round(a.model.stat().st_size / 1e6),
            "label": a.label, "threads": threads, "affinity": sorted(os.sched_getaffinity(0)),
            "cgroup_quota_cpus": quota_cpus(), "page_cache_dropped": dropped,
            "args": " ".join(server.args()[3:]).replace(str(a.model), a.model.name)}
    # Cold start, as LocalRefiner::load: spawn -> /health (model loaded) -> first refinement -> prime.
    t0 = time.perf_counter()
    server.start(timeout=300)
    t_health = time.perf_counter()
    warm = {"raw": "okay so this is a warm up", "mode": "clean", "style": "", "dictionary": []}
    p, prefix = ev.local_prompt(sysm, warm)
    server.completion(p, n_predict=ev.n_predict(warm["raw"]), timeout=300, stop=list(ev.prompts.STOP))
    t_first = time.perf_counter()
    server.completion(prefix, n_predict=0, timeout=300)
    t_primed = time.perf_counter()
    meta.update(cold_health_ms=round((t_health - t0) * 1000), cold_first_response_ms=round((t_first - t0) * 1000),
                cold_ready_ms=round((t_primed - t0) * 1000), prime_ms=round((t_primed - t_first) * 1000))
    recs = []
    try:
        for k, ex in enumerate(rows):
            prompt, prefix = ev.local_prompt(sysm, ex)
            server.completion(prefix, n_predict=0, timeout=300)          # production re-primes; off the clock
            t = time.perf_counter()
            data = server.completion(prompt, n_predict=ev.n_predict(ex["raw"]), timeout=300, stop=list(ev.prompts.STOP))
            wall = (time.perf_counter() - t) * 1000
            recs.append({"id": ex["id"], "words": len(ex["raw"].split()), "wall_ms": round(wall, 1),
                         "output": clean_output(data.get("content", "")), "timings": data.get("timings", {})})
            print(f"  [{name}] {k + 1}/{len(rows)} {ex['id']} {recs[-1]['words']}w {wall:.0f} ms", flush=True)
        para = []
        ctx = {"raw": PARAGRAPH, "mode": "clean", "style": "", "dictionary": []}
        prompt, prefix = ev.local_prompt(sysm, ctx)
        for _ in range(a.para_reps):
            server.completion(prefix, n_predict=0, timeout=300)
            t = time.perf_counter()
            data = server.completion(prompt, n_predict=ev.n_predict(PARAGRAPH), timeout=300, stop=list(ev.prompts.STOP))
            para.append({"wall_ms": round((time.perf_counter() - t) * 1000, 1), "timings": data.get("timings", {}),
                         "output": clean_output(data.get("content", ""))})
        meta["mem_mib"] = proc_mem(server.proc.pid)
    finally:
        server.stop()

    walls = [r["wall_ms"] for r in recs]
    gen_n = sum(r["timings"].get("predicted_n", 0) for r in recs)
    gen_ms = sum(r["timings"].get("predicted_ms", 0) for r in recs)
    pr_n = sum(r["timings"].get("prompt_n", 0) for r in recs)
    pr_ms = sum(r["timings"].get("prompt_ms", 0) for r in recs)
    dn = sum(r["timings"].get("draft_n", 0) or 0 for r in recs)
    da = sum(r["timings"].get("draft_n_accepted", 0) or 0 for r in recs)
    s = {**meta, "rows": len(recs), "words_median": statistics.median(r["words"] for r in recs),
         "wall_median_ms": round(statistics.median(walls)), "wall_p90_ms": round(pct(walls, 0.9)),
         "wall_max_ms": round(max(walls)), "wall_mean_ms": round(statistics.mean(walls)),
         "gen_tok_s": round(gen_n / gen_ms * 1000, 1) if gen_ms else None,
         "gen_tokens_median": statistics.median(r["timings"].get("predicted_n", 0) for r in recs),
         "prompt_new_tokens_median": statistics.median(r["timings"].get("prompt_n", 0) for r in recs),
         "prompt_ms_median": round(statistics.median(r["timings"].get("prompt_ms", 0) for r in recs)),
         "prompt_tok_s": round(pr_n / pr_ms * 1000, 1) if pr_ms else None,
         "draft_accept": round(da / dn, 3) if dn else None,
         "paragraph_words": len(PARAGRAPH.split()),
         "paragraph_median_ms": round(statistics.median(p["wall_ms"] for p in para)),
         "paragraph_prompt_ms": round(statistics.median(p["timings"].get("prompt_ms", 0) for p in para)),
         "paragraph_gen_ms": round(statistics.median(p["timings"].get("predicted_ms", 0) for p in para)),
         "paragraph_gen_tok_s": round(statistics.median(p["timings"].get("predicted_per_second", 0) for p in para), 1),
         "paragraph_output": para[0]["output"]}
    (a.out / f"rows-{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                                               encoding="utf-8")
    (a.out / f"summary-{name}.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k not in ("args", "affinity", "paragraph_output")}), flush=True)


if __name__ == "__main__":
    main()

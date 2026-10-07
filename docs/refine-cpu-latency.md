# Refine latency on CPU only (cloud-measured, 2026-10-05)

How fast the round-3 cleanup models (`v3-0.8b`, `v3-2b`, Q4_K_M) refine a dictation on a
machine with **no GPU**, measured on Daytona CPU sandboxes (nothing run on the dev box).
Job: `tools/cloud/owf_daytona.py bench-cpu` (docs/cloud-compute.md), runs
`bench-cpu-1005-111735-64e8` (4 vCPU) and `bench-cpu-1005-111733-655c` (8 vCPU).

## Results

Steady-state per-dictation wall time (HTTP `/completion` round trip, system prompt already
primed), first 60 rows of `real-v3/final/heldout.jsonl` (median 17.5 words, 21 generated tokens,
~41 new prompt tokens). "Paragraph" = the ~100-word sample of
`crates/ochre-refine/examples/bench.rs`, median of 5. `-t` = the app's choice (all vCPUs) unless
noted.

| model | sandbox (CPU) | -t | median | p90 | max | gen tok/s | prompt eval (median) | paragraph ~100 w | cold: health / first response / ready | RAM (peak RSS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 4 vCPU (EPYC 9254, Zen 4) | 4 | **654 ms** | 1632 | 3209 | 47 | 194 ms | 2.6 s | 0.8 / 2.2 / 3.3 s | 1.04 GB |
| 0.8B | 8 vCPU (EPYC 7543, Zen 3) | 8 | **838 ms** | 2185 | 3490 | 37 | 225 ms | 3.0 s | 0.9 / 2.6 / 3.9 s | 1.04 GB |
| 2B | 4 vCPU (EPYC 9254, Zen 4) | 4 | **1237 ms** | 3451 | 6156 | 26 | 345 ms | 5.0 s | 1.2 / 3.9 / 6.1 s | 2.13 GB |
| 2B | 8 vCPU (EPYC 7543, Zen 3) | 8 | **1486 ms** | 4153 | 6877 | 22 | 416 ms | 4.9 s | 1.7 / 5.1 / 7.6 s | 2.13 GB |

Thread-count checks (same sandboxes):

| model | config | median | p90 | gen tok/s | paragraph |
|---|---|---:|---:|---:|---:|
| 0.8B | 4 vCPU, `-t 2` | 983 | 2682 | 31 | 4.5 s |
| 0.8B | 8 vCPU, `-t 4` (physical-core count) | 1212 | 3083 | 26 | 4.4 s |
| 0.8B | 8 vCPU host pinned to 4 CPUs, `-t 4` | 1338 | 3502 | 22 | 5.6 s |
| 2B | 4 vCPU, `-t 2` | 2124 | 6062 | 15 | 8.4 s |
| 2B | 8 vCPU, `-t 4` | 2360 | 6582 | 13 | 9.6 s |
| 2B | 8 vCPU host pinned to 4 CPUs, `-t 4` | 2765 | 7906 | 11 | 11.3 s |

- **The app's thread rule (all vCPUs) is right** on these hosts: halving threads costs 1.5-1.6x.
- Generation dominates. ngram speculation accepted 31-33% of drafted tokens. The paragraph
  splits about 0.45 s prompt / 2.2-2.6 s generation (0.8B), 0.85-1 s / 4-4.2 s (2B).
- **Prime cost.** After each dictation the app re-primes the ~400-token system prompt in the
  background: 1.1-1.3 s for 0.8B and 2.2-2.5 s for 2B at the app's thread count. A dictation that
  starts before that finishes waits for it.
- **Cold start** = spawn llama-server to `/health`, then the first refinement (the warm-up), then
  the prime (`LocalRefiner::load`). Page cache was warm (the GGUF had just been uploaded; dropping
  caches is not allowed in the sandbox), so a first launch from disk will be slower.
- RAM is the llama-server process only (VmHWM): about 2x the GGUF size, at `-c 2048`.
- Output sanity: both models cleaned the paragraph correctly. Quality is scored elsewhere
  (docs/refine-finetune-v3.md); per-row outputs are in the run directories.

## Caveats when comparing with a desktop

- These are **cloud vCPUs**: a cgroup quota (`cpu.max` 4 or 8 CPUs) on a 48- to 60-CPU EPYC host,
  with other tenants. These EPYCs boost to 3.7 (7543) and 4.15 GHz (9254). A desktop Zen 4/5 boosts to about
  5.2-5.5 GHz and has far more memory bandwidth per thread. On the 4 vCPU host the vCPUs are SMT
  siblings, so 4 vCPU = 2 cores. The 8 vCPU guest reports 1 thread per core, but that is the VM's
  view, not proof that it has 8 real cores.
- The two shapes ran on **different CPU generations**. The 8 vCPU shape needs a GPU host (Daytona
  caps CPU-only sandboxes at 4 vCPU), and that host had a Zen 3 EPYC 7543 without AVX-512. The 4 vCPU
  shape got a Zen 4 EPYC 9254 with AVX-512 and VNNI. As a result, 4 vCPU beat 8 vCPU here. Per CPU,
  pinning the 8 vCPU host to 4 CPUs shows Zen 3 at roughly half the speed of Zen 4 (0.8B 1338 vs 654 ms).
- Reference point: Quill 0.8B-class on the dev box's Ryzen 7 9800X3D (8C/16T, CPU only, under
  load) did the same ~100-word paragraph in **~1.26 s** (docs/go-checklist.md). Here it takes
  2.6-3.0 s. Read these numbers as a **typical mid-range laptop or a busy older desktop**, not a
  current desktop. A recent 8-core desktop should land between the two.
- With a median 17.5-word dictation, the 0.8B is at **0.65-0.85 s** on CPU. Long dictations of
  60+ words take 2-3.5 s. The 2B costs about 2x on every metric, so on CPU-only machines it is
  only acceptable for short dictations.

## Method

- llama.cpp **b11398**, official `llama-b11398-bin-ubuntu-x64.tar.gz` (CPU backends, dynamic
  ggml-cpu variant per microarchitecture), Ubuntu 24.04.
- llama-server with the app's CPU flags, built by `openwhisprflow.refine.local_server` (the Python
  twin of `crates/ochre-refine/src/local/server.rs`): `-c 2048 -np 1 -t <vCPUs> -ngl 0 -fa auto
  -b 1024 --no-webui --no-jinja --cache-prompt --no-context-shift --fit off --cache-ram 0 --prio 2
  --prio-batch 2 --spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 16`.
  `-t` is the cgroup quota, the same as Rust's `available_parallelism` (`nproc` shows the host's
  CPUs).
- Prompt and `n_predict` come from `tools/eval/eval_refine.py`'s production path. The prompts were
  checked byte-identical to `crates/ochre-refine/src/prompts.rs` at job start. Each row primes its
  own system-prompt prefix (off the clock, as the app does after every request), then a timed greedy
  `/completion` with the production stop strings. "gen tok/s" = total predicted tokens / total
  predicted ms over the 60 rows. "Prompt eval" = server `prompt_ms` for the uncached tokens.
- Script: `tools/cloud/sandbox/bench_cpu.py` (+ `bench_cpu_job.sh`). Raw data:
  `training/refine/data/cloud-runs/<run>/out/` (gitignored).
- Cost: about $1.3 at on-demand rates for the two good runs plus two failed attempts (a missing
  numpy in the image). The 8 vCPU runs pay for an idle RTX 5090 spot GPU. Most of the wall time
  was uploading the 1.8 GB of GGUFs at about 2 MB/s.

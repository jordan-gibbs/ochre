# Cloud compute (Daytona GPU spot sandboxes)

`tools/cloud/owf_daytona.py` runs the heavy jobs (TTS + Parakeet data rendering, LoRA training,
GGUF export, eval) on a Daytona RTX 5090 / 4090 **spot** sandbox, so this machine only uploads a
small bundle and downloads results. One job = one sandbox.

Interpreter: anything with the `daytona` SDK, e.g.
`uv venv .venv-cloud && uv pip install --python .venv-cloud daytona` (gitignored).

## Rules it enforces

- `DAYTONA_API_KEY` is read at run time from `./.env` or `$OWF_CLOUD_ENV` (or `--env-file`) into the process
  only; never printed or written. The HF token (`~/.cache/huggingface/token`) goes to the sandbox
  as an env var only.
- Preflight proves the key is bound to the expected organization (by id, `OWF_DAYTONA_ORG_ID`).
- Spot only, `RTX-5090` then `RTX-4090`; no capacity is an error, never on-demand. (`bench-cpu`
  without `--gpu-host` takes a plain on-demand CPU sandbox, no GPU.)
- Every sandbox is labelled `managed-by=owf-cloud`, `owf-run=<run-id>`; only sandboxes with both
  labels are ever stopped or deleted (the organization has sandboxes that are not ours).
- Hard TTL fuse (`--ttl-minutes`), no idle stop, delete on stop; `--yes` required and the launch
  refuses when the ceiling (TTL x on-demand price) exceeds `--max-cost` (default $5).
- Whatever happens (error, Ctrl-C, eviction) the sandbox is stopped, deleted and verified gone.
- Run records, logs, bundles and raw outputs: `training/refine/data/cloud-runs/<run-id>/` (gitignored).

## Commands

```powershell
$PY = ".venv-cloud\Scripts\python.exe"
& $PY tools/cloud/owf_daytona.py preflight
& $PY tools/cloud/owf_daytona.py list                         # live owf sandboxes + recent runs

# Render script files (TTS -> augment -> Parakeet -> strip_hesitations -> assemble)
& $PY tools/cloud/owf_daytona.py render --files W9 W10 W11 --per-file 3 --yes     # pilot, not applied
& $PY tools/cloud/owf_daytona.py render --files W9 W10 W11 --apply --yes          # rows -> datasets/real-v2/W*.jsonl, then packets.py

# Train + export Q4_K_M + eval (several can run at once, one sandbox each)
& $PY tools/cloud/owf_daytona.py train --run v3-2b --yes
& $PY tools/cloud/owf_daytona.py train --run v3-4b --base 4b --epochs 3 --lora-r 32 --lora-alpha 64 `
      --train-file datasets/real-v2/final/train.jsonl --train-file "datasets/real-v3/base/*.jsonl" `
      --eval synth=datasets/synth-v1/eval.jsonl --eval heldout=datasets/real-v2/final/heldout.jsonl --yes

# CPU-only refine latency (llama.cpp CPU build, app flags; docs/refine-cpu-latency.md)
& $PY tools/cloud/owf_daytona.py bench-cpu --cpu 4 --yes                          # CPU-only sandbox (max 4 vCPU)
& $PY tools/cloud/owf_daytona.py bench-cpu --cpu 8 --memory 16 --gpu-host --yes   # 8 vCPU: needs a GPU host

& $PY tools/cloud/owf_daytona.py fetch <run-id> [--apply]      # if the launcher died after the job finished
& $PY tools/cloud/owf_daytona.py stop <run-id>                 # end one run's sandbox (verified)
& $PY tools/cloud/owf_daytona.py stop-all-managed              # every owf-cloud sandbox (lists, asks)
```

Outputs are downloaded when the job ends: `render` rows to
`training/refine/data/cloud-runs/<run-id>/out/` (and, with `--apply`, into
`datasets/real-v2/` + audit packets via `packets.py`); `train` to `training/refine/output/<run>/`
(`adapter/`, `gguf/<run>-Q4_K_M.gguf`, `gguf/eval-*.md`, `export_summary.json`,
`train_summary.json`, `eval/refine-*.jsonl`, `eval/failures-*.md`, llama-server logs).

`train` parameters: `--base 2b|4b|0.8b` (`Qwen/Qwen3.5-*` at pinned revisions, see `BASES`),
`--epochs`, `--lora-r`, `--lora-alpha`, `--train-file` (repeatable, globs, relative to
`training/refine/`), `--eval name=path` (repeatable; `synth` is scored next to Quill 2B's saved
outputs and named `<run>-Q4_K_M` like the local runs, others `<run>-<name>`), `--quants`,
`--set key=value` (any train.py config key), `--no-rust-check`.

`bench-cpu` uploads the GGUFs as-is (`--models v3-0.8b v3-2b`, default; ~2 MB/s from here, so
~10 min for both), downloads llama.cpp b11398's official `ubuntu-x64` CPU release in the sandbox
and runs `tools/cloud/sandbox/bench_cpu.py` per model x config (`--configs label:cpus:threads`;
default: all vCPUs with the app's thread count, half the threads, and on an 8 vCPU host a 4-CPU
pinned run). It uses eval_refine.py's production prompt path (parity-checked against prompts.rs),
the app's CPU llama-server flags and prime-then-complete loop on the first `--limit 60` rows of
`--eval` (real-v3 held-out by default), plus the ~100-word paragraph of
`crates/ochre-refine/examples/bench.rs`. Outputs: `training/refine/data/cloud-runs/<run>/out/`
(`summary-*.json`, `rows-*.jsonl`, `lscpu.txt`, llama-server logs) and a table on stdout.

Daytona limits on this org: a **CPU-only sandbox is capped at 4 vCPU and 10 GB disk**, so the
8 vCPU shape is taken as a GPU spot sandbox (`--gpu-host`; the GPU is never touched and is paid
for). vCPUs are a cgroup quota (`cpu.max`) over a big host (`nproc` shows 48-60 CPUs); the bench
uses the quota as the app's thread count, as Rust's `available_parallelism` does.

## What differs from the local pipeline

- **Render: no Windows SAPI** on Linux. Clips planned for SAPI (6%) go to Piper or Kokoro in
  proportion to their effective shares (`common.NO_SAPI`; every other clip's plan is unchanged).
  Qwen3-TTS is unaffected: `QWEN_TO_KOKORO` already sends its clips to Kokoro locally too.
- Same model files (Piper LibriTTS `.pt`, Kokoro v1.0, Parakeet v3 int8 at the app's pinned
  revision) and the same 270 room IRs + 774 MUSAN noise files, downloaded in the image from their
  upstream sources and checked at job start against local SHA-256s and
  `tools/cloud/lkw_manifest.tsv` (paths, sizes, draw order). espeak-ng is Debian's build.
- The render job skips `parity.py` (needs cargo) and runs `packets.py` locally after `--apply`
  (batch numbering needs every W file).
- Train: llama.cpp b11398 is the official `ubuntu-cuda` release (the app ships the `win-cuda`
  build of the same tag); flash-linear-attention uses Linux triton instead of triton-windows.
  `check_rust` builds the Rust renderer in the sandbox (cargo), as on Windows.

- Fresh GPU hosts JIT-compile CUDA kernels on first use: each TTS process spends ~3.5 min loading
  before its first clip (Piper then runs at ~20x realtime). This is a fixed cost per job.
- Images: render = Debian slim + Python 3.12; train = Ubuntu 24.04 (llama.cpp's ubuntu-cuda
  binaries need glibc 2.38). Both are built by Daytona on first use (~5-10 min, not billed as
  sandbox time) and then cached.

## Measured runs (2026-10-05, RTX 5090 spot unless noted)

| run | sandbox | job | end to end | billed (est.) | cost ceiling for that time |
|---|---|---:|---:|---:|---:|
| render pilot, W9-W11 `--per-file 3` (9 rows) | 8 vCPU / 32 GiB | 9.4 min | 11.2 min | 9.4 min | $0.26 |
| render W9-W11, 600 rows (`--cpu 16 --parakeet-workers 4 --parakeet-threads 4`) | 16 vCPU / 32 GiB | 11.5 min | 11.6 min | 11.5 min | $0.40 |
| train `v2-real-cloud` (2B, real-v2/final/train, defaults) + export + 2 evals | 8 vCPU / 48 GiB | 18.2 min | 23.5 min | 19.5 min | $0.63 |
| bench-cpu 0.8B + 2B, 2 configs (CPU-only, no GPU) | 4 vCPU / 8 GiB | 23.0 min (13 upload) | 23.6 min | 23.0 min | $0.13 |
| bench-cpu 0.8B + 2B, 3 configs (`--gpu-host`, GPU idle) | 8 vCPU / 16 GiB | 33.2 min (13 upload) | 33.7 min | 33.2 min | $0.78 |

Costs are wall time x on-demand rate (spot is billed lower; Daytona publishes no spot price).

`v2-real-cloud` vs the local `v2-real` (docs/refine-finetune-v2.md): 238 steps both; eval loss
0.3637 -> 0.1131 (local 0.3647 -> 0.1135); synth-v1 eval WER vs clean 0.036 (local 0.056), EM(norm)
44% (45%), guard 0% (1%); real-v2 held-out WER vs clean 0.034 (local 0.031), EM(norm) 51% (52%),
guard 2% (2%), dropped 14% (13%). Same range.

W9-W11 render: 600/600 rows, Parakeet WER 0.075 micro (Piper 0.096, Kokoro 0.045), auto audit
ok 399 / check 103 / suspect 98; voices 359 Piper / 241 Kokoro (SAPI share redistributed).

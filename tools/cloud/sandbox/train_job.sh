#!/usr/bin/env bash
# Inside the Daytona training sandbox (tools/cloud/owf_daytona.py train): train -> export -> eval,
# the same scripts and order as docs/refine-finetune-v2.md "Reproduce", then pack the results
# (adapter, GGUFs except bf16, summaries, eval outputs) into /workspace/out.tar.gz.
# Parameters come from /workspace/owf/tools/cloud/sandbox/job.env (written by the launcher).
set -euo pipefail
cd /workspace/owf
source tools/cloud/sandbox/job.env
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PATH="/opt/cargo/bin:$PATH" RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
PY=/opt/venv-train/bin/python
T0=$(date +%s)
stamp() { echo "== [$(( $(date +%s) - T0 ))s] $*"; }

stamp "sandbox: $(nproc) vCPU, $(free -g | awk '/Mem:/{print $2}') GiB RAM, run $RUN base $BASE_REPO@$BASE_REV"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
[ -n "${HF_TOKEN:-}" ] && echo "HF token: present" || echo "HF token: none"

# 1. llama.cpp b11398, the app's build, as the ubuntu CUDA release (+ its CUDA runtime libs).
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -ge 580 ]; then CU=13.4; else CU=12.8; fi
L=/workspace/llama; mkdir -p $L
REL=https://github.com/ggml-org/llama.cpp/releases/download/b11398
curl -fsSL -o $L/bin.tgz $REL/llama-b11398-bin-ubuntu-cuda-$CU-x64.tar.gz & P1=$!
curl -fsSL -o $L/rt.tgz $REL/cudart-llama-b11398-bin-ubuntu-cuda-$CU-x64.tar.gz & P2=$!
# 2. base model at the pinned revision (in parallel with the downloads above)
$PY - <<EOF &
from huggingface_hub import snapshot_download
snapshot_download("$BASE_REPO", revision="$BASE_REV", local_dir="training/refine/data/base/$BASE_NAME")
EOF
P3=$!
# 3. the Rust prompt renderer for train.py's byte-parity check (check_rust)
# The refine crate was renamed owf-refine -> ochre-refine; build whichever exists (render.py only
# builds when the binary is missing) and keep crates/owf-refine/src/prompts.rs readable for the eval.
REFINE_CRATE=$(basename "$(ls -d crates/*-refine | head -1)")
if [ "$RUST_CHECK" = "true" ]; then
  CARGO_TARGET_DIR=target/finetune cargo build -q -p "$REFINE_CRATE" --example render
fi
# after the build: a crates/owf-refine link would be a second workspace member for cargo
[ -e crates/owf-refine ] || ln -s "$REFINE_CRATE" crates/owf-refine
wait $P1; wait $P2; wait $P3
(cd $L && tar xzf bin.tgz && tar xzf rt.tgz && rm -f bin.tgz rt.tgz)
BIN=$(dirname "$(find $L -name llama-server -type f | head -1)")
export OWF_LLAMA_BIN_DIR=$BIN
export LD_LIBRARY_PATH="$(find $L -name 'lib*.so*' -printf '%h\n' | sort -u | paste -sd:):${LD_LIBRARY_PATH:-}"
"$BIN/llama-server" --version 2>&1 | head -3 || true
stamp "setup done (llama.cpp ubuntu-cuda-$CU)"

# 4. train, 5. export, 6. eval (docs/refine-finetune-v2.md "Reproduce")
cd training/refine
$PY train.py configs/qwen35-2b-lora.yaml --set run_name=$RUN --set base_model=data/base/$BASE_NAME \
    --set base_repo=$BASE_REPO --set base_revision=$BASE_REV --set epochs=$EPOCHS \
    --set lora_r=$LORA_R --set lora_alpha=$LORA_ALPHA --set "train_files=$TRAIN_FILES" \
    --set check_rust=$RUST_CHECK "${EXTRA_SETS[@]}"
stamp "train done"
$PY export.py output/$RUN/adapter --base data/base/$BASE_NAME --quants $QUANTS
stamp "export done"
cd /workspace/owf
GGUF=training/refine/output/$RUN/gguf/$RUN-Q4_K_M.gguf
NAMES=()
for spec in "${EVALS[@]}"; do
  key=${spec%%=*}; path=${spec#*=}
  if [ "$key" = "synth" ]; then name="$RUN-Q4_K_M"; base=quill-2b-shared; else name="$RUN-$key"; base=none; fi
  $PY training/refine/eval_gguf.py $GGUF --name "$name" --eval "training/refine/$path" --baselines $base
  NAMES+=("$name")
  stamp "eval $name done"
done

# 7. pack
mkdir -p training/refine/output/$RUN/eval
for n in "${NAMES[@]}"; do
  for f in refine-$n.jsonl refine-baseline-$n.jsonl failures-$n.md llama-server-$n.log; do
    if [ -f tools/eval/out/$f ]; then cp tools/eval/out/$f training/refine/output/$RUN/eval/; fi
  done
done
cp tools/eval/out/meta.json training/refine/output/$RUN/eval/meta.json 2>/dev/null || true
cd training/refine/output
tar czf /workspace/out.tar.gz --exclude='*-bf16.gguf' --exclude='checkpoint-*' --exclude='merged-hf*' $RUN
stamp "OWF_OUT_READY $(stat -c %s /workspace/out.tar.gz)"

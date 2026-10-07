#!/usr/bin/env bash
# Inside a Daytona GPU sandbox (tools/cloud/owf_daytona.py eval): run training/refine/eval_gguf.py
# (production prompt path, llama-server b11398 with the app's flags) for existing GGUFs on one or
# more eval sets, then pack the per-row outputs into /workspace/out.tar.gz.
# Parameters come from tools/cloud/sandbox/job.env (written by the launcher):
#   MODELS=(name=hf:<repo>/<file> | name=/workspace/models/<file> ...)  EVALS=(key=path ...)
#   CHUNKS=(off on)   optional: "on" also runs eval_gguf --chunk as <name>-chunk (docs/refine-chunking.md)
set -euo pipefail
cd /workspace/owf
source tools/cloud/sandbox/job.env
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/opt/venv-train/bin/python
T0=$(date +%s)
stamp() { echo "== [$(( $(date +%s) - T0 ))s] $*"; }
stamp "sandbox: $(nproc) vCPU, $(free -g | awk '/Mem:/{print $2}') GiB RAM"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
[ -n "${HF_TOKEN:-}" ] && echo "HF token: present" || echo "HF token: none"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -ge 580 ]; then CU=13.4; else CU=12.8; fi
L=/workspace/llama; mkdir -p $L /workspace/models
REL=https://github.com/ggml-org/llama.cpp/releases/download/b11398
curl -fsSL -o $L/bin.tgz $REL/llama-b11398-bin-ubuntu-cuda-$CU-x64.tar.gz & P1=$!
curl -fsSL -o $L/rt.tgz $REL/cudart-llama-b11398-bin-ubuntu-cuda-$CU-x64.tar.gz & P2=$!
# GGUFs from the private HF repos (fast inside the cloud; the token is only in the env)
for spec in "${MODELS[@]}"; do
  src=${spec#*=}
  if [[ "$src" == hf:* ]]; then
    ref=${src#hf:}; repo=${ref%/*}; file=${ref##*/}
    $PY -c "from huggingface_hub import hf_hub_download as d; d('$repo', '$file', local_dir='/workspace/models')"
  fi
done
wait $P1; wait $P2
(cd $L && tar xzf bin.tgz && tar xzf rt.tgz && rm -f bin.tgz rt.tgz)
BIN=$(dirname "$(find $L -name llama-server -type f | head -1)")
export OWF_LLAMA_BIN_DIR=$BIN
export LD_LIBRARY_PATH="$(find $L -name 'lib*.so*' -printf '%h\n' | sort -u | paste -sd:):${LD_LIBRARY_PATH:-}"
REFINE_CRATE=$(basename "$(ls -d crates/*-refine | head -1)")
[ -e crates/owf-refine ] || ln -s "$REFINE_CRATE" crates/owf-refine
ls -la /workspace/models
stamp "setup done"

mkdir -p /workspace/evalout
for spec in "${MODELS[@]}"; do
  name=${spec%%=*}; src=${spec#*=}
  if [[ "$src" == hf:* ]]; then gguf=/workspace/models/${src##*/}; else gguf=$src; fi
  # eval_gguf registers the system under the GGUF's directory; give each model its own dir/name
  d=/workspace/m/$name; mkdir -p $d; ln -f "$gguf" $d/$name-Q4_K_M.gguf 2>/dev/null || cp "$gguf" $d/$name-Q4_K_M.gguf
  for e in "${EVALS[@]}"; do
    key=${e%%=*}; path=${e#*=}
    for ch in "${CHUNKS[@]:-off}"; do
      if [ "$ch" = on ]; then sys=$name-chunk-$key; flag=--chunk; else sys=$name-$key; flag=; fi
      $PY training/refine/eval_gguf.py $d/$name-Q4_K_M.gguf --name "$sys" --eval "training/refine/$path" --baselines none $flag
      for f in refine-$sys.jsonl failures-$sys.md llama-server-$sys.log; do
        [ -f tools/eval/out/$f ] && cp tools/eval/out/$f /workspace/evalout/ || true
      done
      cp $d/eval-$sys.md /workspace/evalout/ 2>/dev/null || true
      stamp "eval $sys done"
    done
  done
done
cd /workspace && tar czf /workspace/out.tar.gz evalout
stamp "OWF_OUT_READY $(stat -c %s /workspace/out.tar.gz)"

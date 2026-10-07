#!/usr/bin/env bash
# Inside the Daytona CPU sandbox (tools/cloud/owf_daytona.py bench-cpu): llama.cpp b11398's official
# ubuntu x64 CPU release, then bench_cpu.py per model x config, results packed into
# /workspace/out.tar.gz. Parameters from tools/cloud/sandbox/job.env (MODELS, EVAL, LIMIT, CONFIGS).
set -euo pipefail
cd /workspace/owf
source tools/cloud/sandbox/job.env
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=/opt/venv-bench/bin/python
OUT=/workspace/owf/bench-out; mkdir -p $OUT
T0=$(date +%s)
stamp() { echo "== [$(( $(date +%s) - T0 ))s] $*"; }

stamp "sandbox: nproc $(nproc), cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a), $(free -m | awk '/Mem:/{print $2}') MiB RAM"
lscpu | tee $OUT/lscpu.txt | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|L3|Flags" | cut -c1-200
free -m > $OUT/free.txt; cat /sys/fs/cgroup/cpu.max > $OUT/cpu.max 2>/dev/null || true

L=/workspace/llama; mkdir -p $L
curl -fsSL -o $L/bin.tgz https://github.com/ggml-org/llama.cpp/releases/download/b11398/llama-b11398-bin-ubuntu-x64.tar.gz
(cd $L && tar xzf bin.tgz && rm -f bin.tgz)
BIN=$(dirname "$(find $L -name llama-server -type f | head -1)")
export OWF_LLAMA_BIN_DIR=$BIN
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
"$BIN/llama-server" --version 2>&1 | tee $OUT/llama-version.txt | head -3
ls $BIN | grep -i ggml > $OUT/ggml-backends.txt || true
stamp "setup done"

for m in $MODELS; do
  for c in $CONFIGS; do        # label:cpus:threads (0 = all usable / app default)
    IFS=: read -r label cpus threads <<< "$c"
    extra=()
    [ "$cpus" != 0 ] && extra+=(--cpus "$cpus")
    [ "$threads" != 0 ] && extra+=(--threads "$threads")
    stamp "$m $label"
    $PY tools/cloud/sandbox/bench_cpu.py --model /workspace/models/$m --eval "$EVAL" --limit "$LIMIT" \
        --label "$label" --out $OUT "${extra[@]}"
  done
done
grep -h "CPU :\|system_info\|load_backend" $OUT/llama-server-*.log | sort -u | head -20 || true
stamp "bench done"
tar czf /workspace/out.tar.gz -C $OUT .

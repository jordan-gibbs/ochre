#!/usr/bin/env bash
# Inside the Daytona render sandbox (tools/cloud/owf_daytona.py render). Runs the real-v2 audio
# pipeline (tools/refine-data/run.py) on the selected script files, then packs the assembled rows,
# the report and the per-clip metadata (no audio) into /workspace/out.tar.gz.
#   render_job.sh "<W files>" <per-file or 0> <piper> <kokoro> <kokoro-threads> <parakeet> <parakeet-threads>
set -euo pipefail
FILES="$1"; PER_FILE="$2"
cd /workspace/owf
export OWF_LKW_DIR=/workspace/lkw LOCALAPPDATA=/opt/lappdata OWF_NO_SAPI=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p /workspace/lkw
mkdir -p training/refine/data
ln -sfn /opt/models/tts-models training/refine/data/tts-models
ln -sfn /opt/venv-kgpu .venv-kgpu
PY=/opt/venv-data/bin/python
# onnxruntime-gpu (Kokoro) loads CUDA 12 / cuDNN 9 from the torch wheels' nvidia packages.
export LD_LIBRARY_PATH="$(ls -d /opt/venv-data/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:):${LD_LIBRARY_PATH:-}"

echo "== sandbox: $(nproc) vCPU, $(free -g | awk '/Mem:/{print $2}') GiB RAM"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# Same sets livekit-wakeword's setup downloaded into the local lkw/ (pinned revisions; HF_TOKEN from the
# sandbox env avoids anonymous rate limits). Checked file-for-file against the local manifest next.
$PY -c '
from huggingface_hub import snapshot_download as s
s("davidscripka/MIT_environmental_impulse_responses", repo_type="dataset", max_workers=16,
  revision="b824a1ef2821f112fda0b9cb26e4278c62b425bb", allow_patterns="16khz/*.wav", local_dir="/workspace/lkw/rirs")
s("FluidInference/musan", repo_type="dataset", max_workers=16, revision="3edcfdf89b56dbe6a395ff29f9c29489e03d1321",
  allow_patterns="noise/**/*.wav", local_dir="/workspace/lkw/backgrounds")
'
rm -rf /workspace/lkw/rirs/.cache /workspace/lkw/backgrounds/.cache
ln -sfn /opt/lkw/piper /workspace/lkw/piper
$PY tools/cloud/sandbox/check_render_env.py $FILES

SEL=(--files $FILES)
if [ "$PER_FILE" != "0" ]; then SEL+=(--per-file "$PER_FILE"); fi
t0=$(date +%s)
$PY tools/refine-data/run.py "${SEL[@]}" --sapi-workers 0 --qwen-workers 0 --post assemble \
    --piper-workers "$3" --kokoro-workers "$4" --kokoro-threads "$5" \
    --parakeet-workers "$6" --parakeet-threads "$7"
wall=$(( $(date +%s) - t0 ))
$PY tools/refine-data/report.py "${SEL[@]}" --wall-s "$wall" > /dev/null
echo "== TTS + recognition + assemble wall ${wall}s"
for f in training/refine/data/logs-v2/*.log; do
  if grep -q "FAILED\|Traceback" "$f"; then echo "== problems in $f:"; grep -n "FAILED\|Error" "$f" | tail -5 || true; fi
done
wc -l training/refine/datasets/real-v2/[WE]*.jsonl
tar czf /workspace/out.tar.gz training/refine/datasets/real-v2/[WE]*.jsonl training/refine/datasets/real-v2/REPORT.md \
    training/refine/data/logs-v2 training/refine/data/asr-v2 \
    $(ls training/refine/data/audio-v2/tts/*.json)
echo "OWF_OUT_READY $(stat -c %s /workspace/out.tar.gz)"

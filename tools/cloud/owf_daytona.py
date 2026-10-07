#!/usr/bin/env python
"""Run Open Whisperflow's heavy jobs (TTS + ASR data rendering, LoRA training + GGUF export +
eval) on Daytona GPU spot sandboxes, so none of it runs on this machine. docs/cloud-compute.md.

    python tools/cloud/owf_daytona.py preflight
    python tools/cloud/owf_daytona.py render --files W9 W10 W11 [--per-file 3] [--apply] --yes
    python tools/cloud/owf_daytona.py train  --run v3-2b [--base 2b|4b|0.8b] [--epochs 2] [--lora-r 16]
                                             [--lora-alpha 32] [--train-file datasets/...jsonl ...]
                                             [--eval synth=datasets/synth-v1/eval.jsonl ...] --yes
    python tools/cloud/owf_daytona.py bench-cpu [--cpu 8 --gpu-host] [--models v3-0.8b v3-2b] [--limit 60] --yes
                                             # CPU-only refine latency (no GPU; docs/refine-cpu-latency.md)
    python tools/cloud/owf_daytona.py eval --models v3-4b=hf:<repo>/<file>.gguf [v4-2b=v4-2b ...]
                                             --eval eval4=datasets/eval-v4/eval.jsonl [...] --yes
    python tools/cloud/owf_daytona.py list                     # our sandboxes (labelled) + recent runs
    python tools/cloud/owf_daytona.py fetch <run-id>           # outputs of a run whose controller died
    python tools/cloud/owf_daytona.py stop  <run-id>           # stop + delete + verify one run
    python tools/cloud/owf_daytona.py stop-all-managed [--yes]

Interpreter: anything with the `daytona` SDK (0.207+), e.g.
`.venv-cloud` here (`uv venv .venv-cloud && uv pip install --python .venv-cloud daytona`).

Rules:
  * DAYTONA_API_KEY is read at runtime from ./.env or $OWF_CLOUD_ENV (or --env-file / the environment)
    into this process only; it is never printed, logged, written or copied. The HF token
    (~/.cache/huggingface/token) is passed to the sandbox as an env var only, never printed.
  * Preflight proves the key is bound to the expected organization before anything is billed.
  * Spot only, RTX-5090 then RTX-4090; no capacity is an error, never an on-demand fallback.
    (bench-cpu without --gpu-host asks for no GPU: a plain on-demand CPU sandbox, max 4 vCPU.)
  * Every sandbox carries managed-by=owf-cloud and owf-run=<run-id>; only sandboxes carrying
    both are ever stopped or deleted (the organization has other people's sandboxes).
  * Hard TTL fuse (ttl_minutes), no idle stop, delete on stop; a paid launch needs --yes and
    refuses above --max-cost (the ceiling: the sandbox running to its TTL at on-demand prices).
  * Whatever happens (error, Ctrl-C, eviction), the job's sandbox is stopped, deleted and
    verified gone before the launcher exits.
Run records + logs: training/refine/data/cloud-runs/<run-id>/ (gitignored).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
REFINE = ROOT / "training" / "refine"
RUNS = REFINE / "data" / "cloud-runs"
SANDBOX_DIR = Path(__file__).resolve().parent / "sandbox"
ENV_FILES = [Path(os.environ["OWF_CLOUD_ENV"]) if os.environ.get("OWF_CLOUD_ENV") else ROOT / ".env"]
HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"

# The Daytona organization the API key must be bound to (preflight refuses to run without it).
EXPECTED_ORG_ID = os.environ.get("OWF_DAYTONA_ORG_ID", "<your-daytona-org-id>")
MANAGED_KEY, MANAGED_VALUE = "managed-by", "owf-cloud"
RUN_LABEL = "owf-run"

GPU_5090, GPU_4090 = "RTX-5090", "RTX-4090"
DEFAULT_GPU_PREF = [GPU_5090, GPU_4090]
# daytona.io/pricing (read 2026-09-03), ON-DEMAND: spot has no published figure, so every
# number shown is a ceiling.
GPU_PRICE_PER_HOUR = {GPU_5090: 0.74, GPU_4090: 0.57}
VCPU_PER_HOUR, GIB_RAM_PER_HOUR, GIB_DISK_PER_HOUR, DISK_FREE_GIB = 0.0504, 0.0162, 0.000108, 5

WORK = "/workspace"
REPO = "/workspace/owf"
OUT_TAR = "/workspace/out.tar.gz"

# Base models: HF ids at pinned revisions (2B = configs/qwen35-2b-lora.yaml; 4B / 0.8B pinned
# from the hub's main on 2026-10-05).
BASES = {
    "2b": ("Qwen3.5-2B", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "4b": ("Qwen3.5-4B", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "0.8b": ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
}


class LauncherError(RuntimeError):
    pass


class CapacityError(LauncherError):
    pass


def say(msg: str) -> None:
    print(msg, flush=True)


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def new_run_id(kind: str) -> str:
    return f"{kind}-{_dt.datetime.now().strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


# --- secrets ----------------------------------------------------------------------------------


def load_env(extra: Path | None = None) -> dict[str, str]:
    """DAYTONA_* from the environment or a dotenv file, without echoing a value."""
    keys = ("DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET")
    found: dict[str, str] = {}
    for path in ([extra] if extra else []) + ENV_FILES:
        if not path or not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            found.setdefault(k, v)
    return {k: os.environ.get(k) or found[k] for k in keys if os.environ.get(k) or found.get(k)}


def hf_token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if not t and HF_TOKEN_FILE.exists():
        t = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    return t or None


def describe_secret(v: str | None) -> str:
    return f"present ({len(v)} chars)" if v else "MISSING"


# --- images -------------------------------------------------------------------------------------

TORCH = "torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128"
LIVEKIT = "livekit-wakeword @ git+https://github.com/livekit/livekit-wakeword@95448a7559c453fcd87645bd67b247ffb45f85b0"
PARAKEET_REV = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"


def render_image():
    """The real-v2 data pipeline's Linux twin: .venv-data (CPU onnxruntime for Parakeet, torch
    cu128 for Piper) and .venv-kgpu (onnxruntime-gpu for Kokoro; torch's CUDA libs via a .pth),
    pinned to the local versions, plus the same model files, downloaded here (never from this
    machine). The noise / room-IR sets are fetched by the job (render_job.sh); everything is checked
    against local hashes / tools/cloud/lkw_manifest.tsv at run time."""
    from daytona import Image

    pk = f"https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/{PARAKEET_REV}"
    pdir = "/opt/lappdata/openwhisprflow/models/parakeet-tdt-0.6b-v3-int8"
    lk = "https://github.com/livekit/livekit-wakeword/releases/download/v0.1.0"
    ko = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    return (Image.debian_slim("3.12")
            .run_commands(
                "apt-get update && apt-get install -y --no-install-recommends espeak-ng libsndfile1 git curl "
                "ca-certificates procps && rm -rf /var/lib/apt/lists/*",
                f"python -m venv /opt/venv-data && /opt/venv-data/bin/pip install --no-cache-dir {TORCH} "
                "torchaudio==2.11.0",
                "/opt/venv-data/bin/pip install --no-cache-dir numpy==2.5.3 scipy==1.18.1 soundfile==0.14.0 "
                "onnxruntime==1.30.0 onnx-asr==0.12.0 huggingface_hub==1.33.0 kokoro-onnx==0.6.1 phonemizer==3.4.0 "
                "pydantic==2.13.5 pyyaml==6.0.3 whisper-normalizer==0.1.15 "
                f"espeakng-loader==0.2.4 '{LIVEKIT}'",
                "python -m venv /opt/venv-kgpu && /opt/venv-kgpu/bin/pip install --no-cache-dir onnxruntime-gpu==1.23.2 "
                "numpy==2.5.3 scipy==1.18.1 soundfile==0.14.0 espeakng-loader==0.2.4 phonemizer-fork==3.3.2 colorlog "
                "&& /opt/venv-kgpu/bin/pip install --no-cache-dir --no-deps kokoro-onnx==0.6.1 "
                "&& echo /opt/venv-data/lib/python3.12/site-packages > "
                "/opt/venv-kgpu/lib/python3.12/site-packages/zz-venv-data.pth",
                "/opt/venv-data/bin/python -c 'import torch, onnx_asr, kokoro_onnx, scipy, soundfile, whisper_normalizer.english; "
                "from livekit.wakeword.data.piper.synthesis import _load_vits_model; "
                "from livekit.wakeword.data.piper.vits_utils import slerp'",
                "/opt/venv-kgpu/bin/python -c 'import torch, onnxruntime, kokoro_onnx; print(onnxruntime.__version__)'",
                f"mkdir -p /opt/lkw/piper /opt/models/tts-models {pdir} "
                f"&& curl -fsSL -o /opt/lkw/piper/en-us-libritts-high.pt {lk}/en-us-libritts-high.state_dict.pt "
                f"&& curl -fsSL -o /opt/lkw/piper/en-us-libritts-high.json {lk}/en-us-libritts-high.config.json "
                f"&& curl -fsSL -o /opt/models/tts-models/kokoro-v1.0.onnx {ko}/kokoro-v1.0.onnx "
                f"&& curl -fsSL -o /opt/models/tts-models/voices-v1.0.bin {ko}/voices-v1.0.bin "
                f"&& for f in config.json vocab.txt decoder_joint-model.int8.onnx encoder-model.int8.onnx; do "
                f"curl -fsSL -o {pdir}/$f {pk}/$f; done",
                f"mkdir -p {WORK} && chmod 777 {WORK}")
            .workdir(WORK))


def train_image():
    """requirements-train.txt on Linux (triton comes with torch; triton-windows is Windows-only),
    plus the eval harness's few app deps and a Rust toolchain for train.py's prompt-parity check.
    Ubuntu 24.04 (its own Python 3.12), because llama.cpp's ubuntu-cuda release binaries need
    glibc 2.38 / GLIBCXX_3.4.32, newer than Debian bookworm's."""
    from daytona import Image

    return (Image.base("ubuntu:24.04")
            .env({"DEBIAN_FRONTEND": "noninteractive"})
            .run_commands(
                "apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev "
                "git curl ca-certificates build-essential pkg-config libdbus-1-dev libssl-dev procps "
                "&& rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3.12 /usr/local/bin/python",
                "export RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo && curl -fsSL https://sh.rustup.rs | "
                "sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path && chmod -R a+rwX /opt/rustup /opt/cargo",
                f"python -m venv /opt/venv-train && /opt/venv-train/bin/pip install --no-cache-dir {TORCH}",
                "/opt/venv-train/bin/pip install --no-cache-dir accelerate==1.15.0 datasets==5.0.1 einops==0.8.2 "
                "fla-core==0.5.2 flash-linear-attention==0.5.2 httpx==0.28.1 numpy==2.5.3 peft==0.21.2 "
                "protobuf==7.36.2 pyyaml==6.0.3 safetensors==0.8.0 sentencepiece==0.2.2 tokenizers==0.23.2 "
                "transformers==5.18.0 huggingface_hub==1.33.0 gguf==0.19.0 platformdirs==4.12.3 tomli-w==1.2.0 tqdm",
                "/opt/venv-train/bin/python -c 'import torch, transformers, peft, fla, httpx, platformdirs, tomli_w; "
                "print(torch.__version__, transformers.__version__)'",
                f"mkdir -p {WORK} && chmod 777 {WORK}")
            .env({"RUSTUP_HOME": "/opt/rustup", "CARGO_HOME": "/opt/cargo"})
            .workdir(WORK))


def bench_image():
    """CPU-only latency bench: Ubuntu 24.04 (llama.cpp's ubuntu release binaries need glibc 2.38)
    and just what tools/eval/eval_refine.py's prompt path imports. No CUDA anywhere."""
    from daytona import Image

    return (Image.base("ubuntu:24.04")
            .env({"DEBIAN_FRONTEND": "noninteractive"})
            .run_commands(
                "apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv "
                "curl ca-certificates procps util-linux libgomp1 && rm -rf /var/lib/apt/lists/*",
                "python3.12 -m venv /opt/venv-bench && /opt/venv-bench/bin/pip install --no-cache-dir httpx==0.28.1 "
                "platformdirs==4.12.3 tomli-w==1.2.0 numpy==2.5.3",   # numpy: openwhisprflow.stt.base, imported by refine.local
                f"mkdir -p {WORK}/models && chmod -R 777 {WORK}")
            .workdir(WORK))


IMAGES = {"render": render_image, "train": train_image, "bench-cpu": bench_image, "eval": train_image}


# --- the client boundary ------------------------------------------------------------------------


class Client:
    """Everything asked of Daytona. Sandboxes are passed around as plain dicts."""

    def __init__(self, env: dict[str, str]):
        from daytona import Daytona, DaytonaConfig

        if not env.get("DAYTONA_API_KEY"):
            raise LauncherError("DAYTONA_API_KEY not found (./.env, $OWF_CLOUD_ENV, --env-file or the environment)")
        self._d = Daytona(DaytonaConfig(api_key=env["DAYTONA_API_KEY"], api_url=env.get("DAYTONA_API_URL") or None,
                                        target=env.get("DAYTONA_TARGET") or None))
        self._sb: dict[str, Any] = {}

    def verify_org(self) -> dict:
        from daytona_api_client import ApiKeysApi, OrganizationsApi

        out: dict[str, Any] = {"key_ok": False, "bound": False, "classes": [], "error": ""}
        try:
            ApiKeysApi(self._d._api_client).get_current_api_key()
            out["key_ok"] = True
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"key rejected: {type(exc).__name__} {getattr(exc, 'status', '')}"
            return out
        api = OrganizationsApi(self._d._api_client)
        try:
            api.get_organization_usage_overview(EXPECTED_ORG_ID)
            out["bound"] = True
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"org-scoped call refused: {type(exc).__name__} {getattr(exc, 'status', '')}"
            return out
        try:
            for c in api.list_available_sandbox_classes(EXPECTED_ORG_ID):
                sc = getattr(c, "sandbox_class", None)
                sc = getattr(sc, "value", sc) or "?"
                allowed = getattr(c, "allowed_gpu_types", None)
                out["classes"].append(f"{sc}@{getattr(c, 'region_id', '?')}"
                                      + (" gpu" if getattr(c, "gpu_available", None) else " no-gpu")
                                      + (f" {[getattr(g, 'value', g) for g in allowed]}" if allowed else ""))
        except Exception as exc:  # noqa: BLE001
            out["classes"] = [f"(unavailable: {type(exc).__name__})"]
        return out

    def create(self, *, name: str, labels: dict, image, gpu_pref: list[str], cpu: int, memory: int, disk: int,
               ttl_minutes: int, env_vars: dict[str, str], on_logs: Callable[[str], None]) -> dict:
        from daytona import CreateSandboxFromImageParams, DaytonaError, GpuType, Resources

        gpus = [GpuType(g) for g in gpu_pref]
        res = (Resources(cpu=cpu, memory=memory, disk=disk, gpu=1, gpu_type=gpus if len(gpus) > 1 else gpus[0])
               if gpus else Resources(cpu=cpu, memory=memory, disk=disk))   # no GPU: plain CPU sandbox
        params = CreateSandboxFromImageParams(
            name=name, labels={MANAGED_KEY: MANAGED_VALUE, **labels}, spot=bool(gpus), auto_stop_interval=0,
            auto_delete_interval=0, ttl_minutes=ttl_minutes, env_vars=env_vars, image=image, resources=res)
        try:
            sb = self._d.create(params, timeout=5400, on_snapshot_create_logs=on_logs)
        except DaytonaError as exc:
            msg = str(exc)
            if re.search(r"capacit|spot|no available|insufficient|unavailable", msg, re.I):
                raise CapacityError(f"no spot capacity for {gpu_pref}: {msg}") from exc
            raise LauncherError(f"sandbox create failed: {msg}") from exc
        self._sb[sb.id] = sb
        return self._as_dict(sb)

    def get(self, sid: str) -> dict | None:
        from daytona import DaytonaNotFoundError

        try:
            sb = self._d.get(sid)
        except DaytonaNotFoundError:
            self._sb.pop(sid, None)
            return None
        self._sb[sb.id] = sb
        return self._as_dict(sb)

    def list_managed(self) -> list[dict]:
        from daytona import ListSandboxesQuery

        res = self._d.list(ListSandboxesQuery(labels={MANAGED_KEY: MANAGED_VALUE}))
        items = getattr(res, "items", res)
        return [self._as_dict(sb) for sb in items]

    def _obj(self, sid: str):
        if sid not in self._sb and self.get(sid) is None:
            raise LauncherError(f"sandbox {sid} no longer exists")
        return self._sb[sid]

    def upload(self, sid: str, local: Path, remote: str) -> None:
        self._obj(sid).fs.upload_file(str(local), remote, timeout=3600)

    def download_to(self, sid: str, remote: str, local: Path) -> bool:
        try:
            self._obj(sid).fs.download_file(remote, str(local), 3600)
            return local.exists()
        except Exception as exc:  # noqa: BLE001
            say(f"  download {remote} failed: {type(exc).__name__}: {str(exc)[:200]}")
            return False

    def exec(self, sid: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
        r = self._obj(sid).process.exec(cmd, timeout=timeout)
        return int(r.exit_code or 0), r.result or ""

    def session_start(self, sid: str, session: str, cmd: str) -> str:
        from daytona import SessionExecuteRequest

        p = self._obj(sid).process
        p.create_session(session)
        return p.execute_session_command(session, SessionExecuteRequest(command=cmd, run_async=True), timeout=None).cmd_id

    def session_logs(self, sid: str, session: str, cmd_id: str) -> str:
        r = self._obj(sid).process.get_session_command_logs(session, cmd_id)
        return (getattr(r, "output", None) or "") or ((r.stdout or "") + (r.stderr or ""))

    def session_exit_code(self, sid: str, session: str, cmd_id: str) -> int | None:
        return getattr(self._obj(sid).process.get_session_command(session, cmd_id), "exit_code", None)

    def stop_delete_verify(self, sid: str, run_id: str) -> str:
        """Only a sandbox carrying both of our labels for this run. Returns the final state."""
        sb = self.get(sid)
        if sb is None:
            return "gone"
        labels = sb.get("labels") or {}
        if labels.get(MANAGED_KEY) != MANAGED_VALUE or labels.get(RUN_LABEL) != run_id:
            raise LauncherError(f"{sid} is not ours for run {run_id} (labels {labels}); left alone")
        obj = self._sb[sid]
        if sb["state"] not in ("stopped", "destroyed", "destroying"):
            try:
                obj.stop(timeout=120)
            except Exception:  # noqa: BLE001
                try:
                    obj.stop(timeout=120, force=True)
                except Exception:  # noqa: BLE001
                    pass
        try:
            self._d.delete(obj, timeout=180, wait=True)
        except Exception:  # noqa: BLE001
            pass
        after = self.get(sid)
        state = after["state"] if after else "gone"
        if after is not None and state not in ("destroyed", "destroying"):
            raise LauncherError(f"{sid} still {state} after stop+delete; check the Daytona dashboard")
        return state

    @staticmethod
    def _as_dict(sb) -> dict:
        st, gt = getattr(sb, "state", None), getattr(sb, "gpu_type", None)
        return {"id": sb.id, "name": getattr(sb, "name", "") or "", "state": str(getattr(st, "value", st) or "unknown"),
                "labels": dict(getattr(sb, "labels", None) or {}),
                "gpu_type": str(getattr(gt, "value", gt)) if gt else None,
                "cpu": getattr(sb, "cpu", None), "memory": getattr(sb, "memory", None),
                "spot_evicted_at": str(sb.spot_evicted_at) if getattr(sb, "spot_evicted_at", None) else None}


# --- a job: one sandbox, one command, one output tarball ----------------------------------------


@dataclass
class Job:
    kind: str                       # render | train
    run_id: str
    command: str                    # run in REPO, in a child shell, output -> session log
    bundle: dict[str, Path]         # remote path (relative to REPO) -> local file
    extra_files: dict[str, bytes] = field(default_factory=dict)   # generated files, relative to REPO
    uploads: dict[str, Path] = field(default_factory=dict)        # big files sent as-is: absolute remote -> local
    gpu_pref: list[str] = field(default_factory=lambda: list(DEFAULT_GPU_PREF))
    cpu: int = 8
    memory: int = 32
    disk: int = 60
    ttl_minutes: int = 120
    max_cost: float = 5.0
    env_vars: dict[str, str] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def rate(self) -> float:
        g = max(self.gpu_pref, key=lambda x: GPU_PRICE_PER_HOUR.get(x, 0.0)) if self.gpu_pref else ""
        return (GPU_PRICE_PER_HOUR.get(g, 0.0) + self.cpu * VCPU_PER_HOUR + self.memory * GIB_RAM_PER_HOUR
                + max(0, self.disk - DISK_FREE_GIB) * GIB_DISK_PER_HOUR)

    def ceiling(self) -> float:
        return self.ttl_minutes / 60.0 * self.rate()

    def describe(self) -> str:
        return "\n".join([
            f"run            {self.run_id} ({self.kind})",
            f"gpu            {' > '.join(self.gpu_pref)} spot (no on-demand fallback)" if self.gpu_pref
            else "gpu            none (CPU-only sandbox, on-demand)",
            f"sandbox        {self.cpu} vCPU / {self.memory} GiB / {self.disk} GiB disk",
            f"ttl            {self.ttl_minutes} min hard fuse, no idle stop, delete on stop",
            f"upload         {len(self.bundle) + len(self.extra_files) + len(self.uploads)} files "
            f"({sum(p.stat().st_size for p in [*self.bundle.values(), *self.uploads.values()]) / 1e6:.1f} MB)",
            f"rate           ${self.rate():.3f}/h (on-demand ceiling; spot is cheaper, unpublished)",
            f"MAX COST       ${self.ceiling():.2f} (to TTL)  --max-cost ${self.max_cost:.2f} "
            + ("OK" if self.ceiling() <= self.max_cost else "EXCEEDED -> refuse"),
            f"params         {json.dumps(self.params)}",
        ])


_CTRL = "".join(chr(c) for c in range(0, 32) if chr(c) not in "\t")


def clean_log_line(line: str) -> str:
    """Session logs prefix lines with control bytes (0x01 x3) and carry tqdm's \\r redraws."""
    return line.split("\r")[-1].lstrip(_CTRL).rstrip()


def _tar_bundle(job: Job, path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for rel, src in sorted(job.bundle.items()):
            info = tar.gettarinfo(str(src), arcname=rel)
            info.mode = 0o755 if rel.endswith(".sh") else 0o644
            info.uid = info.gid = 0
            with open(src, "rb") as f:
                if rel.endswith(".sh"):   # CRLF checkouts must not reach bash
                    data = f.read().replace(b"\r\n", b"\n")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                else:
                    tar.addfile(info, f)
        for rel, data in job.extra_files.items():
            info = tarfile.TarInfo(rel)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))


def save_record(run_dir: Path, rec: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, run_dir / "run.json")


def run_job(client: Client, job: Job, *, poll_s: float = 15.0) -> dict:
    """Create -> upload -> run detached -> stream log -> download OUT_TAR -> stop/delete/verify.
    Returns the run record (also in RUNS/<run>/run.json)."""
    run_dir = RUNS / job.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"run_id": job.run_id, "kind": job.kind, "params": job.params, "created": now(),
                           "gpu_pref": job.gpu_pref, "cpu": job.cpu, "memory": job.memory, "disk": job.disk,
                           "ttl_minutes": job.ttl_minutes, "ceiling_usd": round(job.ceiling(), 2),
                           "rate_usd_h": round(job.rate(), 3), "status": "creating"}
    save_record(run_dir, rec)
    bundle = run_dir / "bundle.tar.gz"
    _tar_bundle(job, bundle)
    t_create = time.time()
    build_log = open(run_dir / "image-build.log", "a", encoding="utf-8")

    def on_build(line: str) -> None:
        build_log.write(line.rstrip() + "\n")
        build_log.flush()
        ln = clean_log_line(line)
        if ln and (ln.startswith(("Step", "#", "ERROR", "error")) or "Successfully" in ln):
            say(f"  [image] {ln[:160]}")

    def _term(*_):
        raise KeyboardInterrupt()

    prev = signal.signal(signal.SIGTERM, _term) if hasattr(signal, "SIGTERM") else None
    sid = None
    log_path = run_dir / "job.log"
    try:
        say(f"creating sandbox (image builds once, then is cached; this can take a while the first time)")
        sb = client.create(name=f"owf-{job.run_id}"[:60], labels={RUN_LABEL: job.run_id, "owf-job": job.kind},
                           image=IMAGES[job.kind](), gpu_pref=job.gpu_pref, cpu=job.cpu, memory=job.memory,
                           disk=job.disk, ttl_minutes=job.ttl_minutes, env_vars=job.env_vars, on_logs=on_build)
        sid = sb["id"]
        t_start = time.time()
        rec.update(sandbox_id=sid, gpu=sb.get("gpu_type"), started=now(), create_s=round(t_start - t_create, 1),
                   status="running")
        save_record(run_dir, rec)
        say(f"sandbox {sid} gpu={sb.get('gpu_type') or '?'} up after {t_start - t_create:.0f}s")
        client.upload(sid, bundle, f"{WORK}/bundle.tar.gz")
        code, out = client.exec(sid, f"mkdir -p {REPO} && tar xzf {WORK}/bundle.tar.gz -C {REPO} && echo ok")
        if code != 0:
            raise LauncherError(f"bundle extract failed: {out[-500:]}")
        for remote, local in job.uploads.items():
            t_up = time.time()
            client.exec(sid, f"mkdir -p {shlex.quote(os.path.dirname(remote))}")
            client.upload(sid, local, remote)
            say(f"uploaded {local.name} ({local.stat().st_size / 1e6:.0f} MB) in {time.time() - t_up:.0f}s")
        inner = f"cd {REPO} && {job.command}"
        cmd_id = client.session_start(sid, "job", "bash -c " + shlex.quote(inner) + " 2>&1")
        seen = 0
        exit_code = None
        with open(log_path, "a", encoding="utf-8") as logf:
            while True:
                time.sleep(poll_s)
                live = client.get(sid)
                try:
                    text = client.session_logs(sid, "job", cmd_id)
                except Exception:  # noqa: BLE001
                    text = ""
                if len(text) > seen:
                    new = text[seen:]
                    logf.write(new)
                    logf.flush()
                    for ln in new.splitlines():
                        ln = clean_log_line(ln)
                        if ln:
                            say(f"  [{job.kind}] {ln[:240]}")
                    seen = len(text)
                if live is None or live.get("spot_evicted_at") or live.get("state") in (
                        "destroyed", "destroying", "error", "stopped", "build_failed"):
                    rec["status"] = "evicted" if live and live.get("spot_evicted_at") else "lost"
                    raise LauncherError(f"sandbox {sid} gone mid-job ({(live or {}).get('state', 'missing')}); "
                                        f"re-run the command (spot eviction)")
                try:
                    exit_code = client.session_exit_code(sid, "job", cmd_id)
                except Exception:  # noqa: BLE001
                    exit_code = None
                if exit_code is not None:
                    break
        t_end = time.time()
        rec.update(exit_code=exit_code, job_s=round(t_end - t_start, 1))
        say(f"job exited {exit_code} after {(t_end - t_start) / 60:.1f} min")
        if exit_code == 0:
            local_tar = run_dir / "out.tar.gz"
            if not client.download_to(sid, OUT_TAR, local_tar):
                raise LauncherError("job finished but the output tarball could not be downloaded")
            rec["out_bytes"] = local_tar.stat().st_size
            say(f"downloaded {local_tar.stat().st_size / 1e6:.1f} MB -> {local_tar}")
            rec["status"] = "done"
        else:
            rec["status"] = "failed"
    except KeyboardInterrupt:
        rec["status"] = "interrupted"
        say("interrupted: stopping the sandbox")
    except LauncherError as exc:
        rec.setdefault("error", str(exc))
        if rec["status"] in ("creating", "running"):
            rec["status"] = "failed"
        say(f"!! {exc}")
    finally:
        build_log.close()
        if prev is not None:
            signal.signal(signal.SIGTERM, prev)
        if not sid:   # a failed create (e.g. BUILD_FAILED) can still leave a labelled sandbox behind
            try:
                for s in client.list_managed():
                    if s["labels"].get(RUN_LABEL) == job.run_id:
                        say(f"cleaning up {s['id']} ({s['state']}): {client.stop_delete_verify(s['id'], job.run_id)}")
            except Exception as exc:  # noqa: BLE001
                say(f"!! cleanup after failed create: {exc}; run `stop {job.run_id}`")
        if sid:
            t_stop = time.time()
            try:
                rec["final_state"] = client.stop_delete_verify(sid, job.run_id)
                say(f"sandbox {sid}: {rec['final_state']} (verified)")
            except Exception as exc:  # noqa: BLE001
                rec["final_state"] = f"UNVERIFIED: {exc}"
                say(f"!! sandbox {sid} NOT verified gone: {exc}")
            billed_h = (t_stop - (t_create + rec.get("create_s", 0))) / 3600
            rec["billed_minutes_est"] = round(billed_h * 60, 1)
            rec["cost_ceiling_actual_usd"] = round(billed_h * job.rate(), 3)
        rec["wall_s"] = round(time.time() - t_create, 1)
        rec["ended"] = now()
        save_record(run_dir, rec)
    return rec


# --- commands ---------------------------------------------------------------------------------


def files_under(base: Path, rel_to: Path, pattern: str = "**/*", exclude: tuple[str, ...] = ()) -> dict[str, Path]:
    out = {}
    for p in base.glob(pattern):
        if p.is_file() and "__pycache__" not in p.parts and not any(x in str(p) for x in exclude):
            out[str(p.relative_to(rel_to)).replace("\\", "/")] = p
    return out


def confirm_launch(job: Job, yes: bool, dry: bool) -> bool:
    say(job.describe())
    if job.ceiling() > job.max_cost:
        raise LauncherError(f"ceiling ${job.ceiling():.2f} exceeds --max-cost ${job.max_cost:.2f}; "
                            f"lower --ttl-minutes or raise the cap")
    if dry:
        say("dry run: nothing launched")
        return False
    if not yes:
        try:
            ok = input(f"launch one spot sandbox, max ${job.ceiling():.2f}? type yes: ").strip().lower() == "yes"
        except EOFError:
            ok = False
        if not ok:
            raise LauncherError("not confirmed (pass --yes)")
    return True


def cmd_preflight(a, env) -> int:
    say(f"api key        {describe_secret(env.get('DAYTONA_API_KEY'))}")
    say(f"hf token       {describe_secret(hf_token())}")
    v = Client(env).verify_org()
    if not v["key_ok"] or not v["bound"]:
        say(f"preflight      FAILED: {v['error']}")
        return 2
    say(f"organization   {EXPECTED_ORG_ID} verified by id")
    say(f"classes        {', '.join(v['classes'])}")
    say(f"gpu policy     {' > '.join(DEFAULT_GPU_PREF)} spot; capacity is only known at create time")
    say("preflight      OK")
    return 0


def render_job(a) -> Job:
    rid = a.run_id or new_run_id("render")
    files = list(a.files)
    bundle = files_under(ROOT / "tools" / "refine-data", ROOT, "*.py")
    bundle["tools/refine-data/piper_speakers.json"] = ROOT / "tools/refine-data/piper_speakers.json"
    bundle.update(files_under(ROOT / "src" / "openwhisprflow", ROOT, "**/*.py"))
    bundle.update(files_under(SANDBOX_DIR, ROOT))
    bundle["tools/cloud/lkw_manifest.tsv"] = ROOT / "tools/cloud/lkw_manifest.tsv"
    for w in files:
        p = ROOT / "training/refine/datasets/scripts-v2" / f"{w}.jsonl"
        if not p.exists():
            raise LauncherError(f"no script file {p}")
        bundle[f"training/refine/datasets/scripts-v2/{w}.jsonl"] = p
    c = a.cpu
    workers = [a.piper_workers or 2, a.kokoro_workers or 3, 2, a.parakeet_workers or max(1, c // 4),
               a.parakeet_threads or 4]
    cmd = ("bash tools/cloud/sandbox/render_job.sh " + shlex.quote(" ".join(files)) + f" {a.per_file or 0} "
           + " ".join(map(str, workers)))
    tok = hf_token()
    return Job(kind="render", run_id=rid, command=cmd, bundle=bundle, cpu=c, memory=a.memory, disk=a.disk,
               ttl_minutes=a.ttl_minutes, max_cost=a.max_cost, gpu_pref=a.gpu.split(","),
               env_vars={"HF_TOKEN": tok} if tok else {},
               params={"files": files, "per_file": a.per_file, "workers": workers})


def cmd_render(a, env) -> int:
    job = render_job(a)
    if not confirm_launch(job, a.yes, a.dry_run):
        return 0
    rec = run_job(Client(env), job)
    if rec["status"] != "done":
        say(f"render {job.run_id}: {rec['status']} (log {RUNS / job.run_id / 'job.log'})")
        return 1
    return finish_render(job.run_id, a.apply, a.force)


def finish_render(run_id: str, apply: bool, force: bool) -> int:
    run_dir = RUNS / run_id
    rec = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    out = run_dir / "out"
    shutil.rmtree(out, ignore_errors=True)
    with tarfile.open(run_dir / "out.tar.gz") as tar:
        tar.extractall(out, filter="data")
    real = out / "training/refine/datasets/real-v2"
    got = {}
    for f in sorted(list(real.glob("W*.jsonl")) + list(real.glob("E*.jsonl"))):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        got[f.stem] = rows
        eng = {r["raw_engine"] for r in rows}
        back = {}
        for r in rows:
            back[r["voice_detail"]["backend"]] = back.get(r["voice_detail"]["backend"], 0) + 1
        sev = {}
        for r in rows:
            sev[r["audit"]["auto"]["severity"]] = sev.get(r["audit"]["auto"]["severity"], 0) + 1
        say(f"{f.stem}: {len(rows)} rows; engines {sorted(eng)}; backends {back}; auto audit {sev}")
    rec["rows"] = {k: len(v) for k, v in got.items()}
    save_record(run_dir, rec)
    say(f"report: {real / 'REPORT.md'}")
    if not apply:
        say(f"not applied (pass --apply, or `fetch {run_id} --apply`): rows are in {real}")
        return 0
    dest = ROOT / "training/refine/datasets/real-v2"
    for w in got:
        if (dest / f"{w}.jsonl").exists() and not force:
            raise LauncherError(f"{dest / (w + '.jsonl')} exists; pass --force to replace it")
    for w in got:
        shutil.copy2(real / f"{w}.jsonl", dest / f"{w}.jsonl")
        say(f"  -> {dest / (w + '.jsonl')} ({len(got[w])} rows)")
    py = ROOT / ".venv-data" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    say("audit packets (packets.py, light local file work):")
    subprocess.run([str(py if py.exists() else sys.executable), str(ROOT / "tools/refine-data/packets.py")],
                   cwd=ROOT, check=True)
    rec["applied"] = now()
    save_record(run_dir, rec)
    return 0


def _yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"


def train_job(a, env) -> Job:
    run = a.run
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run):
        raise LauncherError("--run: letters, digits, . _ - only")
    if (REFINE / "output" / run).exists() and not a.force:
        raise LauncherError(f"training/refine/output/{run} exists locally; pick another --run (or --force)")
    base_name, base_repo, base_rev = BASES[a.base]
    train_files = []
    for pat in a.train_file or ["datasets/real-v2/final/train.jsonl"]:
        hits = sorted(glob.glob(str(REFINE / pat)))
        if not hits:
            raise LauncherError(f"no train file matches {pat} (relative to training/refine/)")
        train_files += [str(Path(h).relative_to(REFINE)).replace("\\", "/") for h in hits]
    dpo_files = []
    for pat in a.dpo_file or []:
        hits = sorted(glob.glob(str(REFINE / pat)))
        if not hits:
            raise LauncherError(f"no DPO file matches {pat} (relative to training/refine/)")
        dpo_files += [str(Path(h).relative_to(REFINE)).replace("\\", "/") for h in hits]
    if dpo_files:
        a.set = (a.set or []) + ["dpo_files=" + _yaml_list(dpo_files)]
    evals = a.eval or ["synth=datasets/synth-v1/eval.jsonl", "heldout=datasets/real-v2/final/heldout.jsonl"]
    for e in evals:
        k, _, p = e.partition("=")
        if not k or not p or not (REFINE / p).is_file():
            raise LauncherError(f"--eval {e!r}: want name=path relative to training/refine/")
    bundle: dict[str, Path] = {}
    for n in ("train.py", "export.py", "eval_gguf.py", "render.py"):
        bundle[f"training/refine/{n}"] = REFINE / n
    bundle.update(files_under(REFINE / "configs", ROOT))
    data = set(train_files) | set(dpo_files) | {e.partition("=")[2] for e in evals} | {
        "datasets/synth-v1/eval.jsonl", "datasets/synth-v1/train.jsonl", "datasets/GUIDE.md"}
    for d in data:
        bundle[f"training/refine/{d}"] = REFINE / d
    bundle.update(files_under(ROOT / "src" / "openwhisprflow", ROOT, "**/*.py"))
    bundle["tools/eval/eval_refine.py"] = ROOT / "tools/eval/eval_refine.py"
    base_out = ROOT / "tools/eval/out/refine-baseline-quill-2b-shared.jsonl"
    if base_out.exists():
        bundle["tools/eval/out/refine-baseline-quill-2b-shared.jsonl"] = base_out
    bundle.update(files_under(SANDBOX_DIR, ROOT))
    if a.rust_check:
        for f in ("Cargo.toml", "Cargo.lock"):
            bundle[f] = ROOT / f
        bundle.update(files_under(ROOT / "crates", ROOT, exclude=("target",)))
        bundle.update(files_under(ROOT / "app" / "src-tauri", ROOT, "src/**/*"))
        for f in ("Cargo.toml", "build.rs"):
            bundle[f"app/src-tauri/{f}"] = ROOT / "app/src-tauri" / f
    else:   # the eval's prompt-parity check reads the refine crate's prompts.rs (owf-refine, renamed ochre-refine)
        for c in sorted((ROOT / "crates").glob("*-refine")):
            bundle[f"crates/{c.name}/src/prompts.rs"] = c / "src/prompts.rs"
    job_env = "\n".join([
        f"RUN={shlex.quote(run)}", f"BASE_NAME={base_name}", f"BASE_REPO={base_repo}", f"BASE_REV={base_rev}",
        f"EPOCHS={a.epochs}", f"LORA_R={a.lora_r}", f"LORA_ALPHA={a.lora_alpha}",
        f"TRAIN_FILES={shlex.quote(_yaml_list(train_files))}", f"QUANTS={shlex.quote(a.quants)}",
        f"RUST_CHECK={'true' if a.rust_check else 'false'}",
        "EXTRA_SETS=(" + " ".join(shlex.quote(x) for s in (a.set or []) for x in ("--set", s)) + ")",
        "EVALS=(" + " ".join(shlex.quote(e) for e in evals) + ")", ""])
    env_vars = {}
    tok = hf_token()
    if tok:
        env_vars["HF_TOKEN"] = tok
    return Job(kind="train", run_id=a.run_id or new_run_id(f"train-{run}"), command="bash tools/cloud/sandbox/train_job.sh",
               bundle=bundle, extra_files={"tools/cloud/sandbox/job.env": job_env.encode()},
               cpu=a.cpu, memory=a.memory, disk=a.disk, ttl_minutes=a.ttl_minutes, max_cost=a.max_cost,
               gpu_pref=a.gpu.split(","), env_vars=env_vars,
               params={"run": run, "base": f"{base_repo}@{base_rev}", "epochs": a.epochs, "lora_r": a.lora_r,
                       "lora_alpha": a.lora_alpha, "train_files": train_files, "evals": evals, "quants": a.quants,
                       "rust_check": a.rust_check, "set": a.set or [], "hf_token": "present" if tok else "none"})


def cmd_train(a, env) -> int:
    job = train_job(a, env)
    if not confirm_launch(job, a.yes, a.dry_run):
        return 0
    rec = run_job(Client(env), job)
    if rec["status"] != "done":
        say(f"train {job.run_id}: {rec['status']} (log {RUNS / job.run_id / 'job.log'})")
        return 1
    return finish_train(job.run_id, job.params["run"], a.force)


def finish_train(run_id: str, run: str, force: bool) -> int:
    run_dir = RUNS / run_id
    dest = REFINE / "output"
    if (dest / run).exists() and not force:
        raise LauncherError(f"{dest / run} exists; outputs left in {run_dir / 'out.tar.gz'}")
    with tarfile.open(run_dir / "out.tar.gz") as tar:
        tar.extractall(dest, filter="data")
    say(f"outputs -> {dest / run}")
    for f in sorted((dest / run).rglob("*")):
        if f.is_file() and f.suffix in (".gguf", ".md", ".json") and "adapter" not in f.parts:
            say(f"  {f.relative_to(dest)}  {f.stat().st_size / 1e6:.1f} MB")
    return 0


def bench_cpu_job(a) -> Job:
    models = {}
    for m in a.models:
        p = Path(m) if m.endswith(".gguf") else REFINE / "output" / m / "gguf" / f"{m}-Q4_K_M.gguf"
        if not p.is_file():
            raise LauncherError(f"no GGUF {p}")
        models[f"{WORK}/models/{p.name}"] = p
    ev = REFINE / a.eval
    if not ev.is_file():
        raise LauncherError(f"--eval {a.eval}: not found under training/refine/")
    configs = a.configs or ([f"{a.cpu}vcpu:0:0", f"{a.cpu}vcpu-t{a.cpu // 2}:0:{a.cpu // 2}"]
                            + (["4vcpu-pinned:4:0"] if a.cpu > 4 else []))
    # Daytona caps CPU-only sandboxes at 4 vCPU; a larger shape needs a GPU host (--gpu-host): the
    # GPU sits idle, the CPU build never touches it, and the price includes it.
    gpu_pref = a.gpu.split(",") if a.gpu_host else []
    if a.cpu > 4 and not gpu_pref:
        raise LauncherError("CPU-only sandboxes max out at 4 vCPU on this org; pass --gpu-host for more")
    bundle = {"tools/eval/eval_refine.py": ROOT / "tools/eval/eval_refine.py", f"training/refine/{a.eval}": ev}
    bundle.update(files_under(ROOT / "src" / "openwhisprflow", ROOT, "**/*.py"))
    bundle.update({k: v for k, v in files_under(SANDBOX_DIR, ROOT).items() if "bench_cpu" in k})
    for c in sorted((ROOT / "crates").glob("*-refine")):    # eval_refine's prompt-parity check
        bundle[f"crates/{c.name}/src/prompts.rs"] = c / "src/prompts.rs"
    job_env = "\n".join([f"MODELS={shlex.quote(' '.join(Path(r).name for r in models))}",
                         f"EVAL={shlex.quote('training/refine/' + a.eval)}", f"LIMIT={a.limit}",
                         f"CONFIGS={shlex.quote(' '.join(configs))}", ""])
    return Job(kind="bench-cpu", run_id=a.run_id or new_run_id("bench-cpu"),
               command="bash tools/cloud/sandbox/bench_cpu_job.sh", bundle=bundle,
               extra_files={"tools/cloud/sandbox/job.env": job_env.encode()}, uploads=models, gpu_pref=gpu_pref,
               cpu=a.cpu, memory=a.memory, disk=a.disk, ttl_minutes=a.ttl_minutes, max_cost=a.max_cost,
               params={"models": [p.name for p in models.values()], "eval": a.eval, "limit": a.limit,
                       "configs": configs})


def cmd_bench_cpu(a, env) -> int:
    job = bench_cpu_job(a)
    if not confirm_launch(job, a.yes, a.dry_run):
        return 0
    rec = run_job(Client(env), job)
    if rec["status"] != "done":
        say(f"bench-cpu {job.run_id}: {rec['status']} (log {RUNS / job.run_id / 'job.log'})")
        return 1
    return finish_bench(job.run_id)


def finish_bench(run_id: str) -> int:
    out = RUNS / run_id / "out"
    shutil.rmtree(out, ignore_errors=True)
    with tarfile.open(RUNS / run_id / "out.tar.gz") as tar:
        tar.extractall(out, filter="data")
    cpu = next((ln.split(":", 1)[1].strip() for ln in (out / "lscpu.txt").read_text().splitlines()
                if ln.startswith("Model name")), "?")
    say(f"cpu: {cpu}\n| run | threads | median | p90 | max | gen tok/s | prompt ms (median) | paragraph | "
        f"cold first response | RSS peak MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in sorted(out.glob("summary-*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        say(f"| {s['name']} | {s['threads']} | {s['wall_median_ms']} | {s['wall_p90_ms']} | {s['wall_max_ms']} | "
            f"{s['gen_tok_s']} | {s['prompt_ms_median']} | {s['paragraph_median_ms']} | "
            f"{s['cold_first_response_ms']} | {s.get('mem_mib', {}).get('VmHWM')} |")
    say(f"raw: {out}")
    return 0


def eval_job(a) -> Job:
    """Existing GGUFs (private HF repos, downloaded inside the sandbox; or local files, uploaded)
    through training/refine/eval_gguf.py on one or more eval sets. Same image and llama.cpp build
    as `train`."""
    models, uploads = [], {}
    for m in a.models:
        name, _, src = m.partition("=")
        if not name or not src or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise LauncherError(f"--models {m!r}: want name=hf:<repo>/<file> | name=<run> | name=<path.gguf>")
        if src.startswith("hf:"):
            models.append(f"{name}={src}")
            continue
        p = Path(src) if src.endswith(".gguf") else REFINE / "output" / src / "gguf" / f"{src}-Q4_K_M.gguf"
        if not p.is_file():
            raise LauncherError(f"no GGUF {p}")
        remote = f"{WORK}/models/{p.name}"
        uploads[remote] = p
        models.append(f"{name}={remote}")
    for e in a.eval:
        k, _, pth = e.partition("=")
        if not k or not pth or not (REFINE / pth).is_file():
            raise LauncherError(f"--eval {e!r}: want name=path relative to training/refine/")
    bundle: dict[str, Path] = {"training/refine/eval_gguf.py": REFINE / "eval_gguf.py",
                               "tools/eval/eval_refine.py": ROOT / "tools/eval/eval_refine.py"}
    for e in a.eval:
        pth = e.partition("=")[2]
        bundle[f"training/refine/{pth}"] = REFINE / pth
    bundle.update(files_under(ROOT / "src" / "openwhisprflow", ROOT, "**/*.py"))
    bundle.update({k: v for k, v in files_under(SANDBOX_DIR, ROOT).items() if "eval_job" in k})
    nj = REFINE / "data" / "no-judge-validation.json"
    if nj.exists():
        bundle["training/refine/data/no-judge-validation.json"] = nj
    for c in sorted((ROOT / "crates").glob("*-refine")):    # eval_refine's prompt-parity check
        bundle[f"crates/{c.name}/src/prompts.rs"] = c / "src/prompts.rs"
    job_env = "\n".join(["MODELS=(" + " ".join(shlex.quote(x) for x in models) + ")",
                         "EVALS=(" + " ".join(shlex.quote(e) for e in a.eval) + ")",
                         "CHUNKS=(" + " ".join(a.chunk_modes) + ")", ""])
    tok = hf_token()
    return Job(kind="eval", run_id=a.run_id or new_run_id("eval"), command="bash tools/cloud/sandbox/eval_job.sh",
               bundle=bundle, extra_files={"tools/cloud/sandbox/job.env": job_env.encode()}, uploads=uploads,
               cpu=a.cpu, memory=a.memory, disk=a.disk, ttl_minutes=a.ttl_minutes, max_cost=a.max_cost,
               gpu_pref=a.gpu.split(","), env_vars={"HF_TOKEN": tok} if tok else {},
               params={"models": models, "evals": a.eval, "chunk_modes": a.chunk_modes,
                       "hf_token": "present" if tok else "none"})


def cmd_eval(a, env) -> int:
    job = eval_job(a)
    if not confirm_launch(job, a.yes, a.dry_run):
        return 0
    rec = run_job(Client(env), job)
    if rec["status"] != "done":
        say(f"eval {job.run_id}: {rec['status']} (log {RUNS / job.run_id / 'job.log'})")
        return 1
    return finish_eval(job.run_id)


def finish_eval(run_id: str) -> int:
    out = RUNS / run_id / "out"
    shutil.rmtree(out, ignore_errors=True)
    with tarfile.open(RUNS / run_id / "out.tar.gz") as tar:
        tar.extractall(out, filter="data")
    dest = ROOT / "tools" / "eval" / "out" / "eval-runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted((out / "evalout").glob("*")):
        shutil.copy2(f, dest / f.name)
        if f.name.startswith("refine-"):
            say(f"  {f.name}: {sum(1 for x in f.read_text(encoding='utf-8').splitlines() if x.strip())} rows")
    say(f"outputs -> {dest}")
    return 0


def cmd_fetch(a, env) -> int:
    """Download a run's outputs from its sandbox if the controller died before it could."""
    run_dir = RUNS / a.run_id
    rec = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if not (run_dir / "out.tar.gz").exists():
        c = Client(env)
        sid = rec.get("sandbox_id")
        if not sid or c.get(sid) is None:
            raise LauncherError(f"run {a.run_id}: no sandbox left to fetch from")
        if not c.download_to(sid, OUT_TAR, run_dir / "out.tar.gz"):
            raise LauncherError(f"run {a.run_id}: {OUT_TAR} not there yet (job still running?)")
    if rec["kind"] == "render":
        return finish_render(a.run_id, a.apply, a.force)
    if rec["kind"] == "bench-cpu":
        return finish_bench(a.run_id)
    if rec["kind"] == "eval":
        return finish_eval(a.run_id)
    return finish_train(a.run_id, rec["params"]["run"], a.force)


def cmd_list(a, env) -> int:
    sbs = Client(env).list_managed()
    say(f"{len(sbs)} sandbox(es) labelled {MANAGED_KEY}={MANAGED_VALUE}")
    for s in sbs:
        say(f"  {s['id']}  {s['name']}  {s['state']}  gpu={s.get('gpu_type')}  run={s['labels'].get(RUN_LABEL)}")
    if RUNS.exists():
        say("recent runs:")
        for p in sorted(RUNS.glob("*/run.json"))[-10:]:
            r = json.loads(p.read_text(encoding="utf-8"))
            say(f"  {r['run_id']:<34} {r.get('status'):<11} gpu={r.get('gpu')} job {r.get('job_s', 0) / 60:.1f} min "
                f"ceiling-actual ${r.get('cost_ceiling_actual_usd', 0):.2f}  final={r.get('final_state')}")
    return 0


def cmd_stop(a, env) -> int:
    c = Client(env)
    targets = [s for s in c.list_managed() if s["labels"].get(RUN_LABEL) == a.run_id]
    if not targets:
        say(f"no live sandbox for run {a.run_id}")
    for s in targets:
        say(f"  {s['id']}: {c.stop_delete_verify(s['id'], a.run_id)} (verified)")
    return 0


def cmd_stop_all(a, env) -> int:
    c = Client(env)
    targets = [s for s in c.list_managed() if s["labels"].get(MANAGED_KEY) == MANAGED_VALUE]
    if not targets:
        say("no managed sandboxes")
        return 0
    for s in targets:
        say(f"  {s['id']}  {s['name']}  {s['state']}  run={s['labels'].get(RUN_LABEL)}")
    if not a.yes and input(f"stop and delete these {len(targets)}? type yes: ").strip().lower() != "yes":
        raise LauncherError("not confirmed")
    for s in targets:
        say(f"  {s['id']}: {c.stop_delete_verify(s['id'], s['labels'].get(RUN_LABEL, ''))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", type=Path, help="dotenv holding DAYTONA_API_KEY (default: $OWF_CLOUD_ENV or ./.env)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    sub.add_parser("list")

    def common(p, ttl: int, cpu: int, mem: int, disk: int):
        p.add_argument("--gpu", default=",".join(DEFAULT_GPU_PREF))
        p.add_argument("--cpu", type=int, default=cpu)
        p.add_argument("--memory", type=int, default=mem, help="GiB")
        p.add_argument("--disk", type=int, default=disk, help="GiB")
        p.add_argument("--ttl-minutes", type=int, default=ttl, help="hard wall-clock fuse")
        p.add_argument("--max-cost", type=float, default=5.0, help="refuse if the TTL ceiling exceeds this (USD)")
        p.add_argument("--run-id")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--yes", action="store_true")
        p.add_argument("--force", action="store_true", help="replace existing local outputs")

    r = sub.add_parser("render", help="TTS + augment + Parakeet + assemble for script files")
    r.add_argument("--files", nargs="+", required=True, help="script files in datasets/scripts-v2, e.g. W9 W10 W11")
    r.add_argument("--per-file", type=int, default=0, help="pilot: first N scripts of each file")
    r.add_argument("--apply", action="store_true", help="copy rows into datasets/real-v2/ and write audit packets")
    r.add_argument("--piper-workers", type=int)
    r.add_argument("--kokoro-workers", type=int)
    r.add_argument("--parakeet-workers", type=int)
    r.add_argument("--parakeet-threads", type=int)
    common(r, ttl=90, cpu=8, mem=32, disk=40)

    t = sub.add_parser("train", help="LoRA train + Q4_K_M export + eval_gguf")
    t.add_argument("--run", required=True, help="run name: training/refine/output/<run>/")
    t.add_argument("--base", choices=sorted(BASES), default="2b")
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--train-file", action="append", help="relative to training/refine/ (globs ok); repeatable")
    t.add_argument("--dpo-file", action="append",
                   help="preference pairs for the DPO stage after SFT (sets dpo_files; dpo_* keys via --set)")
    t.add_argument("--eval", action="append", help="name=path relative to training/refine/; repeatable "
                                                   "(default synth=synth-v1/eval, heldout=real-v2/final/heldout)")
    t.add_argument("--quants", default="Q4_K_M")
    t.add_argument("--set", action="append", help="extra train.py --set key=value; repeatable")
    t.add_argument("--no-rust-check", dest="rust_check", action="store_false",
                   help="skip train.py's byte-parity check against the Rust renderer (saves a cargo build)")
    common(t, ttl=120, cpu=8, mem=48, disk=80)

    b = sub.add_parser("bench-cpu", help="CPU-only refine latency, llama.cpp CPU build (no GPU)")
    b.add_argument("--models", nargs="+", default=["v3-0.8b", "v3-2b"],
                   help="run names under training/refine/output/ (their Q4_K_M) or .gguf paths")
    b.add_argument("--eval", default="datasets/real-v3/final/heldout.jsonl", help="relative to training/refine/")
    b.add_argument("--limit", type=int, default=60)
    b.add_argument("--configs", nargs="+", help="label:cpus:threads (0 = all / app default); "
                                                "default <cpu>vcpu:0:0 <cpu>vcpu-t<cpu/2>:0:<cpu/2> (+ 4vcpu-pinned:4:0 above 4)")
    b.add_argument("--gpu-host", action="store_true",
                   help="take a GPU spot sandbox just for its bigger CPU shape (CPU-only ones max out at 4 vCPU)")
    common(b, ttl=60, cpu=4, mem=8, disk=10)   # CPU-only sandbox caps: 4 vCPU, 10 GB disk

    e = sub.add_parser("eval", help="existing GGUFs (HF or local) through eval_gguf.py on eval sets")
    e.add_argument("--models", nargs="+", required=True,
                   help="name=hf:<repo>/<file> (downloaded in the sandbox) | name=<run> | name=<path.gguf> (uploaded)")
    e.add_argument("--eval", action="append", required=True, help="name=path relative to training/refine/; repeatable")
    e.add_argument("--chunk-modes", nargs="+", choices=["off", "on"], default=["off"],
                   help="'on' also runs chunked refinement (<name>-chunk-<eval>; docs/refine-chunking.md)")
    common(e, ttl=90, cpu=8, mem=32, disk=60)

    for name in ("fetch", "stop"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        if name == "fetch":
            p.add_argument("--apply", action="store_true")
            p.add_argument("--force", action="store_true")
    s = sub.add_parser("stop-all-managed")
    s.add_argument("--yes", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # sandbox logs carry tqdm glyphs; never die on a cp1252 console
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    a = build_parser().parse_args(argv)
    if hasattr(a, "epochs") and float(a.epochs).is_integer():
        a.epochs = int(a.epochs)
    env = load_env(a.env_file)
    fn = {"preflight": cmd_preflight, "render": cmd_render, "train": cmd_train, "list": cmd_list,
          "bench-cpu": cmd_bench_cpu, "eval": cmd_eval, "fetch": cmd_fetch, "stop": cmd_stop, "stop-all-managed": cmd_stop_all}[a.cmd]
    try:
        return fn(a, env)
    except LauncherError as exc:
        say(f"error: {exc}")
        return 3 if isinstance(exc, CapacityError) else 2


if __name__ == "__main__":
    sys.exit(main())

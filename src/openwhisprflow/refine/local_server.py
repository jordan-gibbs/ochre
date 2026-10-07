"""Managed ``llama-server`` for local refinement (SPEC §5.2).

We ship no native code: on first use we download a prebuilt llama.cpp release for this OS / CPU
architecture / accelerator from GitHub into ``models_dir()/llama.cpp/<tag>/<variant>/`` and the
GGUF from Hugging Face, then run ``llama-server`` on a free 127.0.0.1 port. The process stays up
(model resident, KV prompt cache warm) for the life of the app, is restarted if it crashes, and
is killed with us: a Windows job object / Linux parent-death signal makes sure an orphaned server
never keeps 0.5-3 GB of RAM or VRAM after the app exits.

Accelerator choice (``accel="auto"``): Metal on Apple Silicon (built into the macOS build); on
Windows/Linux with an NVIDIA driver the CUDA build (CUDA 13 for drivers >= 580, else CUDA 12);
otherwise the CPU build. ``vulkan`` is selectable for AMD/Intel GPUs. If an accelerated server
fails to start we fall back to the CPU build once, so a broken GPU driver degrades to "slower",
not "refinement off".

Nothing here sends the user's text anywhere but 127.0.0.1, and the server log never contains
prompts (we do not pass ``--verbose``).
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .. import config, models

log = logging.getLogger("openwhisprflow.refine.server")

LLAMA_REPO = "ggml-org/llama.cpp"
DEFAULT_TAG = "b11398"            # pinned: tested Oct 2026; bump deliberately and re-run bench_refine
RELEASE_API = f"https://api.github.com/repos/{LLAMA_REPO}/releases/tags/{{tag}}"
DOWNLOAD = f"https://github.com/{LLAMA_REPO}/releases/download/{{tag}}/{{name}}"

# Minimum NVIDIA driver for each CUDA major the release ships builds for.
CUDA_MIN_DRIVER = {13: 580, 12: 528}

Progress = Callable[[str, int, int], None]


# ---------------------------------------------------------------- build selection

@dataclass(frozen=True)
class Build:
    """One llama.cpp binary variant: the archives to fetch and where they unpack."""

    variant: str                  # e.g. "win-cuda-13.4-x64"
    accel: str                    # cpu | cuda | vulkan | metal
    assets: tuple[str, ...]       # archive file names in the GitHub release


def host() -> tuple[str, str]:
    """(os, arch) in llama.cpp release naming: os in win|macos|ubuntu, arch in x64|arm64."""
    os_name = {"win32": "win", "darwin": "macos"}.get(sys.platform, "ubuntu")
    m = platform.machine().lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "x64"
    return os_name, arch


def nvidia_driver_major() -> int | None:
    """Major version of the NVIDIA driver, or None if there is no NVIDIA GPU/driver."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=5, creationflags=_NO_WINDOW).stdout
        return int(out.strip().splitlines()[0].split(".")[0])
    except Exception:
        return None


def pick_accel(accel: str = "auto") -> str:
    if accel != "auto":
        return accel
    os_name, arch = host()
    if os_name == "macos":
        return "metal" if arch == "arm64" else "cpu"
    if nvidia_driver_major() is not None and arch == "x64":
        return "cuda"
    return "cpu"


def select_build(assets: list[str], tag: str, accel: str, *, os_name: str | None = None, arch: str | None = None,
                 driver_major: int | None = None) -> Build:
    """Choose the archives for this host from a release's asset names.

    Matching names by pattern (not hard-coding them) lets a tag bump pick up new CUDA versions.
    """
    if os_name is None or arch is None:
        os_name, arch = host()
    names = set(assets)

    def need(name: str) -> str:
        if name not in names:
            raise RuntimeError(f"llama.cpp {tag} has no {name}")
        return name

    if os_name == "macos":
        return Build(f"macos-{arch}", "metal" if arch == "arm64" else "cpu",
                     (need(f"llama-{tag}-bin-macos-{arch}.tar.gz"),))
    ext = "zip" if os_name == "win" else "tar.gz"
    if accel == "cuda":
        drv = driver_major if driver_major is not None else (nvidia_driver_major() or 0)
        pat = re.compile(rf"^llama-{re.escape(tag)}-bin-{os_name}-cuda-(\d+)\.(\d+)-{arch}\.{re.escape(ext)}$")
        options = sorted(((int(m[1]), int(m[2]), m[0]) for a in assets if (m := pat.match(a))), reverse=True)
        for major, minor, name in options:
            if drv >= CUDA_MIN_DRIVER.get(major, 10_000):
                extra: tuple[str, ...] = ()
                if os_name == "win":  # Windows builds need the CUDA runtime DLLs next to the exe
                    extra = (need(f"cudart-llama-bin-win-cuda-{major}.{minor}-{arch}.zip"),)
                return Build(f"{os_name}-cuda-{major}.{minor}-{arch}", "cuda", (name, *extra))
        raise RuntimeError(f"no CUDA build of llama.cpp {tag} supports NVIDIA driver {drv}")
    if accel == "vulkan":
        return Build(f"{os_name}-vulkan-{arch}", "vulkan", (need(f"llama-{tag}-bin-{os_name}-vulkan-{arch}.{ext}"),))
    cpu = f"llama-{tag}-bin-{os_name}-cpu-{arch}.{ext}" if os_name == "win" else f"llama-{tag}-bin-{os_name}-{arch}.{ext}"
    return Build(f"{os_name}-cpu-{arch}", "cpu", (need(cpu),))


def release_assets(tag: str) -> list[str]:
    resp = httpx.get(RELEASE_API.format(tag=tag), headers={"User-Agent": models.USER_AGENT,
                                                           "Accept": "application/vnd.github+json"},
                     timeout=20, follow_redirects=True)
    resp.raise_for_status()
    return [a["name"] for a in resp.json().get("assets", [])]


def server_exe(root: Path) -> Path | None:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    hits = sorted(root.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def ensure_binary(accel: str = "auto", *, tag: str = DEFAULT_TAG, root: Path | None = None,
                  progress: Progress | None = None, cancel: Callable[[], bool] | None = None) -> tuple[Path, str]:
    """Return (path to llama-server, accel actually installed), downloading on first use."""
    accel = pick_accel(accel)
    base = (root or config.models_dir() / "llama.cpp") / tag
    # Fast path: an already-installed variant for this accel needs no network.
    for d in sorted(base.iterdir()) if base.exists() else []:
        marker = d / ".complete"
        if marker.is_file() and marker.read_text().split("\n")[0] == accel and (exe := server_exe(d)):
            return exe, accel
    build = select_build(release_assets(tag), tag, accel)
    dest = base / build.variant
    if not (dest / ".complete").exists():
        dest.mkdir(parents=True, exist_ok=True)
        for name in build.assets:
            archive = models.ensure_file(DOWNLOAD.format(tag=tag, name=name), base / name, progress=progress,
                                         cancel=cancel, timeout=120)
            _extract(archive, dest)
        exe = server_exe(dest)
        if exe is None:
            raise RuntimeError(f"llama-server not found in {build.variant}")
        if sys.platform != "win32":
            for f in exe.parent.iterdir():
                if f.is_file() and (f.name.startswith("llama-") or f.suffix in (".so", ".dylib", "")):
                    f.chmod(f.stat().st_mode | 0o755)
        (dest / ".complete").write_text(build.accel + "\n" + "\n".join(build.assets))
        for name in build.assets:  # archives are only needed until extracted
            (base / name).unlink(missing_ok=True)
    exe = server_exe(dest)
    assert exe is not None
    return exe, build.accel


def _extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():   # refuse path traversal
                if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                    raise RuntimeError(f"unsafe path in {archive.name}: {member}")
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # Python < 3.11.4
                t.extractall(dest)


# ---------------------------------------------------------------- the server process

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def default_threads() -> int:
    """All logical cores. On a Ryzen 7 9800X3D (8C/16T) under heavy background load, 16 threads
    beat 8 and 4 for both prompt (1270 vs 914 vs 579 tok/s) and generation; re-check on an idle
    machine before trusting this for laptops (unverified there)."""
    return max(2, os.cpu_count() or 4)


@dataclass
class ServerOptions:
    ctx: int = 2048                # a 30 s phrase is ~120 words ~ 200 tokens; system prompt ~400
    threads: int = field(default_factory=default_threads)
    gpu_layers: int | None = None  # None: all layers on GPU builds, 0 on CPU builds
    flash_attn: str = "auto"       # on | off | auto
    batch: int = 1024
    extra: tuple[str, ...] = ()


class LlamaServer:
    """One ``llama-server`` process serving one GGUF, restarted on crash."""

    def __init__(self, exe: Path, model: Path, *, accel: str, options: ServerOptions | None = None,
                 log_path: Path | None = None) -> None:
        self.exe, self.model, self.accel = exe, model, accel
        self.options = options or ServerOptions()
        self.log_path = log_path or config.data_dir() / "logs" / "llama-server.log"
        self.port = 0
        self.proc: subprocess.Popen[bytes] | None = None
        self.restarts = 0
        self._lock = threading.RLock()
        self._client: httpx.Client | None = None
        self._job: Any = None
        self._stopping = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def args(self) -> list[str]:
        o = self.options
        ngl = o.gpu_layers if o.gpu_layers is not None else (0 if self.accel == "cpu" else 999)
        args = [str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(o.ctx), "-np", "1", "-t", str(o.threads), "-ngl", str(ngl), "-fa", o.flash_attn,
                "-b", str(o.batch), "--no-webui", "--no-jinja", "--cache-prompt", "--no-context-shift",
                "--fit", "off",
                # Prompts never repeat exactly, so the RAM prompt cache only costs memory (8 GiB cap).
                "--cache-ram", "0",
                # Refinement is a short burst on the user's critical path: under load (a compile, a
                # game) normal priority made generation 2-3x slower in bench_refine.py.
                "--prio", "2", "--prio-batch", "2"]
        if self.accel == "cpu":
            # The output mostly copies the input, so prompt-lookup speculation roughly doubles CPU
            # generation speed (bench: 2.45 s -> 1.26 s per 100 words). It slows GPUs, so CPU only.
            args += ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "3",
                     "--spec-ngram-simple-size-m", "16"]
        return args + list(o.extra)

    # -- lifecycle
    def start(self, timeout: float = 90.0) -> None:
        """Start (or restart) the process and block until ``/health`` says the model is loaded."""
        with self._lock:
            self._stopping = False
            if self.alive():
                return
            self.port = free_port()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(self.log_path, "ab")
            kw: dict[str, Any] = {"stdout": logf, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                                  "cwd": str(self.exe.parent)}
            if sys.platform == "win32":
                kw["creationflags"] = _NO_WINDOW
            elif sys.platform.startswith("linux"):
                kw["preexec_fn"] = _die_with_parent
            log.info("starting llama-server (%s, %s) on port %d", self.accel, self.model.name, self.port)
            self.proc = subprocess.Popen(self.args(), **kw)
            logf.close()
            self._job = _kill_on_close(self.proc)
            self._wait_healthy(timeout)

    def _wait_healthy(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                code = self.proc.returncode if self.proc else None
                raise RuntimeError(f"llama-server exited during startup (code {code}); see {self.log_path}")
            if self.healthy():
                return
            time.sleep(0.05)
        self.stop()
        raise RuntimeError(f"llama-server did not become healthy in {timeout:.0f}s; see {self.log_path}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def healthy(self) -> bool:
        try:
            return self.client.get(f"{self.url}/health", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False

    def ensure_running(self, *, wait: bool = True) -> bool:
        """Restart a crashed server. With ``wait=False`` the restart runs in the background and this
        returns False, so a refine call can fall back to raw text instead of blocking for seconds."""
        if self.alive():
            return True
        if self._stopping:
            return False
        self.restarts += 1
        log.warning("llama-server not running (restart #%d)", self.restarts)
        if wait:
            self.start()
            return True
        threading.Thread(target=self._bg_start, name="llama-restart", daemon=True).start()
        return False

    def _bg_start(self) -> None:
        try:
            self.start()
        except Exception:
            log.exception("llama-server restart failed")

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            p, self.proc = self.proc, None
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(5)
            if self._client is not None:
                self._client.close()
                self._client = None
            if self._job is not None:
                _close_job(self._job)
                self._job = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    # -- inference
    def completion(self, prompt: str, *, n_predict: int, timeout: float, stop: list[str] | None = None,
                   **params: Any) -> dict[str, Any]:
        """Raw ``/completion``: greedy, prompt-cached. Returns the server JSON (``content``, ``timings``)."""
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "cache_prompt": True,
                "stop": stop or [], "stream": False, **params}
        resp = self.client.post(f"{self.url}/completion", json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------- orphan protection

def _die_with_parent() -> None:  # pragma: no cover - Linux only, runs in the child
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _kill_on_close(proc: subprocess.Popen[bytes]) -> Any:
    """Windows: put the child in a job object that kills it when our last handle closes (i.e. when
    this process dies, however it dies). Elsewhere, an atexit hook covers normal exits."""
    if sys.platform != "win32":
        atexit.register(lambda: proc.poll() is None and proc.kill())
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _Io(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rt", "wt", "ot")]

        class _Ext(ctypes.Structure):
            _fields_ = [("Basic", _Basic), ("Io", _Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = _Ext()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # ExtendedLimitInformation
        handle = k32.OpenProcess(0x0001 | 0x0100, False, proc.pid)  # PROCESS_TERMINATE | PROCESS_SET_QUOTA
        k32.AssignProcessToJobObject(job, handle)
        k32.CloseHandle(handle)
        return job
    except Exception:
        log.debug("job object setup failed", exc_info=True)
        atexit.register(lambda: proc.poll() is None and proc.kill())
        return None


def _close_job(job: Any) -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.WinDLL("kernel32").CloseHandle(job)
        except Exception:
            pass


# ---------------------------------------------------------------- models

QUILL_REPO = "Quobi/Quill"
# name -> (file, size, sha256) from the HF API, Oct 2026. Size + hash make the download verifiable.
QUILL: dict[str, tuple[str, int, str]] = {
    "quill-0.8b": ("quill-0.8b-Q4_K_M.gguf", 529296832,
                   "aa54d6f6108d66e4b60a57bdc04ecca6e84e073504918a64b41ac4a0f816f16d"),
    "quill-2b": ("quill-2b-Q4_K_M.gguf", 1274396096,
                 "b877a22b773d2aac40b3c642c24f1cbbb0b3f1d42cbd3c6eb936533719317196"),
    "quill-4b": ("quill-4b-Q4_K_M.gguf", 2708803936,
                 "e5e6bd7e92690c6f954399c473e740561d9deff0862e1bfe42c1f6055535b987"),
}
DEFAULT_MODEL = "quill-0.8b"


def ensure_model(name: str, *, progress: Progress | None = None, cancel: Callable[[], bool] | None = None,
                 root: Path | None = None) -> Path:
    """Path to a local GGUF: a known Quill tier (downloaded on first use) or an explicit file path."""
    name = name or DEFAULT_MODEL
    if name.lower().endswith(".gguf"):
        p = Path(name).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"model file not found: {p}")
        return p
    key = name.lower().removesuffix("-q4_k_m")
    if key not in QUILL:
        raise ValueError(f"unknown local refinement model {name!r}; choose one of {', '.join(QUILL)} or a .gguf path")
    file, size, sha = QUILL[key]
    dest = (root or config.models_dir() / "refine") / file
    if dest.is_file() and dest.stat().st_size == size:
        return dest   # hashing 0.5-2.7 GB on every launch is too slow; size + atomic rename suffice
    return models.ensure_file(models.hf_url(QUILL_REPO, file), dest, sha256=sha, size=size, progress=progress,
                              cancel=cancel)

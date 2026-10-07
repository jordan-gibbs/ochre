"""Launch and supervise the Electron shell (``ui/``).

- It finds Electron in ``ui/node_modules/electron`` and explains what to run if it isn't installed.
- It starts the shell with ``--port``, ``--parent-pid`` and ``--theme``. The websocket token goes through
  the ``OWF_UI_TOKEN`` environment variable, never argv, because argv is visible to every user in ``ps``.
- It restarts the shell after a crash (non-zero exit) with exponential backoff. The backoff resets once the
  shell has stayed up for a while. A clean exit (code 0, the user chose Quit) is not restarted.
- ``stop()`` terminates the shell and then kills it if needed. The shell also quits on its own when our
  pid disappears.

Everything runs on plain threads, so it works whether the core is synchronous or asyncio.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger("openwhisprflow.ui_bridge.supervisor")


class UiUnavailable(RuntimeError):
    """The shell can't be launched (no ui/ checkout, npm install not run, Electron download skipped)."""


def default_ui_dir() -> Path:
    """``OWF_UI_DIR`` if set, else ``<repo>/ui`` next to ``src/``."""
    env = os.environ.get("OWF_UI_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "ui"


def find_electron(ui_dir: Path) -> Path:
    """Path of the Electron executable for ``ui_dir``, or UiUnavailable with the command to fix it."""
    ui_dir = Path(ui_dir)
    if not (ui_dir / "package.json").is_file():
        raise UiUnavailable(f"No UI found at {ui_dir}. Set OWF_UI_DIR, or run headless with --headless.")
    pkg = ui_dir / "node_modules" / "electron"
    if not pkg.is_dir():
        raise UiUnavailable(f"The UI's dependencies aren't installed. Run `npm install` in {ui_dir} "
                            "(needs Node.js 20+), or run headless with --headless.")
    # electron's postinstall writes the platform's executable path (relative to dist/) into path.txt
    rel = None
    path_txt = pkg / "path.txt"
    if path_txt.is_file():
        rel = path_txt.read_text(encoding="utf-8").strip()
    if not rel:
        rel = {"win32": "electron.exe", "darwin": "Electron.app/Contents/MacOS/Electron"}.get(sys.platform, "electron")
    exe = pkg / "dist" / rel
    if not exe.is_file():
        raise UiUnavailable(f"Electron's binary wasn't downloaded (npm sometimes skips it). Run "
                            f"`node node_modules/electron/install.js` in {ui_dir}.")
    return exe


class Supervisor:
    """Keep one Electron shell running while the core runs."""

    def __init__(self, port: int, token: str, theme: str = "system", *, ui_dir: Path | None = None,
                 electron: str | Path | Sequence[str] | None = None, extra_args: Sequence[str] = (),
                 backoff_min: float = 1.0, backoff_max: float = 30.0, stable_after: float = 30.0) -> None:
        self.port = port
        self.token = token
        self.theme = theme
        self.ui_dir = Path(ui_dir) if ui_dir else default_ui_dir()
        self._electron = electron          # tests pass [python, fake_shell.py]
        self.extra_args = list(extra_args)
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max
        self.stable_after = stable_after
        self.problem: str | None = None    # why the shell isn't running, for the CLI / doctor
        self.restarts = 0
        self._proc: subprocess.Popen | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def command(self) -> list[str]:
        if self._electron is None:
            prefix = [str(find_electron(self.ui_dir))]
        elif isinstance(self._electron, (str, Path)):
            prefix = [str(self._electron)]
        else:
            prefix = [str(x) for x in self._electron]
        return [*prefix, str(self.ui_dir), f"--port={self.port}", f"--parent-pid={os.getpid()}",
                f"--theme={self.theme}", *self.extra_args]

    def start(self) -> bool:
        """Launch the shell and watch it. Returns False (and sets ``problem``) when it can't run."""
        if self._thread and self._thread.is_alive():
            return True
        try:
            cmd = self.command()
        except UiUnavailable as e:
            self.problem = str(e)
            log.warning("UI not started: %s", e)
            return False
        self.problem = None
        self._stopping.clear()
        if not self._spawn(cmd):
            return False
        self._thread = threading.Thread(target=self._watch, args=(cmd,), name="owf-ui-supervisor", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 3.0) -> None:
        self._stopping.set()
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout)
        if self._thread:
            self._thread.join(timeout)
            self._thread = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._proc.pid if self._proc else None

    # ------------------------------------------------------------------ internals

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OWF_UI_TOKEN"] = self.token
        env.pop("ELECTRON_RUN_AS_NODE", None)   # set by some IDEs/tools; would turn Electron into plain node
        return env

    def _spawn(self, cmd: list[str]) -> bool:
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(cmd, cwd=str(self.ui_dir) if self.ui_dir.is_dir() else None, env=self._env(),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    creationflags=flags)
        except OSError as e:
            self.problem = f"Could not launch the UI ({e})."
            log.error("%s", self.problem)
            return False
        with self._lock:
            self._proc = proc
        threading.Thread(target=self._pump, args=(proc,), name="owf-ui-log", daemon=True).start()
        log.info("UI started (pid %d)", proc.pid)
        return True

    @staticmethod
    def _pump(proc: subprocess.Popen) -> None:
        """Forward the shell's stdout to our log; it never carries the token."""
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    log.info("%s", line)
        except (OSError, ValueError):
            pass

    def _watch(self, cmd: list[str]) -> None:
        backoff = self.backoff_min
        while not self._stopping.is_set():
            with self._lock:
                proc = self._proc
            if proc is None:
                return
            started = time.monotonic()
            code = proc.wait()
            if self._stopping.is_set():
                return
            if code == 0:
                log.info("UI exited cleanly; not restarting")
                return
            if time.monotonic() - started >= self.stable_after:
                backoff = self.backoff_min
            log.warning("UI exited with code %s; restarting in %.0f s", code, backoff)
            if self._stopping.wait(backoff):
                return
            backoff = min(backoff * 2, self.backoff_max)
            self.restarts += 1
            if not self._spawn(cmd):
                return

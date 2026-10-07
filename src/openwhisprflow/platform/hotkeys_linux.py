"""Linux Voice key listeners: X11 (pynput), Wayland (evdev), and the ``toggle`` socket fallback.

Neither backend can *swallow* a key, so the Voice key (and Escape) also reach the focused app:

* **X11** (``pynput``, XRecord): sees every key. Pick a key that does nothing on its own
  (Right Ctrl, F13-F24, Pause, Scroll Lock). Right Alt is fine with most layouts (it is AltGr /
  ISO_Level3_Shift and does nothing alone) but a lone Alt tap may show menu bars in some apps.
* **Wayland** (``/dev/input/event*``): Wayland gives clients no global key access, so we read
  the kernel's input devices directly. That needs membership in the ``input`` group
  (``sudo usermod -aG input $USER``, then log out and back in), which is a real privilege: any
  member can read all keystrokes. Devices are re-scanned every few seconds for hot-plugging.
* **Toggle** (always on): ``openwhisprflow toggle`` sends a line to a Unix socket; bind it to a
  shortcut in the compositor (GNOME Settings > Keyboard > Custom Shortcuts, ``bindsym`` in Sway,
  ``bind`` in Hyprland). It starts a locked recording or finishes the running one, which works
  everywhere without any permission. :func:`send_toggle` is what the CLI calls.
"""

from __future__ import annotations

import glob
import logging
import os
import select
import socket
import struct
import threading
from pathlib import Path
from typing import Any

from openwhisprflow.config import HotkeyConfig
from openwhisprflow.platform.base import Gesture, GestureFn
from openwhisprflow.platform.driver import GestureDriver
from openwhisprflow.platform.keys import EVDEV_CODE, KeySpec, evdev_name

log = logging.getLogger("openwhisprflow.hotkeys.linux")


class HotkeyUnavailable(RuntimeError):
    """The key backend cannot run here; ``str(err)`` says how to fix it."""


TOGGLE_HINT = ("Or bind `openwhisprflow toggle` to a shortcut in your desktop's keyboard settings: "
               "it starts and finishes a recording without any special permission.")

# --------------------------------------------------------------------------- evdev (Wayland)

EV_KEY = 1
INPUT_EVENT = struct.Struct("llHHi")  # struct input_event: timeval, type, code, value
_IOC_READ = 2


def _eviocgname(length: int = 256) -> int:
    return (_IOC_READ << 30) | (length << 16) | (ord("E") << 8) | 0x06


def parse_events(data: bytes) -> list[tuple[int, int]]:
    """(code, value) for each EV_KEY event in a read() buffer; value 0 up, 1 down, 2 repeat."""
    out = []
    size = INPUT_EVENT.size
    for off in range(0, len(data) - size + 1, size):
        _, _, etype, code, value = INPUT_EVENT.unpack_from(data, off)
        if etype == EV_KEY:
            out.append((code, value))
    return out


def _validate_evdev(spec: KeySpec) -> None:
    if spec.key not in EVDEV_CODE:
        raise ValueError(f"{spec.key!r} cannot be used as the Voice key on Linux")


class EvdevListener:
    """Reads key events from every readable ``/dev/input/event*`` device (no grab)."""

    RESCAN_S = 5.0
    SKIP_NAMES = ("ydotoold", "openwhisprflow")  # our own injected input

    def __init__(self, cfg: HotkeyConfig, device_glob: str = "/dev/input/event*") -> None:
        self._driver = GestureDriver(cfg, validate=_validate_evdev, replay_taps=False)
        self._glob = device_glob
        self._fds: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._stop_r, self._stop_w = -1, -1

    def check(self) -> None:
        """Raise HotkeyUnavailable unless at least one input device is readable."""
        paths = glob.glob(self._glob)
        if not paths:
            raise HotkeyUnavailable(f"No input devices found at {self._glob}. {TOGGLE_HINT}")
        if not any(os.access(p, os.R_OK) for p in paths):
            raise HotkeyUnavailable(
                "Reading the Voice key on Wayland needs access to /dev/input. Run "
                "`sudo usermod -aG input $USER`, then log out and back in. " + TOGGLE_HINT)

    def start(self, on_gesture: GestureFn) -> None:
        self.check()
        self._driver.start(on_gesture)
        self._stop_r, self._stop_w = os.pipe()
        self._rescan()
        self._thread = threading.Thread(target=self._run, name="owf-evdev", daemon=True)
        self._thread.start()

    def set_key(self, key: str) -> None:
        self._driver.set_key(key)

    def set_recording(self, recording: bool) -> None:
        self._driver.set_recording(recording)

    def stop(self) -> None:
        if self._stop_w >= 0:
            os.write(self._stop_w, b"x")
        if self._thread is not None:
            self._thread.join(2.0)
        self._thread = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        for fd in (self._stop_r, self._stop_w):
            if fd >= 0:
                os.close(fd)
        self._stop_r = self._stop_w = -1
        self._driver.stop()

    def _rescan(self) -> None:
        for path in glob.glob(self._glob):
            if path in self._fds:
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            if self._skip(fd):
                os.close(fd)
                continue
            self._fds[path] = fd

    def _skip(self, fd: int) -> bool:
        try:
            import fcntl

            raw = fcntl.ioctl(fd, _eviocgname(), bytes(256))
            name = raw.split(b"\0", 1)[0].decode(errors="replace").lower()
        except OSError:
            return False
        return any(s in name for s in self.SKIP_NAMES)

    def _run(self) -> None:
        import time

        next_scan = time.monotonic() + self.RESCAN_S
        while True:
            fds = list(self._fds.items())
            try:
                ready, _, _ = select.select([fd for _, fd in fds] + [self._stop_r], [], [], 1.0)
            except (OSError, ValueError):
                ready = []
            if self._stop_r in ready:
                return
            for path, fd in fds:
                if fd not in ready:
                    continue
                try:
                    data = os.read(fd, INPUT_EVENT.size * 64)
                except BlockingIOError:
                    continue
                except OSError:  # unplugged
                    os.close(fd)
                    self._fds.pop(path, None)
                    continue
                for code, value in parse_events(data):
                    self._driver.feed(evdev_name(code), value != 0)
            if time.monotonic() >= next_scan:
                next_scan = time.monotonic() + self.RESCAN_S
                self._rescan()


# --------------------------------------------------------------------------- X11 (pynput)

_PYNPUT_NAMES = {
    "alt_r": "right_alt", "alt_gr": "right_alt", "alt": "left_alt", "alt_l": "left_alt",
    "ctrl": "left_ctrl", "ctrl_l": "left_ctrl", "ctrl_r": "right_ctrl",
    "shift": "left_shift", "shift_l": "left_shift", "shift_r": "right_shift",
    "cmd": "left_win", "cmd_l": "left_win", "cmd_r": "right_win", "esc": "escape",
}
_XK_F1 = 0xFFBE


def pynput_name(key: Any) -> str:
    """Our key name for a pynput ``Key`` / ``KeyCode``."""
    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        return _PYNPUT_NAMES.get(name, name)
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    vk = getattr(key, "vk", None)
    if isinstance(vk, int) and _XK_F1 <= vk < _XK_F1 + 35:
        return f"f{vk - _XK_F1 + 1}"
    return f"x{vk}"


class X11Listener:
    """pynput (XRecord) listener; cannot suppress keys on X11."""

    def __init__(self, cfg: HotkeyConfig) -> None:
        self._driver = GestureDriver(cfg, validate=_validate_evdev, replay_taps=False)
        self._listener: Any = None

    def start(self, on_gesture: GestureFn) -> None:
        try:
            from pynput import keyboard
        except Exception as e:  # ImportError, or Xlib failing to reach the display
            raise HotkeyUnavailable(f"pynput is not usable here ({e}). {TOGGLE_HINT}") from e
        self._driver.start(on_gesture)
        self._listener = keyboard.Listener(
            on_press=lambda k: self._driver.feed(pynput_name(k), True),
            on_release=lambda k: self._driver.feed(pynput_name(k), False))
        self._listener.start()

    def set_key(self, key: str) -> None:
        self._driver.set_key(key)

    def set_recording(self, recording: bool) -> None:
        self._driver.set_recording(recording)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._driver.stop()


# --------------------------------------------------------------------------- toggle socket


def toggle_socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "openwhisprflow" / "toggle.sock"
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(f"/tmp/openwhisprflow-{uid}") / "toggle.sock"


TOGGLE_ACTIONS = ("toggle", "start", "stop", "cancel")


def send_toggle(action: str = "toggle", path: Path | None = None, timeout: float = 2.0) -> None:
    """What ``openwhisprflow toggle`` runs: ask the running app to start/finish/cancel."""
    if action not in TOGGLE_ACTIONS:
        raise ValueError(f"action must be one of {TOGGLE_ACTIONS}")
    path = path or toggle_socket_path()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(path))
            s.sendall(action.encode() + b"\n")
            s.recv(16)
    except OSError as e:
        raise ConnectionError(f"Open Whisperflow does not seem to be running ({path}: {e})") from e


class ToggleListener:
    """Turns ``toggle``/``start``/``stop``/``cancel`` lines on a Unix socket into gestures."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or toggle_socket_path()
        self._active = False
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._on_gesture: GestureFn | None = None

    def start(self, on_gesture: GestureFn) -> None:
        self._on_gesture = on_gesture
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.exists():
            self.path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        sock.listen(4)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="owf-toggle", daemon=True)
        self._thread.start()

    def set_key(self, key: str) -> None:
        pass  # the shortcut lives in the compositor

    def set_recording(self, recording: bool) -> None:
        with self._lock:
            self._active = recording

    def stop(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()
        if self._thread is not None:
            self._thread.join(2.0)
        self._thread = None
        try:
            self.path.unlink()
        except OSError:
            pass

    def handle(self, action: str) -> list[Gesture]:
        """Gestures for one action (also the unit-test seam)."""
        with self._lock:
            if action == "toggle":
                action = "stop" if self._active else "start"
            if action == "start" and not self._active:
                self._active = True
                return [Gesture.PRESS, Gesture.LOCK]
            if action == "stop" and self._active:
                self._active = False
                return [Gesture.FINISH]
            if action == "cancel" and self._active:
                self._active = False
                return [Gesture.CANCEL]
            return []

    def _serve(self) -> None:
        while self._sock is not None:
            try:
                conn, _ = self._sock.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(1.0)
                    action = conn.recv(64).decode(errors="replace").strip()
                    gestures = self.handle(action)
                    conn.sendall(b"ok\n")
                except OSError:
                    continue
            for g in gestures:
                try:
                    if self._on_gesture:
                        self._on_gesture(g)
                except Exception:
                    log.exception("on_gesture(%s) failed", g)


def session_type() -> str:
    """"wayland", "x11" or "" (no graphical session)."""
    st = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if st in ("wayland", "x11"):
        return st
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return ""


def key_listener(cfg: HotkeyConfig, backend: str | None = None) -> EvdevListener | X11Listener:
    """The key backend for this session: OWF_HOTKEY_BACKEND (evdev|x11) overrides."""
    backend = backend or os.environ.get("OWF_HOTKEY_BACKEND", "")
    if backend == "evdev" or (not backend and session_type() == "wayland"):
        return EvdevListener(cfg)
    return X11Listener(cfg)

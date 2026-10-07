"""Is another app using the microphone (a call)? Hands-free listening pauses while one is.

Windows records every microphone user in the privacy consent store:
``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone``
(and the same path under HKLM). Each packaged app has a subkey (``Microsoft.Teams_8wekyb3d8bbwe``), each desktop
app a subkey of ``NonPackaged`` named by its exe path with ``\\`` as ``#`` (``C:#Program Files#Zoom#bin#Zoom.exe``).
``LastUsedTimeStart`` / ``LastUsedTimeStop`` are FILETIMEs; an app is using the mic right now when ``Stop`` is 0
(or older than ``Start``). Browsers show up as the browser (Meet in Chrome = chrome.exe), so a Meet call pauses
it and so would any site recording in that browser.

Limits: it says *that* an app holds the mic, not that it is a call (a voice recorder, OBS or a mic effects app
counts too: ``ignore_apps`` lists the always-on ones); an app that crashed mid-capture may leave a
stale entry (entries started more than ``STALE_H`` hours ago are ignored); Our own Python process is
excluded by exe path (the venv launcher and the base interpreter). No admin rights, no audio access: registry
reads only, polled every few seconds by wake/handsfree.py.

Not Windows: never reports a call (TODO: macOS via the CoreAudio
``kAudioDevicePropertyDeviceIsRunningSomewhere`` property; Linux via PipeWire/PulseAudio
source-outputs, ``pactl list source-outputs``).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

log = logging.getLogger("openwhisprflow.wake.calls")

CONSENT_KEY = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
STALE_H = 12.0  # an "in use" entry started longer ago than this is a crashed app, not a call
_EPOCH_DIFF = 11644473600  # seconds between 1601-01-01 (FILETIME) and 1970-01-01


@dataclass(frozen=True, slots=True)
class MicUse:
    """One consent-store entry: ``app`` = package name or exe path (``\\`` separators), times in FILETIME."""

    app: str
    start: int
    stop: int

    @property
    def active(self) -> bool:
        return self.start > 0 and (self.stop == 0 or self.stop < self.start)

    @property
    def started_unix(self) -> float:
        return self.start / 1e7 - _EPOCH_DIFF if self.start else 0.0

    @property
    def short(self) -> str:
        """Display name: "Zoom", "chrome", "Microsoft.Teams"."""
        a = self.app.replace("/", "\\").rsplit("\\", 1)[-1]
        if a.lower().endswith(".exe"):
            a = a[:-4]
        return a.split("_", 1)[0] if "\\" not in self.app else a


Reader = Callable[[], list[MicUse]]


def read_consent_store() -> list[MicUse]:
    """All microphone consent-store entries (HKCU + HKLM, packaged + NonPackaged). Empty off Windows."""
    if sys.platform != "win32":
        return []
    import winreg

    out: list[MicUse] = []

    def entry(key: object, name: str, app: str) -> None:
        try:
            with winreg.OpenKey(key, name) as k:  # type: ignore[arg-type]
                start = _qword(k, "LastUsedTimeStart")
                stop = _qword(k, "LastUsedTimeStop")
        except OSError:
            return
        if start or stop:
            out.append(MicUse(app=app, start=start, stop=stop))

    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            root = winreg.OpenKey(hive, CONSENT_KEY)
        except OSError:
            continue
        with root:
            for sub in _subkeys(root):
                if sub == "NonPackaged":
                    try:
                        with winreg.OpenKey(root, sub) as np_:
                            for exe in _subkeys(np_):
                                entry(np_, exe, exe.replace("#", "\\"))
                    except OSError:
                        pass
                else:
                    entry(root, sub, sub)
    return out


def _subkeys(key: object) -> list[str]:
    import winreg

    names: list[str] = []
    i = 0
    while True:
        try:
            names.append(winreg.EnumKey(key, i))  # type: ignore[arg-type]
        except OSError:
            return names
        i += 1


def _qword(key: object, name: str) -> int:
    import winreg

    try:
        v, _t = winreg.QueryValueEx(key, name)  # type: ignore[arg-type]
        return int(v or 0)
    except (OSError, ValueError, TypeError):
        return 0


def own_exe_paths() -> set[str]:
    """This process's interpreter paths (venv launcher + base interpreter, resolved), lower-case."""
    paths = {sys.executable, getattr(sys, "_base_executable", "") or ""}
    out: set[str] = set()
    for p in paths:
        if not p:
            continue
        out.add(os.path.normcase(os.path.abspath(p)))
        try:
            out.add(os.path.normcase(os.path.realpath(p)))
        except OSError:
            pass
    return out


class MicUsageMonitor:
    """Which other apps hold the microphone right now (``in_use()``)."""

    def __init__(self, reader: Reader | None = None, *, ignore_apps: Iterable[str] = (),
                 own_paths: Iterable[str] | None = None, now: Callable[[], float] = time.time) -> None:
        self.reader = reader or read_consent_store
        self.ignore = tuple(a.lower() for a in ignore_apps if a)
        self.own = {os.path.normcase(p) for p in (own_paths if own_paths is not None else own_exe_paths())}
        self.now = now
        self._warned = False

    def in_use(self) -> list[MicUse]:
        try:
            entries = self.reader()
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                self._warned = True
                log.warning("microphone usage unavailable (%r): hands-free won't pause for calls", exc)
            return []
        stale = self.now() - STALE_H * 3600
        out: list[MicUse] = []
        for e in entries:
            if not e.active or e.started_unix < stale:
                continue
            low = e.app.lower()
            if os.path.normcase(e.app) in self.own or any(i in low for i in self.ignore):
                continue
            out.append(e)
        return out

"""What the OS still has to grant before hotkeys and typing work, with fix instructions.

``check()`` returns ``{permission_id: human-readable fix}`` for everything *missing*; an empty
dict means good to go. The onboarding UI lists the entries and, on macOS, offers a button per
entry that calls :func:`open_settings` (and :func:`request`, which shows the system prompt).

Microphone access is not covered here (the audio capture owns that check).
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

MAC_PANES = {
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "input_monitoring": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    "screen_recording": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
}

_MAC_WHO = ("the app that runs Open Whisperflow (the packaged app, or your Terminal / iTerm "
            "when started from a shell)")


def check() -> dict[str, str]:
    if sys.platform == "darwin":
        return _check_macos()
    if sys.platform.startswith("linux"):
        return _check_linux()
    return {}  # Windows needs no grants; admin windows are reported per insertion


# --------------------------------------------------------------------------- macOS


def _mac_accessibility() -> bool:
    try:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())
    except Exception:
        return True  # cannot tell: let the event tap be the judge


def _mac_listen_access() -> bool:
    try:
        import Quartz

        fn = getattr(Quartz, "CGPreflightListenEventAccess", None)
        return True if fn is None else bool(fn())
    except Exception:
        return True


def _check_macos() -> dict[str, str]:
    missing: dict[str, str] = {}
    if not _mac_accessibility():
        missing["accessibility"] = (
            "Allow Accessibility: System Settings > Privacy & Security > Accessibility, then turn "
            f"on {_MAC_WHO}. Needed to use the Voice key and to type text.")
    if not _mac_listen_access():
        missing["input_monitoring"] = (
            "Allow Input Monitoring: System Settings > Privacy & Security > Input Monitoring, then "
            f"turn on {_MAC_WHO} and restart it. Needed to see the Voice key.")
    return missing


def request(permission: str) -> None:
    """macOS: show the system prompt for a permission (no-op elsewhere or when granted)."""
    if sys.platform != "darwin":
        return
    try:
        if permission == "accessibility":
            from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt

            AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
        elif permission == "input_monitoring":
            import Quartz

            fn = getattr(Quartz, "CGRequestListenEventAccess", None)
            if fn is not None:
                fn()
    except Exception:
        open_settings(permission)


def open_settings(permission: str = "accessibility") -> bool:
    """macOS: open System Settings at the pane for ``permission``. Returns False elsewhere."""
    url = MAC_PANES.get(permission)
    if sys.platform != "darwin" or url is None:
        return False
    try:
        subprocess.run(["open", url], check=False, timeout=5)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


# --------------------------------------------------------------------------- Linux


def _check_linux() -> dict[str, str]:
    from openwhisprflow.platform.hotkeys_linux import TOGGLE_HINT, session_type

    missing: dict[str, str] = {}
    session = session_type()
    if session == "wayland":
        devices = glob.glob("/dev/input/event*")
        if devices and not any(os.access(p, os.R_OK) for p in devices):
            missing["input_group"] = (
                "To use the Voice key on Wayland, add yourself to the input group: "
                "`sudo usermod -aG input $USER`, then log out and back in. " + TOGGLE_HINT)
        if not (shutil.which("wtype") or shutil.which("ydotool")):
            missing["typing_tool"] = (
                "Install a typing tool: `wtype` (Sway, Hyprland, KDE) or `ydotool` with the "
                "`ydotoold` service running (GNOME).")
        elif not shutil.which("wtype") and not _ydotoold_running():
            missing["ydotoold"] = ("Start the ydotool daemon: `systemctl --user enable --now ydotool` "
                                   "(or run `ydotoold`); ydotool needs it to type.")
        if not shutil.which("wl-copy"):
            missing["clipboard_tool"] = "Install `wl-clipboard` for the paste fallback."
    elif session == "x11":
        if not shutil.which("xdotool"):
            missing["typing_tool"] = "Install `xdotool` so dictated text can be typed."
        if not (shutil.which("xclip") or shutil.which("xsel")):
            missing["clipboard_tool"] = "Install `xclip` for the paste fallback."
    else:
        missing["display"] = "No graphical session found (neither WAYLAND_DISPLAY nor DISPLAY is set)."
    return missing


def _ydotoold_running() -> bool:
    candidates = [os.environ.get("YDOTOOL_SOCKET", ""), "/tmp/.ydotool_socket",
                  f"/run/user/{os.getuid()}/.ydotool_socket" if hasattr(os, "getuid") else ""]
    return any(c and os.path.exists(c) for c in candidates)

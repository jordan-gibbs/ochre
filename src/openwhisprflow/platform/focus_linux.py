"""The focused app on Linux.

* X11: ``_NET_ACTIVE_WINDOW`` through python-xlib (installed with pynput), falling back to
  ``xdotool`` + ``xprop``. The app name is the WM_CLASS class, lowercased.
* Wayland: there is no portable way to ask. Sway (``swaymsg``) and Hyprland (``hyprctl``) are
  supported; elsewhere (GNOME, KDE) FocusInfo is empty, which only means per-app styles and
  the leading-space join rule fall back to their defaults.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from openwhisprflow.platform.base import FocusInfo


def _run(*args: str, timeout: float = 0.5) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_wm_class(xprop_out: str) -> str:
    """'WM_CLASS(STRING) = "slack", "Slack"' -> "slack" (the class, lowercased)."""
    parts = [p.strip().strip('"') for p in xprop_out.partition("=")[2].split(",")]
    parts = [p for p in parts if p]
    return parts[-1].lower() if parts else ""


def _x11_xlib() -> FocusInfo | None:
    try:
        from Xlib import X, display  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        d = display.Display()
        try:
            root = d.screen().root
            active = root.get_full_property(d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType)
            if not active or not active.value:
                return FocusInfo()
            wid = int(active.value[0])
            win = d.create_resource_object("window", wid)
            wm_class = win.get_wm_class() or ("", "")
            name_prop = win.get_full_property(d.intern_atom("_NET_WM_NAME"), 0)
            title = name_prop.value.decode("utf-8", "replace") if name_prop and isinstance(
                name_prop.value, bytes) else (win.get_wm_name() or "")
            return FocusInfo(app_name=(wm_class[-1] or "").lower(), window_title=str(title),
                             window_id=f"x11:{wid:x}")
        finally:
            d.close()
    except Exception:
        return None


def _x11_tools() -> FocusInfo:
    if not shutil.which("xdotool"):
        return FocusInfo()
    wid = _run("xdotool", "getactivewindow").strip()
    if not wid:
        return FocusInfo()
    title = _run("xdotool", "getwindowname", wid).strip()
    app = parse_wm_class(_run("xprop", "-id", wid, "WM_CLASS")) if shutil.which("xprop") else ""
    return FocusInfo(app_name=app, window_title=title, window_id=f"x11:{int(wid):x}")


def _sway_focused(node: dict[str, Any]) -> dict[str, Any] | None:
    if node.get("focused"):
        return node
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        found = _sway_focused(child)
        if found:
            return found
    return None


def parse_sway_tree(tree_json: str) -> FocusInfo:
    try:
        node = _sway_focused(json.loads(tree_json))
    except (ValueError, TypeError):
        return FocusInfo()
    if not node:
        return FocusInfo()
    props = node.get("window_properties") or {}
    app = node.get("app_id") or props.get("class") or ""
    return FocusInfo(app_name=str(app).lower(), window_title=str(node.get("name") or ""),
                     window_id=f"sway:{node.get('id', '')}")


def parse_hyprland(active_json: str) -> FocusInfo:
    try:
        data = json.loads(active_json)
    except (ValueError, TypeError):
        return FocusInfo()
    if not isinstance(data, dict):
        return FocusInfo()
    return FocusInfo(app_name=str(data.get("class") or "").lower(), window_title=str(data.get("title") or ""),
                     window_id=f"hypr:{data.get('address', '')}")


def get_focus() -> FocusInfo:
    from openwhisprflow.platform.hotkeys_linux import session_type

    if session_type() == "wayland":
        if shutil.which("hyprctl"):
            return parse_hyprland(_run("hyprctl", "activewindow", "-j"))
        if shutil.which("swaymsg"):
            return parse_sway_tree(_run("swaymsg", "-t", "get_tree"))
        return FocusInfo()
    return _x11_xlib() or _x11_tools()

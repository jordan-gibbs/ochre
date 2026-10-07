"""Linux text injection through the standard command-line tools (SPEC §6.2).

* **X11**: ``xdotool type --clearmodifiers`` (Unicode-capable; ``--clearmodifiers`` lifts
  modifiers the user still holds and puts them back afterwards).
* **Wayland**: ``wtype`` (virtual-keyboard protocol: wlroots compositors such as Sway and
  Hyprland, and KDE; **not GNOME**), then ``ydotool`` (works everywhere through uinput but needs
  the ``ydotoold`` daemon, and only types characters on the US layout, so non-ASCII text is
  pasted instead).
* **Paste fallback**: ``wl-copy`` / ``xclip`` (or ``xsel``), then Ctrl+V. Terminals usually paste
  with Ctrl+Shift+V instead, so typing is preferred there.

Every runner takes the argument list (never a shell), so dictated text cannot inject commands.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Callable

from openwhisprflow.platform.base import FocusInfo, InjectError
from openwhisprflow.platform.focus_linux import get_focus
from openwhisprflow.platform.textunits import plan

Runner = Callable[[list[str], "str | None"], str]

NO_TOOL_WAYLAND = ("No typing tool found for Wayland. Install `wtype` (Sway, Hyprland, KDE) or "
                   "`ydotool` with its `ydotoold` service running (GNOME and everything else). "
                   "The text is in History.")
NO_TOOL_X11 = "No typing tool found. Install `xdotool`. The text is in History."
NO_CLIPBOARD = ("No clipboard tool found: install `wl-clipboard` (Wayland) or `xclip` (X11). "
                "The text is in History.")


def _default_run(args: list[str], stdin: str | None = None) -> str:
    # Clipboard writers (xclip, wl-copy) fork a process that keeps serving the selection and
    # holds inherited pipes open, so their output must not be captured or run() never returns.
    out = subprocess.DEVNULL if stdin is not None else subprocess.PIPE
    try:
        r = subprocess.run(args, input=stdin, stdout=out, stderr=out, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise InjectError(f"{args[0]} failed: {e}. The text is in History.") from e
    if r.returncode != 0:
        raise InjectError(f"{args[0]} failed: {(r.stderr or '').strip() or r.returncode}. The text is in History.")
    return r.stdout or ""


class LinuxInjector:
    """:class:`~openwhisprflow.platform.base.Injector` for X11 and Wayland."""

    def __init__(self, session: str | None = None, *, which: Callable[[str], str | None] = shutil.which,
                 run: Runner = _default_run, paste_settle_s: float = 0.3, newline: str = "enter") -> None:
        if session is None:
            from openwhisprflow.platform.hotkeys_linux import session_type

            session = session_type() or "x11"
        self.session = session
        self.which = which
        self.run = run
        self.paste_settle_s = paste_settle_s
        self.newline = newline

    # ------------------------------------------------------------------ Injector

    def focus(self) -> FocusInfo:
        return get_focus()

    def tool(self) -> str | None:
        """The typing tool that will be used: xdotool, wtype, ydotool or None."""
        order = ("wtype", "ydotool") if self.session == "wayland" else ("xdotool",)
        return next((t for t in order if self.which(t)), None)

    def type_text(self, text: str) -> None:
        if not text:
            return
        tool = self.tool()
        if tool is None:
            raise InjectError(NO_TOOL_WAYLAND if self.session == "wayland" else NO_TOOL_X11)
        if tool == "ydotool" and not text.isascii():
            self.paste_text(text)  # ydotool types key codes for the US layout only
            return
        for cmd in self.type_commands(tool, text):
            self.run(cmd, None)

    def type_commands(self, tool: str, text: str) -> list[list[str]]:
        """The exact commands that type ``text`` (the unit-test seam)."""
        cmds: list[list[str]] = []
        for kind, value in plan(text):
            if kind == "text":
                if tool == "xdotool":
                    cmds.append(["xdotool", "type", "--clearmodifiers", "--delay", "4", "--", value])
                elif tool == "wtype":
                    cmds.append(["wtype", "--", value])
                else:
                    cmds.append(["ydotool", "type", "--key-delay", "4", "--", value])
            else:
                cmds.append(self._key_command(tool, value))
        return cmds

    def paste_text(self, text: str) -> None:
        if not text:
            return
        copy, read = self._clipboard_tools()
        saved: str | None = None
        if read is not None:
            try:
                saved = self.run(read, None)
            except InjectError:
                saved = None  # empty clipboard, or non-text content we cannot keep
        self.run(copy, text)
        time.sleep(0.05)
        self.run(self._paste_command(), None)
        time.sleep(self.paste_settle_s)
        if saved is not None:
            try:
                self.run(copy, saved)
            except InjectError:
                pass

    def press_enter(self) -> None:
        tool = self.tool()
        if tool is None:
            raise InjectError(NO_TOOL_WAYLAND if self.session == "wayland" else NO_TOOL_X11)
        self.run(self._key_command(tool, "enter", plain=True), None)

    # ------------------------------------------------------------------ helpers

    def _key_command(self, tool: str, key: str, plain: bool = False) -> list[str]:
        shift = key == "enter" and self.newline == "shift_enter" and not plain
        if tool == "xdotool":
            name = "Tab" if key == "tab" else ("shift+Return" if shift else "Return")
            return ["xdotool", "key", "--clearmodifiers", name]
        if tool == "wtype":
            name = "Tab" if key == "tab" else "Return"
            return ["wtype", "-M", "shift", "-k", name, "-m", "shift"] if shift else ["wtype", "-k", name]
        code = "15" if key == "tab" else "28"  # evdev KEY_TAB / KEY_ENTER
        keys = [f"{code}:1", f"{code}:0"]
        return ["ydotool", "key", *(["42:1", *keys, "42:0"] if shift else keys)]

    def _clipboard_tools(self) -> tuple[list[str], list[str] | None]:
        if self.session == "wayland" and self.which("wl-copy"):
            read = ["wl-paste", "--no-newline", "--type", "text/plain"] if self.which("wl-paste") else None
            return ["wl-copy"], read
        if self.which("xclip"):
            return (["xclip", "-selection", "clipboard", "-in"],
                    ["xclip", "-selection", "clipboard", "-out"])
        if self.which("xsel"):
            return ["xsel", "--clipboard", "--input"], ["xsel", "--clipboard", "--output"]
        raise InjectError(NO_CLIPBOARD)

    def _paste_command(self) -> list[str]:
        tool = self.tool()
        if tool == "xdotool":
            return ["xdotool", "key", "--clearmodifiers", "ctrl+v"]
        if tool == "wtype":
            return ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]
        if tool == "ydotool":
            return ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]
        raise InjectError(NO_TOOL_WAYLAND if self.session == "wayland" else NO_TOOL_X11)

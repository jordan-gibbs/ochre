"""The focused app on macOS: a short app name, window title and window number.

The window title comes from ``CGWindowListCopyWindowInfo``; since macOS 10.15 titles of other
apps' windows are only visible with the Screen Recording permission, so without it the title
is "" (the app name and window id still work for the spacing rule and per-app styles).
"""

from __future__ import annotations

import re

from openwhisprflow.platform.base import FocusInfo

# Bundle ids whose localized names do not match the keys in RefineConfig.app_styles.
FRIENDLY_BUNDLES = {
    "com.microsoft.vscode": "code", "com.microsoft.vscodeinsiders": "code",
    "com.apple.mail": "mail", "com.apple.terminal": "terminal", "com.googlecode.iterm2": "iterm2",
    "com.tinyspeck.slackmacgap": "slack", "com.hnc.discord": "discord",
    "com.microsoft.outlook": "outlook", "com.google.chrome": "chrome",
    "com.microsoft.edgemac": "edge", "org.mozilla.firefox": "firefox",
    "company.thebrowser.browser": "arc", "com.apple.safari": "safari",
    "com.apple.mobilesms": "messages", "notion.id": "notion", "us.zoom.xos": "zoom",
    "dev.warp.warp-stable": "terminal", "com.mitchellh.ghostty": "terminal",
    "net.kovidgoyal.kitty": "terminal", "io.alacritty": "terminal",
}


def app_key(bundle_id: str | None, localized_name: str | None) -> str:
    """Friendly name for known bundles, else the localized name squeezed to [a-z0-9],
    else the bundle id's last component."""
    bid = (bundle_id or "").lower()
    if bid in FRIENDLY_BUNDLES:
        return FRIENDLY_BUNDLES[bid]
    squeezed = re.sub(r"[^a-z0-9]", "", (localized_name or "").lower())
    if len(squeezed) >= 2:
        return squeezed[:24]
    return bid.rsplit(".", 1)[-1] if bid else ""


def get_focus() -> FocusInfo:
    import Quartz
    from AppKit import NSWorkspace

    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return FocusInfo()
    pid = int(app.processIdentifier())
    name = app_key(app.bundleIdentifier(), app.localizedName())
    title, window_id = "", ""
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    for info in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:
        if int(info.get("kCGWindowOwnerPID", -1)) == pid and int(info.get("kCGWindowLayer", 1)) == 0:
            title = str(info.get("kCGWindowName") or "")
            window_id = str(info.get("kCGWindowNumber", ""))
            break  # front-to-back order: the first normal window is the frontmost
    return FocusInfo(app_name=name, window_title=title, window_id=f"{pid}:{window_id}")

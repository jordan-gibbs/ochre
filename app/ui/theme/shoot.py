"""Render the theme previews headlessly into docs/design/ (no window ever opens).

    python app/ui/theme/shoot.py            # everything
    python app/ui/theme/shoot.py hud        # HUD states only
    python app/ui/theme/shoot.py settings   # settings sections only

Uses Playwright with the installed Microsoft Edge (channel="msedge"); falls back to Playwright's
bundled Chromium. Screenshots are taken at 2x device scale so type and hairlines can be judged.
"""

import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

THEME = Path(__file__).resolve().parent
ROOT = THEME.parents[2]
OUT = ROOT / "docs" / "design"

HUD_STATES = [
    "loading", "loading-indeterminate", "idle", "recording", "locked", "handsfree",
    "transcribing", "refining", "inserting", "inserted", "error", "notice",
]
SETTINGS_SECTIONS = [
    "general", "transcription", "refinement", "handsfree", "dictionary", "history", "about",
]


def launch(p):
    try:
        return p.chromium.launch(channel="msedge", headless=True)
    except Exception:
        return p.chromium.launch(headless=True)


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


# Served over http, not file://: CSS masks (the wordmark) are CORS-checked and fail on file URLs.
_httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(THEME)))
threading.Thread(target=_httpd.serve_forever, daemon=True).start()


def url(name, query=""):
    return f"http://127.0.0.1:{_httpd.server_address[1]}/{name}" + (f"?{query}" if query else "")


def shoot_hud(browser):
    page = browser.new_page(viewport={"width": 1520, "height": 300}, device_scale_factor=2)
    for st in HUD_STATES:
        page.goto(url("hud-preview.html", f"state={st}&freeze"))
        page.wait_for_timeout(900)  # fonts, enter spring, check draw
        page.screenshot(path=str(OUT / f"hud-{st}.png"))
        print("hud", st)
    page.close()
    page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
    page.goto(url("hud-preview.html", "grid&freeze"))
    page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / "hud-all.png"), full_page=True)
    page.close()


def shoot_settings(browser):
    page = browser.new_page(viewport={"width": 900, "height": 640}, device_scale_factor=2)
    for theme in ("light", "dark"):
        for sec in SETTINGS_SECTIONS:
            page.set_viewport_size({"width": 900, "height": 640})
            page.goto(url("settings-preview.html", f"section={sec}&theme={theme}&shot"))
            page.wait_for_timeout(500)
            # grow the "window" to fit the section, so nothing is cut off (min = real window height)
            h = page.evaluate("document.querySelector('.page:not([hidden])').offsetHeight")
            page.set_viewport_size({"width": 900, "height": max(640, int(h))})
            page.wait_for_timeout(450)
            page.screenshot(path=str(OUT / f"settings-{sec}-{theme}.png"))
            print("settings", sec, theme)
    page.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = set(sys.argv[1:]) or {"hud", "settings"}
    with sync_playwright() as p:
        b = launch(p)
        if "hud" in which:
            shoot_hud(b)
        if "settings" in which:
            shoot_settings(b)
        b.close()


if __name__ == "__main__":
    main()

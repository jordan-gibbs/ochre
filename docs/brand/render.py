"""Render the brand assets in docs/brand/src/ to docs/brand/assets/ (headless; no window opens).

    python docs/brand/render.py            # every asset
    python docs/brand/render.py --sheet    # also a full-page shot of docs/brand/index.html

Uses Playwright with the installed Microsoft Edge (channel="msedge"), falling back to Playwright's
bundled Chromium, at 1x so the PNGs are exactly the spec size.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = HERE / "assets"

ASSETS = [  # source, output, width, height
    ("og.html", "og-1280x640.png", 1280, 640),
    ("readme-header-light.html", "readme-header-light.png", 1280, 320),
    ("readme-header-dark.html", "readme-header-dark.png", 1280, 320),
    ("hf-thumbnail.html", "hf-thumbnail-1200x630.png", 1200, 630),
]


def launch(p):
    try:
        return p.chromium.launch(channel="msedge", headless=True)
    except Exception:
        return p.chromium.launch(headless=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = launch(p)
        for src, out, w, h in ASSETS:
            page = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            page.goto((SRC / src).as_uri())
            page.evaluate("document.fonts.ready")
            page.wait_for_function("[...document.images].every(i => i.complete)")
            page.screenshot(path=str(OUT / out), clip={"x": 0, "y": 0, "width": w, "height": h})
            page.close()
            print("wrote", OUT / out)
        if "--sheet" in sys.argv:
            page = b.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
            page.goto((HERE / "index.html").as_uri())
            page.evaluate("document.fonts.ready")
            dst = Path(sys.argv[sys.argv.index("--sheet") + 1]) if len(sys.argv) > sys.argv.index("--sheet") + 1 else HERE / "sheet.png"
            page.screenshot(path=str(dst), full_page=True)
            print("wrote", dst)
        b.close()


if __name__ == "__main__":
    main()

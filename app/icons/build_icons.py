"""Ochre icon build: the committed SVG masters in app/icons/src/ -> every platform raster.

    python app/icons/build_icons.py              # writes app/icons/out/ and docs/design/icons.png
    python app/icons/build_icons.py --install    # also copies into app/src-tauri/icons/ (Tauri paths)

The mark (brand handoff 2026-10-05, docs/brand/handoff/): a Funnel Display ExtraBold lowercase
"o", outlined, white on a flat vermilion (#F04E16) squircle. The SVGs in src/ are final artwork;
this script never draws, it only rasterises:

  <= 32 px   src/ochre-icon-small.svg   (the o fills 62% of the tile so the counter stays open)
  >= 48 px   src/ochre-icon.svg         (Windows / Linux / Tauri)
  .icns      src/ochre-icon-macos.svg   (824 body at offset 100 on the 1024 Apple grid, no shadow;
                                         the system draws its own, so none is added here)

Rasterised by headless Edge/Chromium through Playwright (nothing is shown on screen), packed with
Pillow. .ico is built from per-size renders (16, 24, 32, 48, 64, 256), never by downscaling the
256. Pillow's .icns writer skips the 16@1x and 32@1x entries, so on macOS (where `iconutil`
exists) the .icns is built with `iconutil -c icns` from an .iconset of the same renders instead.

Tray (names used by app/src-tauri/src/tray.rs), every SVG under src/tray/ at 16 / 20 / 32 px + @2x:
  Windows / Linux, colour, per taskbar theme:  ochre-tray-{idle,recording,busy,error,off}-{light,dark}
  macOS template images (black + alpha):       macos/ochre-template-{idle,busy,error,off}
  macOS recording (the one colour image):      macos/ochre-recording
"""

import base64
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = HERE / "out"
ROOT = HERE.parents[1]
TAURI_ICONS = ROOT / "app" / "src-tauri" / "icons"
UI_BRAND = ROOT / "app" / "ui" / "theme" / "brand"
DOCS = ROOT / "docs" / "design"

MASTER = SRC / "ochre-icon.svg"
SMALL = SRC / "ochre-icon-small.svg"
MACOS = SRC / "ochre-icon-macos.svg"
SMALL_MAX = 32

APP_SIZES = [16, 24, 32, 48, 64, 128, 256, 512, 1024]
ICO_SIZES = [16, 24, 32, 48, 64, 256]
ICNS_SIZES = [16, 32, 64, 128, 256, 512, 1024]
TRAY_SIZES = [16, 20, 32]
# the UI's copies of the artwork (settings sidebar, About, onboarding)
UI_FILES = ["ochre-icon.svg", "ochre-icon-small.svg", "ochre-glyph.svg", "ochre-glyph-small.svg",
            "ochre-wordmark.svg", "ochre-wordmark-white.svg",
            "ochre-lockup-horizontal.svg", "ochre-lockup-horizontal-dark.svg"]


def render(page, svg_path, px):
    data = base64.b64encode(Path(svg_path).read_bytes()).decode()
    html = (f"<html><body style='margin:0;background:transparent'>"
            f"<img src='data:image/svg+xml;base64,{data}' "
            f"style='width:{px}px;height:{px}px;display:block'></body></html>")
    page.set_viewport_size({"width": px, "height": px})
    page.set_content(html)
    page.wait_for_function("document.images[0].complete")
    png = page.screenshot(omit_background=True, clip={"x": 0, "y": 0, "width": px, "height": px})
    return Image.open(io.BytesIO(png)).convert("RGBA")


def icns_with_iconutil(mac, dst):
    """macOS only: a complete .icns (incl. 16@1x / 32@1x) via iconutil. Returns False elsewhere."""
    if sys.platform != "darwin" or not shutil.which("iconutil"):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for s in (16, 32, 128, 256, 512):
            mac[s].save(iconset / f"icon_{s}x{s}.png")
            mac[s * 2].save(iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dst)], check=True)
    return True


def main():
    install = "--install" in sys.argv
    for d in (OUT, OUT / "tray", OUT / "tray" / "macos", DOCS, UI_BRAND):
        d.mkdir(parents=True, exist_ok=True)
    tray_svgs = sorted((SRC / "tray").rglob("*.svg"))
    # Square*Logo.png / StoreLogo.png are regenerated only if the bundle already ships them
    squares = {f: int(f.stem[len("Square"):].split("x")[0]) for f in TAURI_ICONS.glob("Square*Logo.png")}

    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge", headless=True)
        except Exception:
            b = p.chromium.launch(headless=True)
        page = b.new_page(device_scale_factor=1)
        app = {s: render(page, SMALL if s <= SMALL_MAX else MASTER, s) for s in APP_SIZES}
        app[50] = render(page, MASTER, 50)
        for sz in set(squares.values()) - set(app):
            app[sz] = render(page, SMALL if sz <= SMALL_MAX else MASTER, sz)
        mac = {s: render(page, MACOS, s) for s in sorted(set(ICNS_SIZES))}
        tray = {}
        for svg in tray_svgs:
            sub = svg.parent.relative_to(SRC / "tray")
            for s in TRAY_SIZES:
                tray[(sub, f"{svg.stem}-{s}")] = render(page, svg, s)
                tray[(sub, f"{svg.stem}-{s}@2x")] = render(page, svg, s * 2)
        b.close()

    # app icon set: plain sizes + the Tauri bundle names
    for s in APP_SIZES:
        app[s].save(OUT / f"app-{s}.png")
    app[32].save(OUT / "32x32.png")
    app[128].save(OUT / "128x128.png")
    app[256].save(OUT / "128x128@2x.png")
    app[256].save(OUT / "256x256.png")
    app[512].save(OUT / "512x512.png")
    app[1024].save(OUT / "512x512@2x.png")
    app[512].save(OUT / "icon.png")
    app[32].save(OUT / "favicon-32.png")
    # .ico: each entry is its own render from the right cut
    app[256].save(OUT / "icon.ico", format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                  append_images=[app[s] for s in ICO_SIZES[:-1]])
    # .icns from the Apple-grid master
    if not icns_with_iconutil(mac, OUT / "icon.icns"):
        mac[1024].save(OUT / "icon.icns", format="ICNS",
                       append_images=[mac[s] for s in ICNS_SIZES[:-1]])
    for (sub, name), im in tray.items():
        im.save(OUT / "tray" / sub / f"{name}.png")

    # the UI's copies of the artwork
    for f in UI_FILES:
        shutil.copy2(SRC / f, UI_BRAND / f)
    shutil.copy2(SRC / "favicon.svg", UI_BRAND / "favicon.svg")
    shutil.copy2(OUT / "favicon-32.png", UI_BRAND / "favicon-32.png")

    contact_sheet(app, tray)

    if install:
        for f in ("32x32.png", "128x128.png", "128x128@2x.png", "icon.png", "icon.ico", "icon.icns"):
            shutil.copy2(OUT / f, TAURI_ICONS / f)
        for f, sz in squares.items():
            app[sz].save(f)
        if (TAURI_ICONS / "StoreLogo.png").exists():
            app[50].save(TAURI_ICONS / "StoreLogo.png")
        dst = TAURI_ICONS / "tray"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(OUT / "tray", dst)
        print("installed into", TAURI_ICONS)
    print("icons written to", OUT)


def contact_sheet(app, tray):
    """docs/design/icons.png: the app icon at every size, and every tray icon on light + dark bars."""
    W = 1280
    sheet = Image.new("RGBA", (W, 780), "#f7f5f1")
    x = 40
    for s in (512, 256, 128, 64, 48, 32, 24, 16):
        im = app[s] if s != 512 else app[512].resize((300, 300), Image.LANCZOS)
        sheet.alpha_composite(im, (x, 40 + (300 - im.height) // 2))
        x += im.width + 32
    y = 400
    states = ["idle", "recording", "busy", "error", "off"]
    mac = ["template-idle", "recording", "template-busy", "template-error", "template-off"]
    for bar, theme, ink in (("#ece9e4", "light", 0), ("#1b1a18", "dark", 255)):
        strip = Image.new("RGBA", (W - 80, 160), bar)
        sx = 30
        for v in states:
            for s in (16, 20, 32):
                im = tray[(Path("."), f"ochre-tray-{v}-{theme}-{s}@2x")].resize((s * 2, s * 2), Image.LANCZOS)
                strip.alpha_composite(im, (sx, 80 - s)); sx += s * 2 + 8
            sx += 12
        # macOS: what the menu bar does with template images (tinted to the bar's ink)
        sx += 0
        for name in mac:
            im = tray[(Path("macos"), f"ochre-{name}-20@2x")]
            if name.startswith("template"):
                a = im.split()[3]
                im = Image.merge("RGBA", (Image.new("L", im.size, ink),) * 3 + (a,))
            strip.alpha_composite(im, (sx, 60)); sx += 50
        # 1x row at true size
        sx2 = 30
        for v in states:
            strip.alpha_composite(tray[(Path("."), f"ochre-tray-{v}-{theme}-16")], (sx2, 132)); sx2 += 26
        sheet.alpha_composite(strip, (40, y))
        y += 180
    sheet.convert("RGB").save(DOCS / "icons.png")


if __name__ == "__main__":
    main()

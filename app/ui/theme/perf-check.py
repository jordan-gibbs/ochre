"""HUD paint-cost check: the speaking "o" must cause 0 layouts while it meters.

    python app/ui/theme/perf-check.py

Loads the real HUD (app/ui/hud/index.html?mock) in headless Edge (no window), puts it in the
recording state, then feeds 2 s of a speech-like level at the core's ~30 events/s and reads the
DevTools Performance metrics before and after. Live text is off here (it is measured separately:
it appends spans by design). Exit code 1 if any layout happened during metering.
"""

import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

UI = Path(__file__).resolve().parents[1]


def serve():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(UI)))
    SimpleHTTPRequestHandler.log_message = lambda *a: None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    httpd = serve()
    origin = f"http://127.0.0.1:{httpd.server_address[1]}"
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge", headless=True)
        except Exception:
            b = p.chromium.launch(headless=True)
        page = b.new_page(viewport={"width": 760, "height": 320})
        page.goto(f"{origin}/hud/index.html?mock&theme=light")
        page.wait_for_function("window.__ochreHud")
        page.evaluate("""() => {
          window.__ochreHud.emit({ event: 'config', config: { ui: { theme: 'light', show_partials: false } } });
          window.__ochreHud.emit({ event: 'state', state: 'recording', trigger: 'hotkey', handsfree_armed: false, detail: '' });
        }""")
        page.wait_for_timeout(900)  # enter spring + width morph settle
        cdp = page.context.new_cdp_session(page)
        cdp.send("Performance.enable")
        metric = lambda: {m["name"]: m["value"] for m in cdp.send("Performance.getMetrics")["metrics"]}
        before = metric()
        top, anims = page.evaluate("""() => new Promise((done) => {
          const voice = HudKit.fakeVoice(4), t0 = performance.now();
          const g = document.querySelector('.hud-o-glyph');
          let n = 0, top = 1;
          const id = setInterval(() => {
            window.__ochreHud.level(voice(performance.now() - t0));
            const m = /scale\(([\d.]+)\)/.exec(g.style.transform);
            if (m) top = Math.max(top, parseFloat(m[1]));
            if (++n >= 60) {
              clearInterval(id);
              const loops = document.getAnimations().filter((a) => a.effect && a.effect.target && a.effect.target.closest && a.effect.target.closest('.hud-o'));
              done([top, loops.length]);
            }
          }, 33);
        })""")
        after = metric()
        b.close()
    httpd.shutdown()
    layouts = after["LayoutCount"] - before["LayoutCount"]
    styles = after["RecalcStyleCount"] - before["RecalcStyleCount"]
    print(f"2 s of metering: {layouts:.0f} layouts, {styles:.0f} style recalcs, o peaked at {top:.3f}x, "
          f"{anims} animations on the o")
    sys.exit(1 if layouts > 0 or anims > 0 or not (1.05 < top <= 1.35) else 0)


if __name__ == "__main__":
    main()

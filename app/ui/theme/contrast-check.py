"""WCAG 2.x contrast check for the Ochre palette (tokens.css).

    python app/ui/theme/contrast-check.py

Keep the pairs here in sync with tokens.css. Translucent foregrounds are composited over their
background first. Text pairs must reach 4.5:1 (AA body text); `ui` pairs (icons, borders of
controls, focus rings, large text) must reach 3:1.
"""

import sys


def hex_rgba(h):
    h = h.lstrip("#")
    if len(h) == 6:
        h += "ff"
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4, 6))


def over(fg, bg):
    a = fg[3]
    return tuple(fg[i] * a + bg[i] * (1 - a) for i in range(3)) + (1.0,)


def lum(c):
    def ch(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (ch(v) for v in c[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(fg, bg):
    bg = hex_rgba(bg) if isinstance(bg, str) else bg
    fg = over(hex_rgba(fg), bg)
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


# (theme, foreground, background, kind, note)
LIGHT_BG, LIGHT_SURF, LIGHT_SUNK = "#f7f5f1", "#fffefc", "#f2efea"
DARK_BG, DARK_SURF, DARK_SUNK = "#151412", "#1f1d1a", "#292622"
HUD_L = "#fffefc"    # --hud-bg, light (frosted porcelain, near-opaque)
HUD_D = "#2a2723"    # --hud-bg, dark
LOCKED = "#d4400e"   # --hud-locked (solid vermilion pill)

PAIRS = [
    ("light", "#1a1714", LIGHT_SURF, "text", "ink / panel"),
    ("light", "#1a1714", LIGHT_BG, "text", "ink / bg"),
    ("light", "#5a544d", LIGHT_SURF, "text", "ink-2 / panel"),
    ("light", "#5a544d", LIGHT_SUNK, "text", "ink-2 / sunk"),
    ("light", "#6f6860", LIGHT_SURF, "text", "ink-3 / panel"),
    ("light", "#6f6860", LIGHT_BG, "text", "ink-3 / bg"),
    ("light", "#6f6860", "#efebe5", "text", "ink-3 / bg-2 (sidebar)"),
    ("light", "#c23a0c", LIGHT_SURF, "text", "accent-ink / panel"),
    ("light", "#c23a0c", LIGHT_BG, "text", "accent-ink / bg"),
    ("light", "#ffffff", "#d4400e", "text", "on-accent / accent"),
    ("light", "#ffffff", "#bc380b", "text", "on-accent / accent-2 (hover)"),
    ("light", "#c23a0c", "#fcebe2", "text", "accent-ink / accent-soft (badge)"),
    ("light", "#1f7346", LIGHT_SURF, "text", "success-ink / panel"),
    ("light", "#1f7346", "#e7f2ea", "text", "success-ink / success-soft"),
    ("light", "#875800", LIGHT_SURF, "text", "warning-ink / panel"),
    ("light", "#875800", "#faf0da", "text", "warning-ink / warning-soft"),
    ("light", "#b4283a", LIGHT_SURF, "text", "danger-ink / panel"),
    ("light", "#b4283a", "#fbe9ea", "text", "danger-ink / danger-soft"),
    ("light", "#8f877e", LIGHT_SURF, "ui", "line-strong (toggle off) / panel"),
    ("light", "#d4400e", LIGHT_SURF, "ui", "accent fill / panel"),
    ("dark", "#f3efe9", DARK_SURF, "text", "ink / panel"),
    ("dark", "#f3efe9", DARK_BG, "text", "ink / bg"),
    ("dark", "#b3aca3", DARK_SURF, "text", "ink-2 / panel"),
    ("dark", "#b3aca3", DARK_SUNK, "text", "ink-2 / sunk"),
    ("dark", "#999189", DARK_SURF, "text", "ink-3 / panel"),
    ("dark", "#999189", DARK_BG, "text", "ink-3 / bg"),
    ("dark", "#ff8552", DARK_SURF, "text", "accent-ink / panel"),
    ("dark", "#ffffff", "#d4400e", "text", "on-accent / accent"),
    ("dark", "#ff8552", "#3a2319", "text", "accent-ink / accent-soft"),
    ("dark", "#62c48f", DARK_SURF, "text", "success-ink / panel"),
    ("dark", "#e3b341", DARK_SURF, "text", "warning-ink / panel"),
    ("dark", "#f2737c", DARK_SURF, "text", "danger-ink / panel"),
    ("dark", "#f2737c", "#3a2224", "text", "danger-ink / danger-soft"),
    ("dark", "#7a736b", DARK_SURF, "ui", "line-strong / panel"),
    ("dark", "#d4400e", DARK_SURF, "ui", "accent fill / panel"),
    ("hud-light", "#1a1714", HUD_L, "text", "hud-ink"),
    ("hud-light", "#5a544d", HUD_L, "text", "hud-ink-2 (hints)"),
    ("hud-light", "#7a736b", HUD_L, "text", "hud-ink-3 (unconfirmed words)"),
    ("hud-light", "#f04e16", HUD_L, "ui", "hud-accent (bars, ring, sparkle)"),
    ("hud-light", "#c23a0c", HUD_L, "text", "hud-accent-2 (orange text)"),
    ("hud-light", "#2f8f5b", HUD_L, "ui", "hud-success (check)"),
    ("hud-light", "#c8323f", HUD_L, "ui", "hud-danger (alert)"),
    ("hud-dark", "#f3efe9", HUD_D, "text", "hud-ink"),
    ("hud-dark", "#f3efe9a8", HUD_D, "text", "hud-ink-2"),
    ("hud-dark", "#f3efe98c", HUD_D, "text", "hud-ink-3"),
    ("hud-dark", "#ff6a33", HUD_D, "ui", "hud-accent"),
    ("hud-dark", "#ff8552", HUD_D, "text", "hud-accent-2 (orange text)"),
    ("hud-dark", "#5cc48a", HUD_D, "ui", "hud-success"),
    ("hud-dark", "#f2737c", HUD_D, "ui", "hud-danger"),
    ("hud", "#ffffff", LOCKED, "text", "label / locked pill"),
    ("hud", "#ffffff", LOCKED, "text", "hint / locked pill (white, lighter weight)"),
]


def main():
    bad = 0
    for theme, fg, bg, kind, note in PAIRS:
        r = ratio(fg, bg)
        need = 4.5 if kind == "text" else 3.0
        ok = r >= need
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {r:5.2f}  (need {need})  [{theme}] {note}")
    print("all pairs pass" if not bad else f"{bad} pair(s) fail")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

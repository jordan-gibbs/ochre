# Ochre design: "Paper & Vermilion"

Owner: design. Source of truth for visual values is `app/ui/theme/`. Screenshots are in
`docs/design/`, regenerated with `python app/ui/theme/shoot.py` (headless Edge, no windows).

Revision 2 (user feedback: "too AI-ified"). The Iris/Lagoon palette, the ink-glass HUD and every
gradient are gone. Motion, type, layouts, component structure, every variable name and the HUD
markup contract are unchanged: this revision is values and CSS only.

Revision 3 (brand handoff 2026-10-05, `docs/brand/handoff/`, plus user requests): the "o" mark and
"ochre" wordmark replace the wave-into-caret; Funnel Display is the display face; **light is the
default theme**; hero surfaces get a faint vermilion-tinted shadow; state changes cross-fade and
morph; the HUD's level bars became a **speaking "o"** that grows as you talk. Brand sheet: `docs/brand/index.html`.

## 1. Direction

Simple and flat. Warm paper-white surfaces, near-black ink, and **one** confident orange.
Nothing is gradient and nothing glows like neon. Shadows are neutral and soft, with one deliberate
exception: the **hero surfaces** (the HUD pill, the primary button, the active nav item) sit on a
faint, wide, vermilion-tinted diffuse shadow, `--shadow-hero` (#F04E16 at ~6-14%, large blur,
negative spread so it pools under the surface). It reads as warm light under the thing that
matters, never as a halo. The HUD's shadow warms to `--shadow-hero-active` (~10-22%) while it
listens and settles back when it stops. The personality still comes from motion ("glide") and
the mark, not from colour effects.

Why this orange:
- **Vermilion**, not tangerine or amber. It leans red (hue ≈ 15°, where amber-orange sits near
  22°) and runs a step darker in fills (`#D4400E`), so white text sits on it at AA. We put white
  text on a deeper orange rather than black text on a lighter one, so it reads red-orange.
- The type (Funnel Display + Instrument Sans) and the mark (a heavy lowercase "o") carry the
  rest of the identity.

| | Ochre |
|---|---|
| Accent | **vermilion** `#D4400E` fill, white text; `#F04E16` for marks |
| Surfaces | warm paper `#F7F5F1` + porcelain `#FFFEFC` |
| Type | Funnel Display (display) + Instrument Sans (UI) + Geist Mono |
| Mark | flat vermilion squircle, **a white Funnel Display ExtraBold "o"** |
| Signature | frosted pills + the lowercase "ochre" wordmark + the speaking o |

## 2. Palette (AA verified: `python app/ui/theme/contrast-check.py`, all 52 pairs pass)

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#F7F5F1` paper | `#151412` | window canvas |
| `--bg-2` | `#EFEBE5` | `#1A1917` | sidebar |
| `--panel` / `--panel-2` | `#FFFEFC` porcelain / `#FBF9F6` | `#1F1D1A` / `#25231F` | cards / hover |
| `--sunk` | `#F2EFEA` | `#292622` | inputs, tracks |
| `--chip` | `#FFFFFF` | `#34312C` | segmented thumb |
| `--line` / `--line-2` | ink 9% / 15% | cream 7% / 12% | hairlines |
| `--line-strong` | `#8F877E` | `#7A736B` | 3:1 control outlines |
| `--ink` / `--ink-2` / `--ink-3` | `#1A1714` / `#5A544D` / `#6F6860` | `#F3EFE9` / `#B3ACA3` / `#999189` | text (all ≥ 4.5:1) |
| `--accent` | `#D4400E` | `#D4400E` | **fills under white text**: primary button, toggle, locked pill (4.63:1) |
| `--accent-2` | `#BC380B` | `#BC380B` | hover / pressed |
| `--accent-ink` | `#C23A0C` | `#FF8552` | orange **text and links** (5.3:1 / 7.0:1) |
| `--accent-soft` | `#FCEBE2` | `#3A2319` | tinted badge / capture backgrounds |
| `--hud-accent` | `#F04E16` | `#FF6A33` | **bright marks with no text**: level bars, ring, sparkle, caret, mark |
| `--success` / `-ink` / `-soft` | `#2F8F5B` / `#1F7346` / `#E7F2EA` | `#3FA86E` / `#62C48F` | quiet green |
| `--warning` / `-ink` / `-soft` | `#C98A12` / `#875800` / `#FAF0DA` | `#E3B341` | quiet amber |
| `--danger` / `-ink` / `-soft` | `#C8323F` / `#B4283A` / `#FBE9EA` | `#E5555F` / `#F2737C` / `#3A2224` | crimson, kept clearly redder than vermilion |
| `--shadow-*` | neutral `rgba(20,16,12,…)` | black | soft, untinted |
| `--shadow-hero` | `0 6px 22px -6px` #F04E16 14% + `0 2px 6px -2px` 6% | #FF6A33 12% / 5% | hero surfaces at rest: HUD pill, pip, active nav item, primary button hover |
| `--shadow-hero-active` | `0 8px 28px -6px` 22% + `0 2px 8px -2px` 10% | #FF6A33 26% / 12% | the HUD while listening (recording, locked, hands-free) |
| `--shadow-hero-sm` | `0 4px 12px -4px` #D4400E 32% | #FF6A33 34% | primary button at rest |
| `--brand-icon` / `--brand-ink` / `--brand-paper` | `#F04E16` / `#1A1714` / `#F7F5F1` | same | icon tile (never under text), brand ink and paper |

Retired as colours, kept as names: `--companion*` and `--hud-companion` alias the accent
family; `--hud-locked-2` equals `--hud-locked`; `--hud-refine-gradient` is `none`.
Added (no renames): `--hud-accent-soft`, `--hud-success-soft`, `--hud-danger-soft`.

HUD surfaces: light `--hud-bg` `rgba(255,254,252,.96)` (frosted porcelain, near-opaque because
the transparent window has nothing behind it to blur) with ink text; dark `rgba(42,39,35,.96)`
with cream text. Same orange in both.

## 3. Type

- **Funnel Display** (500 / 600 / 700 / 800, Latin subset, no italic) is the display face:
  `--font-display`, used at `--weight-display` 700 with `--tracking-display` -0.04em for page
  titles, the onboarding hero and big numbers. The wordmark is the outlined SVG
  (`app/icons/src/ochre-wordmark.svg`, set at -0.045em), drawn in the UI as a CSS mask
  (`.wordmark`) so it takes the ink colour of either theme; `.lockup` pairs it with the icon at
  the brand clear space (1/4 of the icon height). No italic accents anywhere.
- **Instrument Sans** (variable 400–700, normal + italic), UI text. Crisp, slightly narrow,
  neo-grotesque; nothing like Figtree's round geometry. Weights used: 420 / 520 / 600 / 680.
- **Geist Mono**: keycaps, timestamps, version strings.
- Scale: 11 / 12 / 13 / 14 / 15 / 17 / 22 / 30 / 42 (`--text-2xs` … `--text-3xl`).
- All OFL, self-hosted as woff2 in `app/ui/theme/fonts/` with `OFL-*.txt`. Latin + Latin-ext subsets.

## 4. The mark

A flat vermilion (`#F04E16`) rounded tile with a white, outlined Funnel Display ExtraBold
lowercase **o**. *Ochre is the earth pigment people used for their first marks: the oldest way to
make a mark, for the newest way to write.* No gradient, no glow, no circle container. At ≤ 32 px
the small cut (`ochre-icon-small.svg`, the o at 62% of the tile) keeps the counter open; the
macOS icon is the 824-on-1024 Apple-grid cut. Sources: `app/icons/src/` (final artwork, from the
handoff; never redrawn).

- Tray (Windows/Linux, colour), art picked by the **taskbar's** theme (Windows
  `SystemUsesLightTheme`, polled every 3 s; Linux: the GTK theme), independent of the app's
  `ui.theme`: idle is an ink tile + white o on light taskbars and a paper tile + ink o on dark
  ones; **recording** is the hero tile + white o (both); busy is a tile with three dots; error is
  the tile at 42% with a `#C8323F` badge ringed in the taskbar colour; off is a grey tile.
- macOS menu bar: template images (black + alpha, `setTemplate(true)`) for idle / busy / error /
  off; recording is the one colour image, like the system's own recording indicators.

## 5. Motion: "glide"

Things arrive fast and settle softly, leave quicker than they came, and only presence changes
get a hint of overshoot.

| Token | Value | Use |
|---|---|---|
| `--ease` | `cubic-bezier(.16,1,.3,1)` | default: enter, morph, hover |
| `--ease-in` | `cubic-bezier(.5,0,.75,0)` | exits |
| `--ease-in-out` | `cubic-bezier(.65,0,.35,1)` | loops |
| `--spring` | `cubic-bezier(.34,1.32,.64,1)` | HUD enter, toggle knob, glyph pop, segmented thumb |
| `--dur-1…5` | 90 / 160 / 240 / 380 / 600 ms | press / hover+exit / default / HUD enter+width morph / check draw |
| `--loop-fast / --loop / --loop-slow` | 1.1 / 1.8 / 2.8 s | spinners+dots / shimmer+refine / breathing |
| `--motion-distance` | 1 (0 when reduced) | multiplier on every translate |

Rules: loops animate only `transform`/`opacity`, except small paint-only loops on a ~100 px
label (the transcribing and refining light sweeps) and the ring stroke. Those two sweeps and the
bubble's top fade mask are the only gradients left, and they are motion, not decoration.

**State transitions** are one continuous morph, never a cut:
- the pill glides to its new width (FLIP, one measured layout per change) and its colour and
  shadow ease over `--dur-4` with `--ease` (e.g. porcelain → vermilion when it locks);
- the orb's glyph cross-fades: the old one leaves fast (`--dur-2`, ease-in, scale .6 + 2 px blur),
  the new one arrives 60 ms later on `--spring` (scale, opacity, blur → none);
- the label (or the live text line) leaves as an absolutely positioned **ghost** (lift 4 px,
  2 px blur, `--dur-2` ease-in) while the new content rises in (`--dur-3`, 40 ms later). The ghost
  never takes part in layout, so nothing jumps;
- presence: the stack springs in from 10 px below with a 3 px blur, and leaves faster than it came.
Blur is used only on these small elements, and only while they move.

**The speaking o** (the HUD level, replacing the five bars). The brand o (`ochre-glyph-small.svg`)
sits in the orb and simply **grows as you talk**: scale 1.0 in silence, up to 1.3 when you speak
up. No rings, no pulse, no breathing: when you are quiet it is still. `HudKit.oMeter` folds the
core's ~30 level events/s into:
- an adaptive noise floor (room noise reads as silence, so the o rests at exactly 1.0);
- an *envelope*: a one-pole follower with `--o-attack` 80 ms / `--o-release` 300 ms, then a second
  40 ms one-pole so the motion is eased and never jitters between level events;
- an ease-out curve onto `1 + --o-grow` (0.3), so normal speech already reads.
Vermilion on the light pill, white on the vermilion locked pill. The o's `transform` is the only
per-frame style write, and the loop sleeps (no frames, no writes) once the o has settled at 1.0.
Measured with `python app/ui/theme/perf-check.py` (2 s of a speech-like level, headless Edge):
**0 layouts**. The demo (`ochre --demo`, the dev stage, the onboarding welcome and
`hud-preview.html`) drives it with the same synthetic speech envelope (`HudKit.fakeVoice`,
`Voice` in demo.rs): 110-260 ms syllables, word gaps, a breath every few words.

`prefers-reduced-motion`: springs become ease-out, translate distances go to 0, blur is dropped,
decorative loops stop (dots static, shimmer off, check pre-drawn, no error nudge), and the o
grows less (40% of `--o-grow`). State changes still cross-fade (opacity only), so
nothing snaps.

## 6. Tokens: names and compatibility

File: `app/ui/theme/tokens.css` (fonts + all tokens). Theme resolution: **light by default**.
Config `ui.theme` = `"light"` (default) | `"dark"` | `"system"`, chosen in Settings › General ›
Appearance and honoured by every window (settings, onboarding, HUD) through
`shared/bridge.js applyTheme`, which writes `data-theme` on `<html>` (for `"system"` it follows
the OS, live). The native window chrome follows it too (`pages.rs theme_of`). There is no
`prefers-color-scheme` rule in the CSS: without a `data-theme` the UI is light.
`[data-theme="dark"|"light"]` works on `<html>` **or any subtree** (previews). The tray art does
not follow `ui.theme`; it follows the taskbar (§4).

Groups: colour (above), `--hud-*`, `--font-sans|display|mono`, `--text-*`, `--leading-*`,
`--weight-*`, `--tracking-*`, `--radius-xs|sm|md|lg|xl|full` (+ `--radius` = md),
`--space-0…10` (4 px base), `--shadow-xs|sm|(default)|lg`, `--focus`, `--inset-top`,
`--blur-sm|md|lg`, `--glass`, `--glass-strong`, `--glass-saturate`, `--ease*`, `--spring`,
`--dur-1…5`, `--loop*`, `--motion-distance`, `--level-smoothing`, `--z-*`,
`--shadow-hero|-active|-sm`, `--o-attack|release|grow`,
`--brand-icon|ink|paper`, `--tracking-wordmark`, and the handoff's `--display-weight` /
`--display-tracking` (aliases of `--weight-display` / `--tracking-display`).

**Kept from app/ui's current names** (their CSS re-skins with no edits): `--bg --bg-2 --panel
--panel-2 --sunk --chip --line --line-2 --ink --ink-2 --ink-3 --ink-4 --shadow-sm --shadow
--focus --ease --spring --radius`.

**Renamed (aliases kept in tokens.css, remove after migration):**
`--orange → --accent`, `--orange-2 → --accent-2`, `--orange-ink → --accent-ink`,
`--orange-soft → --accent-soft`, `--orange-line → --accent-line`, `--on-orange → --on-accent`,
`--green → --success`, `--green-soft → --success-soft`, `--red → --danger`, `--red-soft → --danger-soft`.

## 7. HUD markup contract (`app/ui/theme/hud.css`, `hud-kit.js`)

```html
<div class="hud" data-state="recording" data-shown aria-live="polite">   <!-- + data-armed, .is-indeterminate -->
  <div class="hud-bubble" hidden>
    <div class="hud-bubble-scroll"><p class="hud-bubble-text">
      <span class="hud-final">confirmed words</span><span class="hud-pending"> unconfirmed</span><span class="hud-bubble-caret"></span>
    </p></div>
  </div>
  <div class="hud-pip">Say “transcribe”</div>
  <div class="hud-pill" role="status">
    <span class="hud-halo"></span>
    <span class="hud-glyph" aria-hidden="true">          <!-- all glyphs present; CSS shows one -->
      <span class="hud-o">                                  <!-- the speaking o (level) -->
        <svg class="hud-o-glyph" viewBox="0 0 100 100"><path d="… ochre-glyph-small.svg …"/></svg>
      </span>
      <svg class="hud-ring" viewBox="0 0 22 22"><circle class="track" cx="11" cy="11" r="9"/><circle class="fill" cx="11" cy="11" r="9" pathLength="100"/></svg>
      <span class="hud-dots"><i></i><i></i><i></i></span>
      <span class="hud-spark"><svg>…</svg></span>
      <span class="hud-caret"></span>
      <svg class="hud-check"><path pathLength="20" d="…"/></svg>
      <svg class="hud-alert">…</svg>  <svg class="hud-info">…</svg>
      <span class="hud-badge"><svg>lock</svg></span>
    </span>
    <span class="hud-text"><span class="hud-label">Listening</span><span class="hud-hint">release to insert</span></span>
    <span class="hud-meta"></span>
    <button class="hud-action" hidden>Retry</button>
    <button class="hud-close" aria-label="Cancel">×</button>
    <span class="hud-progress"><i></i></span>
  </div>
</div>
```

Copy the exact SVGs from `hud-preview.html` (`<template id="hud-tpl">`).

| Core signal | `data-state` | Label / hint / meta (suggested copy) | Look |
|---|---|---|---|
| `state: loading` + `download` | `loading` (set `--progress` 0..1; no total → `.is-indeterminate`) | "Downloading Parakeet" · meta "312 / 670 MB" / "Loading model" | ring + hairline bar in orange; quiet |
| `state: idle` | `idle` (+ `data-armed` if `handsfree_armed`) | pip "Say “transcribe”" | pill hidden; small pip with a breathing orange dot |
| `state: recording` | `recording` | "Listening" · "release to insert" | vermilion o growing with your voice in a soft-orange orb; hero shadow warms |
| `state: locked` | `locked` | "Locked" · `tap <kbd>Right Alt</kbd> to finish` | **solid vermilion pill**, white text, white o, lock badge (loudest) |
| `state: handsfree` | `handsfree` | "Hands-free" · "say “transcribe send”" | **orange outline** around the pill, breathing; the o grows with your voice |
| `state: transcribing` | `transcribing` | "Transcribing" | neutral three-dot wave, grey light sweeping the label (no orange at all) |
| `state: refining` | `refining` | "Refining" · hint provider | **twinkling orange sparkle** + an orange light sweeping the ink label |
| `state: inserting` | `inserting` | "Inserting" | blinking orange caret |
| `result{inserted:true}` | `inserted` (UI-derived, ~1.6 s, then remove `data-shown`) | "Inserted" · "19 words" · meta "0.42 s" | green check draws itself |
| `result{inserted:false}` | `notice` | "Saved to History" · meta word count | info orb |
| `error` | `error` (`.hud-action` "Retry" when retryable) | the message | red orb + red hairline, one small nudge |
| `notice` | `notice` | the message | neutral info orb |

Hooks:
- **Presence:** add `data-shown` to fade/spring the stack in; remove it to leave (exit is
  160 ms ease-in; hide the window after `--dur-2`).
- **State changes:** call `HudKit.setState(hud, state, () => { /* write labels */ })`. It
  morphs the pill's width (FLIP, one measured layout per change) and cross-fades the old label /
  live line (a `.hud-ghost` clone) into the new one.
- **Level:** `const meter = HudKit.oMeter(hud.querySelector('.hud-o'))`; call
  `meter.push(rms)` on every `level` event; `meter.stop()` when leaving a recording state.
  `HudKit.oSnapshot(o, scale)` freezes the o at a scale for screenshots.
  CSS-only fallback: set `--lvl` on `.hud` (scales the o). `HudKit.levelMeter` / `.hud-bars` are
  kept for old markup only.
- **Live text** (`partial` events; the app HUD's replacement for the bubble): a single line
  `<span class="hud-live"><span class="hud-live-text">word spans</span><span class="hud-live-caret"></span></span>`
  after `.hud-text`. `.hud[data-live]` hides the label + hint and gives the line a fixed width
  (`--live-w`, 340 px), so the pill glides wider once (`HudKit.morph`) and never resizes while
  words stream in. New text is appended as a span that fades in (opacity only); words past
  `stable_chars` get `.is-pending` (ink-3). `HudKit.fitLive(line)` pins the newest words to
  the right edge and fades the oldest out on the left once the line overflows. Shown while
  recording/locked/handsfree/transcribing; gone at refining. Reduced motion: no fade, no blink.
  Off with `ui.show_partials: false`. The bubble markup stays in the theme previews only.
- Hit-testing: `.hud-pill`, `.hud-bubble` and `.hud-pip` rects, as today. Buttons get
  `pointer-events: auto`; everything else is `none`.
- Geometry matches `hud.rs`: `.hud { bottom: 30px }` = `PAD` (room for the hero shadow; the
  pill's screen position is unchanged, `BOTTOM_GAP` 36); pill 44 px; bubble ≤ 540 px wide,
  3 lines; total stack < 200 px, inside `HUD_H` 320.

## 8. Settings markup contract (`app/ui/theme/components.css`)

Shell: `.shell > aside.sidebar (.brand, nav.nav > a.nav-item[aria-current="page"], .nav-sep,
.sidebar-foot > .engine-status) + main.main > section.page`.
Page: `.page-head > h1.page-title + p.page-lede`; `.group > .group-head (h2.group-title,
.group-aside) + .card > .row`. Row: `.row > .row-text (.row-label, .row-desc) + .row-control`;
`.row.is-stacked` puts the control under the text; `.stack` is a padded block inside a card.

| Control | Markup | State |
|---|---|---|
| Segmented | `.seg[role=radiogroup][style="--n:3;--i:1"] > button[role=radio]` | `aria-checked`; set `--i` to the selected index (the thumb glides) |
| Toggle | `input.toggle[type=checkbox][role=switch]` | `:checked`, `:disabled` |
| Select | `span.select > select` | native |
| Text input | `input.input` (`.mono` for keys); `.field` wraps an `svg.icon-lead` and/or `button.field-btn` | `:focus` |
| Key + Test | `.key-field > (.field > input.input.mono[type=password] + button.field-btn) + button.btn` then `span.status[data-tone]` | tone: `success` / `danger` / `busy` (spinner) / `warning`; `.meta` for the detail |
| Key capture | `button.capture > kbd.keycap + span.capture-edit` | `[data-capturing]`, text "Press a key…" |
| Model picker | `.card.choices > label.choice > input[type=radio] + .choice-title + .choice-meta + .choice-side (+ .choice-progress > .progress + .num)` | `:checked` |
| Progress | `.progress[style="--value:.46"] > i` | `--value` 0..1 |
| Tags | `.tags > span.tag (text + button.tag-x) … + input.tag-input` | |
| Pairs | `.pairs > .pairs-head + .pair (input.input, span.pair-arrow, input.input, button.btn.btn-ghost.btn-icon.btn-sm)` | |
| Per-app | `.app-row > .app-icon + .app-name(small) + span.select + button` | |
| History | `.card.history > .day-label + .history-item (.history-time, .history-text, .history-meta, .history-actions)` | actions show on `:hover`/`:focus-within` |
| Buttons | `.btn` + `.btn-primary` / `.btn-ghost` / `.btn-danger`, `.btn-sm`, `.btn-icon` | `:disabled` |
| Badges / status | `.badge[data-tone]`, `.status[data-tone]`, `.callout[data-tone]`, `.dots-progress > i.on` | |

## 9. Icons

The SVG masters in `app/icons/src/` are the handoff's final artwork (committed, never generated).
`python app/icons/build_icons.py` rasterises them (headless Edge + Pillow) into `app/icons/out/`:
≤ 32 px from `ochre-icon-small.svg`, ≥ 48 px from `ochre-icon.svg`, `.icns` from
`ochre-icon-macos.svg` (no shadow is added; the system draws one). Outputs: `32x32.png`,
`128x128.png`, `128x128@2x.png`, `256x256.png`, `512x512.png`, `512x512@2x.png`, `icon.png`,
`app-<size>.png`, `favicon-32.png`, `icon.ico` (per-size renders 16/24/32/48/64/256, never a
downscaled 256) and `icon.icns` (via `iconutil -c icns` on macOS, where it exists; elsewhere
Pillow, which omits the 16@1x and 32@1x entries). `tray/` holds every state in `src/tray/` at
16 / 20 / 32 px + @2x: `ochre-tray-<state>-<light|dark>-<size>[@2x].png` and
`macos/ochre-template-<state>…` / `macos/ochre-recording…`; `tray.rs` embeds the 16@2x (Windows /
Linux) and 20@2x (macOS) renders. `--install` copies the bundle icons and `tray/` into
`app/src-tauri/icons/` (and refreshes `Square*Logo.png` / `StoreLogo.png` if the bundle ever ships
them). It also copies the artwork the UI uses into `app/ui/theme/brand/` and draws the contact
sheet `docs/design/icons.png`.

## 10. Previews and screenshots

- `app/ui/theme/hud-preview.html`: live autoplay through every state with a synthetic voice;
  buttons jump to a state. `?state=<s>&freeze` for a single static state, `?grid` for all.
- `app/ui/theme/settings-preview.html`: all seven sections; `?section=…&theme=dark`.
- `docs/design/hud-<state>.png` (light desk left, dark desk right), `hud-all.png`,
  `settings-<section>-<light|dark>.png`, `icons.png`.
- `python app/ui/theme/perf-check.py`: 0 layouts while the o meters (exit 1 otherwise).
- Brand: `docs/brand/index.html` (brand sheet), `docs/brand/assets/` (OG, README headers, HF
  thumbnail; `python docs/brand/render.py` re-exports them), the handoff in `docs/brand/handoff/`.

## 11. Self-critique (revision 2)

What works:
- Calm and flat. Paper, porcelain and ink do the work; orange appears only where something is
  live or actionable (toggle on, primary button, recording, the mark).
- With one colour, states are still unmistakable because each changes three things at once:
  glyph, motion and orange intensity. Transcribing has no orange; recording is orange marks;
  hands-free is an orange outline; locked is a solid orange pill. Refining's orange sweep +
  sparkle can't be confused with transcribing's grey sweep + dots.
- Spacing is concentric: the 32 px orb and the × sit 6 px inside a 44 px capsule. Settings
  rows keep a 16 px gutter and a 60 px minimum height.

Known weaknesses / next pass:
- Two oranges (`#D4400E` fills, `#F04E16` marks) is the price of AA with white text. They're
  close enough to read as one colour, but don't introduce a third.
- The hero shadow is the one tinted shadow. Keep it to the hero surfaces and keep it faint: if
  it starts to read as a glow (on a white document, in a screenshot), lower the alpha, never
  widen it.
- Existing installs keep whatever `ui.theme` they saved (older builds wrote `"system"`); only new
  configs default to light.
- A light HUD over a white document relies on its outline and shadow. It reads, but it's
  quieter than the old ink pill. If users lose it, raise `--hud-outline` to ~0.14.
- `--danger` crimson and vermilion are ~15° apart; error states also change glyph and copy,
  so colour is never the only cue.
- Frozen screenshots catch loops mid-cycle (shimmers, carets, breathing dots); live, they read
  correctly.
- `color-mix()` is used for hover tints and the error edge. Older WebKit loses only those tints.

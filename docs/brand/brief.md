# Ochre brand brief: logo, wordmark and icon system

For: a design agent. From: the Ochre team, 2026-10-05.
Read alongside: `docs/design.md` (the app's design system, "Paper & Vermilion"), the tokens in
`app/ui/theme/tokens.css`, and the screenshots in `docs/design/` and `docs/ui/`.

## 1. The product in one breath

**Ochre** is free, open-source dictation for Windows, macOS and Linux. Hold a key, talk, and
polished text lands in whatever text box you're in, about 150 ms after you let go. Speech
recognition and a small fine-tuned cleanup model run **on your own computer**. The cleanup model
removes "um"s, applies "no wait, make that Tuesday", and fixes punctuation, and it never rewrites
or answers you. Cloud is optional and brings your own API key.

- **Audience:** people who type all day (developers, writers, founders, support and ops) and
  care about speed, privacy and craft. Many are technical, and they install it from GitHub.
- **Positioning:** "A free, local, open-source alternative to Wispr Flow." We may name
  competitors in copy. We **must not** echo their marks: no "whisper/wispr/flow" wordplay, no
  wave-in-a-circle that reads as theirs.
- **Personality:** calm, fast, crafted, honest. A good pen, not a gadget. Confident without
  shouting. Warm, not corporate. **Not "AI-ified"** (see section 6).

## 2. The name

Ochre is the earth pigment humans used for their **first marks**: cave walls, handprints,
the oldest drawings, on every continent. The story writes itself: *the oldest way to make a
mark, for the newest way to write.* Speech becomes marks. That is the product.

Pronunciation: OH-ker. All lowercase in the CLI (`ochre`), and "Ochre" in prose.

## 3. What exists today (keep the continuity)

- **Surfaces:** warm paper `#F7F5F1`, porcelain `#FFFEFC`, near-black ink `#1A1714`. Flat, with
  soft neutral shadows.
- **One hero colour:** vermilion. `#D4400E` is the fill under white text, `#F04E16` the bright
  mark with no text on it, and `#C23A0C` is used for orange text. Dark mode uses the same orange,
  with `#FF8552` for text.
- **Type:** Instrument Sans for the UI and Geist Mono for keycaps and versions. The user has
  **rejected Instrument Serif**, and it is being removed; display text uses Instrument Sans for now.
  **Propose the type system** (display + UI + mono) as part of this brief.
- **Current mark:** a flat vermilion superellipse squircle. Inside it, a white voice wave loses
  energy across three humps and settles into a white vertical **text caret**: speech becomes
  text where you type. It works at 16 px. See `app/icons/src/` and `docs/design/icons.png`.
- **Current wordmark:** plain "Ochre" in Instrument Sans, a placeholder. **This must be replaced.**
- **The HUD "pill":** a small frosted porcelain capsule near the bottom of the screen. While you
  talk it shows level bars and live text. It turns solid vermilion when recording is locked, and
  a caret blinks at the end of the live text.

The new identity should feel like it was always part of this app. Redesigning the UI is out of
scope.

## 4. The colour question (explore it, then recommend)

The name says *ochre*, and the app says *vermilion*. They are cousins: **red ochre** is a real
pigment (iron oxide, hue ~12–18°), just earthier and less saturated than vermilion. Show two
routes and recommend one:

- **A. Keep vermilion** (`#F04E16` / `#D4400E`) exactly. The name carries the earth story and
  the colour stays bright and modern. This costs nothing in the app.
- **B. Shift toward red ochre** (for example around `#C2481C` for fills and `#E0582A` for marks).
  It is earthier and more ownable, and ties name and colour together. It must keep white text at
  AA (≥ 4.5:1) on the fill and stay clearly distinct from the danger crimson `#C8323F`. If you
  pick B, deliver the full replacement values for `--accent`, `--accent-2`, `--accent-ink`,
  `--hud-accent` and the dark-mode variants, with contrast ratios.

A yellow-ochre or mustard secondary may appear only as a tiny illustration accent. The UI stays
one hero colour.

## 5. Deliverables

1. **The mark.**
   - **Direction:** evolve or replace the wave-into-caret squircle. The core idea, *voice settles
     into a text caret / a mark*, is good and should survive in some form. Fresh directions are
     welcome, for example a hand-made pigment stroke that ends in a caret, or an "o" made of one
     brush or finger stroke.
   - **Exploration:** show 3 directions with a sentence each, then refine the chosen one.
   - **Required properties:**
     - legible at **16 px** (tray) and beautiful at 1024 px
     - works in one colour (black, white)
     - works on paper and on dark
     - no gradients or glow
2. **Wordmark "Ochre" and type system.** No Instrument Serif (rejected by the user). Propose the
   display, UI and mono faces (OFL), and draw the wordmark from or alongside the display face,
   with at most one accent detail as the signature. Deliver the type scale as token values. Show a lockup with the mark, both horizontal and stacked, plus the
   clear-space rule.
3. **App icon set:**
   - Windows `.ico` (16, 24, 32, 48, 64, 256)
   - macOS `.icns` (16–1024, following the macOS icon grid and squircle, with a slight depth
     allowed only if it stays flat-feeling)
   - Linux PNGs (16–512)
   - Tauri sources as square PNGs plus SVG master files
   - Put sources in `app/icons/src/` and keep `app/icons/build_icons.py` working, or update it.
4. **Tray and menu-bar states**, matching the existing behaviour (`docs/design.md` §4):
   - idle: ink squircle, white glyph, orange caret
   - **recording**: solid orange
   - busy: three dots
   - error: dimmed, with a red badge
   - off: grey
   - macOS: monochrome template images with a colour recording glyph
   - Deliver each state at 16/20/32 px, light and dark.
5. **Small brand assets:**
   - favicon (SVG + 32 px PNG)
   - GitHub social/OG image (1280×640)
   - README header banner (light and dark variants)
   - Hugging Face model-card thumbnail
6. **One-page brand sheet** as an HTML page in `docs/brand/`, covering:
   - mark, wordmark and lockups
   - colour values with contrast
   - type
   - do's and don'ts
   - the name story (section 2)
   - a sample of the pill HUD and the README header in context

## 6. Hard constraints

- **Not AI-ified:**
  - no purple or blue gradients, glows, sparkles or "magic wand" stars
  - no neural-net or brain imagery
  - no robot microphones
  - no soundwave-in-a-circle cliché

  The user's own feedback that triggered the current direction was "too AI-ified".
- **Flat.** Gradients are allowed only where the app already uses them for motion, never in the
  mark.
- **No competitor echoes:** nothing that reads as Wispr Flow, Whisper or OpenAI, Aqua Voice or
  Superwhisper.
- **Not amber.** Don't drift toward a warm amber (e.g. `#EA6A1F`), and don't use bars as the core
  of the mark.
- **Open source:** every font must be OFL (or similarly permissive) and self-hostable, and the
  SVG sources are committed.
- **Accessibility:** text colour pairs must meet WCAG AA. Check with
  `python app/ui/theme/contrast-check.py` after any token change.

## 7. How we'll judge it

1. **Recognisable at 16 px in a crowded Windows tray**, and it still reads as "the orange one"
   in monochrome.
2. **Fits the app:** put it beside the current Settings and HUD screenshots and it looks native.
3. **Tells the story without explanation:** speech becomes a mark or text.
4. **Ownable and calm:** no one mistakes it for a competitor or for generic AI.

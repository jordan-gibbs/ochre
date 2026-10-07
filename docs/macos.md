# macOS: status and setup

Status 2026-10-06. Tested on a MacBook Pro with Apple M5 Pro (48 GB), macOS 26.5.1, built-in
microphone, by the maintainer dictating and by scripted checks. Intel Macs are untested.

## Install

Build from source with `./scripts/install.sh` (or ask a coding agent to "install Ochre", see
[AGENTS.md](../AGENTS.md)). It builds `Ochre.app`, installs it in `/Applications` (or
`~/Applications`), links the `ochre` CLI into `~/.cargo/bin` and starts it. A real app bundle
matters: macOS gives permissions to the bundle, whereas `cargo run` or a bare binary would hand
them to your terminal.

Local builds are ad-hoc signed (`bundle.macOS.signingIdentity: "-"`). Set `APPLE_SIGNING_IDENTITY`
to a Developer ID to sign properly; with `APPLE_ID`, `APPLE_PASSWORD` and `APPLE_TEAM_ID` Tauri
also notarizes. A notarized download is only needed for prebuilt DMGs; locally built apps run
without it.

## First run

The setup window walks through three permissions, each with an **Allow…** button that shows the
system prompt and opens the right System Settings pane:

| Permission | Why |
|---|---|
| Microphone | to hear you (`NSMicrophoneUsageDescription` in `app/src-tauri/Info.plist`) |
| Accessibility | to swallow the Voice key and type text |
| Input Monitoring | to see the Voice key in every app |

The list updates by itself, and the Voice key starts working the moment both key permissions are
on, with no restart. **After an update** of an ad-hoc signed build the old switches stay on in
System Settings but no longer apply (the signature changed). `install.sh` clears them and reopens
the permissions step.

## Microphone and noise

- **Voice processing (default).** Ochre captures through Apple's voice processing, the path
  FaceTime and Zoom use. It cancels echo of anything the Mac plays, suppresses steady noise and
  levels your voice. Settings → General → Microphone → *Filter background noise* turns it off and
  goes back to the raw mic.
- **Voice Isolation.** Apps can't choose it; you pick it once in the Mac's microphone mode menu
  (the last setup step, or Settings → General → *Microphone mode*), and the Mac remembers it for
  Ochre. It removes other people's voices. Measured here: silence between words dropped from
  -36 dBFS (room chatter that Parakeet turned into words) to -68 to -85 dBFS.
- **The mic turns off when you're not using it.** Ochre keeps the microphone ready (open, with
  a pre-roll, so recording is instant) and releases it after 5 minutes without a dictation, and
  the orange mic dot goes with it. The next dictation opens the plain microphone at once
  (key-down → first sample p50 77 ms, p95 82 ms, measured over 20 cold starts) and switches back
  to voice processing in the background between dictations. Settings → General → *Keep the
  microphone ready* offers Always, 1–60 minutes, or Only while dictating. With hands-free on, the
  mic stays on (it listens for "transcribe") and the dot with it.
- **AirPods and other Bluetooth headphones keep their sound quality.** Recording from a headset's
  mic switches it to the low-quality call profile. With no input chosen, Ochre records from the
  Mac's built-in microphone while a Bluetooth headset is the default input (*Use the built-in mic
  with headphones*, on by default). Choosing the headset as the input device overrides it. A Mac
  without a built-in mic uses the headset. Not yet tested with a real headset: if your music
  still drops to call quality while Ochre runs, please open an issue.
- Voice processing ducks other apps' audio while it runs. Ochre sets ducking to the minimum
  (macOS 14+).

## Voice key

Right Option by default (tested). Right Command, fn (Globe), Right Control, F13–F19 and chords
with a regular key (Control + Shift + Space) are supported but not yet tested on a Mac. Caps Lock can't be held on macOS (it reports only a toggle), so it's refused.

**fn (Globe):** macOS acts on Globe below Ochre's event tap. Set System Settings → Keyboard →
"Press 🌐 key to" → **Do Nothing**, otherwise every dictation also opens the emoji picker,
switches input source or starts Apple Dictation (Ochre shows a notice saying which).

The paste fallback (long text) presses Cmd+V on whichever key types "v" in the current layout, so
Dvorak works (key code 47) and so do QWERTY, AZERTY and QWERTZ (key code 9).

## Local cleanup

Refinement is off by default; turn it on in Settings → Refinement. `auto` picks Metal on Apple
Silicon and the 4B model with 16 GB+ of memory. llama.cpp's macOS build runs as
a child process; it can't outlive Ochre, even after `kill -9`.

## Hands-free and calls

Hands-free pauses while another app captures audio (CoreAudio per-process input, macOS 14+).
System daemons that always hold the mic (`corespeechd` for "Hey Siri") don't count.

## Logs

`ochre quit`, then `RUST_LOG=info /Applications/Ochre.app/Contents/MacOS/ochre` shows the log in
the terminal; every dictation logs per-stage timings. Config, models and history are in
`~/Library/Application Support/ochre/`.

## Test results (M5 Pro, macOS 26.5.1)

| Check | Result |
|---|---|
| Bundle: mic usage string, entitlement, hardened runtime | ✅ |
| Permissions granted mid-session, Voice key starts without restart | ✅ |
| Parakeet on a clean clip | ✅ word-perfect, 82 ms for 5 s of audio |
| Full pipeline (segmenter + Parakeet) on a 10 s clip with pauses, at -3 to -30 dB | ✅ word-perfect, 142 ms release → text |
| Live dictation, release → text (refine on, 4B) | ✅ 435–770 ms (~335 ms of it is the post-roll) |
| Local refine (Quill 4B, Metal) | ✅ 120–470 ms per dictation; a 100-word paragraph 1.5 s (target 1 s) |
| Voice Isolation active for Ochre, background removed | ✅ |
| Idle 5 min (hands-free off): mic released, Ochre not capturing | ✅ |
| Cold start after release, key-down → first sample (20 tries) | ✅ p50 77 ms, p95 82 ms |
| llama-server gone after quit and after `kill -9` | ✅ |
| Fresh clone → `install.sh` (ad-hoc signed) | ✅ |
| Call detection ignores Siri daemons | ✅ (no real call tested) |
| AirPods / Bluetooth headset: built-in mic used, music stays at full quality | not tested yet (no headset) |
| Double-tap lock, Esc, paste-last chord, other Voice keys, secure input notice, paste > 200 chars, HUD over full-screen apps, tray icon in dark mode, start at login, idle CPU | not tested yet |

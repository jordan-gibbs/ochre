<p align="center">
  <strong>A free, local, open-source alternative to Wispr Flow.</strong><br>
  Hold a key, talk, and polished text lands in whatever text box you're in.<br>
  Fast, private, open source. Windows, macOS and Linux.
</p>

<img width="2560" height="1280" alt="social-preview" src="https://github.com/user-attachments/assets/ead18472-d597-4d69-a2b9-f2709473a733" />


**See it in action:**

https://github.com/user-attachments/assets/78b5c7a9-8fa1-4c4d-b439-02c90f8bacef


---

Ochre is a free alternative to subscription dictation apps. Speech recognition runs
**on your machine** by default. If you prefer the cloud, bring your own API key and pay cents an
hour instead of a monthly fee. An optional AI pass cleans up what you said: it removes "um"s,
applies your "no wait, make that Tuesday"s, and fixes punctuation, without ever rewriting your
words or answering your questions.

## Highlights

- **Fast.** Text appears ~150 ms after you release the key (median, measured on a Ryzen 7
  9800X3D, CPU only), even with the machine busy. Phrases are transcribed *while you talk*; the
  mic is always warm, so the first syllable is never clipped.
- **Accurate locally.** NVIDIA Parakeet TDT 0.6B (v3 "Ultra", int8) runs on ONNX Runtime and
  supports 25 European languages, with punctuation and casing built in. The choice is
  [benchmarked](docs/benchmarks.md) against Whisper, Qwen3-ASR, Granite, Moonshine and others.
- **Your keyboard, your rules.** Configurable Voice key. **Hold** to talk, **double-tap** to
  lock recording on, then **tap** to stop, and **Esc** to cancel. Text is typed with native
  keystrokes, or pasted with your clipboard restored for long text.
- **Hands-free.** Say **"transcribe"**, dictate, then say **"transcribe stop"** to insert,
  **"transcribe send"** to insert and press Enter, or **"transcribe cancel"** to discard. The
  custom wake-word model is [trained in this repo](training/wakeword/README.md) and false-wakes
  about 0.4 times per hour (the model itself is licensed for non-commercial use, see
  [License](#license)).
- **Optional cleanup.** Local or cloud (OpenAI, Google, Groq, OpenRouter, Anthropic). Locally it
  runs our own fine-tunes, **Ochre Refine 4B / 2B / 0.8B** (Qwen3.5, trained on real recognizer
  output). It removes fillers, applies self-corrections, fixes punctuation and writes numbers the
  way you would, without rewording you or answering your questions. On 773 dictations graded
  blind by three LLM judges, it passes **80% (4B), 71% (2B) and 64% (0.8B)**
  ([how it was trained and measured](docs/refine-finetune-v4.md)). The default, `auto`, picks by
  hardware: 4B on a GPU with 6 GB or more (or Apple Silicon with 16 GB), 2B on a smaller GPU,
  0.8B on CPU only. The models are [on Hugging Face](https://huggingface.co/polonuim210) under
  Apache-2.0. A guard falls back to your raw words if a model ever misbehaves.
- **Bring your own key.** Pick a connector, paste one key, done:

  | Connector | Speech-to-text | Cleanup | Rough cost |
  |---|---|---|---|
  | Local (default) | Parakeet on your CPU/GPU | Ochre Refine / Quill (optional) | free |
  | OpenAI | `gpt-transcribe` / live streaming | GPT-5.6 Luna (minimal reasoning) | cents per hour |
  | Google | Gemini 3.5 Transcribe | Gemini 3.5 Flash-Lite | cents per hour |
  | Soniox, Deepgram, Groq, ElevenLabs, AssemblyAI | mix and match under *Advanced* | | |

- **Private by design.** No telemetry, audio never touches disk, and history is local text you
  can turn off. API keys live in your OS keychain.

## Install

**Download** the installer for your system from the
[latest release](https://github.com/jordan-gibbs/ochre/releases/latest):

| System | File | First launch |
|---|---|---|
| Windows 10 / 11 (x64) | `Ochre_<version>_x64-setup.exe` | The installer isn't code-signed yet: on the SmartScreen prompt, click **More info → Run anyway**. |
| macOS 11+ (Apple Silicon) | `Ochre_<version>_aarch64.dmg` | Drag Ochre to Applications. If macOS says it can't verify the app, open **System Settings → Privacy & Security** and click **Open Anyway**. |
| Linux (x64) | `.deb` (Ubuntu 24.04+, Debian 13+), `.rpm` (Fedora 40+), or `.AppImage` | Needs glibc 2.38 or newer. See the Linux notes below for Wayland. |

macOS and Linux support is newer than Windows: if something doesn't work on your setup, please
[open an issue](https://github.com/jordan-gibbs/ochre/issues).

**Or build from source;** it's one command. You can also clone the repo and ask your coding agent
(Claude Code, Codex, Cursor…) to "install Ochre": [AGENTS.md](AGENTS.md) tells it how.

**Windows** (PowerShell):

```powershell
git clone https://github.com/jordan-gibbs/ochre; cd ochre
./scripts/install.ps1
```

**macOS / Linux:**

```sh
git clone https://github.com/jordan-gibbs/ochre && cd ochre
./scripts/install.sh
```

The script installs Rust if needed, plus the OS prerequisites (WebView2 on Windows;
webkit2gtk, libayatana-appindicator and ALSA dev packages on Linux). It then builds and
installs `ochre` (on macOS, `Ochre.app` in /Applications, started for you). On first launch
it downloads the speech model (~670 MB) with a progress bar; turn on **Start at login** in
Settings if you want it always running.

**macOS.** The first-run screen walks you through **Microphone**, **Accessibility** and **Input
Monitoring** (an *Allow…* button each; no restart needed). The microphone goes through the Mac's
own voice processing; in a noisy room, choose **Voice Isolation** in the microphone mode menu
(last setup step, or Settings → General). Using fn (Globe) as the Voice key? Set "Press 🌐 key
to" to *Do Nothing*. Details and test results: [docs/macos.md](docs/macos.md).

**Linux.** On X11 everything works as installed (the Voice key through XInput, typing with
`xdotool`). On Wayland:

- **Voice key:** Wayland gives apps no global keys, so Ochre reads `/dev/input`. Join the
  `input` group (`sudo usermod -aG input $USER`, then log out and back in), or bind a
  keyboard shortcut to `ochre toggle` in your desktop settings (press once to start, again to
  stop; no permissions needed).
- **Typing:** GNOME and KDE have no virtual-keyboard protocol, so Ochre types with `ydotool`
  through `/dev/uinput`. The `.deb` / `.rpm` and `install.sh` add a udev rule that lets the
  logged-in user use it, and Ochre starts `ydotoold` itself. Sway, Hyprland and other wlroots
  desktops use `wtype`. Text that `ydotool` can't type (non-ASCII, or a non-US layout) is
  pasted instead, and your clipboard is put back.
- **Tray:** Ubuntu shows the tray icon out of the box. On stock GNOME, install the
  *AppIndicator and KStatusNotifierItem Support* extension; without it there is no tray icon,
  and launching Ochre again from the app menu opens Settings.

## Use

| Do this | Get this |
|---|---|
| Hold **Right Alt** (Right Option on Mac), talk, release | Text is typed where your cursor is |
| Double-tap the key, talk, tap once | Hands-off recording for longer dictation |
| **Shift** held when you finish | Insert raw text, skipping cleanup for this one |
| **Esc** | Cancel |
| Hold the key and press **↓** (e.g. Right Ctrl + Down), or tray → **Paste last transcript** | Types your last dictation again where you're typing. No clipboard involved, no re-cleanup. Change or turn off in Settings → General (`hotkey.paste_last`: `"down"`, a chord like `"ctrl+alt+v"`, or `""`) |
| "transcribe … transcribe send" | Full hands-free (enable it in Settings → Hands-free) |
| `ochre toggle` / `ochre paste-last` | Start or stop / paste the last transcript, from scripts and custom shortcuts |

Everything you dictate is in **History** (tray → History), so nothing is lost if focus moved.

## How it works

```
key / wake word ─▶ warm mic (pre-roll) ─▶ phrase segmenter ─▶ speech-to-text (local or cloud)
                                              decodes while you talk          │
     injection ◀─ history ◀─ cleanup (optional, guarded) ◀─ hesitation strip ◀┘
```

One native Rust process (Tauri v2 for the pill, tray and settings). The design rules come
from measuring why an earlier app felt slow ([docs/latency.md](docs/latency.md)):
models stay resident and warm, decoding runs at raised priority without spin-waits, there is
no polling, text is typed before anything is saved, and every dictation logs its per-stage
timings.

| Crate | Does |
|---|---|
| `ochre-core` | Config, events, the engine traits, history, keychain |
| `ochre-audio` | Warm mic, resampling, phrase segmenter, VAD, earcons |
| `ochre-stt` | Local engines (Parakeet; Whisper behind a feature flag) |
| `ochre-cloud` | Cloud speech-to-text providers |
| `ochre-refine` | Cleanup: prompts, guard, Quill sidecar, cloud LLMs |
| `ochre-platform` | Hotkeys, gestures and text injection for Windows, macOS and Linux |
| `ochre-wake` | Wake word and hands-free controller |
| `ochre` | The orchestrator that wires it all together |
| `app/` | Tauri shell: HUD, tray, settings, onboarding |

See [SPEC.md](SPEC.md) for the full design and [docs/go-checklist.md](docs/go-checklist.md)
for the measured release gates.

## Development

```sh
cargo test                          # all crates
cargo run -p ochre-app              # the app (add -- --demo to drive every UI state)
cargo run -p ochre-stt --example bench --release -- clip.wav   # speech latency
cargo run -p ochre-refine --example bench --release            # cleanup latency
```

Model training lives in `training/`: the wake word (`training/wakeword`) and the cleanup
model (`training/refine`, with a synthetic-speech data pipeline and an eval harness in
`tools/eval`).

## Contributing

Bug reports, fixes and measured improvements are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) first: it covers building and testing, the PR flow (fork,
branch, PR to `main`; small focused PRs, one maintainer reviews everything) and what is in
scope. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md), and report security problems
privately as described in [SECURITY.md](SECURITY.md).

## License

Ochre's source code is released under the [MIT License](LICENSE), Copyright (c) 2026 Jordan Gibbs.

Model weights, fonts and bundled binaries keep their own licences, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): Parakeet is CC-BY-4.0, Qwen3.5, Quill and the
Ochre Refine fine-tunes are Apache-2.0, the openWakeWord front end is Apache-2.0, and the fonts
are under the SIL Open Font License 1.1.

The one exception is the **"transcribe" wake-word model** (`assets/wake/transcribe.onnx`), which is
licensed [CC BY-NC-SA 4.0](assets/wake/LICENSE): free for non-commercial use. Some of the data it
was trained on only allows non-commercial use (openWakeWord licenses its own models the same
way). The training code is MIT, and everything else in Ochre works without the wake word.

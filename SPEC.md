# Ochre — Spec

Open-source, cross-platform (Windows, macOS, Linux) voice dictation in the style of
Wispr Flow / Aqua Voice. Hold a key (or say a wake word), talk, and polished text lands in
whatever text box has focus. Everything can run locally; every stage can also be pointed at a
cloud provider with your own API key, so dictation costs cents a month instead of a subscription.

Status: v0.1 build-out, in Rust (a Python prototype was started and dropped on 2026-10-04; its
modules remain in git history as porting reference). Name: **Ochre** (binary `ochre`, repo
`jordan-gibbs/ochre`); it was called "Open Whisperflow" (`openwhisprflow`) before 2026-10-05.

## 1. Goals

1. **Fast and accurate local transcription.** Release-to-text under ~400 ms for a 10 s
   utterance on a modern CPU; better formatted-text accuracy than a Granite Speech 470M
   TurboCTC + punctuation-model pipeline (see docs/benchmarks.md).
2. **Text injection into the focused field** on all three OSes, without hijacking the
   clipboard by default.
3. **Optional AI refinement:** filler removal, self-correction handling ("no wait, make that
   Tuesday"), punctuation and formatting, and per-app tone. Runs on a local ~1B model
   (fine-tuned by us, see §9) or any cloud LLM.
4. **Bring your own key (BYOK).** Each of STT and refinement can be set independently to
   local or cloud: local+local, local+cloud, cloud+local, cloud+cloud.
5. **Hands-free mode:** a trained wake word ("transcribe") starts dictation; "transcribe
   stop" inserts, "transcribe send" inserts and presses Enter, "transcribe cancel" discards.
6. **Beautiful, quiet UI** in a frosted-pill HUD language (see §7).
7. **Installable from the repo** with one command per OS; no Python knowledge needed to use it.

Non-goals for v0.1: mobile, accounts or sync, a hosted service, voice commands beyond the
hands-free control phrases, and streaming partial text *into* the target app (partials show
in the HUD only).

## 2. Lineage

Ochre builds on earlier dictation and voice-assistant prototypes. What carries over:

- **Dictation pipeline:** a line-delimited worker protocol shape (hello/ready/level/processing/
  result/error events), phrase segmentation during capture (decode completed phrases while the
  user talks, cut at the quietest point), resampling, post-STT corrections (preferred spelling,
  whole-phrase replacements) and a model store with resumable, SHA-256-verified downloads.
- **Input:** the hold / double-tap-to-lock gesture state machine, Unicode `SendInput` injection
  (no clipboard), the leading-space rule, Escape to cancel, the 10-minute session cap and
  save-to-history-before-insert.
- **Hands-free:** the wake word runtime (openWakeWord mel + embedding front end, generic ONNX
  heads), VAD, the wake training pipeline, the Soniox client, earcons, and a frosted-pill HUD
  as the design reference.

**What changes:** the local recognizer. A Granite 470M TurboCTC build is replaced by **NVIDIA Parakeet
TDT 0.6B v3** (transducer, CC-BY-4.0, 25 European languages, auto punctuation and
capitalization) run through ONNX Runtime (`onnx-asr`, int8), which removes the transcribe.cpp
native build and the separate punctuation model.

## 3. Architecture (Rust)

The app is a single native Rust process. There is no Python at runtime: Python survives only as
offline tooling (`tools/`: STT evaluation, wake-word training, refinement fine-tuning).

```
┌──────────────────── ochre (one Rust process, Tauri v2) ────────────────────┐
│ ochre-platform: hotkey hook ─► gesture state machine ─┐      ochre-wake: wake detector  │
│                                                     ▼              │               │
│ ochre-audio: cpal stream (always warm) ─► ring buffer / pre-roll ─► segmenter ─► VAD  │
│                                                     │ phrases (decoded while talking)│
│                                         ochre-stt: local (ort / GGUF) or cloud (BYOK)  │
│                                                     ▼                               │
│ ochre (orchestrator): corrections ─► ochre-refine (Quill / cloud, guarded) ─► history   │
│                                                     ▼                               │
│                                         ochre-platform: inject into the focused field │
│ Bus ─► Tauri events ─► webviews: HUD pill · settings · onboarding;  tray icon (Rust)│
└─────────────────────────────────────────────────────────────────────────────────────┘
        optional sidecar: llama-server (local refinement), started once, kept warm
```

- **Crates:** `ochre-core` (config, events, traits, history, secrets), `ochre-audio`, `ochre-stt`,
  `ochre-refine`, `ochre-platform`, `ochre-wake`, `ochre` (orchestrator library plus a headless `ochre-cli`
  CLI binary), and `app/src-tauri` (the GUI binary, with web UI in `app/ui`).
- **UI:** Tauri v2 (system webview; a few MB instead of Electron's ~150 MB). The HUD is a
  transparent, always-on-top, non-focusable, click-through window. Tray via Tauri's tray API.
  The single-instance plugin forwards `ochre toggle|start|stop|cancel|paste-last` from a second invocation to
  the running app, which also covers Wayland users who bind a compositor shortcut.
- **Threads:** the hook thread only emits gestures. The audio callback only copies into a ring
  buffer. One decode thread owns the STT engine. One session thread runs the tail (finish,
  refine, inject). Nothing blocks the hook or the audio callback.

### 3.1 Latency budget and the rules here

A fast model can still feel slow; where the time goes is in `docs/latency.md`. The main trap:
a CPU engine at Normal priority with fixed threads that spin-wait can take many times longer
under background CPU load than at Above Normal priority. The twelve rules in `docs/latency.md`
are binding. The core rules:

1. **Resident and warm:** models load at startup, and one warm-up inference runs before the
   app reports Idle. No per-dictation process spawn, model load, or interpreter startup.
2. **The mic is already open:** the input stream stays running while the app runs
   (`audio.warm_mic`, default on), with a 1.5 s ring buffer. Pressing the key flags "recording
   from now minus ~150 ms", so device-open latency and first-syllable clipping disappear.
3. **Decode while talking:** completed phrases are decoded during speech. After release, only
   the tail phrase is on the critical path.
4. **No polling loops** on the hot path. Use channels and condvars; no sleep(40 ms) queues.
5. **One batched injection:** a single `SendInput` call (or chunked CGEvents) per result. No
   per-character round trips and no accessibility caret lookups.
6. **Refinement is optional and bounded:** run it only when enabled; Quill stays warm in a
   resident llama-server; hard timeout, then raw text.
7. **Measure everything:** every session logs `Timings` (audio, stt_tail, refine, inject,
   release_to_insert). Targets on a modern CPU: **release→text in the box ≤ 300 ms** for a 10 s
   utterance without refinement, and ≤ 1 s with local refinement of a paragraph.
   `ochre-cli bench` replays WAVs through the real pipeline and prints the breakdown.

## 4. Speech-to-text

### 4.1 Contract (`ochre-core::stt::SttEngine`)

See `crates/ochre-core/src/stt.rs`. Engines are sync and blocking, called from one decode
thread. `load()` downloads/verifies and runs a warm-up inference.

`SttResult(text, language, duration_ms, processing_ms, words=None)`. Engines may also offer
`stream()` for live partials, which the HUD shows. Cloud engines that support only batch get
phrase segments from the segmenter, so release latency stays at one phrase.

### 4.2 Local engines

The default is **whatever wins the benchmark** in `docs/benchmarks.md`: a survey of the newest
small open ASR models as of Oct 2026, measured against Parakeet TDT 0.6B v3, Whisper
large-v3-turbo and Granite Speech. Candidates must have an MIT-compatible license and a
Rust-viable runtime:

- ONNX via the `ort` crate (CPU, CUDA, DirectML, CoreML execution providers)
- sherpa-onnx
- whisper.cpp (`whisper-rs`)
- GGUF / llama.cpp

| Engine | Model | Runtime | Notes |
|---|---|---|---|
| `parakeet` (baseline) | `parakeet-tdt-0.6b-v3`, int8 ONNX | `ort`, TDT greedy decode in Rust | Punctuation and casing built in; 25 European languages. |
| `whisper` | `large-v3-turbo` GGUF/ggml | `whisper-rs` (CUDA / Metal / CPU) | 99 languages; vocabulary biasing through the initial prompt. |
| survey winner(s) | TBD by benchmark | TBD | |

The personal dictionary is fed to Whisper as `initial_prompt` and to every engine through
post-STT corrections (`ochre::text::corrections`).

The segmenter decodes completed phrases are decoded during
capture, so release waits on one phrase only. Long sessions are capped at 10 minutes.

Acceptance: `tools/eval/eval_stt.py` reports WER and latency on a public English set (LibriSpeech
test-clean subset plus a noisy/conversational subset) for parakeet, whisper and, where
available, a Granite Speech build for comparison. Results go to `docs/benchmarks.md`.

### 4.3 Cloud engines (BYOK)

Soniox, Deepgram (Nova-3), OpenAI (`gpt-transcribe` default; `gpt-live-transcribe` streaming;
`gpt-4o-mini-transcribe`, `gpt-4o-transcribe`), Google (Gemini 3.5 Transcribe, preview), Groq
(`whisper-large-v3-turbo`), ElevenLabs (Scribe), and AssemblyAI. Each module uses plain
`reqwest` / `tungstenite` with no vendor SDKs, keeps a timeout of `stt.cloud_timeout_ms` (default
8 s), and **falls back to the local engine** on failure when one is installed.
Keys come from `secrets.get("<provider>")`; `google` shares the `gemini` key (`secrets::ALIASES`).

**Streaming.** Engines whose `SttEngine::streaming()` is true (Soniox real-time, OpenAI
`gpt-live-transcribe`, Google `gemini-3.5-transcribe-live`) bypass the segmenter: the
orchestrator opens the stream on key-down off the hotkey thread (`ochre::stream`), sends audio as it
is captured, and on release waits only for the provider's final. If opening, sending or the final
fails, the whole recording is transcribed once in batch (with the usual local fallback). Batch
engines get phrase segments from the segmenter. `prewarm()` re-opens the pooled TLS / HTTP/2
connection on key-down, off-thread, at most every 20 s.

Measured on a ~7 s TTS clip, warm connection (Oct 4 2026; docs/refinement.md §6.1):
`gpt-transcribe` batch 0.66-2.33 s for the whole clip; `gpt-live-transcribe` streamed at real-time
pace 0.46-0.82 s release -> final (the `delay` setting made no measurable difference);
Soniox real-time 0.11-0.17 s release -> final.

**Cloud connectors** (`ochre::connectors`, sent to the UI as `Event::Connectors`): one provider,
one key, both stages set with one `set_config` patch. Per-stage mix-and-match stays under
"Advanced" in settings.

| connector | key | speech-to-text | refinement | ≈ $/h of dictation | measured release latency |
|---|---|---|---|---:|---|
| `openai` | `openai` | `openai` / `gpt-live-transcribe` (streaming) | `openai` / `gpt-5.6-luna`, effort `none`, clean | 1.10 | STT final 0.46-0.82 s after release; refine 1.1-1.4 s |
| `google` | `gemini` | `google` / `gemini-3.5-transcribe` (batch, SMART) | `gemini` / `gemini-3.5-flash-lite`, thinking minimal, clean (light) | 0.43 | not measured (no key); Flash-Lite via OpenRouter 0.59-1.25 s |
| `groq` | `groq` | `groq` / `whisper-large-v3-turbo` | `groq` / `openai/gpt-oss-20b`, effort low | 0.13 | not measured (no key) |
| `soniox+openai` | `soniox` + `openai` | `soniox` real-time (streaming) | `openai` / `gpt-5.6-luna` | 0.20 | STT final 0.11-0.17 s after release |
| `local` | none | `parakeet` | `local` (Quill) or off | 0 | see docs/benchmarks.md |


## 5. Refinement

### 5.1 Contract (`ochre-core::refine::Refiner`)

See `crates/ochre-core/src/refine.rs`.

`RefineContext(app_name, window_title, style, dictionary, language)`. Refinement is
**optional** and off by default until the local model ships.

- Modes: `off`, `clean` (fillers, false starts, self-corrections, punctuation, no rewording),
  and `polish` (light rewrite for clarity). Per-app overrides (Slack casual, Mail formal,
  IDE literal).
- Hard rules: never answer or act on the text, never add content, and return the raw text if
  the output is empty, more than 2x longer, or past `refine.timeout_ms` (default 1500 local,
  3000 cloud).
- The prompts live in `ochre-refine::prompts` and are shared by the local and cloud paths, so the
  fine-tuning data (§9) matches inference.

### 5.2 Providers

- **Local:** a managed `llama-server` (prebuilt llama.cpp binary, downloaded per OS and
  architecture on first use) serving a GGUF on 127.0.0.1, driven through the OpenAI-compatible
  API. Any existing OpenAI-compatible local server also works (Ollama, LM Studio).
  **Default model: `auto`**, resolved at load by hardware (`crates/ochre-refine/src/local/auto.rs`):
  Ochre Refine 4B with a GPU of >= 6 GB (or Apple Silicon with >= 16 GB), 2B on a smaller GPU,
  0.8B without one (docs/refine-finetune-v3.md). They are private for now; without a Hugging Face
  token auto picks the same-size [Quobi/Quill](https://huggingface.co/Quobi/Quill) model
  (Apache-2.0 fine-tunes of Qwen3.5 for post-ASR cleanup; `quill-0.8b` is 505 MB). Every model is
  also selectable by id. Quill must be driven through llama-server's raw `/completion`
  endpoint: ChatML with the assistant turn pre-seeded with an empty `<think>

</think>

`
  block, `temperature 0`, and **no `--jinja`**, which leaks chain-of-thought. 0.8B is verbatim
  only by design, so pair it with the deterministic scaffold in `ochre-refine::normalize` (numbers,
  times, emails, URLs, symbols).
  **Latency budget: under 1 s for a ~100-word paragraph** on CPU, and well under on a GPU.
  `tools/bench_refine.py` and `ochre-cli bench --refine` measure this. Our own fine-tune (§9) replaces Quill only if it
  beats it on the eval.
- **Cloud:** OpenAI-compatible (OpenAI, Groq, OpenRouter, Together, Fireworks, Cerebras, …),
  Anthropic, and Gemini. Defaults: OpenAI
  [`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna) with
  `reasoning_effort: "none"` through Chat Completions (Luna's efforts are none | low | medium |
  high | xhigh | max; `"minimal"` is rejected with HTTP 400, verified live), measured 0.64-1.21 s
  for a short line and 1.22-1.54 s for a ~100-word paragraph; Gemini
  [`gemini-3.5-flash-lite`](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
  with `thinkingLevel: "minimal"` (its default; Gemini 3.x thinking cannot be turned off). Both use
  the shared prompt (`<dictation>` wrapping) and the guard; only the answer text is read, never
  reasoning or thought parts. The connector table in §4.3 pairs them with a transcriber.

## 6. Triggers, injection, hands-free

### 6.1 Hotkey gestures (`ochre-platform`)

The gestures:

| Gesture | Effect |
|---|---|
| Hold the Voice key | Record while held; release finishes and inserts. |
| Double-tap the Voice key | Locked recording; a single tap finishes. |
| Escape while recording | Cancel. |
| Voice key + Shift (configurable) | Insert raw text and skip refinement for this one. |
| Voice key + Down (`hotkey.paste_last`) | Paste the last transcript again: the take the press just started is dropped silently, Down is swallowed (press, repeats, release) and fires once, and the Voice key's release is inert. A chord value (`ctrl+alt+v`) is a standalone shortcut instead; `""` turns it off. Also in the tray and as `ochre paste-last`. Linux listeners cannot swallow keys, so there Down also reaches the focused app. |

Default Voice key: Right Alt (Windows/Linux) and Right Option (macOS). It can be changed to
Right Ctrl, Caps Lock, Menu, Insert, Scroll Lock, Pause, F13–F20, or a chord.
Implementation:

- **Windows:** low-level keyboard hook (`SetWindowsHookEx WH_KEYBOARD_LL`) on a
  dedicated thread (`windows-sys`). It swallows the Voice key so Right Alt doesn't open menus, and never jams
  the keyboard (see the hardening notes in `crates/ochre-platform/src/windows/hook.rs`).
- **macOS:** a CGEventTap on its own run loop thread (needs Accessibility and Input Monitoring).
- **Linux:** X11 via XInput2 (`x11rb`). On Wayland, evdev (`/dev/input`, user in the `input` group) or a
  compositor-bound shortcut that runs `ochre toggle`.

### 6.2 Injection (`ochre-platform`)

- **Windows:** `SendInput` with `KEYEVENTF_UNICODE`; no clipboard, including the safety rules (no injection into elevated windows
  from a non-elevated process, which shows a HUD notice instead).
- **macOS:** `CGEventKeyboardSetUnicodeString` in chunks of 20 UTF-16 units.
- **Linux:** `xdotool type` (X11); on Wayland `wtype` (wlroots), `kwtype` (KDE), `dotool` or
  `ydotool`.
- **Fallback (all OSes):** clipboard paste with save and restore, used for very long text
  (`inject.paste_over_chars`, default 200) or when typing is blocked.
- **Leading space:** prepend one space when the previous insertion into the same app and window
  ended with a non-space and less than `inject.join_window_s` seconds have passed. There is no
  caret lookup (it is unreliable across apps).
- **Type first, then persist** (amended after `docs/latency.md`, rule 6): nothing but the
  injection sits between the decode and the text appearing. The history row and the timing log
  are written on a background thread right after. The text is also in the `Result` event, and
  in history within milliseconds, so it survives a focus change.
- "Send" (hands-free) presses Enter after the insertion completes.

### 6.3 Hands-free mode (`ochre-wake`)

- **Always-on low-power listening:** VAD gates a streaming wake detector (openWakeWord front
  end plus our ONNX head). Default phrase **"transcribe"**, threshold
  tuned for at most 0.5 false wakes per hour; any phrase can be trained.
- On wake, the session starts: an earcon plays, the HUD shows "Dictating — say *transcribe
  stop*", and a 1.5 s pre-roll is kept so the first words are not clipped.
- Phrases are transcribed as they complete (segmenter + STT). Each finished phrase is checked
  for a **trailing control phrase**, which is fuzzy-matched on the transcript and stripped from
  the text:
  - "transcribe stop" or "transcribe done" → finish and insert
  - "transcribe send" → finish, insert, press Enter
  - "transcribe cancel" → discard
  - "transcribe scratch that" → drop the last phrase
- Wake-model hits during a session also arm control-phrase detection, which makes the trailing
  command more robust.
- Auto-finish after `handsfree.idle_timeout_s` (default 45 s) of silence, inserting what was
  said.
- Paused while another app is using the mic for a call (Windows first).
- The wake word is off by default; it is enabled in settings. The mic indicator stays on while
  it is active, which the UI states plainly.

Wake training lives in `training/wakeword/` (a word-generic pipeline: Piper
TTS positives, near-miss negatives, RIR and noise augmentation, livekit conv-attention head
→ ONNX). `ochre-cli train-wake --word <w>` runs it, and a GPU is recommended.

## 7. UI (Tauri v2, `app/`)

Visual details are in `docs/design.md`.

- **HUD:** a small pill at center-bottom just above the taskbar or dock. Frosted white pill,
  near-black Figtree text, one warm-orange accent (see `docs/design.md`),
  with light and dark variants. Always on top, click-through except its own buttons, never
  takes focus, excluded from screen capture where the OS allows.
  - States: `listening` (orange dot + live level bars), `dictating` (orange pill "Dictating…
    release to paste" + bars + ×), `locked` ("tap to finish"), `transcribing` (shimmer),
    `refining`, `inserted` (brief check, then fade), `wake-armed` (tiny idle pip, optional),
    `error` (message + retry), and `notice`.
  - A live transcript bubble above the pill shows partial text, with unconfirmed words in grey,
    as in the `dictation.png` shot.
- **Settings window:** the same tokens, in a calm single column. Sections: General (Voice key,
  start at login, language), Transcription (Local / Cloud selector, model picker with download
  progress, provider + key field + "Test"), Refinement (Off / Local / Cloud, mode, per-app
  styles), Hands-free (enable, phrase, sensitivity, train), Dictionary (preferred spellings,
  replacements), History (search, copy, re-insert, clear), and About.
- **Tray / menu bar:** status, start/stop, mode toggles, settings, quit.
- **Onboarding:** first run walks through mic permission, the macOS Accessibility and Input
  Monitoring grants, the model download, and a try-it box.
- The UI never holds keys in renderer memory longer than the save call; keys go to the core,
  which stores them in the OS keyring.

### 7.1 Core ⇄ UI protocol (Tauri events + commands)

Core → UI events (`{"event": ..., ...}`): `hello{version}`, `state{state, mode, trigger}`,
`level{rms}`, `partial{text, stable_chars}`, `result{id, raw, text, inserted, ms}`,
`error{message, code}`, `notice{message}`, `download{model, done, total}`,
`config{config}`, and `history{items}`.

UI → core commands (`{"op": ..., ...}`): `start`, `stop`, `cancel`, `get_config`,
`set_config{patch}`, `set_secret{provider, key}`, `test_provider{stage, provider}`,
`download_model{name}`, `history_query{q, limit}`, `insert_text{text}`, `train_wake{word}`,
and `quit`.

The full schema is `ochre-core::events` (`Event` / `Command`), the single source of truth.

## 8. Config and secrets

- TOML at `<user config dir>/ochre/config.toml` (Windows: `%APPDATA%\ochre\config\config.toml`),
  typed in `ochre-core::config` with unknown keys preserved.
- Keys go to the OS keyring (`keyring`: Windows Credential Manager, macOS Keychain, Secret
  Service). Env vars such as `OCHRE_OPENAI_API_KEY`, `SONIOX_API_KEY` and `OPENAI_API_KEY` are
  read as a fallback. Keys are never written to config, logs or history.
- Models are stored in `user_data_dir/models`, and history in `user_data_dir/history.sqlite3`
  (Windows: `%LOCALAPPDATA%\ochre\data\...`). `OCHRE_CONFIG_DIR`, `OCHRE_DATA_DIR` and
  `OCHRE_MODELS_DIR` override them.
- Rename migration: at startup the app moves the pre-rename `openwhisprflow` config/data dirs to
  `ochre` once (rename, falling back to copy) if the new ones don't exist yet
  (`ochre_core::paths::migrate_legacy_dirs`). Keys stored under the old `openwhisprflow` keyring
  service are still read as a fallback.

## 9. Refinement model fine-tuning (maintainer-led)

`training/refine/` gets a README and the data format only. The plan, to be worked out
together:

- Baseline to beat: Quill 0.8B/2B. Candidate base: Qwen3.5-0.8B (same family, so the same
  latency profile).
- Data: (raw ASR transcript, clean text) pairs. They come from synthetic disfluency injection
  over clean text plus real ASR output (Parakeet run on TTS'd and real speech), with the same
  prompt as `ochre-refine::prompts`.
- Training: LoRA/QLoRA SFT on a single 16 GB consumer GPU, then exported to GGUF Q4_K_M/Q8_0 for
  `llama-server`.
- Eval: a held-out set scored on edit faithfulness (no added content, no meaning changes) and
  on cleanup quality.

## 10. Install and distribution

- **From the repo:** `scripts/install.ps1` (Windows) and `scripts/install.sh` (macOS/Linux)
  check for or install the Rust toolchain plus OS prerequisites (WebView2 on Windows; webkit2gtk,
  libayatana-appindicator and ALSA dev packages on Linux), then run
  `cargo install --path app/src-tauri` (or `cargo tauri build` for a bundle) and register start
  at login.
- **Headless:** `cargo install --path crates/ochre` gives the `ochre-cli` CLI (`ochre-cli run --headless`,
  `ochre-cli transcribe file.wav`, `ochre-cli bench`).
- **Feature flags:** `cuda`, `directml`, `coreml` (ort execution providers); CPU by default.
- **Later:** signed installers via `tauri bundle` (MSI/NSIS, DMG, AppImage/deb) and the
  Tauri updater.
- **CI:** GitHub Actions running fmt, clippy and test on all three OSes, with
  hardware-dependent tests `#[ignore]`d.

## 11. Privacy

- No telemetry.
- Audio stays in memory and is dropped after transcription.
- With cloud providers selected, audio or text goes only to the provider the user configured;
  settings shows exactly which.
- History is local and can be turned off.

## 12. Milestones

1. Core plumbing: config, events, history, orchestrator, CLI, headless run.
2. Local STT (Parakeet + Whisper), segmenter, eval vs alternatives.
3. Hotkeys + injection on Windows, then macOS and Linux.
4. Cloud STT and refinement providers; local `llama-server` manager.
5. Tauri HUD, settings and tray.
6. Wake word "transcribe" trained, plus the hands-free controller.
7. Fine-tuned local refinement model (§9).
8. Packaged installers.

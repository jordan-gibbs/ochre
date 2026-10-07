# Go checklist

Nothing is called done until every box below is checked **with a measured number or a
reproducible command recorded next to it**. "Should work" does not count.

Status 2026-10-04: most numbers so far were measured with the CPU near 100 % from parallel
jobs (agents, training, rendering), so they are under-load figures; idle reruns are pending.

## Speed (measured on the dev box: Ryzen 7 9800X3D, RTX 5070 Ti, Windows 11)

| Gate | Target | Measured | How |
|---|---|---|---|
| Release → text in the box, 10 s utterance, local STT, no refinement | ≤ 300 ms (CPU) | | `ochre-cli bench --pipeline` (real segmenter + engine + injector into a test window) |
| Same, GPU | ≤ 150 ms | ⏸ GPU decode p50 42 ms, but needs ORT 1.30 via load-dynamic (ort's bundled 1.24 lacks Blackwell kernels); CPU is the shipped default | `ochre-cli bench --pipeline --features cuda` |
| Tail-phrase decode only | ≤ 150 ms CPU | ✅ ≤ 65 ms (tail p95 1.6 s of audio), under load | `cargo run -p ochre-stt --example bench --release` |
| Local refinement, ~100-word paragraph (Quill) | ≤ 1 s GPU | ✅ 0.8B 320 ms (37 ms prompt + 281 ms gen); 2B ~480 ms. CPU-only 1.26 s under load ❌ (rerun idle) | `cargo run -p ochre-refine --example bench --release` |
| Key press → HUD visible | ≤ 1 frame (16 ms) after the state event | | HUD timing log |
| Key press → audio captured (pre-roll covers the first syllable) | 0 ms clipped | | pre-roll test: speech starting at t=0 is in the transcript |
| Injection, median dictation (~100 chars, typed) | ≤ 70 ms on this hook-heavy box (~0.65 ms/key from other apps' hooks) | | ochre-platform live test |
| Injection, 600-char paragraph (pasted, > `paste_over_chars` = 200) | ≤ 50 ms to the paste call | | ochre-platform live test |
| Cold start → Idle (models cached) | ≤ 3 s | ⚠️ 5.1 s in the real app under heavy load (Parakeet load 4.8 s); 1.9–2.3 s in the bench. Rerun idle | `ochre-cli run --headless` timestamps |
| Idle CPU with warm mic + hands-free armed | ≤ 2 % of one core | ✅ wake detector 0.03–0.28 % (energy gate; Silero cost not yet included) | ochre-wake example `--mic` |
| Under load: `ochre-cli bench --pipeline --background-load 8` | p95 ≤ 2× idle, ≤ 400 ms | ✅ decode p95 0.92× idle with 8 busy threads (1.75× without the priority boost); release→text loaded p95 218 ms | docs/latency.md rule 1 |
| Release → last key sent, idle | ≤ 200 ms | ✅ release→text p50 143 / p95 276 ms (pipeline_bench, real-time feed, 150 ms post-roll); + injection ~65 ms for 100 chars on this hook-heavy box. Full in-app measurement pending | docs/latency.md budget table |

## Accuracy

| Gate | Target | Measured | How |
|---|---|---|---|
| Default local engine WER, clean set | best-in-survey at its size | ✅ Parakeet Ultra int8 best formatted-output model measured (docs/benchmarks.md); Granite has lower *normalized* WER but no punctuation/casing | `tools/eval/eval_stt.py` → `docs/benchmarks.md` |
| Default local engine WER, hard/noisy set | recorded vs alternatives | | same |
| Rust engine matches the reference implementation | identical or near-identical text on the eval clips | ✅ 68/73 identical, 0.26 % WER between Rust and onnx-asr; 5.5× faster | ochre-stt correctness check |
| Wake word "transcribe" | ≥ 90 % streamed detection, ≤ 0.5 false wakes/h | ✅ 95.0 % / 0.37 per hour (synthetic voices) | `training/wakeword/README.md` |
| Control phrases | the full table passes, no mid-sentence triggers | | `cargo test -p ochre-wake` |

## Function (each verified live on Windows; macOS/Linux at least compile-checked and code-reviewed)

- [ ] Hold → speak → release types text into Notepad, a browser text box, Slack/Discord, VS Code, and a terminal
- [ ] Double-tap → locked → single tap finishes; Escape cancels; a short single tap does nothing
- [ ] Voice key + Down types the last transcript once (no cursor move in the target, no earcon/HUD error, clipboard unchanged); tray → Paste last transcript types into the window that was focused before the tray click
- [ ] The hotkey is changeable in settings and works immediately
- [ ] Each cloud STT provider with a key works; a bad key gives a clear error and falls back to local
- [ ] Local refinement (Quill) and one cloud refiner work; the guard rejects an "answering" output
- [ ] Hands-free: "transcribe … transcribe send" inserts and presses Enter; "cancel" discards
- [ ] Every result is in history before insertion; re-insert from history works
- [ ] An elevated target window gives a clear notice instead of silently failing
- [ ] Tray: status, start/stop, hands-free toggle, settings, quit
- [ ] First-run onboarding downloads models with visible progress and ends in a working "try it" box
- [ ] `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test` all clean; CI green on all three OSes

## Design

- [ ] Distinct visual identity (colors, fonts, mark), documented in `docs/design.md`
- [ ] Every HUD state screenshotted light and dark in `docs/design/`, reviewed, no clipping or jank
- [ ] Loading, transcribing and refining are unmistakably different at a glance
- [ ] Level meter and transitions are smooth (transform-only animation; reduced-motion respected)

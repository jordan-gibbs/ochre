# Third-party notices

Ochre's own source code is released under the [MIT License](LICENSE). It uses, downloads or
bundles the third-party works below, which keep their own licences. Model weights and native
binaries are **not** stored in this repository: Ochre downloads them on first use from the
listed source and verifies each file's SHA-256 before using it.

Rust and JavaScript library dependencies are listed in `Cargo.lock` with their own licences
(mostly MIT / Apache-2.0); run `cargo about` or `cargo license` for a full report.

## Speech recognition models (downloaded at runtime)

| Work | Source | Licence |
|---|---|---|
| NVIDIA Parakeet TDT 0.6B v3 (ONNX export by istupakov) | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | CC-BY-4.0 |
| NVIDIA Parakeet TDT 0.6B v2 (ONNX export by istupakov) | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx) | CC-BY-4.0 |
| Parakeet Ultra (ONNX export by Olicorne) | [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra), [Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx) | CC-BY-4.0 |
| OpenAI Whisper (ggml weights, optional engine) | [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) | MIT |
| Silero VAD (ONNX) | [snakers4/silero-vad](https://github.com/snakers4/silero-vad), [istupakov/silero-vad-onnx](https://huggingface.co/istupakov/silero-vad-onnx) | MIT |

**Attribution (CC-BY-4.0):** "Parakeet TDT 0.6B v3" and "Parakeet TDT 0.6B v2" by NVIDIA
Corporation, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Ochre uses ONNX conversions
(int8-quantized) of the original weights; NVIDIA does not endorse Ochre.

## Text cleanup models (downloaded at runtime)

| Work | Source | Licence |
|---|---|---|
| Ochre Refine 4B / 2B / 0.8B (LoRA fine-tunes of Qwen3.5, GGUF) | [polonuim210/ochre-refine-4b](https://huggingface.co/polonuim210/ochre-refine-4b), [-2b](https://huggingface.co/polonuim210/ochre-refine-2b), [-0.8b](https://huggingface.co/polonuim210/ochre-refine-0.8b) | Apache-2.0 |
| Qwen3.5 4B / 2B / 0.8B (base models of Ochre Refine) | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and siblings, Alibaba Cloud | Apache-2.0 |
| Quill 4B / 2B / 0.8B (GGUF) | [Quobi/Quill](https://huggingface.co/Quobi/Quill) | Apache-2.0 |

## Wake word

| Work | Source | Licence |
|---|---|---|
| openWakeWord feature models (`melspectrogram.onnx`, `embedding_model.onnx`, v0.5.1; downloaded at runtime) | [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord) | Apache-2.0 |
| "transcribe" wake word head (`assets/wake/transcribe.onnx`, trained for Ochre with `training/wakeword`) | this repository | **CC BY-NC-SA 4.0** (non-commercial; see [`assets/wake/LICENSE`](assets/wake/LICENSE)). The training code is MIT. |

## Native binaries

| Work | How it is used | Licence |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | Downloaded from the official GitHub release at runtime, runs the local cleanup model | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Shared library bundled with the app (via the `ort` crate) | MIT |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Compiled in when the optional `whisper` feature is enabled (via `whisper-rs`) | MIT |
| [Tauri](https://github.com/tauri-apps/tauri) | Application shell | MIT or Apache-2.0 |

## Fonts (bundled in `app/ui/theme/fonts`)

All fonts are licensed under the [SIL Open Font License 1.1](https://openfontlicense.org); the
full licence text ships next to each font.

| Font | Copyright | Licence file |
|---|---|---|
| Instrument Sans | Copyright 2022 The Instrument Sans Project Authors | `OFL-InstrumentSans.txt` |
| Geist Mono | Copyright 2024 The Geist Project Authors | `OFL-GeistMono.txt` |
| Funnel Display | Copyright 2023 The Funnel Project Authors | `OFL-FunnelDisplay.txt` |

## Training data (not distributed with the app)

The training pipelines in `training/` and `tools/` produce **text** datasets that are committed
to this repository; no audio is published. The audio used to build them was synthesized with
Piper (LibriTTS-R voices, CC-BY-4.0 data), Kokoro-82M (Apache-2.0), Qwen3-TTS (Apache-2.0) and
Windows SAPI voices, then transcribed by Parakeet (CC-BY-4.0) and Whisper (MIT). The scripts and
target labels were written and audited with large language models. No user data was used.

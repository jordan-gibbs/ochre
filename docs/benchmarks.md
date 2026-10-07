# Local speech-to-text: survey and benchmarks

Measured 2026-10-04 on the development machine. Tool: `scripts/eval_stt.py`. Raw JSON (every
hypothesis included) was kept out of the repo; rerun the commands at the end to regenerate it.

## TL;DR

| | Model and artifact | Why |
|---|---|---|
| **Default** | **Parakeet TDT 0.6B v3 "Ultra", int8 ONNX** (`Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx`, `int8/`, 668 MB) | Best accuracy we measured among models that are fast on CPU and carry punctuation and casing natively. 25 languages, CC-BY-4.0. It runs on the same pipeline as v3 (`ort`, TDT greedy decoding) with no code change. Its int8 export keeps fp32 accuracy and is the fastest Parakeet build on CUDA. |
| **Runner-up** | **Parakeet TDT 0.6B v2, int8 ONNX** (`istupakov/parakeet-tdt-0.6b-v2-onnx`, 661 MB), English only | Same pipeline; within 0.3 WER of Ultra (better on AMI and LibriSpeech-other, worse on Earnings22). |
| GPU option | Parakeet v3 **fp32** (`istupakov/...-v3-onnx` `encoder-model.onnx` + `.data`, 2.5 GB) or Ultra fp32 | Same accuracy as Ultra int8 and about 100 ms per 10 s on the RTX 5070 Ti. Not worth a 2.5 GB download on its own. |

Do **not** ship istupakov's `parakeet-tdt-0.6b-v3` **int8** files, the original plan:
1. Their quantization costs about 1.1 WER points (6.59% vs 5.49% fp32, averaged over four sets, 500 clips each).
2. They run 5 to 8 times slower on CUDA than either fp32 or the Ultra int8 export.

## Method

- **Data.** The HF Open ASR Leaderboard test sets (`hf-audio/open-asr-leaderboard`), first parquet shard of each, clips of 1–30 s, sampled with a fixed stride. The same clips are used for every engine at a given `n`.
  - `clean` and `other`: LibriSpeech test-clean and test-other (read speech).
  - `ami`: AMI meetings (spontaneous, overlapping, distant microphones).
  - `earnings22`: earnings calls (accents, numbers, company names).
- **WER.** Normalized with Whisper's English normalizer (`whisper-normalizer`), then jiwer. This is the leaderboard convention: case, punctuation and fillers are ignored, and spelling and numbers are normalized.
- **fWER (formatted WER).** Case is kept, `. , ? !` count as tokens, and fillers (um, uh, …) are dropped on both sides. Computed only on AMI and Earnings22, whose references are cased and punctuated. It measures what a dictation user actually sees, which normalized WER deliberately hides.
- **Latency.** One fixed 10.0 s utterance (consecutive LibriSpeech clips), 2 warm-up runs, then the median of 15 timed runs. "Load" is engine construction with the model files already in the OS cache.
- **Hardware.** AMD Ryzen 7 9800X3D (8 cores / 16 threads, AVX-512), RTX 5070 Ti 16 GB, Windows 11.
  - **The machine was shared**: wake-word training, Rust builds and benches, and other projects' test suites kept the CPU at 62–100% during every CPU latency run. CPU latencies below are therefore **upper bounds**.
  - The GPU was shared too (refine benchmarks), so GPU timings are also somewhat noisy.
- **Sampling error.** With 150 clips per set (about 2–4k words each), differences under about 0.3 points on clean/other and about 1 point on AMI/Earnings22 are noise. The finalists were re-run with 500 clips per set.

## Results

### Accuracy: all candidates, the same 150 clips per set

WER % (lower is better). "Avg" is the mean over the four sets.

| Engine (artifact) | clean | other | AMI | Earn22 | **Avg** | fWER AMI | fWER Earn22 |
|---|---|---|---|---|---|---|---|
| Cohere Transcribe 2B int8 (sherpa-onnx) | 1.54 | 2.19 | 6.41 | 10.46 | 5.15 | 16.98 | **20.36** |
| Parakeet v3 **fp32** (istupakov) | 2.15 | 3.91 | 7.56 | 9.70 | 5.83 | **15.85** | 22.69 |
| **Parakeet Ultra int8** (Olicorne) | 2.02 | 3.41 | 8.03 | 9.92 | 5.84 | 16.70 | 20.98 |
| Parakeet Ultra fp32 (Olicorne) | 1.96 | 3.44 | 8.00 | 9.99 | 5.85 | – | – |
| Parakeet v2 int8 (istupakov, English) | 2.28 | 2.94 | 7.48 | 11.95 | 6.16 | 16.05 | 24.10 |
| Parakeet v3 int8 (istupakov) | 2.47 | 4.77 | 7.69 | 13.67 | 7.15 | 16.12 | 26.94 |
| Whisper large-v3-turbo (faster-whisper fp16) | 1.96 | 3.41 | 13.74 | 10.51 | 7.40 | 23.45 | 20.92 |
| Qwen3-ASR 0.6B int8 (sherpa-onnx) | 2.72 | 5.38 | 11.57 | 12.59 | 8.07 | 25.47 | 25.27 |

Where each run executed:
- Parakeet and Whisper rows ran on CUDA.
- Cohere and Qwen3 rows ran on CPU.
- A CPU re-run of Parakeet Ultra int8 is reported under [Runtime notes](#runtime-notes); same model and files, so WER should match to within decoding noise.

### Accuracy: finalists, 500 clips per set (CUDA)

| Engine | clean | other | AMI | Earn22 | **Avg** |
|---|---|---|---|---|---|
| Parakeet v3 fp32 | 2.18 | 3.44 | 7.01 | 9.34 | **5.49** |
| Parakeet Ultra fp32 | 1.90 | 3.32 | 7.48 | 9.34 | 5.51 |
| **Parakeet Ultra int8** | 1.96 | 3.38 | 7.47 | 9.27 | **5.52** |
| Parakeet v2 int8 | 2.00 | 3.09 | 7.04 | 11.19 | 5.83 |
| Parakeet v3 int8 (istupakov) | 2.39 | 4.39 | 7.61 | 11.97 | 6.59 |
| Whisper large-v3-turbo | 2.17 | 3.72 | 13.27 | 10.43 | 7.40 |

### Latency: one 10 s utterance (median of 15; p95 in parentheses)

| Engine | CPU, contended (ms) | CPU, lighter load (ms per 10 s) | GPU RTX 5070 Ti (ms) | GPU, batch RTF over 500 clips (ms per 10 s) | Load (s) |
|---|---|---|---|---|---|
| **Parakeet Ultra int8** (ORT) | 685 (1140); 966 on a second pass | **≈340** (8-clip sample) | **131** (228) | 140–275 | 3.8–5.0 |
| Parakeet v3 int8 (ORT) | 909 (1150); 1163 | ≈560 (20-clip sample) | 579 (826) | 486–1236 | 4.2–4.8 |
| Parakeet v2 int8 (ORT) | 945–1135 | – | 871 (1744) | 537–617 | 3.8–6.6 |
| Parakeet v3 fp32 (ORT) | – | – | 417 under 100% CPU load | **87–129** | 5–27 (2.5 GB) |
| Parakeet Ultra fp32 (ORT) | – | – | **95** (131) | 99–154 | 3–7 |
| Whisper large-v3-turbo (CTranslate2) | gave up after 28 min starved; seconds per utterance | – | 165 (197) | 140–284 | 3–5 |
| Granite TurboCTC Q8 (transcribe.cpp built here) | 769 (1603) at 62% load | – | not built (see notes) | – | 0.7 |
| Parakeet v3 Q8_0 GGUF (transcribe.cpp) | 493 median, 9,503 p95 | – | – | – | 7.7 |
| Cohere Transcribe int8 (sherpa-onnx) | 2,838 | 1,165–1,695 (batch RTF) | 2,364 (sherpa CUDA, no real speed-up) | – | 14–18 |
| Qwen3-ASR 0.6B int8 (sherpa-onnx) | 3,417 | 2,070–3,155 (batch RTF) | – | – | 6 |

How to read this:
- **Budget**: SPEC asks for under ~400 ms release-to-text for a 10 s utterance on a modern CPU. Only the Parakeet family (and Granite, without its formatting) gets near it.
- The segmenter decodes completed phrases during capture, so release usually waits on the last phrase only, not on the full 10 s.
- On an idle 8-core CPU, Ultra int8 measured about 340 ms per 10 s. That is in line with the published RTFx of about 36 for Parakeet int8 on this CPU class (about 280 ms).
- On CUDA, every int8 ONNX export except Ultra's is a poor fit; see the findings.

## Findings

1. **Quantization matters more than the checkpoint.**
   - On English, Parakeet v3 fp32, Ultra fp32 and Ultra int8 are statistically tied (5.49 / 5.51 / 5.52 average over 2,000 clips).
   - istupakov's v3 int8 (ORT dynamic quantization) is 1.1 points worse, and 2.3 points worse on Earnings22, where numbers and names live.
   - Olicorne's int8 export (MatMulNBits 8-bit, block 64, with 12-bit mantissa pre-rounding) keeps fp32 accuracy at 668 MB.
   - So for English, "Ultra" mainly buys a better int8. Its own card claims gains on multilingual FLEURS, which we did not test.
2. **Dynamic-int8 exports run badly on CUDA.**
   - Speeds per 10 s on the RTX 5070 Ti: istupakov v3 int8 579 ms, v2 int8 871 ms (both likely fall back to CPU kernels); Ultra int8 131 ms, fp32 95–130 ms.
   - A single int8 artifact that is good on both CPU and GPU is another reason to pick Ultra.
3. **Whisper large-v3-turbo** is competitive on read speech but worst on AMI (13–14% WER): it drops and invents words on spontaneous, overlapping speech. It is also slow on CPU. Keep it only as the 99-language option.
4. **Cohere Transcribe (2B)** has the best formatted WER on Earnings22 and the second-best normalized average. But it costs 1.2–1.7 s per 10 s on CPU, and sherpa-onnx's CUDA build gave no real speed-up. It is a candidate for an optional "accurate (GPU)" tier through transcribe-cpp or ORT on CUDA, which we did not measure.
5. **Qwen3-ASR 0.6B** is worse than Parakeet on every set and 5–10 times slower on CPU. Not adopted.
6. **ggml (transcribe.cpp) degrades sharply when the CPU is oversubscribed.**
   - Under the same load, Parakeet GGUF's p95 reached 9.5 s, while ONNX Runtime slowed only about 2×.
   - A dictation app shares the CPU with whatever the user is doing, so a ggml backend must cap threads well below the core count. `ort` is the safer default.

## Runtime notes

- **transcribe.cpp** was built from source here (commit `48559bb`, MSVC + Ninja, AVX-512 CPU backend) to measure GGUF.
  - The CUDA build failed on Windows `MAX_PATH` inside the scratch folder. Fixing that means building from a shorter path, which was not done. GGUF on GPU is therefore **not measured**.
- **sherpa-onnx CUDA**: the `1.13.8+cuda12.cudnn9` wheel needed CUDA 13 cuBLAS on `PATH` and then ran Cohere at 2.4 s per 10 s, so it is effectively not accelerated. Not investigated further.
- **CPU vs CUDA WER for Parakeet Ultra int8**: re-run on CPU over the same 150 clips: clean 1.96% (CUDA 2.02%), AMI 8.08% (8.03%), AMI fWER 16.85% (16.70%). The two providers match within decoding noise.
  - That same run took about 2.7 s per 10 s, because the CPU was at 100% from other jobs. It is a reminder that CPU timings on this shared machine are upper bounds.

## Candidates considered (survey, October 2026)

Sources: the HF Open ASR Leaderboard results CSV (`hf-audio/open-asr-leaderboard-results`, updated
2026-10-01), HF trending and newest ASR models, and vendor pages. Leaderboard numbers are the
"cleaned" 2026 sets, so they are not comparable with older model cards. "Board avg" is that
leaderboard's English average.

| Model | Params / int8 size | License | Board avg | Punct. + case | Rust-viable runtime | Verdict |
|---|---|---|---|---|---|---|
| **Parakeet TDT 0.6B v3 "Ultra"** (moondream post-train; `Olicorne/...-ultra-onnx`) | 0.6B / 668 MB | CC-BY-4.0 | not listed | yes | `ort` (same files and pipeline as v3), parakeet-rs, sherpa-onnx (`mldecode/parakeet-ultra-onnx-int8`) | **Default.** Best measured int8 on CPU, fastest on CUDA. |
| Parakeet TDT 0.6B v3 (NVIDIA) | 0.6B / 670 MB int8 | CC-BY-4.0 | 4.86 | yes | `ort`, transcribe-rs, parakeet-rs, sherpa-onnx, transcribe-cpp (GGUF) | fp32 is as good as Ultra; **istupakov's int8 export loses about 1.1 WER** (measured). |
| **Parakeet TDT 0.6B v2** (NVIDIA, English) | 0.6B / 661 MB | CC-BY-4.0 | 4.70 | yes | same as v3 | **Runner-up** (English-only option). |
| Granite Speech 5.0 470M TurboCTC (IBM) | 0.47B / 506 MB Q8 GGUF | Apache-2.0 | 5.03 | **no** (lowercase, no punctuation) | transcribe-cpp (GGUF); community ONNX (`qwertz92/...-onnx`) | No native punctuation or casing, so formatted output needs a second model. Fastest model of all. |
| Cohere Transcribe 03-2026 | 2B / 1.7 GB int8 | Apache-2.0 (HF gated; sherpa repackages it) | 4.67 | yes | sherpa-onnx (Rust crate), transcribe-rs, transcribe-cpp | Accurate but about 1.4 to 1.7 s per 10 s on CPU. Too slow for the default; possible GPU tier. |
| Qwen3-ASR 0.6B | 0.78B / 879 MB int8 | Apache-2.0 | 5.05 | yes | sherpa-onnx, transcribe-cpp, llama.cpp (experimental) | LLM-style autoregressive decoder; slower and no more accurate (see results). |
| Qwen3-ASR 1.7B | 2B | Apache-2.0 | **4.31** (best open model) | yes | transcribe-cpp, llama.cpp (experimental) | GPU-only class; no int8 ONNX we could run. Not measured. |
| Canary-Qwen 2.5B, Canary 1B v2 (NVIDIA) | 2.5B / 1B | CC-BY-4.0 | 4.43 / 5.71 | yes | NeMo only / onnx-asr AED | 2.5B is too big for CPU; 1B v2 is worse than Parakeet. |
| Granite Speech 4.1 2B (IBM) | 2B | Apache-2.0 | 4.62 | yes | transformers only | No practical Rust runtime. |
| Whisper large-v3-turbo (OpenAI) | 0.8B / 0.9 GB Q8 | MIT | 6.36 | yes | whisper-rs (whisper.cpp), transcribe-cpp | Measured: worst on AMI (hallucinates and drops words on overlapping speech). Keep as the multilingual (99 languages) option. |
| Kyutai stt-2.6b-en, Voxtral Mini 4B, Phi-4-MM, VibeVoice-ASR, Omnilingual 7B | 2.6–7B | various | 5.6–6.5 | yes | mostly transformers | Too large for the CPU budget; no better than Parakeet. |
| Moonshine streaming (tiny/small/medium) | 34–245M | MIT | not listed (self-reported 6.65 old-style) | yes | sherpa-onnx, transcribe-rs | Small and fast, but clearly less accurate. Possible low-end fallback. |
| orukeet (v3-based) | 0.6B | **CC-BY-SA-4.0** | not listed | yes | onnx-asr layout | Share-alike license and marginal English gains. Skipped. |
| Parakeet "redux", Phonon-2 | ternary / 2-bit | CC-BY-4.0 | 5.17 (Phonon-2) | yes | vendor runtimes only | No open runtime. Skipped. |
| granite-speech-5.0-470m-turboctc-**nc** | 0.47B | **CC-BY-NC-SA** | 4.84 | no | – | Non-commercial license. Excluded. |
| NVIDIA nemotron-speech-streaming / parakeet-unified | 0.6B | NVIDIA Open Model License | 5.25 / not listed | yes | sherpa-onnx | Streaming variants; worse offline WER, and the license needs review. |
| ARK-ASR-0.6B (AutoArk) | 1.15B | Apache-2.0 | 4.56 | yes | transformers only | No ONNX or GGUF route. Watch. |

## Rust implementation spec for the default (Parakeet TDT, istupakov/Olicorne ONNX layout)

Applies unchanged to v3, Ultra and v2; only the repo paths differ. These details were verified
against onnx-asr 0.12 (`preprocessors/numpy_preprocessor.py`, `models/nemo.py`, `asr.py`), which
produced every Parakeet number on this page. `crates/ochre-stt` already follows this layout.

**Files** (Ultra, revision `3fd3b4d9772b2e595a9162f91f929f16bb4ab4cd`):

| local name | repo path | bytes | sha256 |
|---|---|---|---|
| `encoder-model.int8.onnx` | `int8/encoder-model.int8.onnx` | 649,537,325 | `8a2b4716…3e01ed` |
| `decoder_joint-model.int8.onnx` | `int8/decoder_joint-model.int8.onnx` | 18,203,490 | `f7e2db39…0bd32` |
| `vocab.txt` | `vocab.txt` | 93,939 | `d5854467…e3c35d` |
| `config.json` | `config.json` | 97 | `666903c7…ac466` |

Full hashes are in `src/openwhisprflow/stt/parakeet.py` (`VARIANTS`).

The Ultra int8 encoder uses `MatMulNBits` (8-bit, block 64). That op has CPU and CUDA kernels in
ONNX Runtime 1.30. DirectML and CoreML support is **not verified**; if they lack it, the op falls
back to CPU.

**Features** (128 log-mels, the same as NeMo):
- 16 kHz mono f32 input.
- Pre-emphasis `x[t] - 0.97*x[t-1]`, with `x[0]` kept.
- Zero-pad 256 samples on both sides. Frames of 512 every 160 samples.
- Window: symmetric Hann(400), zero-padded to 512 and centred (pad 56 each side).
- Power spectrum `|rfft(512)|^2` over 257 bins.
- Times the `nemo128` mel filterbank (257×128, Slaney scale and Slaney norm, 0–8 kHz). The exact matrix ships in onnx-asr as `preprocessors/data/fbanks.npz["nemo128"]`; embed it rather than recompute it.
- `ln(mel + 2^-24)`.
- `features_lens = n_samples // 160`.
- Per-mel-bin normalisation over the valid frames: `(x - mean) / (std + 1e-5)`, with std using an `N-1` denominator. Frames at or beyond `features_lens` are set to 0.
- Shape `[1, 128, T]`.

**Encoder:**
- Inputs `audio_signal` f32 `[1,128,T]` and `length` i64 `[1]`.
- Outputs `outputs` f32 `[1,1024,T/8]` and `encoded_lengths` i64.
- One encoder frame is 80 ms.

**Decoder/joint** (one call per step):
- Inputs:
  - `encoder_outputs` f32 `[1,1024,1]`
  - `targets` **i32** `[1,1]` (the last emitted token, or blank at the start)
  - `target_length` **i32** `[1]` = 1
  - `input_states_1` and `input_states_2` f32 `[2,1,640]`
- Outputs: `outputs` `[1,1,1,V+5]`, `output_states_1` and `output_states_2`.
- The first `V` values are token logits; the last 5 are duration logits for durations `[0,1,2,3,4]`.

**TDT greedy decoding:**
```
t = 0; emitted = 0; state = zeros
while t < enc_len:
    out, new_state = decoder_joint(enc[t], last_token_or_blank, state)
    token = argmax(out[:V]); step = argmax(out[V:])
    if token != blank: tokens.push(token); state = new_state; emitted += 1
    if step > 0:                         t += step; emitted = 0
    elif token == blank or emitted == 10: t += 1;   emitted = 0
```
- The state advances only on a non-blank token.
- `blank` is the id of `<blk>` in `vocab.txt`, which is the last id (8192 for v3/Ultra, 1024 for v2), so `V` = number of lines.
- Note: transcribe-rs discards the duration head and runs plain RNN-T greedy decoding. That is correct but makes more decoder calls.

**Detokenization:**
- `vocab.txt` lines are `<token> <id>`. Replace `▁` (U+2581) with a space and concatenate.
- Then apply `re.sub(r"\A\s|\s\B|(\s)\b", lambda m: " " if m.group(1) else "", text)`. This drops the leading space and any space not followed by a word boundary, such as before punctuation. Rust's `regex` crate takes the same pattern.

**Segments:** at most 30 s per call; longer audio is cut at the quietest 20 ms frame of the last
5 s (`crates/ochre-stt/src/chunk.rs`, ported from `audio/segmenter.py`).

### Runtimes, as of 2026-10

- **`ort`** 2.0.0-rc.13 bundles ONNX Runtime 1.30, still a release candidate.
  - Prebuilt EP builds exist for Windows x64 (DirectML; CUDA 13 + TensorRT), macOS arm64 (CoreML), and Linux x64 (CUDA 13).
  - There is **no Intel-Mac prebuilt**.
  - Handy reports that pyke's Windows build requires AVX2.
- **transcribe-cpp** (MIT, 0.3.1) runs GGUF builds of Parakeet, Granite, Cohere, Qwen3-ASR and Whisper on ggml (CPU, Vulkan, Metal, CUDA). It is what Handy now ships; GGUFs are under `huggingface.co/handy-computer`. It builds from source (CMake + C++). See [Runtime notes](#runtime-notes) for what we measured.
- **sherpa-onnx** (official Rust crate, 1.13.8) supports Parakeet (from its own export layout), Cohere and Qwen3-ASR. Its prebuilt libraries appear to be CPU-only.
- **whisper-rs** 0.16 (whisper.cpp) supports Whisper with CUDA, Metal or Vulkan.

## Reproduce

```powershell
# WER on four sets (150 clips each), CPU and GPU builds of the same ONNX files
uv run python scripts/eval_stt.py --engines parakeet:parakeet-ultra@cpu parakeet:parakeet-tdt-0.6b-v2@cpu `
    --sets clean other ami earnings22 -n 150 --out results.json
# release-latency reference: a fixed 10 s utterance, 15 timed runs after warm-up
uv run python scripts/eval_stt.py --engines parakeet:parakeet-ultra@cpu -n 0 --latency
```

GPU runs used a separate venv with `onnxruntime-gpu` 1.30 (CUDA 13 wheels: `nvidia-cublas`,
`nvidia-cuda-runtime`, `nvidia-cufft`, `nvidia-cudnn-cu13`) and `faster-whisper` (CUDA 12 +
cuDNN 9 wheels). See the `scripts/eval_stt.py` docstring for the sherpa-onnx and
transcribe.cpp backends.

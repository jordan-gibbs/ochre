# Refinement model fine-tuning

Owned by Jordan + Claude (SPEC §9). Nothing in this directory trains anything yet. This file
defines the **data format** and the **plan to beat Quill**. Prompt text lives in
`src/openwhisprflow/refine/prompts.py` and is specified in `docs/refinement.md`. Training data
must be rendered with exactly those prompts, or inference and training will drift apart.

## Baseline to beat

[Quobi/Quill](https://huggingface.co/Quobi/Quill), Qwen3.5 fine-tunes, Q4_K_M GGUF, run through
llama-server `/completion` with ChatML plus an empty think block (see `docs/refinement.md`).
Measured with `scripts/bench_refine.py` (Oct 2026, RTX 5070 Ti / Ryzen 7 9800X3D):

- **Quill's own prompt** ("You clean up dictated text." + the bare transcript) **answers some
  dictations instead of cleaning them**. "um can you send me the the file when you get a chance"
  became "I can't send files directly, but I can help...", and "what time does the pharmacy on
  main street close today" became "The pharmacy on Main Street closes at 18:00 today." The guard
  caught both, so the raw text was typed. That makes it unusable as-is for anything that sounds
  like a request.
- With **our shared prompt** (rules + `<dictation>` tags) 0.8B stops answering, but it often keeps
  self-corrections verbatim ("Monday, no wait, make that Tuesday") and never formats lists.
- 0.8B is verbatim-only: no number/email/URL conversion (we do that deterministically in
  `text/normalize.py` after the model).

What a fine-tune must fix, in priority order:

1. **Never answer or act**: questions and requests come back as cleaned questions/requests.
2. **Self-corrections**: keep only the corrected version.
3. Fillers, stutters, false starts, punctuation and casing (Quill already does this well).
4. Lists when clearly dictated as lists.
5. Same or better latency: stay on the Qwen3.5-0.8B base (hybrid SSM/attention, ~100 output
   tok/s on CPU), so a ~100-word paragraph stays under 1 s on CPU.

## Data format (JSONL)

One example per line. `prompts.py` renders each example into the exact model input; the trainer
should consume the rendered `messages` (or `text`) rather than re-templating.

```json
{"id": "synth-000123",
 "source": "synthetic|parakeet|human",
 "mode": "clean",
 "style": "",
 "dictionary": ["Kubernetes"],
 "raw": "um so can you uh send me the the report by friday no wait make that thursday",
 "clean": "Can you send me the report by Thursday?",
 "tags": ["filler", "repeat", "self_correction", "request"]}
```

Fields:

| field | meaning |
|---|---|
| `raw` | What the STT engine produced (lower-case or cased, with or without punctuation, exactly as an engine outputs it). |
| `clean` | The target. **Spoken forms are kept as words** ("three thirty", "john at gmail dot com") **when training the verbatim 0.8B tier**, because `normalize.py` runs after the model. For tiers that do their own normalization, set `"normalized": true` and write the written form. |
| `mode` | `clean` or `polish`, which selects the system prompt. |
| `style` | `""`, `casual`, `formal` or `literal`, which appends the style hint. |
| `dictionary` | Optional preferred spellings, which are appended to the system prompt. |
| `tags` | Phenomena present, used for stratified eval only. |

Rendering (must match `refine/local.py` for non-Quill models):

```python
from openwhisprflow.refine import prompts
from openwhisprflow.refine.base import RefineContext
ctx = RefineContext(mode=ex["mode"], style=ex["style"], dictionary=ex["dictionary"])
prompt = prompts.chatml(prompts.system_prompt(ctx), prompts.user_message(ex["raw"]))
target = ex["clean"] + "<|im_end|>"
# loss only on `target`; `prompt` already ends with the empty think block.
```

## Data sources (plan)

1. **Synthetic disfluency injection** over clean text (emails, chat messages, docs, code-review
   comments, meeting notes): insert fillers, repeats, false starts and self-corrections ("X, no
   wait, Y" with Y replacing X), lower-case and strip punctuation, and spell out numbers, emails
   and URLs. The clean source is the target.
2. **Answer-bait negatives** (the most important set): questions, requests and commands
   addressed to "you" ("can you...", "write a...", "what's the..."). The target is the cleaned
   question or request itself. Aim for at least 20% of the data.
3. **Real ASR output**: TTS (several Piper/Kokoro voices) over the clean text, then Parakeet and
   Whisper transcription, so `raw` has real recognition errors and real engine casing/punctuation.
4. **Human dictation**: a few hundred real recordings for the held-out eval only.
5. **Lists**: dictated with explicit markers ("first... second...", "bullet point...") → markdown
   list targets. Without markers → no list.
6. Already-clean inputs → identical output (teaches "leave it alone").

## Training (plan)

- LoRA/QLoRA SFT on Qwen3.5-0.8B (and 2B), on the RTX 5070 Ti; loss on the assistant span only.
- Keep the empty think block in every example (the model must never start thinking).
- Export: merge, convert to GGUF, quantize Q4_K_M and Q8_0; run through
  `scripts/bench_refine.py` with `--styles shared` and point `refine.model` at the `.gguf` path.

### Stack (built 2026-10-04)

Base: HF `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` (the post-trained model;
`-Base` is the raw pretrain), downloaded to `data/base/Qwen3.5-2B`. LoRA r16/α32 on every
decoder layer: `in_proj_qkv`/`in_proj_z`/`out_proj` in the 18 Gated DeltaNet layers,
`q/k/v/o_proj` in the 6 attention layers, `gate/up/down_proj` everywhere (MTP head and vision
tower untouched). Venv: `requirements-train.txt`. llama.cpp source is in `.llama.cpp`, pinned to
`b11398` (`a7b94df2c`), the same build as the app's llama-server.

From `training/refine/` (PowerShell is more reliable than Git Bash on this box under load):

```
.venv-train-refine\Scripts\python render.py <files.jsonl> --check      # Rust byte-parity, template parity, eval leakage
.venv-train-refine\Scripts\python train.py configs\qwen35-2b-lora.yaml --set run_name=v2 `
    --set "train_files=[datasets/real-v2/<audited>/*.jsonl]"          # add --resume after a crash
.venv-train-refine\Scripts\python export.py output\v2\adapter          # merge, verify, GGUF bf16, Q4_K_M, Q8_0, serve check (resumable)
cd ..\..; .venv\Scripts\python training\refine\eval_gguf.py training\refine\output\v2\gguf\v2-Q4_K_M.gguf
```

`eval_gguf.py` runs `tools/eval/eval_refine.py` on `synth-v1/eval.jsonl` without an LLM judge.
It writes `tools/eval/out/refine-<system>.jsonl` for the blind judge and prints the deterministic
table next to Quill 2B. `--quick-judge` adds gpt-4.1-mini.

## Eval

Held-out set (human + Parakeet), scored on:

- **Faithfulness**: no added content, no meaning change, no answers (an LLM judge plus the
  rule checks in `refine/guard.py`; the guard reject rate is a metric in its own right).
- **Cleanup quality**: WER against a human-cleaned reference, plus per-tag accuracy
  (self-correction applied, fillers removed, list formatted).
- **Latency**: `bench_refine.py` paragraph median on CPU (< 1 s) and GPU.

Replace Quill as the default only if the fine-tune wins on faithfulness and does not lose on
quality or latency.

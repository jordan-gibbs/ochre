# Refinement fine-tune v2 (`v2-real`): results

2026-10-04. Qwen3.5-2B + LoRA (r16/α32, all 24 layers), 2 epochs on **2,112 audited real-audio
pairs** (`training/refine/datasets/real-v2/final/train.jsonl`). The pipeline that made them:
TTS of 4,000 scripts → Parakeet → `strip_hesitations`, then two independent auditors (Opus,
Sonnet) per row and an Opus adjudicator for disagreements. Export: Q4_K_M GGUF (1.2 GB, the
same size as Quill 2B), served by the app's llama-server. Train + export + eval wall time:
**9 min 40 s** on the RTX 5070 Ti.

## Blind judgement (Opus judges; model names hidden, both models shuffled together)

| test set | Quill 2B | **v2-real** | only v2 passes | only Quill passes |
|---|---:|---:|---:|---:|
| synth-v1 eval (100, different generators) | 59% | **76%** | 28 | 11 |
| real-v2 held-out (150, real Parakeet output) | 51% | **75%** | 47 | 11 |

| held-out details | Quill 2B | v2-real |
|---|---:|---:|
| acted on / answered the text | 0 | 0 |
| added content | 3 | 3 |
| dropped content | 17 | 10 |
| self-corrections applied | 7/12 | 10/14 |

On the synth-v1 eval: acted 4 → 1, added 7 → 3, dropped 13 → 7, self-corrections 15/21 → 19/22.

## Deterministic metrics

| | Quill 2B | v2-real |
|---|---:|---:|
| WER vs target, synth-v1 eval | 0.097 | 0.056 |
| WER vs target, held-out | 0.086 | 0.031 |
| guard rejections, synth-v1 eval | 5% | 1% |
| latency median, GPU | 117 ms | 79 ms |

Exact match on synth-v1 rose only 42% → 45%, because those older references keep numbers as
spoken words and v2-real writes digits (GUIDE rule 7, decided 2026-10-04).

## Remaining failure modes (from the judges' reasons)

- Self-corrections where the cue was misrecognized ("fifteenth, **weight**, the twentieth").
- Voice commands at the very end ("Best regards new line Alex").
- Trailing-off queries ending in "." instead of "...".
- Residual "like" in casual speech; occasional dropped "and".
- Recognition garbage the model passes through ("rejects" for "regex", a stray "Thumb.").

Next data round: target these, add real human recordings for evaluation, and consider a 0.8B
distillation for CPU-only machines.

## Reproduce

```powershell
cd training\refine
.\.venv-train-refine\Scripts\python train.py configs\qwen35-2b-lora.yaml --set run_name=v2-real --set "train_files=[datasets/real-v2/final/train.jsonl]"
.\.venv-train-refine\Scripts\python export.py output\v2-real\adapter
cd ..\..
.\.venv\Scripts\python training\refine\eval_gguf.py training\refine\output\v2-real\gguf\v2-real-Q4_K_M.gguf
.\.venv\Scripts\python training\refine\eval_gguf.py training\refine\output\v2-real\gguf\v2-real-Q4_K_M.gguf --name v2-real-heldout --eval training\refine\datasets\real-v2\final\heldout.jsonl
```

Blind judging: `tools/eval/out/judge-v1/` (chunks, `KEY-v3.json`, `verdicts-v3.json`).

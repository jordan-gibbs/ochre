# Refinement fine-tune v4: results

2026-10-05. Round 4 of the cleanup model. Rendering, training and evaluation all ran on Daytona
RTX 5090 spot sandboxes, and nothing heavy ran locally. The step-by-step record is in
`docs/refine-round4-log.md`, and the failure taxonomy that drove the data is in
`docs/refine-round4-failures.md`.

## What changed

- **Measurement first.** `eval-v4` (`training/refine/datasets/eval-v4/`) has 551 held-out
  dictations in 13 categories. Separate eval writers wrote them, they were rendered and
  recognized in the cloud, and two auditors plus an adjudicator checked them. The round-3 sets
  (150 + 60 + 12) are kept, so 773 rows are judged in total.
- **Judging protocol v4** (`tools/eval/JUDGE_V4.md`):
  - 3 independent blind Opus judges per row, scored by majority vote.
  - Identical outputs are merged under one letter.
  - Letters are shuffled per row and per judge.
  - Reported: pass / harmful / acted rates, per-category results, Fleiss kappa, and paired
    bootstrap 95% CIs.
- **GUIDE rule 13 (recoverability):** a misrecognized word is fixed only when the intended word
  can be inferred from the transcript. Otherwise the raw word is kept. Inventing a different
  word is a failure.
- **Targeted real-audio data:** 5,210 round-4 scripts aimed at the round-3 failures
  (self-corrections, filler vs. meaningful words, misrecognition, numbers, long dictations,
  voice commands, trailing off, AI prompts, terminal). They were rendered with TTS and
  augmentation, transcribed by Parakeet, and audited by two auditors plus an adjudicator
  (rubric v6). 3,661 were kept. `real-v4/train-real.jsonl` holds 6,143 pairs: round 3 plus
  round 4, with near-duplicates of any eval row removed.
- **Distillation set (2B / 0.8B):**
  - Simulator: `tools/refine-data/asr_noise_sim.py` is an ASR-noise simulator fitted to
    Parakeet's errors. Out of sample (W12–W51, 3,800 clips) its corpus WER is 0.0908 against
    0.0925 for the real recognizer (`docs/asr-noise-sim.md`).
  - Scripts: 4,000 from Claude writers, 18,960 from Codex (`codex exec`, gpt-6-astra), and the
    W12–W64 scripts again. They went through the simulator, and a sample was kept only when its
    target stayed recoverable. That gave 24,193 pairs.
  - Audit: a stratified 10% sample was checked by two auditors (agreement 96.5%) with an Opus
    adjudicator. 93.4% were accepted, 5.8% fixed and 0.8% dropped. Codex and Claude writers
    came out equal.
  - Final set: 23,991 pairs (`real-v4/sim-audited.jsonl`). v3-4b agrees word for word with
    78% of the targets.
- **DPO (2B test):** 467 preference pairs. Chosen is the audited target; rejected is a clear
  v3-2b failure on a training row. Two auditors checked a sample, and 92% of the filtered sample
  pairs were "worse" by both. Implemented as an optional stage in `train.py`
  (`dpo_files`, `dpo_*`).

## Runs

All runs use LoRA r16/α32 with lr 2e-4 (cosine) and effective batch 16, the v3 recipe.

| run | base | data | epochs |
|---|---|---|---|
| v4-4b | Qwen3.5-4B | real (6,143) | 2 |
| v4-2b-real | Qwen3.5-2B | real | 2 |
| v4-2b-real-dpo | Qwen3.5-2B | real, then DPO (β 0.1, lr 2e-5, 2 ep, +0.2 NLL) | 2 + 2 |
| **v4-2b** | Qwen3.5-2B | real ×2 + distillation (23,991) | 1 |
| v4-0.8b-real | Qwen3.5-0.8B | real | 3 |
| **v4-0.8b** | Qwen3.5-0.8B | real ×2 + distillation | 1.5 |

## Results (protocol v4, 773 rows × 9 models, 3 judges)

Inter-judge agreement over 2,873 distinct outputs: Fleiss kappa 0.94 for pass, 0.93 for harmful.
Every pass was unanimous on 95.7% of outputs. No model ever answered or acted on a dictation.

| run | **all (773)** | eval-v4 (551) | held-out (150) | new (60) | casual (12) | harmful |
|---|---:|---:|---:|---:|---:|---:|
| v3-4b | 74.8% | 70.1% | 86.7% | 85.0% | 91.7% | 14.9% |
| **v4-4b** | **79.8%** | **75.7%** | **91.3%** | 86.7% | 91.7% | **9.8%** |
| v3-2b | 64.7% | 58.3% | 79.3% | 81.7% | 91.7% | 22.6% |
| v4-2b-real | 69.2% | 65.0% | 78.7% | 78.3% | 100% | 15.8% |
| v4-2b-real-dpo | 66.0% | 62.6% | 75.3% | 68.3% | 91.7% | 12.5% |
| **v4-2b** | **71.0%** | **67.2%** | 78.0% | **85.0%** | 91.7% | **14.0%** |
| v3-0.8b | 55.6% | 48.8% | 74.0% | 68.3% | 75.0% | 25.1% |
| v4-0.8b-real | 62.1% | 59.2% | 72.0% | 58.3% | 91.7% | 17.9% |
| **v4-0.8b** | **63.8%** | **59.2%** | **74.7%** | **73.3%** | 91.7% | **17.3%** |

Paired comparisons on the same 773 rows. "Only A / only B" counts the rows where exactly one of
the two passes. The CIs come from a 10,000-sample paired bootstrap.

| comparison | only A | only B | pass diff (95% CI) | harmful diff (95% CI) |
|---|---:|---:|---:|---:|
| v4-4b vs v3-4b | 76 | 37 | **+5.0 [+2.3, +7.8]** | **−5.0 [−7.1, −2.8]** |
| v4-2b vs v3-2b | 101 | 52 | **+6.3 [+3.2, +9.4]** | **−8.7 [−11.6, −5.7]** |
| v4-2b-real vs v3-2b | 88 | 53 | +4.5 [+1.6, +7.5] | −6.9 [−9.8, −3.9] |
| v4-2b-real-dpo vs v3-2b | 92 | 82 | +1.3 [−2.1, +4.7] | −10.1 [−13.1, −7.1] |
| v4-0.8b vs v3-0.8b | 106 | 43 | **+8.2 [+5.2, +11.3]** | **−7.8 [−10.9, −4.8]** |
| v4-0.8b-real vs v3-0.8b | 93 | 43 | +6.5 [+3.6, +9.4] | −7.2 [−10.3, −4.3] |
| v4-2b (distill) vs v4-2b-real | 67 | 53 | +1.8 [−1.0, +4.7] | −1.8 [−4.3, +0.6] |
| v4-0.8b (distill) vs v4-0.8b-real | 65 | 52 | +1.7 [−1.0, +4.4] | −0.5 [−3.2, +2.2] |
| v4-2b-real-dpo vs v4-2b-real | 27 | 52 | −3.2 [−5.6, −1.0] | −3.2 [−5.2, −1.3] |

Per category, eval-v4 pass rate for v3 → v4 at each size:

| category | n | 4B | 2B | 0.8B |
|---|---:|---|---|---|
| self_correction | 64 | 70 → 81 | 55 → 73 | 38 → 70 |
| voice_command | 34 | 62 → 82 | 50 → 71 | 44 → 65 |
| trailing_off | 34 | 68 → 76 | 41 → 59 | 35 → 53 |
| long (multi-paragraph) | 42 | 43 → 60 | 14 → 19 | 5 → 10 |
| misrecognition | 37 | 62 → 70 | 59 → 59 | 46 → 51 |
| numbers | 51 | 82 → 76 | 67 → 76 | 65 → 65 |
| technical | 33 | 76 → 70 | 61 → 61 | 42 → 58 |
| terminal | 19 | 79 → 74 | 63 → 63 | 32 → 47 |
| filler_contrast | 61 | 77 → 82 | 72 → 82 | 59 → 72 |

## Promotion

The rule: a model is promoted only if its pass rate beats its v3 counterpart with a 95% CI that
excludes zero, and its harmful-failure rate does not go up.

- **v4-4b: promoted** (+5.0 [+2.3, +7.8], harmful −5.0).
- **v4-2b: promoted** (+6.3 [+3.2, +9.4], harmful −8.7). v4-2b-real also passes the rule. The
  distillation model is shipped because it has the higher point estimate and fewer harmful
  failures; the gap between the two is not significant.
- **v4-0.8b: promoted** (+8.2 [+5.2, +11.3], harmful −7.8).
- **DPO: not promoted.** It cut harmful failures (12.5% vs 15.8%) but lost 3.2 points of pass
  rate against the same SFT recipe. It became conservative: more `misrecognition_unfixed`, more
  punctuation, and fillers kept. It stays behind its config flag.

Shipped as `ochre-refine-{4b,2b,0.8b}-v4-Q4_K_M.gguf` in the private HF repos, with the v3
files kept. `crates/ochre-refine/src/local/install.rs` points at v4.

## Takeaways

- The targeted real-audio data does most of the work. At every size the real-only v4 run beats
  v3 significantly.
- Distillation adds a non-significant +1.7–1.8 points on top of the real-only runs for 2B and
  0.8B. It helps most on the round-3 legacy "new" set (2B 78 → 85, 0.8B 58 → 73), which suggests
  it improves generalization to unseen styles more than the round-4 categories.
- Self-corrections, voice commands and trailing off improved the most. Long multi-paragraph
  dictations improved at 4B (43 → 60%) but remain the weakest category for the small models.
  Misrecognized words (fixed when inferable, kept when not) are still the top failure type at
  every size.
- 4B dropped slightly on three round-4 categories: numbers 82% → 76% (3 of 51 rows),
  technical 76% → 70% (2 of 33) and terminal 79% → 74% (1 of 19). Each shift is within noise
  but worth watching.
- Caveat: the distillation runs hold out a 10% eval split of a file in which every real row
  appears twice, so their eval loss is optimistic. The judged held-out sets are unaffected.

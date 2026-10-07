# Chunked refinement, repeat collapse, re-dictation

2026-10-05. Three changes around the v4 cleanup models (`docs/refine-finetune-v4.md`). No new
data was generated. Every model run happened on Daytona, and judging used protocol v4
(`tools/eval/JUDGE_V4.md`).

**Results in brief**

- **Chunked refinement** (`crates/ochre-refine/src/chunk.rs`, `[refine] chunk_long`) is built and
  judged, and it does not help. 2B: +0.0 pts, 0.8B: -1.6, 4B: -21.0 [-32.3, -9.7] on the
  dictations it splits. `"auto"` (the default) therefore enables it for no model. `"on"` remains
  available for testing.
- **Repeat collapse** (`text::collapse_repeats`) removes a phrase restarted several times in a
  row before refinement. It fires on 9 of 34,728 eval and training transcripts, and none of
  those targets keeps the repeat.
- **Re-dictation** (`[refine] redictation_raw`, on by default): if you say the same words again
  within 60 s after Ochre changed them, the repeat is typed as heard. The earlier text is never
  replaced, for the reasons given below.

## 1. Chunked refinement for long input

### Threshold and piece size, from the round-4 verdicts

These are the judged pass rates by raw word count over all 773 rows of the final round-4
judging (`tools/eval/out/judge-v4/final/majority.json`):

| words | n | 4B | 2B | 0.8B |
|---|---:|---:|---:|---:|
| 0-19 | 452 | 83% | 79% | 73% |
| 20-39 | 134 | 84% | 72% | 69% |
| 40-59 | 106 | 78% | 64% | 49% |
| 60-79 | 13 | 85% | 62% | 62% |
| 80-99 | 16 | 38% | 44% | 31% |
| 100-129 | 11 | 27% | 27% | 18% |
| 130-159 | 34 | 62% | 26% | 9% |
| 160-199 | 7 | 71% | 0% | 14% |

The small models drop sharply above about 80 words and are at their best under about 40. The
defaults follow from that: dictations of 80 words or more are split (`chunk_min_words = 80`),
into pieces of about 40 words (`TARGET_WORDS`). A paragraph of 60 words or fewer stays whole.

### Boundary rules

A boundary is used only when nothing that must be read together can straddle it. All the rules
are in `chunk.rs`:

- **Corrections.** No cut next to a self-correction cue ("no wait", "actually", "sorry",
  "scratch that", "I mean", "or rather", "make that", ...). The cue may not be in the first four
  words after the cut or the last four words before it. A sentence with a strong cue anywhere in
  it cannot start a piece.
- **Short sentences.** Never split off a sentence under 5 words. Fragments like "No sorry."
  and "Some context." stay with their neighbours.
- **Mid-thought periods.** Never where the recognizer's period falls mid-thought:
  - either sentence starts lowercase;
  - the next sentence starts with a continuation word ("because", "which", "for", ...);
  - the previous sentence is a dependent clause ("If we don't hear back by noon.");
  - the previous sentence ends on a function word.
- **Lists** (GUIDE rule 8). A dictated list stays in one piece, from its lead-in sentence
  through the sentence after its last item cue. Item cues are "bullet", "number one", "step 2",
  "new line", "colon", or two or more sentences opening with "first / second / ... / finally".
- **"new paragraph".** A command that clearly stands alone (after a sentence end, or
  capitalized after a comma) is always a boundary. The command is removed and the pieces are
  joined with a blank line, as GUIDE rule 9 asks. "new line" is never a boundary.
- **Joining.** Pieces are joined with a single space (raw is single-spaced), or with `\n\n` at a
  paragraph command. Text below the threshold, or text with no usable boundary, is refined
  exactly as before: one request with the same prompt.

The app refines the pieces one after another, because llama-server runs with one slot (`-np 1`).
Each request reuses the cached system prompt (prompt cache plus the background primer), so a
piece only evaluates its own tokens: about 20 extra prompt tokens per extra piece for the chat
template. Each piece is guarded on its own (`guard::apply`), and a rejected or failed piece
falls back to its own raw text. If every piece fails, the whole raw text is used, as before.
Each piece gets the configured timeout.

The eval harness uses a byte-identical Python port, `src/openwhisprflow/refine/chunk.py`.
`tools/eval/chunk_parity.py` runs both on a table, 3,000 fuzz cases and every raw in eval-v4,
the real-v3 held-out sets and real-v4 train-real: 16,856 of 16,856 cases are identical, and
3,443 of them split.

### Evaluation

- **Run.** Daytona run `eval-1005-204618-24da` (RTX 5090 spot, 15.4 min) covered v4-4b, v4-2b
  and v4-0.8b, chunked (`eval_gguf.py --chunk`) and unchunked, on all four sets (773 rows),
  using the production prompt path and llama.cpp b11398. Command:
  `owf_daytona.py eval --models ... --eval eval4=... --eval heldout=... --eval new=... --eval casual=... --chunk-modes off on`.
- **Reproducibility.** The unchunked rerun matches the round-4 outputs byte for byte (773 of 773
  at every size). In chunk mode, every row the chunker leaves whole (711 of 773) gets the same
  text as unchunked, so only the 62 split rows can differ: 56 from eval-v4 (40 of the 42
  "long" rows) and 6 from held-out.
- **Judging.** The 62 split rows × 6 runs were judged blind by 3 Opus judges: 18 packets,
  5.4 distinct outputs per row. Judges graded `typed`, the text the app would insert, with the
  per-piece guard fallback included (`build_judge_v4.py` manifest `"field": "typed"`).
  Inter-judge Fleiss kappa was 0.876 for pass and 0.888 for harmful, and 93.7% of pass verdicts
  were unanimous.
- **Full sets.** `tools/eval/chunk_compare.py full` folds the split rows back into the full
  sets. A row the chunker leaves whole has a paired difference of exactly 0.

Paired differences, chunked minus unchunked, with 95% CIs from a 10,000-sample paired bootstrap:

| model | scope | n | pass, unchunked → chunked | only chunked / only unchunked passes | pass diff (95% CI) | harmful diff (95% CI) |
|---|---|---:|---|---:|---:|---:|
| 4B | split rows | 62 | 48.4% → 27.4% | 2 / 15 | **−21.0 [−32.3, −9.7]** | +9.7 [+0.0, +21.0] |
| 4B | long (42) | 42 | 24 → 15 rows | 2 / 11 | −21.4 [−38.1, −7.1] | +9.5 [−4.8, +23.8] |
| 4B | eval-v4 | 551 | 75.7% → 73.7% | 2 / 13 | −2.0 [−3.4, −0.7] | +1.1 [−0.2, +2.4] |
| 4B | all | 773 | 79.8% → 78.1% | 2 / 15 | −1.7 [−2.7, −0.6] | +0.8 [+0.0, +1.7] |
| 2B | split rows | 62 | 24.2% → 24.2% | 6 / 6 | +0.0 [−11.3, +11.3] | +1.6 [−11.3, +16.1] |
| 2B | long (42) | 42 | 8 → 8 rows | 4 / 4 | +0.0 [−11.9, +11.9] | −7.1 [−23.8, +9.5] |
| 2B | eval-v4 | 551 | 67.2% → 67.2% | 5 / 5 | +0.0 [−1.1, +1.1] | +0.0 [−1.5, +1.5] |
| 2B | all | 773 | 71.0% → 71.0% | 6 / 6 | +0.0 [−0.9, +0.9] | +0.1 [−1.0, +1.3] |
| 0.8B | split rows | 62 | 14.5% → 12.9% | 1 / 2 | −1.6 [−8.1, +3.2] | −4.8 [−14.5, +4.8] |
| 0.8B | long (42) | 42 | 4 → 4 rows | 1 / 1 | +0.0 [−7.1, +7.1] | +0.0 [−11.9, +14.3] |
| 0.8B | eval-v4 | 551 | 59.2% → 59.0% | 1 / 2 | −0.2 [−0.9, +0.4] | −0.2 [−1.3, +0.9] |
| 0.8B | all | 773 | 63.8% → 63.7% | 1 / 2 | −0.1 [−0.6, +0.3] | −0.4 [−1.2, +0.4] |

How the table was built:

- The "long" rows cover the 40 split long rows; the other 2 long rows are identical in both
  modes.
- Full-set unchunked rates are the round-4 results, which the rerun reproduces exactly.
  Chunked rates are those minus the paired difference.
- The new judging of the split rows agrees with round 4 on the long category: unchunked
  60% / 20% / 10%.

**Decision: `"auto"` is off at every size.** At 2B and 0.8B the gain is zero, and at 4B chunking
is significantly worse. `AUTO_MODELS` is empty, and `"auto"` (the default) behaves like
`"off"`.

**Why it doesn't help.** Failure types are much the same in both modes: misrecognitions,
punctuation and casing, missed corrections. A long row fails when any one of its ten or so
sentences has an error, and splitting does not lower the per-sentence error rate. The word-count
curve above mixes length with how dense and noisy these dictations are. Splitting also removes
context. At 4B, punctuation and casing failures rise from 7 to 11 and dropped content from 4 to
6. Most of these happen at a piece boundary, where the model can no longer merge sentences the
recognizer broke ("…rebuild the wheel a new hub. | which would be…"), or where an enumeration
runs across the cut.

### Latency (RTX 5090, 62 split rows, sequential pieces, 3.2 pieces on average)

| model | unchunked median / p95 | chunked median / p95 | extra, median / mean / p95 |
|---|---:|---:|---:|
| 4B | 652 / 739 ms | 970 / 1,391 ms | +336 / +365 / +713 ms |
| 2B | 354 / 405 ms | 472 / 614 ms | +110 / +128 / +259 ms |
| 0.8B | 278 / 318 ms | 370 / 592 ms | +95 / +117 / +289 ms |

Generated tokens are the same in both modes (for example 2B: 9,728 vs 9,872 over the 62 rows).
Evaluated prompt tokens rise 26% (11,271 to 14,180), from the chat template repeated per
piece; the system prompt stays cached. The extra time is per-request overhead: about 42 ms
(0.8B), 49 ms (2B) and 150 ms (4B) per extra piece on this GPU, more on a slower one. The
timing includes the re-prime between pieces, as the app's background primer does it.

## 2. Repeat collapse before refinement

`crates/ochre/src/text.rs` `collapse_repeats`. It runs right after `strip_hesitations` in the
app's pre-pass (`text::prepass`: hesitations, then repeats, then the dictionary), whether
refinement is on or off.

- **What counts as a repeat.** A phrase said several times in a row: 3 or more copies of at
  least 3 words, or exactly 2 copies of at least 5 words. The match ignores case and ASCII
  punctuation, and non-ASCII letters must match exactly. The phrases are collapsed into one
  copy.
- **What is kept.** The first copy is kept, so the sentence-start casing survives, with the last
  copy's final token, so its punctuation ends the thought. "The 2.8B long. The 2.8B long. The
  2.8B long." becomes "The 2.8B long."
- **What is never collapsed.** One- and two-word repeats ("very, very good", "no no no"); a
  phrase that is all one word or only numbers (a dictated code or count); any phrase holding a
  voice command ("new line", "new paragraph", "bullet", "comma", "period", "colon").
- **Eval check** (`training/refine/datasets`: eval-v4 eval + raw-all, real-v3 held-out / new /
  casual / train, real-v4 train-real + sim-audited, synth-v1 eval, real-v2; 34,728 distinct
  raws).
  - A first version (2 copies of 3 or more words) fired 79 times. In 17 of those the target
    keeps the repeat: deliberate repetition ("Wait for it, wait for it", "What a night, what a
    night", "The thing is, the thing is"), parallel sentences ("...the final round. The final
    round is a half day"), and a dictated command list ("make new line, make new line, make
    install").
  - The shipped rules fire 9 times. All 9 are restarts ("Write a short apology to the write a
    short apology to the writer", "She said the invoice was she said the invoice was already
    paid"), and in every one the target drops the repeat. None conflicts with a target.
- **Python mirror.** The pass is mirrored in `tools/refine-data/owf_text.py`
  (`collapse_repeats`, and `app_prepass` now matches `text::prepass`). `parity.py` compares
  strip, collapse and the full pre-pass against `examples/text_parity.rs`: 15,770 of 15,770
  cases are identical, including fuzz cases with injected repeats and 10,703 real recognizer
  outputs. Training data built from now on gets the same raw the app produces. Existing
  datasets were not rebuilt, and of their raws only the 9 above would change.

## 3. Re-dictation means the fix was wrong

**Rule** (`crates/ochre/src/app.rs`, `redictated`). A dictation counts as a re-dictation when:

- the previous dictation ended less than `redictation_window_s` ago (default 60 s);
- the previous dictation's typed text differed from its raw text;
- its raw words match the new dictation's raw words (`text::same_words_key`: lowercased, with
  punctuation and spacing ignored, compared after hesitations, repeats and dictionary).

**What happens.** The new dictation is typed as heard: hesitations, repeat collapse and
dictionary are applied, but not the refine model.

- A notice says "Said again, so typed as heard".
- The history row is saved with `refiner = ""` and the new column `note = "redictation"`. The
  column is added to older history files on open. Settings › History shows the row as "Said
  again: as heard".
- Because the raw insert changed nothing, a third take is refined again.
- The rule does not apply to raw-gesture dictations or when refinement is off.
- Setting: `[refine] redictation_raw = true` and `redictation_window_s = 60.0`. It appears in
  Settings › Refinement › "Said again, typed as heard".

**Replacing the previous output: not implemented.** It cannot be made provably safe with what
Ochre knows:

- **No read-back.** The injector only writes, as synthesized keystrokes or a clipboard paste. No
  cross-platform API here can read the field back to confirm it still ends with Ochre's text.
  UI Automation and AX text patterns exist in only some apps, and Linux has nothing.
- **Typed text is not field text.** Apps autocorrect, auto-pair brackets and quotes, turn
  quotes curly, and accept completions on space or Enter (Slack mentions, emoji, IDE
  completion). N backspaces would delete the wrong characters.
- **Newlines are Enter key presses** (`ochre-platform` `text::plan`). In chat apps a
  multi-paragraph dictation can already have been sent, and the backspaces would then land in
  whatever the user has started typing next.
- **Focus is the top-level window**, not the control or caret (`FocusInfo.window_id`), so
  "same window" does not prove "same field, caret unmoved".
- **"No other input since" cannot be proved.**
  - On Windows the keyboard hook could see keys, and the mouse hook only stamps a time.
  - The macOS and Linux listeners see only the hotkey.
  - Edits that are not input events (a collaborator in a shared document, app-side
    reformatting, autosave) are invisible on every platform.
- **Undo is not uniform.** Ctrl+Z can undo more or less than Ochre's insertion, depending on
  the app.

Deleting user text is the one outcome ruled out, so only the raw insert is
implemented. `join_window_s` / `JoinMemory` (same window id within 20 s) is enough to decide
whether to add a leading space, which is harmless when wrong. It is not enough to justify a
deletion.

## Reproduce

```text
python tools/eval/chunk_parity.py --from training/refine/datasets/eval-v4/eval.jsonl ...
.venv-data/Scripts/python tools/refine-data/parity.py --from training/refine/datasets/real-v2
owf_daytona.py eval --models v4-4b=hf:... v4-2b=hf:... v4-0.8b=hf:... --eval eval4=datasets/eval-v4/eval.jsonl \
    --eval heldout=... --eval new=... --eval casual=... --chunk-modes off on --yes
python tools/eval/chunk_compare.py prepare tools/eval/out/eval-runs/<run> tools/eval/out/judge-v4/chunking
python tools/eval/build_judge_v4.py --manifest tools/eval/out/judge-v4/chunking/manifest.json   # then 3 judges
python tools/eval/tally_judge_v4.py tools/eval/out/judge-v4/chunking --pairs v4-4b-chunk:v4-4b v4-2b-chunk:v4-2b v4-0.8b-chunk:v4-0.8b
python tools/eval/chunk_compare.py full tools/eval/out/judge-v4/chunking
```

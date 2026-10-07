# ASR-noise simulator (round 4, Phase D)

`tools/refine-data/asr_noise_sim.py` turns a spoken script (`spoken`: lowercase, `,` pauses,
`-` cut-offs, `...`, um/uh) into a simulated Parakeet transcript (`raw_asr_sim`). It then runs the
app pre-pass (`owf_text.app_prepass`, the byte-identical port of the Rust stripper) to get `raw`.
It is pure Python with no models, and the model is a 500 KB table (`asr_sim/model.json`). Fitting
takes about 2 minutes on a laptop; simulating takes about 2 ms per row.

## Method

**Data.** The fit uses the real-v2 rows from W1–W11 and R2 recognized by Parakeet TDT 0.6B v3
(4,611 rows). The split is deterministic, by id hash: `sha256("asr-sim|"+id)[:8] % 100 < 15`
puts a row in the held-out set. That gives 3,895 fit rows and 716 held-out rows; the held-out ids
are in `asr_sim/heldout_ids.json`. Batches W12 and later landed while this was being built. They
are opt-in (`--all-batches`), so the model stays reproducible.

**Alignment.** `tts_text` tokens (with the marker after each word) are aligned to `raw_asr` tokens
by a weighted Levenshtein:
- A near-miss substitution costs `1 - 0.5·similarity`. Hesitation forms match each other.
- Merges ("stand up" → "standup") and splits are 2:1 / 1:2 moves.
- A spoken number span aligns many-to-one to a digit token when the values agree
  ("ten thirty" → "1030", "twelve percent" → "12%", "fifteen dollars" → "to$15").
- Symbol spans ("staging dot example dot com" → "staging.example.com") and spelled letters
  ("a w s" → "AWS") also align many-to-one.

**What is learned** (all counts, with additive backoff):

| component | what it models |
|---|---|
| word errors | Per-word substitution tables (word → {replacement: count}) and deletion counts. These back off to a word-class rate (function, short, common, rare, proper-noun-like from `intended`/dictionary, number word, repeat, cut-off fragment). When a word has no table, a novel substitution is drawn from the recognizer's output vocabulary by spelling similarity. |
| conventions | 259 words whose dominant "error" is a recognizer convention ("dont" → "don't", "its" → "it's", "ok" → "okay"). These apply at a fixed rate and are not scaled by noise. Also: repeat collapse ("to to" → "to", 21%), spelled-letter merge (44%), "dot" joins (23%), and learned merges/splits ("stand up"/"standup", "to do"/"todo", "sign up"/"signup"). |
| utterance noise | The noise multiplier `m` is observed errors ÷ expected errors per clip, kept as the full empirical distribution in 6 length buckets. The share of clips with `m = 0` falls from 65% (≤ 6 words) to 11% (> 60 words), and short clips have a heavier tail. Every scaled error probability is multiplied by `m`. |
| truncation | Leading or trailing loss of ≥ 2 words (1–5% of clips by length, sized as a fraction of the clip). Mid-clip deletion bursts of ≥ 4 words. An end-of-clip hallucinated word in 4.5% of clips (mostly "Yeah."). |
| punctuation | After each output token: P(none / comma / stop) given the spoken marker after the word (none / pause / trail / cut / hesitation), the words since the last break (7 buckets), the next spoken word, and the current word. The levels back off fine to coarse. A stop becomes "?" or "." based on the first question word in the sentence's first three words and its position ("can@0" vs "can@1"). There is a separate end-of-utterance table, and no break inside a spoken number. |
| casing | First token capitalized (97%), capital after a stop (94%), per-word casing ("I", days, languages, names, "API", "PM"), and spurious mid-sentence capitals by context. After a spoken pause with no punctuation the next word is capitalized 27% of the time (a phrase-cut artifact), otherwise 1–3%. |
| numbers | Number spans are typed small / mid / large / million / ordinal / lone ordinal / multi-group / decimal / pct / money. P(digits) by type ranges from 1.5% (lone "first") and 3% (< 10) to 20% (≥ 100, times) and 36% (decimals). Formats: "12%", "$15" glued to the previous word ("under$20"), "10,000" (4-digit numbers get a comma 1/3 of the time), "1030" (never "10:30" in the data), "19th", "1.8.0". |
| hesitations | Only used when they are spoken (`--keep-hesitations`; the first ~4,000 renders spoke them). Outcomes: 62% "Um"/"Uh"/"UMM", which the pre-pass then strips (or not, for "UM"); 20% dropped; 18% misheard as a word ("no", "oh", "or", "some", "thumb", "zum"). |

**Apply.** Hesitations are removed the way the TTS step does it (`common.strip_spoken_hesitations`).
The tokens are parsed with their markers and number spans are found. Then `m` is drawn, followed by
the edge, collapse and letter events. Each word is kept, substituted, deleted, merged or split, and
insertions are added. Finally punctuation and casing are applied, the tokens are joined, and the
app pre-pass runs. Output is deterministic for a given seed.

## Validation (held-out 716 rows × 5 seeds, `asr_sim/validation.json`)

Reference = `spoken` minus hesitations, normalized. WER is measured on `raw` (after the pre-pass).
Number renderings and contractions count as errors on both sides.

| statistic | real | sim | rel |
|---|---:|---:|---:|
| WER mean (per utterance) | 0.137 | 0.125 | −9% |
| WER median | 0.086 | 0.071 | −17% |
| WER q75 / q90 | 0.167 / 0.313 | 0.154 / 0.286 | −8% / −9% |
| corpus WER | 0.102 | 0.096 | −6% |
| share of rows with WER = 0 | 0.200 | 0.259 | +30% |
| substitution / deletion / insertion rate | 0.053 / 0.032 / 0.017 | 0.049 / 0.032 / 0.016 | −9% / −1% / −6% |
| commas per 100 words | 2.88 | 2.78 | −4% |
| periods per 100 words | 10.18 | 10.22 | +0% |
| "?" per 100 words | 0.42 | 0.47 | +11% |
| sentence breaks per 100 words (non-final) | 8.16 | 8.26 | +1% |
| mid-clause breaks per 100 words | 0.83 | 0.83 | −0% |
| mid-sentence capitalization rate | 0.043 | 0.046 | +8% |
| number spans rendered as digits | 0.102 | 0.114 | +12% |

Almost every statistic is within 15%. The two outside it are the WER = 0 share (+30%) and the WER
median (−17%). Most of that gap is split variance rather than model bias:
- The real held-out rows are noisier than the fit rows: 20.0% of held-out rows have WER = 0,
  against 23.7% of fit rows.
- On an in-sample check (710 fit rows, also in `validation.json`), every main statistic is within
  14%, with WER = 0 at 23.7% real vs 24.9% sim and mean WER at −3%.
- The remainder is a small generalization gap: per-word tables learned on 3.9k rows are slightly
  optimistic for unseen text.

The subset of held-out rows whose TTS did not speak hesitations (250 rows, mostly W9–W11 and R2)
fits worse: sub −16%, del +52%, commas −18%, "?" −16%. It is small and high-variance; a single
whole-clip loss in one seed moves the deletion rate a lot. Those batches also have more commas in
real Parakeet output than the spoken markers predict, probably because of TTS prosody (voice and
pause scale), which the simulator does not see.

## What it does not capture

- **Acoustics.** There is no voice, room, noise or speaking rate. The noise level is drawn
  independently of the text, so the simulator cannot tell that a given voice mangles names.
- **Phonetic confusions for unseen words** are approximated by spelling similarity
  ("auntie" → "auto"), not sound ("castillo" → "Loy"). Proper nouns and jargon get class-level
  rates (18% substitution for proper-noun-like words).
- **Context-dependent rewrites** (homophones by sense such as "role"/"roll", "no wait" → "no weight")
  appear only when they are in a word's learned table.
- **Long-range garbling and hallucinated phrases.** Contiguous whole-phrase rewrites ("Did you
  finish slides?" for "to do finish slides") come out as independent word errors.
- **Spoken-symbol and number rendering for rare forms** (IPs, versions, "a.m./p.m.") is rough.
  Numbers inside symbol spans stay as words ("ten dot zero.five").
- **Phrase cutting.** The real segmenter (2 s / 200 ms rules) is not simulated. Its effects
  (breaks and capitals at pauses) are absorbed into the punctuation and casing tables.

## Usage

```
python tools/refine-data/asr_noise_sim.py fit                   # -> asr_sim/model.json, heldout_ids.json
python tools/refine-data/asr_noise_sim.py validate              # -> asr_sim/validation.json (prints table)
python tools/refine-data/asr_noise_sim.py make-validation-set   # -> datasets/asr-sim/heldout-pairs.jsonl
python tools/refine-data/asr_noise_sim.py apply --in scripts.jsonl --out rows.jsonl [--seed N] [--keep-hesitations]
```

`apply` reads script rows `{id, spoken, intended, dictionary, context, style, tags, ...}`. It
writes the same rows plus `raw` (simulated, after the pre-pass), `raw_asr_sim`, `clean` = `intended`,
`source` = `"sim-v4"` and `sim_seed`. The per-row seed is `sha256("sim-v4|id|seed")`, so different
`--seed` values give independent noisy copies of a script. Pass `intended` and `dictionary`: they
mark proper nouns, which get higher error rates and their written casing.

From Python: `from asr_noise_sim import apply; apply(spoken, seed, dictionary=[...], intended=...)`
returns `raw`. `load_sim().raw(...)` returns `(raw_asr_sim, raw)`.

**Validation set for the cloud check.** `training/refine/datasets/asr-sim/heldout-pairs.jsonl`
holds 400 held-out rows, chosen as rows with a claude or adjudicator audit and no "drop" in either.
Each row appears twice in the eval format: `simval-real-<n>` with the real `raw`, and
`simval-sim-<n>` with a simulated `raw` from the same `tts_text`. Both carry the row's audited
`clean`. The simulation speaks hesitations when that row's TTS did (218 of 400), so the two sides
match the same pipeline. A simulated clip that came out empty is re-drawn, because the audit
already removed unusable real clips. `asr_sim/heldout_pairs_map.json` maps `<n>` to the source
row id, `tts_text`, and the real and simulated `raw_asr`.

## Out-of-sample check on W12-W51 (`--fresh`)

The first downstream check (v3-4b on `heldout-pairs.jsonl`) was confounded: its W1-W11 rows were
in v3's training set, so the real side scored far better (word error vs target 0.013 real vs
0.102 sim). The check was redone on round-4 batches W12-W51, which neither the simulator fit nor
any v3 model saw:

```
python tools/refine-data/asr_noise_sim.py validate --fresh             # -> asr_sim/validation-fresh.json
python tools/refine-data/asr_noise_sim.py make-validation-set --fresh  # -> datasets/asr-sim/fresh-pairs.jsonl (400 pairs)
python tools/refine-data/asr_sim/fresh_check.py                        # after the cloud eval
```

**Recognizer statistics (3,800 real rows vs 5 simulated copies each).** Corpus WER 0.0925 real vs
0.0908 sim (-2%), mean WER 0.124 vs 0.117 (-6%), q90 0.318 vs 0.308. The simulator gives a few
more perfect clips (share with WER 0: 0.31 vs 0.39) and a little more deletion and less
substitution than real Parakeet (del +28%, sub -13%, ins -20%). Punctuation densities are within
15% (commas -15%, periods -3%, sentence breaks -5%). The round-4 TTS did not speak hesitations,
so the no-hesitation subset is the whole set.

**Downstream (v3-4b, cloud run `eval-1005-155709-f02f`).** Against the audited target, which is
recoverable from the *real* raw only, v3-4b's word error is 0.043 on real raw and 0.123 on
simulated raw. Most of that gap is simulated lexical errors that no model can undo. When only the
227 of 400 simulated samples that `distill_v4.patch_target` accepts are kept (target recoverable
from the simulated raw, the same filter the distillation set uses), the result is word error 0.038
real vs 0.056 sim, and word-level exact match 0.73 vs 0.60. Filtered simulated rows are a little
harder than real ones, not easier, so they will not teach the model to skip real fixes.

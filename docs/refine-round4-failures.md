# Round-4 failure taxonomy (Phase B)

Sources:

- **eval-v4 baseline** (this round): v3-4b / v3-2b / v3-0.8b on 752 rows (530 new eval-v4 rows +
  the 222 round-3 rows), blind side-by-side, one Opus judge per row (protocol v4 packets,
  `tools/eval/out/judge-v4/baseline/`). Every failing output carries one primary `fail_type`.
  2,256 graded outputs, 879 failures.
- **judge-v3** (round 3): 222 rows x 6 models, failure reasons as free text
  (`tools/eval/out/failures-v4/judge-v3-failures.jsonl`, 153 failures of the three v3 models).
  Its rule tags agree with the picture below: R5 filler (63), R6 punctuation (68), dropped
  content (43, all harmful), self-correction (40 + 7 misapplied, nearly all harmful), reword /
  changed word (~70 together, all harmful), numbers (20), false starts (20), trailing off (10).

Baseline pass rates (1 judge, for mining only; the reported comparison uses 3 judges):

| | all 752 | eval-v4 530 | round-3 held-out 150 | new 60 | casual 12 |
|---|---:|---:|---:|---:|---:|
| v3-4b | 74.7% | 69.4% | 86.7% | 88.3% | 91.7% |
| v3-2b | 64.6% | 58.5% | 78.0% | 80.0% | 91.7% |
| v3-0.8b | 55.7% | 49.1% | 72.7% | 68.3% | 75.0% |

eval-v4 is harder than the round-3 sets: fresh writers, topics and names the models never
saw, and harsher recognizer output. Weakest eval-v4 categories (4b / 2b / 0.8b pass): long
dictations 24 / 10 / 5% (n = 21), trailing off 59 / 38 / 32%, self-corrections 72 / 52 / 38%,
misrecognition-heavy 62 / 60 / 46%, voice commands 65 / 50 / 44%, casual 65 / 61 / 57%.

## Taxonomy, ranked by harm x frequency

Count = majority-failed outputs over the three v3 models (eval-v4 baseline); harmful = the judge
marked it harmful (changes what the text says or would embarrass the user).

| rank | failure type | count | harmful | 4b / 2b / 0.8b | where |
|---:|---|---:|---:|---|---|
| 1 | **misrecognition invented** (a wrong or odd word replaced by a guess, synonym or "more sensible" word) | 171 | 171 | 43 / 65 / 63 | self-corrections, trailing off, misrecognition, work |
| 2 | **content dropped** (a clause, value, unit or word silently removed) | 98 | 92 | 23 / 29 / 46 | casual, held-out, self-corrections, long |
| 3 | **reword** (paraphrase, changed person/number/tense) | 65 | 57 | 11 / 30 / 24 | filler contrast, work |
| 4 | **missed self-correction** (both versions kept, cue left in) | 56 | 49 | 11 / 20 / 25 | self-corrections, **long** |
| 5 | **misrecognition unfixed** (an inferable slip left in: muscles/mussels, fog/fob, a dictionary name) | 171 | 13 | 40 / 45 / 86 | misrecognition, self-corrections, casual |
| 6 | number format (wrong value or form) | 34 | 27 | 4 / 16 / 14 | numbers |
| 7 | content added | 29 | 29 | 9 / 13 / 7 | trailing off, AI prompts |
| 8 | punctuation / casing | 52 | 9 | 10 / 16 / 26 | casual, filler contrast |
| 9 | meaningful word dropped as filler | 13 | 13 | 8 / 2 / 3 | casual |
| 10 | filler kept | 31 | 0 | 8 / 7 / 16 | casual |
| 11 | voice command missed | 25 | 2 | 8 / 9 / 8 | voice commands |
| 12 | misapplied correction | 10 | 9 | 4 / 4 / 2 | |
| 13 | trailing off (`...` missing or invented ending) | 13 | 5 | 6 / 4 / 3 | trailing off |
| | already-clean changed, list structure, terminal format, other | 21 | 2 | | |

Examples (raw -> v3 output; judge's reason):

- invented: "Dar dash CZF backup slash hype dash records..." -> `tar -czf backup/archive-records.tar.gz`
  ("hype" replaced by an invented "archive"); "The omnisci cabinet..." -> "The oncall cabinet".
- dropped: "Explain what this rejects does, and then Thumb. Simplify it" -> "Explain what this does,
  and then simplify it" (deletes "rejects" instead of fixing it to "regex"); a height limit of
  "13 feet six inches" -> "13 feet".
- reword: "In anyone give me a quick rundown..." -> "Can you give me..."; "Ours quit" -> "Our quit".
- missed correction: "...down from about 500 last week. No, the week before." kept "last week";
  "the rabies booster scratch, that the distemper booster" kept both.
- unfixed: "The muscles were amazing tonight" (a food truck: mussels); "get in with the fog" (fob).

## What drives the round-4 data

1. **Misrecognized words, both directions (ranks 1 and 5, 342 failures).** The model must fix a
   slip whose intended word is certain from context, and keep the raw word otherwise (GUIDE
   rule 13, added this round). Every real-audio row with a recognition error teaches this
   when its target is audited for recoverability, so the main lever is *more audited real-audio
   rows with recognition errors*: the misrecognition category (500 scripts), plus noisy audio
   across all categories. The distillation data must not teach mind-reading: the simulator's
   content-word errors are propagated into the target (the raw word is kept, rule 13).
2. **Content preservation (ranks 2, 3, 7).** Long and medium multi-clause dictations whose
   target keeps every clause: long_dictation (300 + the same 300 re-rendered with clean audio,
   `W52`), work and casual rows with several clauses, minimal-edit rows that must not be
   reworded.
3. **Self-corrections, especially inside long text and with garbled cues (ranks 4, 12):**
   600 scripts, 25% contrast rows where the cue words are content.
4. Trailing off (300), numbers (300), voice commands (300), filler contrast (500), AI prompts
   (200), casual (200), technical/terminal (200).

Wave 2 (after this analysis) adds ~1,200 scripts weighted to ranks 1-4: misrecognition in strong
context (near-homophones and slips that context resolves, next to names and rare words it does
not), self-corrections inside 60-150-word dictations, casual multi-clause messages, trailing off,
and numbers with units.

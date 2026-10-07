# synth-v1

500 synthetic dictation-cleanup pairs, generated 2026-10-04 by five Sonnet agents following
`../GUIDE.md`, one bucket each. Hesitations (um, uh, er, hmm…) are deliberately absent from
`raw`, because `owf::text::strip_hesitations` removes them before the model ever sees the text.

| file | bucket |
|---|---|
| `A.jsonl` | self-corrections and restarts, with meaningful-cue traps |
| `B.jsonl` | false starts, fragments, stutters, trailing off |
| `C.jsonl` | punctuation, casing, run-ons, proper nouns, dictionary spellings |
| `D.jsonl` | never answer: AI prompts, questions and commands; already-clean identity |
| `E.jsonl` | lists, voice commands, discourse fillers, context variety |

`train.jsonl` (400) and `eval.jsonl` (100, 20 per bucket) are a deterministic split (seed
20261004). Never train on `eval.jsonl`.

# Judging protocol v4 (blind, side by side, 3 independent judges)

You are one of three independent judges grading a dictation-cleanup model. Ochre turns speech
into text: the recognizer's output (`raw`, after um/uh were deleted deterministically) goes to a
small model whose job is to clean it per `training/refine/datasets/GUIDE.md`. You grade the
model's outputs.

Read the GUIDE (rules 1-13 and the recoverability rule) before grading. Grade each packet row
on its own.

## What you see per row

- `context`, `style`, `dictionary`: what the model was told (where the person is typing).
- `raw`: **the only text the model saw**.
- `expected`: an audited reference cleanup. It is **one** correct answer, not the only one.
- `candidates`: letters -> outputs from different models, shuffled per row and per judge, with
  identical outputs merged under one letter. You never learn which model wrote what.

## Pass / fail

An output **passes** when a careful user who dictated `raw` would accept it as typed, i.e. it
follows the GUIDE with no real error. Concretely it passes when:

- the meaning and every content word of the intended dictation are kept (GUIDE removals aside);
- self-corrections are resolved the way the speaker meant; meaningless fillers, stutters,
  false starts and hesitation residue are gone; meaningful uses are kept;
- numbers, times, money, emails, URLs follow rule 7 (equivalent careful forms are fine:
  "March 13" vs "March 13th", "3 PM" vs "3 pm");
- sentence breaks, `?`, casing, apostrophes and names are right;
- voice commands / dictated lists / trailing-off `...` are handled (rules 4, 8, 9);
- questions and requests are kept as such, never answered or acted on (rule 1);
- misrecognized words are fixed only when the intended word is inferable; otherwise the raw word
  is kept (rule 13). Inventing a different word is a failure; keeping an unfixable raw word is
  not.

Do **not** fail an output for: optional commas, a comma vs a period between two short clauses,
an exclamation mark vs a period, colon vs period before a request, "OK" vs "okay", a different
but equally correct written number form, or for not matching `expected` word for word. If
`expected` itself looks wrong, grade against the GUIDE, not against `expected`.

## Harmful

`harmful = true` when the failure changes what the text says or would embarrass the user:
dropped or added content, a changed word or value (number, name, day), a reworded sentence,
a misapplied or missed self-correction that leaves the wrong value, an answered/acted request,
an invented fix of a misrecognized word, a meaningful word removed as filler. Cosmetic misses
(a filler left in, a missing `?`, casing, a missed `...`, list formatting) are failures but not
harmful.

## Failure type (one primary per failing output)

`missed_correction`, `misapplied_correction`, `filler_kept`, `meaningful_dropped`,
`content_dropped`, `content_added`, `reword`, `misrecognition_invented`,
`misrecognition_unfixed` (an obvious, inferable slip left in), `number_format`,
`punctuation_casing`, `voice_command`, `trailing_off`, `list_structure`, `acted_answered`,
`already_clean_changed`, `terminal_format`, `other`.

## Output

Write one JSON object per packet row, in packet order, as JSON Lines, to the path you were given:

```json
{"key": "<row key>", "grades": [{"letter": "A", "pass": true, "harmful": false, "acted": false, "fail_type": "", "reason": ""},
                                {"letter": "B", "pass": false, "harmful": true, "acted": false, "fail_type": "content_dropped", "reason": "drops 'before Friday'"}]}
```

Grade every letter of every row. `reason` is one short line (empty on pass). Be consistent: two
outputs with the same error get the same verdict.

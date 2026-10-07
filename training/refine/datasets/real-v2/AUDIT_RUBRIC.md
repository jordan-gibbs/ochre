# real-v2 audit rubric (v6: fix limits scale with length 2026-10-05; R13 added; R5 revised)

You are auditing training pairs for Open Whisperflow's dictation-cleanup model. Each row is one
dictation: a script of what a person **said** (`spoken`) was spoken by a TTS voice, passed through
room/noise/mic simulation, transcribed by the app's real speech recognizer, and run through the
app's deterministic hesitation stripper. The result is `raw`: **the only thing the model will
see**. `clean` is the target the model will be trained to produce from `raw`.

Your job, per row: decide whether `clean` is the right target **for this `raw`**, and output
`accept`, `fix_clean` (with the corrected full text), or `drop` (with a reason).

Two auditors label every batch independently and their outputs are merged mechanically: rows
where you both chose the same decision (and, for `fix_clean`, the same text) are kept; everything
else goes to a human. So follow this procedure literally, prefer the most conservative
reading, and never fix things that are merely a matter of taste.

## 1. What is in a row

| field | meaning |
|---|---|
| `context`, `style`, `dictionary` | where the person is typing, style hint, preferred spellings (all passed to the model) |
| `spoken` | exactly what was said: lowercase, `-` = word cut off, `,` = pause, `...` = trailing off, um/uh/er/hmm included |
| `raw_asr` | recognizer output before hesitations were stripped (for your understanding only) |
| `raw` | **model input**: recognizer output after the app's pre-pass (um/uh/er/hmm/mm removed, dictionary spellings applied) |
| `clean` | current target (the script author's `intended`) |
| `raw_engine` | `parakeet-tdt-0.6b-v3-int8` (default engine) or `whisper-large-v3-turbo` |
| `voice`, `audio_conditions` | which TTS voice and acoustic conditions produced the audio (context only) |
| `tts_text`, `orig_id` | only on re-rendered rows (`batch-r2-*`, id ending `-r2`): `tts_text` is what the TTS voice actually read, i.e. `spoken` with um/uh/er/hmm removed (TTS voices turned hesitations into words such as "Number" or "her"; real speakers don't). Audit against `spoken` as usual: the hesitations the stripper would remove simply never reached the audio. |
| `audit_auto` | deterministic hints: `wer` of raw vs spoken, `missing` (target tokens that were said but are absent from raw: **recoverability suspects**), `not_in_spoken`, `flags`. Hints, not verdicts. |

## 2. The decision procedure (apply in this order; stop at the first `drop`)

### Step 1. Drop the row if any of these hold

| code | when |
|---|---|
| `garbage` | `raw` is empty, mostly unintelligible, or a repetition loop. |
| `non_english` | `raw` is wholly or partly in another language or script (Whisper sometimes transcribes accented English as another language). |
| `hallucination` | `raw` contains a plausible word, phrase or sentence the speaker never said and that is not a mishearing of anything in `spoken`: typically Whisper's "Thank you.", a lone trailing "you", "Thanks for watching!", "Bye.", a subtitle credit, or an invented sentence; also a short phrase like "Yeah." or "Okay." decoded from trailing noise when `spoken` has nothing there. (Training on it would teach the model either to delete real words or to keep noise.) |
| `hesitation_misheard_as_word` | where `spoken` has a hesitation (um/uh/er/hmm/ah), `raw` has a **real English word** that could be read as meant ("him", "air", "there", "allude", "on"), and leaving it out of `clean` would mean deleting a word a reader of `raw` would take as content. Not a drop if the residue is a non-word or a pure interjection ("Padam", "Zump", "Hm", "Mm-hmm", "Uh-huh" used as a filler, "Ah"); see step 2. |
| `unrecoverable_meaning` | the recognizer lost or garbled the message so badly that the best target inferable from `raw` is not a sensible dictation, or fixing `clean` would take rewriting more than about 5 words or more than 2 separate places **per 40 words of `raw`** (v6: a 120-word dictation may take up to 6 places / 15 words; a 30-word one still 2 / 5), or a self-correction can no longer be resolved from `raw` (e.g. the corrected value is missing). |
| `script_defect` | `spoken` and `intended` disagree in a way unrelated to recognition (the target says something never spoken), or the script is not a plausible dictation. |

### Step 2. The recoverability rule: make `clean` inferable from `raw`

The model sees only `raw` (plus context, style and dictionary). **Every word in `clean` must be
traceable to `raw`.** Go through each difference between `raw` and `spoken`:

**Recoverable: `clean` keeps the intended form (no change needed).**
- Hesitations, stutters, repeated words, cut-off fragments ("I th", "wh-"), false starts: present
  or absent in `raw`, `clean` drops them (GUIDE rules 2–4). A lone non-word / interjection blip
  where `spoken` had a hesitation ("Padam!", "Hm", "Ah") is also dropped from `clean`.
- Casing, punctuation and sentence breaks the recognizer got wrong (GUIDE rule 6), including
  phrase cuts that capitalize mid-sentence ("the The deck").
- Homophones when the meaning is unambiguous (there/their/they're, to/too/two, its/it's,
  your/you're, write/right).
- Split or merged words ("git hub" → GitHub, "every one" → everyone, "data base" → database).
- Dictionary terms and widely known proper nouns/products in any near spelling or split form
  ("cooper netties" is **not** near; "kuber netes" is) when context makes them unambiguous.
- Spelling variant changes introduced by the recognizer (Parakeet sometimes writes "summarise"
  for a US speaker): `clean` follows the speaker's script variant **only** if `raw` has it;
  otherwise use `raw`'s variant. (Rule 10: never change a spelling variant the input has.)
- Numbers, times, money, emails and URLs in a different but equivalent form ("25%" vs "twenty
  five percent", "3:30" vs "three thirty"): `clean` writes them per GUIDE rule 7, with the
  **value from `raw`**.
- A self-correction whose cue the recognizer dropped or changed ("Friday. No. Wait, Thursday",
  "Friday, Thursday") as long as both the retracted and the corrected value are still in `raw`
  in order: `clean` keeps only the correction.

**Not recoverable: `fix_clean` so `clean` follows `raw`.**
- A word misheard as a different real word, when the original cannot be inferred from `raw` and
  context: `clean` uses `raw`'s word ("Ask Korea about it" stays "Ask Korea about it", not
  "Priya"). If the result no longer reads as something a person would write, drop
  (`unrecoverable_meaning`).
- A name not in the dictionary: `clean` uses `raw`'s spelling ("Pria", "Shawn", "Catherine"),
  with proper casing. Exception: the dictionary has the name, or raw's version is a split/near
  form of a well-known product/brand (see above).
- A word the recognizer dropped: remove it from `clean` (do not re-insert it, even if the
  sentence becomes slightly less grammatical). If its loss changes or breaks the meaning, drop.
- A word the recognizer added (not a hallucinated phrase, e.g. an extra "the" or "and"): `clean`
  may drop it if it is clearly a recognition artefact of a stutter/repeat; if it is a plausible
  content word the speaker could have said, keep it in `clean` (fix_clean) or drop if it changes
  the meaning.
- A different number value ("15" for "fifty"): `clean` uses `raw`'s value.
- Trailing off: if the recognizer lost the last words, `clean` ends where `raw` ends (with `...`
  if `raw` visibly ends mid-thought, e.g. on "to", "the", "and"; otherwise as `raw` ends).

### Step 2b. Check the script itself: does `clean` match what was said?

The scripts were machine-written and their um/uh/er were partly machine-inserted. Compare
`clean` with `spoken` word by word, ignoring the disfluencies the GUIDE removes:
- Every content word of `clean` must have been said (in its spoken form) and every word that was
  said and not removed by a GUIDE rule (hesitation, stutter, false start, retracted
  self-correction, meaningless filler) must be in `clean`. A difference is a script defect:
  `fix_clean` to what was said (reason `guide_violation`), or `drop` (`script_defect`) if more
  than 5 words or 2 places are affected.
- An inserted hesitation that changed what was said: inside a number or name ("twenty um five",
  "pri uh ya"), between words that only make sense together, or turning a correction cue into
  something else. If `clean` assumed the uninterrupted reading and `raw` supports it, `accept`;
  if the hesitation makes the target's reading unrecoverable from `raw`, follow step 2 / step 1.
- **Structure the speaker did not dictate is a violation**: headings and labels such as
  "Title:", "Subject:", "Summary:", "Action items:" or a bolded/markdown heading, bullets or
  numbering, or line breaks, unless the speaker said them (e.g. "subject line", "new line",
  "bullet point", "first… second…"; see R8/R9). Remove the added structure with `fix_clean`
  (reason `guide_violation`) and keep the words that were actually said in prose; if a heading
  word was actually spoken ("title colon..."), keep it as words in the sentence, unless the
  speaker clearly dictated a title line.

### Step 3. Check `clean` against every GUIDE rule (training/refine/datasets/GUIDE.md)

Fix only clear violations, with the smallest edit:

| rule | `clean` must |
|---|---|
| R1 never answer / act | keep questions as questions and requests as requests, even in `ai_chat` ("Translate this to Spanish." stays that, untranslated) |
| R2 self-corrections | keep only the corrected version; drop the cue ("no wait", "sorry", "I mean", "actually" when corrective, "scratch that", "or rather", "make that") |
| R3 disfluencies | drop false starts, stutters, fragments, accidental repeats; keep deliberate repetition ("very, very good") |
| R4 trailing off | drop an abandoned fragment followed by a restart; a dictation that ends mid-thought keeps every word and ends with `...` (three ASCII periods) |
| R5 discourse fillers (revised 2026-10-05) | remove like / you know / I mean / basically / sort of / kind of / literally / so / okay so / right whenever meaningless, **in every context including casual** ("hey man so can we" -> "Hey man, can we"; "like ten minutes" -> "10 minutes"); keep when meaningful ("so that", "so" = therefore, "so good", "I like", "looks like", "what kind of", "turn right", "I mean it"); keep the casual register (slang, gonna, nah, lol) |
| R6 punctuation & casing | proper sentence breaks, commas where required, `?` on questions, apostrophes, `I`, capitalized names and products (dictionary spelling wins) |
| R7 numbers etc. | times, percentages, money, versions, measurements, emails, URLs, dates as a careful writer types them ("3:30", "25%", "$40", "john@gmail.com", "March 13th"); small counts in prose may stay words ("three people"); "one or two things" stays words |
| R8 lists / structure | only when dictated as a list ("first… second…", "bullet point", "number one…"): lead-in on its own line, then `- ` or `1. ` items one per line; never invent lists, headings ("Title:", "Subject:"), labels or line breaks the speaker did not dictate |
| R9 voice commands | "new line" → `\n`, "new paragraph" → `\n\n`, spoken "period"/"comma"/"question mark" used as commands → the mark |
| R10 faithful | no added words, facts, greetings or sign-offs; no summarizing; no change of tense, person, dialect, spelling variant or register; never censor; keep intentional slang/grammar |
| R11 already clean | if `raw` is already a perfect cleanup, `clean` must equal `raw` character for character |
| R12 terminal / literal | no final period on commands; flags, paths and casing as spoken |
| R13 misrecognized words | a slip whose intended word is inferable with confidence (homophone, split/near spelling of a known or dictionary term) is fixed; otherwise `clean` keeps `raw`'s word: never a different guess, a synonym, or a deletion that hides it (this is the recoverability rule of step 2 stated as a target rule) |
| filler rule | `clean` never contains um/uh/er/hmm/mm/mhm |

### Step 4. Decide

- `accept`: `clean` is **a** correct target for `raw` (steps 2 and 3 found nothing wrong).
- `fix_clean`: you changed `clean`. Output the **entire** new text.
- `drop`: step 1 matched, or the fix would exceed step 1's `unrecoverable_meaning` limit.

## 3. Rules that keep two auditors in agreement

1. **Wrong, not different.** Change `clean` only when it violates a rule above or the
   recoverability rule. If it is one acceptable answer, `accept`, even if you would have written
   it differently.
2. **Optional punctuation is never a fix**: a comma before "and"/"but"/"so", after a short
   introductory phrase, around "please" or "thanks"; a colon vs a period before a request;
   splitting vs joining two short clauses with a comma or period; an exclamation vs a period.
   Only fix: a missing `?` on a question, a missing sentence break in a run-on of two or more
   full sentences, a sentence break in the middle of a clause, wrong casing, a missing apostrophe.
3. **Minimal edits.** A `fix_clean` text differs from the old `clean` only where a rule required
   it. Keep every other character (including the author's optional commas) identical.
4. **Spelling of fixes is deterministic**: a word taken from `raw` keeps `raw`'s spelling;
   casing follows GUIDE rule 6; numbers follow rule 7 using the same style as the rest of `clean`.
5. **When in doubt between `fix_clean` and `drop`**: choose `drop` if the fix needs more than 5
   changed words or more than 2 places; otherwise `fix_clean`. (per 40 words of `raw`, see `unrecoverable_meaning`; long dictations are not dropped just because they contain several independent, recoverable fixes, as long as the fixed text still reads as a sensible dictation).
6. **When in doubt between `accept` and `fix_clean`** on a judgment call (is this filler
   meaningful? is this homophone unambiguous?): `accept`. The script author already decided.
7. Do not use `audit_auto` as a verdict. A `missing` token is often fine (an equivalent form the
   checker could not match); a clean `wer` of 0 can still hide a GUIDE violation.

Batches named `batch-rx-*` re-send rows whose `raw` changed after their first batch was written
(the app's hesitation stripper was fixed to also remove all-caps "UMM"/"HMM"). Audit them from
scratch; the decision replaces the earlier one for that id.

## 4. Output format

One JSON object per input row, same order, as JSON Lines:

```json
{"id": "s2-W1-0004", "decision": "fix_clean", "clean": "Can you summarise the thread from yesterday?", "reason": "spelling_from_raw", "rules": ["R10"], "note": "raw has 'summarise'; keep the input's variant"}
{"id": "s2-W1-0005", "decision": "accept", "reason": "ok"}
{"id": "s2-W1-0006", "decision": "drop", "reason": "hallucination", "note": "Whisper appended 'Thank you.'"}
```

- `decision`: `accept` | `fix_clean` | `drop`
- `clean`: required for `fix_clean` only: the full new target (use `\n` for line breaks)
- `reason`: accept → `ok`; fix_clean → `unrecoverable_word`, `dropped_by_recognizer`,
  `number_value`, `spelling_from_raw`, `guide_violation`, `trailing_off`; drop → the step 1 codes
- `rules`: GUIDE rules involved (`R1`–`R13`, `RECOVER`), optional for accept
- `note`: one short line, optional

## 5. Worked examples

**A. Accept: recognizer errors are all recoverable.**
```
spoken: um so can you send me the the deck by friday no wait thursday
raw:    So can you send me the the deck by Friday? No wait. Thursday.
clean:  Can you send me the deck by Thursday?
```
Hesitation stripped, stutter and correction are in `raw`, the meaningless "so" is an R5
removal, the bad sentence breaks are R6 fixes. → `accept`.

**B. Fix: a name misheard as another real word.**
```
spoken: ask priya to review the pr before lunch
raw:    Ask Korea to review the PR before lunch.
clean:  Ask Priya to review the PR before lunch.
```
"Priya" is not in the dictionary and cannot be inferred from "Korea". → `fix_clean`,
`clean`: "Ask Korea to review the PR before lunch.", reason `unrecoverable_word`.
(Same row with `dictionary: ["Priya"]` and raw "Ask Pria to review…" → `accept`: the dictionary
makes the spelling recoverable.)

**C. Fix: the recognizer dropped a word.**
```
spoken: the build is uh really slow on windows
raw:    The build is slow on Windows.
clean:  The build is really slow on Windows.
```
"really" is gone and cannot be inferred. → `fix_clean`: "The build is slow on Windows.",
reason `dropped_by_recognizer`.

**D. Drop: Whisper hallucination.**
```
spoken: er make this more formal please
raw:    And there. Make this more formal, please. Thank you.
clean:  Make this more formal, please.
```
"Thank you." was never said. → `drop`, reason `hallucination`.

**E. Accept: a hesitation residue that is not a word.**
```
spoken: um translate this paragraph into french no wait german
raw:    Padam! Translate this paragraph into French, no wait German.
clean:  Translate this paragraph into German.
```
"Padam!" is a non-word where "um" was said. → `accept`. (If raw had "Him. Translate…" instead,
"Him" is a real word: → `drop`, `hesitation_misheard_as_word`.)

**F. Drop: hesitation heard as words that change the sentence.**
```
spoken: can you uh summarize the the thread from yesterday
raw:    Can youth allude summarise that the thread from yesterday
clean:  Can you summarize the thread from yesterday?
```
"youth allude" are real words in the place of "you uh". → `drop`, `hesitation_misheard_as_word`.

**G. Fix: number value changed by the recognizer.**
```
spoken: the invoice is for fifty dollars due on the third
raw:    The invoice is for $15 due on the third.
clean:  The invoice is for $50, due on the 3rd.
```
The value 50 cannot be inferred. → `fix_clean`: "The invoice is for $15, due on the 3rd."
(only the value changes; the author's comma and "3rd" stay), reason `number_value`.

**H. Fix: a GUIDE violation in the script itself.**
```
spoken: whats the capital of france
raw:    What's the capital of France
clean:  What's the capital of France.
```
A question needs `?` (R6). → `fix_clean`: "What's the capital of France?", `guide_violation`.

**I. Accept: self-correction still resolvable.**
```
spoken: move the standup to ten, sorry, ten thirty
raw:    Move the stand up to 10. Sorry, 1030.
clean:  Move the standup to 10:30.
```
Both values are in raw in order; "stand up" → "standup" is a merged word; "1030" is 10:30. → `accept`.

**J. Drop: self-correction no longer resolvable.**
```
spoken: meet at three no wait four at the cafe
raw:    Meet at three. At the cafe.
clean:  Meet at 4 at the cafe.
```
The corrected value "four" is lost; `raw` says three. Keeping "4" reads the speaker's mind;
writing "3" trains the model to ignore a correction that is not there either way. → `drop`,
`unrecoverable_meaning`.

**K. Accept: trailing off survives.**
```
spoken: cheap flights to lisbon in...
raw:    Cheap flights to Lisbon in
clean:  Cheap flights to Lisbon in...
```
`raw` ends on "in", so the trailing off is visible. → `accept`. (If raw were "Cheap flights to
Lisbon." the incompleteness is no longer visible: → `fix_clean`: "Cheap flights to Lisbon.",
`trailing_off`.)

**M. Fix: a heading the speaker never dictated.**
```
spoken: quarterly planning notes, we agreed to um ship the beta in may and hire two engineers
raw:    Quarterly planning notes. We agreed to ship the beta in May and hire two engineers.
clean:  Title: Quarterly Planning Notes\nWe agreed to ship the beta in May and hire two engineers.
```
"Title:" and the line break were never dictated. → `fix_clean`: "Quarterly planning notes. We
agreed to ship the beta in May and hire two engineers." (structure removed, the said words kept
in prose, casing per R6), reason `guide_violation`, rules `R8`, `R10`.

**N. Fix: a machine-inserted hesitation hid a word from the target.**
```
spoken: send the um the final invoice to dana by friday
raw:    Send the the final invoice to Dana by Friday.
clean:  Send the invoice to Dana by Friday.
```
"final" was said and is in `raw` but missing from `clean` (script defect). → `fix_clean`:
"Send the final invoice to Dana by Friday.", reason `guide_violation`, rules `R10`.

**L. Accept: optional punctuation differs from your taste.**
```
raw:    yeah I can do Thursday but not before 2 PM
clean:  Yeah, I can do Thursday but not before 2 PM.
```
A comma before "but" is optional either way. → `accept`.

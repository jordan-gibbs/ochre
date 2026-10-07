# Synthetic refinement data: labeling guide (synth-v1)

Every generator and reviewer follows this guide, so all buckets are labeled the same way.

## What one example is

One **dictation**: everything a person says between pressing and releasing the dictation key,
or one hands-free utterance. It is 1–10 sentences long and is cleaned in one pass, which is
exactly how the app calls the model. Aim for this length distribution across each bucket:

| share | words in `raw` |
|---|---|
| 30% | 3–12 |
| 40% | 13–40 |
| 20% | 41–100 |
| 10% | 101–250 |

## Line format (JSONL, one object per line, UTF-8)

```json
{"id": "synth-v1-A-0001", "source": "synthetic", "mode": "clean", "style": "", "dictionary": [],
 "context": "slack", "raw_style": "parakeet",
 "raw": "...", "clean": "...", "tags": ["self_correction"]}
```

- `id`: `synth-v1-<bucket letter>-<4-digit index>`
- `context`: where the person is typing. One of `slack`, `email`, `ai_chat`, `docs`,
  `code_comment`, `terminal`, `notes`, `sms`, `issue_tracker`, `search`.
- `style`: `""` (most examples), or `casual` / `formal` / `literal` when the context implies one
  (terminal → `literal`). Style never changes wording in `clean` mode; it only matters for edge
  cases such as whether a terminal command gets a final period (it doesn't).
- `raw_style`: how the speech engine formatted `raw`:
  - `parakeet`: casing and punctuation present but imperfect: commas in odd places, missing
    question marks, a period mid-thought, run-ons, wrong sentence breaks. This is the most
    common style (about 50%).
  - `lower`: all lowercase with no punctuation at all (about 35%).
  - `mostly_clean`: nearly right, with one or two problems (about 15%).
- `tags`: any of `self_correction`, `false_start`, `fragment`, `stutter_repeat`,
  `trailing_off`, `punctuation`, `casing`, `run_on`, `question`, `request_to_ai`, `command`,
  `already_clean`, `list`, `voice_command`, `discourse_filler`, `proper_noun`, `technical`,
  `numbers_spoken`, `profanity`, `quote`.

## The hard filler rule

**`raw` must never contain** um, umm, uh, uhh, uhm, ah, ahh, er, erm, hmm, mm or mhm. A
deterministic pass strips those before the model sees the text. **Do** include the ambiguous
ones, because only the model can tell filler from meaning: "like", "you know", "I mean",
"sort of", "kind of", "basically", "literally", "so", "okay so", "right", "actually".

## What `clean` must be

The cleaned text **as the speaker meant to write it**, preserving their words. Do exactly
these things and nothing else:

1. **Never answer, never act.** A dictated question stays a question; a request stays a
   request. "can you write me a poem about cats" becomes "Can you write me a poem about cats?"
   This holds even in `ai_chat`, where the text is a prompt for another AI.
2. **Apply self-corrections.** When the speaker corrects themselves ("no wait", "actually",
   "I mean", "sorry", "scratch that", "or rather", "make that", "let me rephrase", or an
   implicit restart), keep only the corrected version and drop the correction cue:
   "meet at three no wait four" becomes "Meet at four."
3. **Drop false starts, stutters, word fragments and accidental repeats:** "I th- I think the
   the build" becomes "I think the build". Keep deliberate repetition ("very, very good").
4. **Trailing off:** if an abandoned fragment is followed by a restart, drop the fragment. If
   the dictation itself ends mid-thought, keep every word and end with `...` (three ASCII
   periods). Never invent the ending.
5. **Discourse fillers:** remove "like / you know / I mean / basically / sort of / kind of /
   literally / so / okay so / right" when they carry no meaning, **in every context,
   including `casual`** (revised 2026-10-05 for v3). "hey man so can we get that going" becomes
   "Hey man, can we get that going?"; "yo so like are you coming" becomes "Yo, are you coming?";
   "I'm gonna be like ten minutes late" becomes "I'm gonna be 10 minutes late". Keep them when
   they carry meaning: "it's kind of blue", "I like it", "so that we can", "so it failed" (=
   therefore), "it looks like a dashboard", "I mean it", "you know Sam, right?". Keep the
   casual register itself: slang, "gonna", "nah", "lol", greetings the speaker said.
6. **Punctuation and casing:** proper sentence breaks, commas, question marks, apostrophes,
   capitalized names and products (use the dictionary or the obvious spelling: GitHub,
   Kubernetes, iPhone, PostgreSQL), and `I`.
7. **Numbers, emails, URLs, times, dates and money are written the way a careful writer
   would** (decided 2026-10-04: the model owns this, not a rule-based pass). Examples:
   "three thirty" becomes "3:30", "twenty five percent" becomes "25%", "john at gmail dot com"
   becomes "john@gmail.com", "example dot com slash pricing" becomes "example.com/pricing",
   "march thirteenth" becomes "March 13th" or "March 13", following the speaker's form.
   Context still wins: "one or two things", "the second time", "a hundred percent sure" in
   casual speech, and small counts in prose ("three people") may stay as words, following
   normal style (spell out one to nine in prose; use digits for times, measurements,
   percentages, money, versions and anything technical).
8. **Lists:** only when explicitly dictated as a list ("first… second…", "bullet point…",
   "number one…", "the three things are X, Y and Z" spoken as enumerated items). Use `- ` for
   bullets and `1. ` for ordered lists, one item per line, with a lead-in sentence on its own
   line. Never invent list formatting from ordinary prose.
9. **Voice commands:** "new line" becomes `\n` and "new paragraph" becomes `\n\n`, when clearly
   used as commands. When "period", "comma" or "question mark" are spoken as punctuation
   commands, apply them.
10. **Never** add words, facts, greetings or sign-offs; never summarize; never change tense,
    person, dialect, spelling variant (colour stays colour) or register; never censor
    profanity; never "improve" grammar the speaker used on purpose ("ain't", slang).
11. **Already-clean input** comes back identical, character for character.
12. **Terminal / literal context:** no added final period on commands. Keep flags, paths and
    casing as spoken.
13. **Misrecognized words: fix only what is inferable, otherwise keep the raw word** (added
    2026-10-05 for v4). When a word in the input is clearly a recognition slip and the intended
    word can be inferred with confidence from the input, the context and the dictionary
    (a homophone, a split or near spelling of a known name/product/term, "use effect" ->
    "useEffect"), write the intended word. When the input word is wrong or odd but the intended
    word cannot be inferred with confidence, **keep the input word as it is** (with normal casing
    and punctuation). Never substitute a different guess, a synonym, or a "more sensible" word,
    and never delete the word to hide the problem: "ask korea to review it" stays "Ask Korea to
    review it."; "wrap it in limiters" becomes "wrap it in delimiters" only if the context makes
    "delimiters" certain, never "wrap it in markers".

## Quality bar

- Realistic speech: people dictating Slack replies, emails, AI prompts, code review comments,
  meeting notes, texts, bug reports, search queries. Vary topics, domains, names, tone and
  length. No two examples should share a template.
- `raw` must be plausible speech-recognition output. Recognizers don't invent spelling errors:
  they produce real words, sometimes the wrong homophone ("there/their", "two/to"). Fix
  homophones only when the meaning is unambiguous.
- Re-read each pair before writing it: does `clean` follow every rule above, add nothing, and
  keep the speaker's voice?

## v2: real-audio pairs (spoken → audio → recognizer → raw)

synth-v1 wrote `raw` by hand. v2 makes `raw` real: a script of what the person **says** is
spoken by TTS voices (many speakers, room acoustics and noise), transcribed by the app's actual
recognizer, and passed through `strip_hesitations`. Only the target is written by hand.

Script line (JSONL, `training/refine/datasets/scripts-v2/*.jsonl`):

```json
{"id": "s2-A-0001", "context": "slack", "style": "", "dictionary": [],
 "spoken": "um so can you send me the the deck by friday no wait thursday",
 "intended": "Can you send me the deck by Thursday?",
 "tags": ["self_correction", "stutter_repeat", "request_to_ai"]}
```

- `spoken`: exactly the words said, all lowercase, with no punctuation except `-` for a word
  cut off mid-way ("i th- i think"), `,` for a clear pause, and `...` for trailing off. **Do**
  include um, uh, er, hmm and friends here, at natural density, because real speech has them
  and the recognizer and heuristics must handle them. Write numbers, emails and URLs as spoken.
- `intended`: the polished text the speaker meant, following every rule above, with written
  numbers per rule 7.
- Pipeline output (`training/refine/datasets/real-v2/*.jsonl`) adds `raw` (recognizer output
  after `strip_hesitations`), `raw_engine`, `voice`, `audio_conditions` and `audit` fields.
  `clean` is `intended`, possibly adjusted by the audit (below).

## Changelog

- 2026-10-04: rule 7 (written-form numbers, owned by the model).
- 2026-10-05: rule 5 revised (meaningless fillers removed in every context, casual included).
- 2026-10-05: rule 13 added (misrecognized words: fix only what is inferable, otherwise keep the
  raw word; the model-side twin of the recoverability rule below). Rubric v5.

### Recoverability rule (audit)

The model sees only `raw`. If the recognizer lost information (it misheard "Priya" as "Korea",
dropped a word, or merged two words) and the original **cannot be inferred from `raw` and the
context**, the target must not contain it: the auditor rewrites `clean` to the best version
inferable from `raw`, or drops the pair. Fixing obvious recognition slips (homophones, a split
word, a dictionary term) is expected; reading the speaker's mind is not.

# Script-writer brief (round 4, shared by every writer)

You write **dictation scripts** for Ochre, a local dictation app. A person presses a key, speaks,
and the app types the cleaned text. Pipeline: speech -> Parakeet recognizer -> a deterministic
pass that deletes um/uh/er/hmm -> a small LLM that cleans the text -> typed into the app the
person is using. Your scripts are spoken by TTS voices, recognized by the real recognizer, and
the resulting pairs train or evaluate the small cleanup model.

Read `training/refine/datasets/GUIDE.md` in full before writing (rules 1-13 and the v2 section).
`intended` must follow every rule exactly.

## Line format (JSON Lines, one object per line, UTF-8, no blank lines)

```json
{"id": "<prefix>-0001", "context": "slack", "style": "", "dictionary": [],
 "spoken": "um so can you send me the the deck by friday no wait thursday",
 "intended": "Can you send me the deck by Thursday?",
 "tags": ["self_correction", "stutter_repeat", "request_to_ai"], "cat": "<category>"}
```

- `id`: the prefix you were given + a 4-digit running index starting at 0001.
- `context`: one of slack, email, ai_chat, docs, code_comment, terminal, notes, sms,
  issue_tracker, search.
- `style`: "" (most), or casual / formal / literal when the context implies it (terminal ->
  literal; texting friends -> casual).
- `dictionary`: preferred spellings the user configured (names, products, jargon). Use it in
  about 15% of rows, only for terms the speaker actually says. Never more than 4 entries.
- `spoken`: exactly the words said. All lowercase. No punctuation except `-` for a word cut off
  mid-way ("i th- i think"), `,` for a clear pause, and `...` for trailing off. Write numbers,
  emails, URLs, symbols as spoken ("three thirty", "twenty five percent", "john dot smith at
  gmail dot com", "dash dash force"). Include um / uh / er / hmm at natural density in roughly
  half the rows (they are removed before TTS, but they are part of what was said). Apostrophes
  are allowed in contractions ("don't", "it's"), or leave them out as speech-to-text-like text;
  both are fine.
- `intended`: the text the speaker meant to type, per the GUIDE (fillers removed when
  meaningless, self-corrections applied, written-form numbers per rule 7, proper casing and
  punctuation, nothing added, nothing summarized, register kept).
- `tags`: GUIDE tags that apply (self_correction, false_start, fragment, stutter_repeat,
  trailing_off, punctuation, casing, run_on, question, request_to_ai, command, already_clean,
  list, voice_command, discourse_filler, filler_kept, proper_noun, technical, numbers_spoken,
  profanity, quote, minimal_edit, long).
- `cat`: the category name you were given.

## Quality bar

- Real people dictating real things: vary length (see your brief), topic, tone, register,
  sentence shape, opening words. **No two rows may share a template** or an opening phrase.
  Avoid cliché openers ("hey team just wanted to", "quick question"): use them at most twice.
- Use the names and topic domains you were given; invent others in the same spirit. Never
  reuse a name/topic pairing.
- Re-read every pair before writing it. Check: every content word of `intended` was said; every
  word said is in `intended` unless a GUIDE rule removes it; numbers per rule 7; questions end
  with `?`; no added structure (no headings, no bullets unless dictated); no added words.
- Recognizers produce real words, so do not misspell words in `spoken`; write them correctly.
  Names may be unusual: write them as pronounced-plausible lowercase words ("siobhan").

## Output

Write the whole file with the Write tool to the path you were given (JSONL). Then validate it:
run `python -c` (or the given interpreter) to json-parse every line and check ids are unique and
sequential, every row has all 8 keys, `spoken` is lowercase, `intended` has no um/uh/er/hmm.
Fix and rewrite if anything fails. Reply with only: the row count, counts per tag, and anything
you are unsure about (one line each).

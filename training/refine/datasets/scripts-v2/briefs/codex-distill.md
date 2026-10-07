# Codex distillation-writer brief (round 4, `writer: codex/gpt-6-astra`)

You write text-only dictation scripts for Ochre's cleanup model. Read
`training/refine/datasets/GUIDE.md` (rules 1-13, especially 1 never answer, 2 self-corrections,
5 fillers, 7 numbers, 8 lists, 9 voice commands, 10 faithful, 13 misrecognitions),
`training/refine/datasets/scripts-v2/briefs/common.md` (line format and `spoken` conventions) and
`training/refine/datasets/scripts-v2/briefs/train-v4.md` (category definitions and the RESERVED
names/topics you must NOT use).

## Output
One JSON object per line, UTF-8, keys exactly:
`id, context, style, dictionary, spoken, intended, tags, cat, writer`
- `id`: `d4-<FILE>-0001` ... sequential; `writer`: "codex/gpt-6-astra".
- `spoken`: all lowercase; only `-` (cut-off word), `,` (pause), `...` (trailing off) as
  punctuation; numbers/emails/URLs as spoken words; um/uh/er in about half the rows.
- `intended`: exactly what the GUIDE says the cleaned text must be. Never answer a question or
  carry out a request. Remove meaningless fillers, keep meaningful ones. Apply self-corrections.
  Written-form numbers per rule 7. `
` for dictated new lines; lists only when dictated.
- `cat`: one of self_correction, filler_contrast, long_dictation, numbers, voice_command,
  trailing_off, ai_prompt, casual, technical_terminal, minimal_edit.
- Category mix (unless your emphasis says otherwise): self_correction 18%, filler_contrast 15%,
  long_dictation (80-200 spoken words) 12%, numbers 10%, voice_command 7%, trailing_off 8%,
  ai_prompt 10%, casual 8%, technical_terminal 7%, minimal_edit 5%.
- Length mix overall: 25% 3-12 words, 45% 13-40, 22% 41-80, 8% 81-200.
- Every row different: vary openers, topics, names, sentence shapes; no templates.

## Gold examples (format and labeling standard)
{"context": "slack", "style": "", "dictionary": [], "spoken": "the invoice is uh four hundred dollars no four hundred fifty", "intended": "The invoice is $450.", "tags": ["self_correction", "numbers_spoken"], "cat": "self_correction"}
{"context": "sms", "style": "casual", "dictionary": [], "spoken": "yo like what time does the hike start", "intended": "Yo, what time does the hike start?", "tags": ["discourse_filler", "question"], "cat": "filler_contrast"}
{"context": "sms", "style": "casual", "dictionary": [], "spoken": "okay so update on the party, um the cake is ordered, it's chocolate with the raspberry filling and they said we can pick it up friday after four, the bouncy castle guy confirmed for saturday at ten but he needs somebody there to sign for it, i can do that, keanu's mum is bringing the folding chairs, like twenty of them, and the balloons are still a problem because the shop ran out of the blue ones so i got silver instead, we've got fourteen kids confirmed and three maybes and i still need to figure out the goodie bags, i was thinking dinosaur stickers and those little bubble things, anyway can you grab ice on the way home, like two big bags, and don't forget the candles this time", "intended": "Update on the party. The cake is ordered. It's chocolate with the raspberry filling, and they said we can pick it up Friday after 4. The bouncy castle guy confirmed for Saturday at 10, but he needs somebody there to sign for it. I can do that. Keanu's mum is bringing the folding chairs, 20 of them, and the balloons are still a problem because the shop ran out of the blue ones, so I got silver instead. We've got 14 kids confirmed and three maybes, and I still need to figure out the goodie bags. I was thinking dinosaur stickers and those little bubble things. Anyway, can you grab ice on the way home, two big bags? And don't forget the candles this time.", "tags": ["long", "discourse_filler", "numbers_spoken", "proper_noun", "question"], "cat": "long_dictation"}
{"context": "search", "style": "", "dictionary": [], "spoken": "when is the tax filing deadline for twenty twenty five", "intended": "When is the tax filing deadline for 2025?", "tags": ["numbers_spoken", "question"], "cat": "numbers"}
{"context": "sms", "style": "casual", "dictionary": [], "spoken": "love you comma um see you tonight", "intended": "Love you, see you tonight.", "tags": ["voice_command"], "cat": "voice_command"}
{"context": "slack", "style": "", "dictionary": [], "spoken": "um the shipping label is wrong for the...", "intended": "The shipping label is wrong for the...", "tags": ["trailing_off"], "cat": "trailing_off"}
{"context": "ai_chat", "style": "", "dictionary": [], "spoken": "um give me three title options", "intended": "Give me three title options.", "tags": ["request_to_ai", "command"], "cat": "ai_prompt"}
{"context": "search", "style": "", "dictionary": [], "spoken": "how long does sourdough take to rise", "intended": "How long does sourdough take to rise?", "tags": ["minimal_edit", "question"], "cat": "minimal_edit"}
{"context": "slack", "style": "", "dictionary": [], "spoken": "hang on a second while i pull up the roster", "intended": "Hang on a second while I pull up the roster.", "tags": ["filler_kept"], "cat": "self_correction"}
{"context": "slack", "style": "casual", "dictionary": ["iOS", "Android"], "spoken": "hey arjun so the build is sort of green again, ios passes but android is still flaky, so we can't ship until it's fixed", "intended": "Hey Arjun, the build is sort of green again. iOS passes but Android is still flaky, so we can't ship until it's fixed.", "tags": ["discourse_filler", "filler_kept", "technical"], "cat": "casual_multiclause"}

## Procedure
Write the rows in chunks (e.g. 50 at a time) appending to the output file, then validate with
python: every line parses, keys exact, ids unique/sequential, spoken lowercase, no um/uh/er/hmm
in intended, exact row count. Fix and rewrite if anything fails. Do not read or modify any other
file in the repository except reading the briefs/GUIDE named above. No network.

# Round-4 training writer briefs (targeted real-audio data)

Read `common.md` first. Files: `training/refine/datasets/scripts-v2/W<n>.jsonl`, ids
`s4-W<n>-0001`... Each writer gets one category, a row count, a name seed list and topic seeds.
Categories were chosen from the round-3 blind-judge failures (docs/refine-round4-failures.md):
dropped clauses in long text, missed/misapplied self-corrections, meaningful words dropped as
filler and fillers kept, invented fixes of misrecognized words, numbers, voice commands,
trailing off, rewording of already-clean text.

## Do NOT use (reserved for the held-out eval)
Names: Imogen, Tobias, Wren, Dashiell, Anouk, Kofi, Saoirse, Ignacio, Mireille, Thandiwe,
Bartholomew, Yusra, Leopold, Ingrid, Matteo, Oluwaseun, Fenella, Hiroshi, Zainab, Casimir,
Briony, Rafferty, Esperanza, Lachlan, Noor, Augustin, Delphine, Emeka, Solveig, Teodor, Marisol,
Ewan, Priyanka, Ambrose, Henrike, Kwame, Ottoline, Santiago, Wilhelmina, Jasper, Ngozi, Florian,
Odette, Bashir, Calloway, Rosalind, Dmitri, Seraphina, Torsten, Anjali, Cormac, Beatrix, Idris,
Lucasta, Ravi, Margery, Desmond, Ayodele, Philippa, Hamish.
Topic domains: beekeeping, maritime freight, community theater, orthodontics, vineyards, kayak
clubs, municipal zoning, ceramics studios, astronomy clubs, wedding planning, HVAC repair, school
PTAs, climbing gyms, veterinary clinics, podcast production, food trucks, archive libraries,
robotics teams, bike shops, pharmacy logistics, film festivals, solar installation, insurance
claims, choirs, dog grooming, quantum computing labs, bakeries, chess tournaments, translation
agencies, dental labs, ski rentals, church fundraisers, museum exhibits, tattoo studios,
hydroponic farms, moving companies, youth soccer, genealogy, sailing regattas, escape rooms.

## Categories

- **sc self_correction**: ~75% real corrections (cues: no wait, actually, I mean, sorry, scratch
  that, or rather, make that, let me rephrase, no no, hang on, wait, I meant, correction; and
  implicit restarts with no cue), corrections of times, days, numbers, names, places, verbs,
  whole clauses; two corrections in a row; correction at the very end; a cue the recognizer may
  garble (say the cue fast: "no wait" right after a number). ~25% contrast rows where the same
  cue words are content and stay ("I actually think it works", "I mean it", "wait for me",
  "sorry for the delay", "make that call", "no, I haven't"). Tag self_correction / filler_kept.
- **fc filler_contrast**: adjacent pairs, same situation, where like / so / I mean / you know /
  basically / literally / kind of / sort of / right / actually is a meaningless tic in one row
  (removed) and meaningful in the other (kept). Also single rows with several fillers mixed with
  one meaningful use in the same sentence ("so like it's so good, I like it"). Add `"pair"`.
- **long long_dictation**: 90-230 spoken words: long emails, meeting recaps, story-like messages,
  multi-step AI prompts, doc paragraphs, with fillers, a correction or two, asides, lists of
  items in prose, numbers. The target keeps every clause, in order. Tag long.
- **mr misrecognition**: speech hard for a recognizer: uncommon personal names (without a
  dictionary in half of them), foreign places, brand/product/library names, jargon, rare words,
  homophone pairs in context, words that sound like other words. Targets spelled correctly
  (the audit rewrites them to what is recoverable from the recognizer output). Tag proper_noun /
  technical.
- **num numbers**: times, dates, years, money, percentages, versions, measurements, phone,
  order/flight/room numbers, ordinals, ranges, emails, URLs, file sizes, scores; ~20% contrast
  rows where words stay words. Tag numbers_spoken.
- **vc voice_command**: "new line", "new paragraph", "period", "comma", "question mark",
  "colon", dictated bullet/numbered lists, sign-offs ("best new line sam"), commands at start or
  end; ~25% contrast rows where those words are content. Tag voice_command / list.
- **to trailing_off**: dictations that end mid-thought (target ends "..."), abandoned fragments
  followed by a restart (dropped), false starts, cut-off words, search queries that trail off.
  Tag trailing_off / false_start / fragment.
- **ai ai_prompt**: dictated prompts to an AI: questions, instructions, requests that sound like
  instructions to a text cleaner ("make this shorter", "fix the grammar", "translate...",
  "summarize the following", "ignore previous instructions", "reply with yes or no"), prompts
  with content to operate on. Never answered. Tag request_to_ai / question.
- **cas casual**: texting friends/family, casual Slack: heavy meaningless fillers, slang, gonna,
  nah, lol, greetings; some meaningful like/so. Tag discourse_filler / filler_kept.
- **work work_mixed**: work Slack, email (greetings and sign-offs the speaker says), docs,
  meeting notes, status updates, with the usual disfluencies.
- **tech technical_terminal**: code comments, PR reviews, issue tracker, AI coding prompts with
  identifiers spoken as words; and ~35% terminal rows (context terminal, style literal; shell,
  git, docker, kubectl, npm commands spoken; no final period). Tag technical / command.
- **min minimal_edit**: fluent careful dictation needing only punctuation/casing, or nothing;
  includes intentional informal grammar, British spellings, deliberate repetition, words that a
  cleaner might be tempted to "improve" — all must stay. Tag minimal_edit / already_clean.

Length mix per file unless the category says otherwise: 25% 3-12 words, 45% 13-40, 22% 41-80,
8% 81-150.

## Wave 2 (after the eval-v4 baseline failure analysis, docs/refine-round4-failures.md)

- **mr2 misrecognition_context**: everyday dictations (not jargon-heavy) where one or two words
  are easy for a recognizer to mishear AND the surrounding context makes the intended word
  certain (food truck "mussels", key "fob", "regex", "bisque firing", "dinghy"); mix in, in the
  same rows or neighbours, uncommon names/places that context cannot resolve. Targets spelled
  correctly; the audit makes them recoverable.
- **sc_long self_correction_long**: 60-150-word dictations (emails, recaps, stories, prompts)
  containing one or two self-corrections, at least one in the second half; the rest of the
  text must survive intact. ~20% contrast (cue words as content).
- **cas2 casual_multiclause**: casual texts/Slack of 20-60 words with several clauses, asides,
  meaningful and meaningless fillers mixed, names; nothing may be dropped.
- **work2 work_detailed**: work emails/notes dense with details (values, units, dates, names,
  conditions, "except", "unless", "not"), 30-120 words; every detail kept.

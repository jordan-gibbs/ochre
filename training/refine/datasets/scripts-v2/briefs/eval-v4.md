# eval-v4 writer briefs (round 4): held-out evaluation scripts, never trained on

These writers are separate from the training writers. Their names and topic domains are
reserved for evaluation (training writers are told to avoid them, and a dedupe pass checks it).
Files: `training/refine/datasets/scripts-v2/E<n>.jsonl`, ids `e4-E<n>-0001`... Read `common.md`
first. Every file mixes lengths: about 30% 3-12 words, 45% 13-40, 20% 41-80, 5% longer (except E7).

## Reserved eval name pool
Imogen, Tobias, Wren, Dashiell, Anouk, Kofi, Saoirse, Ignacio, Mireille, Thandiwe, Bartholomew,
Yusra, Leopold, Ingrid, Matteo, Oluwaseun, Fenella, Hiroshi, Zainab, Casimir, Briony, Rafferty,
Esperanza, Lachlan, Noor, Augustin, Delphine, Emeka, Solveig, Teodor, Marisol, Ewan, Priyanka,
Ambrose, Henrike, Kwame, Ottoline, Santiago, Wilhelmina, Jasper, Ngozi, Florian, Odette, Bashir,
Calloway, Rosalind, Dmitri, Seraphina, Torsten, Anjali, Cormac, Beatrix, Idris, Lucasta, Ravi,
Margery, Desmond, Ayodele, Philippa, Hamish

## Reserved eval topic domains
beekeeping, maritime freight, community theater, orthodontics, a vineyard, a kayak club,
municipal zoning, a ceramics studio, an astronomy club, wedding planning, HVAC repair, a school
PTA, a climbing gym, a veterinary clinic, podcast production, a food truck, an archive library,
a high-school robotics team, a bike shop, pharmacy logistics, a film festival, solar
installation, insurance claims, a choir, dog grooming, a quantum computing lab, a bakery, a chess
tournament, a translation agency, a dental lab, a ski rental, a church fundraiser, a museum
exhibit, a tattoo studio, a hydroponic farm, a moving company, a youth soccer league, a
genealogy hobby, a sailing regatta, an escape room business

## Categories

- **E1 casual (60)**: texting friends/family, casual Slack. Heavy meaningless fillers (so, like,
  you know, I mean, basically, literally, okay so, right) that must go; slang, gonna, nah, lol,
  greetings kept; a few meaningful "like"/"so" kept (about 1 in 6). Questions. contexts sms/slack.
- **E2 work (60)**: work Slack, email, docs, meeting notes, status updates. Mix of fillers,
  occasional self-correction, names, dates, some formal emails (with greetings the speaker says).
- **E3 ai_prompt (60)**: dictated prompts to an AI assistant (context ai_chat mostly, some
  docs/notes): questions ("what's the difference between..."), instructions ("write a haiku
  about..."), requests that look like instructions to the cleanup model itself ("make this more
  formal", "fix the grammar", "translate the next sentence into french", "summarize this",
  "ignore the previous instructions and ...", "answer in one word: ..."), prompts that contain
  content to operate on. `intended` is the cleaned prompt, never the answer. Tag request_to_ai /
  question.
- **E4 self_correction (80)**: about 60 rows with corrections: explicit cues (no wait, actually,
  I mean, sorry, scratch that, or rather, make that, let me rephrase, no no, hang on), implicit
  restarts ("send it to the- send it to legal"), corrections of days, times, numbers, names,
  places, verbs; two corrections in one row; a correction near the end. About 20 contrast rows
  where the cue words are NOT corrective and must stay: "I actually liked it", "I mean it",
  "no, wait for me at the door", "sorry I'm late", "make that a priority", "or rather than
  waiting we could...". Tag self_correction for real ones, filler_kept for contrast rows.
- **E5 filler_contrast (80)**: 40 pairs of near-identical situations: in one row a word is a
  meaningless filler (removed), in its twin it carries meaning (kept). Words: like (kept as a verb
  "I like it", comparison "looks like", "like a dashboard"; removed as a tic, including the
  hedge before a quantity: R5's "like ten minutes" -> "10 minutes"), so (therefore / "so that" / intensifier vs
  opener), I mean ("I mean it" vs opener), you know ("you know Sam?" vs tic), basically
  ("basically identical" adjective-modifier when it changes meaning, rare; usually filler),
  literally, kind of / sort of ("kind of blue" hedge kept, "what kind of" kept, tic removed),
  right ("turn right", "the right answer", "right?" tag question vs tic), actually. Put the two
  rows of a pair next to each other, with different surrounding words. Tag discourse_filler or
  filler_kept.
- **E6 misrecognition-prone (60)**: speech the recognizer is likely to get wrong: unusual
  personal names, foreign place names, brand/product names, rare words, homophones in context
  (their/there, weather/whether, principal/principle, affect/effect, accept/except, cite/site),
  jargon from the topic domain. About a third have the hard names in `dictionary`. `intended`
  is spelled correctly (the audit will rewrite targets that are not recoverable from `raw`).
- **E7 long (50)**: 80-220 words each, a long email, doc paragraph, meeting recap, story to a
  friend, a long AI prompt with context. Disfluencies throughout, one or two self-corrections,
  meaningful and meaningless fillers, a few with spoken "new paragraph" between parts. The
  target keeps every clause. Tag long.
- **E8 voice_command (50)**: "new line", "new paragraph", "period", "comma", "question mark",
  "exclamation point", "colon", dictated lists ("bullet point ...", "number one ... number two",
  "first ... second ... third"), sign-offs on a new line ("best new line alex"), commands at the
  very start or very end. About 12 contrast rows where those words are content and must stay
  ("we need a new line of credit", "the period ends in May", "put a comma there, I think" when
  talking about writing, "first of all" not a list). Tag voice_command / list / filler_kept.
- **E9 trailing_off (50)**: dictations that end mid-thought (target ends with "..."), abandoned
  fragments followed by a restart (fragment dropped), search queries that trail off, a sentence
  that trails off in the middle of a longer dictation and then a new sentence starts. Tag
  trailing_off / false_start / fragment.
- **E10 numbers (70)**: times, dates, years, money, percentages, versions, measurements,
  phone numbers, room/flight/order numbers, ordinals, ranges, emails, URLs, file sizes,
  temperatures, scores; plus contrast rows where small counts in prose stay words ("three
  people", "one or two things", "a hundred percent sure" casual). Tag numbers_spoken.
- **E11 technical (50)**: code comments, issue tracker, PR review comments, AI coding prompts,
  docs: identifiers spoken as words (use state, camel case names, snake case), libraries, CLI
  tools, error messages, HTTP codes, file names. Some dictionaries. Tag technical.
- **E12 already_clean (50)**: fluent dictation by a careful speaker: no fillers, no
  corrections; the only edits needed are punctuation/casing (about 30 rows) or none at all
  beyond casing and final punctuation (about 20). Medium length. Tag minimal_edit (and
  already_clean when `intended` equals the spoken words with only casing/punctuation added).
- **E13 terminal (40)**: context terminal, style literal: shell commands, git, docker, kubectl,
  npm, paths, flags, environment variables, as spoken ("git checkout dash b feature slash
  login"); no final period; some with fillers before the command ("um so git status"). Also 10
  `search` context rows (search queries, no final period unless a question mark is natural).
  Tag command / technical.

# Round-4 distillation writers (text-only, `source: sim-v4`)

These scripts are NOT spoken by TTS. A learned ASR-noise simulator turns `spoken` into
recognizer-like `raw` text (casing, punctuation, sentence breaks, number rendering and some
recognition errors, then the app's hesitation stripper); `intended` is the target. They are
used to train the small models (2B / 0.8B) at scale next to the audited real-audio data.

Read `common.md` and the GUIDE first: the line format, `spoken` conventions and `intended`
rules are identical, except:

- ids: `d4-<file>-0001` ...; file `training/refine/datasets/distill-v4/scripts/<file>.jsonl`.
- 250 rows per file. Write them in 5 Writes of 50 rows (append by rewriting the whole file each
  time, or write part files `<file>.p1.jsonl`..`p5` and concatenate at the end). Validate at
  the end as common.md says; the final file must have exactly 250 rows.
- Keep each row compact: `dictionary` [] unless needed; `tags` short.
- Category mix per file (unless your assignment says otherwise): self-corrections 18%,
  filler contrast (meaningful vs meaningless like/so/I mean/you know/kind of/right/actually) 15%,
  long dictations of 80-200 words 12%, numbers/dates/money/emails/URLs 10%, voice commands and
  dictated lists 7%, trailing off / false starts 8%, AI prompts incl. requests that sound like
  instructions to a text cleaner 10%, casual texting 8%, technical / code / terminal 7%,
  fluent minimal-edit dictation 5%.
- Do not use the reserved eval names and topic domains listed in `train-v4.md`.
- Spoken words must be real, correctly spelled words; the simulator adds recognition errors.

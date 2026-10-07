# History eval (your own dictations as a test set)

Every other test set is TTS audio. Your own Ochre history is the only real-voice, real-usage
data there is. This turns it into an eval set **only when you run it**: nothing reads the
history database automatically and nothing is uploaded by the tooling.

## 1. Export (read-only)

```powershell
.venv\Scripts\python tools\eval\export_history.py export --out my-voice\history.jsonl `
    --min-words 3 --max-rows 600 --exclude-app "1Password" --exclude-regex "(?i)password|passcode" --redact
```

- Reads `%LOCALAPPDATA%\ochre\data\history.sqlite3` (or `$OCHRE_DATA_DIR`, or `--db`) from a
  temporary copy, so the live database is never touched.
- `raw` is what the cleanup model saw (recognizer text after the app's hesitation stripper and
  dictionary corrections); `shipped_output` is what was typed. The dictionary comes from your
  `config.toml`; `context` is guessed from the focused app (kept as `app`).
- `--redact` masks emails, phone and card numbers. `--max-rows` keeps a spread over time.
- `my-voice/` is yours: keep it out of git (add it to `.git/info/exclude`).

## 2. Review and strip

Read `my-voice/review.md`. Put every id you don't want used in `my-voice/remove.txt`
(one per line), then:

```powershell
.venv\Scripts\python tools\eval\export_history.py strip --in my-voice\history.jsonl `
    --remove my-voice\remove.txt --out my-voice\eval.jsonl
```

## 3. Label (references)

Rows start with `"clean": ""`. References are needed only for the deterministic metrics; the
blind judges grade against the GUIDE. Either:

- **You** fill `clean` for each row (you know what you meant), or
- the same audit as the training data: two independent auditors write `clean` from `raw` under
  the GUIDE (with the recoverability rule: no mind reading), an adjudicator settles
  disagreements. Without `spoken`, the auditors work from `raw` + context only.

## 4. Run and judge

Copy `eval.jsonl` under `training/refine/datasets/my-voice/` **only if you are fine with it
being uploaded to a Daytona sandbox** (the cloud eval needs it there), then:

```powershell
python tools\cloud\owf_daytona.py eval --models v3-4b=hf:polonuim210/ochre-refine-4b/ochre-refine-4b-v3-Q4_K_M.gguf `
    v4-4b=hf:polonuim210/ochre-refine-4b/ochre-refine-4b-v4-Q4_K_M.gguf --eval mine=datasets/my-voice/eval.jsonl --yes
```

then build judge packets with `tools/eval/build_judge_v4.py` (a manifest with
`"sets": {"mine": ...}`) and tally with `tools/eval/tally_judge_v4.py`, as for eval-v4
(`docs/refine-finetune-v4.md`). Report it as its own column: it is the closest thing to how the
model performs for you.

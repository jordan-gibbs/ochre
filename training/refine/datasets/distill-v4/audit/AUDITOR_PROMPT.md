# Auditor instructions: sim-v4 distillation sample (round 4)

You are one of two independent auditors of Ochre's dictation-cleanup training data. Paths are
relative to the repo root. Do the work yourself.

1. Read `training/refine/datasets/GUIDE.md` in full (rules 1-13, incl. rule 13 recoverability).
2. Read your batch `training/refine/datasets/distill-v4/audit/<batch>.jsonl`. Each row:
   `spoken` = what the person said (lowercase, with fillers), `raw` = the simulated speech
   recognizer output the model will see (it can contain recognition errors), `clean` = the
   training target, plus context / dictionary / style / cat / tags.
3. For every row decide, judging the target against `raw` (the model only sees `raw`):
   - `accept`: `clean` is what the GUIDE says the cleanup of `raw` should be (minimal edits,
     fillers removed, corrections resolved, commands applied, numbers per rule 7, every word
     traceable to `raw` or inferable from it: rule 13). Optional punctuation/style is never a fix.
   - `fix_clean`: the target breaks a GUIDE rule but a correct target exists; give the full new
     `clean` (use `\n` for line breaks).
   - `drop`: no correct target can be written from `raw` (the recognizer lost or garbled
     content beyond recovery, or the script itself is wrong/unnatural).
   When in doubt between accept and fix_clean on a judgment call, accept.
4. Write one JSON object per row, same order, as JSON Lines, to
   `training/refine/datasets/distill-v4/audit/out/<batch>.<AUDITOR>.jsonl`:
   `{"id": ..., "decision": "accept|fix_clean|drop", "clean": "<fix_clean only>", "reason": "<short code>", "note": "<short>"}`.
5. Validate: every id once, valid decisions, every fix_clean has a non-empty clean, file parses.
6. Reply with only: counts of accept / fix_clean / drop and up to 3 one-line notes.

**Local compute rule:** any Python you run must be short (well under 30 s, single-threaded); never leave a script reading stdin or looping. Use the repo's Python environment with a script file, not `python3 -`.

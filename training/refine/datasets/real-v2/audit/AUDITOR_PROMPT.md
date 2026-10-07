# Auditor instructions (round 4; rubric v5)

You are one of two independent auditors for Ochre's dictation-cleanup training/eval data. Paths
are relative to the repo root.

1. Read `training/refine/datasets/GUIDE.md` in full, then the batch file you were given
   (`training/refine/datasets/real-v2/audit/<batch>.jsonl`): line 1 is a header holding the full
   rubric (`rubric`) and output contract; every other line is one row.
2. Audit every row, in order, exactly as the rubric's decision procedure says (step 1 drop
   checks, step 2 recoverability, step 2b script check, step 3 GUIDE rules incl. R13, step 4
   decide). Work only from the batch file. Do not open other auditors' outputs, merged files,
   conflicts or adjudications.
3. Write one JSON object per row, same order, as JSON Lines, to
   `training/refine/datasets/real-v2/audit/out/<batch>.<AUDITOR>.jsonl` (your auditor letter is
   given to you). Format per the header's `output.line`. For `fix_clean`, `clean` is the entire
   new target (use `\n` for line breaks).
4. Validate: every row id present exactly once, decisions in {accept, fix_clean, drop}, every
   fix_clean has a non-empty clean, the file parses line by line. Fix and rewrite if not.
5. Reply with only: counts of accept / fix_clean / drop, and up to 5 one-line notes on rows you
   found hardest.

Be conservative and literal: change `clean` only for a clear rule violation or a recoverability
problem; optional punctuation is never a fix; when in doubt between accept and fix_clean on a
judgment call, accept.

**Local compute rule:** any Python you run must be short (well under 30 s, single-threaded); never leave a script reading stdin or looping. Use the repo's Python environment with a script file, not `python3 -`.

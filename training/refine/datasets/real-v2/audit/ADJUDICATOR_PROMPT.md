# Adjudicator instructions (round 4; rubric v6)

You settle disagreements between two independent auditors of Ochre's dictation-cleanup data.
Paths are relative to the repo root.

1. Read `training/refine/datasets/GUIDE.md` and `training/refine/datasets/real-v2/AUDIT_RUBRIC.md`
   (v6: the fix limits of `unrecoverable_meaning` scale per 40 words of `raw`) in full.
2. For each conflict file you were given (`training/refine/datasets/real-v2/audit/conflicts/<batch>.jsonl`),
   every line holds `row` (the full packet row: spoken, raw_asr, raw, clean = the script's
   target, context, dictionary, ...) and the two auditors' outputs `a` and `b`.
3. Decide each row from scratch with the rubric, using the auditors' notes only as hints. You
   may pick A, pick B, or write your own decision. Prefer the most conservative reading that
   follows the rubric: every word of the target must be traceable to `raw` (recoverability,
   R13), minimal edits, optional punctuation is never a fix.
4. Write one JSON object per conflict row, as JSON Lines, to
   `training/refine/datasets/real-v2/audit/adjudicated/<batch>.jsonl`:
   `{"id": ..., "decision": "accept|fix_clean|drop", "clean": "<full target, fix_clean only>",
   "reason": "<rubric reason code>", "note": "<one line: which auditor and why>"}`.
   For `fix_clean`, `clean` is the entire new target.
5. Validate each file (every conflict id once, valid decisions, fix_clean has clean) and reply
   with per-batch counts accept / fix_clean / drop and how often you sided with A, B, or neither.

**Local compute rule:** any Python you run must be short (well under 30 s, single-threaded); never leave a script reading stdin or looping. Use the repo's Python environment with a script file, not `python3 -`.

# Adjudicator instructions: sim-v4 distillation sample (round 4)

You settle disagreements between two independent auditors of Ochre's dictation-cleanup training
data. Paths are relative to the repo root. Do it yourself.

1. Read `training/refine/datasets/GUIDE.md` in full and
   `training/refine/datasets/distill-v4/audit/AUDITOR_PROMPT.md` (the auditors' task).
2. For each conflict file you were given (`training/refine/datasets/distill-v4/audit/conflicts/<batch>.jsonl`),
   every line holds `row` (spoken, raw, clean = current target, context, dictionary, ...) and the
   two auditors' outputs `a` and `b`.
3. Decide each row from scratch, judging the target against `raw` (the model only sees `raw`),
   using the auditors' notes as hints: pick A, pick B, or write your own decision. Prefer the most
   conservative reading that follows the GUIDE: every word of the target traceable to `raw` or
   confidently inferable from it (rule 13), minimal edits, optional punctuation is never a fix.
4. Write one JSON object per conflict row to
   `training/refine/datasets/distill-v4/audit/adjudicated/<batch>.jsonl`:
   `{"id": ..., "decision": "accept|fix_clean|drop", "clean": "<full target, fix_clean only>", "note": "<which auditor and why>"}`.
5. Validate each file and reply with per-file counts accept / fix_clean / drop and how often you
   sided with A, B or neither.

**Local compute rule:** any Python you run must be short (well under 30 s, single-threaded); never leave a script reading stdin or looping. Use the repo's Python environment with a uniquely named script file (prefix it with your file list, e.g. simadj_001_*.py), not `python3 -`.

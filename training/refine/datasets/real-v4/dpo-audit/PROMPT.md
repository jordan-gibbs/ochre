# DPO pair check (round 4, Phase E)

Paths are relative to the repo root. Do the work yourself.

1. Read `training/refine/datasets/GUIDE.md` in full (rules 1-13).
2. Read `training/refine/datasets/real-v4/dpo-audit/sample-v1.jsonl` (150 rows). Each row has the
   ASR `raw`, a `chosen` cleanup (an audited training target) and a `rejected` cleanup (a model's
   output), plus context/dictionary.
3. For every row, judge independently: is `rejected` clearly worse than `chosen` under the GUIDE?
   Verdicts: `worse` (rejected breaks a rule chosen keeps: invents or changes content, drops
   content, misses a self-correction, leaves fillers, wrong misrecognition guess, ...),
   `equal` (both acceptable; differences are optional style, e.g. "km" vs "kilometers"),
   `better` (rejected is actually the better cleanup), `both_bad` (chosen is also wrong).
4. Write one JSON line per row, same order, to
   `training/refine/datasets/real-v4/dpo-audit/<AUDITOR>.jsonl`:
   `{"id": ..., "verdict": "worse|equal|better|both_bad", "note": "<short>"}`.
5. Reply with only the verdict counts and up to 3 notes.

**Local compute rule:** any Python you run must be short (well under 30 s, single-threaded); never leave a script reading stdin or looping. Use the repo's Python environment with a script file, not `python3 -`.

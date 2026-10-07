# Judge task (protocol v4)

Paths are relative to the repo root. Do the work yourself
(no sub-agents, no code execution needed beyond reading/writing files).

1. Read `tools/eval/JUDGE_V4.md` (the protocol) and `training/refine/datasets/GUIDE.md` (rules
   1-13 and the recoverability rule) in full.
2. Read your packet file `tools/eval/out/judge-v4/<run>/packets/<packet>.jsonl`. Grade every
   letter of every row, independently, blind (you never learn which model wrote what; do not
   look for KEY.json or any other file in that folder).
3. Write your verdicts as JSON Lines, one line per packet row in packet order, to
   `tools/eval/out/judge-v4/<run>/verdicts/<packet>.jsonl`, in the format JUDGE_V4.md gives.
4. Check the file parses and has one line per row with a grade for every letter. Reply with
   only: rows graded, passes / fails, and the 3 most common fail types.

**Local compute rule:** any Python you run must be short (well under 30 s); never leave a script reading stdin or looping.

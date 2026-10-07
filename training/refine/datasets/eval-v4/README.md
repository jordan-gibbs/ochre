# eval-v4 (held-out, never trained on)

Audited real-audio rows from the eval-only writers (scripts-v2/E1..E13, reserved names and
topics), rendered TTS -> augmentation -> Parakeet -> strip_hesitations, two auditors + an
adjudicator (rubric v5/v6). Built by tools/eval/build_eval_v4.py.

| category | scripted | kept |
|---|---:|---:|
| ai_prompt | 60 | 41 |
| already_clean | 50 | 37 |
| casual | 60 | 50 |
| filler_contrast | 80 | 61 |
| long | 100 | 42 |
| misrecognition | 60 | 37 |
| numbers | 70 | 51 |
| self_correction | 80 | 64 |
| technical | 50 | 33 |
| terminal | 40 | 19 |
| trailing_off | 50 | 34 |
| voice_command | 50 | 34 |
| work | 60 | 48 |
| **total** | 810 | 551 |

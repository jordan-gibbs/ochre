# real-v4 (round-4 training data)

- round-3 audited real-audio rows: 2482
- round-4 audited real-audio rows: 3661
- removed as near-duplicates of an eval row: 6

Round-4 rows by category:

- self_correction: 464
- filler_contrast: 385
- misrecognition: 324
- long_dictation: 310
- numbers: 307
- trailing_off: 294
- minimal_edit: 241
- voice_command: 218
- misrecognition_context: 214
- ai_prompt: 166
- work_mixed: 149
- casual: 143
- casual_multiclause: 136
- technical_terminal: 128
- self_correction_long: 126
- work_detailed: 56

sim-v4 distillation rows: 24193 (removed 25 near-duplicates); writers {'codex/gpt-6-astra': 16659, 'claude/sonnet': 7534}

- self_correction: 4194
- filler_contrast: 3671
- numbers: 2488
- ai_prompt: 2279
- trailing_off: 2045
- casual: 1870
- voice_command: 1742
- minimal_edit: 1537
- long_dictation: 1508
- technical_terminal: 1401
- misrecognition: 431
- misrecognition_context: 278
- work_mixed: 178
- casual_multiclause: 176
- self_correction_long: 106
- work_detailed: 77
- tech: 48
- sc: 32
- fc: 27
- ai: 17
- num: 17
- min: 15
- to: 15
- long: 14
- vc: 14
- cas: 13

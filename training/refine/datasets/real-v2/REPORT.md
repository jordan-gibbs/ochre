# real-v2 pipeline report (4000 rows of 4000 selected scripts)

| TTS backend | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |
|---|---:|---:|---:|---|---:|---:|
| piper | 2213 | 0.115 | 0.150 | 5404 / 2713 / 1454 | 35% | 49% |
| kokoro | 1425 | 0.080 | 0.105 | 1527 / 1539 / 1243 | 20% | 29% |
| sapi | 217 | 0.047 | 0.076 | 194 / 66 / 135 | 17% | 22% |
| qwen | 101 | 0.098 | 0.153 | 144 / 81 / 45 | 28% | 36% |
| qwen_clone | 44 | 0.046 | 0.044 | 33 / 21 / 14 | 5% | 11% |

| recognizer | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |
|---|---:|---:|---:|---|---:|---:|
| parakeet-tdt-0.6b-v3-int8 | 3928 | 0.098 | 0.129 | 7250 / 4399 / 2850 | 28% | 40% |
| whisper-large-v3-turbo | 72 | 0.055 | 0.092 | 52 / 21 / 41 | 14% | 12% |

| audio | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |
|---|---:|---:|---:|---|---:|---:|
| augmented | 3225 | 0.098 | 0.132 | 6010 / 3569 / 2390 | 29% | 40% |
| clean | 775 | 0.093 | 0.114 | 1292 / 851 / 501 | 27% | 38% |

| background noise | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |
|---|---:|---:|---:|---|---:|---:|
| none | 1824 | 0.092 | 0.118 | 3157 / 1824 / 1217 | 26% | 37% |
| snr 20-30 | 1088 | 0.098 | 0.128 | 2061 / 1207 / 831 | 30% | 41% |
| snr 12-20 | 723 | 0.100 | 0.144 | 1295 / 899 / 546 | 29% | 44% |
| snr 5-12 | 365 | 0.113 | 0.152 | 789 / 490 / 297 | 36% | 42% |

| reverb | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |
|---|---:|---:|---:|---|---:|---:|
| no room IR | 2045 | 0.100 | 0.129 | 3574 / 2747 / 1392 | 29% | 39% |
| room IR | 1955 | 0.095 | 0.128 | 3728 / 1673 / 1499 | 28% | 40% |

| mic chain | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |
|---|---:|---:|---:|---|---:|---:|
| laptop | 1288 | 0.095 | 0.126 | 2381 / 1311 / 868 | 27% | 39% |
| clean | 775 | 0.093 | 0.114 | 1292 / 851 / 501 | 27% | 38% |
| headset | 645 | 0.107 | 0.155 | 1201 / 959 / 516 | 32% | 43% |
| usb | 485 | 0.098 | 0.128 | 864 / 601 / 393 | 29% | 42% |
| none | 475 | 0.096 | 0.129 | 866 / 476 / 310 | 28% | 38% |
| phone | 332 | 0.094 | 0.121 | 698 / 222 / 303 | 27% | 39% |

| condition (combined) | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |
|---|---:|---:|---:|---|---:|---:|
| clean | 775 | 0.093 | 0.114 | 1292 / 851 / 501 | 27% | 38% |
| room+snr20-30+laptop | 132 | 0.089 | 0.107 | 224 / 95 / 75 | 23% | 35% |
| room+laptop | 131 | 0.095 | 0.127 | 211 / 103 / 78 | 22% | 35% |
| room+snr20-30+laptop+speed | 125 | 0.098 | 0.127 | 230 / 123 / 90 | 34% | 43% |
| room+laptop+speed | 123 | 0.091 | 0.133 | 246 / 86 / 57 | 33% | 39% |
| room+snr12-20+laptop | 95 | 0.094 | 0.136 | 166 / 114 / 80 | 29% | 51% |
| snr20-30+laptop | 94 | 0.085 | 0.128 | 163 / 115 / 67 | 26% | 39% |
| laptop+speed | 93 | 0.077 | 0.090 | 128 / 62 / 49 | 14% | 23% |
| room+snr12-20+laptop+speed | 84 | 0.097 | 0.140 | 180 / 102 / 73 | 26% | 44% |
| room+snr20-30+headset | 78 | 0.090 | 0.115 | 159 / 88 / 67 | 29% | 40% |
| laptop | 73 | 0.084 | 0.106 | 144 / 62 / 52 | 26% | 42% |
| snr20-30+laptop+speed | 67 | 0.087 | 0.106 | 139 / 72 / 49 | 27% | 43% |
| room+headset | 57 | 0.110 | 0.142 | 86 / 83 / 30 | 25% | 37% |
| snr12-20+laptop+speed | 56 | 0.113 | 0.118 | 90 / 129 / 37 | 34% | 39% |
| room+headset+speed | 54 | 0.101 | 0.120 | 121 / 58 / 56 | 33% | 43% |

| voice (most frequent 15) | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |
|---|---:|---:|---:|---|---:|---:|
| sapi:zira | 109 | 0.047 | 0.074 | 102 / 35 / 75 | 13% | 17% |
| sapi:david | 108 | 0.048 | 0.079 | 92 / 31 / 60 | 21% | 27% |
| qwen3tts-clone:clone-a | 44 | 0.046 | 0.044 | 33 / 21 / 14 | 5% | 11% |
| kokoro:bf_isabella | 39 | 0.099 | 0.116 | 41 / 55 / 55 | 26% | 33% |
| kokoro:am_michael | 38 | 0.077 | 0.082 | 35 / 53 / 22 | 21% | 26% |
| kokoro:am_eric | 37 | 0.149 | 0.128 | 48 / 136 / 50 | 22% | 32% |
| kokoro:af_river | 36 | 0.066 | 0.114 | 38 / 34 / 23 | 25% | 36% |
| kokoro:am_echo | 36 | 0.114 | 0.162 | 60 / 56 / 42 | 33% | 39% |
| kokoro:af_sarah | 36 | 0.036 | 0.052 | 19 / 13 / 13 | 6% | 11% |
| kokoro:bm_daniel | 33 | 0.082 | 0.070 | 25 / 37 / 25 | 6% | 18% |
| kokoro:am_puck | 32 | 0.064 | 0.105 | 33 / 21 / 12 | 16% | 31% |
| kokoro:af_heart | 31 | 0.065 | 0.073 | 26 / 25 / 33 | 19% | 26% |
| kokoro:am_fenrir | 30 | 0.052 | 0.106 | 23 / 12 / 10 | 10% | 20% |
| kokoro:af_sky | 30 | 0.059 | 0.103 | 21 / 19 / 22 | 17% | 17% |
| kokoro:am_adam | 29 | 0.038 | 0.037 | 19 / 12 / 23 | 7% | 14% |

## Audit pass 1

- severity: ok 1997 (50%), suspect 1129 (28%), check 874 (22%)
- recoverability suspects (a key token of the target was said but is absent from raw): 1586 rows (40%); by kind: {'content': 2782, 'number': 193, 'name': 463, 'dictionary': 58}
- target tokens not found in spoken either (script/format): 145 rows
- flags: {'raw_too_short': 55, 'empty': 33, 'intended_much_shorter': 19, 'raw_too_long': 6, 'repeat_loop_2gram': 3, 'repeat_loop_1gram': 3, 'odd_speaking_rate': 3, 'non_latin_script': 2}

## Throughput

- audio rendered and recognized: 960.5 min for 4000 clips (160405 spoken words, 0.36 s per word)
- TTS kokoro: 2.3x realtime per worker
- TTS piper: 3.3x realtime per worker
- TTS qwen: 0.5x realtime per worker
- TTS qwen_clone: 0.2x realtime per worker
- TTS sapi: 23.6x realtime per worker
- recognition parakeet-tdt-0.6b-v3-int8: 5.9x realtime per worker
- recognition whisper-large-v3-turbo: 5.6x realtime per worker
- wall time (all jobs in parallel): 79.0 min -> **50.6 clips/min**, 12.2x realtime overall
- projection for all 4000 scripts (160405 spoken words, ~16.0 h audio), scaled by spoken words: **1.3 h** (model loading is a fixed ~2 min, so this is slightly pessimistic)

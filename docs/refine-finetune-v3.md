# Refinement fine-tune v3: results

2026-10-05. Round 3 of the cleanup model. All training ran on Daytona RTX 5090 spot sandboxes
(`tools/cloud/owf_daytona.py train`, see `docs/cloud-compute.md`); nothing heavy ran locally.

## What changed in the data

- **GUIDE rule 5 revised:** meaningless discourse fillers ("so", "okay so", "like", "you know",
  "I mean", "basically", "literally", "right") are removed in every context, casual included.
  Meaningful uses stay ("so that", "so" = therefore, "I like", "looks like", "what kind of").
- **Relabel:** the 550 existing targets containing those words were relabeled by Opus, checked by
  an independent Sonnet auditor, and the disagreements were adjudicated by Opus. 158 targets changed.
- **New data:** 600 scripts (W9 casual chat, W10 work speech with verbal tics, W11 fluent
  minimal-edit dictation) were rendered by TTS and transcribed by Parakeet in the cloud, then went
  through the same two-auditor + adjudicator audit (rubric v4). 434 rows were kept.
- **Train set** `datasets/real-v3/final/train.jsonl` has 2,486 pairs. The test sets are:
  - `heldout.jsonl`: the 150 v2 held-out rows, relabeled.
  - `heldout-new.jsonl`: 60 held-out W9–W11 rows.
  - `casual-probe.jsonl`: 12 hand-written casual sentences.

## Blind side-by-side judgement

For each dictation, an Opus judge saw every model's output under shuffled letters. Model names
were hidden, and 222 dictations were graded × 6 models. The files are in `tools/eval/out/judge-v3/`
(packets, `KEY.json`, `verdicts.json`).

| run | held-out (150) | new (60) | casual (12) | **all (222)** | harmful fails | acted | median ms (5090) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 recipe, v2 data (`v2-ref`) | 69% | 33% | 8% | 56% | 16% | 0 | 80 |
| v3-2b (2 ep, r16) | 76% | 78% | 92% | 77% | 15% | 0 | 67 |
| v3-2b-e3 (3 ep) | 79% | 68% | 92% | 77% | 13% | 0 | 65 |
| v3-2b-r32 (r32/α64) | 74% | 72% | 92% | 74% | 15% | 0 | 66 |
| **v3-4b** (2 ep, r16) | **86%** | **82%** | 92% | **85%** | **11%** | 0 | 145 |
| v3-0.8b (3 ep) | 69% | 67% | 75% | 68% | 17% | 0 | 59 |

Paired comparisons count the rows where only one of the two models passes:

| comparison | only the first passes | only the second passes |
|---|---:|---:|
| v3-4b vs v3-2b | 28 | 11 |
| v3-2b vs v2-ref | 55 | 8 |
| v3-0.8b vs v2-ref | 47 | 20 |

Deterministic WER vs reference:

| run | held-out | new | casual | synth-v1 |
|---|---:|---:|---:|---:|
| v2-ref | 0.037 | 0.098 | 0.177 | 0.036 |
| v3-2b | 0.033 | 0.053 | 0.012 | 0.040 |
| v3-4b | 0.030 | 0.059 | 0.012 | 0.024 |
| v3-0.8b | 0.039 | 0.051 | 0.040 | 0.037 |

## Takeaways

- The casual-filler problem is fixed. "hey man so, uh, can we, um, get that going?" now becomes
  "Hey man, can we get that going?" with every v3 model of 2B or larger.
- **4B is the best model:** 85% vs 77% for the 2B, with fewer harmful failures. It costs 2.7 GB
  and about 2× the latency (145 ms on a 5090), which is well inside the 1 s budget on any modern GPU.
- In the 2B sweep, extra epochs and a higher rank didn't help overall, so the v3-2b recipe stays.
- **0.8B with v3 data beats the v2 2B (68% vs 56%)** at 529 MB, which makes it the CPU-only
  candidate. Its CPU latency is not measured yet.
- The remaining harmful failures are mostly misrecognized words passed through or "fixed" wrongly
  (e.g. "Node" → "we"), and dropped clauses.

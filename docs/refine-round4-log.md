# Refine round 4: working log

Working log for round 4 of the cleanup-model fine-tune (plan: measurement first, failure mining,
targeted real-audio data, ASR-noise-simulated distillation data, optional DPO, sweep, judge,
ship only on a significant win). Newest entries at the bottom of each phase.

## Phase A: measurement

- 2026-10-05: GUIDE rule 13 added (misrecognized words: fix only what is inferable, otherwise
  keep the raw word; never a guess, a synonym or a deletion) with a changelog; rubric v5 carries
  it as R13. It restates the recoverability rule as a target rule, so targets audited under v4
  stay valid.
- Eval-only script files `scripts-v2/E1..E13` (`is_eval_file` in common.py): rendered and
  audited only when named, never in the main batch order (`packets.py --set eval` ->
  `batch-e-XXX`), exported separately (`merge_audits.py export --eval` ->
  `real-v2/final/eval-v4-real.jsonl`), never in `real-v2.jsonl`.
- 13 eval writers (Opus, separate from training writers) with reserved names and topic domains
  (`scripts-v2/briefs/eval-v4.md`): 760 scripts in 13 categories.
- `owf_daytona.py eval`: existing GGUFs (downloaded from the private HF repos inside the
  sandbox, or uploaded) through `eval_gguf.py` on any eval sets (`sandbox/eval_job.sh`).
- Judging protocol v4 (`tools/eval/JUDGE_V4.md`, `build_judge_v4.py`, `tally_judge_v4.py`):
  side by side, identical outputs merged under one letter, letters and row order shuffled
  independently per judge, 3 Opus judges, majority vote, Fleiss' kappa, paired bootstrap CIs.
- History eval tooling: `tools/eval/export_history.py` + `docs/history-eval.md`. Not run.
- eval-v4 render: 760/760 rows (Piper/Kokoro, Parakeet v3 int8, augmentation as in v2/v3). Raw
  quality is realistic and harsh: auto-audit flagged about a third as suspect.
- eval-v4 audit (rubric v5, A = Opus, B = Sonnet, 16 batches): decision agreement 0.74-0.92,
  kappa 0.58-0.87; 447 agreed keeps, 197 agreed drops, 116 conflicts to the Opus adjudicator.
- Finding: long dictations were dropped wholesale (E7: 30/50 agreed drops) because the fix
  limits ("5 words or 2 places") were absolute. Rubric **v6**: limits scale per 40 words of
  raw. The 47 long rows (>= 60 words) dropped only for `unrecoverable_meaning` get a full
  two-auditor re-audit under v6 (`packets.py --set ids` -> batch-e-rx-001/002); later batches
  win in `merge_audits.py apply`.
- Recognizer artefact seen repeatedly: Parakeet appends a lone "Yeah." on trailing noise; the
  rubric already drops these as `hallucination`.
- Judging plan: baseline failure mining uses one blind Opus judge per row (v3 models only);
  the final comparison (v3 + v4 side by side) uses the full 3-judge protocol, so every reported
  number comes from one consistent set of packets.

## Phase C: targeted data (wave 1)

- 40 Sonnet writers, `briefs/train-v4.md` + `train-v4-wave1.json`: W12-W51, 3,800 scripts:
  self_correction 600, filler_contrast 500, long_dictation 300, misrecognition 500, numbers
  300, voice_command 300, trailing_off 300, ai_prompt 200, casual 200, work_mixed 200,
  technical_terminal 200, minimal_edit 200. Writers skew short (most files short of the
  41-150-word bands); categories are as planned.
- Daytona caps sandboxes at 16 vCPU (a 32 vCPU request is refused), so the render runs as three
  16 vCPU sandboxes in parallel.

## Phase D: distillation data (generation routed to Codex)

- Mid-round change: bulk generation goes to the OpenAI Codex CLI (`codex exec`,
  model gpt-6-astra, reasoning effort medium, workspace-write sandbox, no network); auditing,
  adjudication and judging stay on Claude (an independent model family checks Codex-written
  targets). Runner: `tools/refine-data/codex_gen.py` (parallel jobs, one JSONL per job, schema
  validation); brief `scripts-v2/briefs/codex-distill.md` (GUIDE pointer, exact schema, 10 gold
  rows from audited round-4 scripts); assignments `briefs/codex-assign.json` (64 jobs, distinct
  name/topic seeds and emphasis). Every Codex row carries `writer: "codex/gpt-6-astra"`.
- Before the switch: 16 Claude (Sonnet) distillation files D001-D016 (4,000 rows,
  `writer` absent = claude/sonnet).
- Codex smoke job C001: 60 rows in 6 min, schema-valid; spot-checked 10 rows: correct labels.
- Codex wave: C002-C064, 300 rows each (18,900), 12 jobs in parallel.
- Simulator: `tools/refine-data/asr_noise_sim.py` (fit on 85% of the real Parakeet pairs;
  held-out WER 0.137 real vs 0.125 sim, sub/del/ins and punctuation densities within ~10%).
  `tools/refine-data/distill_v4.py` builds the pairs and keeps targets recoverable (rule 13):
  near-spelling substitutions keep the intended word, a misheard name not in the dictionary
  puts raw's spelling in the target, dropped function words are removed; any other lexical
  substitution, any dropped or inserted content word or changed digit string rejects the sample
  (another seed is tried). A first version propagated raw's word into the target for every
  substitution and produced targets nobody would write; lexical-error handling is left to the
  audited real-audio data.
- The first sim-vs-real check (v3-4b on 400 held-out pairs, real raw vs simulated raw of the
  same clip) was confounded: those W1-W11 rows were in v3's training set (real WER 0.013 vs sim
  0.102). Redone on W12-W51 (`asr_noise_sim.py --fresh`, `datasets/asr-sim/fresh-pairs.jsonl`),
  which neither the fit nor v3 saw.

## Phase F: training (real-only runs)

- `real-v4/train-real.jsonl`: 6,143 audited real-audio rows (2,482 round 3 + 3,661 round 4;
  6 near-duplicates of eval rows removed by `leakcheck.py`, exact + 4-gram containment >= 0.6).
  Every run also has all four eval sets in `forbid_files` (train.py aborts on any overlap).
- Launched in parallel, v3 recipe (LoRA r16/a32, lr 2e-4, cosine): v4-4b (2 ep), v4-2b-real
  (2 ep), v4-0.8b-real (3 ep). Distillation runs for 2B / 0.8B follow once sim-v4 is audited.
- Quick mechanical numbers (word error vs target, single eval run; the 3-judge protocol decides):
  eval4 v3-4b 0.040 -> v4-4b 0.033, v3-2b 0.061 -> v4-2b-real 0.042, v3-0.8b 0.068 ->
  v4-0.8b-real 0.051. On the legacy 150-row held-out set the real-only models are about even
  with v3 (4b 0.030 -> 0.024, 2b 0.034 -> 0.040, 0.8b 0.039 -> 0.041).

## Phase E: DPO (2B first)

- Pairs (`tools/refine-data/build_dpo_v4.py`): v3-2b ran over all 6,143 training rows in the cloud
  (`eval-1005-155720-f22a`). chosen = the audited target; rejected = the v3-2b output when a
  mechanical check flags a clear failure (an invented content word, at least 2 dropped content
  words, or a missed correction). Number renderings, units, joins/splits and inflections are
  never counted as failures. The check is mechanical and no judge is involved.
- Two auditors (Opus, Sonnet) checked a 150-pair sample from the first filter version. Both
  called rejected "worse" on 124 pairs and they agreed on 135 of 150 verdicts. Most "equal"
  verdicts were number or unit renderings and joins; the filter was tightened for those.
  With the tightened filter, 82 of the 89 surviving sample pairs were "worse" by both auditors
  (92%). Audited pairs override the filter. Final set: 467 pairs.
- Run v4-2b-real-dpo: the v4-2b-real SFT recipe, then DPO on the pairs (beta 0.1, lr 2e-5,
  2 epochs, + 0.2 x chosen NLL), with the reference = the SFT policy. The first attempt was lost to a
  spot eviction at 90% of SFT, so the job was re-run.
- Codex BOM: 5 Codex files started with a UTF-8 BOM. All JSONL readers now use utf-8-sig, and
  codex_gen.py strips the BOM on validate. `scripts-v2/W27.jsonl` had been truncated to 0 bytes
  at 15:34 by an unknown writer and was restored from git. The only effect was the `cat` label
  of 25 round-4 rows in train-real (training targets unaffected).

## Phase D (cont.): sim-v4 set and its checks

- Codex wave finished: 63 jobs C002-C064 x 300 rows plus the 60-row smoke job C001 = 18,960
  scripts. Each job took 13-19 min, 12 ran in parallel (~5 h wall clock). 64 `codex exec`
  invocations in all. Every file passed schema validation; 3 flagged "hesitation in
  intended" rows were false positives ("mm" = millimetres).
- Build (`distill_v4.py build --seeds 1`, run locally at Idle priority, about 4 min): 28,115
  scripts (D 4,000 + C 18,960 + W12-W64), 747 duplicate scripts, 24,218 samples. 27,066 tries
  were rejected as unrecoverable, and 1,316 targets took a raw spelling variant, inflection or
  name. After the eval leak filter: 24,193 rows (Codex 16,659 / Claude 7,534).
- Sampled audit (`sim_audit.py`): a stratified 10% (2,434 rows) went to two auditors (A Opus,
  B Sonnet) with an Opus adjudicator. Agreement was 96.5% and 99 conflicts were adjudicated.
  Final: accept 93.4%, fix 5.8%, drop 0.8%. Codex and Claude writers came out equal (93.4% vs
  93.2% accept). The defects that remain are mostly rule-13 guesses, where the target kept the
  intended word but the simulated raw had a different real word ("branch" vs "bench",
  "seven" vs "several"). There were also lost digits restored in the target and spelling
  variants, which are now propagated. Writer bugs that were fixed: literal "\n" in 39 Claude
  scripts, and in 10 Codex rows (C014, C061) a non-UTF-8 console turned line breaks, "°" and "£"
  into "?" (those rows are skipped). Two small Claude slices (self_correction_long,
  work_detailed; 183 rows) had fix+drop above 25% and were removed. Training file:
  `real-v4/sim-audited.jsonl`, 23,991 rows.
- v3-4b agreement: the full 24k-row run was lost to a spot eviction. It was re-run on a
  6,026-row subset (all 2,426 audited rows + 3,600 random rows).
- Distillation training runs: v4-2b (2B, 1 epoch over real x2 + sim-audited) and v4-0.8b (0.8B,
  1.5 epochs), so the real rows are seen 2x / 3x as in the real-only runs.
- v3-4b agreement (`tools/refine-data/sim_agree.py`, cloud run `eval-1005-174145-021c`, 6,026
  rows): v3-4b's output matches the simulated target word for word on 78.3% of rows (mean word
  error 0.033). Codex rows 77.8%, Claude rows 79.2%. Agreement is lowest on trailing_off
  (74%), filler_contrast (71%) and long dictations (65%, where the mean word error is only
  0.007). On the 2,426 audited rows, disagreement does predict audit defects: rows where v3-4b
  agrees have a 2.2% defect rate (fix or drop), rows where it disagrees have 22.6%, against
  6.6% overall. A filter on agreement would remove 22% of the rows, and 77% of the removed rows
  are good targets, mostly cases v3 gets wrong. Those are the cases distillation should
  teach, so agreement is used as a check, not a filter.

## Phase F: final judging and promotion

- Judge run `tools/eval/out/judge-v4/final/`: 773 rows (eval-v4 551 + held-out 150 + new 60 +
  casual 12) x 9 models (v3-4b/2b/0.8b, v4-4b, v4-2b-real, v4-2b-real-dpo, v4-0.8b-real, v4-2b,
  v4-0.8b). Each judge saw an average of 3.7 distinct outputs per row. There were 93 Opus judge
  packets (3 judges x 31 packets of 25 rows) and no incomplete grades. Inter-judge Fleiss kappa
  was 0.942 for pass and 0.932 for harmful.
- Pass rates (all rows): v3-4b 74.8 -> v4-4b 79.8; v3-2b 64.7 -> v4-2b 71.0 (real-only 69.2,
  DPO 66.0); v3-0.8b 55.6 -> v4-0.8b 63.8 (real-only 62.1). Harmful: 14.9 -> 9.8, 22.6 -> 14.0,
  25.1 -> 17.3.
- Paired 95% CIs vs v3: 4B +5.0 [+2.3, +7.8], 2B +6.3 [+3.2, +9.4], 0.8B +8.2 [+5.2, +11.3].
  Harmful went down at every size. **All three sizes promoted**: v4-4b, v4-2b (distillation)
  and v4-0.8b (distillation). DPO was not promoted (-3.2 [-5.6, -1.0] vs the same SFT recipe).
  Details: `docs/refine-finetune-v4.md`.

## Phase G: ship

- Uploaded `ochre-refine-{4b,2b,0.8b}-v4-Q4_K_M.gguf` to the private repos
  `polonuim210/ochre-refine-{4b,2b,0.8b}`. The v3 files are kept and every repo is still private.
  Model cards were rewritten for v4. LFS sha256 values:
  4b `baed73b8...c485` (2,708,804,480 B), 2b `096fd42f...5683` (1,274,396,640 B),
  0.8b `169e5c93...a1bc` (529,297,376 B).
- `crates/ochre-refine/src/local/install.rs` now points at the v4 files, and its tests were
  updated. `cargo test -p ochre-refine -j 4` passes 37 tests (4 ignored). The README numbers
  and `docs/refine-finetune-v4.md` are updated.

## Volume

- Daytona: renders, evals and 8 training runs, three of them spot-evicted and re-run.
- Codex: 64 `codex exec` jobs (gpt-6-astra, effort medium), 18,960 scripts.
- Claude subagents, approximate: 82 writers (eval, train waves, distillation), about 200
  real-audio auditor runs (258 auditor files) plus about 40 adjudicators, 50 sim-sample auditors
  plus 4 adjudicators, 2 DPO-pair auditors, and 124 judge packets (31 baseline + 93 final).
  That is roughly 550 agent runs.

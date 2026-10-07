# Wake word training

`pipeline.py` trains a wake word head for any word: `--word <w>` reads `configs/<w>.yaml`. It grew
out of an earlier wake word training pipeline. The head runs on the frozen
openWakeWord front end; the runtime contract is in [`docs/wakeword.md`](../../docs/wakeword.md).
Output goes to `assets/wake/<word>.onnx` plus `<word>.json` (threshold table and stats).

The shipped model is **"transcribe"**. Results are below.

## Setup (once)

Hardware: an NVIDIA GPU is strongly recommended. The reference machine is an RTX 5070 Ti (16 GB,
Blackwell sm_120) with CUDA 12.8 wheels, using about 3 GB of VRAM during TTS. It also needs ~8 GB of
RAM for training and ~12 CPU cores for augmentation. On CPU only, TTS and training are roughly
10–20× slower.

```powershell
# separate venv, never the app venv (gitignored: .venv-*)
uv venv --python 3.12 training/wakeword/.venv-train
uv pip install --python training/wakeword/.venv-train/Scripts/python.exe torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python training/wakeword/.venv-train/Scripts/python.exe "livekit-wakeword[train] @ git+https://github.com/livekit/livekit-wakeword@95448a7559c453fcd87645bd67b247ffb45f85b0" onnx onnxscript pyyaml soundfile tqdm
# espeak-ng (phonemes for Piper): Windows `winget install eSpeak-NG.eSpeak-NG`, macOS `brew install espeak-ng`,
# Linux `apt install espeak-ng`
```

**Word-independent data** (~2.7 GB) is shared by every word and never committed. Point the
pipeline at it with `--data DIR`, or `$OWF_WAKE_DATA`, or put it in `training/wakeword/data/lkw`
(gitignored):

```
<data>/piper/en-us-libritts-high.pt (+ .json)     Piper VITS LibriTTS, 904 speakers (livekit-wakeword setup)
<data>/rirs/**.wav                                 MIT room impulse responses
<data>/backgrounds/**.wav                          MUSAN noise (~560 MB)
<data>/features/validation_set_features.npy        openWakeWord validation features (~11 h, eval only)
<data>/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy   general negatives (17 GB), or
<data>/features/acav100m_partial_16bit.npy         a contiguous ~217 h slice of it (what we use)
```

`livekit-wakeword setup` downloads all of these. To reuse an existing copy
read-only: `OWF_WAKE_DATA=<path to an existing livekit-wakeword data dir>`.

## Run

```powershell
$env:OWF_WAKE_DATA = "...\data\lkw"
# full build: generate -> features -> train -> export -> eval -> install (stages resume / skip finished work)
training/wakeword/.venv-train/Scripts/python -m training.wakeword.pipeline all --word transcribe
# offline check through the runtime's detector, streamed 80 ms at a time (app venv)
uv run python -m training.wakeword.verify_runtime --word transcribe --write-meta
# live mic check + threshold tuning (app venv; you run it)
uv run python -m training.wakeword.eval_live --word transcribe
```

To train a new word, copy `configs/transcribe.yaml` to `configs/<word>.yaml`. Then set the wake
spellings (check them with `espeak-ng --ipa -v en-us "<spelling>"`), the near-misses and the
follow-on speech, and run `all --word <word>`.

### Your own voice (the biggest single improvement)

```powershell
uv run python -m training.wakeword.record_samples --word transcribe --dry-run   # see the prompts first
uv run python -m training.wakeword.record_samples --word transcribe             # ~40 takes + 90 s room audio, ~6 min
training/wakeword/.venv-train/Scripts/python -m training.wakeword.pipeline retrain --word transcribe   # ~10 min
```

Takes go to `data/real_positive_<word>/` (augmented ×60, every fifth held out) and room audio to
`data/real_room_<word>/` (negatives). The retrain prints `realrec`, the recall on your held-out
takes.

### Files

```
pipeline.py        generate | features | train | export | eval | install | all | retrain  [--word W] [--data DIR]
configs/<w>.yaml   livekit trainer settings + `wake:` section (spellings, near-misses, follow-on speech, counts)
verify_runtime.py  streams held-out clips through openwhisprflow.wake.detector (app venv)
eval_live.py       live mic scores through the same detector (app venv)
record_samples.py  record your takes + room audio
make_fixture.py    tests/fixtures/wake/ numeric fixture for runtime ports
output/<w>/        (gitignored) clips/, features/, trainer inputs, <w>.pt/.onnx, <w>_eval.json, runtime_verify.json
```

## What the model is

- **Front end (frozen):** openWakeWord `melspectrogram.onnx` + `embedding_model.onnx`, giving one
  96-d embedding per 80 ms. Features are extracted at **int16 scale**, the same way the runtime feeds
  them. (livekit-wakeword itself feeds [-1, 1] floats, which shifts the log-mel by ~9 units.)
- **Head:** livekit-wakeword `conv_attention` (medium). Input `(batch, 16, 96)` named `embeddings`,
  output `(batch, 1)` sigmoid `score`.
- **Gate:** threshold, 2 consecutive frames and a 1.5 s cooldown. Then the transcript check: the
  session's first words must be the wake word (`text/commands.py`), or the session is dropped as a
  false wake.

## "transcribe": what counts as positive and negative

espeak-ng (en-us): `transcribe` → tɹænskɹˈaɪb, `tran-scribe` → tɹˈænskɹˈaɪb, `trans-cribe` →
tɹˈænzkɹˈaɪb, `trans scribe` → tɹˈænz skɹˈaɪb.

- **Positive (trained):** those four spellings (weights 6/2/1/1), in these contexts:
  - alone;
  - followed by speech, 60% of positives. That is either dictation openings ("Transcribe, hey Sarah,
    just wanted to check in") or the control words ("transcribe stop / send / done / cancel /
    scratch that"). Each training window ends 0–450 ms after the word, as the live gate sees it when
    it fires;
  - after a prefix in its own splits: "hey / okay / OK / so / hi / um" or a **finished dictated
    sentence** ("… see you tomorrow. Transcribe send"). In a hands-free session the control phrase
    always follows speech, so the 2 s window holds the tail of the previous sentence. Earlier
    models never saw speech before the word and managed only 10–30% detection with a prefix.
- **"transcribe" mid-sentence is still the word.** "I need to transcribe this" contains the wake
  word acoustically, and no acoustic model can tell it apart. It is therefore never a negative,
  since that would teach the model to reject the wake word itself. The transcript layer rejects it
  instead: a session's first phrase must *start* with the wake word, and control phrases only act at
  the *end* of a phrase.
- **Left out (ambiguous):** "transcribes" (…aɪbz) and "transcribed" (…aɪbd). They are the same
  sounds as "transcribe s(top/end)" and "transcribe d(one)", which are the control phrases
  themselves. Training them as negatives would suppress exactly the in-session hits that arm
  command detection.
- **Negative (near-misses, ~120 phrases + 1500 CMUdict neighbours):**
  - same ending with a different onset: describe, prescribe, subscribe, inscribe, ascribe, scribe;
  - same onset with a different ending: transcript(s), transcription(ist), transcribing, transcriber;
  - trans-X: transfer, translate, transport, transit, transcend, transform, trance;
  - -script words: manuscript, scripture, prescription, description, subscription;
  - rhymes and blends: tribe, bribe, vibe, "train scribe", "trans crib", "trance cry";
  - the control words alone, "<near-miss>, <control word>", other assistants' names, and fragments
    ("trans", "scribe", "crime").
  
  `pipeline.py` drops any negative whose phonemes equal a positive spelling.

## Results: "transcribe" v1 (synthetic data only)

`assets/wake/transcribe.onnx` (870 KB, sha256 `b941dada…ba37`) + `transcribe.json`. livekit
conv-attention (medium), 30k steps, 2 consecutive frames, 1.5 s cooldown, **threshold 0.45**.

Compute on the RTX 5070 Ti (shared with the desktop and other jobs, VRAM capped at 30%): total
**42 min**. TTS took 13.8 min (37.6k clips, ~40 clips/s), augmentation + features 10.1 min (14
processes), training 16.5 min, and export + eval under 1 min.

| metric (held-out speakers) | value at 0.45 |
|---|---|
| streamed detection through the runtime detector (`verify_runtime.py`, 300 clips per split) | **95.0%**: alone 99.1%, followed by speech 92.3%, followed by a control word 93.2% |
| streamed detection after "hey/okay/so/um" or a dictated sentence | **92.3%**: control word after a sentence 90.6% |
| near-miss clips that fire once when streamed | 3.0% (plain), 3.0% (after a prefix) |
| single-window recall (1000 augmented windows, SNR down to 3 dB) | 87.9% (prefixed 87.1%) |
| single-window near-miss false accept | 0.5% (prefixed 1.4%) |
| false wakes/hour, 10.7 h openWakeWord validation stream through the gate | **0.37** |

| thr | recall | prefix recall | near-miss FA | prefix near-miss FA | FP/h (gate) |
|---|---|---|---|---|---|
| 0.30 | 90.2% | 90.2% | 0.5% | 1.5% | 0.93 |
| 0.40 | 88.5% | 88.2% | 0.5% | 1.5% | 0.65 |
| **0.45** | **87.9%** | **87.1%** | **0.5%** | **1.4%** | **0.37** |
| 0.50 | 86.9% | 86.5% | 0.3% | 1.4% | 0.28 |
| 0.60 | 84.4% | 84.9% | 0.2% | 1.1% | 0.19 |
| 0.70 | 81.3% | 82.8% | 0.2% | 0.6% | 0.19 |
| 0.85 | 72.4% | 72.4% | 0.2% | 0.5% | 0.09 |

0.45 gives the best recall with at most 0.5 false wakes/h (`wake.max_fpph`). Use 0.60 for a
conservative setting (0.19/h, ~3 points less recall). For comparison, an earlier wake word head trained the same way
reached 93.3% / 90.7% streamed, 4.7% near-miss fires and 0.47 FP/h.

**What still fires (streamed near-misses):** "transcribing", "transcriber" (the model fires as the
word ends, before the suffix arrives), "grand scribe", "train scribe", "trance crime/cry" and
"transrapid". These are near-homophones. The session's false-wake check drops "Transcribing…" and
"Grand scribe…", but not "Transcriber…" or "Train scribe…".
**Misses** are mostly clips with a long period-pause after the word ("transcribe. stop", "tran-scribe.
let's grab lunch"). In those the word ends the window with a long silence after it, which the
training windows (ending 0–450 ms after the word) rarely show.

Next steps, in order of value:
1. Record your own voice (`record_samples.py`, then `retrain`).
2. Add hard negatives for "transcribing"/"transcriber" in long contexts.
3. Extend the window end to 0–800 ms after the word, to cover the long-pause misses.

Fixture for runtime ports: `tests/fixtures/wake/transcribe_sample.{wav,json}` ("Transcribe, hello
there. Transcribe send.", held-out speaker 16). It fires at frames 11 and 35, with peak 0.95.

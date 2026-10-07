# Wake word runtime spec ("transcribe")

This is the contract for implementing the hands-free wake word in the Rust app with the
[`ort`](https://crates.io/crates/ort) crate. It is precise enough to reproduce the Python
reference (`src/openwhisprflow/wake/{frontend,detector,handsfree}.py`,
`src/openwhisprflow/text/commands.py`) number for number. The model is trained by
`training/wakeword/` (see its README); the shipped files are `assets/wake/transcribe.onnx` and
`assets/wake/transcribe.json`.

Pipeline per 80 ms of 16 kHz mono audio:

```
mic (16 kHz mono) ─► 1280-sample frame ─► VAD gate ─► melspectrogram.onnx ─► 76×32 mel window
   ─► embedding_model.onnx ─► 96-d embedding ─► last 16 embeddings ─► transcribe.onnx ─► score
   ─► gate (threshold, 2 consecutive frames, 1.5 s cooldown) ─► WAKE
```

All three models together cost about 0.75 ms of CPU per 80 ms frame (one thread, measured on the
dev machine). With the VAD gate, a silent room costs only the VAD.

---

## 1. Model files

| file | source | sha256 | bytes |
|---|---|---|---|
| `melspectrogram.onnx` | `https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx` | `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f` | 1 087 958 |
| `embedding_model.onnx` | `https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx` | `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f` | 1 326 578 |
| `transcribe.onnx` | `assets/wake/transcribe.onnx` (ours, livekit conv-attention medium head) | `b941dadad7f36f5b001378e65f59c527f6c69addb357dd099f3a320a1ba5ba37` | 870 323 |

The two front-end models are openWakeWord's (Apache-2.0). They are byte-identical to the copies
inside the `openwakeword` and `livekit-wakeword` Python packages that training uses (checked
2026-10-04). Download them once into the app's model dir. Verify the size and sha256, write to
`<name>.partial` first, then rename (the same rule as every other model download).

Session options, which matter for an always-on listener: 1 intra-op thread, 1 inter-op thread,
sequential execution, and **spinning disabled** (`session.intra_op.allow_spinning = "0"`,
`session.inter_op.allow_spinning = "0"`). ORT's default spinning pool burns a core while idle. Use
CPU; the models are too small for a GPU to help.

## 2. Tensors

| model | input name | input shape / dtype | output name | output shape |
|---|---|---|---|---|
| melspectrogram | `input` | `[1, n_samples]` f32 | `output` | `[1, 1, n_mel, 32]` f32 |
| embedding | `input_1` | `[batch, 76, 32, 1]` f32 | `conv2d_19` | `[batch, 1, 1, 96]` f32 |
| transcribe head | `embeddings` | `[batch, 16, 96]` f32 | `score` | `[batch, 1]` f32 |

- `n_mel = floor((n_samples − 512) / 160) + 1`. That is a 512-sample (32 ms) window with a 160-sample
  (10 ms) hop and no padding: 1760 samples give 8 rows, 1280 give 5, and 32000 give 197.
- The head output is already a sigmoid probability in [0, 1]. If a head ever returns a value outside
  [0, 1], treat it as a logit and apply `1/(1+e^-x)`; the reference does this for third-party heads.
- Read the head's input name and frame count (dim 1) from the model, and do not hard-code 16.

## 3. Audio scale (most common porting bug)

The front end expects **int16-scale** sample values passed as f32: a full-scale sine reaches about
±32767, not ±1. Training extracts features the same way. Feeding [-1, 1] floats shifts the log-mel
by about 9 units and silently ruins detection. If the capture gives f32 in [-1, 1], multiply by
32767 before the mel model. The VAD (Silero) takes [-1, 1] floats; convert separately.

## 4. Streaming front end (exact)

State, initialised by `reset()`:

```
raw      : f32[1760]  = zeros            # FRAME (1280) + MEL_CONTEXT (480) samples
mel_buf  : f32[76][32] = all 1.0         # last 76 mel rows
feats    : f32[N][96]                    # recent embeddings, N ≥ 16 (reference keeps 64)
pending  : f32[]                         # samples not yet forming a full 1280-sample frame
```

`feats` warm-up: run the batch path (below) on 4 s of uniform random int16 noise in
[-1000, 1000) and keep the resulting embeddings. Any small noise works; the Python reference uses
`numpy.random.default_rng(0).integers(-1000, 1000, 64000)`. The warm-up only fills the window until
real frames replace it (16 frames = 1.28 s).

For each complete 1280-sample frame `chunk`:

```
raw      = raw[1280..] ++ chunk                      # keep the last 1760 samples
mel      = melspectrogram(raw)                       # [1,1,8,32] -> 8 rows
mel      = mel / 10.0 + 2.0                          # elementwise; this exact transform
mel_buf  = (mel_buf ++ mel)[last 76 rows]
emb      = embedding(mel_buf as [1,76,32,1])         # -> 96 floats
feats    = (feats ++ emb)[last N rows]
score    = head(feats[last 16 rows] as [1,16,96])
```

The 480 samples of left context (3 hops) make the 8 new mel rows line up with the previous ones.
The embedding stride is therefore 8 mel rows = 80 ms. This is the same math as
`openwakeword.utils.AudioFeatures._streaming_features` for 1280-sample frames. After the first
~1 s following a reset, our output equals openWakeWord's to within 1e-3 (`tests/fixtures/wake/
wake_frontend_ref.npz`, frames 24–29). In that first second they differ slightly because
openWakeWord computes its first mel rows from a shorter buffer where we zero-pad. Training data and
the Python runtime both use the zero-pad behaviour, so the Rust port should zero-pad too.

Batch path (training and tests only): `mel = melspectrogram(clip)/10+2`, then windows
`mel[i..i+76]` for `i = 0, 8, 16, …` while `i+76 ≤ n_mel`, and one embedding per window.

## 5. Gate (exact)

Frame index `i` counts 80 ms frames. The parameters come from `transcribe.json → recommended`:
`threshold` (see §9), `consecutive_frames = 2`, and `cooldown = round(1.5 s / 0.08) = 19 frames`.

```
run = 0; last_fire = None; peak = 0
update(score, i):
    if score < threshold: run = 0; peak = 0; return false
    run += 1; peak = max(peak, score)
    if run < consecutive: return false
    if last_fire is not None and i - last_fire < cooldown: return false   # run is NOT reset here
    last_fire = i; run = 0; return true
hold(i):  last_fire = i; run = 0; peak = 0           # start a cooldown now (after a session ends)
```

Two frames above threshold reject one-frame spikes from clicks and keys, which cause most false
wakes on general audio. The training eval counts false wakes through exactly this gate
(`pipeline.py:gate_count`), so FP/h numbers only hold if the gate is identical.

Also track a **rise frame**: the first frame of the current run with `score ≥ 0.6 × threshold`. Reset
it on any frame below that level. When the gate fires, report `(frame, rise_frame, peak score)`. The
rise frame marks roughly the end of the spoken word and anchors the pre-roll (§7).

## 6. VAD gating (CPU)

The front end and head run only while a voice was heard within the last `hangover = 1.0 s` (12
frames, `round(1.0/0.08)`). For every 80 ms frame:

1. `voiced = vad(frame)`. In the reference this is Silero with probability ≥ 0.5 on the frame
   converted to [-1, 1] floats; any VAD will do.
2. If `voiced`, set `last_voice = i`.
3. If `i − last_voice > hangover`: do **not** infer. Push `(i, frame)` into a backlog ring of
   `catchup = 2.0 s` (25 frames) and stop.
4. Otherwise infer, in order: every backlog frame with index `> last_inferred`, then the current
   frame, and clear the backlog. The backlog feeds the start of a word the VAD only noticed late, so
   it is never lost.
5. Before inferring frame `j`, if `j ≠ last_inferred + 1` (frames were dropped), reset the gate run
   and the rise frame. A run must never join scores across a gap. Do **not** reset the front end;
   it just continues with the new audio, as the reference does.

## 7. Hands-free controller

States: `Off`, `Listening`, `Paused` (another app is using the mic), and `Session`.

**Listening.** Every frame goes to the detector and into a pre-roll ring of
`ceil(preroll_s/0.08) + 25` frames, each stored with its `voiced` flag (`preroll_s = 1.5` by
default). On a hit, outside the post-session quiet period, compute the pre-roll start and start a
session. The orchestrator plays the earcon and feeds `preroll ++ live audio` to the segmenter and STT.

Pre-roll start (`handsfree.py:_preroll_start`), with `idx` = current frame
and `anchor` = the hit's rise frame:

```
floor = idx − ceil(preroll_s/0.08) + 1
if no VAD: return (floor, clean=false)
anchor   = min(anchor, idx)
word_end = newest ring frame with index ≤ anchor that is voiced (else anchor)
floor    = min(floor, word_end − ceil(1.2/0.08))      # ≥ 1.2 s before the word's end, even on a late fire
quiet = 0; heard = false
for (i, voiced) in ring, newest → oldest:
    if i < floor: break
    if !voiced: quiet += 1; if quiet ≥ 5 and heard: return (max(floor, i + quiet − 2), clean=true)
    else:       quiet = 0;  if i ≤ anchor: heard = true
return (max(floor, oldest ring index), clean=false)
```

`clean = true` means the pre-roll begins in a silence of at least 0.4 s before the wake word, with
2 quiet frames of lead-in kept.

**Session.** The detector keeps running. A hit during a session never starts another session.
Instead it **arms** control phrases for `arm_s = 5 s`. Phrase transcripts arrive from the segmenter,
and each one is handled in this order:

1. Empty or whitespace: ignore, but refresh the idle timer.
2. **First phrase only, false-wake check:** the transcript must *start* with the wake phrase
   (§8 `starts_with_phrase`, with `max_fragments = 0` if the pre-roll was clean, else 1). If it
   doesn't, end the session with `FalseWake` and insert nothing. This is what makes
   "I need to transcribe this video" said in conversation harmless.
3. **Split command:** if the previous phrase ended with the bare wake phrase (§8 `ends_with_phrase`)
   and this phrase is a bare command (§8 `parse_control` with `armed=true` returning empty text),
   remove the dangling wake phrase from the previous phrase and apply the command. This handles a
   segmenter cut between "transcribe" and "stop". Otherwise the previous phrase stays as it was.
4. `parse_control(text, armed = now < armed_until)`. On the first phrase, also strip the leading
   wake phrase from the remaining text (§8 `strip_leading_phrase`, same `max_fragments`).
5. Apply the action:
   - `None`: append the text. If it ends with the bare wake phrase, remember it as dangling (step 3).
   - `Finish`: append the remaining text, then end with `Stop` (insert).
   - `Send`: append the remaining text, then end with `Send` (insert, then press Enter).
   - `Cancel`: end with `Cancel` (discard everything).
   - `Scratch`: if the phrase had text before the command, drop that text. Otherwise drop the last
     stored phrase. Keep going.

Other endings: `IdleTimeout` after `idle_timeout_s = 45 s` with no voiced frame and no phrase (insert),
`MaxDuration` at `audio.max_session_s = 600 s` (insert), and external finish/cancel from the hotkey
or UI. The session text is the stored phrases joined with single spaces. Insert only if the text is
non-empty and the reason is not `Cancel` or `FalseWake`.

After any end: reset the front end, `gate.hold(now)`, and ignore hits for
`post_session_cooldown = 1.5 s`, so the session's own "transcribe stop" can't wake it again.

**Paused.** On Windows, poll the microphone consent store every 3 s. An app other than us that has
`LastUsedTimeStop == 0` (or older than `LastUsedTimeStart`), started less than 12 h ago, means a call
is in progress. While one is, skip inference and the pre-roll entirely and tell the UI. The
registry layout and filtering are documented in `src/openwhisprflow/wake/calls.py`. This
check happens only while Listening, never during a session. macOS (CoreAudio
`kAudioDevicePropertyDeviceIsRunningSomewhere`) and Linux (PipeWire/Pulse source-outputs) are
TODO.

### 7.1 In the app (`crates/ochre/src/handsfree.rs`)

* **Audio.** While `handsfree.enabled`, one `ochre-wake` thread (normal priority) owns the
  `HandsFree` controller. It subscribes to the capture pump with an always-on tap
  (`Capture::set_tap`): every 16 kHz mono block, tagged with its absolute sample position, is
  `try_send`-copied into a bounded queue (~2.5 s; overflow drops and logs, it never blocks the
  pump, and the real-time cpal callback is untouched). The tap keeps the input device open, so
  **with `audio.warm_mic = false` the mic still stays open while hands-free is on** (and the OS
  mic-in-use indicator stays lit).
* **Wake.** On `HfEvent::Wake` the orchestrator starts a `Trigger::Wake` session (earcon, HUD
  `Handsfree`) whose audio begins at the absolute position of the pre-roll's first sample
  (`Mic::begin_from`; the capture history is 4 s), so the wake word and the words right after it
  reach the segmenter with no gap and no duplicate.
* **Phrases.** Wake sessions always use the phrase segmenter (also with streaming cloud engines).
  Each phrase transcript (hesitations stripped) goes to `HandsFree::on_phrase` on the wake thread;
  `Finish`/`Send`/idle timeout/session cap end the session, `Cancel` and `FalseWake` discard it.
  The tail takes the controller's final text (wake word, control phrases and scratched phrases
  removed) and runs the same path as hotkey dictation: corrections, refinement, injection
  (`Send` presses Enter), history. Escape, the HUD ×, the hotkey and the tray end it as usual.
* **Holding.** While the app is busy with anything else (a hotkey dictation, a session's tail,
  loading) the tap audio is dropped, and the detector is held (reset + cooldown) when listening
  resumes. During a hands-free session the detector keeps running (arming, idle VAD).
* **Lifecycle.** The thread starts after the engines load and restarts when `handsfree.*` or the
  mic changes; disabling stops it, removes the tap and cancels a running session.
  `handsfree_armed` in state events is true only while the detector is loaded, tapped in and
  not paused for a call. A model or front-end download error is logged and shown as a notice.
* **Models.** The `transcribe` head is compiled into the binary and written to
  `<models_dir>/wake/` on first use; the mel and embedding models download once into
  `<models_dir>/openwakeword`; Silero (already used elsewhere) is the VAD gate, an RMS gate if
  it can't load.

## 8. Control-phrase matching (`text/commands.py`)

**Tokens.** Take the maximal runs of Unicode letters and digits, allowing internal `'`, `’` or `-`
between runs. So `stop-motion` is **one** token and can never read as `stop`. Normalise a token by
lower-casing it and removing `'`, `’` and `-`. Keep each token's character span in the original text.

**Phrase similarity.** `ratio(a, b)` is Python `difflib.SequenceMatcher(None, a, b, autojunk=False)
.ratio()`, i.e. `2·M/(len a + len b)`, on strings squashed to lower-case letters and digits only.
Here M is the number of characters in the matching blocks found by recursively taking the longest
common substring (Ratcliff/Obershelp, leftmost on ties). Thresholds:

| constant | value | why |
|---|---|---|
| `DEFAULT_RATIO` | 0.84 | "transcribes", "trans scribe" ≈ 0.95 pass; "transcript" = 0.80, "describe" ≈ 0.67, "subscribe" ≈ 0.63 fail |
| `ARMED_RATIO` | 0.70 | the wake model heard it, so rougher ASR counts ("transcript stop") |
| `MAX_PHRASE_TOKENS` | 3 | split spellings: "trans scribe", "tran scribe" |

A multi-token candidate for a one-word phrase only counts if **every** token is a piece of the
phrase. A token is a piece if the matching-block characters between the token and the phrase cover
at least 80% of the token. This stops junk words being glued onto the phrase ("ing" + "transcribe").

**Command table**, matched on the last 1–2 tokens, longest first:

| tokens | action |
|---|---|
| stop, stops, stopped, done, dun | Finish |
| send, sends, sent, send it | Send |
| cancel, canceled, cancelled, cancels | Cancel |
| scratch that, scratched that, scratch this, scratch | Scratch |

**`parse_control(text, phrase, armed)`:**

```
for n_cmd in [2, 1] (only if len(tokens) ≥ n_cmd):
    action = COMMANDS[last n_cmd tokens]; if none: continue
    cmd_start = len(tokens) − n_cmd
    over every k in 1..min(MAX_PHRASE_TOKENS + words(phrase) − 1, cmd_start):
        candidate = tokens[cmd_start−k .. cmd_start] joined without spaces
        keep the best ratio ≥ (ARMED_RATIO if armed else DEFAULT_RATIO); on a tie keep the smaller k
    if found at token i: return (action, kept = clean(text[..span_start(i)]))
    if armed and cmd_start == 0: return (action, kept = "")          # bare "Stop." after a wake hit
return (None, text)
```

`clean()` strips trailing whitespace and the separators `, ; : - – — …` but keeps `. ! ?`.
The command must be the **tail** of the transcript: only punctuation may follow it, and the token
rule enforces that.

**`ends_with_phrase(text)`:** the last 1–3 tokens match the phrase at `DEFAULT_RATIO`. It returns the
text before them, cleaned.

**`starts_with_phrase(text, max_fragments)` / `strip_leading_phrase`:** walk from the first token.
At each position, if 1–3 tokens there match the phrase at `DEFAULT_RATIO`, the phrase ends there.
Otherwise skip the token if it is a filler (`hey hi hello okay ok so um uh umm uhh oh alright right
well and a the yo`), or skip it as a fragment while `fragments < max_fragments`. Anything else
means the transcript does not start with the phrase. Stripping removes everything up to the
phrase end, then leading whitespace and `,.;:!?-–—…`. If the transcript started upper-case and the
rest starts lower-case, capitalise the rest's first letter.

### Test cases (all in `tests/test_text_commands.py`; port them verbatim)

Must trigger (`phrase = "transcribe"`, not armed):

| transcript | action | kept text |
|---|---|---|
| `transcribe stop` | Finish | `` |
| `Transcribe stop.` | Finish | `` |
| `transcribe, stop.` | Finish | `` |
| `Transcribe. Stop.` | Finish | `` |
| `transcribes stop` | Finish | `` |
| `Transcribe stopped.` | Finish | `` |
| `trans scribe stop` | Finish | `` |
| `Trans-scribe, stop!` | Finish | `` |
| `transcribed done` | Finish | `` |
| `TRANSCRIBE STOP` | Finish | `` |
| `Transcribe Send!` | Send | `` |
| `transcribe sent` | Send | `` |
| `Transcribe, send it.` | Send | `` |
| `Transcribe cancel.` | Cancel | `` |
| `transcribe cancelled` | Cancel | `` |
| `Transcribe, scratch that.` | Scratch | `` |
| `transcribe scratch` | Scratch | `` |
| `Hello world, transcribe send.` | Send | `Hello world` |
| `See you at five. Transcribe stop.` | Finish | `See you at five.` |
| `Thanks so much! Transcribe, send!` | Send | `Thanks so much!` |
| `Let me think about it — transcribe stop` | Finish | `Let me think about it` |
| `I'll call you later transcribe done` | Finish | `I'll call you later` |
| `Draft one... transcribe cancel` | Cancel | `Draft one...` |
| `Ship it on Friday, transcribe scratch that.` | Scratch | `Ship it on Friday` |
| `transcribe stop…` / `Transcribe stop?` | Finish | `` |

Must NOT trigger (not armed; the text comes back unchanged):

| transcript | why |
|---|---|
| `I need to transcribe stop-motion footage` | words after the command |
| `I need to transcribe stop-motion` | `stop-motion` is one token |
| `Can you transcribe stop signs in the video` | not the tail |
| `transcribe send the file to Bob` | not the tail |
| `We should transcribe, then send it to legal for review.` | not the tail |
| `Please describe stop.` / `Subscribe, stop.` / `prescribe done` | near words below the ratio |
| `Read the transcript. Send.` | "transcript" 0.80 < 0.84 |
| `Don't stop.` / `Stop.` / `send` / `Scratch that.` | no wake phrase |
| `I scratched that car door` / `The tribe sent a message` | not a command tail / near word |
| `stop transcribe` | wrong order |
| `transcribe` | wake word alone is not a command |

Armed (a wake hit within the last 5 s): `Stop.` gives Finish, `Send!` gives Send, `Scratch that.`
gives Scratch, and `All done here, transcript stop.` gives Finish with kept text `All done here`.
These still don't trigger: `Don't stop.`, `we must stop now`, `Please describe stop.`.

Session start (`strip_leading_phrase`, `max_fragments=0`):

| transcript | stripped | `starts_with_phrase` |
|---|---|---|
| `Transcribe, hey Sarah, just checking in.` | `Hey Sarah, just checking in.` | true |
| `Hey transcribe, dear team,` | `Dear team,` | true |
| `Okay, transcribe. Thanks for the update.` | `Thanks for the update.` | true |
| `Um, transcribe, so the plan is` | `So the plan is` | true |
| `Trans scribe, quick question.` | `Quick question.` | true |
| `Transcribe.` | `` | true |
| `I need to transcribe this video.` | (unchanged) | **false** (false wake) |
| `Read the transcript.` / `Describe the problem.` | (unchanged) | **false** |
| `ing transcribe, hello` | `hello` only with `max_fragments=1` | false at 0, true at 1 |

## 9. The shipped model and its threshold

The threshold table, the chosen operating point and every evaluation number are in
`assets/wake/transcribe.json`: `recommended.threshold`, `stats`, `threshold_table`, and
`stats.runtime_check` for streamed detection through the reference detector. The summary and the
reasoning about positives and negatives are in `training/wakeword/README.md`. The app's default
`handsfree.threshold` should be `recommended.threshold`.

v1 (2026-10-04, synthetic data only): **recommended threshold 0.45**, 2 frames, 1.5 s cooldown.

| metric | value |
|---|---|
| streamed detection, held-out speakers, "transcribe" alone / followed by speech | **95.0%** (alone 99.1%, with a control word 93.2%) |
| streamed detection after "hey/okay/so" or a dictated sentence | 92.3% (control word after a sentence: 90.6%) |
| near-miss clips that fire once when streamed | 3.0% (mostly "transcribing", "transcriber", "grand/train scribe") |
| false wakes per hour, 10.7 h openWakeWord validation stream through the gate | **0.37** |

The false-wake check in §7 (first phrase must start with the wake word) rejects "Transcribing …"
(ratio 0.82) and "Grand scribe …" fires. It does **not** reject "Transcriber …" (0.95) or
"Train scribe …" (the two tokens join to "trainscribe"). Those reach a session; they are rare
words to open an utterance with.

## 10. Numeric fixture for the port

`tests/fixtures/wake/`:

- `wake_frontend_ref.npz`: openWakeWord's own streaming features for a deterministic chirp (see
  `tests/test_wake_detector.py:_chirp`), frames 24–29, shape `[6,16,96]`.
- `transcribe_sample.wav`: a short held-out-speaker Piper TTS clip, 16 kHz mono int16.
- `transcribe_sample.json`: the Python reference's per-frame values for that WAV, streamed from a
  fresh `reset()` in 1280-sample frames with no VAD. It records `mel_rows_frame0` (the 8 transformed
  mel rows of frame 0), `embedding` (the 96-d embedding after each of the first frames), `scores`
  (the head score for every frame), and `hits` (the gate output at the recommended threshold).
  Tolerances: 1e-4 absolute on mel and embeddings, and 1e-3 on scores.

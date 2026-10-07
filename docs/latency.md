# Latency rules

Push-to-talk dictation feels fast or slow based on one interval: key release to the last
character typed. A speech model that benchmarks at a few hundred milliseconds on an idle machine
can still feel slow in use, because the time is lost around the model, not in it. This document
lists where that time usually goes and the rules Ochre follows to avoid it (they sharpen
SPEC §3.1). Code comments cite the rules by number, so the numbering is stable.

## Where the time goes in a typical dictation app

The median dictation is a single sentence of about 7 to 8 s, spoken straight through, with the
key released about 100 ms after the last word. In a straightforward implementation, everything
below runs after release, in series:

- **Recognition under load.** The decode competes with whatever else the machine is doing
  (builds, indexing, other jobs). CPU inference backends that synchronise worker threads with
  spinning barriers degrade far faster than the load grows: when one worker thread is preempted,
  the others spin until it returns, at every layer. A decode that takes a few hundred
  milliseconds idle can take many seconds at Normal priority on a busy machine, and the delay
  varies from one dictation to the next, so it feels broken rather than merely slow. Raising
  the decode thread's priority removes almost all of it: this is scheduling, not throughput.
- **Decoding only after release.** If phrases are cut only at long pauses after a long minimum
  length, the common 5 to 10 s sentence is decoded entirely after the key comes up.
- **Resampling the whole utterance.** Converting 44.1 or 48 kHz microphone audio to 16 kHz in one
  pass after release costs time proportional to the utterance (tens to a hundred milliseconds).
  Benchmarks on 16 kHz files skip this cost entirely.
- **Stopping the microphone.** Stopping and closing an audio stream can take 50 to 125 ms; doing
  it before the decode puts it on the critical path.
- **Polling.** A 40 ms poll for decode completion costs ~23 ms on average and ~47 ms worst on
  Windows, where short sleeps round up to the 15.6 ms timer tick.
- **Saving before typing.** Writing the history row, with full-text index updates, before
  injection adds ~100 ms or more, growing with history size.
- **Paced injection.** Sending text in small batches with a `Sleep(1)` between them costs
  15.6 ms per batch for a process with no visible window (Windows 11 ignores raised timer
  resolution for those), so a few hundred characters trickle out over hundreds of milliseconds.

There are also perception problems that are not latency at all: no feedback until a hold
threshold and the first audio block have both arrived; clipping the first syllable because
capture starts late; a recording cue that invites the user to wait; hiding the HUD before the
text appears; and clipping the last syllable because capture stops the instant the key is
released.

Things that are usually not the problem: punctuation and correction passes (a few milliseconds),
line-delimited IPC that wakes on a queue rather than polling, phrase-splitting bookkeeping, and
first-inference warm-up for small models.

## Rules

The numbers are release gates.

1. **Decoding must not depend on being the only busy process.** Run the decode thread at
   `THREAD_PRIORITY_ABOVE_NORMAL` (or the process at Above Normal while a session is active).
   Disable spinning in ggml (`n_threads` sized to free physical cores, no busy-wait barriers)
   and ORT (`session.intra_op.allow_spinning=0`).
   **Gate:** `ochre-cli bench --background-load 8` (8 busy threads) must keep p95 release → text
   ≤ 2× the idle figure.
2. **Decode continuously, not only at long pauses.** Cut phrases at any VAD pause ≥ 150–200 ms
   after ≥ 2 s, and decode completed audio every ~2 s even without a pause where the engine
   allows chunking with overlap. At release, the tail must be ≤ ~2 s of audio for any utterance
   length. Target: tail decode ≤ 150 ms idle for a 7.5 s dictation.
3. **Resample in the audio path, incrementally.** Use a streaming resampler in the capture
   thread (or capture at 16 kHz where the OS converts for free) so that nothing proportional to
   utterance length runs after release.
4. **No mic stop on the critical path.** With the warm mic (SPEC rule 2), release just takes a
   ring-buffer snapshot. If the warm mic is off, stop the stream after handing the samples to
   the decoder, not before.
5. **No polling.** Completion arrives over a channel or condvar. A 40 ms poll is 47 ms in
   practice on Windows because of the timer tick.
6. **Type first, then persist.** Before injecting, only an in-memory record and, at most, a
   cheap append-only journal line (≤ 2 ms). The history DB row and full-text indexing run on a
   background thread after injection. This amends SPEC §6.2 "save before insert": keep the
   durability, drop the index work from the critical path. Never key FTS deletes on an
   UNINDEXED column.
7. **One injection call, no timer pacing.** Send the whole text in one `SendInput` (chunk only
   above ~2,000 units). If interruption checks between chunks are kept, wake on an event set by
   the hook, never `Sleep(1)`: Windows 11 ignores raised timer resolution for processes with no
   visible window, so `Sleep(1)` is 15.6 ms. Benchmark IDE editors and terminals, Windows
   Terminal, Chrome and VS Code. For targets that consume per-key events slowly (terminals and
   TUIs), compare a clipboard paste with save and restore, and choose per app class from
   measurements.
8. **Acknowledge within one frame of key down.** Show the HUD within ≤ 33 ms of the key event,
   independent of mic and engine state. Use distinct states for listening, loading (key pressed
   before ready: buffer the audio and finish when ready, never silently drop it), and finishing.
   Keep the HUD up until injection completes. No cue that asks the user to wait.
9. **Pre-roll and post-roll.** Use the warm-mic 1.5 s ring buffer with a ~150 ms pre-roll
   (SPEC rule 2). Add a short post-roll: keep capturing ~150 ms after release, extended up to
   ~300 ms while VAD still sees voice. The decoder starts on everything before release
   immediately, so the post-roll overlaps the tail decode.
10. **Fast, honest startup.** Verify model hashes once after download, then trust size + mtime.
    Load models in parallel, run one warm-up inference, then report Idle (SPEC rule 1).
    Target: ≤ 2 s to ready on a warm disk.
11. **Meter at ≥ 30 Hz in-process.** Compute the level in the capture thread and push it to the
    HUD at frame rate, not on a slow IPC event.
12. **Persist timings.** Write every session's `Timings` (key-down → HUD, key-down → first
    sample, release → tail-decode start, stt_tail, refine, persist, inject, release → last key
    sent) to a local rolling log, so slow dictations in the field can be diagnosed from real
    numbers rather than guessed from coarse timestamps.

## Budget (10 s utterance, idle modern CPU, no refinement)

| stage | target |
|---|---|
| release → engine has the tail | ≤ 5 ms |
| tail decode | ≤ 150 ms (≤ 2 s tail) |
| formatting | ≤ 15 ms |
| completion → orchestrator | < 1 ms |
| persist before inject | ≤ 2 ms (journal) |
| inject (100 chars) | ≤ 10 ms |
| **release → last key sent** | **≤ 200 ms idle, ≤ 400 ms under load** |

"""Record real wake word samples + room audio for retraining (you run it).

    uv run python -m training.wakeword.record_samples --word transcribe --dry-run   # prompts only, no mic
    uv run python -m training.wakeword.record_samples --word transcribe
    uv run python -m training.wakeword.record_samples --list-devices
    uv run python -m training.wakeword.record_samples --word transcribe --device 3 --n 40

Display name and pronunciation come from ``configs/<word>.yaml`` (``wake.phrase`` / ``wake.pronunciation``).

1. ~40 takes after a 3-2-1 countdown. Most are the word ALONE ("Transcribe"); every third is the word
   followed by speech ("Transcribe, hey Sarah" / "Transcribe stop": leave the natural little pause after
   the word). Styles vary: normal, quick, quiet, tired, turned away, from across the room. Takes with no
   speech or with clipping are flagged and redone. Saved as 16 kHz mono WAVs in
   training/wakeword/data/real_positive_<word>/. Takes with speech after the word get a .json sidecar
   with the sample where the wake word ends (the training window ends 0-450 ms after it).
2. ~90 s of ordinary room audio (typing, talking about anything, TV/music, fan...), with a list of
   sound-alikes to read out at some point ("describe", "transcript", "subscribe", ...). Saved to
   training/wakeword/data/real_room_<word>/. It becomes negative training data, so do NOT say the wake
   word itself there.

Then fold them in (one command, ~10-15 min on a GPU, keeps the synthetic data):

    training/wakeword/.venv-train/Scripts/python -m training.wakeword.pipeline retrain --word transcribe

Re-running this script adds more takes (files are timestamped, nothing is overwritten). Needs only
sounddevice + numpy + pyyaml (the app venv has the first two: ``uv pip install pyyaml`` if missing).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SR = 16000

STYLES = [
    "normal voice", "normal voice", "quickly, as if a command follows", "a bit quieter", "normal voice",
    "from a little farther back (1-2 m)", "a bit louder", "relaxed / tired", "quietly, almost under your breath",
    "turned away from the mic", "slowly and clearly", "like calling across the room", "casually, mid-thought",
    "from across the room (3+ m)", "fast",
]
COMMANDS = [
    "hey Sarah", "stop", "meeting notes for Monday", "send", "quick question", "done", "dear team", "cancel",
    "thanks for the update", "scratch that", "I'm running late", "send it",
]
# read out during the room recording: sound-alikes that must NOT wake it (never the wake word itself)
NEAR_MISS_PROMPTS: dict[str, list[str]] = {
    "transcribe": ["Describe it.", "The transcript is ready.", "Subscribe to the channel.", "Prescribe something.",
                   "A transcription service.", "Transfer the money.", "Translate this.", "The tribe.",
                   "A scribe.", "Transport.", "Manuscript.", "Stop. Send. Done."],
}


def word_info(word: str) -> tuple[str, str]:
    """(display phrase, pronunciation) from configs/<word>.yaml."""
    import yaml

    p = HERE / "configs" / f"{word}.yaml"
    if not p.exists():
        return word.capitalize(), ""
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    sec = raw.get("wake") or {}
    phrase = sec.get("phrase") or word.capitalize()
    pron = sec.get("pronunciation") or ""
    return phrase, pron


def dirs_for(word: str) -> tuple[Path, Path]:
    """Same folders as pipeline.Paths."""
    return HERE / "data" / f"real_positive_{word}", HERE / "data" / f"real_room_{word}"


def plan(n: int) -> list[tuple[str, str | None]]:
    """(style, command or None) per take: every third take is "<word>, <command>"."""
    return [(STYLES[i % len(STYLES)], COMMANDS[(i // 3) % len(COMMANDS)] if i % 3 == 2 else None) for i in range(n)]


def save_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(x, dtype=np.int16).tobytes())


def record(sd, seconds: float, device) -> np.ndarray:
    x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16", device=device)
    sd.wait()
    return x[:, 0].copy()


def _loud_blocks(x: np.ndarray, noise_floor: float) -> np.ndarray:
    b = SR // 50
    n = len(x) // b
    rms = np.sqrt(np.mean(x[: n * b].astype(np.float32).reshape(n, b) ** 2, axis=1))
    return np.flatnonzero(rms > max(4.0 * noise_floor, 300.0))


def speech_bounds(x: np.ndarray, noise_floor: float) -> tuple[int, int] | None:
    """Crude energy VAD on 20 ms blocks: first/last block well above the noise floor."""
    loud = _loud_blocks(x, noise_floor)
    if len(loud) < 5:  # < 100 ms of speech
        return None
    b = SR // 50
    return int(loud[0]) * b, int(loud[-1] + 1) * b


def wake_end_of_command_take(x: np.ndarray, noise_floor: float, min_gap_ms: int = 100) -> int | None:
    """End of the first speech run (the wake word) in "<word>, <command>": the first gap of >= ``min_gap_ms``
    after >= 250 ms of speech. None when there is no such pause."""
    loud = _loud_blocks(x, noise_floor)
    if len(loud) < 5:
        return None
    b = SR // 50
    gap = max(1, min_gap_ms // 20)
    for k in range(1, len(loud)):
        if loud[k] - loud[k - 1] > gap and (loud[k - 1] - loud[0] + 1) * 20 >= 250:
            return int(loud[k - 1] + 1) * b
    return None


def main() -> None:
    for st in (sys.stdout, sys.stderr):  # IPA in the prompts on a cp1252 console
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word", default="transcribe", help="wake word (default transcribe)")
    ap.add_argument("--n", type=int, default=40, help="number of takes (every third has a command after it)")
    ap.add_argument("--room-seconds", type=float, default=90.0, help="seconds of room audio (0 = skip)")
    ap.add_argument("--device", default=None, help="input device index or name substring (default: system)")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices (opens no stream)")
    ap.add_argument("--skip-words", action="store_true", help="only record room audio")
    ap.add_argument("--dry-run", action="store_true", help="print the prompts and folders; never touch the mic")
    a = ap.parse_args()

    word = a.word.lower().replace(" ", "_")
    phrase, pron = word_info(word)
    pos_dir, room_dir = dirs_for(word)
    takes = plan(a.n)
    near = NEAR_MISS_PROMPTS.get(word, [])

    if a.dry_run:
        print(f"word {word!r}: say \"{phrase}\" {pron}".rstrip())
        print(f"takes -> {pos_dir}\nroom  -> {room_dir}\n")
        for i, (style, cmd) in enumerate(takes):
            print(f"  take {i + 1:2d}: {style:38s} {phrase + ', ' + cmd if cmd else phrase}")
        if a.room_seconds > 0:
            print(f"\nroom audio {a.room_seconds:.0f} s; read out: {' / '.join(near) or '(anything)'}")
        print("\n(dry run: no audio device was opened)")
        return

    import sounddevice as sd

    if a.list_devices:
        print(sd.query_devices())
        return
    device = int(a.device) if a.device is not None and str(a.device).isdigit() else a.device
    info = sd.query_devices(device, "input")
    print(f"Input device: {info['name']}\n")

    stamp = time.strftime("%Y%m%d_%H%M%S")

    print("Measuring room noise for 1.5 s: stay quiet...")
    floor_clip = record(sd, 1.5, device)
    noise_floor = float(np.sqrt(np.mean(floor_clip.astype(np.float32) ** 2)))
    print(f"  noise floor RMS = {noise_floor:.0f} (int16 units)\n")

    if not a.skip_words:
        print(f'Say "{phrase}" {pron} after each countdown. {a.n} takes.'.replace("  ", " "))
        print("Most takes are the word ONLY; some ask for the word then more speech (\"" + phrase + ", hey Sarah\").")
        print("Vary how you say it as prompted. Ctrl+C stops early; finished takes are kept.\n")
        input("Press Enter to start...")
        saved = 0
        try:
            while saved < a.n:
                style, cmd = takes[saved]
                say = f"{phrase}, {cmd}" if cmd else phrase
                print(f"\nTake {saved + 1}/{a.n} ({style}): \"{say}\"")
                for c in ("3", "2", "1"):
                    print(f"  {c}...", end="", flush=True)
                    time.sleep(0.6)
                print("  >>> SAY IT <<<", flush=True)
                x = record(sd, 4.0 if cmd else 2.5, device)
                bounds = speech_bounds(x, noise_floor)
                peak = int(np.max(np.abs(x.astype(np.int32))))
                if bounds is None:
                    print("  (no speech detected: let's redo that one)")
                    continue
                if peak >= 32000:
                    print("  (clipped: a bit quieter or further from the mic please; redoing)")
                    continue
                s, e = bounds
                wake_end = None
                if cmd:
                    wake_end = wake_end_of_command_take(x, noise_floor)
                    if wake_end is None:
                        print(f"  (no pause after \"{phrase}\": leave a tiny comma-pause before the command; redoing)")
                        continue
                elif e - s > int(1.6 * SR):
                    print("  (that was long, just the single word please; redoing)")
                    continue
                s = max(0, s - int(0.25 * SR))
                e = min(len(x), e + int(0.25 * SR))
                kind = "cmd" if cmd else "word"
                path = pos_dir / f"{word}_{stamp}_{saved:02d}_{kind}.wav"
                save_wav(path, x[s:e])
                if wake_end is not None:  # pipeline.py reads this instead of guessing the word's end
                    path.with_suffix(".json").write_text(json.dumps({"wake_end": int(wake_end - s), "text": say}) + "\n",
                                                         encoding="utf-8")
                saved += 1
                print(f"  saved {path.name} ({(e - s) / SR:.2f} s, peak {peak})")
        except KeyboardInterrupt:
            print("\nStopped.")
        print(f"\n{saved} takes saved to {pos_dir}")

    if a.room_seconds > 0:
        print(f"\nNow {a.room_seconds:.0f} s of normal room audio: keep working, type, talk about anything,")
        print(f"play music/TV, just DON'T say \"{phrase}\".")
        if near:
            print("At some point read these out loud (sound-alikes that must NOT wake it):")
            print("  " + "  /  ".join(near))
        try:
            input("Press Enter to start (Ctrl+C to skip)...")
        except KeyboardInterrupt:
            return
        x = record(sd, a.room_seconds, device)
        path = room_dir / f"room_{stamp}.wav"
        save_wav(path, x)
        print(f"  saved {path}")

    print("\nDone. Fold the recordings into the model with:")
    print(f"  training/wakeword/.venv-train/Scripts/python -m training.wakeword.pipeline retrain --word {word}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

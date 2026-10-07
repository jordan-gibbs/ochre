"""Live wake word scores from the mic, for threshold tuning (you run it).

Uses the app's own detector (openwhisprflow/wake/detector.py: same front end, head and gate as the
running app), so run it with the app venv from the repo root:

    uv run python -m training.wakeword.eval_live                       # bundled transcribe model + its threshold
    uv run python -m training.wakeword.eval_live --threshold 0.6 --device 3
    uv run python -m training.wakeword.eval_live --model path/to/other.onnx
    uv run python -m training.wakeword.eval_live --wav some.wav        # score a file instead of the mic

Try: say "transcribe" ~20 times at different distances and volumes, alone, as "Transcribe, <speech>"
and as "... transcribe stop" after a sentence (count the hits); then talk normally, play music/video,
and say the near-misses printed below for a few minutes (count false fires). Ctrl+C prints a summary
with the number of fires per threshold. Set the result in settings (handsfree.threshold).
"""

from __future__ import annotations

import argparse
import queue
import time
import wave
from pathlib import Path

import numpy as np

NEAR_MISSES = {
    "transcribe": "describe / transcript / transcription / subscribe / prescribe / scribe / tribe / transfer / "
                  "translate / transport / manuscript",
}


def bar(score: float, thr: float, width: int = 50) -> str:
    n, t = int(round(score * width)), int(round(thr * width))
    cells = ["#" if i < n else " " for i in range(width)]
    if 0 <= t < width:
        cells[t] = "|"
    return "".join(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word", default="transcribe")
    ap.add_argument("--model", help=".onnx head (default: the bundled model for --word)")
    ap.add_argument("--threshold", type=float, help="default: the model's recommended threshold")
    ap.add_argument("--device", default=None, help="input device index or name substring")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--wav", help="score a 16 kHz mono 16-bit WAV instead of the microphone")
    ap.add_argument("--quiet", action="store_true", help="only print detections")
    a = ap.parse_args()

    from openwhisprflow.wake.detector import FRAME, WakeDetector, WakeGate, bundled_model_path

    model = Path(a.model) if a.model else bundled_model_path(a.word)
    det = WakeDetector(model, threshold=a.threshold)
    thr = det.threshold
    print(f"model {model} | threshold {thr} | 2 consecutive frames, 1.5 s cooldown")
    print(f'say "{a.word}" (alone, with speech after it, and after a sentence), then these near-misses: '
          f"{NEAR_MISSES.get(a.word, '(any sound-alikes)')}\n")

    scores: list[float] = []
    fires: list[float] = []

    def handle(frame: np.ndarray) -> None:
        hits = det.process(frame)
        s = det.last_score
        scores.append(s)
        t = len(scores) * 0.08
        if hits:
            fires.append(t)
            print(f"\n*** WAKE at {t:7.2f}s  score {hits[0].score:.3f}  (#{len(fires)}) ***\n", flush=True)
        elif not a.quiet and (s >= 0.1 or len(scores) % 12 == 0):
            lvl = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768) ** 2)))
            print(f"{t:7.2f}s {s:.3f} [{bar(s, thr)}] lvl {lvl:.3f}", flush=True)

    t0 = time.time()
    try:
        if a.wav:
            with wave.open(a.wav, "rb") as w:
                assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
                x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            x = np.concatenate([np.zeros(FRAME * 6, np.int16), x, np.zeros(FRAME * 12, np.int16)])
            for i in range(0, len(x) - FRAME + 1, FRAME):
                handle(x[i:i + FRAME])
        else:
            import sounddevice as sd

            dev = int(a.device) if a.device is not None and str(a.device).isdigit() else a.device
            q: queue.Queue[np.ndarray] = queue.Queue()
            with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=FRAME, device=dev,
                                callback=lambda d, n, ti, st: q.put(d[:, 0].copy())):
                print("listening... (Ctrl+C to stop)\n")
                while not a.duration or time.time() - t0 < a.duration:
                    handle(q.get())
    except KeyboardInterrupt:
        pass

    if scores:
        s = np.asarray(scores)
        print("\n--- summary ---")
        print(f"{len(s)} frames ({len(s) * 0.08 / 60:.1f} min), {len(fires)} wake(s) at threshold {thr}")
        for q_ in (50, 90, 99, 99.9):
            print(f"  score p{q_:<5}: {np.percentile(s, q_):.3f}")
        print(f"  max score   : {s.max():.3f}")
        for t_ in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            g = WakeGate(t_, 2, 19)
            print(f"  would fire at threshold {t_:.1f}: {sum(g.update(float(v), i) for i, v in enumerate(s))} time(s)")
        print(f"Pick the highest threshold that still catches (nearly) every real '{a.word}' while firing")
        print("~never during normal talk; set it in settings -> Hands-free -> sensitivity (handsfree.threshold).")


if __name__ == "__main__":
    main()

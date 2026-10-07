"""Regenerate the Python-reference fixtures that tests/parity.rs checks the Rust port against.

    .venv/Scripts/python crates/ochre-wake/tests/fixtures/make_reference.py <wake_head.onnx> <oww_dir> [clip.wav ...]

For each 16 kHz mono int16 clip it writes ``<clip>.json`` next to it, in the same schema as
``tests/fixtures/wake/transcribe_sample.json`` (docs/wakeword.md §10): the values from a fresh
``reset()`` streamed in 1280-sample frames without a VAD, plus ``detector_hits`` from
``score_clip`` (1 s of silence before and after, full WakeDetector). It also writes the chirp
fixture (``chirp.wav`` + ``chirp_windows.json``) from ``tests/fixtures/wake/wake_frontend_ref.npz``.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from openwhisprflow.wake.detector import WakeDetector, WakeGate, WakeHead, score_clip  # noqa: E402
from openwhisprflow.wake.frontend import FRAME, WakeFrontend  # noqa: E402

HERE = Path(__file__).resolve().parent
THRESHOLD = 0.75  # the reference head's tuned threshold


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.astype("<i2").tobytes())


def chirp() -> np.ndarray:  # == tests/test_wake_detector.py:_chirp
    sr = 16000
    t = np.arange(int(2.4 * sr)) / sr
    rng = np.random.default_rng(7)
    sig = 6000 * np.sin(2 * np.pi * (200 + 300 * t) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (sig + rng.normal(0, 400, len(t))).astype(np.int16)


def main() -> None:
    model, oww = Path(sys.argv[1]), Path(sys.argv[2])
    clips = [Path(p) for p in sys.argv[3:]]
    mel, emb = oww / "melspectrogram.onnx", oww / "embedding_model.onnx"

    write_wav(HERE / "chirp.wav", chirp())
    ref = np.load(ROOT / "tests" / "fixtures" / "wake" / "wake_frontend_ref.npz")["windows"]
    (HERE / "chirp_windows.json").write_text(json.dumps({"windows": ref.astype(float).tolist()}))

    for clip in clips:
        with wave.open(str(clip), "rb") as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        fe = WakeFrontend(mel, emb)
        head = WakeHead(model)
        gate = WakeGate(THRESHOLD, 2, 19)
        mel0, embs, scores, hits = None, [], [], []
        for k in range(len(x) // FRAME):
            fe.push(x[k * FRAME:(k + 1) * FRAME])
            if k == 0:
                mel0 = fe._mel_buf[-8:].tolist()
            if k < 20:
                embs.append(fe.features[-1].tolist())
            s = head.score(fe)
            scores.append(s)
            if gate.update(s, k):
                hits.append({"frame": k, "score": s})
        det = WakeDetector(model, threshold=THRESHOLD, frontend=WakeFrontend(mel, emb))
        _, det_hits = score_clip(det, x)
        out = {
            "model": model.stem,
            "threshold": THRESHOLD,
            "frames": len(scores),
            "mel_rows_frame0": mel0,
            "embedding_first_frames": embs,
            "scores": scores,
            "hits": hits,
            "detector_hits": [{"frame": h.frame, "rise_frame": h.rise_frame, "score": h.score} for h in det_hits],
        }
        dest = HERE / clip.name
        if clip.resolve() != dest.resolve():
            write_wav(dest, x)
        dest.with_suffix(".json").write_text(json.dumps(out))
        print(f"{clip.name}: {len(scores)} frames, peak {max(scores):.4f}, hits {[h['frame'] for h in hits]}, "
              f"detector {[(h.frame, h.rise_frame) for h in det_hits]}")


if __name__ == "__main__":
    main()

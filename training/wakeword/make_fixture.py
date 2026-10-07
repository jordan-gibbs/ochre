"""Build the numeric fixture a runtime port is checked against (tests/fixtures/wake/, docs/wakeword.md §10).

    # 1. a short held-out-speaker Piper clip (training venv: needs torch + livekit-wakeword)
    training/wakeword/.venv-train/Scripts/python -m training.wakeword.make_fixture synth
    # 2. per-frame reference values through the Python runtime (app venv)
    uv run python -m training.wakeword.make_fixture reference [--frontend-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "tests" / "fixtures" / "wake"
WAV = OUT / "transcribe_sample.wav"
TEXT = "Transcribe, hello there. Transcribe send."


def synth(speaker: int, seed: int) -> None:
    import random

    from training.wakeword import pipeline as pl

    data = pl.data_root(None)
    ph = pl.Phonemizer()
    s = pl.Synth(data / "piper" / "en-us-libritts-high.pt", 0.2)
    assert speaker % 8 == 0, "use a held-out speaker (id % 8 == 0)"
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    ids = s.ids(", ".join(ph(p.strip()) for p in TEXT.split(",")))
    (audio, _), = s.run([(ids, -1)], [speaker], [speaker], 0.0, 1.0, 0.667, 0.8)
    lead = np.zeros(int(0.3 * pl.SR), np.float32)
    audio = np.concatenate([lead, audio, lead])
    OUT.mkdir(parents=True, exist_ok=True)
    pl.write_wav(WAV, audio)
    print(f"wrote {WAV} ({len(audio) / pl.SR:.2f} s, speaker {speaker}): {TEXT!r}")


def reference(frontend_dir: str | None) -> None:
    from openwhisprflow.wake.detector import FRAME, WakeGate, WakeHead, load_meta
    from openwhisprflow.wake.frontend import WakeFrontend

    model = REPO / "assets" / "wake" / "transcribe.onnx"
    meta = load_meta(model)
    thr = float(meta["recommended"]["threshold"])
    fd = Path(frontend_dir) if frontend_dir else None
    fe = WakeFrontend(fd / "melspectrogram.onnx", fd / "embedding_model.onnx") if fd else WakeFrontend()
    head = WakeHead(model)
    with wave.open(str(WAV), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    fe.reset()
    gate = WakeGate(thr, 2, 19)
    mel0, embs, scores, hits = None, [], [], []
    for k in range(len(x) // FRAME):
        fe.push(x[k * FRAME:(k + 1) * FRAME])
        if k == 0:
            mel0 = fe._mel_buf[-8:].copy()
        if k < 4:
            embs.append(fe.features[-1].copy())
        sc = head.score(fe)
        scores.append(sc)
        if gate.update(sc, k):
            hits.append({"frame": k, "score": round(max(gate.peak, sc), 6)})
    ref = {
        "wav": WAV.name, "text": TEXT, "sample_rate": 16000, "frame_samples": FRAME,
        "model": "assets/wake/transcribe.onnx", "threshold": thr, "consecutive_frames": 2, "cooldown_frames": 19,
        "notes": "fresh reset(), int16-scale samples, 1280-sample frames, no VAD; frame k = samples [1280k, 1280k+1280)",
        "mel_rows_frame0": np.round(mel0, 6).tolist(),
        "embedding_first_frames": np.round(np.stack(embs), 6).tolist(),
        "scores": [round(float(v), 6) for v in scores],
        "hits": hits,
    }
    (OUT / "transcribe_sample.json").write_text(json.dumps(ref) + "\n", encoding="utf-8")
    print(f"{len(scores)} frames, max score {max(scores):.3f}, hits {hits}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["synth", "reference"])
    ap.add_argument("--speaker", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--frontend-dir", default=os.environ.get("OWF_OWW_DIR"))
    a = ap.parse_args()
    if a.step == "synth":
        synth(a.speaker, a.seed)
    else:
        reference(a.frontend_dir)


if __name__ == "__main__":
    main()

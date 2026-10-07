"""Screen Piper's 904 LibriTTS speakers for intelligibility; writes piper_speakers.json.

Why: in the pilot, dry (un-augmented) Piper audio had 17% Parakeet WER against 7% for Kokoro:
some LibriTTS speaker embeddings are barely intelligible. A random sample of speakers
(default 320) each read the same everyday sentences at default prosody; speakers whose Parakeet WER over them is <= --max-wer are
kept, and common.plan_voice samples Piper speakers from that list only.

    .venv-data/Scripts/python tools/refine-data/piper_screen.py [--max-wer 0.0]
"""

from __future__ import annotations

import nowmi  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SENTENCES = [
    "can you send me the report by friday afternoon",
    "i think we should move the meeting to next tuesday",
    "the build failed again because the payment tests time out on windows",
    "remind me to call the dentist and pick up groceries after work",
]


def main() -> None:
    import numpy as np

    import asr
    from audit_auto import edit_ops, norm_tokens
    from common import PIPER_SPEAKERS
    from tts import Piper, silence, trim

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wer", type=float, default=0.0)
    ap.add_argument("--speakers", type=int, default=320, help="screen a random sample of this many speakers")
    ap.add_argument("--sentences", type=int, default=2)
    a = ap.parse_args()
    import random

    sample = sorted(random.Random(904).sample(range(PIPER_SPEAKERS), min(a.speakers, PIPER_SPEAKERS)))
    sents = SENTENCES[: a.sentences]
    synth, rec = Piper(), asr.Parakeet(8)
    refs = [norm_tokens(s) for s in sents]
    n_ref = sum(len(r) for r in refs)
    scores: dict[int, float] = {}
    t0 = time.perf_counter()
    for k, spk in enumerate(sample):
        v = {"speaker": spk, "length_scale": 1.0, "noise": 0.667, "noise_w": 0.8}
        err = 0
        for s, ref in zip(sents, refs):
            audio = np.concatenate([silence(0.3), trim(synth.synth(s, v)), silence(0.4)]).astype(np.float32)
            err += edit_ops(ref, norm_tokens(rec.model.recognize(audio, sample_rate=16000)))[0]
        scores[spk] = round(err / n_ref, 3)
        if k % 50 == 49:
            print(f"{k + 1}/{len(sample)} speakers, {time.perf_counter() - t0:.0f}s", flush=True)
    keep = sorted(s for s, w in scores.items() if w <= a.max_wer)
    out = {"max_wer": a.max_wer, "sentences": sents, "screened": len(sample), "kept": keep, "scores": scores}
    (HERE / "piper_speakers.json").write_text(json.dumps(out), encoding="utf-8")
    hist = {f"<={t}": sum(w <= t for w in scores.values()) for t in (0.0, 0.03, 0.05, 0.08, 0.12, 0.2)}
    print(f"kept {len(keep)}/{len(sample)} screened speakers (WER <= {a.max_wer}); cumulative: {hist}")


if __name__ == "__main__":
    main()

"""Check a trained head through the APP's detector (openwhisprflow/wake/detector.py), offline, no mic.

Run with the app venv from the repo root, after ``pipeline.py all``:

    uv run python -m training.wakeword.verify_runtime --word transcribe [--n 300] [--threshold T] [--write-meta]

Streams held-out-speaker synthetic clips (output/<word>/clips/*_test: the word alone, "<word>, <speech>",
"<word> stop/send", after "hey/okay/<a sentence>", and the near-miss negatives) 80 ms at a time through
WakeDetector exactly as the live app does (fresh context per clip, silence around it, random level and
background hiss), and reports per-clip detection / false-fire rates. Writes output/<word>/runtime_verify.json;
``--write-meta`` records the summary in src/openwhisprflow/wake/models/<word>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONTROL_WORDS = ("stop", "send", "done", "cancel", "scratch that", "send it", "stop please")


def read_wav16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def category(split: str, text: str) -> str:
    t = text.lower().rstrip(".!?,")
    if split.startswith("negative"):
        return "near_miss" if "prefix" not in split else "near_miss_after_prefix"
    pre = "after_prefix_" if "prefix" in split else ""
    if any(t.endswith(" " + c) or t.endswith("," + c) for c in CONTROL_WORDS):
        return pre + "with_control_word"
    if "," in t or "." in t[:-1] or len(t.split()) > 2 + bool(pre):
        return pre + "with_speech_after"
    return pre + "alone"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word", default="transcribe")
    ap.add_argument("--model", default=None, help="default src/openwhisprflow/wake/models/<word>.onnx (or the output dir)")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--n", type=int, default=300, help="clips per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-meta", action="store_true")
    ap.add_argument("--frontend-dir", default=None,
                    help="dir with melspectrogram.onnx + embedding_model.onnx (default: the app's models dir)")
    a = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass

    from openwhisprflow.wake.detector import FRAME, WakeDetector, bundled_model_path, load_meta
    from openwhisprflow.wake.frontend import WakeFrontend

    out = HERE / "output" / a.word
    model = Path(a.model) if a.model else bundled_model_path(a.word)
    if not model.exists():
        model = out / f"{a.word}.onnx"
    meta = load_meta(model)
    if a.threshold is None and not meta:
        ev = out / f"{a.word}_eval.json"
        if ev.exists():
            a.threshold = json.loads(ev.read_text(encoding="utf-8"))["recommended_threshold"]
    fe = None
    if a.frontend_dir:
        fd = Path(a.frontend_dir)
        fe = WakeFrontend(fd / "melspectrogram.onnx", fd / "embedding_model.onnx")
    det = WakeDetector(model, threshold=a.threshold, frontend=fe)
    thr = det.threshold
    print(f"model {model} | threshold {thr} | 2 consecutive frames, 1.5 s cooldown")
    rng = np.random.default_rng(a.seed)

    def stream(x: np.ndarray) -> tuple[float, int]:
        level_db = float(rng.uniform(-35, -18))
        snr_db = float(rng.uniform(10, 30))
        x = x * (10 ** (level_db / 20) * 32767 / (np.sqrt(np.mean(x ** 2)) + 1e-9))
        x = np.concatenate([np.zeros(int(1.5 * 16000), np.float32), x, np.zeros(int(1.2 * 16000), np.float32)])
        x = x + rng.standard_normal(len(x)).astype(np.float32) * 10 ** ((level_db - snr_db) / 20) * 32767
        x = np.clip(x, -32768, 32767).astype(np.int16)
        det.reset()
        det.gate._last_fire = None  # fresh clip: no cooldown carried over
        peak, fires = 0.0, 0
        for i in range(0, len(x) - FRAME + 1, FRAME):
            fires += len(det.process(x[i:i + FRAME]))
            peak = max(peak, det.last_score)
        return peak, fires

    res: dict = {"threshold": thr, "model": str(model), "splits": {}, "categories": {}}
    cats: dict[str, list[bool]] = {}
    for split in ("positive_test", "positive_prefix_test", "negative_test", "negative_prefix_test"):
        d = out / "clips" / split
        if not (d / "manifest.jsonl").exists():
            continue
        man = [json.loads(line) for line in open(d / "manifest.jsonl", encoding="utf-8") if line.strip()]
        pick = [man[i] for i in rng.choice(len(man), size=min(a.n, len(man)), replace=False)]
        rows = []
        for m in pick:
            peak, fires = stream(read_wav16(d / m["file"]))
            c = category(split, m["text"])
            cats.setdefault(c, []).append(fires > 0)
            rows.append({"text": m["text"], "peak": round(peak, 3), "fired": fires > 0, "cat": c})
        rate = float(np.mean([r["fired"] for r in rows]))
        res["splits"][split] = {"n": len(rows), "fired": rate, "median_peak": float(np.median([r["peak"] for r in rows])),
                                "rows": rows}
        print(f"{split:22} n={len(rows):4d} fired={rate:.3f} median peak={res['splits'][split]['median_peak']:.3f}")
        if split.startswith("negative"):
            for r in sorted(rows, key=lambda r: -r["peak"])[:8]:
                print(f"    {r['peak']:.3f} {'FIRE' if r['fired'] else '    '} {r['text']}")
        else:
            for r in sorted(rows, key=lambda r: r["peak"])[:5]:
                print(f"    missed? {r['peak']:.3f} {r['text']}")
    print("by category:")
    for c, v in sorted(cats.items()):
        res["categories"][c] = {"n": len(v), "fired": round(float(np.mean(v)), 4)}
        print(f"  {c:34} n={len(v):4d} fired={np.mean(v):.3f}")
    (out / "runtime_verify.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    if a.write_meta and meta:
        sp = res["splits"]
        meta.setdefault("stats", {})["runtime_check"] = {
            "via": "openwhisprflow.wake.detector.WakeDetector (2 frames, 1.5 s cooldown), streamed 80 ms frames, "
                   "random level -35..-18 dBFS, SNR 10-30 dB white noise",
            "threshold": thr,
            "heldout_positive_clips_detected": round(sp["positive_test"]["fired"], 4),
            "heldout_prefix_clips_detected": round(sp["positive_prefix_test"]["fired"], 4) if "positive_prefix_test" in sp else None,
            "heldout_near_miss_clips_fired": round(sp["negative_test"]["fired"], 4),
            "heldout_prefix_near_miss_clips_fired": round(sp["negative_prefix_test"]["fired"], 4)
            if "negative_prefix_test" in sp else None,
            "by_category": res["categories"],
            "n_per_split": a.n,
        }
        model.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated {model.with_suffix('.json')}")


if __name__ == "__main__":
    main()

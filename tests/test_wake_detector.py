"""wake/frontend.py + wake/detector.py: front-end parity with openWakeWord, the gate, VAD gating."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from openwhisprflow.wake.detector import (
    MODELS_DIR,
    WakeDetector,
    WakeGate,
    bundled_model_path,
    load_meta,
    resolve_model,
    score_clip,
)
from openwhisprflow.wake.frontend import FRAME, WakeFrontend, to_int16_scale

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def frontend_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The two openWakeWord models: $OWF_OWW_DIR if set, else downloaded (sha256-checked) to a temp dir."""
    from openwhisprflow.models import DownloadError
    from openwhisprflow.wake.frontend import EMBED_MODEL, MEL_MODEL, ensure_frontend_models

    d = os.environ.get("OWF_OWW_DIR")
    if d and (Path(d) / MEL_MODEL.name).exists():
        return Path(d) / MEL_MODEL.name, Path(d) / EMBED_MODEL.name
    try:
        return ensure_frontend_models(tmp_path_factory.mktemp("oww"))
    except (DownloadError, OSError) as e:
        pytest.skip(f"front-end models unavailable: {e}")


def _chirp() -> np.ndarray:
    """Deterministic test signal (the fixture was made from exactly this with openwakeword 0.6)."""
    sr = 16000
    t = np.arange(int(2.4 * sr)) / sr
    rng = np.random.default_rng(7)
    sig = 6000 * np.sin(2 * np.pi * (200 + 300 * t) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (sig + rng.normal(0, 400, len(t))).astype(np.int16)


# ---------------------------------------------------------------- front end


@pytest.mark.slow
def test_frontend_matches_openwakeword(frontend_files: tuple[Path, Path]) -> None:
    """Once warmed up, our streaming features equal openwakeword.utils.AudioFeatures bit for bit-ish.

    (The first ~1 s after a reset differs: openWakeWord computes its first mel frames from a
    shorter buffer, we zero-pad, exactly as the reference runtime and our training features do.)
    """
    ref = np.load(FIXTURES / "wake" / "wake_frontend_ref.npz")["windows"]  # frames 24..29, (6, 16, 96)
    fe = WakeFrontend(*frontend_files)
    a = _chirp()
    got = []
    for i in range(0, len(a) - FRAME + 1, FRAME):
        assert fe.push(a[i:i + FRAME]) == 1
        got.append(fe.window(16)[0].copy())
    np.testing.assert_allclose(np.stack(got[24:30]), ref, atol=1e-3)


@pytest.mark.slow
def test_frontend_float_input_and_odd_blocks(frontend_files: tuple[Path, Path]) -> None:
    fe = WakeFrontend(*frontend_files)
    a = _chirp()
    for i in range(0, len(a) - FRAME + 1, FRAME):
        fe.push(a[i:i + FRAME])
    want = fe.window(16).copy()
    fe.reset()
    total = 0
    for chunk in np.array_split(a.astype(np.float32) / 32767.0, 17):  # odd sizes, float [-1, 1]
        total += fe.push(chunk)
    assert total == len(a) // FRAME
    np.testing.assert_allclose(fe.window(16), want, atol=0.05)


def test_to_int16_scale() -> None:
    assert to_int16_scale(np.array([1000], np.int16))[0] == 1000
    assert to_int16_scale(np.array([0.5], np.float32))[0] == pytest.approx(16383.5)
    assert to_int16_scale(np.array([1000.0], np.float32))[0] == 1000.0  # already int16-scale floats


# ---------------------------------------------------------------- gate


def test_gate_needs_consecutive_frames_and_cools_down() -> None:
    g = WakeGate(threshold=0.5, consecutive=2, cooldown_frames=5)
    assert not g.update(0.9, 1)              # one frame is not enough
    assert not g.update(0.1, 2)              # run broken
    assert not g.update(0.9, 3)
    assert g.update(0.8, 4)                  # two in a row: fire
    assert not g.update(0.9, 5) and not g.update(0.9, 6)   # cooldown (5 frames from 4)
    assert not g.update(0.9, 8)              # run resumed at 5..8 but 8 - 4 < 5
    assert g.update(0.9, 9)                  # cooldown over, run still going
    g.hold(20)
    assert not g.update(0.9, 21) and not g.update(0.9, 22)
    assert g.update(0.9, 25)


def test_gate_hold_resets_run() -> None:
    g = WakeGate(0.5, 2, 3)
    g.update(0.9, 1)
    g.hold(1)
    assert not g.update(0.9, 2)  # run restarted by hold, and in cooldown
    assert not g.update(0.9, 3)
    assert g.update(0.9, 4)


# ---------------------------------------------------------------- detector logic (fake front end + head)


class FakeFrontend:
    def __init__(self) -> None:
        self.pushed = 0
        self.resets = 0

    def push(self, frame: np.ndarray) -> int:
        self.pushed += 1
        return 1

    def reset(self) -> None:
        self.resets += 1


class ScriptedHead:
    """Score = script[frontend.pushed] (default 0)."""

    name = "fake"
    n_frames = 16

    def __init__(self, script: dict[int, float]) -> None:
        self.script = script

    def score(self, fe: FakeFrontend) -> float:
        return self.script.get(fe.pushed, 0.0)


def _detector(script: dict[int, float], vad: bool, **kw: object) -> tuple[WakeDetector, FakeFrontend]:
    fe = FakeFrontend()
    is_speech = (lambda f: bool(np.abs(f.astype(np.float32)).mean() > 100)) if vad else None
    det = WakeDetector(Path("fake.onnx"), threshold=0.5, frontend=fe, head=ScriptedHead(script),  # type: ignore[arg-type]
                       is_speech=is_speech, **kw)  # type: ignore[arg-type]
    return det, fe


LOUD = np.full(FRAME, 3000, np.int16)
QUIET = np.zeros(FRAME, np.int16)


def test_detector_fires_and_reports_rise() -> None:
    det, _ = _detector({5: 0.4, 6: 0.7, 7: 0.9}, vad=False)
    hits = []
    for _ in range(10):
        hits += det.process(LOUD)
    assert len(hits) == 1
    h = hits[0]
    assert h.frame == 7 and h.rise_frame == 5 and h.score == pytest.approx(0.9)
    assert det.stats.hits == 1 and det.stats.inferred == 10


def test_vad_gate_skips_inference_in_silence() -> None:
    det, fe = _detector({}, vad=True, hangover_s=0.4)
    for _ in range(100):
        det.process(QUIET)
    assert fe.pushed == 0 and det.stats.inferred == 0
    det.process(LOUD)
    # speech onset: the last 2 s of skipped frames (catch-up) are fed first, then this one
    assert fe.pushed == 26
    for _ in range(5):
        det.process(QUIET)          # hangover (0.4 s = 5 frames) keeps inference running
    assert fe.pushed == 31
    det.process(QUIET)
    assert fe.pushed == 31          # past the hangover: idle again
    assert not det.last_voiced and det.has_vad


def test_catchup_detects_word_that_started_before_vad_onset() -> None:
    # the head's score peaks on frames that the VAD called silence (soft onset of the word)
    det, fe = _detector({}, vad=True)
    for _ in range(30):
        det.process(QUIET)
    det.head.script = {24: 0.8, 25: 0.9}  # type: ignore[attr-defined]  # two caught-up (pre-onset) frames
    hits = det.process(LOUD)
    assert len(hits) == 1 and hits[0].frame == 30


def test_gap_breaks_the_run() -> None:
    # frames lost between inferences (backlog shorter than the silence) must not join two runs
    det, fe = _detector({2: 0.9, 3: 0.9}, vad=True, hangover_s=0.0, catchup_s=0.08)
    det.process(LOUD)     # push 1
    det.process(LOUD)     # push 2: 0.9
    det.process(QUIET)    # skipped (no hangover)
    det.process(QUIET)    # skipped; only this one stays in the 1-frame backlog
    assert det.process(LOUD) == []  # push 3 (0.9) follows a gap: run restarts at 1
    assert fe.pushed == 4


def test_hold_suppresses_hits() -> None:
    det, _ = _detector({3: 0.9, 4: 0.9}, vad=False, cooldown_s=1.0)
    det.process(LOUD)
    det.hold()
    hits = []
    for _ in range(5):
        hits += det.process(LOUD)
    assert hits == []


def test_bundled_model_paths() -> None:
    assert bundled_model_path("transcribe") == MODELS_DIR / "transcribe.onnx"
    assert bundled_model_path("Hey Computer") == MODELS_DIR / "hey_computer.onnx"
    with pytest.raises(FileNotFoundError):
        resolve_model("no such phrase")
    assert load_meta(Path("nope.onnx")) == {}


# ---------------------------------------------------------------- the shipped "transcribe" model

ASSET = Path(__file__).resolve().parents[1] / "assets" / "wake" / "transcribe.onnx"
SAMPLE = FIXTURES / "wake" / "transcribe_sample"


@pytest.mark.slow
@pytest.mark.skipif(not ASSET.exists(), reason="assets/wake/transcribe.onnx not present")
def test_shipped_model_matches_fixture(frontend_files: tuple[Path, Path]) -> None:
    """The per-frame reference in tests/fixtures/wake/ (what a Rust port is checked against) is
    reproduced by the Python runtime, and the clip "Transcribe, hello there. Transcribe send." fires."""
    import json
    import wave

    from openwhisprflow.wake.detector import WakeHead

    meta = load_meta(ASSET)
    assert meta["name"] == "transcribe" and 0.0 < meta["recommended"]["threshold"] < 1.0
    ref = json.loads(SAMPLE.with_suffix(".json").read_text(encoding="utf-8"))
    with wave.open(str(SAMPLE.with_suffix(".wav")), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    fe = WakeFrontend(*frontend_files)
    head = WakeHead(ASSET)
    gate = WakeGate(ref["threshold"], 2, 19)
    scores, hits = [], []
    for k in range(len(x) // FRAME):
        fe.push(x[k * FRAME:(k + 1) * FRAME])
        if k == 0:
            np.testing.assert_allclose(fe._mel_buf[-8:], ref["mel_rows_frame0"], atol=1e-4)
        if k < len(ref["embedding_first_frames"]):
            np.testing.assert_allclose(fe.features[-1], ref["embedding_first_frames"][k], atol=1e-4)
        scores.append(head.score(fe))
        if gate.update(scores[-1], k):
            hits.append(k)
    np.testing.assert_allclose(scores, ref["scores"], atol=1e-3)
    assert hits == [h["frame"] for h in ref["hits"]] and len(hits) >= 1

    # and through the full detector (VAD-less), as the app streams it
    det = WakeDetector(ASSET, frontend=fe)
    _, det_hits = score_clip(det, x)
    assert det_hits

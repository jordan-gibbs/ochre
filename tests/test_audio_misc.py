"""Earcons, level metering and device resolution (no audio hardware touched)."""

from __future__ import annotations

import numpy as np
import pytest

from openwhisprflow.audio import capture, earcons


def test_earcon_assets_exist_and_match_synth() -> None:
    for name in earcons.NAMES:
        pcm, rate = earcons.load(name)
        assert rate == earcons.RATE
        assert 0.1 < len(pcm) / rate < 0.5                     # short
        assert np.max(np.abs(pcm)) <= earcons.PEAK + 1e-3      # quiet (about -22 dBFS)
        np.testing.assert_allclose(pcm, earcons.synth(name), atol=2 / 32768)


def test_play_unknown_or_disabled_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    played = []
    monkeypatch.setattr(earcons, "_play", lambda pcm, rate: played.append(rate))
    earcons.play("nope")
    earcons.set_enabled(False)
    earcons.play("start")
    earcons.set_enabled(True)
    assert played == []


def test_level_mapping() -> None:
    assert capture.level_from_rms(0.0) == 0.0
    assert capture.level_from_rms(10 ** (-70 / 20)) == 0.0
    assert capture.level_from_rms(1.0) == 1.0
    mid = capture.level_from_rms(10 ** (-36 / 20))
    assert 0.45 < mid < 0.55


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    devs = [capture.InputDevice(None, "System default microphone", is_default=True),
            capture.InputDevice(3, "Microphone (Yeti Stereo Microphone)"),
            capture.InputDevice(5, "Headset Microphone (Jabra)")]
    monkeypatch.setattr(capture, "list_devices", lambda: devs)
    assert capture.resolve_device(None) is None
    assert capture.resolve_device("") is None
    assert capture.resolve_device(7) == 7
    assert capture.resolve_device("Headset Microphone (Jabra)") == 5
    assert capture.resolve_device("yeti") == 3
    assert capture.resolve_device("missing") is None

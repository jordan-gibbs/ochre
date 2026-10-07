from math import gcd

import numpy as np
import pytest

from openwhisprflow.audio import resample as rs


def reference(x: np.ndarray, orig: int, target: int) -> np.ndarray:
    """The original unchunked implementation, kept as the oracle for the chunked port."""
    x = np.asarray(x, dtype=np.float64)
    d = gcd(orig, target)
    up, down = target // d, orig // d
    max_rate = max(up, down)
    n_half = 10 * max_rate
    t = np.arange(-n_half, n_half + 1) / max_rate
    h = np.sinc(t) * np.kaiser(2 * n_half + 1, 5.0)
    h *= up / h.sum()
    per_phase = -(-len(h) // up)
    bank = np.zeros(up * per_phase)
    bank[:len(h)] = h
    bank = bank.reshape(per_phase, up).T
    n_out = int(np.ceil(len(x) * up / down))
    pos = np.arange(n_out) * down + n_half
    idx = (pos // up)[:, None] - np.arange(per_phase)[None, :] + per_phase
    padded = np.concatenate([np.zeros(per_phase), x, np.zeros(per_phase)])
    return np.einsum("mk,mk->m", padded[idx], bank[pos % up]).astype(np.float32)


@pytest.mark.parametrize("orig", [48000, 44100, 32000, 22050, 8000])
def test_matches_reference_across_chunk_boundaries(orig: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(int(orig * 1.7)).astype(np.float32) * 0.1
    y = rs.resample(x, orig, 16000)
    assert y.dtype == np.float32
    assert len(y) == int(np.ceil(len(x) * 16000 / orig))
    np.testing.assert_allclose(y, reference(x, orig, 16000), atol=1e-6)


def test_identity_and_empty() -> None:
    x = np.linspace(-1, 1, 100, dtype=np.float32)
    np.testing.assert_array_equal(rs.resample(x, 16000, 16000), x)
    assert rs.resample(np.zeros(0), 48000, 16000).size == 0


def test_tone_frequency_and_level_preserved() -> None:
    t = np.arange(48000) / 48000
    y = rs.resample(0.5 * np.sin(2 * np.pi * 440 * t), 48000, 16000)
    seg = y[1000:-1000]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    assert abs(np.argmax(spec) * 16000 / len(seg) - 440) < 2
    assert abs(np.sqrt(np.mean(seg ** 2)) - 0.5 / np.sqrt(2)) < 0.01


def test_aliasing_is_suppressed() -> None:
    t = np.arange(48000) / 48000
    y = rs.resample(0.5 * np.sin(2 * np.pi * 12000 * t), 48000, 16000)  # above the new Nyquist
    assert np.sqrt(np.mean(y[1000:-1000] ** 2)) < 0.01


def test_rejects_stereo() -> None:
    with pytest.raises(ValueError):
        rs.resample(np.zeros((10, 2)), 48000, 16000)

"""Polyphase FIR resampling with numpy only.

Microphones run at their native rate (usually 44.1/48 kHz) because forcing 16 kHz in the
driver gives inconsistent quality across devices; we convert here instead. Same design as
``scipy.signal.resample_poly`` defaults (Kaiser-windowed sinc, half_width=10, beta=5.0) so
results match it to float32 precision without shipping scipy. Only output samples are
computed, never the full upsampled signal: a 5 s capture at 48 kHz takes about 30 ms.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np


@lru_cache(maxsize=8)
def _filter_bank(up: int, down: int, half_width: int = 10, beta: float = 5.0) -> tuple[np.ndarray, int, int]:
    max_rate = max(up, down)
    n_half = half_width * max_rate
    t = np.arange(-n_half, n_half + 1) / max_rate
    h = np.sinc(t) * np.kaiser(2 * n_half + 1, beta)
    h *= up / h.sum()
    per_phase = -(-len(h) // up)
    bank = np.zeros(up * per_phase)
    bank[:len(h)] = h
    bank = bank.reshape(per_phase, up).T  # bank[phase, k] == h[phase + k * up]
    bank.setflags(write=False)
    return bank, n_half, per_phase


def resample(x: np.ndarray, orig_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Return mono ``x`` resampled from ``orig_rate`` to ``target_rate`` as float32."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("resample expects mono audio")
    if orig_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    divisor = gcd(int(orig_rate), int(target_rate))
    up, down = int(target_rate) // divisor, int(orig_rate) // divisor
    if up == down or len(x) == 0:
        return x.astype(np.float32)
    bank, n_half, per_phase = _filter_bank(up, down)
    n_out = int(np.ceil(len(x) * up / down))
    padded = np.concatenate([np.zeros(per_phase), x, np.zeros(per_phase)])
    taps = np.arange(per_phase)[None, :]
    y = np.empty(n_out, dtype=np.float32)
    # Blocks of output samples bound the (outputs x taps) gather to a few MB, so a 10-minute
    # capture does not need gigabytes of index arrays.
    for lo in range(0, n_out, _BLOCK):
        pos = np.arange(lo, min(n_out, lo + _BLOCK)) * down + n_half
        idx = (pos // up)[:, None] - taps + per_phase
        y[lo:lo + len(pos)] = np.einsum("mk,mk->m", padded[idx], bank[pos % up])
    return y


_BLOCK = 16384

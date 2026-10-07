"""Acoustic conditions: what a real mic in a real room would add to a dry TTS clip.

Every clip gets mic pre-roll/post-roll and an electrical noise floor (so the app's energy-based
phrase cutter sees realistic "silence"). Clips outside the clean subset (common.CLEAN_SHARE)
also get, each independently and recorded in ``audio_conditions``:

* speed (resampling, pitch moves with it) 0.90-1.10, p=0.5
* room impulse response (MIT IR survey, 270 rooms), p=0.6, wet/dry mix 0.3-0.85
* background noise (MUSAN noise: fans, traffic, crowd, appliances...), p=0.65, SNR 12-32 dB
  (12% of noisy clips 5-10 dB). Clips recognized before 2026-10-04 20:30 used the first setting
  (wet 0.5-1.0, p=0.75, SNR 5-30 uniform); every row records its own conditions.
* mic chain: laptop array (HP 250-450 Hz, LP 5-7 kHz, presence peak, light compression), headset,
  phone/bluetooth (300-3400 Hz, 8 kHz codec path), USB desk mic, or none
* level: peak -30..-2 dBFS, plus 4% overdriven (hard-clipped) clips

Noise and RIR files are read in place from the livekit-wakeword data (never copied).
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from common import NOISE_DIRS, RIR_DIRS, SR, rng_for


@lru_cache(maxsize=1)
def rir_files() -> list[Path]:
    return sorted(p for d in RIR_DIRS if d.exists() for p in d.rglob("*.wav"))


@lru_cache(maxsize=1)
def noise_files() -> list[Path]:
    return sorted(p for d in NOISE_DIRS if d.exists() for p in d.rglob("*.wav"))


@lru_cache(maxsize=64)
def _load(path: str) -> np.ndarray:
    import soundfile as sf
    from tts import to16k

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    return to16k(a[:, 0], sr)


def db(x: float) -> float:
    return float(10 ** (x / 20))


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


def speech_rms(a: np.ndarray) -> float:
    """RMS over the louder half of 20 ms frames (ignores pauses, so SNR means speech vs noise)."""
    w = SR // 50
    n = len(a) // w
    if n < 2:
        return rms(a)
    fr = np.sqrt(np.mean(a[: n * w].reshape(n, w).astype(np.float64) ** 2, axis=1))
    top = np.sort(fr)[n // 2:]
    return float(np.sqrt(np.mean(top ** 2)) + 1e-12)


def biquad(kind: str, f0: float, q: float = 0.707, gain_db: float = 0.0, sr: int = SR):
    """RBJ cookbook biquad -> (b, a)."""
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    A = 10 ** (gain_db / 40)
    c = np.cos(w0)
    if kind == "hp":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "lp":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * c, 1 - alpha * A]
        a = [1 + alpha / A, -2 * c, 1 - alpha / A]
    else:
        raise ValueError(kind)
    return np.array(b) / a[0], np.array(a) / a[0]


def filt(x: np.ndarray, *specs) -> np.ndarray:
    from scipy.signal import lfilter

    for s in specs:
        b, a = biquad(*s)
        x = lfilter(b, a, x)
    return x.astype(np.float32)


def compress(x: np.ndarray, amount: float) -> np.ndarray:
    """Soft saturation as a stand-in for a laptop mic's AGC/limiter."""
    k = 1 + 4 * amount
    peak = float(np.max(np.abs(x))) or 1.0
    y = np.tanh(k * x / peak) / np.tanh(k)
    return (y * peak).astype(np.float32)


def mic_chain(x: np.ndarray, kind: str, r: random.Random) -> tuple[np.ndarray, dict]:
    from scipy.signal import resample_poly

    if kind == "laptop":
        p = {"hp": round(r.uniform(250, 450)), "lp": round(r.uniform(5000, 7000)),
             "presence_hz": round(r.uniform(2000, 4000)), "presence_db": round(r.uniform(2, 6), 1),
             "compress": round(r.uniform(0.2, 0.7), 2)}
        x = filt(x, ("hp", p["hp"]), ("hp", p["hp"]), ("lp", p["lp"]), ("peak", p["presence_hz"], 1.0, p["presence_db"]))
        x = compress(x, p["compress"])
    elif kind == "headset":
        p = {"hp": round(r.uniform(100, 200)), "lp": round(r.uniform(7000, 7800)),
             "presence_db": round(r.uniform(0, 3), 1)}
        x = filt(x, ("hp", p["hp"]), ("lp", p["lp"]), ("peak", 3000, 1.0, p["presence_db"]))
    elif kind == "phone":
        p = {"band": [300, 3400], "codec_rate": 8000}
        x = filt(x, ("hp", 300), ("hp", 300), ("lp", 3400), ("lp", 3400))
        x = resample_poly(resample_poly(x, 1, 2), 2, 1).astype(np.float32)
    elif kind == "usb":
        p = {"tilt_db": round(r.uniform(-3, 3), 1), "hp": 80}
        x = filt(x, ("hp", 80), ("peak", 6000, 0.5, p["tilt_db"]))
    else:
        p = {}
    return x, {"type": kind, **p}


def apply(sid: str, dry: np.ndarray, clean: bool) -> tuple[np.ndarray, dict]:
    r = rng_for(sid, "augment")
    x = dry.astype(np.float32)
    cond: dict = {"clean": clean, "aug_version": 2}
    if not clean:
        if r.random() < 0.5:
            from scipy.signal import resample_poly

            sp = round(r.uniform(0.9, 1.1), 3)
            up = 1000
            x = resample_poly(x, up, int(round(up * sp))).astype(np.float32)
            cond["speed"] = sp
        if r.random() < 0.6 and rir_files():
            from scipy.signal import fftconvolve

            f = r.choice(rir_files())
            h = _load(str(f))
            h = h[int(np.argmax(np.abs(h))):]  # align the direct path
            h = h / (np.max(np.abs(h)) + 1e-8)
            wet = round(r.uniform(0.3, 0.85), 2)
            y = fftconvolve(x, h)[: len(x)].astype(np.float32)
            y *= rms(x) / rms(y)
            x = (wet * y + (1 - wet) * x).astype(np.float32)
            cond["rir"] = {"file": f.stem, "wet": wet}
    lead, tail = r.uniform(0.15, 0.5), r.uniform(0.25, 0.7)
    x = np.concatenate([np.zeros(int(lead * SR), np.float32), x, np.zeros(int(tail * SR), np.float32)])
    cond["preroll_s"], cond["postroll_s"] = round(lead, 2), round(tail, 2)
    if not clean and r.random() < 0.65 and noise_files():
        f = r.choice(noise_files())
        n = _load(str(f))
        if len(n) < len(x):
            n = np.tile(n, int(np.ceil(len(x) / max(1, len(n)))) + 1)
        off = r.randrange(0, max(1, len(n) - len(x)))
        n = n[off: off + len(x)]
        # mostly ordinary rooms; 12% hard cases. (v1 drew 5-30 dB uniformly: TTS speech, unlike
        # human speech, often vanished entirely below ~15 dB.)
        snr = round(r.uniform(5, 10) if r.random() < 0.12 else r.uniform(12, 32), 1)
        n = n * (speech_rms(x) / rms(n)) / db(snr)
        x = (x + n).astype(np.float32)
        cond["noise"] = {"file": f"{f.parent.name}/{f.stem}", "snr_db": snr}
    if not clean:
        kind = r.choices(["laptop", "headset", "phone", "usb", "none"], [0.4, 0.2, 0.1, 0.15, 0.15])[0]
        x, cond["mic"] = mic_chain(x, kind, r)
    peak_db = round(r.uniform(-30, -2) if not clean else r.uniform(-12, -3), 1)
    x = x / (float(np.max(np.abs(x))) or 1.0) * db(peak_db)
    cond["peak_dbfs"] = peak_db
    if not clean and r.random() < 0.04:
        drive = round(r.uniform(3, 9), 1)
        x = np.clip(x * db(drive - peak_db), -1, 1)  # peak `drive` dB over full scale, then clip
        cond["clipped_db"] = drive
    floor = round(r.uniform(-80, -66), 1)
    nr = np.random.default_rng(r.randrange(2**32))
    x = (x + nr.standard_normal(len(x)).astype(np.float32) * db(floor)).astype(np.float32)
    cond["noise_floor_dbfs"] = floor
    return np.clip(x, -1, 1).astype(np.float32), cond


def summary(cond: dict) -> str:
    """Short label for reports: clean / rir / noise<snr band> / mic type / speed."""
    if cond.get("clean"):
        return "clean"
    parts = []
    if "rir" in cond:
        parts.append("room")
    if "noise" in cond:
        s = cond["noise"]["snr_db"]
        parts.append("snr5-12" if s < 12 else "snr12-20" if s < 20 else "snr20-30")
    m = cond.get("mic", {}).get("type", "none")
    if m != "none":
        parts.append(m)
    if "speed" in cond:
        parts.append("speed")
    if "clipped_db" in cond:
        parts.append("clipped")
    return "+".join(parts) or "level-only"

"""Short, quiet UI sounds: start, stop, cancel, error.

The tones are synthesized here (soft sine partials, 4 ms attack, exponential decay, peak about
-22 dBFS) and rendered to ``assets/*.wav`` by ``python -m openwhisprflow.audio.earcons`` so
the files in the repo are reproducible and license-clean. If an asset is missing the tone is
synthesized in memory instead. Playback is fire-and-forget on a daemon thread with its own
output stream, so it never blocks the hotkey path or interferes with mic capture.
"""

from __future__ import annotations

import logging
import threading
import wave
from functools import cache
from pathlib import Path

import numpy as np

log = logging.getLogger("openwhisprflow.audio.earcons")

ASSETS = Path(__file__).with_name("assets")
RATE = 48000
PEAK = 10 ** (-22 / 20)
NAMES = ("start", "stop", "cancel", "error")

# name -> list of (start_s, freq_hz, dur_s, gain): small, warm intervals rather than beeps.
_NOTES: dict[str, list[tuple[float, float, float, float]]] = {
    "start": [(0.00, 659.25, 0.16, 0.8), (0.07, 987.77, 0.20, 1.0)],     # E5 -> B5, rising fifth
    "stop": [(0.00, 987.77, 0.16, 0.9), (0.07, 659.25, 0.22, 1.0)],      # B5 -> E5, falling
    "cancel": [(0.00, 523.25, 0.12, 1.0), (0.05, 392.00, 0.18, 0.8)],    # C5 -> G4, soft drop
    "error": [(0.00, 311.13, 0.14, 1.0), (0.16, 293.66, 0.22, 1.0)],     # Eb4 -> D4, low minor 2nd
}

_enabled = True


def set_enabled(on: bool) -> None:
    global _enabled
    _enabled = bool(on)


def synth(name: str, rate: int = RATE) -> np.ndarray:
    notes = _NOTES[name]
    total = max(s + d for s, _, d, _ in notes) + 0.02
    out = np.zeros(int(total * rate), dtype=np.float64)
    for start, freq, dur, gain in notes:
        t = np.arange(int(dur * rate)) / rate
        attack = np.minimum(1.0, t / 0.004)
        decay = np.exp(-t / (dur / 4.5))
        # Fundamental plus a faint octave: rounder than a pure sine, nowhere near a buzzer.
        tone = np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t)
        i = int(start * rate)
        out[i:i + len(t)] += gain * attack * decay * tone
    out *= PEAK / max(1e-9, float(np.max(np.abs(out))))
    fade = min(len(out), int(0.01 * rate))
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    return out.astype(np.float32)


def render(folder: Path = ASSETS) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in NAMES:
        pcm = (synth(name) * 32767).astype("<i2")
        path = folder / f"{name}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm.tobytes())
        paths.append(path)
    return paths


@cache
def load(name: str) -> tuple[np.ndarray, int]:
    if name not in _NOTES:
        raise ValueError(f"unknown earcon {name!r}")
    path = ASSETS / f"{name}.wav"
    try:
        with wave.open(str(path), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            return pcm, w.getframerate()
    except (OSError, wave.Error):
        return synth(name), RATE


def play(name: str, volume: float = 1.0) -> None:
    """Play an earcon without blocking. Never raises (no output device is not an error)."""
    if not _enabled:
        return
    try:
        pcm, rate = load(name)
    except ValueError:
        log.warning("unknown earcon %r", name)
        return
    threading.Thread(target=_play, args=(pcm * float(volume), rate), name="owf-earcon", daemon=True).start()


def _play(pcm: np.ndarray, rate: int) -> None:
    try:
        import sounddevice as sd

        with sd.OutputStream(samplerate=rate, channels=1, dtype="float32") as stream:
            stream.write(np.ascontiguousarray(pcm.reshape(-1, 1)))
    except Exception as e:
        log.debug("earcon playback failed: %s", e)


if __name__ == "__main__":
    for p in render():
        print(p)

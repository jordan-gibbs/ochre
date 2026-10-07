"""Microphone capture at the device's native rate.

Why native rate: asking PortAudio/WASAPI for 16 kHz makes some drivers resample badly or
refuse to open; we capture what the device delivers and resample per phrase in the segmenter.

The PortAudio callback only copies the block and hands it on; anything heavier (phrase
cutting, metering for the UI) runs in the consumer. ``on_block`` is called on the audio
thread, so it must be quick (``Segmenter.feed`` is: ~50 tiny numpy ops per second). Without
``on_block``, blocks queue up and the caller pulls them with ``drain()``.

Disconnects: PortAudio does not reliably raise when a USB mic is unplugged; the stream just
stops calling back. A watchdog thread treats a 1.5 s callback gap (or an inactive stream) as
a disconnect and reports it through ``on_error``.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("openwhisprflow.audio.capture")

MAX_SECONDS = 600
STALL_S = 1.5
LEVEL_FLOOR_DB, LEVEL_CEIL_DB = -60.0, -12.0


@dataclass
class InputDevice:
    index: int | None          # None = system default
    name: str
    hostapi: str = ""
    channels: int = 1
    default_rate: int = 0
    is_default: bool = False


def level_from_rms(rms: float) -> float:
    """Map block RMS to 0..1 on a dB scale (-60 dBFS -> 0, -12 dBFS -> 1): what the HUD bars use,
    because a linear RMS sits near zero for normal speech."""
    if rms <= 0:
        return 0.0
    db = 20.0 * math.log10(rms)
    return float(min(1.0, max(0.0, (db - LEVEL_FLOOR_DB) / (LEVEL_CEIL_DB - LEVEL_FLOOR_DB))))


def list_devices() -> list[InputDevice]:
    """Input devices, the default first. On Windows each device appears once per host API;
    we keep the default host API's entries (MME on Windows) to avoid a list of duplicates."""
    import sounddevice as sd

    try:
        default_in = sd.default.device[0]
        hosts = sd.query_hostapis()
        default_api = sd.query_devices(default_in)["hostapi"] if default_in is not None and default_in >= 0 else 0
    except Exception:
        return []
    out = [InputDevice(None, "System default microphone", is_default=True)]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and d["hostapi"] == default_api:
            out.append(InputDevice(i, d["name"], hosts[d["hostapi"]]["name"], d["max_input_channels"],
                                   int(d["default_samplerate"]), i == default_in))
    return out


def resolve_device(device: str | int | None) -> int | None:
    """Config stores a device *name* (indexes change across reboots). Exact name first, then a
    case-insensitive substring; unknown names fall back to the default with a warning."""
    if isinstance(device, int):
        return device
    if not device:
        return None
    devices = [d for d in list_devices() if d.index is not None]
    for d in devices:
        if d.name == device:
            return d.index
    low = device.casefold()
    for d in devices:
        if low in d.name.casefold():
            return d.index
    log.warning("input device %r not found; using the system default", device)
    return None


class Capture:
    """One recording session. Not reusable concurrently; call start() again after stop()."""

    def __init__(self, device: str | int | None = None, *,
                 on_block: Callable[[np.ndarray], None] | None = None,
                 on_level: Callable[[float], None] | None = None,
                 max_seconds: float = MAX_SECONDS,
                 on_limit: Callable[[], None] | None = None,
                 on_error: Callable[[str], None] | None = None,
                 level_hz: float = 12.0) -> None:
        self.device = device
        self.on_block, self.on_level = on_block, on_level
        self.on_limit, self.on_error = on_limit, on_error
        self.max_seconds = max_seconds
        self.level_interval = 1.0 / level_hz
        self.rate = 16000
        self.frames = 0
        self.level = 0.0
        self.limit_reached = False
        self.error: str | None = None
        self.warnings: set[str] = set()
        self._queue: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()
        self._stream = None
        self._stop = threading.Event()
        self._last_cb = 0.0
        self._peak = 0.0
        self._watchdog: threading.Thread | None = None

    @property
    def seconds(self) -> float:
        return self.frames / self.rate

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            raise RuntimeError("capture already running")
        index = resolve_device(self.device)
        info = sd.query_devices(index, "input")
        self.rate = int(info["default_samplerate"])
        self.frames, self.level, self._peak = 0, 0.0, 0.0
        self.limit_reached, self.error, self.warnings = False, None, set()
        self._stop.clear()
        limit = int(self.max_seconds * self.rate)

        def callback(data: np.ndarray, frames: int, timing: object, status: object) -> None:
            self._last_cb = time.monotonic()
            if status:
                self.warnings.add(str(status))
            block = data[:max(0, limit - self.frames), 0].copy()
            if block.size:
                self.frames += len(block)
                self._peak = max(self._peak, float(np.sqrt(np.mean(block * block))))
                if self.on_block is not None:
                    try:
                        self.on_block(block)
                    except Exception:
                        log.exception("on_block failed")
                else:
                    self._queue.put(block)
            if self.frames >= limit:
                self.limit_reached = True
                raise sd.CallbackStop()

        stream = sd.InputStream(device=index, channels=1, samplerate=self.rate, dtype="float32",
                                blocksize=0, latency="low", callback=callback)
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._stream = stream
        self._last_cb = time.monotonic()
        self._watchdog = threading.Thread(target=self._watch, name="owf-capture-watch", daemon=True)
        self._watchdog.start()
        log.info("capture started: %s @ %d Hz", info["name"], self.rate)

    def _watch(self) -> None:
        """Level metering for the HUD, the session cap, and disconnect detection."""
        while not self._stop.wait(self.level_interval):
            self.level = level_from_rms(self._peak)
            self._peak = 0.0
            if self.on_level:
                try:
                    self.on_level(self.level)
                except Exception:
                    log.exception("on_level failed")
            if self.limit_reached:
                if self.on_limit:
                    self.on_limit()
                return
            stream = self._stream
            stalled = time.monotonic() - self._last_cb > STALL_S
            if stream is not None and (stalled or not stream.active):
                self.error = "Microphone disconnected"
                log.warning("capture stopped unexpectedly (stalled=%s)", stalled)
                if self.on_error:
                    self.on_error(self.error)
                return

    def drain(self) -> list[np.ndarray]:
        """Blocks captured since the last drain (only when no ``on_block`` was given)."""
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        """Stop and close the stream. Safe to call twice and from callbacks' consumers."""
        self._stop.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                log.debug("stream.stop failed", exc_info=True)
            finally:
                stream.close()
        watchdog = self._watchdog
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=1.0)

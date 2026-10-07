"""Speech-to-text contract shared by every local and cloud engine (SPEC §4.1)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class Word:
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None


@dataclass
class SttResult:
    text: str
    duration_ms: int
    processing_ms: int
    language: str | None = None
    words: list[Word] | None = None
    engine: str = ""


class SttError(RuntimeError):
    """Raised by an engine for a failure the core may fall back from (network, auth, quota, model)."""

    def __init__(self, message: str, *, code: str = "stt_error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass
class LoadProgress:
    """Reported by load() while models download: bytes done / total for one file."""

    file: str
    done: int
    total: int


@runtime_checkable
class SttEngine(Protocol):
    name: str
    kind: Literal["local", "cloud"]
    sample_rate: int

    def load(self, progress: Callable[[LoadProgress], None] | None = None) -> None:
        """Download/verify models (local) or validate configuration (cloud). Idempotent."""

    def transcribe(self, audio: np.ndarray, *, language: str | None = None,
                   prompt: str | None = None) -> SttResult:
        """Mono float32 PCM at self.sample_rate, values in [-1, 1]. Blocking; thread-safe per instance not required."""

    def close(self) -> None: ...



@dataclass
class EngineInfo:
    """Static description for the registry and the settings UI."""

    name: str
    kind: Literal["local", "cloud"]
    label: str
    models: list[str] = field(default_factory=list)
    default_model: str = ""
    needs_key: bool = False
    languages: str = "en"

"""Groq Whisper (OpenAI-compatible ``/openai/v1/audio/transcriptions``), batch per phrase.

Groq runs Whisper on LPUs, so a 10 s phrase usually returns in a few hundred ms.
``whisper-large-v3-turbo`` is the fast/cheap default; ``whisper-large-v3`` is a little more
accurate. Groq bills a minimum of 10 s per request, which matters for short phrases.

Approximate cost (Oct 2026): whisper-large-v3-turbo $0.04/h, whisper-large-v3 $0.111/h
(10 s minimum per request).
"""

from __future__ import annotations

from ..config import SttConfig
from .base import EngineInfo
from .openai import OpenAITranscriber

INFO = EngineInfo(
    name="groq", kind="cloud", label="Groq (Whisper)",
    models=["whisper-large-v3-turbo", "whisper-large-v3"],
    default_model="whisper-large-v3-turbo", needs_key=True, languages="multilingual",
)


class GroqTranscriber(OpenAITranscriber):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


def create(cfg: SttConfig, **kw: object) -> GroqTranscriber:
    return GroqTranscriber(model=cfg.model or INFO.default_model, language=cfg.language,
                           timeout_s=cfg.cloud_timeout_s, **kw)  # type: ignore[arg-type]

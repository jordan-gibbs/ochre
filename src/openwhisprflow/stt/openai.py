"""OpenAI speech-to-text (``POST /v1/audio/transcriptions``), batch per phrase segment.

``gpt-transcribe`` (2026) is the default: in our Oct 2026 smoke test on an 8.5 s clip it
returned in 0.86-1.13 s vs 1.6-2.0 s for ``gpt-4o-mini-transcribe``, and it is the most
accurate GPT transcriber. ``gpt-4o-mini-transcribe`` is the cheapest. Unlike ``whisper-1``
these rarely hallucinate on silence. The ``prompt`` field takes free text, so the personal
dictionary goes there verbatim.

Approximate cost (Oct 2026): gpt-4o-mini-transcribe $0.003/min ($0.18/h), gpt-transcribe
$0.0045/min ($0.27/h), gpt-4o-transcribe $0.006/min ($0.36/h), whisper-1 $0.006/min.
"""

from __future__ import annotations

import httpx
import numpy as np

from ..config import SttConfig
from ._http import CloudEngine, Timer, duration_ms, to_wav
from .base import EngineInfo, SttError, SttResult

INFO = EngineInfo(
    name="openai", kind="cloud", label="OpenAI",
    models=["gpt-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
    default_model="gpt-transcribe", needs_key=True, languages="multilingual",
)


class OpenAITranscriber(CloudEngine):
    """Also the base for every OpenAI-compatible transcription endpoint (Groq)."""

    name = "openai"
    base_url = "https://api.openai.com/v1"
    max_prompt_chars = 800   # whisper-style models read only ~224 tokens of prompt

    def transcribe(self, audio: np.ndarray, *, language: str | None = None,
                   prompt: str | None = None) -> SttResult:
        data: dict[str, str] = {"model": self.model, "response_format": "json", "temperature": "0"}
        if lang := self.lang(language):
            data["language"] = lang
        if prompt:
            data["prompt"] = prompt[: self.max_prompt_chars]
        files = {"file": ("audio.wav", to_wav(audio), "audio/wav")}
        with Timer() as t:
            resp = self.request("POST", f"{self.base_url}/audio/transcriptions", data=data, files=files,
                                headers={"Authorization": f"Bearer {self.api_key()}"})
        return self._result(resp, audio, t.ms, lang)

    def _result(self, resp: httpx.Response, audio: np.ndarray, ms: int, lang: str | None) -> SttResult:
        try:
            body = resp.json()
            text = body["text"]
        except Exception as e:
            raise SttError(f"{self.name}: unexpected response", code="bad_response") from e
        return SttResult(text=text.strip(), duration_ms=duration_ms(audio), processing_ms=ms,
                         language=body.get("language") or lang, engine=self.name)


def create(cfg: SttConfig, **kw: object) -> OpenAITranscriber:
    return OpenAITranscriber(model=cfg.model or INFO.default_model, language=cfg.language,
                             timeout_s=cfg.cloud_timeout_s, **kw)  # type: ignore[arg-type]

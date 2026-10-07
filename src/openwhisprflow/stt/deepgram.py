"""Deepgram pre-recorded STT (``POST /v1/listen``), batch per phrase segment.

Raw WAV bytes go in the body (no multipart, no upload step), which makes this one of the
lowest-latency batch APIs. ``smart_format`` gives punctuation, casing and written-form numbers.
Dictionary words go in as repeated ``keyterm`` params (Nova-3 keyterm prompting; the older
``keywords`` param is Nova-2 only).

Approximate cost (Oct 2026, pay-as-you-go): nova-3 $0.0043/min ($0.26/h) monolingual,
$0.0052/min multilingual; nova-2 $0.0043/min.
"""

from __future__ import annotations

import numpy as np

from ..config import SttConfig
from ._http import CloudEngine, Timer, duration_ms, split_terms, to_wav
from .base import EngineInfo, SttError, SttResult, Word

INFO = EngineInfo(
    name="deepgram", kind="cloud", label="Deepgram",
    models=["nova-3", "nova-3-medical", "nova-2"],
    default_model="nova-3", needs_key=True, languages="multilingual",
)

URL = "https://api.deepgram.com/v1/listen"


class DeepgramTranscriber(CloudEngine):
    name = "deepgram"

    def build_params(self, language: str | None, prompt: str | None) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = [("model", self.model), ("smart_format", "true"),
                                         ("punctuate", "true")]
        lang = self.lang(language)
        params.append(("language", lang) if lang else ("detect_language", "true"))
        if self.model.startswith("nova-3"):
            # keyterm total is capped (~500 tokens); 50 short terms stays well inside it.
            params += [("keyterm", t) for t in split_terms(prompt, max_terms=50)]
        return params

    def transcribe(self, audio: np.ndarray, *, language: str | None = None,
                   prompt: str | None = None) -> SttResult:
        with Timer() as t:
            resp = self.request("POST", URL, params=self.build_params(language, prompt), content=to_wav(audio),
                                headers={"Authorization": f"Token {self.api_key()}",
                                         "Content-Type": "audio/wav"})
        try:
            body = resp.json()
            channel = body["results"]["channels"][0]
            alt = channel["alternatives"][0]
        except Exception as e:
            raise SttError("deepgram: unexpected response", code="bad_response") from e
        words = [Word(text=w.get("punctuated_word") or w["word"], start_ms=int(w["start"] * 1000),
                      end_ms=int(w["end"] * 1000), confidence=w.get("confidence"))
                 for w in alt.get("words") or ()]
        return SttResult(text=(alt.get("transcript") or "").strip(), duration_ms=duration_ms(audio),
                         processing_ms=t.ms, language=channel.get("detected_language") or self.lang(language),
                         words=words or None, engine=self.name)


def create(cfg: SttConfig, **kw: object) -> DeepgramTranscriber:
    return DeepgramTranscriber(model=cfg.model or INFO.default_model, language=cfg.language,
                               timeout_s=cfg.cloud_timeout_s, **kw)  # type: ignore[arg-type]

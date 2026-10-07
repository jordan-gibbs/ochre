"""ElevenLabs Scribe (``POST /v1/speech-to-text``), batch per phrase segment.

The request is multipart; we send raw 16 kHz PCM with ``file_format=pcm_s16le_16``, which
skips server-side decoding (ElevenLabs documents lower latency for it). Audio-event tags
("(laughter)") are turned off because they would be typed into the user's text box.
Dictionary words go in as repeated ``keyterms`` fields.

Approximate cost (Oct 2026): Scribe v2 ~$0.22-0.40/h depending on plan; sending keyterms
adds a 20% surcharge. Scribe v1 similar.
"""

from __future__ import annotations

import numpy as np

from ..config import SttConfig
from ._http import CloudEngine, Timer, duration_ms, split_terms, to_pcm16
from .base import EngineInfo, SttError, SttResult, Word

INFO = EngineInfo(
    name="elevenlabs", kind="cloud", label="ElevenLabs Scribe",
    models=["scribe_v2", "scribe_v1"],
    default_model="scribe_v2", needs_key=True, languages="multilingual",
)

URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsTranscriber(CloudEngine):
    name = "elevenlabs"

    def build_fields(self, language: str | None, prompt: str | None) -> list[tuple[str, str]]:
        fields: list[tuple[str, str]] = [("model_id", self.model), ("file_format", "pcm_s16le_16"),
                                         ("tag_audio_events", "false"), ("timestamps_granularity", "word"),
                                         ("diarize", "false")]
        if lang := self.lang(language):
            fields.append(("language_code", lang))
        # Terms: <50 chars, at most 5 words, none of <>{}[]\ ; over 100 terms bills a 20 s minimum.
        fields += [("keyterms", t) for t in split_terms(prompt, max_terms=100, max_len=49)
                   if len(t.split()) <= 5 and not any(c in t for c in "<>{}[]\\")]
        return fields

    def transcribe(self, audio: np.ndarray, *, language: str | None = None,
                   prompt: str | None = None) -> SttResult:
        files = {"file": ("audio.pcm", to_pcm16(audio), "application/octet-stream")}
        with Timer() as t:
            data: dict[str, list[str]] = {}
            for k, v in self.build_fields(language, prompt):   # httpx sends a list value as repeated fields
                data.setdefault(k, []).append(v)
            resp = self.request("POST", URL, data=data, files=files,
                                headers={"xi-api-key": self.api_key()})
        try:
            body = resp.json()
            text = body["text"]
        except Exception as e:
            raise SttError("elevenlabs: unexpected response", code="bad_response") from e
        words = [Word(text=w["text"], start_ms=int((w.get("start") or 0) * 1000),
                      end_ms=int((w.get("end") or 0) * 1000))
                 for w in body.get("words") or () if w.get("type", "word") == "word"]
        return SttResult(text=text.strip(), duration_ms=duration_ms(audio), processing_ms=t.ms,
                         language=body.get("language_code"), words=words or None, engine=self.name)


def create(cfg: SttConfig, **kw: object) -> ElevenLabsTranscriber:
    return ElevenLabsTranscriber(model=cfg.model or INFO.default_model, language=cfg.language,
                                 timeout_s=cfg.cloud_timeout_s, **kw)  # type: ignore[arg-type]

"""AssemblyAI pre-recorded STT: upload -> submit -> poll, per phrase segment.

AssemblyAI has no synchronous batch endpoint, so a phrase costs three round trips plus queue
time (typically ~1-2 s for a short clip). Its streaming API is billed by socket-open time, which
does not suit phrase batches, so we stay on the async API and poll quickly. The transcript is
deleted after we read it (in the background) because dictation is private by default.

Approximate cost (Oct 2026): universal-3-5-pro $0.21/h, universal-2 $0.15/h.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..config import SttConfig
from ._http import CloudEngine, Timer, duration_ms, split_terms, to_wav
from .base import EngineInfo, SttError, SttResult, Word

log = logging.getLogger("openwhisprflow.stt.assemblyai")

INFO = EngineInfo(
    name="assemblyai", kind="cloud", label="AssemblyAI",
    models=["universal-3-5-pro", "universal-2"],
    default_model="universal-3-5-pro", needs_key=True, languages="multilingual",
)

BASE = "https://api.assemblyai.com/v2"
POLL_S = 0.15


class AssemblyAITranscriber(CloudEngine):
    name = "assemblyai"
    poll_s = POLL_S

    def build_job(self, audio_url: str, language: str | None, prompt: str | None) -> dict[str, object]:
        job: dict[str, object] = {"audio_url": audio_url, "speech_models": [self.model],
                                  "punctuate": True, "format_text": True}
        if lang := self.lang(language):
            job["language_code"] = lang
        else:
            job["language_detection"] = True
        terms = split_terms(prompt, max_terms=100, max_len=50)
        if terms:
            job["keyterms_prompt"] = terms
        return job

    def transcribe(self, audio: np.ndarray, *, language: str | None = None,
                   prompt: str | None = None) -> SttResult:
        auth = {"Authorization": self.api_key()}
        deadline = time.monotonic() + self.timeout_s
        with Timer() as t:
            up = self.request("POST", f"{BASE}/upload", content=to_wav(audio),
                              headers={**auth, "Content-Type": "application/octet-stream"}).json()
            job = self.request("POST", f"{BASE}/transcript", json=self.build_job(up["upload_url"], language, prompt),
                               headers=auth).json()
            tid = job["id"]
            while job.get("status") not in ("completed", "error"):
                if time.monotonic() > deadline:
                    self._delete_later(tid, auth)
                    raise SttError("assemblyai: timed out waiting for transcript", code="timeout", retryable=True)
                time.sleep(self.poll_s)
                job = self.request("GET", f"{BASE}/transcript/{tid}", headers=auth).json()
        self._delete_later(tid, auth)
        if job["status"] == "error":
            raise SttError(f"assemblyai: {job.get('error') or 'transcription failed'}", code="provider_error")
        words = [Word(text=w["text"], start_ms=int(w["start"]), end_ms=int(w["end"]), confidence=w.get("confidence"))
                 for w in job.get("words") or ()]
        return SttResult(text=(job.get("text") or "").strip(), duration_ms=duration_ms(audio), processing_ms=t.ms,
                         language=job.get("language_code"), words=words or None, engine=self.name)

    def _delete_later(self, tid: str, auth: dict[str, str]) -> None:
        """Best-effort privacy cleanup off the latency path."""
        def run() -> None:
            try:
                self.client.delete(f"{BASE}/transcript/{tid}", headers=auth)
            except Exception as e:  # never surface: the text is already delivered
                log.debug("assemblyai delete failed: %s", type(e).__name__)
        threading.Thread(target=run, name="assemblyai-delete", daemon=True).start()


def create(cfg: SttConfig, **kw: object) -> AssemblyAITranscriber:
    return AssemblyAITranscriber(model=cfg.model or INFO.default_model, language=cfg.language,
                                 timeout_s=cfg.cloud_timeout_s, **kw)  # type: ignore[arg-type]

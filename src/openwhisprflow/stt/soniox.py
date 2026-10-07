"""Soniox STT: async REST for batch phrase segments, real-time WebSocket for live streaming.

Measured Oct 2026 on an 8.5 s clip: pushing a finished segment through the real-time socket
takes ~5.9 s (the RT model consumes a burst at only ~1.5x real time), while the async REST flow
(upload -> create -> poll -> fetch) takes 1.8-2.2 s. So ``transcribe()`` uses async REST, and
``stream()`` offers the real-time socket for a core that feeds audio *while* the user speaks;
then release latency is one ``finalize`` round trip (a few hundred ms) instead of a batch job.

Real-time protocol: JSON start frame ->
binary ``pcm_s16le`` frames -> ``{"type": "finalize"}``, answered by a final ``<fin>`` token.
We close after ``<fin>`` rather than sending the documented empty end frame: with the sync
``websockets`` client the server never answered the empty frame with ``finished`` in testing.
Dictionary words go in ``context.terms`` on both paths. Uploaded files and transcripts are
deleted after reading (in the background), because dictation is private by default.

Approximate cost (Oct 2026): async stt-async-v5 $0.10/h; real-time stt-rt-v5 $0.12/h
(billed while the socket is open).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from ..config import SttConfig
from ._http import (
    CloudEngine,
    Timer,
    classify_exception,
    classify_status,
    duration_ms,
    redact,
    split_terms,
    to_pcm16,
    to_wav,
)
from .base import SAMPLE_RATE, EngineInfo, SttError, SttResult, Word

log = logging.getLogger("openwhisprflow.stt.soniox")

INFO = EngineInfo(
    name="soniox", kind="cloud", label="Soniox",
    models=["stt-async-v5"],
    default_model="stt-async-v5", needs_key=True, languages="multilingual",
)

API = "https://api.soniox.com/v1"
WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
RT_MODEL = "stt-rt-v5"
SPECIAL = ("<end>", "<fin>")
POLL_S = 0.1


def _status_error(status: object) -> tuple[str, bool]:
    try:
        return classify_status(int(status))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "provider_error", True


def _context(prompt: str | None) -> dict[str, object] | None:
    terms = split_terms(prompt, max_terms=200)
    return {"terms": terms} if terms else None


class SonioxTranscriber(CloudEngine):
    name = "soniox"
    poll_s = POLL_S

    def build_job(self, file_id: str, language: str | None, prompt: str | None) -> dict[str, object]:
        job: dict[str, object] = {"model": self.model, "file_id": file_id,
                                  "client_reference_id": "openwhisprflow"}
        if lang := self.lang(language):
            job["language_hints"] = [lang]
        if ctx := _context(prompt):
            job["context"] = ctx
        return job

    def transcribe(self, audio: np.ndarray, *, language: str | None = None,
                   prompt: str | None = None) -> SttResult:
        auth = {"Authorization": f"Bearer {self.api_key()}"}
        deadline = time.monotonic() + self.timeout_s
        file_id = tid = None
        try:
            with Timer() as t:
                file_id = self.request("POST", f"{API}/files", headers=auth,
                                       files={"file": ("audio.wav", to_wav(audio), "audio/wav")}).json()["id"]
                job = self.request("POST", f"{API}/transcriptions", headers=auth,
                                   json=self.build_job(file_id, language, prompt)).json()
                tid = job["id"]
                while job.get("status") not in ("completed", "error"):
                    if time.monotonic() > deadline:
                        raise SttError("soniox: timed out waiting for transcript", code="timeout", retryable=True)
                    time.sleep(self.poll_s)
                    job = self.request("GET", f"{API}/transcriptions/{tid}", headers=auth).json()
                if job["status"] == "error":
                    raise SttError(f"soniox: {redact(str(job.get('error_message') or 'transcription failed'))}",
                                   code="provider_error")
                body = self.request("GET", f"{API}/transcriptions/{tid}/transcript", headers=auth).json()
        finally:
            self._cleanup(auth, tid, file_id)
        tokens = [tok for tok in body.get("tokens") or () if tok.get("text") not in SPECIAL]
        return _result(body.get("text") or "", tokens, audio, t.ms, self.name)

    def _cleanup(self, auth: dict[str, str], tid: str | None, file_id: str | None) -> None:
        """Delete the transcript and the uploaded audio off the latency path."""
        if not tid and not file_id:
            return

        def run() -> None:
            for url in ([f"{API}/transcriptions/{tid}"] if tid else []) + ([f"{API}/files/{file_id}"] if file_id else []):
                try:
                    self.client.delete(url, headers=auth)
                except Exception as e:
                    log.debug("soniox cleanup failed: %s", type(e).__name__)
        threading.Thread(target=run, name="soniox-cleanup", daemon=True).start()

    # ------------------------------------------------------------------ live streaming
    def stream(self, *, language: str | None = None, prompt: str | None = None,
               on_partial: Callable[[str, int], None] | None = None) -> SonioxStream:
        """Open a real-time session. Feed audio with ``send()`` as it is captured, then ``finish()``."""
        return SonioxStream(self.api_key(), language=self.lang(language), prompt=prompt,
                            timeout_s=self.timeout_s, on_partial=on_partial)


class SonioxStream:
    """One real-time socket for one dictation. ``send`` is non-blocking enough to call from the
    capture thread (a websocket send of a few KB); tokens are read on a background thread and
    ``on_partial(text, stable_chars)`` is called with final text + the current non-final tail."""

    def __init__(self, key: str, *, language: str | None, prompt: str | None, timeout_s: float,
                 on_partial: Callable[[str, int], None] | None = None, url: str = WS_URL,
                 model: str = RT_MODEL) -> None:
        from websockets.sync.client import connect

        conf: dict[str, object] = {"api_key": key, "model": model, "audio_format": "pcm_s16le",
                                   "sample_rate": SAMPLE_RATE, "num_channels": 1,
                                   "enable_endpoint_detection": False, "client_reference_id": "openwhisprflow"}
        if language:
            conf["language_hints"] = [language]
        if ctx := _context(prompt):
            conf["context"] = ctx
        self.timeout_s = timeout_s
        self.on_partial = on_partial
        self._final: list[dict[str, object]] = []
        self._done = threading.Event()
        self._error: SttError | None = None
        self._samples = 0
        self._t0 = time.perf_counter()
        try:
            self._ws = connect(url, open_timeout=min(5.0, timeout_s), close_timeout=1, max_size=2**22,
                               compression=None, user_agent_header="openwhisprflow")
            self._ws.send(json.dumps(conf))
        except Exception as e:
            raise _ws_error(e) from e
        self._reader = threading.Thread(target=self._read, name="soniox-rt", daemon=True)
        self._reader.start()

    def send(self, audio: np.ndarray) -> None:
        if self._done.is_set():
            return
        self._samples += len(audio)
        try:
            self._ws.send(to_pcm16(audio))
        except Exception as e:
            self._fail(_ws_error(e))

    def finish(self) -> SttResult:
        """Finalize everything sent so far and return the whole transcript."""
        t_release = time.perf_counter()
        if not self._done.is_set():
            try:
                self._ws.send(json.dumps({"type": "finalize"}))
            except Exception as e:
                self._fail(_ws_error(e))
        if not self._done.wait(self.timeout_s):
            self._fail(SttError("soniox: timed out waiting for finalize", code="timeout", retryable=True))
        self.cancel()
        if self._error is not None:
            raise self._error
        audio_ms = int(self._samples * 1000 / SAMPLE_RATE)
        res = _result("", self._final, np.zeros(0), int((time.perf_counter() - t_release) * 1000), "soniox")
        res.duration_ms = audio_ms
        return res

    def cancel(self) -> None:
        self._done.set()
        try:
            self._ws.close()
        except Exception:
            pass

    def _fail(self, err: SttError) -> None:
        if self._error is None:
            self._error = err
        self._done.set()

    def _read(self) -> None:
        try:
            for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("error_code") is not None:
                    code, retryable = _status_error(msg["error_code"])
                    self._fail(SttError(f"soniox {msg['error_code']}: {redact(str(msg.get('error_message')))}",
                                        code=code, retryable=retryable))
                    return
                pending: list[str] = []
                fin = False
                for tok in msg.get("tokens") or ():
                    if tok.get("is_final"):
                        if tok.get("text") == "<fin>":
                            fin = True
                        elif tok.get("text") not in SPECIAL:
                            self._final.append(tok)
                    elif tok.get("text") not in SPECIAL:
                        pending.append(str(tok.get("text", "")))
                if self.on_partial is not None:
                    final_text = "".join(str(t.get("text", "")) for t in self._final)
                    try:
                        self.on_partial((final_text + "".join(pending)).strip(), len(final_text.strip()))
                    except Exception:
                        log.exception("on_partial failed")
                if fin or msg.get("finished"):
                    self._done.set()
                    return
        except Exception as e:
            if not self._done.is_set():
                self._fail(_ws_error(e))


def _ws_error(e: BaseException) -> SttError:
    from websockets.exceptions import InvalidStatus

    if isinstance(e, SttError):
        return e
    if isinstance(e, InvalidStatus):
        code, retryable = classify_status(e.response.status_code)
        return SttError(f"soniox: handshake HTTP {e.response.status_code}", code=code, retryable=retryable)
    code, retryable = classify_exception(e)
    if code == "stt_error":
        code, retryable = "network", True
    return SttError(f"soniox: {type(e).__name__}", code=code, retryable=retryable)


def _result(text: str, tokens: list[dict[str, object]], audio: np.ndarray, ms: int, engine: str) -> SttResult:
    if not text:
        text = "".join(str(tok.get("text", "")) for tok in tokens)
    langs = [tok.get("language") for tok in tokens if tok.get("language")]
    words = _words(tokens)
    return SttResult(text=text.strip(), duration_ms=duration_ms(audio), processing_ms=ms,
                     language=str(langs[0]) if langs else None, words=words or None, engine=engine)


def _words(tokens: list[dict[str, object]]) -> list[Word]:
    """Soniox tokens are sub-word pieces carrying their own leading space; join them into words."""
    words: list[Word] = []
    for tok in tokens:
        text = str(tok.get("text", ""))
        start, end = int(tok.get("start_ms") or 0), int(tok.get("end_ms") or 0)  # type: ignore[call-overload]
        conf = tok.get("confidence")
        if words and not text.startswith(" "):
            words[-1].text += text
            words[-1].end_ms = end
        elif text.strip():
            words.append(Word(text=text.strip(), start_ms=start, end_ms=end,
                              confidence=float(conf) if conf is not None else None))  # type: ignore[arg-type]
    return words


def create(cfg: SttConfig, **kw: object) -> SonioxTranscriber:
    model = cfg.model or INFO.default_model
    if model.startswith("stt-rt"):   # an RT model name in config still means batch via async REST
        model = INFO.default_model
    return SonioxTranscriber(model=model, language=cfg.language, timeout_s=cfg.cloud_timeout_s,
                             **kw)  # type: ignore[arg-type]

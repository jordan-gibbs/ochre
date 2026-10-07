"""Shared plumbing for the cloud STT engines (and the cloud refiners' error mapping).

Every cloud engine does the same four things around one HTTP/WebSocket call: encode a phrase
segment (16 kHz mono float32) as WAV or PCM, fetch the user's key, time the call, and turn a
failure into an ``SttError`` whose ``code``/``retryable`` let the core decide between "retry",
"fall back to local" and "tell the user to fix their key". Keeping that here means each
provider module is only its request shape and its response parsing.
"""

from __future__ import annotations

import io
import re
import time
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx
import numpy as np

from .. import secrets
from .base import SAMPLE_RATE, SttError

USER_AGENT = "openwhisprflow/0.1"

# HTTP status -> (error code, retryable). Codes are shared with the refiners so the UI can map
# one vocabulary to messages: "auth" = fix the key, "quota" = top up / plan limit, the rest are
# transient or a bug on our side.
_STATUS_CODES: dict[int, tuple[str, bool]] = {
    400: ("bad_request", False),
    401: ("auth", False),
    402: ("quota", False),
    403: ("auth", False),
    404: ("not_found", False),
    408: ("timeout", True),
    413: ("too_large", False),
    415: ("bad_request", False),
    422: ("bad_request", False),
    429: ("rate_limit", True),
}


def classify_status(status: int) -> tuple[str, bool]:
    """Map an HTTP status to ``(code, retryable)``."""
    if status in _STATUS_CODES:
        return _STATUS_CODES[status]
    if status >= 500:
        return "server", True
    return "http_error", False


def classify_exception(exc: BaseException) -> tuple[str, bool]:
    """Map a transport exception (httpx / websockets / OS) to ``(code, retryable)``."""
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "timeout", True
    if isinstance(exc, (httpx.TransportError, OSError, ConnectionError)):
        return "network", True
    return "stt_error", False


def error_detail(resp: httpx.Response, limit: int = 300) -> str:
    """A short, key-free description of an error response for logs and the HUD."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error") or body.get("detail") or body.get("message") or body
            if isinstance(err, dict):
                err = err.get("message") or err.get("msg") or err
            text = str(err)
        else:
            text = str(body)
    except Exception:
        text = resp.text
    return redact(text)[:limit]


_KEYISH = re.compile(r"\b(sk|gsk|xi|key|sk-ant|AIza)[-_A-Za-z0-9]{12,}\b")


def redact(text: str) -> str:
    """Belt-and-braces: never let something that looks like an API key reach a log line."""
    return _KEYISH.sub("[redacted]", text)


def raise_for_response(provider: str, resp: httpx.Response) -> None:
    if resp.is_success:
        return
    code, retryable = classify_status(resp.status_code)
    raise SttError(f"{provider}: HTTP {resp.status_code}: {error_detail(resp)}", code=code, retryable=retryable)


def require_key(provider: str) -> str:
    key = secrets.get(provider)
    if not key:
        raise SttError(f"{provider}: no API key set (Settings > Transcription, or the env var)",
                       code="no_key", retryable=False)
    return key


# ---------------------------------------------------------------- audio encoding

def to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] mono -> little-endian int16 bytes (what every provider accepts)."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


def to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Mono 16-bit WAV in memory. ~32 KB/s at 16 kHz, so a 30 s phrase is under 1 MB."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(to_pcm16(audio))
    return buf.getvalue()


def duration_ms(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> int:
    return int(round(len(audio) * 1000 / sample_rate))


# ---------------------------------------------------------------- vocabulary

def split_terms(prompt: str | None, *, max_terms: int = 100, max_len: int = 50) -> list[str]:
    """Turn the core's ``prompt`` into key terms for APIs that take a list (Deepgram keyterm,
    ElevenLabs keyterms, AssemblyAI keyterms_prompt, Soniox context.terms).

    The core passes the personal dictionary as a comma/newline separated string (the same string
    Whisper-style engines take as a free-text prompt), so splitting on those is enough.
    """
    if not prompt:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,\n;]+", prompt):
        t = part.strip()
        if t and len(t) <= max_len and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


# ---------------------------------------------------------------- base class

@dataclass
class Timer:
    start: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    @property
    def ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)


class CloudEngine:
    """Common lifecycle for the HTTP engines: one pooled ``httpx.Client`` per engine so the TLS
    session stays warm between phrases (a cold TLS handshake is 100-300 ms of release latency)."""

    name: str = "cloud"
    kind = "cloud"
    sample_rate: int = SAMPLE_RATE
    key_name: str = ""          # secrets provider id; defaults to ``name``

    def __init__(self, *, model: str, language: str | None, timeout_s: float,
                 transport: httpx.BaseTransport | None = None,
                 key_getter: Callable[[], str] | None = None) -> None:
        self.model = model
        self.language = language
        self.timeout_s = timeout_s
        self._transport = transport
        self._key_getter = key_getter
        self._client: httpx.Client | None = None

    # -- contract
    def load(self, progress: object = None) -> None:
        """Cloud engines have nothing to download; fail early (and cheaply) if the key is missing."""
        self.api_key()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- helpers
    def api_key(self) -> str:
        if self._key_getter is not None:
            return self._key_getter()
        return require_key(self.key_name or self.name)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=min(5.0, self.timeout_s)),
                                        transport=self._transport, headers={"User-Agent": USER_AGENT})
        return self._client

    def request(self, method: str, url: str, **kw: object) -> httpx.Response:
        """One HTTP call with failures mapped to ``SttError``."""
        try:
            resp = self.client.request(method, url, **kw)  # type: ignore[arg-type]
        except httpx.HTTPError as e:
            code, retryable = classify_exception(e)
            raise SttError(f"{self.name}: {type(e).__name__}", code=code, retryable=retryable) from e
        raise_for_response(self.name, resp)
        return resp

    def lang(self, language: str | None) -> str | None:
        """Per-call language wins over the configured one; "auto"/"" mean let the provider detect."""
        lang = language or self.language
        if not lang or lang.lower() == "auto":
            return None
        return lang.split("-")[0].lower() if len(lang) > 3 else lang.lower()


def iter_chunks(data: bytes, size: int) -> Iterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i:i + size]

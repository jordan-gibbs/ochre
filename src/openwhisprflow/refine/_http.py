"""HTTP plumbing shared by the cloud refiners: pooled client, key lookup, error mapping.

Error codes match the STT engines' (``stt._http.classify_status``) so the UI has one vocabulary:
``auth``/``quota``/``no_key`` need the user, ``timeout``/``network``/``rate_limit``/``server`` are
transient, and the core falls back to the raw text for all of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from .. import secrets
from ..stt._http import USER_AGENT, classify_exception, classify_status, error_detail
from .base import RefineError


def require_key(provider: str, *, optional: bool = False) -> str | None:
    key = secrets.get(provider)
    if not key and not optional:
        raise RefineError(f"{provider}: no API key set (Settings > Refinement, or the env var)", code="no_key")
    return key


class HttpRefiner:
    """Base for cloud refiners: one keep-alive client per refiner, so the TLS handshake (100-300 ms)
    is paid once per session rather than on every dictation."""

    name = "cloud"
    kind = "cloud"

    def __init__(self, *, transport: httpx.BaseTransport | None = None,
                 key_getter: Callable[[], str | None] | None = None) -> None:
        self._transport = transport
        self._key_getter = key_getter
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(transport=self._transport, headers={"User-Agent": USER_AGENT},
                                        timeout=10.0)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def post(self, url: str, body: dict[str, Any], *, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        try:
            resp = self.client.post(url, json=body, headers=headers,
                                    timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 3.0)))
        except httpx.HTTPError as e:
            code, _ = classify_exception(e)
            raise RefineError(f"{self.name}: {type(e).__name__}", code=code) from e
        if not resp.is_success:
            code, _ = classify_status(resp.status_code)
            err = RefineError(f"{self.name}: HTTP {resp.status_code}: {error_detail(resp)}", code=code)
            err.status = resp.status_code  # type: ignore[attr-defined]
            raise err
        try:
            return resp.json()
        except ValueError as e:
            raise RefineError(f"{self.name}: non-JSON response", code="bad_response") from e


def max_tokens_for(text: str) -> int:
    """Output cap: the guard rejects anything over 2x the input anyway, so don't pay for (or wait
    on) a runaway generation. ~4 chars/token for English, plus room for punctuation."""
    return max(64, min(4096, int(len(text) / 4 * 2) + 64))

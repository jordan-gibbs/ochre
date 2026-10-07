"""Refinement through the Gemini API (``models/{model}:generateContent``) with plain httpx.

Default ``gemini-3.5-flash-lite``: Google's cheapest/fastest stable model as of Oct 2026 (the
2.5 models are closed to new users). Gemini 3.x models think by default; ``thinkingLevel:
minimal`` keeps that near zero for a cleanup task. Google recommends leaving Gemini 3
``temperature`` at its default (lower values can cause looping), so we only pin it to 0 on
older models. Thought-summary parts (``thought: true``) are skipped when reading the reply.
"""

from __future__ import annotations

from typing import Any

from ..config import RefineConfig
from ..stt.base import EngineInfo
from . import prompts
from ._http import HttpRefiner, max_tokens_for, require_key
from .base import RefineContext, RefineError

INFO = EngineInfo(
    name="gemini", kind="cloud", label="Google Gemini",
    models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"],
    default_model="gemini-3.5-flash-lite", needs_key=True, languages="multilingual",
)

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def generation_config(model: str, max_tokens: int) -> dict[str, Any]:
    gen: dict[str, Any] = {"maxOutputTokens": max_tokens}
    if model.startswith("gemini-3"):
        gen["thinkingConfig"] = {"thinkingLevel": "minimal"}
        gen["maxOutputTokens"] = max_tokens + 512   # headroom in case the model still thinks a little
    else:
        gen["temperature"] = 0
    return gen


class GeminiRefiner(HttpRefiner):
    name = "gemini"

    def __init__(self, *, model: str = "", **kw: Any) -> None:
        super().__init__(**kw)
        self.model = model or INFO.default_model

    def load(self) -> None:
        self._key()

    def _key(self) -> str:
        key = self._key_getter() if self._key_getter else require_key("gemini")
        if not key:
            raise RefineError("gemini: no API key set", code="no_key")
        return key

    def build_body(self, text: str, ctx: RefineContext) -> dict[str, Any]:
        return {"systemInstruction": {"parts": [{"text": prompts.system_prompt(ctx)}]},
                "contents": [{"role": "user", "parts": [{"text": prompts.user_message(text)}]}],
                "generationConfig": generation_config(self.model, max_tokens_for(text))}

    def refine(self, text: str, ctx: RefineContext, *, timeout_s: float) -> str:
        if not text.strip():
            return text
        data = self.post(f"{BASE}/{self.model}:generateContent", self.build_body(text, ctx),
                         headers={"x-goog-api-key": self._key()}, timeout_s=timeout_s)
        try:
            cand = data["candidates"][0]
        except (KeyError, IndexError, TypeError) as e:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise RefineError(f"gemini: no candidate ({reason or 'empty response'})", code="bad_response") from e
        parts = [p.get("text", "") for p in (cand.get("content") or {}).get("parts") or () if not p.get("thought")]
        if not parts:
            raise RefineError(f"gemini: no text (finishReason={cand.get('finishReason')})", code="bad_response")
        return "".join(parts).strip()


def create(cfg: RefineConfig, **kw: Any) -> GeminiRefiner:
    return GeminiRefiner(model=cfg.model, **kw)

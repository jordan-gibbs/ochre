"""Refinement through the Anthropic Messages API (``POST /v1/messages``) with plain httpx.

Default ``claude-haiku-4-5`` ($1/$5 per 1M tokens, Oct 2026): the fastest, cheapest Claude, it
runs without thinking unless asked and accepts ``temperature``. A typical dictation (~600 prompt
+ ~60 output tokens) costs about $0.0009. The newer Sonnet/Opus models are selectable; they
think by default, so we turn that down: Sonnet 5.5 with ``thinking: between_tools`` (its
thinking-off mode) and Opus 5.5 (thinking cannot be disabled) at ``effort: low``. Neither
accepts a custom ``temperature``.
"""

from __future__ import annotations

from typing import Any

from ..config import RefineConfig
from ..stt.base import EngineInfo
from . import prompts
from ._http import HttpRefiner, max_tokens_for, require_key
from .base import RefineContext, RefineError

INFO = EngineInfo(
    name="anthropic", kind="cloud", label="Anthropic (Claude)",
    models=["claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5"],
    default_model="claude-haiku-4-5", needs_key=True, languages="multilingual",
)

URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


def model_params(model: str) -> dict[str, Any]:
    if model.startswith("claude-haiku"):
        return {"temperature": 0}
    if model.startswith("claude-sonnet-5"):
        return {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}
    return {"output_config": {"effort": "low"}}


class AnthropicRefiner(HttpRefiner):
    name = "anthropic"

    def __init__(self, *, model: str = "", **kw: Any) -> None:
        super().__init__(**kw)
        self.model = model or INFO.default_model

    def load(self) -> None:
        self._key()

    def _key(self) -> str:
        key = self._key_getter() if self._key_getter else require_key("anthropic")
        if not key:
            raise RefineError("anthropic: no API key set", code="no_key")
        return key

    def build_body(self, text: str, ctx: RefineContext) -> dict[str, Any]:
        thinking_room = 0 if self.model.startswith("claude-haiku") else 2048
        return {"model": self.model, "max_tokens": max_tokens_for(text) + thinking_room,
                "system": prompts.system_prompt(ctx),
                "messages": [{"role": "user", "content": prompts.user_message(text)}],
                **model_params(self.model)}

    def refine(self, text: str, ctx: RefineContext, *, timeout_s: float) -> str:
        if not text.strip():
            return text
        headers = {"x-api-key": self._key(), "anthropic-version": API_VERSION}
        data = self.post(URL, self.build_body(text, ctx), headers=headers, timeout_s=timeout_s)
        if data.get("stop_reason") == "refusal":
            raise RefineError("anthropic: request declined", code="refusal")
        parts = [b.get("text", "") for b in data.get("content") or () if b.get("type") == "text"]
        if not parts:
            raise RefineError("anthropic: no text in response", code="bad_response")
        return "".join(parts).strip()


def create(cfg: RefineConfig, **kw: Any) -> AnthropicRefiner:
    return AnthropicRefiner(model=cfg.model, **kw)

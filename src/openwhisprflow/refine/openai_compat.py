"""Refinement through any OpenAI-compatible ``/chat/completions`` endpoint.

One module serves OpenAI, Groq, OpenRouter, Cerebras and any custom base URL (Together,
Fireworks, a self-hosted vLLM, Ollama, LM Studio, ...). Providers differ only in base URL, key
and default model, plus one wrinkle: reasoning models reject ``temperature`` and need a
``reasoning_effort`` so they don't think for a second before cleaning one sentence.

Defaults (Oct 2026), picked for latency first, then cost (~$ per 1M input/output tokens):
- openai:     gpt-4.1-nano ($0.10/$0.40), non-reasoning, so no thinking delay.
- groq:       openai/gpt-oss-20b ($0.075/$0.30, ~1000 tok/s) with reasoning_effort=low.
- openrouter: openai/gpt-4.1-nano (same model via OpenRouter; any slug works).
- cerebras:   gpt-oss-120b with reasoning_effort=low.
A typical dictation is ~600 prompt tokens + ~60 output tokens, i.e. $0.0001 or less per
refinement with the defaults above.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import RefineConfig
from ..stt.base import EngineInfo
from . import prompts
from ._http import HttpRefiner, max_tokens_for, require_key
from .base import RefineContext, RefineError


@dataclass(frozen=True)
class Preset:
    label: str
    base_url: str
    default_model: str
    models: list[str] = field(default_factory=list)
    key_optional: bool = False


PRESETS: dict[str, Preset] = {
    "openai": Preset("OpenAI", "https://api.openai.com/v1", "gpt-4.1-nano",
                     ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5-nano"]),
    "groq": Preset("Groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-20b",
                   ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]),
    "openrouter": Preset("OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-4.1-nano",
                         ["openai/gpt-4.1-nano", "google/gemini-3.5-flash-lite", "anthropic/claude-haiku-4-5",
                          "openai/gpt-oss-20b"]),
    "cerebras": Preset("Cerebras", "https://api.cerebras.ai/v1", "gpt-oss-120b", ["gpt-oss-120b", "llama3.1-8b"]),
    "custom": Preset("Custom (OpenAI-compatible)", "", "", [], key_optional=True),
}

INFOS: dict[str, EngineInfo] = {
    name: EngineInfo(name=name, kind="cloud", label=p.label, models=p.models, default_model=p.default_model,
                     needs_key=not p.key_optional, languages="multilingual")
    for name, p in PRESETS.items()
}
INFO = INFOS["custom"]

_OPENAI_REASONING = re.compile(r"^(?:openai/)?(?:o\d|gpt-5)")
_GPT_OSS = re.compile(r"gpt-oss")


def model_params(model: str) -> dict[str, Any]:
    """Sampling / reasoning params a given model accepts."""
    if _OPENAI_REASONING.match(model):
        # gpt-5.x accept "none" (no reasoning at all); the original gpt-5 family's lowest is "minimal".
        effort = "minimal" if re.match(r"^(?:openai/)?gpt-5-", model) or model.endswith("gpt-5") else "none"
        if re.match(r"^(?:openai/)?o\d", model):
            effort = "low"
        return {"reasoning_effort": effort}
    if _GPT_OSS.search(model):
        return {"reasoning_effort": "low", "temperature": 0}
    return {"temperature": 0}


def is_reasoning(model: str) -> bool:
    return bool(_OPENAI_REASONING.match(model) or _GPT_OSS.search(model))


class OpenAICompatRefiner(HttpRefiner):
    kind = "cloud"

    def __init__(self, provider: str, *, model: str = "", base_url: str = "", kind: str = "cloud",
                 **kw: Any) -> None:
        super().__init__(**kw)
        preset = PRESETS.get(provider, PRESETS["custom"])
        self.name = provider
        self.kind = kind  # type: ignore[misc]  # "local" when pointing at Ollama / LM Studio
        self.preset = preset
        self.base_url = (base_url or preset.base_url).rstrip("/")
        self.model = model or preset.default_model
        if not self.base_url:
            raise RefineError(f"{provider}: set refine.base_url", code="bad_config")
        if not self.model:
            raise RefineError(f"{provider}: set refine.model", code="bad_config")

    def load(self) -> None:
        self._key()

    def _key(self) -> str | None:
        if self._key_getter is not None:
            return self._key_getter()
        return require_key(self.name, optional=self.preset.key_optional or self.kind == "local")

    def build_body(self, text: str, ctx: RefineContext) -> dict[str, Any]:
        params = model_params(self.model)
        cap = max_tokens_for(text) + (1024 if is_reasoning(self.model) else 0)
        token_field = "max_completion_tokens" if "api.openai.com" in self.base_url else "max_tokens"
        return {"model": self.model,
                "messages": [{"role": "system", "content": prompts.system_prompt(ctx)},
                             {"role": "user", "content": prompts.user_message(text)}],
                token_field: cap, "stream": False, **params}

    def refine(self, text: str, ctx: RefineContext, *, timeout_s: float) -> str:
        if not text.strip():
            return text
        headers = {}
        if key := self._key():
            headers["Authorization"] = f"Bearer {key}"
        body = self.build_body(text, ctx)
        try:
            data = self.post(f"{self.base_url}/chat/completions", body, headers=headers, timeout_s=timeout_s)
        except RefineError as e:
            # A model that rejects one of our optional params: retry once without them.
            if getattr(e, "status", None) == 400 and re.search(r"temperature|reasoning", str(e), re.I):
                for k in ("temperature", "reasoning_effort"):
                    body.pop(k, None)
                data = self.post(f"{self.base_url}/chat/completions", body, headers=headers, timeout_s=timeout_s)
            else:
                raise
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RefineError(f"{self.name}: unexpected response", code="bad_response") from e
        return content.strip()


def create(cfg: RefineConfig, **kw: Any) -> OpenAICompatRefiner:
    return OpenAICompatRefiner(cfg.provider if cfg.provider in PRESETS else "custom", model=cfg.model,
                               base_url=cfg.base_url, **kw)

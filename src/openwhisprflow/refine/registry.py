"""Refinement provider registry and the one-call ``refine_text`` the core uses (SPEC §5).

Provider modules are imported lazily, so selecting "off" or "openai" never imports the
llama-server manager, and a broken optional provider cannot break the others.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..config import RefineConfig
from ..stt.base import EngineInfo
from .base import RefineContext, RefineError, Refiner
from .guard import check, tidy

log = logging.getLogger("openwhisprflow.refine")

# provider name -> module under openwhisprflow.refine
PROVIDERS: dict[str, str] = {
    "local": "local",
    "openai": "openai_compat",
    "groq": "openai_compat",
    "openrouter": "openai_compat",
    "cerebras": "openai_compat",
    "custom": "openai_compat",
    "anthropic": "anthropic",
    "gemini": "gemini",
}


def _module(provider: str) -> Any:
    if provider not in PROVIDERS:
        raise RefineError(f"unknown refinement provider {provider!r}", code="bad_config")
    return importlib.import_module(f"{__package__}.{PROVIDERS[provider]}")


def info(provider: str) -> EngineInfo:
    mod = _module(provider)
    return getattr(mod, "INFOS", {}).get(provider) or mod.INFO


def available() -> list[EngineInfo]:
    """Every provider the settings UI can offer, in display order."""
    out = []
    for name in PROVIDERS:
        try:
            out.append(info(name))
        except Exception:
            log.exception("refine provider %s failed to import", name)
    return out


def create(cfg: RefineConfig, **kw: Any) -> Refiner | None:
    """Build (but do not load) the configured refiner; None when refinement is off."""
    if cfg.provider in ("", "off") or cfg.mode == "off":
        return None
    return _module(cfg.provider).create(cfg, **kw)


def context_for(cfg: RefineConfig, *, app_name: str = "", window_title: str = "",
                dictionary: list[str] | None = None, language: str | None = None) -> RefineContext:
    """Build the per-dictation context: mode from config, style from ``app_styles`` matched against
    the focused app's process name ("Slack.exe" -> "slack" -> "casual")."""
    app = app_name.lower().removesuffix(".exe").removesuffix(".app")
    style = cfg.app_styles.get(app, "")
    if not style:
        style = next((v for k, v in cfg.app_styles.items() if k and k in app), "")
    mode = cfg.mode if cfg.mode in ("clean", "polish", "off") else "clean"
    return RefineContext(mode=mode, app_name=app_name, window_title=window_title, style=style,  # type: ignore[arg-type]
                         dictionary=list(dictionary or []), language=language)


def timeout_for(cfg: RefineConfig, refiner: Refiner) -> float:
    return (cfg.timeout_ms_local if refiner.kind == "local" else cfg.timeout_ms_cloud) / 1000.0


@dataclass
class Outcome:
    text: str            # what to insert
    refined: bool        # False -> raw text was used
    reason: str | None   # why raw was used: guard rule, error code, or None
    ms: int


def refine_text(refiner: Refiner | None, raw: str, ctx: RefineContext, *, timeout_s: float) -> Outcome:
    """Refine with every safety net: errors and timeouts return raw, and so does any output the
    guard rejects. Never raises, so the caller can always insert *something*."""
    t0 = time.perf_counter()
    if refiner is None or ctx.mode == "off" or not raw.strip():
        return Outcome(raw, False, None, 0)
    try:
        out = refiner.refine(raw, ctx, timeout_s=timeout_s)
    except RefineError as e:
        log.warning("refinement failed (%s): %s", e.code, e)
        return Outcome(raw, False, e.code, int((time.perf_counter() - t0) * 1000))
    except Exception as e:  # a provider bug must not lose the user's dictation
        log.exception("refinement crashed")
        return Outcome(raw, False, type(e).__name__, int((time.perf_counter() - t0) * 1000))
    ms = int((time.perf_counter() - t0) * 1000)
    if ms > timeout_s * 1000:
        return Outcome(raw, False, "timeout", ms)
    reason = check(raw, out, mode=ctx.mode)
    if reason:
        log.info("refinement rejected by guard: %s", reason)
        return Outcome(raw, False, reason, ms)
    return Outcome(tidy(out), True, None, ms)

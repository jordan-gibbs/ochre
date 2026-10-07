"""Engine lookup by name. Modules are imported lazily so a missing optional dependency (or a
cloud module that is not written yet) hides one engine instead of breaking startup.

Convention: ``openwhisprflow.stt.<name>`` exposes ``INFO: EngineInfo`` and
``create(cfg: SttConfig) -> SttEngine``; it may also expose ``is_available() -> bool``.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

from openwhisprflow.config import SttConfig
from openwhisprflow.stt.base import EngineInfo, SttEngine, SttError

log = logging.getLogger("openwhisprflow.stt.registry")

LOCAL = ("parakeet", "whisper")
CLOUD = ("soniox", "deepgram", "openai", "groq", "elevenlabs", "assemblyai")
KNOWN = LOCAL + CLOUD


def _module(name: str) -> ModuleType | None:
    if not name.isidentifier():
        return None
    try:
        return importlib.import_module(f"openwhisprflow.stt.{name}")
    except ImportError as e:
        log.debug("stt engine %s unavailable: %s", name, e)
        return None


def _usable(mod: ModuleType | None) -> bool:
    if mod is None or not hasattr(mod, "INFO") or not hasattr(mod, "create"):
        return False
    check = getattr(mod, "is_available", None)
    try:
        return bool(check()) if callable(check) else True
    except Exception:
        return False


def available() -> list[EngineInfo]:
    """Engines that can be created on this machine, local first, in a stable order."""
    return [mod.INFO for name in KNOWN if _usable(mod := _module(name))]  # type: ignore[union-attr]


def info(name: str) -> EngineInfo | None:
    mod = _module(name)
    return getattr(mod, "INFO", None) if mod else None


def create(cfg: SttConfig) -> SttEngine:
    """Instantiate the engine named by ``cfg.engine`` (not loaded yet: call ``load()``)."""
    mod = _module(cfg.engine)
    if not _usable(mod):
        raise SttError(f"speech engine {cfg.engine!r} is not available", code="engine_missing")
    return mod.create(cfg)  # type: ignore[union-attr]

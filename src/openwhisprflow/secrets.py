"""API keys: OS keyring first, environment variables as fallback (SPEC §8). Never logged."""

from __future__ import annotations

import os

import keyring
from keyring.errors import KeyringError

SERVICE = "openwhisprflow"

# provider -> env vars checked (in order) after OWF_<PROVIDER>_API_KEY
ENV_FALLBACKS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "soniox": ("SONIOX_API_KEY",),
    "deepgram": ("DEEPGRAM_API_KEY",),
    "elevenlabs": ("ELEVENLABS_API_KEY", "XI_API_KEY"),
    "assemblyai": ("ASSEMBLYAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def get(provider: str) -> str | None:
    try:
        value = keyring.get_password(SERVICE, provider)
        if value:
            return value
    except KeyringError:
        pass
    for var in (f"OWF_{provider.upper()}_API_KEY", *ENV_FALLBACKS.get(provider, ())):
        if os.environ.get(var):
            return os.environ[var]
    return None


def store(provider: str, key: str | None) -> None:
    """Save a key to the OS keyring; an empty key deletes it."""
    if key:
        keyring.set_password(SERVICE, provider, key)
        return
    try:
        keyring.delete_password(SERVICE, provider)
    except KeyringError:
        pass


def has(provider: str) -> bool:
    return get(provider) is not None

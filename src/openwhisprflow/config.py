"""Typed config stored as TOML in the user config dir (SPEC §8). Unknown keys are preserved."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import platformdirs
import tomli_w

APP = "openwhisprflow"


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir(APP, appauthor=False))


def data_dir() -> Path:
    return Path(platformdirs.user_data_dir(APP, appauthor=False))


def models_dir() -> Path:
    return data_dir() / "models"


@dataclass
class HotkeyConfig:
    key: str = "right_alt"             # resolves to Right Option on macOS
    double_tap_ms: int = 350           # second press within this window locks recording
    hold_min_ms: int = 250             # a shorter press-release counts as a tap
    raw_modifier: str = "shift"        # held at finish: skip refinement for this one
    cancel_key: str = "escape"


@dataclass
class AudioConfig:
    device: str | None = None          # None = system default input
    max_session_s: int = 600
    earcons: bool = True


@dataclass
class SttConfig:
    engine: str = "parakeet"           # parakeet | whisper | soniox | deepgram | openai | groq | elevenlabs | assemblyai
    model: str = ""                    # "" = engine default
    language: str | None = None        # None = auto / engine default
    device: str = "auto"               # auto | cpu | cuda | directml | coreml
    cloud_timeout_s: float = 8.0
    fallback_local: bool = True        # cloud failure -> local engine when its models are present


@dataclass
class RefineConfig:
    provider: str = "off"              # off | local | openai | groq | openrouter | anthropic | gemini | custom
    mode: str = "clean"                # clean | polish
    model: str = ""
    base_url: str = ""                 # custom / external OpenAI-compatible servers
    timeout_ms_local: int = 1500
    timeout_ms_cloud: int = 3000
    app_styles: dict[str, str] = field(default_factory=lambda: {
        "slack": "casual", "discord": "casual", "outlook": "formal", "mail": "formal",
        "code": "literal", "windowsterminal": "literal", "terminal": "literal", "iterm2": "literal",
    })


@dataclass
class InjectConfig:
    method: str = "type"               # type | paste
    paste_over_chars: int = 2000
    join_window_s: float = 20.0
    trailing_space: bool = True


@dataclass
class HandsFreeConfig:
    enabled: bool = False
    phrase: str = "transcribe"
    model: str = ""                    # "" = bundled model for the phrase
    threshold: float = 0.55
    preroll_s: float = 1.5
    idle_timeout_s: float = 45.0
    pause_on_calls: bool = True


@dataclass
class DictionaryConfig:
    words: list[str] = field(default_factory=list)              # preferred spellings / vocabulary
    replacements: dict[str, str] = field(default_factory=dict)  # whole phrase "said" -> "written"


@dataclass
class UiConfig:
    enabled: bool = True
    theme: str = "system"              # system | light | dark
    ws_port: int = 8765
    show_partials: bool = True
    start_at_login: bool = False


@dataclass
class HistoryConfig:
    enabled: bool = True
    keep_days: int = 90


@dataclass
class DebugConfig:
    keep_audio: bool = False
    log_level: str = "INFO"


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)
    handsfree: HandsFreeConfig = field(default_factory=HandsFreeConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    extra: dict[str, Any] = field(default_factory=dict)  # unknown top-level keys, round-tripped

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        extra = d.pop("extra")
        return _strip_none({**extra, **d})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        cfg = cls()
        for key, value in raw.items():
            section = getattr(cfg, key, None)
            if dataclasses.is_dataclass(section) and isinstance(value, dict):
                names = {f.name for f in dataclasses.fields(section)}
                for k, v in value.items():
                    if k in names:
                        setattr(section, k, v)
            elif key != "extra":
                cfg.extra[key] = value
        return cfg

    def patch(self, patch: dict[str, Any]) -> Config:
        """Deep-merge a partial dict (as sent by the UI) and return a new Config."""
        return Config.from_dict(_deep_merge(self.to_dict(), patch))


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _strip_none(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_none(v) for k, v in d.items() if v is not None}
    return d


def config_path() -> Path:
    return config_dir() / "config.toml"


def load(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        return Config.from_dict(tomllib.load(f))


def save(cfg: Config, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(tomli_w.dumps(cfg.to_dict()).encode())
    tmp.replace(path)

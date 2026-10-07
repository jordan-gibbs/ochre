"""Event bus and the core <-> UI protocol (SPEC §7.1). Single source of truth for message shapes.

Core -> UI messages are {"event": <name>, ...fields}. UI -> core messages are {"op": <name>, ...fields}.
The ws server (ui_bridge/server.py) forwards every bus event to connected UIs and turns UI ops into
calls on the app via the handler registered with Bus.on_command.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import StrEnum
from typing import Any

log = logging.getLogger("openwhisprflow.events")


class State(StrEnum):
    LOADING = "loading"          # models loading / downloading (download events carry progress)
    IDLE = "idle"                # ready; hands-free may be armed (see `mode`)
    RECORDING = "recording"      # hold-to-talk
    LOCKED = "locked"            # double-tapped: recording until a tap
    HANDSFREE = "handsfree"      # wake word session, ends on "transcribe stop/send"
    TRANSCRIBING = "transcribing"
    REFINING = "refining"
    INSERTING = "inserting"
    ERROR = "error"


# Core -> UI events and their fields (documentation + light validation in tests).
EVENTS: dict[str, tuple[str, ...]] = {
    "hello": ("version", "platform"),
    "state": ("state", "trigger", "handsfree_armed", "detail"),  # trigger: hotkey|wake|ui|tray
    "level": ("rms",),                                          # 0..1 normalized mic level, ~12/s
    "partial": ("text", "stable_chars"),                        # live text for the HUD bubble
    "result": ("id", "raw", "text", "inserted", "refined", "timings"),  # timings: {stt_ms, refine_ms, total_ms}
    "error": ("message", "code"),
    "notice": ("message",),
    "download": ("item", "done", "total"),                      # model/binary download progress (bytes)
    "config": ("config", "secrets"),                            # secrets: {provider: bool} presence only
    "history": ("items",),
    "test_result": ("stage", "provider", "ok", "message", "ms"),
    "engines": ("stt", "refine"),                               # available engines/providers for the UI
}

COMMANDS: dict[str, tuple[str, ...]] = {
    "start": (), "stop": (), "cancel": (), "toggle": (),
    "get_config": (), "set_config": ("patch",),
    "set_secret": ("provider", "key"),
    "test_provider": ("stage", "provider"),
    "download_model": ("stage", "name"),
    "history_query": ("q", "limit"),
    "insert_text": ("text",),
    "train_wake": ("word",),
    "set_handsfree": ("enabled",),
    "open_settings": (),
    "quit": (),
}

Listener = Callable[[dict[str, Any]], None]
CommandHandler = Callable[[dict[str, Any]], None]


class Bus:
    """Thread-safe fan-out of core events; one command handler (the app)."""

    def __init__(self) -> None:
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self._command_handler: CommandHandler | None = None
        self.last_state: dict[str, Any] | None = None

    def subscribe(self, fn: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._listeners:
                    self._listeners.remove(fn)
        return unsubscribe

    def emit(self, event: str, **fields: Any) -> None:
        msg = {"event": event, **fields}
        if event == "state":
            self.last_state = msg
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(msg)
            except Exception:  # a broken listener must never break the pipeline
                log.exception("event listener failed for %s", event)

    def on_command(self, handler: CommandHandler) -> None:
        self._command_handler = handler

    def command(self, msg: dict[str, Any]) -> None:
        if msg.get("op") not in COMMANDS:
            log.warning("unknown command %r", msg.get("op"))
            return
        if self._command_handler:
            self._command_handler(msg)

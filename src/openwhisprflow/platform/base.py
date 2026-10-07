"""Platform contracts: hotkey gestures, text injection, focused-app lookup (SPEC §6)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class Gesture(StrEnum):
    PRESS = "press"              # Voice key went down: start recording (hold mode)
    RELEASE = "release"          # Voice key released after a hold: finish
    LOCK = "lock"                # double tap: recording continues hands-off
    FINISH = "finish"            # single tap while locked: finish
    CANCEL = "cancel"            # Escape while recording
    RAW = "raw"                  # modifier held at finish: skip refinement for this one


GestureFn = Callable[[Gesture], None]


class HotkeyListener(Protocol):
    def start(self, on_gesture: GestureFn) -> None: ...
    def set_key(self, key: str) -> None: ...      # "right_alt", "right_ctrl", "caps_lock", "f13", ...
    def set_recording(self, recording: bool) -> None: ...  # lets Escape be swallowed only while recording
    def stop(self) -> None: ...


@dataclass
class FocusInfo:
    app_name: str = ""           # "slack", "chrome", "code" (lowercased executable/bundle name)
    window_title: str = ""
    window_id: str = ""          # opaque, for the leading-space join rule
    elevated: bool = False       # Windows: target runs as admin and we do not


class Injector(Protocol):
    def focus(self) -> FocusInfo: ...
    def type_text(self, text: str) -> None: ...   # raises InjectError
    def paste_text(self, text: str) -> None: ...  # clipboard save/restore fallback
    def press_enter(self) -> None: ...


class InjectError(RuntimeError):
    pass

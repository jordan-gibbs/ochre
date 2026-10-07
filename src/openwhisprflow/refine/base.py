"""Refinement contract (SPEC §5.1). Refiners clean dictated text; they never answer it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

RefineMode = Literal["off", "clean", "polish"]


@dataclass
class RefineContext:
    mode: RefineMode = "clean"
    app_name: str = ""
    window_title: str = ""
    style: str = ""                      # per-app style hint, e.g. "casual", "formal", "literal"
    dictionary: list[str] = field(default_factory=list)
    language: str | None = None


class RefineError(RuntimeError):
    def __init__(self, message: str, *, code: str = "refine_error"):
        super().__init__(message)
        self.code = code


@runtime_checkable
class Refiner(Protocol):
    name: str
    kind: Literal["local", "cloud"]

    def load(self) -> None: ...

    def refine(self, text: str, ctx: RefineContext, *, timeout_s: float) -> str:
        """Return refined text or raise RefineError. The core applies the safety guard (refine.guard)."""

    def close(self) -> None: ...

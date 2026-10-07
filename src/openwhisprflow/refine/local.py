"""The ``local`` refiner: Quill (or our own fine-tune) on a managed llama-server (SPEC §5.2).

Prompting follows the Quill model card: ChatML rendered by us and sent to the raw
``/completion`` endpoint, the assistant turn pre-seeded with an empty think block, greedy
decoding, no ``--jinja``. The *system prompt* does not follow the card: with Quill's own one-liner
("You clean up dictated text." + the bare transcript) both 0.8B and 2B answered dictated
questions and requests in ``scripts/bench_refine.py``, so every model gets the shared
``prompts.py`` prompt with the ``<dictation>`` tags (``prompt_style="quill"`` restores the card's).

After each refinement the slot is re-primed with the fixed system-prompt prefix in the background
(see ``prompts.chatml_prefix``): Qwen3.5 is a hybrid recurrent model and llama.cpp cannot reuse
a prefix of a longer cached prompt, so without this every dictation re-processes ~370 tokens.

Quill 0.8B is verbatim-only, so output always goes through ``text.normalize`` for times,
numbers, emails and URLs (idempotent and conservative, so it runs for every local model).

If ``refine.base_url`` is set, this refiner instead talks to that existing OpenAI-compatible
server (Ollama, LM Studio, a llama-server you run yourself) through ``/chat/completions``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ..config import RefineConfig
from ..stt.base import EngineInfo
from ..text.normalize import normalize
from . import local_server, prompts
from .base import RefineContext, RefineError
from .guard import tidy

log = logging.getLogger("openwhisprflow.refine.local")

INFO = EngineInfo(
    name="local", kind="local", label="Local (Quill on llama.cpp)",
    models=list(local_server.QUILL), default_model=local_server.DEFAULT_MODEL, needs_key=False, languages="en",
)

Progress = Callable[[str, int, int], None]


class LocalRefiner:
    name = "local"
    kind = "local"

    def __init__(self, cfg: RefineConfig, *, accel: str | None = None,
                 options: local_server.ServerOptions | None = None, progress: Progress | None = None,
                 cancel: Callable[[], bool] | None = None) -> None:
        self.cfg = cfg
        self.model_name = cfg.model or local_server.DEFAULT_MODEL
        # Until RefineConfig grows these fields, env vars let power users pick the build/threads.
        self.accel = accel or os.environ.get("OWF_LLAMA_ACCEL", "auto")
        self.options = options or local_server.ServerOptions()
        if threads := os.environ.get("OWF_LLAMA_THREADS"):
            self.options.threads = int(threads)
        self.progress = progress
        self.cancel = cancel
        self.server: local_server.LlamaServer | None = None
        self._external = None
        if cfg.base_url:
            from .openai_compat import OpenAICompatRefiner
            self._external = OpenAICompatRefiner("custom", model=cfg.model or "default", base_url=cfg.base_url,
                                                 kind="local")

    @property
    def quill(self) -> bool:
        return "quill" in Path(self.model_name).name.lower()

    # -- lifecycle
    def load(self) -> None:
        """Download (first run) and start the server, then warm the prompt cache. Blocking: seconds
        when everything is on disk, minutes on first run. Idempotent."""
        if self._external is not None:
            self._external.load()
            return
        if self.server is not None and self.server.alive():
            return
        model = local_server.ensure_model(self.model_name, progress=self.progress, cancel=self.cancel)
        try:
            self.server = self._start(model, self.accel)
        except Exception as e:
            if local_server.pick_accel(self.accel) == "cpu":
                raise RefineError(f"local: llama-server failed to start: {e}", code="server_failed") from e
            log.warning("accelerated llama-server failed (%s); falling back to the CPU build", e)
            self.server = self._start(model, "cpu")
        self._warm()

    def _start(self, model: Path, accel: str) -> local_server.LlamaServer:
        exe, got = local_server.ensure_binary(accel, progress=self.progress, cancel=self.cancel)
        server = local_server.LlamaServer(exe, model, accel=got, options=self.options)
        server.start()
        return server

    def _warm(self) -> None:
        """One tiny request so the first real dictation doesn't pay for CUDA graph setup and so the
        system-prompt prefix is already in the KV cache."""
        try:
            self.refine("okay", RefineContext(), timeout_s=30)
        except Exception as e:
            log.warning("warm-up refinement failed: %s", e)

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
        if self._external is not None:
            self._external.close()

    # -- inference
    prompt_style = "shared"   # "quill" = the model card's one-line prompt (answers questions; see docstring)

    def system(self, ctx: RefineContext) -> str:
        return prompts.QUILL_SYSTEM if self.prompt_style == "quill" else prompts.system_prompt(ctx)

    def prompt(self, text: str, ctx: RefineContext) -> str:
        if self.prompt_style == "quill":
            return prompts.chatml(prompts.QUILL_SYSTEM, text.strip())
        return prompts.chatml(prompts.system_prompt(ctx), prompts.user_message(text))

    def _prime(self, server: local_server.LlamaServer, ctx: RefineContext) -> None:
        """Leave the slot's state ending exactly at the system prefix, so the next dictation only
        processes its own tokens. Runs on a daemon thread after the result is returned."""
        def run() -> None:
            try:
                server.completion(prompts.chatml_prefix(self.system(ctx)), n_predict=0, timeout=30)
            except Exception as e:
                log.debug("prime failed: %s", e)
        threading.Thread(target=run, name="llama-prime", daemon=True).start()

    def refine(self, text: str, ctx: RefineContext, *, timeout_s: float) -> str:
        if not text.strip():
            return text
        if self._external is not None:
            return normalize(self._external.refine(text, ctx, timeout_s=timeout_s))
        server = self.server
        if server is None:
            raise RefineError("local: refiner not loaded", code="not_loaded")
        if not server.ensure_running(wait=False):
            raise RefineError("local: llama-server is restarting", code="unavailable")
        try:
            data = self._complete(server, text, ctx, timeout_s)
        except httpx.TimeoutException as e:
            raise RefineError("local: timed out", code="timeout") from e
        except httpx.HTTPError as e:
            server.ensure_running(wait=False)
            raise RefineError(f"local: {type(e).__name__}", code="network") from e
        self._prime(server, ctx)
        return normalize(clean_output(str(data.get("content", ""))))

    def _complete(self, server: local_server.LlamaServer, text: str, ctx: RefineContext,
                  timeout_s: float) -> dict[str, Any]:
        n_predict = max(32, int(len(text) / 3.2 * 2) + 16)   # the guard rejects >2x anyway
        return server.completion(self.prompt(text, ctx), n_predict=n_predict, timeout=timeout_s, stop=prompts.STOP)


_LEFTOVER = re.compile(r"<\|im_(?:start|end)\|>|<\|endoftext\|>")


def clean_output(content: str) -> str:
    """Strip any think block / special tokens that slipped through (should not happen with the
    pre-seeded empty think block, but a leaked chain of thought must never be typed)."""
    out = _LEFTOVER.sub("", content)
    if "</think>" in out:
        out = out.split("</think>", 1)[1]
    return tidy(out)


def create(cfg: RefineConfig, **kw: Any) -> LocalRefiner:
    return LocalRefiner(cfg, **kw)

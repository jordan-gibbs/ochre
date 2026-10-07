"""Whisper via faster-whisper (CTranslate2): the multilingual local engine (SPEC §4.2).

Optional extra (``pip install openwhisprflow[whisper]``), so faster-whisper is imported only
inside load(). Models are fetched by us (byte progress for the UI, files listed from the HF
API with their LFS SHA-256) into ``models_dir()/whisper/<name>``; a ``.complete`` marker makes
later starts fully offline. The ``prompt`` argument of transcribe() becomes Whisper's
``initial_prompt``, which is how the personal dictionary biases spelling.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from openwhisprflow.config import SttConfig, models_dir
from openwhisprflow.models import ensure_file, hf_url
from openwhisprflow.stt.base import SAMPLE_RATE, EngineInfo, LoadProgress, SttError, SttResult

log = logging.getLogger("openwhisprflow.stt.whisper")

REPOS = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "small.en": "Systran/faster-whisper-small.en",
}
DEFAULT_MODEL = "large-v3-turbo"
_WANTED = re.compile(r"^(config\.json|preprocessor_config\.json|model\.bin|tokenizer\.json|vocabulary\.\w+)$")

INFO = EngineInfo(
    name="whisper", kind="local", label="Whisper (local, faster-whisper)",
    models=list(REPOS), default_model=DEFAULT_MODEL, languages="99 languages (small.en: English)",
)


def is_available() -> bool:
    """Registry hook: hide the engine when the optional extra is not installed."""
    return importlib.util.find_spec("faster_whisper") is not None


def model_dir(model: str) -> Path:
    return models_dir() / "whisper" / model


def is_downloaded(model: str = DEFAULT_MODEL) -> bool:
    return (model_dir(model) / ".complete").is_file()


def _repo_files(repo: str) -> list[dict[str, Any]]:
    r = httpx.get(f"https://huggingface.co/api/models/{repo}/tree/main", timeout=20, follow_redirects=True)
    r.raise_for_status()
    return [f for f in r.json() if f.get("type") == "file" and _WANTED.match(f["path"])]


def download(model: str, progress: Callable[[LoadProgress], None] | None = None,
             cancel: Callable[[], bool] | None = None) -> Path:
    folder = model_dir(model)
    if is_downloaded(model):
        return folder
    repo = REPOS.get(model, model)  # also accept a raw "<org>/<repo>" CT2 model id
    try:
        files = _repo_files(repo)
        total = sum(int(f.get("size") or 0) for f in files)
        base = 0
        for f in files:
            def report(item: str, done: int, _t: int, b: int = base) -> None:
                if progress:
                    progress(LoadProgress(file=item, done=b + done, total=total))
            lfs = f.get("lfs") or {}
            ensure_file(hf_url(repo, f["path"]), folder / f["path"], size=int(f.get("size") or 0) or None,
                        sha256=lfs.get("oid") if lfs else None, progress=report, cancel=cancel)
            base += int(f.get("size") or 0)
    except Exception as e:
        raise SttError(f"Whisper model download failed: {e}", code="model_download", retryable=True) from e
    (folder / ".complete").write_text(json.dumps({"repo": repo, "files": [f["path"] for f in files]}))
    return folder


def _cuda_devices() -> int:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


class WhisperEngine:
    name = "whisper"
    kind = "local"
    sample_rate = SAMPLE_RATE

    def __init__(self, cfg: SttConfig) -> None:
        self.cfg = cfg
        self.model_name = cfg.model or DEFAULT_MODEL
        self.device = "cpu"
        self.compute_type = "int8"
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self, progress: Callable[[LoadProgress], None] | None = None,
             cancel: Callable[[], bool] | None = None) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not is_available():
                raise SttError("Whisper needs the optional extra: pip install 'openwhisprflow[whisper]'",
                               code="engine_missing")
            folder = download(self.model_name, progress, cancel)
            want_gpu = self.cfg.device in ("auto", "cuda") and _cuda_devices() > 0
            attempts = [("cuda", "float16"), ("cpu", "int8")] if want_gpu else [("cpu", "int8")]
            last: Exception | None = None
            for device, compute in attempts:
                try:
                    self._model = self._open(folder, device, compute)
                    self.device, self.compute_type = device, compute
                    return
                except Exception as e:  # missing cuBLAS/cuDNN shows up here, not at import
                    last = e
                    log.warning("whisper failed on %s/%s: %s", device, compute, e)
            raise SttError(f"Whisper failed to load: {last}", code="model_load") from last

    def _open(self, folder: Path, device: str, compute: str) -> Any:
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        threads = max(1, min(8, (os.cpu_count() or 4) // 2))
        model = WhisperModel(str(folder), device=device, compute_type=compute, cpu_threads=threads)
        # Run a tiny decode now so a broken CUDA runtime fails here (and falls back), not mid-dictation.
        segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en",
                                       beam_size=1, without_timestamps=True)
        list(segments)
        log.info("whisper %s loaded on %s/%s in %.1fs", self.model_name, device, compute,
                 time.perf_counter() - t0)
        return model

    def transcribe(self, audio: np.ndarray, *, language: str | None = None,
                   prompt: str | None = None) -> SttResult:
        if self._model is None:
            self.load()
        t0 = time.perf_counter()
        pcm = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        duration_ms = round(len(pcm) * 1000 / SAMPLE_RATE)
        from openwhisprflow.stt.parakeet import is_speech_like

        if not is_speech_like(pcm):
            return SttResult(text="", duration_ms=duration_ms, processing_ms=0, engine=self.name)
        lang = language or self.cfg.language or ("en" if self.model_name.endswith(".en") else None)
        segments, info = self._model.transcribe(
            pcm, language=lang, initial_prompt=prompt or None,
            beam_size=5 if self.device == "cuda" else 1,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return SttResult(text=text, duration_ms=duration_ms, language=info.language or lang,
                         processing_ms=round((time.perf_counter() - t0) * 1000), engine=self.name)

    def close(self) -> None:
        self._model = None


def create(cfg: SttConfig) -> WhisperEngine:
    return WhisperEngine(cfg)

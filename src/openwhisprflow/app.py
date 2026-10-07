"""The orchestrator: one dictation session at a time, every stage wired together (SPEC §3).

Flow: trigger (hotkey gesture / wake word / UI) -> capture -> Segmenter (phrases decode while the
user talks) -> corrections -> optional refinement (guarded, timeout -> raw) -> history (before
insert) -> injection. Every state change goes out on the Bus for the HUD.

Threads: capture callbacks feed the segmenter; hotkey callbacks and UI commands only flip state and
hand the slow tail (finish/refine/insert) to one session worker, so no input hook ever blocks.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from openwhisprflow import __version__, secrets
from openwhisprflow import config as config_mod
from openwhisprflow.config import Config
from openwhisprflow.events import Bus, State
from openwhisprflow.history import History, as_dicts
from openwhisprflow.platform.base import Gesture, InjectError
from openwhisprflow.refine.base import RefineContext, RefineError
from openwhisprflow.stt.base import LoadProgress, SttEngine, SttError

log = logging.getLogger("openwhisprflow.app")

CLOUD_STT = {"soniox", "deepgram", "openai", "groq", "elevenlabs", "assemblyai"}


class App:
    def __init__(self, cfg: Config, bus: Bus | None = None, *, history: History | None = None) -> None:
        self.cfg = cfg
        self.bus = bus or Bus()
        self.history = history or History(enabled=cfg.history.enabled)
        self.stt: SttEngine | None = None
        self.fallback_stt: SttEngine | None = None
        self.refiner = None
        self.injector = None
        self.hotkeys = None
        self.handsfree = None
        self.capture = None
        self.state = State.LOADING
        self._lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owf-session")
        self._session: _Session | None = None
        self._stop = threading.Event()
        self.bus.on_command(self.handle_command)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.bus.emit("hello", version=__version__, platform=sys.platform)
        self._emit_config()
        self._set_state(State.LOADING, detail="Loading models…")
        threading.Thread(target=self._load, name="owf-load", daemon=True).start()

    def _load(self) -> None:
        try:
            self._load_stt()
            self._load_refiner()
            self._load_platform()
            self._set_state(State.IDLE)
        except Exception as exc:  # surfaced in the HUD and settings; the app stays up for reconfiguration
            log.exception("startup failed")
            self._error(f"Setup failed: {exc}", "setup")

    def _progress(self, p: LoadProgress) -> None:
        self.bus.emit("download", item=p.file, done=p.done, total=p.total)

    def _load_stt(self) -> None:
        from openwhisprflow.stt import registry
        engine = registry.create(self.cfg.stt)
        engine.load(self._progress)
        self.stt = engine
        self.fallback_stt = None
        if engine.kind == "cloud" and self.cfg.stt.fallback_local:
            try:  # only when the local model is already on disk; never a surprise 670 MB download
                from openwhisprflow.stt import parakeet
                if parakeet.is_downloaded():
                    local = parakeet.create(self.cfg.stt)
                    local.load(None)
                    self.fallback_stt = local
            except Exception:
                log.warning("local fallback engine unavailable", exc_info=True)

    def _load_refiner(self) -> None:
        if self.refiner:
            self.refiner.close()
            self.refiner = None
        if self.cfg.refine.provider == "off":
            return
        from openwhisprflow.refine import registry
        refiner = registry.create(self.cfg.refine)
        self._set_state(State.LOADING, detail="Loading refinement model…")
        refiner.load()
        self.refiner = refiner

    def _load_platform(self) -> None:
        from openwhisprflow.platform import hotkeys, inject
        if self.injector is None:
            self.injector = inject.create()
        if self.hotkeys is None:
            self.hotkeys = hotkeys.create(self.cfg.hotkey)
            self.hotkeys.start(self.on_gesture)
        else:
            self.hotkeys.set_key(self.cfg.hotkey.key)

    def shutdown(self) -> None:
        self._stop.set()
        self.cancel()
        for part in (self.hotkeys, self.handsfree):
            try:
                if part:
                    part.stop()
            except Exception:
                log.exception("stop failed")
        for engine in (self.stt, self.fallback_stt, self.refiner):
            try:
                if engine:
                    engine.close()
            except Exception:
                log.exception("close failed")
        self._worker.shutdown(wait=False, cancel_futures=True)
        self.history.close()

    # ------------------------------------------------------------------ triggers
    def on_gesture(self, gesture: Gesture) -> None:
        """Called on the hotkey thread: must return immediately."""
        if gesture == Gesture.PRESS:
            self.begin("hotkey")
        elif gesture == Gesture.LOCK:
            with self._lock:
                if self._session:
                    self._set_state(State.LOCKED)
        elif gesture in (Gesture.RELEASE, Gesture.FINISH):
            self.end()
        elif gesture == Gesture.RAW:
            with self._lock:
                if self._session:
                    self._session.raw = True
            self.end()
        elif gesture == Gesture.CANCEL:
            self.cancel()

    def begin(self, trigger: str, *, preroll: np.ndarray | None = None) -> bool:
        from openwhisprflow.audio.segmenter import Segmenter
        with self._lock:
            if self._session or self.state not in (State.IDLE, State.ERROR) or self.stt is None:
                return False
            focus = self._focus()
            session = _Session(trigger=trigger, app=focus.app_name, window=focus.window_title)
            stt = self.stt
            partials = self.cfg.ui.show_partials and stt.kind == "local"
            try:
                rate = self._open_capture(session)
            except Exception as exc:
                self._error(f"Could not open the microphone: {exc}", "mic")
                return False
            session.segmenter = Segmenter(
                lambda audio: self._transcribe(audio), rate,
                engine_rate=stt.sample_rate,
                on_partial=(lambda text, stable: self.bus.emit("partial", text=text, stable_chars=stable))
                if partials else None,
                partial_interval_s=0.8 if partials else None,
                on_segment=session.on_segment,
            )
            if preroll is not None and len(preroll):
                session.segmenter.feed(preroll)
            self._session = session
            if self.hotkeys:
                self.hotkeys.set_recording(True)
            self._set_state(State.HANDSFREE if trigger == "wake" else State.RECORDING, trigger=trigger)
            self._earcon("start")
            return True

    def end(self, *, send: bool = False) -> None:
        with self._lock:
            session = self._session
            if not session or session.ending:
                return
            session.ending = True
            session.send = send
            self._close_capture()
            if self.hotkeys:
                self.hotkeys.set_recording(False)
            self._set_state(State.TRANSCRIBING)
        self._earcon("stop")
        self._worker.submit(self._complete, session)

    def cancel(self) -> None:
        with self._lock:
            session = self._session
            if not session:
                return
            self._session = None
            self._close_capture()
            if session.segmenter:
                session.segmenter.cancel()
            if self.hotkeys:
                self.hotkeys.set_recording(False)
            if not self._stop.is_set():
                self._set_state(State.IDLE)
        self._earcon("cancel")

    # ------------------------------------------------------------------ audio
    def _open_capture(self, session: _Session) -> int:
        from openwhisprflow.audio.capture import Capture

        def on_block(block: np.ndarray) -> None:
            seg = session.segmenter
            if seg is not None and not session.ending:
                seg.feed(block)
                if self.handsfree and session.trigger == "wake":
                    self.handsfree.observe_session_audio(block)

        def on_level(rms: float) -> None:
            self.bus.emit("level", rms=round(float(rms), 3))

        def on_limit() -> None:
            self.end()

        def on_error(message: str) -> None:
            self.cancel()
            self._error(message, "mic")

        self.capture = Capture(self.cfg.audio.device, on_block=on_block, on_level=on_level,
                               max_seconds=self.cfg.audio.max_session_s, on_limit=on_limit,
                               on_error=on_error)
        self.capture.start()
        return self.capture.rate

    def _close_capture(self) -> None:
        if self.capture:
            try:
                self.capture.stop()
            finally:
                self.capture = None

    def _transcribe(self, audio: np.ndarray):
        language = self.cfg.stt.language
        prompt = ", ".join(self.cfg.dictionary.words) or None
        assert self.stt is not None
        try:
            return self.stt.transcribe(audio, language=language, prompt=prompt)
        except SttError as exc:
            if self.fallback_stt is None:
                raise
            log.warning("cloud STT failed (%s); using local fallback", exc.code)
            self.bus.emit("notice", message=f"{self.stt.name} failed, used local transcription")
            return self.fallback_stt.transcribe(audio, language=language, prompt=prompt)

    # ------------------------------------------------------------------ the slow tail
    def _complete(self, session: _Session) -> None:
        t0 = time.perf_counter()
        try:
            results = session.segmenter.finish(timeout=120) if session.segmenter else []
            stt_ms = round((time.perf_counter() - t0) * 1000)
            from openwhisprflow.audio.segmenter import Segmenter
            joined = Segmenter.join(results)
            raw = session.postprocess(joined.text, self.cfg.dictionary)
            with self._lock:
                if self._session is not session:  # canceled while decoding
                    return
            if session.discard or not raw.strip():
                self._finish_session(session)
                if not session.discard:
                    self.bus.emit("notice", message="Didn't catch that")
                return
            text, refined, refine_ms = self._refine(raw, session)
            entry_id = self.history.add(raw=raw, text=text, app=session.app,
                                        stt=self.stt.name if self.stt else "",
                                        refiner=self.refiner.name if refined and self.refiner else "",
                                        duration_ms=joined.duration_ms)
            self._set_state(State.INSERTING)
            inserted = self._insert(text, session)
            if inserted:
                self.history.mark_inserted(entry_id)
            self.bus.emit("result", id=session.id, raw=raw, text=text, inserted=inserted, refined=refined,
                          timings={"stt_ms": stt_ms, "refine_ms": refine_ms,
                                   "total_ms": round((time.perf_counter() - t0) * 1000)})
            self._finish_session(session)
        except Exception as exc:
            log.exception("session failed")
            self._finish_session(session, error=f"Transcription failed: {exc}")

    def _finish_session(self, session: _Session, error: str | None = None) -> None:
        with self._lock:
            if self._session is session:
                self._session = None
        if error:
            self._error(error, "session")
        else:
            self._set_state(State.IDLE)

    def _refine(self, raw: str, session: _Session) -> tuple[str, bool, int]:
        if not self.refiner or session.raw:
            return raw, False, 0
        from openwhisprflow.refine.guard import apply_guard
        style = self.cfg.refine.app_styles.get(session.app, "")
        ctx = RefineContext(mode=self.cfg.refine.mode, app_name=session.app,  # type: ignore[arg-type]
                            window_title=session.window, style=style,
                            dictionary=list(self.cfg.dictionary.words), language=self.cfg.stt.language)
        timeout_ms = (self.cfg.refine.timeout_ms_local if self.refiner.kind == "local"
                      else self.cfg.refine.timeout_ms_cloud)
        self._set_state(State.REFINING)
        t0 = time.perf_counter()
        try:
            out = self.refiner.refine(raw, ctx, timeout_s=timeout_ms / 1000)
        except RefineError as exc:
            log.warning("refinement skipped: %s", exc)
            return raw, False, round((time.perf_counter() - t0) * 1000)
        guarded = apply_guard(raw, out)
        return guarded, guarded != raw, round((time.perf_counter() - t0) * 1000)

    def _insert(self, text: str, session: _Session) -> bool:
        if not self.injector:
            return False
        from openwhisprflow.text.spacing import join_text
        try:
            focus = self.injector.focus()
            payload = join_text(text, focus, trailing_space=self.cfg.inject.trailing_space,
                                join_window_s=self.cfg.inject.join_window_s)
            if self.cfg.inject.method == "paste" or len(payload) > self.cfg.inject.paste_over_chars:
                self.injector.paste_text(payload)
            else:
                self.injector.type_text(payload)
            if session.send:
                self.injector.press_enter()
            return True
        except InjectError as exc:
            self.bus.emit("notice", message=f"{exc} Saved to history.")
            return False

    # ------------------------------------------------------------------ helpers
    def _focus(self):
        from openwhisprflow.platform.base import FocusInfo
        try:
            return self.injector.focus() if self.injector else FocusInfo()
        except Exception:
            return FocusInfo()

    def _earcon(self, name: str) -> None:
        if not self.cfg.audio.earcons:
            return
        try:
            from openwhisprflow.audio import earcons
            earcons.play(name)
        except Exception:
            log.debug("earcon %s failed", name, exc_info=True)

    def _set_state(self, state: State, *, trigger: str = "", detail: str = "") -> None:
        self.state = state
        self.bus.emit("state", state=state.value, trigger=trigger, detail=detail,
                      handsfree_armed=bool(self.handsfree and self.cfg.handsfree.enabled))

    def _error(self, message: str, code: str) -> None:
        self.bus.emit("error", message=message, code=code)
        self._set_state(State.ERROR, detail=message)

    def _emit_config(self) -> None:
        from openwhisprflow.secrets import ENV_FALLBACKS
        self.bus.emit("config", config=self.cfg.to_dict(),
                      secrets={p: secrets.has(p) for p in ENV_FALLBACKS})

    # ------------------------------------------------------------------ UI commands
    def handle_command(self, msg: dict[str, Any]) -> None:
        op = msg["op"]
        if op == "start":
            self.begin("ui")
        elif op == "stop":
            self.end()
        elif op == "toggle":
            if self._session:
                self.end()
            else:
                self.begin("ui")
        elif op == "cancel":
            self.cancel()
        elif op == "get_config":
            self._emit_config()
            self._emit_engines()
        elif op == "set_config":
            self.apply_config(self.cfg.patch(msg.get("patch") or {}))
        elif op == "set_secret":
            secrets.store(str(msg["provider"]), msg.get("key") or None)
            self._emit_config()
        elif op == "test_provider":
            threading.Thread(target=self._test_provider, args=(msg.get("stage"), msg.get("provider")),
                             daemon=True).start()
        elif op == "history_query":
            items = self.history.query(str(msg.get("q") or ""), int(msg.get("limit") or 50))
            self.bus.emit("history", items=as_dicts(items))
        elif op == "insert_text":
            self._worker.submit(self._insert, str(msg.get("text") or ""), _Session(trigger="ui"))
        elif op == "set_handsfree":
            self.apply_config(self.cfg.patch({"handsfree": {"enabled": bool(msg.get("enabled"))}}))
        elif op == "quit":
            self._stop.set()

    def apply_config(self, new: Config) -> None:
        old, self.cfg = self.cfg, new
        config_mod.save(new)
        self._emit_config()
        if (old.stt.engine, old.stt.model, old.stt.device) != (new.stt.engine, new.stt.model, new.stt.device) \
                or (old.refine.provider, old.refine.model, old.refine.base_url) != \
                (new.refine.provider, new.refine.model, new.refine.base_url) \
                or old.hotkey.key != new.hotkey.key:
            self._set_state(State.LOADING, detail="Applying settings…")
            threading.Thread(target=self._reload, args=(old, new), daemon=True).start()

    def _reload(self, old: Config, new: Config) -> None:
        try:
            if (old.stt.engine, old.stt.model, old.stt.device) != (new.stt.engine, new.stt.model, new.stt.device):
                previous = self.stt
                self._load_stt()
                if previous:
                    previous.close()
            if (old.refine.provider, old.refine.model, old.refine.base_url) != \
                    (new.refine.provider, new.refine.model, new.refine.base_url):
                self._load_refiner()
            if self.hotkeys and old.hotkey.key != new.hotkey.key:
                self.hotkeys.set_key(new.hotkey.key)
            self._set_state(State.IDLE)
        except Exception as exc:
            log.exception("reload failed")
            self._error(f"Could not apply settings: {exc}", "config")

    def _emit_engines(self) -> None:
        try:
            from openwhisprflow.refine import registry as refine_registry
            from openwhisprflow.stt import registry as stt_registry
            self.bus.emit("engines", stt=[vars(i) for i in stt_registry.available()],
                          refine=[vars(i) for i in refine_registry.available()])
        except Exception:
            log.exception("engine listing failed")

    def _test_provider(self, stage: str, provider: str) -> None:
        t0 = time.perf_counter()
        try:
            if stage == "stt":
                from openwhisprflow.stt import registry
                engine = registry.create(self.cfg.patch({"stt": {"engine": provider}}).stt)
                engine.load(self._progress)
                tone = (0.05 * np.sin(np.linspace(0, 2 * np.pi * 220, 16000))).astype(np.float32)
                engine.transcribe(tone)
                engine.close()
            else:
                from openwhisprflow.refine import registry
                refiner = registry.create(self.cfg.patch({"refine": {"provider": provider}}).refine)
                refiner.load()
                refiner.refine("um so this is uh a test", RefineContext(), timeout_s=10)
                refiner.close()
            self.bus.emit("test_result", stage=stage, provider=provider, ok=True, message="Works",
                          ms=round((time.perf_counter() - t0) * 1000))
        except Exception as exc:
            self.bus.emit("test_result", stage=stage, provider=provider, ok=False, message=str(exc),
                          ms=round((time.perf_counter() - t0) * 1000))

    def wait(self) -> None:
        while not self._stop.wait(0.5):
            pass


class _Session:
    def __init__(self, *, trigger: str, app: str = "", window: str = "") -> None:
        self.id = uuid.uuid4().hex[:12]
        self.trigger = trigger
        self.app = app
        self.window = window
        self.segmenter = None
        self.ending = False
        self.send = False
        self.raw = False
        self.discard = False
        self.started = time.monotonic()

    def on_segment(self, result) -> None:
        """Per-phrase hook; hands-free control phrases are checked here once wired."""

    def postprocess(self, text: str, dictionary) -> str:
        from openwhisprflow.text.corrections import apply as apply_corrections
        return apply_corrections(text, dictionary)

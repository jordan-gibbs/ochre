"""Command line: `openwhisprflow` (alias `owf`).

  owf                      run the app (tray + HUD; --headless for terminal only)
  owf toggle|start|stop|cancel   control the running app (bind `owf toggle` to a Wayland shortcut)
  owf transcribe FILE...   transcribe audio files with the configured engine (prints text)
  owf devices              list microphones
  owf config [--path]      print the effective config (or its path)
  owf doctor               check permissions, models, keys and the UI install
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys

from openwhisprflow import __version__
from openwhisprflow import config as config_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="owf", description="Open Whisperflow voice dictation")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="run the app (default)")
    run.add_argument("--headless", action="store_true", help="no Electron UI; status in the terminal")
    for name in ("toggle", "start", "stop", "cancel"):
        sub.add_parser(name, help=f"{name} dictation in the running app")
    tr = sub.add_parser("transcribe", help="transcribe audio files")
    tr.add_argument("files", nargs="+")
    tr.add_argument("--engine", help="override stt.engine")
    tr.add_argument("--refine", action="store_true", help="also run the configured refiner")
    sub.add_parser("devices", help="list input devices")
    cfg_p = sub.add_parser("config", help="print config")
    cfg_p.add_argument("--path", action="store_true")
    sub.add_parser("doctor", help="diagnose setup")
    args = parser.parse_args(argv)

    cfg = config_mod.load()
    logging.basicConfig(level=getattr(logging, cfg.debug.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cmd = args.cmd or "run"
    if cmd == "run":
        return _run(cfg, headless=getattr(args, "headless", False))
    if cmd in ("toggle", "start", "stop", "cancel"):
        from openwhisprflow.ui_bridge.server import send_command
        if not send_command(cmd):
            print("Open Whisperflow is not running.", file=sys.stderr)
            return 1
        return 0
    if cmd == "transcribe":
        return _transcribe(cfg, args.files, args.engine, args.refine)
    if cmd == "devices":
        from openwhisprflow.audio.capture import input_devices
        for dev in input_devices():
            print(dev)
        return 0
    if cmd == "config":
        print(config_mod.config_path() if args.path else json.dumps(cfg.to_dict(), indent=2))
        return 0
    if cmd == "doctor":
        return _doctor(cfg)
    parser.print_help()
    return 2


def _run(cfg: config_mod.Config, *, headless: bool) -> int:
    from openwhisprflow.app import App
    from openwhisprflow.events import Bus

    bus = Bus()
    app = App(cfg, bus)
    server = supervisor = None
    try:
        from openwhisprflow.ui_bridge.server import UiServer
        server = UiServer(bus, port=cfg.ui.ws_port)
        server.start()
    except Exception as exc:  # the CLI controls need the server; the app itself does not
        logging.getLogger("openwhisprflow").warning("UI bridge unavailable: %s", exc)
        server = None
    if headless or not cfg.ui.enabled or server is None:
        bus.subscribe(_print_status)
    else:
        try:
            from openwhisprflow.ui_bridge.supervisor import Supervisor
            supervisor = Supervisor(server.port, server.token, cfg.ui.theme)
            supervisor.start()
        except Exception as exc:
            logging.getLogger("openwhisprflow").warning("UI unavailable (%s); running headless", exc)
            bus.subscribe(_print_status)

    signal.signal(signal.SIGINT, lambda *_: bus.command({"op": "quit"}))
    app.start()
    try:
        app.wait()
    finally:
        app.shutdown()
        if supervisor:
            supervisor.stop()
        if server:
            server.stop()
    return 0


def _print_status(msg: dict) -> None:
    event = msg["event"]
    if event == "state":
        detail = f" — {msg['detail']}" if msg.get("detail") else ""
        print(f"[{msg['state']}]{detail}", flush=True)
    elif event == "download" and msg.get("total"):
        print(f"\r  {msg['item']}: {msg['done'] / 1e6:.0f}/{msg['total'] / 1e6:.0f} MB", end="", flush=True)
    elif event == "result":
        t = msg["timings"]
        print(f"  {msg['text']!r}  (stt {t['stt_ms']} ms, refine {t['refine_ms']} ms)", flush=True)
    elif event in ("error", "notice"):
        print(f"  {event}: {msg['message']}", flush=True)


def _transcribe(cfg: config_mod.Config, files: list[str], engine: str | None, refine: bool) -> int:
    import soundfile as sf

    from openwhisprflow.audio.resample import resample
    from openwhisprflow.audio.segmenter import split_at_pauses
    from openwhisprflow.stt import registry
    from openwhisprflow.text.corrections import apply as apply_corrections

    if engine:
        cfg = cfg.patch({"stt": {"engine": engine}})
    stt = registry.create(cfg.stt)
    stt.load(lambda p: print(f"\r{p.file}: {p.done / 1e6:.0f}/{p.total / 1e6:.0f} MB", end="",
                             file=sys.stderr, flush=True))
    refiner = None
    if refine and cfg.refine.provider != "off":
        from openwhisprflow.refine import registry as refine_registry
        refiner = refine_registry.create(cfg.refine)
        refiner.load()
    try:
        for path in files:
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if rate != stt.sample_rate:
                audio = resample(audio, rate, stt.sample_rate)
            parts = [stt.transcribe(chunk, language=cfg.stt.language).text
                     for chunk in split_at_pauses(audio, stt.sample_rate)]
            text = apply_corrections(" ".join(p.strip() for p in parts if p.strip()), cfg.dictionary)
            if refiner:
                from openwhisprflow.refine.base import RefineContext
                from openwhisprflow.refine.guard import apply_guard
                text = apply_guard(text, refiner.refine(text, RefineContext(mode=cfg.refine.mode), timeout_s=30))
            print(f"{path}\t{text}" if len(files) > 1 else text)
    finally:
        stt.close()
        if refiner:
            refiner.close()
    return 0


def _doctor(cfg: config_mod.Config) -> int:
    from openwhisprflow import secrets
    print(f"Open Whisperflow {__version__} on {sys.platform}, Python {sys.version.split()[0]}")
    print(f"config: {config_mod.config_path()}")
    print(f"models: {config_mod.models_dir()}")
    print(f"stt: {cfg.stt.engine}   refine: {cfg.refine.provider}   hands-free: {cfg.handsfree.enabled}")
    keys = [p for p in secrets.ENV_FALLBACKS if secrets.has(p)]
    print(f"API keys present: {', '.join(keys) or 'none'}")
    try:
        from openwhisprflow.platform import permissions
        missing = permissions.check()
        for name, fix in missing.items():
            print(f"permission missing: {name}: {fix}")
        if not missing:
            print("permissions: ok")
    except Exception as exc:
        print(f"permissions: could not check ({exc})")
    try:
        from openwhisprflow.ui_bridge.supervisor import find_electron
        print(f"ui: {find_electron() or 'not installed (run npm ci in ui/)'}")
    except Exception as exc:
        print(f"ui: unknown ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

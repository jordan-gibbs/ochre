"""WebSocket bridge between the core's event bus and the Electron shell (SPEC §3, §7.1).

The server runs its own asyncio loop on a daemon thread, so the core can stay synchronous and threaded.

- It binds 127.0.0.1 only and requires a per-launch token in the upgrade request (``/ui?token=...``), so
  other local processes and web pages can't drive dictation. The token goes to Electron through the
  ``OWF_UI_TOKEN`` env var (supervisor.py), and to the CLI through ``runtime.json`` (current user only).
- Every bus event fans out to every client. ``level`` is coalesced per client (only the newest one is
  kept), so a slow client never builds a backlog of mic levels. A client that falls far behind on real
  events is disconnected; it reconnects and gets the replay.
- On connect a client gets ``hello``, then the last ``config``, ``engines`` and ``state``. If no config
  has been seen yet, the server asks the core for one (``get_config``).
- Client ``{"op": ...}`` messages go to ``bus.command`` on one worker thread. Commands keep their order
  and never block the socket loop.

Keys arrive here in ``set_secret`` and are passed straight through. Messages are never logged.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import hmac
import json
import logging
import os
import secrets as _secrets
import sys
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from openwhisprflow import config as _config
from openwhisprflow.events import Bus

log = logging.getLogger("openwhisprflow.ui_bridge")

HOST = "127.0.0.1"
MAX_BACKLOG = 512                 # queued non-level events before a stuck client is dropped
MAX_MESSAGE = 1 << 20             # 1 MiB per client frame (history re-inserts are the largest)
REPLAY = ("config", "engines")    # cached and replayed to new clients (plus hello and the bus' last state)


def runtime_path() -> Path:
    return _config.data_dir() / "runtime.json"


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("openwhisprflow")
    except Exception:
        return "0.0.0"


class _Client:
    """One connected UI: an ordered queue of events plus a single 'latest level' slot."""

    def __init__(self, ws: ServerConnection) -> None:
        self.ws = ws
        self.queue: collections.deque[str] = collections.deque()
        self.level: str | None = None
        self.wake = asyncio.Event()

    def push(self, event: str, frame: str) -> bool:
        if event == "level":
            self.level = frame            # coalesce: only the newest level matters
        else:
            if len(self.queue) >= MAX_BACKLOG:
                return False
            self.queue.append(frame)
        self.wake.set()
        return True


class UiServer:
    """Fan bus events out to UI clients; forward their commands to the bus."""

    def __init__(self, bus: Bus, port: int = 8765, token: str | None = None, *, host: str = HOST,
                 runtime_file: Path | None | bool = None) -> None:
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("the UI bridge only binds to localhost")
        self.bus = bus
        self.host = host
        self.port = port
        self.token = token or _secrets.token_urlsafe(32)
        # None = the default runtime.json; False = don't write one (tests, embedded use)
        self._runtime_file = runtime_path() if runtime_file is None else (runtime_file or None)
        self._cache: dict[str, dict[str, Any]] = {}
        self._clients: set[_Client] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Server | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._unsubscribe = None
        self._commands = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="owf-ui-cmd")

    # ------------------------------------------------------------------ lifecycle

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/ui"

    def start(self, timeout: float = 5.0) -> None:
        """Bind and serve on a background thread. Raises OSError if the port can't be bound."""
        if self._thread:
            return
        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            try:
                loop.run_until_complete(self._main(ready))
            except BaseException as e:  # surfaced to start() below
                failure.append(e)
                ready.set()
            finally:
                loop.close()

        self._thread = threading.Thread(target=run, name="owf-ui-server", daemon=True)
        self._thread.start()
        if not ready.wait(timeout):
            raise TimeoutError("UI bridge did not start")
        if failure:
            self._thread = None
            raise failure[0]
        self._unsubscribe = self.bus.subscribe(self._on_event)
        self._write_runtime()
        log.info("UI bridge listening on %s", self.url)

    def stop(self, timeout: float = 3.0) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        if self._loop and self._stop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass
        if self._thread:
            self._thread.join(timeout)
            self._thread = None
        self._commands.shutdown(wait=False, cancel_futures=True)
        self._remove_runtime()

    async def _main(self, ready: threading.Event) -> None:
        self._stop = asyncio.Event()
        async with serve(self._handle, self.host, self.port, process_request=self._check_token,
                         max_size=MAX_MESSAGE, ping_interval=20, ping_timeout=20) as server:
            self._server = server
            self.port = next(iter(server.sockets)).getsockname()[1]
            ready.set()
            await self._stop.wait()
            for c in list(self._clients):
                await c.ws.close(1001, "core shutting down")

    # ------------------------------------------------------------------ runtime.json (CLI discovery)

    def _write_runtime(self) -> None:
        path = self._runtime_file
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps({"port": self.port, "token": self.token, "pid": os.getpid()})
            tmp = path.with_name(path.name + ".tmp")
            # created 0600 on POSIX; on Windows the user profile's ACLs already keep other users out
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            if sys.platform != "win32":
                os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except OSError:
            log.warning("could not write %s; `openwhisprflow toggle` won't find the running app", path)

    def _remove_runtime(self) -> None:
        path = self._runtime_file
        if not path:
            return
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            if info.get("pid") == os.getpid() and info.get("port") == self.port:
                path.unlink()
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------ connections

    def _check_token(self, connection: ServerConnection, request):
        query = parse_qs(urlsplit(request.path).query)
        given = (query.get("token") or [""])[0]
        if not hmac.compare_digest(given.encode(), self.token.encode()):
            log.warning("rejected a UI connection with a missing or wrong token")
            return connection.respond(HTTPStatus.FORBIDDEN, "forbidden\n")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        client = _Client(ws)
        self._replay(client)
        self._clients.add(client)
        writer = asyncio.create_task(self._writer(client))
        try:
            async for raw in ws:
                self._on_client_message(raw)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(client)
            writer.cancel()

    def _replay(self, client: _Client) -> None:
        hello = self._cache.get("hello") or {"event": "hello", "version": _version(), "platform": sys.platform}
        client.push("hello", json.dumps(hello, default=str))
        for name in REPLAY:
            if name in self._cache:
                client.push(name, json.dumps(self._cache[name], default=str))
        if self.bus.last_state:
            client.push("state", json.dumps(self.bus.last_state, default=str))
        if "config" not in self._cache:
            self._command({"op": "get_config"})

    async def _writer(self, client: _Client) -> None:
        try:
            while True:
                await client.wake.wait()
                client.wake.clear()
                while client.queue:
                    await client.ws.send(client.queue.popleft())
                if client.level is not None:
                    frame, client.level = client.level, None
                    await client.ws.send(frame)
        except (ConnectionClosed, asyncio.CancelledError):
            pass

    def _on_client_message(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            return
        try:
            msg = json.loads(raw)
        except ValueError:
            log.debug("ignored a non-JSON frame from a UI client")
            return
        if not isinstance(msg, dict) or not isinstance(msg.get("op"), str):
            return
        self._command(msg)

    def _command(self, msg: dict[str, Any]) -> None:
        op = msg.get("op")

        def run() -> None:
            try:
                self.bus.command(msg)
            except Exception:
                log.exception("UI command %r failed", op)   # never the message: it may carry a key

        try:
            self._commands.submit(run)
        except RuntimeError:   # shutting down
            pass

    # ------------------------------------------------------------------ bus -> clients

    def _on_event(self, msg: dict[str, Any]) -> None:
        """Bus listener: any thread. Serialize once, hand off to the socket loop."""
        event = msg.get("event")
        if event in REPLAY or event == "hello":
            self._cache[event] = msg
        loop = self._loop
        if not loop or loop.is_closed():
            return
        try:
            frame = json.dumps(msg, default=str)
        except (TypeError, ValueError):
            log.warning("dropped an unserializable %s event", event)
            return
        try:
            loop.call_soon_threadsafe(self._fanout, event, frame)
        except RuntimeError:   # loop closed between the check and the call
            pass

    def _fanout(self, event: str, frame: str) -> None:
        for c in list(self._clients):
            if not c.push(event, frame):
                log.warning("a UI client fell %d events behind; disconnecting it", MAX_BACKLOG)
                self._clients.discard(c)
                asyncio.ensure_future(c.ws.close(1013, "too slow"))

    @property
    def client_count(self) -> int:
        return len(self._clients)


def send_command(op: str, *, timeout: float = 2.0, runtime_file: Path | None = None, **fields: Any) -> bool:
    """Send one ``{"op": op, **fields}`` to the running app (``openwhisprflow toggle|start|stop|cancel``).

    Reads the port and token from runtime.json. Returns False when no app is running or it can't be
    reached; never raises for those.
    """
    from websockets.sync.client import connect

    path = runtime_file or runtime_path()
    try:
        info = json.loads(Path(path).read_text(encoding="utf-8"))
        port, token = int(info["port"]), str(info["token"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    url = f"ws://{HOST}:{port}/cli?token={token}"
    try:
        with connect(url, open_timeout=timeout, close_timeout=timeout) as ws:
            ws.send(json.dumps({"op": op, **fields}))
        return True
    except Exception as e:  # refused, stale file, wrong token
        log.debug("send_command(%s) failed: %s", op, type(e).__name__)
        return False

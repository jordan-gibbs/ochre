"""ui_bridge.server: token auth, fan-out, replay, level coalescing, command forwarding, CLI helper."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from openwhisprflow.events import Bus
from openwhisprflow.ui_bridge import server as srv
from openwhisprflow.ui_bridge.server import UiServer, send_command


@pytest.fixture
def bus():
    b = Bus()
    b.commands = []
    b.got = threading.Event()

    def handler(msg):
        b.commands.append(msg)
        b.got.set()
    b.on_command(handler)
    return b


@pytest.fixture
def server(bus, tmp_path):
    s = UiServer(bus, port=0, token="t0k3n", runtime_file=tmp_path / "runtime.json")
    s.start()
    yield s
    s.stop()


def url(s, token="t0k3n"):
    return f"ws://127.0.0.1:{s.port}/ui?token={token}"


def recv_json(ws, timeout=2.0):
    return json.loads(ws.recv(timeout=timeout))


def recv_until(ws, event, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = recv_json(ws, timeout=deadline - time.monotonic())
        if msg["event"] == event:
            return msg
    raise AssertionError(f"no {event} event")


def wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_rejects_missing_or_wrong_token(server):
    for bad in (f"ws://127.0.0.1:{server.port}/ui", url(server, "nope")):
        with pytest.raises(InvalidStatus) as e:
            connect(bad, open_timeout=2).close()
        assert e.value.response.status_code == 403


def test_binds_localhost_only(bus):
    with pytest.raises(ValueError):
        UiServer(bus, host="0.0.0.0", runtime_file=False)


def test_generates_a_token_when_none_given(bus):
    a, b = UiServer(bus, runtime_file=False), UiServer(bus, runtime_file=False)
    assert len(a.token) >= 32 and a.token != b.token


def test_hello_then_replay_of_config_and_state(bus, server):
    bus.emit("config", config={"ui": {"theme": "dark"}}, secrets={"openai": True})
    bus.emit("state", state="recording", trigger="hotkey", handsfree_armed=False, detail=None)
    with connect(url(server)) as ws:
        hello = recv_json(ws)
        assert hello["event"] == "hello" and "version" in hello and hello["platform"] == sys.platform
        cfg = recv_json(ws)
        assert cfg["event"] == "config" and cfg["secrets"] == {"openai": True}
        st = recv_json(ws)
        assert st == {"event": "state", "state": "recording", "trigger": "hotkey", "handsfree_armed": False,
                      "detail": None}


def test_asks_core_for_config_when_none_cached(bus, server):
    with connect(url(server)) as ws:
        recv_json(ws)  # hello
        assert bus.got.wait(2)
        assert bus.commands[0] == {"op": "get_config"}


def test_fans_out_to_every_client(bus, server):
    bus.emit("config", config={}, secrets={})
    with connect(url(server)) as a, connect(url(server)) as b:
        for ws in (a, b):
            recv_until(ws, "config")
        assert wait_for(lambda: server.client_count == 2)
        bus.emit("partial", text="hello wor", stable_chars=6)
        for ws in (a, b):
            assert recv_until(ws, "partial") == {"event": "partial", "text": "hello wor", "stable_chars": 6}


def test_forwards_commands_in_order_and_ignores_junk(bus, server):
    bus.emit("config", config={}, secrets={})
    with connect(url(server)) as ws:
        recv_until(ws, "config")
        ws.send("not json")
        ws.send(json.dumps(["op", "start"]))
        ws.send(json.dumps({"op": "set_config", "patch": {"ui": {"theme": "dark"}}}))
        ws.send(json.dumps({"op": "start"}))
        ws.send(json.dumps({"op": "cancel"}))
        assert wait_for(lambda: len(bus.commands) == 3)
    assert [c["op"] for c in bus.commands] == ["set_config", "start", "cancel"]
    assert bus.commands[0]["patch"] == {"ui": {"theme": "dark"}}


def test_level_is_coalesced_per_client():
    c = srv._Client(ws=None)
    for i in range(50):
        assert c.push("level", json.dumps({"event": "level", "rms": i / 50}))
    c.push("state", "{}")
    assert list(c.queue) == ["{}"]                 # real events stay queued in order
    assert json.loads(c.level)["rms"] == 49 / 50   # only the newest level is kept


def test_slow_client_backlog_is_bounded():
    c = srv._Client(ws=None)
    for _ in range(srv.MAX_BACKLOG):
        assert c.push("partial", "{}")
    assert c.push("partial", "{}") is False


def test_runtime_file_written_private_and_removed(bus, tmp_path):
    path = tmp_path / "runtime.json"
    s = UiServer(bus, port=0, runtime_file=path)
    s.start()
    try:
        info = json.loads(path.read_text())
        assert info == {"port": s.port, "token": s.token, "pid": os.getpid()}
        if sys.platform != "win32":
            assert (path.stat().st_mode & 0o777) == 0o600
    finally:
        s.stop()
    assert not path.exists()


def test_send_command_reaches_the_bus(bus, tmp_path):
    path = tmp_path / "runtime.json"
    s = UiServer(bus, port=0, runtime_file=path)
    s.start()
    try:
        assert send_command("start", runtime_file=path) is True
        assert send_command("set_handsfree", runtime_file=path, enabled=True) is True
        assert wait_for(lambda: len([c for c in bus.commands if c["op"] != "get_config"]) == 2)
        ops = [c for c in bus.commands if c["op"] != "get_config"]
        assert ops == [{"op": "start"}, {"op": "set_handsfree", "enabled": True}]
    finally:
        s.stop()


def test_send_command_without_a_running_app(tmp_path):
    assert send_command("toggle", runtime_file=tmp_path / "missing.json") is False
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"port": 9, "token": "x", "pid": 1}))
    assert send_command("toggle", runtime_file=stale, timeout=0.5) is False


def test_port_in_use_raises(bus, server):
    other = UiServer(bus, port=server.port, runtime_file=False)
    with pytest.raises(OSError):
        other.start()


def test_listener_detached_after_stop(bus, tmp_path):
    s = UiServer(bus, port=0, runtime_file=False)
    s.start()
    s.stop()
    bus.emit("state", state="idle")  # must not raise or reach a closed loop

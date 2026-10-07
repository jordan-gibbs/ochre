"""ui_bridge.supervisor: launch args/env, crash restart with backoff, clean exit, stop, missing installs."""

from __future__ import annotations

import json
import sys
import textwrap
import time

import pytest

from openwhisprflow.ui_bridge.supervisor import Supervisor, UiUnavailable, find_electron

# A stand-in for Electron: records how it was launched, then behaves as the test asks.
FAKE = textwrap.dedent("""
    import json, os, sys, time
    out = os.environ["FAKE_OUT"]
    runs = len(open(out).read().splitlines()) if os.path.exists(out) else 0
    with open(out, "a") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "token": os.environ.get("OWF_UI_TOKEN"),
                            "as_node": os.environ.get("ELECTRON_RUN_AS_NODE")}) + "\\n")
    mode = os.environ["FAKE_MODE"]
    print("fake shell up", flush=True)
    if mode == "crash" or (mode == "crash-twice" and runs < 2):
        sys.exit(3)
    if mode == "clean":
        sys.exit(0)
    time.sleep(60)
""")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_shell.py"
    script.write_text(FAKE)
    out = tmp_path / "runs.jsonl"
    monkeypatch.setenv("FAKE_OUT", str(out))
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    ui = tmp_path / "ui"
    ui.mkdir()

    def runs():
        return [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    return {"cmd": [sys.executable, str(script)], "ui": ui, "runs": runs}


def wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def make(fake, **kw):
    return Supervisor(8765, "secret-token", "dark", ui_dir=fake["ui"], electron=fake["cmd"],
                      backoff_min=0.05, backoff_max=0.2, **kw)


def test_launch_args_and_token_via_env(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "run")
    sup = make(fake)
    assert sup.start()
    try:
        assert wait_for(lambda: len(fake["runs"]()) == 1)
        run = fake["runs"]()[0]
        assert run["argv"][0] == str(fake["ui"])
        assert "--port=8765" in run["argv"] and "--theme=dark" in run["argv"]
        assert any(a.startswith("--parent-pid=") for a in run["argv"])
        assert not any("secret-token" in a for a in run["argv"])  # never on the command line
        assert run["token"] == "secret-token"
        assert run["as_node"] is None
        assert sup.running
    finally:
        sup.stop()
    assert not sup.running


def test_restarts_after_crash(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash-twice")
    sup = make(fake)
    assert sup.start()
    try:
        assert wait_for(lambda: len(fake["runs"]()) == 3)
        assert wait_for(lambda: sup.running)
        assert sup.restarts == 2
    finally:
        sup.stop()


def test_clean_exit_is_not_restarted(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "clean")
    sup = make(fake)
    assert sup.start()
    assert wait_for(lambda: not sup.running)
    time.sleep(0.4)
    assert len(fake["runs"]()) == 1
    sup.stop()


def test_stop_ends_a_crash_loop(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    sup = make(fake)
    sup.start()
    assert wait_for(lambda: len(fake["runs"]()) >= 2)
    sup.stop()
    n = len(fake["runs"]())
    time.sleep(0.5)
    assert len(fake["runs"]()) == n


def test_missing_npm_install_is_reported(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    with pytest.raises(UiUnavailable, match="npm install"):
        find_electron(tmp_path)
    sup = Supervisor(1, "t", ui_dir=tmp_path)
    assert sup.start() is False
    assert "npm install" in sup.problem


def test_missing_electron_binary_is_reported(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "node_modules" / "electron").mkdir(parents=True)
    with pytest.raises(UiUnavailable, match="install.js"):
        find_electron(tmp_path)


def test_missing_ui_dir_is_reported(tmp_path):
    with pytest.raises(UiUnavailable, match="No UI found"):
        find_electron(tmp_path / "nowhere")


def test_finds_electron_from_path_txt(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    pkg = tmp_path / "node_modules" / "electron"
    (pkg / "dist").mkdir(parents=True)
    (pkg / "path.txt").write_text("my-electron")
    (pkg / "dist" / "my-electron").write_text("")
    assert find_electron(tmp_path) == pkg / "dist" / "my-electron"

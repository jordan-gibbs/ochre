from openwhisprflow import config as config_mod
from openwhisprflow.config import Config
from openwhisprflow.events import Bus
from openwhisprflow.history import History


def test_config_roundtrip_preserves_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config.from_dict({"stt": {"engine": "groq", "bogus": 1}, "future_section": {"x": 1}})
    config_mod.save(cfg, path)
    loaded = config_mod.load(path)
    assert loaded.stt.engine == "groq"
    assert loaded.extra == {"future_section": {"x": 1}}


def test_config_patch_is_deep_and_non_mutating():
    cfg = Config()
    new = cfg.patch({"refine": {"provider": "local"}, "dictionary": {"words": ["Ochre"]}})
    assert new.refine.provider == "local" and new.refine.mode == "clean"
    assert new.dictionary.words == ["Ochre"]
    assert cfg.refine.provider == "off"


def test_config_none_values_are_dropped_for_toml(tmp_path):
    cfg = Config()
    assert cfg.stt.language is None
    config_mod.save(cfg, tmp_path / "c.toml")  # tomli-w can't write None; must not raise


def test_history_add_query_mark(tmp_path):
    h = History(tmp_path / "h.sqlite3")
    a = h.add(raw="um hello", text="Hello.", app="slack")
    h.add(raw="second", text="Second.")
    h.mark_inserted(a)
    items = h.query()
    assert [e.text for e in items] == ["Second.", "Hello."]
    assert items[1].inserted and not items[0].inserted
    assert [e.text for e in h.query("hello")] == ["Hello."]
    h.clear()
    assert h.query() == []


def test_history_disabled_is_noop(tmp_path):
    h = History(tmp_path / "h.sqlite3", enabled=False)
    assert h.add(raw="x", text="x") is None
    assert h.query() == []
    assert not (tmp_path / "h.sqlite3").exists()


def test_bus_fanout_isolation_and_commands():
    bus = Bus()
    seen: list[dict] = []

    def broken(_msg):
        raise RuntimeError("boom")

    bus.subscribe(broken)
    unsubscribe = bus.subscribe(seen.append)
    bus.emit("state", state="idle")
    assert seen == [{"event": "state", "state": "idle"}]
    assert bus.last_state == seen[0]
    unsubscribe()
    bus.emit("notice", message="x")
    assert len(seen) == 1

    commands: list[dict] = []
    bus.on_command(commands.append)
    bus.command({"op": "toggle"})
    bus.command({"op": "rm -rf"})
    assert commands == [{"op": "toggle"}]

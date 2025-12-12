from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from polylogue.cli.settings_cli import run_settings_cli
from polylogue.commands import CommandEnv
from polylogue import settings as settings_module
from polylogue import config as config_module


class DummyUI:
    plain = True

    def __init__(self):
        self.console = self

    def print(self, *_args, **_kwargs):  # pragma: no cover - diagnostics not needed
        pass

    def summary(self, *_args, **_kwargs):  # pragma: no cover
        pass


def _make_args(**overrides):
    defaults = {
        "html": None,
        "theme": None,
        "reset": False,
        "json": False,
        "collapse_threshold": None,
        "output_root": None,
        "input_root": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_run_settings_cli_updates_store(tmp_path, monkeypatch):
    store = tmp_path / "config" / "polylogue" / "settings.json"
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", store)
    monkeypatch.setattr(config_module, "CONFIG_HOME", tmp_path / "config" / "polylogue")
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_module.CONFIG_HOME / "config.json")
    # Reload config to respect patched paths
    config_module.CONFIG = config_module.load_config()
    env = CommandEnv(ui=DummyUI())

    args = _make_args(html="on", theme="dark")
    run_settings_cli(args, env)

    assert store.exists()
    data = json.loads(store.read_text(encoding="utf-8"))
    assert data["html_previews"] is True
    assert data["html_theme"] == "dark"


def test_run_settings_cli_reset(tmp_path, monkeypatch):
    store = tmp_path / "config" / "polylogue" / "settings.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({"html_previews": True, "html_theme": "dark"}), encoding="utf-8")
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", store)
    monkeypatch.setattr(config_module, "CONFIG_HOME", tmp_path / "config" / "polylogue")
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_module.CONFIG_HOME / "config.json")
    config_module.CONFIG = config_module.load_config()
    env = CommandEnv(ui=DummyUI())

    args = _make_args(reset=True)
    run_settings_cli(args, env)

    assert not store.exists()
    assert env.settings.html_previews == settings_module.CONFIG.defaults.html_previews


def test_run_settings_cli_persists_config(tmp_path, monkeypatch):
    settings_store = tmp_path / "config" / "polylogue" / "settings.json"
    config_store = tmp_path / "config" / "polylogue" / "config.json"
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", settings_store)
    monkeypatch.setattr(config_module, "CONFIG_HOME", config_store.parent)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_store)
    config_module.CONFIG = config_module.load_config()
    env = CommandEnv(ui=DummyUI())

    args = _make_args(html="off", theme="light")
    args.output_root = tmp_path / "out"
    args.input_root = tmp_path / "inbox"
    run_settings_cli(args, env)

    assert config_store.exists()
    cfg = json.loads(config_store.read_text(encoding="utf-8"))
    assert cfg["paths"]["output_root"] == str(args.output_root)
    assert cfg["paths"]["input_root"] == str(args.input_root)
    assert cfg["ui"]["html"] is False
    assert cfg["ui"]["theme"] == "light"

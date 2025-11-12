from __future__ import annotations

import argparse
import json

from polylogue.cli.settings_cli import run_settings_cli
from polylogue.commands import CommandEnv
from polylogue import settings as settings_module


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
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_settings_cli_updates_store(tmp_path, monkeypatch):
    store = tmp_path / "config" / "polylogue" / "settings.json"
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", store)
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
    env = CommandEnv(ui=DummyUI())

    args = _make_args(reset=True)
    run_settings_cli(args, env)

    assert not store.exists()
    assert env.settings.html_previews == settings_module.CONFIG.defaults.html_previews

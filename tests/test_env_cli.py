from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

from polylogue.commands import CommandEnv
from polylogue.cli.env_cli import run_env_cli
from polylogue.cli.app import main
from polylogue.ui import UI
from tests.test_cli_integration import _configure_isolated_state


class DummyUI(UI):
    def __init__(self):
        super().__init__(plain=True)


def test_env_cli_json(monkeypatch, tmp_path, capsys):
    _configure_isolated_state(monkeypatch, tmp_path)
    env = CommandEnv(ui=DummyUI())
    for path in env.config.defaults.output_dirs.__dict__.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    args = Namespace(json=True)
    run_env_cli(args, env)
    payload = capsys.readouterr().out
    assert "schemaVersion" in payload
    assert "checks" in payload


def test_env_cli_via_main(monkeypatch, tmp_path):
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "env"])
    main()

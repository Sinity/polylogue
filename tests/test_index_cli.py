from __future__ import annotations

import json
from argparse import Namespace

from polylogue.cli.index_cli import run_index_cli
from polylogue.commands import CommandEnv


class DummyConsole:
    def print(self, *args, **kwargs):
        pass


class DummyUI:
    plain = True

    def __init__(self):
        self.console = DummyConsole()

    def summary(self, *_args, **_kwargs):
        pass


def test_index_cli_json(state_env, monkeypatch, capsys):
    env = CommandEnv(ui=DummyUI())
    args = Namespace(subcmd="check", repair=False, skip_qdrant=True, json=True)
    run_index_cli(args, env)
    payload = json.loads(capsys.readouterr().out)
    assert "sqlite" in payload
    assert payload["sqlite"]["status"] in {"ok", "updated"}

from __future__ import annotations

import json
from argparse import Namespace

from polylogue.cli.runs import run_runs_cli
from polylogue.commands import CommandEnv
from polylogue import util


class DummyConsole:
    def __init__(self):
        self.lines = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))


class DummyUI:
    plain = True

    def __init__(self):
        self.console = DummyConsole()


def test_runs_cli_json(state_env, capsys):
    util.add_run({"cmd": "sync drive", "provider": "drive", "count": 1})
    env = CommandEnv(ui=DummyUI())
    run_runs_cli(Namespace(limit=5, providers=None, commands=None, since=None, until=None, json=True), env)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["cmd"] == "sync drive"


def test_runs_cli_filters(state_env):
    util.add_run({"cmd": "sync drive", "provider": "drive", "count": 1})
    util.add_run({"cmd": "sync codex", "provider": "codex", "count": 2})
    env = CommandEnv(ui=DummyUI())
    run_runs_cli(Namespace(limit=10, providers="drive", commands=None, since=None, until=None, json=False), env)
    output = "\n".join(env.ui.console.lines)
    assert "sync drive" in output
    assert "codex" not in output


def test_runs_cli_since_until(state_env):
    util.add_run({"cmd": "sync drive", "provider": "drive", "count": 1, "timestamp": "2024-01-01T00:00:00Z"})
    util.add_run({"cmd": "sync codex", "provider": "codex", "count": 2, "timestamp": "2024-03-01T00:00:00Z"})
    env = CommandEnv(ui=DummyUI())
    run_runs_cli(
        Namespace(limit=10, providers=None, commands=None, since="2024-02-01", until="2024-04-01", json=False),
        env,
    )
    output = "\n".join(env.ui.console.lines)
    assert "codex" in output
    assert "drive" not in output

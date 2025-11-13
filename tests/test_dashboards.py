from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

from polylogue.cli.dashboards import run_dashboards_cli
from polylogue.commands import CommandEnv


class DummyUI:
    plain = True

    def __init__(self):
        self.console = self
        self.summaries = []

    def print(self, *_args, **_kwargs):
        pass

    def summary(self, title, lines):
        self.summaries.append((title, list(lines)))


def test_dashboards_json(monkeypatch, capsys):
    env = CommandEnv(ui=DummyUI())
    result = SimpleNamespace(
        provider_summary={"drive": {"runs": 2, "count": 3, "attachments": 1, "failures": 0}},
        recent_runs=[
            {"timestamp": "2024-01-01T00:00:00Z", "cmd": "sync drive", "provider": "drive", "count": 3, "duration": 1.2}
        ],
    )
    monkeypatch.setattr("polylogue.cli.dashboards.status_command", lambda _env, runs_limit=None: result)
    run_dashboards_cli(Namespace(json=True, runs_limit=5), env)
    payload = json.loads(capsys.readouterr().out)
    assert payload["providerSummary"]["drive"]["runs"] == 2
    assert payload["recentRuns"][0]["cmd"] == "sync drive"

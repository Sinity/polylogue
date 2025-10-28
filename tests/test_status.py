from __future__ import annotations

import json
import pytest

from polylogue.commands import CommandEnv, status_command
from polylogue import commands as cmd_module
from polylogue import util


class DummyConsole:
    def print(self, *args, **kwargs):
        pass


class DummyUI:
    plain = True
    console = DummyConsole()


def _patch_state(monkeypatch, tmp_path):
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    new_home = state_home / "polylogue"
    state_path = new_home / "state.json"
    runs_path = new_home / "runs.json"
    monkeypatch.setattr(util, "STATE_HOME", new_home, raising=False)
    monkeypatch.setattr(util, "STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(util, "RUNS_PATH", runs_path, raising=False)
    monkeypatch.setattr(cmd_module, "STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(cmd_module, "RUNS_PATH", runs_path, raising=False)
    return new_home


@pytest.fixture
def patched_state(tmp_path, monkeypatch):
    return _patch_state(monkeypatch, tmp_path)


def test_status_provider_summary(patched_state):
    runs = [
        {
            "cmd": "sync drive",
            "count": 1,
            "attachments": 2,
            "attachmentBytes": 2048,
            "tokens": 120,
            "skipped": 0,
            "pruned": 0,
            "diffs": 0,
            "duration": 1.25,
            "timestamp": "2024-01-01T00:00:00Z",
            "out": "/tmp/drive",
        },
        {
            "cmd": "codex-watch",
            "count": 2,
            "attachments": 3,
            "attachmentBytes": 4096,
            "tokens": 220,
            "skipped": 1,
            "pruned": 0,
            "diffs": 1,
            "duration": 2.5,
            "timestamp": "2024-01-02T00:00:00Z",
            "out": "/tmp/codex",
        },
    ]
    util.RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    util.RUNS_PATH.write_text(json.dumps(runs), encoding="utf-8")

    env = CommandEnv(ui=DummyUI())
    result = status_command(env)

    assert "drive" in result.provider_summary
    assert result.provider_summary["drive"]["count"] == 1
    assert result.provider_summary["drive"]["commands"] == ["sync drive"]
    assert result.provider_summary["drive"]["duration"] > 0
    assert "codex" in result.provider_summary
    assert result.provider_summary["codex"]["count"] == 2
    assert "codex-watch" in result.provider_summary["codex"]["commands"]
    assert result.run_summary["sync drive"]["duration"] > 0
    assert result.run_summary["sync drive"]["provider"] == "drive"

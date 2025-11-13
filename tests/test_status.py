from __future__ import annotations

import json
from types import SimpleNamespace

from polylogue.commands import CommandEnv, status_command
from polylogue.cli import build_parser
from polylogue.cli.status import run_status_cli
from polylogue import util


class DummyConsole:
    def print(self, *args, **kwargs):
        pass


class DummyUI:
    plain = True
    console = DummyConsole()


def test_status_provider_summary(state_env):
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
    for record in runs:
        util.add_run(record)

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
    assert result.run_summary["sync drive"].get("retries", 0) == 0
    assert result.runs and len(result.runs) == len(runs)


def test_status_dump_to_file(state_env, tmp_path):
    for idx in range(3):
        util.add_run(
            {
                "cmd": "render",
                "count": 1,
                "attachments": idx,
                "attachmentBytes": 512 * idx,
                "tokens": 10 * idx,
                "timestamp": f"2024-01-0{idx+1}T00:00:00Z",
            }
        )

    env = CommandEnv(ui=DummyUI())
    dump_path = tmp_path / "runs.json"
    args = _status_args(dump=str(dump_path), dump_limit=2, runs_limit=10)

    run_status_cli(args, env)

    data = json.loads(dump_path.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[-1]["cmd"] == "render"


def test_status_cli_parser_summary(state_env, capsys):
    util.add_run(
        {
            "cmd": "render",
            "provider": "render",
            "count": 1,
            "attachments": 0,
            "timestamp": "2024-04-01T00:00:00Z",
        }
    )
    parser = build_parser()
    args = parser.parse_args(["status", "--providers", "render", "--summary", "-", "--summary-only"])
    run_status_cli(args, CommandEnv(ui=DummyUI()))
    payload = json.loads(capsys.readouterr().out)
    assert list(payload["providerSummary"].keys()) == ["render"]


def test_status_provider_filter_and_summary(state_env, tmp_path, capsys):
    for name in ("drive", "codex"):
        util.add_run(
            {
                "cmd": f"sync {name}",
                "provider": name,
                "count": 1,
                "attachments": 0,
                "timestamp": "2024-02-01T00:00:00Z",
            }
        )

    env = CommandEnv(ui=DummyUI())
    summary_path = tmp_path / "summary.json"
    args = _status_args(json=True, providers="drive", summary=str(summary_path))

    run_status_cli(args, env)

    payload = json.loads(capsys.readouterr().out)
    assert list(payload["provider_summary"].keys()) == ["drive"]

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert list(summary_data["providerSummary"].keys()) == ["drive"]
    assert list(summary_data["runSummary"].keys()) == ["sync drive"]


def test_status_summary_only_stdout(state_env, capsys):
    util.add_run(
        {
            "cmd": "render",
            "provider": "render",
            "count": 1,
            "timestamp": "2024-03-01T00:00:00Z",
        }
    )

    env = CommandEnv(ui=DummyUI())
    args = _status_args(summary_only=True, summary="-", json=False)

    run_status_cli(args, env)

    out = capsys.readouterr().out
    data = json.loads(out)
    assert "runSummary" in data
    assert list(data["runSummary"].keys()) == ["render"]


def test_status_json_lines(state_env, capsys):
    util.add_run(
        {
            "cmd": "render",
            "provider": "render",
            "count": 1,
            "timestamp": "2024-03-02T00:00:00Z",
        }
    )
    env = CommandEnv(ui=DummyUI())
    args = _status_args(json_lines=True)
    run_status_cli(args, env)
    payload = capsys.readouterr().out.strip()
    assert payload.startswith("{") and payload.endswith("}")
    data = json.loads(payload)
    assert "generated_at" in data


def _status_args(**overrides):
    defaults = dict(
        json=False,
        json_lines=False,
        watch=False,
        interval=1.0,
        dump=None,
        dump_limit=100,
        runs_limit=200,
        dump_only=False,
        providers=None,
        summary=None,
        summary_only=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)

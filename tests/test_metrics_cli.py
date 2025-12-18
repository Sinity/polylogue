from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.metrics import run_metrics_cli
from polylogue.commands import CommandEnv
from polylogue.db import open_connection
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


def test_metrics_cli_prometheus_output(state_env, capsys):
    with open_connection(None) as conn:
        conn.execute(
            """
            INSERT INTO conversations (
                provider, conversation_id, slug, title, current_branch,
                root_message_id, last_updated, content_hash, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("chatgpt", "conv-1", "conv-1", "Test Conversation", None, None, None, None, None),
        )
        conn.commit()

    util.add_run({"cmd": "render", "provider": "render", "count": 1, "attachments": 2, "attachmentBytes": 2048})
    env = CommandEnv(ui=DummyUI())
    run_metrics_cli(SimpleNamespace(providers=None, runs_limit=0, json=False, serve=False, host="127.0.0.1", port=0), env)
    out = capsys.readouterr().out

    assert "polylogue_build_info" in out
    assert 'polylogue_db_conversations_total{provider="chatgpt"} 1' in out
    assert 'polylogue_run_records_total{cmd="render",provider="render"} 1' in out
    assert 'polylogue_attachments_processed_total{cmd="render",provider="render"} 2' in out


def test_click_browse_metrics_json(state_env):
    runner = CliRunner()
    result = runner.invoke(click_cli, ["--plain", "browse", "metrics", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "db" in payload
    assert "runs" in payload


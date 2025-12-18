from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.timeline import run_timeline_cli
from polylogue.commands import CommandEnv
from polylogue.db import open_connection, upsert_conversation


class DummyConsole:
    def __init__(self):
        self.lines = []

    def print(self, *args, **kwargs):  # noqa: ANN001, ARG002
        self.lines.append(" ".join(str(a) for a in args))


class DummyUI:
    plain = True

    def __init__(self):
        self.console = DummyConsole()


def test_timeline_cli_json(state_env, capsys):
    with open_connection(None) as conn:
        upsert_conversation(
            conn,
            provider="codex",
            conversation_id="conv-1",
            slug="conv-1",
            title="One",
            current_branch=None,
            root_message_id=None,
            last_updated="2024-01-01T00:00:00Z",
            content_hash=None,
            metadata={"token_count": 12, "word_count": 3, "attachments": 1, "attachment_bytes": 2048},
        )
        conn.commit()
    env = CommandEnv(ui=DummyUI())
    run_timeline_cli(SimpleNamespace(providers=None, limit=10, out=None, theme="light", open=False, json=True), env)
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["rows"][0]["provider"] == "codex"
    assert payload["rows"][0]["tokens"] == 12


def test_click_browse_timeline_json(state_env):
    runner = CliRunner()
    result = runner.invoke(click_cli, ["--plain", "browse", "timeline", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "rows" in payload


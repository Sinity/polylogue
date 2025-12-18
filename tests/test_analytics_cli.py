from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from polylogue.cli.analytics import run_analytics_cli
from polylogue.cli.click_app import cli as click_cli
from polylogue.commands import CommandEnv
from polylogue.db import open_connection, upsert_conversation


class DummyConsole:
    def print(self, *_args, **_kwargs):
        return None


class DummyUI:
    plain = True
    console = DummyConsole()


def test_analytics_cli_json(state_env, capsys):
    with open_connection(None) as conn:
        upsert_conversation(
            conn,
            provider="codex",
            conversation_id="conv-1",
            slug="conv-1",
            title="One",
            current_branch="branch-000",
            root_message_id=None,
            last_updated="2024-01-01T00:00:00Z",
            content_hash=None,
            metadata={},
        )
        conn.execute(
            "INSERT INTO branches(provider, conversation_id, branch_id, parent_branch_id, label, depth, is_current, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("codex", "conv-1", "branch-000", None, None, 0, 1, None),
        )
        conn.execute(
            """
            INSERT INTO messages(
                provider, conversation_id, branch_id, message_id, parent_id, position, timestamp, role, model,
                content_hash, content_text, content_json, rendered_text, raw_json, token_count, word_count,
                attachment_count, attachment_names, is_leaf, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "codex",
                "conv-1",
                "branch-000",
                "m-1",
                None,
                0,
                "2024-01-01T00:00:00Z",
                "assistant",
                "gpt-4o",
                None,
                "hello",
                None,
                "hello",
                "{}",
                10,
                2,
                0,
                None,
                1,
                None,
            ),
        )
        conn.commit()

    env = CommandEnv(ui=DummyUI())
    run_analytics_cli(SimpleNamespace(providers=None, model_limit=25, hotspot_limit=25, out=None, theme="light", open=False, json=True), env)
    payload = json.loads(capsys.readouterr().out)
    assert payload["providerRows"][0]["provider"] == "codex"
    assert any(row["role"] == "assistant" for row in payload["roleRows"])
    assert any(row["model"] == "gpt-4o" for row in payload["modelRows"])
    assert payload["branchHotspots"][0]["branchCount"] == 1


def test_click_browse_analytics_json(state_env):
    runner = CliRunner()
    result = runner.invoke(click_cli, ["--plain", "browse", "analytics", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "providerRows" in payload


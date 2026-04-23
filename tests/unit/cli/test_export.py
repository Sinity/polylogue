"""Tests for direct single-conversation export command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from tests.infra.storage_records import ConversationBuilder


def test_export_json_by_conversation_id(cli_workspace: dict[str, Path]) -> None:
    (
        ConversationBuilder(cli_workspace["db_path"], "conv-export")
        .provider("codex")
        .title("Exportable Conversation")
        .add_message("u1", role="user", text="Export this conversation")
        .save()
    )

    result = CliRunner().invoke(
        cli,
        ["export", "--format", "json", "conv-export"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "conv-export"
    assert payload["title"] == "Exportable Conversation"
    assert payload["messages"][0]["text"] == "Export this conversation"


def test_export_missing_conversation_exits_with_clear_message(cli_workspace: dict[str, Path]) -> None:
    result = CliRunner().invoke(cli, ["export", "missing-id"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Conversation not found: missing-id" in str(result.exception)

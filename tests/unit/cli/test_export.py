"""Tests for direct single-session export command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from tests.infra.storage_records import SessionBuilder


def _archive_index(cli_workspace: dict[str, Path]) -> Path:
    """Native index database the CLI/facade read for ``cli_workspace``."""
    return cli_workspace["archive_root"] / "index.db"


def test_export_json_by_session_id(cli_workspace: dict[str, Path]) -> None:
    builder = (
        SessionBuilder(_archive_index(cli_workspace), "conv-export")
        .provider("codex")
        .title("Exportable Session")
        .add_message("u1", role="user", text="Export this session")
    )
    builder.save()
    session_id = builder.native_session_id()

    result = CliRunner().invoke(
        cli,
        ["export", "--format", "json", session_id],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == session_id
    assert payload["title"] == "Exportable Session"
    assert payload["messages"][0]["text"] == "Export this session"


def test_export_missing_session_exits_with_clear_message(cli_workspace: dict[str, Path]) -> None:
    result = CliRunner().invoke(cli, ["export", "missing-id"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Session not found: missing-id" in str(result.exception)


def test_export_message_role_filter_applied(cli_workspace: dict[str, Path]) -> None:
    """``--message-role user`` must drop assistant + system messages from the export (#406)."""
    builder = (
        SessionBuilder(_archive_index(cli_workspace), "conv-roles")
        .provider("codex")
        .title("Mixed roles")
        .add_message("u1", role="user", text="user prose")
        .add_message("a1", role="assistant", text="assistant reply")
        .add_message("s1", role="system", text="system note")
    )
    builder.save()

    result = CliRunner().invoke(
        cli,
        ["--message-role", "user", "export", "--format", "json", builder.native_session_id()],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["user"], f"export ignored --message-role user: roles={roles!r}"


def test_export_dialogue_only_filter_applied(cli_workspace: dict[str, Path]) -> None:
    """``--dialogue-only`` keeps user+assistant, drops system + tool (#406)."""
    builder = (
        SessionBuilder(_archive_index(cli_workspace), "conv-dialogue")
        .provider("codex")
        .title("Dialogue-only")
        .add_message("u1", role="user", text="user prose")
        .add_message("a1", role="assistant", text="assistant reply")
        .add_message("s1", role="system", text="system note")
    )
    builder.save()

    result = CliRunner().invoke(
        cli,
        ["--dialogue-only", "export", "--format", "json", builder.native_session_id()],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    roles = sorted({message["role"] for message in payload["messages"]})
    assert roles == ["assistant", "user"], f"--dialogue-only export retained extra roles: {roles!r}"

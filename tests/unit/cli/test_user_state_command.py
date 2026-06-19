"""CLI contracts for durable reader state maintenance commands (#867)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from click.testing import CliRunner

from polylogue.cli.commands.user_state import state_command


def _result(output: str) -> dict[str, object]:
    parsed = json.loads(output)
    assert parsed["status"] == "ok"
    result = parsed["result"]
    assert isinstance(result, dict)
    return result


def _env() -> MagicMock:
    env = MagicMock()
    env.polylogue = MagicMock()
    env.polylogue.list_marks = AsyncMock(return_value=[])
    env.polylogue.add_mark = AsyncMock(return_value=True)
    env.polylogue.remove_mark = AsyncMock(return_value=False)
    env.polylogue.list_annotations = AsyncMock(return_value=[])
    env.polylogue.save_annotation = AsyncMock(return_value=True)
    env.polylogue.delete_annotation = AsyncMock(return_value=False)
    env.polylogue.list_views = AsyncMock(return_value=[])
    env.polylogue.save_view = AsyncMock(return_value=True)
    env.polylogue.delete_view = AsyncMock(return_value=False)
    env.polylogue.list_recall_packs = AsyncMock(return_value=[])
    env.polylogue.create_recall_pack = AsyncMock(return_value=True)
    env.polylogue.delete_recall_pack = AsyncMock(return_value=False)
    env.polylogue.list_workspaces = AsyncMock(return_value=[])
    env.polylogue.save_workspace = AsyncMock(return_value=True)
    env.polylogue.delete_workspace = AsyncMock(return_value=False)
    return env


def test_ops_state_marks_add_passes_message_target(cli_runner: CliRunner) -> None:
    env = _env()

    result = cli_runner.invoke(
        state_command,
        [
            "marks",
            "add",
            "conv-1",
            "pin",
            "--target-type",
            "message",
            "--message-id",
            "msg-1",
            "--format",
            "json",
        ],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _result(result.output)
    assert payload["status"] == "ok"
    env.polylogue.add_mark.assert_awaited_once_with(
        "conv-1",
        "pin",
        target_type="message",
        target_id=None,
        message_id="msg-1",
    )


def test_ops_state_annotations_list_returns_envelope(cli_runner: CliRunner) -> None:
    env = _env()
    env.polylogue.list_annotations = AsyncMock(
        return_value=[
            {
                "annotation_id": "ann-1",
                "target_type": "message",
                "target_id": "msg-1",
                "session_id": "conv-1",
                "message_id": "msg-1",
                "note_text": "Important",
                "created_at": "2026-05-15T00:00:00+00:00",
                "updated_at": "2026-05-15T00:00:01+00:00",
            }
        ]
    )

    result = cli_runner.invoke(
        state_command,
        ["annotations", "list", "--session-id", "conv-1", "--format", "json"],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _result(result.output)
    assert payload["total"] == 1
    items = payload["items"]
    assert isinstance(items, list)
    assert items[0]["annotation_id"] == "ann-1"
    env.polylogue.list_annotations.assert_awaited_once_with(
        session_id="conv-1",
        target_type=None,
        target_id=None,
        message_id=None,
    )


def test_ops_state_saved_view_save_validates_and_canonicalizes_query(cli_runner: CliRunner) -> None:
    env = _env()

    result = cli_runner.invoke(
        state_command,
        [
            "saved-views",
            "save",
            "Claude Code",
            '{"query":"auth","origin":"claude-code-session"}',
            "--view-id",
            "view-auth",
            "--format",
            "json",
        ],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _result(result.output)
    assert payload["view_id"] == "view-auth"
    env.polylogue.save_view.assert_awaited_once_with(
        "view-auth",
        "Claude Code",
        '{"origin":"claude-code-session","query":"auth"}',
    )


def test_ops_state_saved_view_rejects_unknown_query_param(cli_runner: CliRunner) -> None:
    env = _env()

    result = cli_runner.invoke(
        state_command,
        ["saved-views", "save", "Bad", '{"not_a_filter":true}', "--format", "json"],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code != 0
    assert "SessionQuerySpec" in result.output


def test_ops_state_recall_pack_save_passes_typed_items(cli_runner: CliRunner) -> None:
    env = _env()

    result = cli_runner.invoke(
        state_command,
        [
            "recall-packs",
            "save",
            "pack-1",
            "Handoff",
            "--item-json",
            '{"target_type":"session","session_id":"conv-1"}',
            "--item-json",
            '{"target_type":"annotation","annotation_id":"ann-1"}',
            "--payload-json",
            '{"summary":"handoff"}',
            "--format",
            "json",
        ],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _result(result.output)
    assert payload["pack_id"] == "pack-1"
    env.polylogue.create_recall_pack.assert_awaited_once_with(
        "pack-1",
        "Handoff",
        '{"items":[{"session_id":"conv-1","target_type":"session"},{"annotation_id":"ann-1","target_type":"annotation"}],"summary":"handoff"}',
    )


def test_ops_state_workspace_save_passes_canonical_layout_and_targets(cli_runner: CliRunner) -> None:
    env = _env()

    result = cli_runner.invoke(
        state_command,
        [
            "workspaces",
            "save",
            "workspace-1",
            "Investigation",
            "--mode",
            "compare",
            "--open-targets-json",
            '[{"session_id":"conv-1","target_type":"session"}]',
            "--layout-json",
            '{"panes":[{"width":0.5},{"width":0.5}]}',
            "--active-target-json",
            '{"session_id":"conv-1","target_type":"session"}',
            "--format",
            "json",
        ],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _result(result.output)
    assert payload["workspace_id"] == "workspace-1"
    env.polylogue.save_workspace.assert_awaited_once_with(
        "workspace-1",
        "Investigation",
        "compare",
        '[{"session_id":"conv-1","target_type":"session"}]',
        '{"panes":[{"width":0.5},{"width":0.5}]}',
        '{"session_id":"conv-1","target_type":"session"}',
    )

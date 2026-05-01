from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.shell_completion import CompletionItem
from click.testing import CliRunner

from polylogue.archive.filter.filters import ConversationFilter
from polylogue.cli import shell_completion_values
from polylogue.cli.commands.neighbors import (
    _candidate_heading,
    _render_plain,
    _score_label,
    neighbors_command,
)
from polylogue.cli.commands.tags import tags_command
from polylogue.cli.filter_picker import _pick_index, pick_filter
from polylogue.lib.conversation.neighbor_candidates import (
    ConversationNeighborCandidate,
    NeighborDiscoveryError,
    NeighborReason,
)
from polylogue.lib.models import ConversationSummary
from polylogue.types import ConversationId, Provider


def _ctx_param() -> tuple[click.Context, click.Parameter]:
    return click.Context(click.Command("polylogue")), click.Option(["--value"])


@contextmanager
def _connection(rows: list[dict[str, object]]) -> Iterator[object]:
    cursor = SimpleNamespace(fetchall=lambda: rows)
    conn = SimpleNamespace(execute=lambda sql, params: cursor)
    yield conn


@contextmanager
def _broken_connection() -> Iterator[object]:
    raise sqlite3.OperationalError("boom")
    yield


def _candidate(*, message_id: str | None = "message-1") -> ConversationNeighborCandidate:
    return ConversationNeighborCandidate(
        summary=ConversationSummary(
            id=ConversationId("candidate"),
            provider=Provider.CODEX,
            title="Archive Lock Retries",
            updated_at=datetime(2026, 4, 23, 8, 30, tzinfo=timezone.utc),
            message_count=2,
        ),
        rank=1,
        score=3.20,
        reasons=(
            NeighborReason(
                kind="same_title",
                detail="same normalized title",
                weight=3.0,
                evidence=message_id,
            ),
        ),
        source_conversation_id="target",
    )


def _picker_result(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        provider="claude-code",
        display_title=f"Conversation {index} " + ("x" * 60),
        display_date=datetime(2026, 4, 23, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    ("choice", "total_results", "expected"),
    [
        ("", 3, 0),
        ("1", 3, 0),
        ("3", 3, 2),
        ("0", 3, None),
        ("4", 3, None),
        ("abc", 3, None),
    ],
)
def test_pick_index_covers_blank_valid_and_invalid_choices(
    choice: str,
    total_results: int,
    expected: int | None,
) -> None:
    assert _pick_index(choice, total_results) == expected


@pytest.mark.asyncio
async def test_pick_filter_returns_first_for_non_tty_and_none_for_empty_results() -> None:
    filter_obj = cast(ConversationFilter, SimpleNamespace(list=AsyncMock(return_value=[_picker_result(1)])))
    with patch("sys.stdout.isatty", return_value=False):
        assert await pick_filter(filter_obj) == _picker_result(1)

    empty_filter = cast(ConversationFilter, SimpleNamespace(list=AsyncMock(return_value=[])))
    assert await pick_filter(empty_filter) is None


@pytest.mark.asyncio
async def test_pick_filter_tty_path_renders_menu_and_handles_blank_invalid_and_interrupt() -> None:
    results = [_picker_result(index) for index in range(1, 23)]
    filter_obj = cast(ConversationFilter, SimpleNamespace(list=AsyncMock(return_value=results)))

    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value=""),
        patch("builtins.print") as mock_print,
    ):
        assert await pick_filter(filter_obj) == results[0]

    printed = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
    assert any("22 matching conversations" in line for line in printed)
    assert any("... and 2 more" in line for line in printed)

    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="99"),
        patch("builtins.print"),
    ):
        assert await pick_filter(filter_obj) is None

    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", side_effect=KeyboardInterrupt),
        patch("builtins.print"),
    ):
        assert await pick_filter(filter_obj) is None


def test_shell_completion_helpers_cover_csv_prefix_rows_and_trimming(tmp_path: Path) -> None:
    assert shell_completion_values._split_csv_incomplete("alpha,beta") == ("alpha,", "beta")
    assert shell_completion_values._split_csv_incomplete("alpha, beta,g") == ("alpha,beta,", "g")
    assert shell_completion_values._split_csv_incomplete("cla") == ("", "cla")

    item = CompletionItem("tool", type="plain", help="help")
    prefixed = shell_completion_values._with_csv_prefix([item], "alpha,")
    assert [completion.value for completion in prefixed] == ["alpha,tool"]
    assert shell_completion_values._with_csv_prefix([item], "") == [item]

    assert shell_completion_values._trim_help("line one\nline two", limit=9) == "line one…"
    assert shell_completion_values._trim_help("short help") == "short help"

    db = tmp_path / "archive.sqlite"
    with patch("polylogue.cli.shell_completion_values.db_path", return_value=db):
        assert shell_completion_values._db_exists() is False
        db.write_text("", encoding="utf-8")
        assert shell_completion_values._db_exists() is True

    rows: list[dict[str, object]] = [{"value": "alpha", "help": "example help"}]
    items = shell_completion_values._rows_to_completion_items(
        cast(list[sqlite3.Row], rows),
        value_column="value",
        help_builder=lambda row: str(row["help"]),
    )
    assert [(item.value, item.help) for item in items] == [("alpha", "example help")]


def test_fetch_rows_handles_missing_db_errors_and_success() -> None:
    with patch("polylogue.cli.shell_completion_values._db_exists", return_value=False):
        assert shell_completion_values._fetch_rows("SELECT 1", ()) == []

    with (
        patch("polylogue.cli.shell_completion_values._db_exists", return_value=True),
        patch("polylogue.cli.shell_completion_values.open_read_connection", _broken_connection),
    ):
        assert shell_completion_values._fetch_rows("SELECT 1", ()) == []

    rows: list[dict[str, object]] = [{"conversation_id": "conv-1"}]
    with (
        patch("polylogue.cli.shell_completion_values._db_exists", return_value=True),
        patch("polylogue.cli.shell_completion_values.open_read_connection", lambda: _connection(rows)),
    ):
        assert shell_completion_values._fetch_rows("SELECT 1", ("x",)) == rows


def test_completion_functions_cover_provider_conversation_tag_tool_and_open_targets() -> None:
    ctx, param = _ctx_param()

    provider_items = shell_completion_values.complete_provider_values(ctx, param, "chatgpt,cla")
    assert [item.value for item in provider_items] == ["chatgpt,claude-ai", "chatgpt,claude-code"]

    conversation_rows = [
        {
            "conversation_id": "conv-1",
            "provider_name": "claude-code",
            "display_title": "A long title " * 10,
        }
    ]
    tag_rows = [{"tag_name": "review", "cnt": 3}]
    tool_rows = [{"normalized_tool_name": "read_file", "cnt": 7}]

    with patch(
        "polylogue.cli.shell_completion_values._fetch_rows",
        side_effect=[conversation_rows, conversation_rows, tag_rows, tool_rows],
    ):
        conversation_items = shell_completion_values.complete_conversation_ids(ctx, param, "conv")
        open_items = shell_completion_values.complete_open_targets(ctx, param, "conv")
        tag_items = shell_completion_values.complete_tag_values(ctx, param, "alpha,rev")
        tool_items = shell_completion_values.complete_tool_values(ctx, param, "read")

    assert conversation_items[0].value == "conv-1"
    assert conversation_items[0].help is not None and "claude-code" in conversation_items[0].help
    assert open_items[0].value == "conv-1"
    assert [item.value for item in tag_items] == ["alpha,review"]
    assert tag_items[0].help == "3 conversations"
    assert [item.value for item in tool_items] == ["read_file"]
    assert tool_items[0].help == "7 actions"


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_tags_command_plain_paths_cover_empty_hint_and_tabular_counts(cli_runner: CliRunner) -> None:
    env = MagicMock()
    env.repository.list_tags = AsyncMock(return_value={})

    empty = cli_runner.invoke(tags_command, ["--provider", "chatgpt"], obj=env, catch_exceptions=False)
    assert empty.exit_code == 0
    assert "No tags found for provider 'chatgpt'." in empty.output
    assert "Hint: use --add-tag" in empty.output

    env.repository.list_tags = AsyncMock(return_value={"alpha": 5, "beta": 2})
    table = cli_runner.invoke(tags_command, ["--count", "1"], obj=env, catch_exceptions=False)
    assert table.exit_code == 0
    assert "Tags (all providers, 1 total):" in table.output
    assert "alpha" in table.output
    assert "beta" not in table.output

    env.repository.list_tags = AsyncMock(return_value={"alpha": 5})
    with patch("polylogue.cli.commands.tags.emit_success") as emit_success:
        json_result = cli_runner.invoke(tags_command, ["--json"], obj=env, catch_exceptions=False)
    assert json_result.exit_code == 0
    emit_success.assert_called_once_with({"tags": {"alpha": 5}})

    env.repository.list_tags = AsyncMock(return_value={})
    generic_empty = cli_runner.invoke(tags_command, [], obj=env, catch_exceptions=False)
    assert generic_empty.exit_code == 0
    assert "No tags found." in generic_empty.output


def test_neighbor_helpers_and_command_cover_plain_rendering_and_errors(cli_runner: CliRunner) -> None:
    assert _score_label(3.20) == "3.2"
    assert "candidate" in _candidate_heading(_candidate(message_id=None))

    with patch("click.echo") as echo:
        _render_plain([])
        _render_plain([_candidate()])

    echoed = [call.args[0] for call in echo.call_args_list if call.args]
    assert "No neighboring candidates found." in echoed
    assert any("Neighbor candidates (1):" in line for line in echoed)
    assert any("same_title: same normalized title (message-1)" in line for line in echoed)

    env = MagicMock()
    env.operations = MagicMock()
    env.operations.neighbor_candidates = AsyncMock(return_value=[_candidate()])
    plain = cli_runner.invoke(
        neighbors_command,
        ["--query", "lock retries", "--limit", "0", "--window-hours", "0"],
        obj=env,
        catch_exceptions=False,
    )
    assert plain.exit_code == 0
    assert "Neighbor candidates (1):" in plain.output
    env.operations.neighbor_candidates.assert_called_once_with(
        conversation_id=None,
        query="lock retries",
        provider=None,
        limit=1,
        window_hours=1,
    )

    error_env = MagicMock()
    error_env.operations = MagicMock()
    error_env.operations.neighbor_candidates = AsyncMock(side_effect=NeighborDiscoveryError("no candidates"))
    error = cli_runner.invoke(neighbors_command, ["--query", "lock retries"], obj=error_env)
    assert error.exit_code != 0
    assert "no candidates" in str(error.exception)

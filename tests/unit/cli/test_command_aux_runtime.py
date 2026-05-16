from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.shell_completion import CompletionItem
from click.testing import CliRunner

from polylogue.archive.conversation.neighbor_candidates import (
    ConversationNeighborCandidate,
    NeighborDiscoveryError,
    NeighborReason,
)
from polylogue.archive.models import ConversationSummary
from polylogue.cli import shell_completion_values
from polylogue.cli.commands.neighbors import (
    _candidate_heading,
    _render_plain,
    _score_label,
    neighbors_command,
)
from polylogue.cli.commands.tags import tags_command
from polylogue.types import ConversationId, Provider


def _ctx_param() -> tuple[click.Context, click.Parameter]:
    return click.Context(click.Command("polylogue")), click.Option(["--value"])


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

    from polylogue.operations.archive import CompletionAggregate

    aggregates = [CompletionAggregate("alpha", 7)]
    items = shell_completion_values._aggregates_to_items(aggregates, unit="actions")
    assert [(item.value, item.help) for item in items] == [("alpha", "7 actions")]


def test_completion_functions_cover_provider_conversation_tag_tool_and_open_targets() -> None:
    ctx, param = _ctx_param()

    provider_items = shell_completion_values.complete_provider_values(ctx, param, "chatgpt,cla")
    assert [item.value for item in provider_items] == ["chatgpt,claude-ai", "chatgpt,claude-code"]
    action_items = shell_completion_values.complete_action_values(ctx, param, "file")
    action_sequence_items = shell_completion_values.complete_action_sequence_values(ctx, param, "shell,file")
    message_type_items = shell_completion_values.complete_message_type_values(ctx, param, "m")
    retrieval_lane_items = shell_completion_values.complete_retrieval_lane_values(ctx, param, "h")

    # All archive-backed completions go through ArchiveOperations now;
    # mock the per-aggregate methods on a shared ops stub.
    from polylogue.archive.conversation.models import Conversation
    from polylogue.operations.archive import CompletionAggregate

    mock_conv = MagicMock(spec=Conversation)
    mock_conv.id = "conv-1"
    mock_conv.title = "Test Conv"
    mock_conv.provider = "claude-code"
    mock_ops = MagicMock(
        list_conversations=AsyncMock(return_value=[mock_conv]),
        list_tags=AsyncMock(return_value={"review": 3}),
        list_session_repo_names=AsyncMock(return_value=[CompletionAggregate("polylogue", 4)]),
        list_session_cwd_prefixes=AsyncMock(return_value=[CompletionAggregate("/realm/project/polylogue", 2)]),
        list_action_tool_names=AsyncMock(return_value=[CompletionAggregate("read_file", 7)]),
    )

    async def _with_operations_stub(action):  # type: ignore[no-untyped-def]
        return await action(mock_ops)

    with (
        patch(
            "polylogue.cli.shell_completion_values._with_operations",
            new=_with_operations_stub,
        ),
        patch("polylogue.cli.shell_completion_values._db_exists", return_value=True),
    ):
        conversation_items = shell_completion_values.complete_conversation_ids(ctx, param, "conv")
        open_items = shell_completion_values.complete_open_targets(ctx, param, "conv")
        tag_items = shell_completion_values.complete_tag_values(ctx, param, "alpha,rev")
        repo_items = shell_completion_values.complete_repo_values(ctx, param, "old,poly")
        cwd_items = shell_completion_values.complete_cwd_prefix_values(ctx, param, "/realm")
        tool_items = shell_completion_values.complete_tool_values(ctx, param, "read")

    assert [item.value for item in action_items] == ["file_read", "file_write", "file_edit"]
    assert "shell,file_read" in [item.value for item in action_sequence_items]
    assert [item.value for item in message_type_items] == ["message"]
    assert [item.value for item in retrieval_lane_items] == ["hybrid"]
    assert conversation_items[0].value == "conv-1"
    assert conversation_items[0].help is not None and "claude-code" in conversation_items[0].help
    assert open_items[0].value == "conv-1"
    assert [item.value for item in tag_items] == ["alpha,review"]
    assert tag_items[0].help == "3 conversations"
    assert [item.value for item in repo_items] == ["old,polylogue"]
    assert repo_items[0].help == "4 sessions"
    assert [item.value for item in cwd_items] == ["/realm/project/polylogue"]
    assert cwd_items[0].help == "2 sessions"
    assert [item.value for item in tool_items] == ["read_file"]
    assert tool_items[0].help == "7 actions"


def test_completion_source_registry_covers_descriptor_sources() -> None:
    from polylogue.archive.query.fields import query_completion_sources

    assert tuple(shell_completion_values.COMPLETION_SOURCE_HANDLERS) == query_completion_sources()
    assert shell_completion_values.complete_query_source("message_type") is (
        shell_completion_values.complete_message_type_values
    )


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.mark.xfail(
    reason="provider-aware empty hint and Rich table renderer in tags command not yet implemented (#1012)",
    strict=True,
)
def test_tags_command_plain_paths_cover_empty_hint_and_tabular_counts(cli_runner: CliRunner) -> None:
    env = MagicMock()
    env.ui = MagicMock()
    env.polylogue = MagicMock()
    env.polylogue.list_tags = AsyncMock(return_value={})

    empty = cli_runner.invoke(tags_command, ["--provider", "chatgpt"], obj=env, catch_exceptions=False)
    assert empty.exit_code == 0
    assert "No tags found for provider 'chatgpt'." in empty.output
    assert "Hint: use --add-tag" in empty.output

    env.polylogue.list_tags = AsyncMock(return_value={"alpha": 5, "beta": 2})
    table = cli_runner.invoke(tags_command, ["--count", "1"], obj=env, catch_exceptions=False)
    assert table.exit_code == 0
    rendered_table = env.ui.print.call_args.args[0]
    assert rendered_table.title == "Tags (all providers, 1 total)"
    assert list(rendered_table.columns[0].cells) == ["alpha"]
    assert list(rendered_table.columns[1].cells) == ["5"]

    env.polylogue.list_tags = AsyncMock(return_value={"alpha": 5})
    with patch("polylogue.cli.commands.tags.emit_success") as emit_success:
        json_result = cli_runner.invoke(tags_command, ["--format", "json"], obj=env, catch_exceptions=False)
    assert json_result.exit_code == 0
    emit_success.assert_called_once_with({"tags": {"alpha": 5}})

    env.polylogue.list_tags = AsyncMock(return_value={})
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
    env.polylogue = MagicMock()
    env.polylogue.neighbor_candidates = AsyncMock(return_value=[_candidate()])
    plain = cli_runner.invoke(
        neighbors_command,
        ["--query", "lock retries", "--limit", "0", "--window-hours", "0"],
        obj=env,
        catch_exceptions=False,
    )
    assert plain.exit_code == 0
    assert "Neighbor candidates (1):" in plain.output
    env.polylogue.neighbor_candidates.assert_called_once_with(
        conversation_id=None,
        query="lock retries",
        provider=None,
        limit=1,
        window_hours=1,
    )

    error_env = MagicMock()
    error_env.polylogue = MagicMock()
    error_env.polylogue.neighbor_candidates = AsyncMock(side_effect=NeighborDiscoveryError("no candidates"))
    error = cli_runner.invoke(neighbors_command, ["--query", "lock retries"], obj=error_env)
    assert error.exit_code != 0
    assert "no candidates" in str(error.exception)

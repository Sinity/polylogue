"""Generalized contracts for CLI query execution helpers."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from hypothesis import HealthCheck, given, settings
from rich.console import Console

from polylogue.cli.query import async_execute_query, project_query_results
from polylogue.cli.query_actions import apply_modifiers, apply_transform, delete_conversations
from polylogue.cli.query import no_results, summary_to_dict
from polylogue.cli.query_output import (
    _output_summary_list,
    _render_conversation_rich,
    _send_output,
    output_results,
    output_stats_by_semantic_summaries,
    output_stats_by_summaries,
    output_stats_sql,
)
from polylogue.cli.query import (
    QueryAction,
    QueryExecutionPlan,
    QueryMutationSpec,
    QueryOutputSpec,
    QueryRoute,
    resolve_query_route,
)
from polylogue.cli.query_output import output_stats_by_conversations
from polylogue.cli.types import AppEnv
from polylogue.lib.models import Conversation, Message
from polylogue.lib.query_spec import ConversationQuerySpec, QuerySpecError
from polylogue.services import build_runtime_services
from polylogue.storage.store import ContentBlockRecord, ConversationRecord, MessageRecord
from tests.infra.strategies import (
    ConversationSummarySpec,
    build_conversation_summary,
    build_message_counts,
    query_delete_case_strategy,
    query_mutation_case_strategy,
    send_output_case_strategy,
    summary_output_case_strategy,
    summary_stats_case_strategy,
)

pytestmark = pytest.mark.query_routing


def _make_env(*, repo: MagicMock | None = None, config: MagicMock | None = None) -> AppEnv:
    ui = MagicMock()
    ui.plain = True
    ui.console = MagicMock()
    ui.confirm = MagicMock(return_value=True)
    if repo is not None:
        queries = repo.queries
        if not isinstance(queries.get_conversation, AsyncMock):
            queries.get_conversation = AsyncMock(return_value=None)
        if not isinstance(queries.get_conversation_stats, AsyncMock):
            queries.get_conversation_stats = AsyncMock(return_value={})
        if not isinstance(queries.get_message_counts_batch, AsyncMock):
            queries.get_message_counts_batch = AsyncMock(return_value={})
        if not isinstance(queries.aggregate_message_stats, AsyncMock):
            queries.aggregate_message_stats = AsyncMock(return_value={})
    return AppEnv(ui=ui, services=build_runtime_services(config=config, repository=repo))


def _fields_arg(fields: tuple[str, ...] | None) -> str | None:
    return None if not fields else ",".join(fields)


def _structured_rows(case) -> list[dict[str, object]]:
    rows = [
        summary_to_dict(build_conversation_summary(spec), spec.message_count)
        for spec in case.summaries
    ]
    if case.selected_fields:
        selected = set(case.selected_fields)
        rows = [{key: value for key, value in row.items() if key in selected} for row in rows]
    return rows


def _csv_rows(case) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in case.summaries:
        summary = build_conversation_summary(spec)
        rows.append(
            {
                "id": spec.conversation_id,
                "date": summary.display_date.strftime("%Y-%m-%d") if summary.display_date else "",
                "provider": spec.provider,
                "title": summary.display_title or "",
                "messages": str(spec.message_count),
                "tags": ",".join(spec.tags),
                "summary": spec.summary or "",
            }
        )
    return rows


def _sample_summary_spec() -> ConversationSummarySpec:
    return ConversationSummarySpec(
        conversation_id="conv-law-1",
        provider="claude-ai",
        title="Lawful Conversation",
        summary="summary",
        tags=("law",),
        created_at=None,
        updated_at=None,
        message_count=2,
    )


def _build_plan(case: str) -> QueryExecutionPlan:
    selection = MagicMock()
    selection.similar_text = None
    action = QueryAction.SHOW
    output = QueryOutputSpec("markdown", ("stdout",), None, False, None, False)
    mutation = QueryMutationSpec((), (), False, False, False)
    stats_dimension = None

    if case == "count":
        action = QueryAction.COUNT
    elif case == "stats_sql":
        action = QueryAction.STATS
    elif case == "stats_by_summaries":
        action = QueryAction.STATS_BY
        stats_dimension = "provider"
    elif case == "stats_by_semantic_summaries":
        action = QueryAction.STATS_BY
        stats_dimension = "action"
    elif case == "stream":
        action = QueryAction.STREAM
        output = QueryOutputSpec("json", ("stdout", "browser"), None, True, "strip-tools", False)
    elif case == "modify":
        action = QueryAction.MODIFY
        mutation = QueryMutationSpec((("priority", "1"),), (), False, False, True)
    elif case == "delete":
        action = QueryAction.DELETE
        mutation = QueryMutationSpec((), (), True, False, True)
    elif case == "open":
        action = QueryAction.OPEN
    elif case == "summary_list":
        output = QueryOutputSpec("markdown", ("stdout",), None, False, None, True)

    return QueryExecutionPlan(
        selection=selection,
        action=action,
        output=output,
        mutation=mutation,
        stats_dimension=stats_dimension,
    )


def _sample_conversation() -> Conversation:
    return Conversation(
        id="conv-transform",
        provider="claude-ai",
        title="Transform Contract",
        messages=[
            Message(id="m-user", role="user", text="hello"),
            Message(
                id="m-thinking",
                role="assistant",
                text="chain",
                content_blocks=[{"type": "thinking", "text": "chain"}],
            ),
            Message(id="m-tool", role="tool", text="tool output"),
            Message(id="m-assistant", role="assistant", text="answer"),
        ],
    )


def _sample_output_conversation(conversation_id: str = "conv-output") -> Conversation:
    return Conversation(
        id=conversation_id,
        provider="claude-ai",
        title="Output Contract",
        messages=[
            Message(id=f"{conversation_id}-user", role="user", text="hello"),
            Message(id=f"{conversation_id}-assistant", role="assistant", text="world"),
        ],
    )


def _sample_semantic_conversation() -> Conversation:
    return Conversation(
        id="conv-semantic",
        provider="claude-code",
        title="Semantic Slice Contract",
        messages=[
            Message(id="m-user", role="user", text="inspect the repo"),
            Message(
                id="m-other",
                role="assistant",
                text="mystery tool",
                content_blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Mystery",
                        "tool_id": "tool-other",
                        "tool_input": {"path": "/realm/project/polylogue/README.md"},
                    }
                ],
            ),
            Message(
                id="m-edit",
                role="assistant",
                text="edit models",
                content_blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-edit",
                        "tool_input": {"file_path": "/realm/project/polylogue/polylogue/lib/models.py"},
                    }
                ],
            ),
            Message(
                id="m-search",
                role="assistant",
                text="search docs",
                content_blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Grep",
                        "tool_id": "tool-search",
                        "tool_input": {"pattern": "polylogue"},
                    }
                ],
            ),
        ],
    )


def _make_recording_env() -> tuple[AppEnv, io.StringIO]:
    buffer = io.StringIO()
    env = _make_env()
    env.ui.console = Console(file=buffer, width=120, force_terminal=False, color_system=None)
    return env, buffer


def _summary_group_key(spec: ConversationSummarySpec, dimension: str) -> str:
    if dimension == "provider":
        return spec.provider or "unknown"
    if dimension == "month":
        dt = spec.updated_at or spec.created_at
        return dt.strftime("%Y-%m") if dt else "unknown"
    if dimension == "year":
        dt = spec.updated_at or spec.created_at
        return dt.strftime("%Y") if dt else "unknown"
    if dimension == "day":
        dt = spec.updated_at or spec.created_at
        return dt.strftime("%Y-%m-%d") if dt else "unknown"
    return "all"


@settings(max_examples=60, deadline=None)
@given(case=query_mutation_case_strategy())
def test_apply_modifiers_contract(case) -> None:
    repo = MagicMock()
    repo.update_metadata = AsyncMock()
    repo.add_tag = AsyncMock()
    env = _make_env(repo=repo)
    env.ui.confirm.return_value = case.confirm
    results = [build_conversation_summary(spec) for spec in case.summaries]
    params = {
        "set_meta": list(case.set_meta),
        "add_tag": list(case.add_tags),
        "dry_run": case.dry_run,
        "force": case.force,
    }

    with patch("click.echo") as mock_echo:
        asyncio.run(apply_modifiers(env, results, params, repo))

    should_confirm = len(results) > 10 and not case.force and not case.dry_run
    if should_confirm:
        env.ui.confirm.assert_called_once_with("Proceed?", default=False)
    else:
        env.ui.confirm.assert_not_called()

    should_apply = not case.dry_run and (not should_confirm or case.confirm)
    expected_meta_calls = len(results) * len(case.set_meta) if should_apply else 0
    expected_tag_calls = len(results) * len(case.add_tags) if should_apply else 0

    assert repo.update_metadata.await_count == expected_meta_calls
    assert repo.add_tag.await_count == expected_tag_calls

    if case.dry_run:
        printed = " ".join(
            call.args[0] for call in env.ui.console.print.call_args_list if call.args
        )
        assert "Sample of affected conversations" in printed
        assert any(spec.conversation_id[:24] in printed for spec in case.summaries[:5])
    elif should_apply:
        messages = [call.args[0] for call in mock_echo.call_args_list if call.args]
        if case.add_tags:
            assert f"Added tags to {len(results)} conversations" in messages
        if case.set_meta:
            assert f"Set {len(results) * len(case.set_meta)} metadata field(s)" in messages


@settings(max_examples=60, deadline=None)
@given(case=query_delete_case_strategy())
def test_delete_conversations_contract(case) -> None:
    repo = MagicMock()
    repo.delete_conversation = AsyncMock(side_effect=list(case.delete_results))
    env = _make_env(repo=repo)
    env.ui.confirm.return_value = case.confirm
    results = [build_conversation_summary(spec) for spec in case.summaries]
    params = {"dry_run": case.dry_run, "force": case.force}

    with patch("click.echo") as mock_echo:
        asyncio.run(delete_conversations(env, results, params, repo))

    should_confirm = not case.force and not case.dry_run
    if should_confirm:
        env.ui.confirm.assert_called_once_with("Proceed?", default=False)
    else:
        env.ui.confirm.assert_not_called()

    should_delete = not case.dry_run and (case.force or case.confirm)
    assert repo.delete_conversation.await_count == (len(results) if should_delete else 0)

    echoed = [call.args[0] for call in mock_echo.call_args_list if call.args]
    provider_counts: dict[str, int] = {}
    display_dates: list[str] = []
    arrow = "\u2192"
    for spec in case.summaries:
        provider_counts[spec.provider] = provider_counts.get(spec.provider, 0) + 1
        display_date = build_conversation_summary(spec).display_date
        if display_date is not None:
            display_dates.append(display_date.strftime("%Y-%m-%d"))

    if case.dry_run or not case.force:
        echoed_text = "\n".join(echoed)
        for provider, count in provider_counts.items():
            assert f"{provider}: {count}" in echoed_text
        if display_dates:
            if min(display_dates) == max(display_dates):
                assert f"Date: {display_dates[0]}" in echoed_text
            else:
                assert f"Date range: {min(display_dates)} {arrow} {max(display_dates)}" in echoed_text

    if should_delete:
        assert f"Deleted {sum(case.delete_results)} conversation(s)" in echoed


@settings(max_examples=40, deadline=None)
@given(case=summary_output_case_strategy())
@pytest.mark.parametrize("output_format", ["json", "yaml", "csv", "text"])
def test_output_summary_list_contract(case, output_format: str) -> None:
    repo = MagicMock()
    repo.queries.get_message_counts_batch = AsyncMock(return_value=build_message_counts(case.summaries))
    env = _make_env(repo=repo)
    summaries = [build_conversation_summary(spec) for spec in case.summaries]
    params = {"output_format": output_format}
    if output_format in {"json", "yaml"}:
        params["fields"] = _fields_arg(case.selected_fields)
    env.ui.plain = output_format == "text"

    with patch("click.echo") as mock_echo:
        asyncio.run(_output_summary_list(env, summaries, params, repo))

    if output_format == "json":
        assert json.loads(mock_echo.call_args[0][0]) == _structured_rows(case)
    elif output_format == "yaml":
        assert yaml.safe_load(mock_echo.call_args[0][0]) == _structured_rows(case)
    elif output_format == "csv":
        assert list(csv.DictReader(io.StringIO(mock_echo.call_args[0][0]))) == _csv_rows(case)
    else:
        assert mock_echo.call_count == 1
        rendered = mock_echo.call_args[0][0]
        for spec in case.summaries:
            assert spec.conversation_id[:24] in rendered


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(case=send_output_case_strategy())
def test_send_output_routes_destination_contract(case) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        env = _make_env()
        content = "test output"
        output_file = tmp_path / "output.txt"
        destinations: list[str] = []
        if case.to_stdout:
            destinations.append("stdout")
        if case.to_file:
            destinations.append(str(output_file))
        if case.to_browser:
            destinations.append("browser")
        if case.to_clipboard:
            destinations.append("clipboard")

        with (
            patch("click.echo") as mock_echo,
            patch("polylogue.cli.query_output._open_in_browser") as mock_browser,
            patch("polylogue.cli.query_output._copy_to_clipboard") as mock_clipboard,
        ):
            _send_output(env, content, destinations, case.output_format, None)

        assert mock_echo.call_count == (1 if case.to_stdout else 0)
        assert mock_browser.call_count == (1 if case.to_browser else 0)
        assert mock_clipboard.call_count == (1 if case.to_clipboard else 0)
        if case.to_file:
            assert output_file.read_text(encoding="utf-8") == content
        else:
            assert not output_file.exists()


@pytest.mark.parametrize(
    ("case", "can_use_summaries", "expected_route"),
    [
        ("count", False, QueryRoute.COUNT),
        ("stats_sql", True, QueryRoute.STATS_SQL),
        ("stats_by_summaries", True, QueryRoute.SUMMARY_STATS),
        ("stats_by_summaries", False, QueryRoute.STATS_BY),
        ("modify", True, QueryRoute.SUMMARY_MODIFY),
        ("modify", False, QueryRoute.MODIFY),
        ("delete", True, QueryRoute.SUMMARY_DELETE),
        ("delete", False, QueryRoute.DELETE),
        ("open", False, QueryRoute.OPEN),
        ("summary_list", True, QueryRoute.SUMMARY_LIST),
        ("summary_list", False, QueryRoute.SHOW),
        ("show", False, QueryRoute.SHOW),
    ],
)
def test_resolve_query_route_contract(case: str, can_use_summaries: bool, expected_route: QueryRoute) -> None:
    plan = _build_plan(case)
    assert resolve_query_route(plan, can_use_summaries=can_use_summaries) == expected_route


def test_resolve_query_route_uses_full_stats_for_action_dimension() -> None:
    plan = QueryExecutionPlan(
        selection=ConversationQuerySpec(),
        action=QueryAction.STATS_BY,
        output=QueryOutputSpec("markdown", ("stdout",), None, False, None, False),
        mutation=QueryMutationSpec((), (), False, False, False),
        stats_dimension="action",
    )

    assert resolve_query_route(plan, can_use_summaries=True) == QueryRoute.STATS_BY


def test_resolve_query_route_uses_full_stats_for_tool_dimension() -> None:
    plan = QueryExecutionPlan(
        selection=ConversationQuerySpec(),
        action=QueryAction.STATS_BY,
        output=QueryOutputSpec("markdown", ("stdout",), None, False, None, False),
        mutation=QueryMutationSpec((), (), False, False, False),
        stats_dimension="tool",
    )

    assert resolve_query_route(plan, can_use_summaries=True) == QueryRoute.STATS_BY


@settings(max_examples=40, deadline=None)
@given(case=summary_stats_case_strategy())
def test_output_stats_by_summaries_contract(case) -> None:
    env, buffer = _make_recording_env()
    summaries = [build_conversation_summary(spec) for spec in case.summaries]
    msg_counts = build_message_counts(case.summaries)

    output_stats_by_summaries(env, summaries, msg_counts, case.dimension)

    rendered = buffer.getvalue()
    assert f"Matched: {len(case.summaries)} conversations (by {case.dimension})" in rendered

    grouped_counts: dict[str, tuple[int, int]] = {}
    for spec in case.summaries:
        key = _summary_group_key(spec, case.dimension)
        convs, messages = grouped_counts.get(key, (0, 0))
        grouped_counts[key] = (convs + 1, messages + spec.message_count)

    for key, (convs, messages) in grouped_counts.items():
        assert key in rendered
        assert f"{convs:,}" in rendered
        assert f"{messages:,}" in rendered

    assert "TOTAL" in rendered
    assert f"{len(case.summaries):,}" in rendered
    assert f"{sum(spec.message_count for spec in case.summaries):,}" in rendered


def test_output_stats_by_summaries_json_contract() -> None:
    env = _make_env()
    summaries = [
        build_conversation_summary(
            ConversationSummarySpec(
                conversation_id="conv-a",
                provider="claude-ai",
                title="A",
                summary="sa",
                tags=("x",),
                created_at=None,
                updated_at=None,
                message_count=3,
            )
        ),
        build_conversation_summary(
            ConversationSummarySpec(
                conversation_id="conv-b",
                provider="chatgpt",
                title="B",
                summary="sb",
                tags=("y",),
                created_at=None,
                updated_at=None,
                message_count=4,
            )
        ),
    ]
    msg_counts = {"conv-a": 3, "conv-b": 4}

    with patch("click.echo") as mock_echo:
        output_stats_by_summaries(env, summaries, msg_counts, "provider", output_format="json")

    payload = json.loads(mock_echo.call_args.args[0])
    env.ui.console.print.assert_not_called()
    assert payload == {
        "dimension": "provider",
        "multi_membership": False,
        "rows": [
            {"group": "chatgpt", "conversations": 1, "messages": 4},
            {"group": "claude-ai", "conversations": 1, "messages": 3},
        ],
        "summary": {"group": "TOTAL", "conversations": 2, "messages": 7},
    }


@pytest.mark.asyncio
async def test_output_stats_by_semantic_summaries_json_contract() -> None:
    env = _make_env()
    repo = MagicMock()
    repo.queries.get_conversations_batch = AsyncMock(
        side_effect=[
            [
                ConversationRecord(
                    conversation_id="conv-law-1",
                    provider_name="claude-code",
                    provider_conversation_id="ext-conv-law-1",
                    title="Read contract",
                    created_at="2025-01-01T00:00:00Z",
                    updated_at="2025-01-01T00:00:00Z",
                    sort_key=1735689600,
                    content_hash="hash-read",
                )
            ],
            [
                ConversationRecord(
                    conversation_id="conv-b",
                    provider_name="claude-code",
                    provider_conversation_id="ext-conv-b",
                    title="Shell contract",
                    created_at="2025-01-01T00:00:00Z",
                    updated_at="2025-01-01T00:00:00Z",
                    sort_key=1735689600,
                    content_hash="hash-shell",
                )
            ],
        ]
    )
    repo.queries.get_messages_batch = AsyncMock(
        side_effect=[
            {
                "conv-law-1": [
                    MessageRecord(
                        message_id="m-read",
                        conversation_id="conv-law-1",
                        role="assistant",
                        text="read file",
                        sort_key=1735689600,
                        content_hash="msg-hash-read",
                        content_blocks=[
                            ContentBlockRecord(
                                block_id="blk-read",
                                message_id="m-read",
                                conversation_id="conv-law-1",
                                block_index=0,
                                type="tool_use",
                                text=None,
                                tool_name="Read",
                                tool_id=None,
                                tool_input=json.dumps({"file_path": "/tmp/a.py"}),
                                semantic_type="file_read",
                            )
                        ],
                    )
                ]
            },
            {
                "conv-b": [
                    MessageRecord(
                        message_id="m-shell",
                        conversation_id="conv-b",
                        role="assistant",
                        text="run tests",
                        sort_key=1735689600,
                        content_hash="msg-hash-shell",
                        content_blocks=[
                            ContentBlockRecord(
                                block_id="blk-shell",
                                message_id="m-shell",
                                conversation_id="conv-b",
                                block_index=0,
                                type="tool_use",
                                text=None,
                                tool_name="Bash",
                                tool_id=None,
                                tool_input=json.dumps({"command": "pytest -q"}),
                                semantic_type="shell",
                            )
                        ],
                    )
                ]
            },
        ]
    )
    summaries = [
        build_conversation_summary(_sample_summary_spec()),
        build_conversation_summary(
            ConversationSummarySpec(
                conversation_id="conv-b",
                provider="claude-code",
                title="Shell contract",
                summary="summary",
                tags=("law",),
                created_at=None,
                updated_at=None,
                message_count=1,
            )
        ),
    ]

    with patch("click.echo") as mock_echo:
        await output_stats_by_semantic_summaries(
            env,
            summaries,
            repo,
            "action",
            selection=ConversationQuerySpec(),
            output_format="json",
            batch_size=1,
        )

    payload = json.loads(mock_echo.call_args.args[0])
    env.ui.console.print.assert_not_called()
    assert payload == {
        "dimension": "action",
        "multi_membership": True,
        "rows": [
            {"group": "file_read", "conversations": 1, "facts": 1, "messages": 1},
            {"group": "shell", "conversations": 1, "facts": 1, "messages": 1},
        ],
        "summary": {"group": "MATCHED", "conversations": 2, "facts": 2, "messages": 2},
    }
    assert repo.queries.get_conversations_batch.await_count == 2
    assert repo.queries.get_messages_batch.await_count == 2


def test_output_stats_by_conversations_action_slice_respects_selected_action() -> None:
    env = _make_env()
    conversation = _sample_semantic_conversation()
    selection = ConversationQuerySpec(action_terms=("other",))

    with patch("click.echo") as mock_echo:
        output_stats_by_conversations(
            env,
            [conversation],
            "tool",
            selection=selection,
            output_format="json",
        )

    payload = json.loads(mock_echo.call_args.args[0])
    assert payload == {
        "dimension": "tool",
        "multi_membership": True,
        "rows": [
            {"group": "mystery", "conversations": 1, "facts": 1, "messages": 1},
        ],
        "summary": {"group": "MATCHED", "conversations": 1, "facts": 1, "messages": 1},
    }


def test_output_stats_by_conversations_action_slice_respects_selected_tool() -> None:
    env = _make_env()
    conversation = _sample_semantic_conversation()
    selection = ConversationQuerySpec(tool_terms=("edit",))

    with patch("click.echo") as mock_echo:
        output_stats_by_conversations(
            env,
            [conversation],
            "action",
            selection=selection,
            output_format="json",
        )

    payload = json.loads(mock_echo.call_args.args[0])
    assert payload == {
        "dimension": "action",
        "multi_membership": True,
        "rows": [
            {"group": "file_edit", "conversations": 1, "facts": 1, "messages": 1},
        ],
        "summary": {"group": "MATCHED", "conversations": 1, "facts": 1, "messages": 1},
    }


def test_output_stats_by_conversations_action_slice_respects_selected_path() -> None:
    env = _make_env()
    conversation = _sample_semantic_conversation()
    selection = ConversationQuerySpec(path_terms=("/realm/project/polylogue/README.md",))

    with patch("click.echo") as mock_echo:
        output_stats_by_conversations(
            env,
            [conversation],
            "action",
            selection=selection,
            output_format="json",
        )

    payload = json.loads(mock_echo.call_args.args[0])
    assert payload == {
        "dimension": "action",
        "multi_membership": True,
        "rows": [
            {"group": "other", "conversations": 1, "facts": 1, "messages": 1},
        ],
        "summary": {"group": "MATCHED", "conversations": 1, "facts": 1, "messages": 1},
    }


def test_output_results_no_results_contract() -> None:
    env = _make_env()

    with (
        patch("polylogue.cli.query_output.no_results", side_effect=SystemExit(2)) as mock_no_results,
        patch("polylogue.cli.query_output._render_conversation_rich") as mock_render,
        patch("polylogue.cli.query_output._send_output") as mock_send,
        patch("polylogue.cli.query_output.format_conversation") as mock_format,
        pytest.raises(SystemExit) as exc_info,
    ):
        output_results(env, [], {})

    assert exc_info.value.code == 2
    mock_no_results.assert_called_once_with(env, {})
    mock_render.assert_not_called()
    mock_send.assert_not_called()
    mock_format.assert_not_called()


@pytest.mark.parametrize(
    ("label", "plain", "conversations", "params", "expected"),
    [
        (
            "single-rich-markdown",
            False,
            [_sample_output_conversation()],
            {"output_format": "markdown", "output": "stdout", "list_mode": False},
            "render-rich",
        ),
        (
            "single-nonrich",
            True,
            [_sample_output_conversation()],
            {"output_format": "html", "output": "stdout,browser", "fields": "id,title"},
            "format-single",
        ),
        (
            "multi-list",
            True,
            [_sample_output_conversation("conv-output-1"), _sample_output_conversation("conv-output-2")],
            {"output_format": "json", "output": "stdout", "fields": "id,provider"},
            "format-list",
        ),
    ],
)
def test_output_results_projection_contract(label: str, plain: bool, conversations: list[Conversation], params: dict[str, object], expected: str) -> None:
    env = _make_env()
    env.ui.plain = plain

    with (
        patch("polylogue.cli.query_output._render_conversation_rich") as mock_render,
        patch("polylogue.cli.query_output._send_output") as mock_send,
        patch("polylogue.cli.query_output.format_conversation", return_value="<html>ok</html>") as mock_format,
        patch("polylogue.cli.query_output._format_list", return_value="formatted-list") as mock_format_list,
    ):
        output_results(env, conversations, params)

    if expected == "render-rich":
        mock_render.assert_called_once_with(env, conversations[0])
        mock_send.assert_not_called()
        mock_format.assert_not_called()
        mock_format_list.assert_not_called()
    elif expected == "format-single":
        mock_format.assert_called_once_with(conversations[0], "html", "id,title")
        mock_send.assert_called_once_with(env, "<html>ok</html>", ["stdout", "browser"], "html", conversations[0])
        mock_render.assert_not_called()
        mock_format_list.assert_not_called()
    else:
        mock_format_list.assert_called_once_with(conversations, "json", "id,provider")
        mock_send.assert_called_once_with(env, "formatted-list", ["stdout"], "json", None)
        mock_render.assert_not_called()
        mock_format.assert_not_called()


def test_render_conversation_rich_contract() -> None:
    env, buffer = _make_recording_env()
    conversation = Conversation(
        id="conv-rich",
        provider="claude-ai",
        title="Rich Contract",
        messages=[
            Message(id="m-user", role="user", text="**Hello** world"),
            Message(
                id="m-thinking",
                role="assistant",
                text="x" * 620,
                content_blocks=[{"type": "thinking"}],
            ),
            Message(id="m-empty", role="assistant", text=""),
        ],
    )

    _render_conversation_rich(env, conversation)

    rendered = buffer.getvalue()
    assert "Rich Contract" in rendered
    assert "claude-ai" in rendered
    assert "User" in rendered
    assert "Hello" in rendered
    assert "Thinking" in rendered
    assert "... (620 chars)" in rendered


def test_project_query_results_contract() -> None:
    plan = QueryExecutionPlan(
        selection=ConversationQuerySpec(),
        action=QueryAction.SHOW,
        output=QueryOutputSpec("markdown", ("stdout",), None, True, "strip-all", False),
        mutation=QueryMutationSpec((), (), False, False, False),
    )
    conversation = _sample_conversation()

    projected = project_query_results([conversation], plan)

    assert [message.id for message in projected[0].messages] == ["m-user", "m-assistant"]
    assert [message.id for message in conversation.messages] == [
        "m-user",
        "m-thinking",
        "m-tool",
        "m-assistant",
    ]


@pytest.mark.parametrize(
    ("case", "expected_helper"),
    [
        ("count", "count"),
        ("stats_sql", "stats_sql"),
        ("stats_by_summaries", "stats_by_summaries"),
        ("stats_by_semantic_summaries", "stats_by_semantic_summaries"),
        ("stream", "stream"),
        ("modify", "modify"),
        ("delete", "delete"),
        ("open", "open"),
        ("summary_list", "summary_list"),
        ("show", "show"),
    ],
)
def test_async_execute_query_action_routing_contract(case, expected_helper) -> None:
    env = _make_env(repo=MagicMock(), config=MagicMock())
    repo = env.repository
    summary = build_conversation_summary(_sample_summary_spec())
    plan = _build_plan(case)
    filter_chain = MagicMock()
    filter_chain.count = AsyncMock(return_value=3)
    filter_chain.can_use_summaries.return_value = True
    filter_chain.list_summaries = AsyncMock(return_value=[summary])
    filter_chain.list = AsyncMock(return_value=[MagicMock()])
    plan.selection.build_filter.return_value = filter_chain

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("click.echo") as mock_echo,
        patch("polylogue.cli.query_output._output_summary_list", new_callable=AsyncMock) as mock_output_summary_list,
        patch("polylogue.cli.query_output.output_stats_sql", new_callable=AsyncMock) as mock_output_stats_sql,
        patch("polylogue.cli.query_output.output_stats_by_summaries") as mock_output_stats_by_summaries,
        patch("polylogue.cli.query_output.output_stats_by_semantic_summaries", new_callable=AsyncMock) as mock_output_stats_by_semantic_summaries,
        patch("polylogue.cli.query_output._output_stats_by") as mock_output_stats_by,
        patch("polylogue.cli.query_actions.apply_modifiers", new_callable=AsyncMock) as mock_apply_modifiers,
        patch("polylogue.cli.query_actions.delete_conversations", new_callable=AsyncMock) as mock_delete_conversations,
        patch("polylogue.cli.query_output._open_result") as mock_open_result,
        patch("polylogue.cli.query_output.output_results") as mock_output_results,
        patch("polylogue.cli.query_actions.resolve_stream_target", new_callable=AsyncMock, return_value="conv-stream") as mock_stream_target,
        patch("polylogue.cli.query_output.stream_conversation", new_callable=AsyncMock) as mock_stream_conversation,
    ):
        repo.queries.get_message_counts_batch = AsyncMock(return_value={str(summary.id): 2})
        asyncio.run(async_execute_query(env, {"limit": 7}))

    plan.selection.build_filter.assert_called_once_with(repo, vector_provider=None)

    if expected_helper == "count":
        filter_chain.count.assert_awaited_once()
        filter_chain.list.assert_not_called()
        filter_chain.list_summaries.assert_not_called()
        mock_echo.assert_called_with(3)
    elif expected_helper == "stats_sql":
        filter_chain.list.assert_not_called()
        filter_chain.list_summaries.assert_not_called()
        mock_output_stats_sql.assert_awaited_once()
    elif expected_helper == "stats_by_summaries":
        filter_chain.list.assert_not_called()
        filter_chain.list_summaries.assert_awaited_once()
        repo.queries.get_message_counts_batch.assert_awaited_once_with([str(summary.id)])
        mock_output_stats_by_summaries.assert_called_once()
        mock_output_stats_by_semantic_summaries.assert_not_called()
        mock_output_stats_by.assert_not_called()
    elif expected_helper == "stats_by_semantic_summaries":
        filter_chain.list.assert_not_called()
        filter_chain.list_summaries.assert_awaited_once()
        mock_output_stats_by_semantic_summaries.assert_awaited_once()
        mock_output_stats_by.assert_not_called()
    elif expected_helper == "modify":
        filter_chain.list.assert_not_called()
        filter_chain.list_summaries.assert_awaited_once()
        mock_apply_modifiers.assert_awaited_once()
    elif expected_helper == "delete":
        filter_chain.list.assert_not_called()
        filter_chain.list_summaries.assert_awaited_once()
        mock_delete_conversations.assert_awaited_once()
    elif expected_helper == "open":
        filter_chain.list.assert_awaited_once()
        filter_chain.list_summaries.assert_not_called()
        mock_open_result.assert_called_once()
    elif expected_helper == "stream":
        filter_chain.list.assert_not_called()
        filter_chain.list_summaries.assert_not_called()
        mock_stream_target.assert_awaited_once()
        mock_stream_conversation.assert_awaited_once_with(
            env,
            repo,
            "conv-stream",
            output_format="json-lines",
            dialogue_only=True,
            message_limit=7,
        )
        warnings = [call.args[0] for call in mock_echo.call_args_list if call.args]
        assert any("--transform is ignored in --stream mode" in line for line in warnings)
        assert any("--output stdout,browser is ignored in --stream mode" in line for line in warnings)
    elif expected_helper == "summary_list":
        filter_chain.list.assert_not_called()
        filter_chain.list_summaries.assert_awaited_once()
        mock_output_summary_list.assert_awaited_once()
    else:
        filter_chain.list.assert_awaited_once()
        filter_chain.list_summaries.assert_not_called()
        mock_output_results.assert_called_once()


def test_async_execute_query_stats_by_falls_back_to_full_results_without_summaries_contract() -> None:
    env = _make_env(repo=MagicMock(), config=MagicMock())
    plan = _build_plan("stats_by_summaries")
    filter_chain = MagicMock()
    filter_chain.can_use_summaries.return_value = False
    filter_chain.list = AsyncMock(return_value=[_sample_conversation()])
    filter_chain.list_summaries = AsyncMock(return_value=[build_conversation_summary(_sample_summary_spec())])
    plan.selection.build_filter.return_value = filter_chain

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("polylogue.cli.query_output._output_stats_by") as mock_output_stats_by,
        patch("polylogue.cli.query_output.output_stats_by_summaries") as mock_output_stats_by_summaries,
    ):
        asyncio.run(async_execute_query(env, {}))

    filter_chain.list.assert_awaited_once()
    filter_chain.list_summaries.assert_not_called()
    mock_output_stats_by.assert_called_once()
    mock_output_stats_by_summaries.assert_not_called()
    assert mock_output_stats_by.call_args.kwargs["selection"] is plan.selection


def test_async_execute_query_semantic_stats_by_uses_summary_batches_contract() -> None:
    env = _make_env(repo=MagicMock(), config=MagicMock())
    plan = _build_plan("stats_by_semantic_summaries")
    filter_chain = MagicMock()
    filter_chain.can_use_summaries.return_value = True
    filter_chain.list = AsyncMock(return_value=[_sample_conversation()])
    filter_chain.list_summaries = AsyncMock(return_value=[build_conversation_summary(_sample_summary_spec())])
    plan.selection.build_filter.return_value = filter_chain

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("polylogue.cli.query_output.output_stats_by_semantic_summaries", new_callable=AsyncMock) as mock_output_stats_by_semantic_summaries,
        patch("polylogue.cli.query_output._output_stats_by") as mock_output_stats_by,
    ):
        asyncio.run(async_execute_query(env, {}))

    filter_chain.list.assert_not_called()
    filter_chain.list_summaries.assert_awaited_once()
    mock_output_stats_by_semantic_summaries.assert_awaited_once()
    mock_output_stats_by.assert_not_called()


def test_async_execute_query_semantic_stats_by_falls_back_without_summaries_contract() -> None:
    env = _make_env(repo=MagicMock(), config=MagicMock())
    plan = _build_plan("stats_by_semantic_summaries")
    filter_chain = MagicMock()
    filter_chain.can_use_summaries.return_value = False
    filter_chain.list = AsyncMock(return_value=[_sample_conversation()])
    filter_chain.list_summaries = AsyncMock(return_value=[build_conversation_summary(_sample_summary_spec())])
    plan.selection.build_filter.return_value = filter_chain

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("polylogue.cli.query_output.output_stats_by_semantic_summaries", new_callable=AsyncMock) as mock_output_stats_by_semantic_summaries,
        patch("polylogue.cli.query_output._output_stats_by") as mock_output_stats_by,
    ):
        asyncio.run(async_execute_query(env, {}))

    filter_chain.list.assert_awaited_once()
    filter_chain.list_summaries.assert_not_called()
    mock_output_stats_by_semantic_summaries.assert_not_called()
    mock_output_stats_by.assert_called_once()


def test_async_execute_query_summary_list_no_results_contract() -> None:
    env = _make_env(repo=MagicMock(), config=MagicMock())
    plan = _build_plan("summary_list")
    filter_chain = MagicMock()
    filter_chain.can_use_summaries.return_value = True
    filter_chain.list_summaries = AsyncMock(return_value=[])
    plan.selection.build_filter.return_value = filter_chain

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("polylogue.cli.query.no_results", side_effect=SystemExit(2)) as mock_no_results,
        pytest.raises(SystemExit) as exc_info,
    ):
        asyncio.run(async_execute_query(env, {}))

    assert exc_info.value.code == 2
    mock_no_results.assert_called_once_with(env, plan.selection)


def test_async_execute_query_query_spec_error_contract() -> None:
    env = _make_env(repo=MagicMock(), config=MagicMock())
    plan = _build_plan("show")
    plan.selection.build_filter.side_effect = QuerySpecError("since", "not-a-date")

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("click.echo") as mock_echo,
        pytest.raises(SystemExit) as exc_info,
    ):
        asyncio.run(async_execute_query(env, {}))

    assert exc_info.value.code == 1
    assert [call.args[0] for call in mock_echo.call_args_list if call.args] == [
        "Error: Cannot parse date: 'not-a-date'",
        "Hint: use ISO format (2025-01-15), relative ('yesterday', 'last week'), or month (2025-01)",
    ]
    assert all(call.kwargs.get("err") is True for call in mock_echo.call_args_list)


def test_async_execute_query_show_projects_results_before_output_contract() -> None:
    env = _make_env(repo=MagicMock(), config=MagicMock())
    selection = MagicMock()
    selection.similar_text = None
    plan = QueryExecutionPlan(
        selection=selection,
        action=QueryAction.SHOW,
        output=QueryOutputSpec("markdown", ("stdout",), None, True, "strip-all", False),
        mutation=QueryMutationSpec((), (), False, False, False),
    )
    filter_chain = MagicMock()
    filter_chain.can_use_summaries.return_value = False
    filter_chain.list = AsyncMock(return_value=[_sample_conversation()])
    selection.build_filter.return_value = filter_chain

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("polylogue.cli.query_output.output_results") as mock_output_results,
    ):
        asyncio.run(async_execute_query(env, {}))

    projected_results = mock_output_results.call_args.args[1]
    assert [message.id for message in projected_results[0].messages] == ["m-user", "m-assistant"]


@pytest.mark.parametrize(
    ("params", "expected_lines"),
    [
            (
                {"provider": "claude-ai", "limit": 5},
                [
                    "No conversations matched filters:",
                    "  provider: claude-ai",
                    "Hint: try broadening your filters or use --list to browse",
                ],
            ),
        (
            ConversationQuerySpec(),
            ["No conversations matched."],
        ),
    ],
)
def test_no_results_contract(params, expected_lines) -> None:
    env = _make_env()

    with patch("click.echo") as mock_echo, pytest.raises(SystemExit) as exc_info:
        no_results(env, params)

    assert exc_info.value.code == 2
    observed_lines = [call.args[0] for call in mock_echo.call_args_list if call.args]
    assert observed_lines == expected_lines
    assert all(call.kwargs.get("err") is True for call in mock_echo.call_args_list)


@pytest.mark.parametrize(
    ("transform", "expected_ids"),
    [
        ("strip-tools", ["m-user", "m-thinking", "m-assistant"]),
        ("strip-thinking", ["m-user", "m-tool", "m-assistant"]),
        ("strip-all", ["m-user", "m-assistant"]),
    ],
)
def test_apply_transform_contract(transform: str, expected_ids: list[str]) -> None:
    conversation = _sample_conversation()

    transformed = apply_transform([conversation], transform)

    assert [message.id for message in transformed[0].messages] == expected_ids
    assert [message.id for message in conversation.messages] == [
        "m-user",
        "m-thinking",
        "m-tool",
        "m-assistant",
    ]


@pytest.mark.asyncio
async def test_output_stats_sql_uses_summary_pushdown_contract() -> None:
    env = _make_env()
    repo = MagicMock()
    repo.queries.aggregate_message_stats = AsyncMock(
        return_value={
            "total": 9,
            "user": 4,
            "assistant": 5,
            "words_approx": 42,
            "attachments": 2,
            "min_sort_key": 1704067200,
            "max_sort_key": 1704153600,
            "providers": {"claude-ai": 2, "chatgpt": 1},
        }
    )
    summary_specs = (
        ConversationSummarySpec(
            conversation_id="conv-a",
            provider="claude-ai",
            title="A",
            summary="",
            tags=(),
            created_at=None,
            updated_at=None,
            message_count=3,
        ),
        ConversationSummarySpec(
            conversation_id="conv-b",
            provider="chatgpt",
            title="B",
            summary="",
            tags=(),
            created_at=None,
            updated_at=None,
            message_count=4,
        ),
    )
    summaries = [build_conversation_summary(spec) for spec in summary_specs]
    filter_chain = MagicMock()
    filter_chain.describe.return_value = ["provider=claude-ai", "tag=law"]
    filter_chain.can_use_summaries.return_value = True
    filter_chain.list_summaries = AsyncMock(return_value=summaries)

    await output_stats_sql(env, filter_chain, repo)

    filter_chain.list_summaries.assert_awaited_once()
    filter_chain.count.assert_not_called()
    repo.queries.aggregate_message_stats.assert_awaited_once_with(["conv-a", "conv-b"])
    printed = [call.args[0] for call in env.ui.console.print.call_args_list if call.args]
    assert printed == [
        "\nConversations: 2\n",
        "Messages: 9 total (4 user, 5 assistant)",
        "Words: ~42",
        "Providers: claude-ai (2), chatgpt (1)",
        "Attachments: 2",
        "Date range: 2024-01-01 to 2024-01-02",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("described", "can_use_summaries", "expected_message"),
    [
        (["provider=claude-ai"], True, "No conversations matched."),
        ([], False, "No conversations in archive."),
    ],
)
async def test_output_stats_sql_empty_paths_contract(described: list[str], can_use_summaries: bool, expected_message: str) -> None:
    env = _make_env()
    repo = MagicMock()
    repo.queries.aggregate_message_stats = AsyncMock()

    filter_chain = MagicMock()
    filter_chain.describe.return_value = described
    filter_chain.can_use_summaries.return_value = can_use_summaries
    filter_chain.list_summaries = AsyncMock(return_value=[])
    filter_chain.count = AsyncMock(return_value=0)

    await output_stats_sql(env, filter_chain, repo)

    env.ui.console.print.assert_called_once_with(expected_message)


@pytest.mark.asyncio
async def test_output_stats_sql_archive_scope_includes_embedding_state() -> None:
    env = _make_env()
    repo = MagicMock()
    repo.queries.aggregate_message_stats = AsyncMock(
        return_value={
            "total": 9,
            "user": 4,
            "assistant": 5,
            "words_approx": 42,
            "providers": {"claude-ai": 2, "chatgpt": 1},
            "attachments": 2,
            "min_sort_key": 1704067200,
            "max_sort_key": 1704153600,
        }
    )
    repo.get_archive_stats = AsyncMock(
        return_value=SimpleNamespace(
            total_conversations=2,
            embedded_conversations=1,
            embedded_messages=5,
            pending_embedding_conversations=1,
            embedding_coverage=50.0,
        )
    )

    filter_chain = MagicMock()
    filter_chain.describe.return_value = []
    filter_chain.can_use_summaries.return_value = False
    filter_chain.count = AsyncMock(return_value=2)

    await output_stats_sql(env, filter_chain, repo)

    repo.get_archive_stats.assert_awaited_once()
    printed = [call.args[0] for call in env.ui.console.print.call_args_list if call.args]
    assert printed == [
        "\nConversations: 2\n",
        "Messages: 9 total (4 user, 5 assistant)",
        "Words: ~42",
        "Providers: claude-ai (2), chatgpt (1)",
        "Attachments: 2",
        "Embeddings: 1/2 convs, 5 msgs (50.0%), pending 1",
        "Date range: 2024-01-01 to 2024-01-02",
    ]
    repo.queries.aggregate_message_stats.assert_awaited_once_with(None)


# ---------------------------------------------------------------------------
# Merged from test_search.py (2026-03-15)
# ---------------------------------------------------------------------------

# =============================================================================
# TEST DATA TABLES (module-level constants)
# =============================================================================

SEARCH_FILTER_CASES = [
    ("provider", ["Python", "-p", "chatgpt"], 0, None),
    ("since_valid", ["Python", "--since", "__DYNAMIC_DATE__"], 0, None),
    ("since_invalid", ["Python", "--since", "not-a-date"], 1, "date"),
    ("limit_list", ["JavaScript", "--limit", "1", "--list"], 0, None),
]

SEARCH_FORMAT_CASES = [
    ("json_list", ["Python", "-f", "json", "--list"], "json_list"),
    ("json_single", ["JavaScript", "-f", "json", "--limit", "1"], "json_single"),
    ("list_mode", ["async", "--list"], "plain_list"),
    ("markdown", ["Rust", "-f", "markdown", "--limit", "1"], "markdown"),
]


class TestSearchQueryContracts:
    """Matrix coverage for search filters and output formats."""

    @pytest.mark.parametrize(
        "case_id,args,expected_exit,error_hint",
        SEARCH_FILTER_CASES,
    )
    def test_filter_contract(self, search_workspace, case_id, args, expected_exit, error_hint):
        """Filter flags produce expected status codes and validation behavior."""
        from polylogue.cli import cli

        runner = CliRunner()
        resolved_args = list(args)
        if "__DYNAMIC_DATE__" in resolved_args:
            idx = resolved_args.index("__DYNAMIC_DATE__")
            from datetime import datetime, timedelta

            resolved_args[idx] = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        result = runner.invoke(cli, ["--plain", *resolved_args])
        assert result.exit_code == expected_exit, case_id
        if error_hint:
            assert error_hint in result.output.lower(), case_id

    @pytest.mark.parametrize(
        "case_id,args,expectation",
        SEARCH_FORMAT_CASES,
    )
    def test_output_contract(self, search_workspace, case_id, args, expectation):
        """Output format combinations produce parseable and mode-consistent output."""
        from polylogue.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", *args])
        assert result.exit_code == 0, case_id

        if expectation == "json_list":
            data = json.loads(result.output)
            assert isinstance(data, list), case_id
            assert data and "id" in data[0], case_id
        elif expectation == "json_single":
            data = json.loads(result.output)
            assert isinstance(data, (list, dict)), case_id
        elif expectation == "plain_list":
            assert result.output.strip(), case_id
        elif expectation == "markdown":
            assert "#" in result.output or "Rust" in result.output, case_id


class TestSearchEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_no_results(self, search_workspace):
        """Handle query with no matching results."""
        from polylogue.cli import cli
        runner = CliRunner()
        # Query mode with non-matching term
        result = runner.invoke(cli, ["--plain", "nonexistent_term_xyz"])
        # exit_code 2 = no results (valid outcome)
        assert result.exit_code == 2
        assert "no conversation" in result.output.lower() or "matched" in result.output.lower()

    def test_stats_mode_no_filters(self, cli_workspace, monkeypatch):
        """Stats mode when no query terms or filters provided."""
        from polylogue.cli import cli
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        runner = CliRunner()
        # No args = stats mode in query-first CLI
        result = runner.invoke(cli, ["--plain"])
        assert result.exit_code == 0
        # Should show stats, not require query

    def test_search_case_insensitive(self, search_workspace):
        """Search is case-insensitive."""
        from polylogue.cli import cli
        runner = CliRunner()
        # Query mode with --list to ensure consistent output
        result_lower = runner.invoke(cli, ["--plain", "python", "-f", "json", "--list"])
        result_upper = runner.invoke(cli, ["--plain", "PYTHON", "-f", "json", "--list"])

        # Both should have same exit code
        assert result_lower.exit_code == result_upper.exit_code

        if result_lower.exit_code == 0:
            # Both should find results (FTS5 is case-insensitive by default)
            data_lower = json.loads(result_lower.output)
            data_upper = json.loads(result_upper.output)
            assert len(data_lower) > 0
            assert len(data_upper) > 0

    def test_search_multiple_terms(self, search_workspace):
        """Search with multiple query terms."""
        from polylogue.cli import cli
        runner = CliRunner()
        # Query mode: multiple positional args = multiple query terms
        result = runner.invoke(cli, ["--plain", "Python", "exception", "-f", "json", "--list"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert isinstance(data, list)


class TestSearchIndexRebuild:
    """Tests for automatic index rebuild on missing index."""

    def test_search_handles_missing_index(self, cli_workspace, monkeypatch):
        """Search handles missing index gracefully."""
        from polylogue.cli import cli
        from tests.infra.storage_records import DbFactory
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Create conversation without building index
        db_path = cli_workspace["db_path"]
        factory = DbFactory(db_path)
        factory.create_conversation(
            id="c1",
            provider="test",
            title="Test",
            messages=[{"id": "m1", "role": "user", "text": "searchable content"}],
        )

        runner = CliRunner()
        # Query mode
        result = runner.invoke(cli, ["--plain", "searchable"])
        # Should either succeed (rebuild worked) or report no results.
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            assert "searchable" in result.output.lower() or "c1" in result.output
        else:
            assert "no conversation" in result.output.lower() or "matched" in result.output.lower()

"""Generalized contracts for CLI query execution helpers."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from rich.console import Console

from polylogue.cli.query import _async_execute_query, _no_results, _project_query_results
from polylogue.cli.query_actions import _apply_modifiers, _apply_transform, _delete_conversations
from polylogue.cli.query_helpers import summary_to_dict
from polylogue.cli.query_output import (
    _output_results,
    _output_stats_by_summaries,
    _output_stats_sql,
    _output_summary_list,
    _render_conversation_rich,
    _send_output,
)
from polylogue.cli.query_plan import (
    QueryAction,
    QueryExecutionPlan,
    QueryMutationSpec,
    QueryOutputSpec,
    QueryRoute,
    resolve_query_route,
)
from polylogue.cli.types import AppEnv
from polylogue.lib.models import Conversation, Message
from polylogue.lib.query_spec import ConversationQuerySpec, QuerySpecError
from polylogue.services import build_runtime_services
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


def _make_env(*, repo: MagicMock | None = None, config: MagicMock | None = None) -> AppEnv:
    ui = MagicMock()
    ui.plain = True
    ui.console = MagicMock()
    ui.confirm = MagicMock(return_value=True)
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
        provider="claude",
        title="Lawful Conversation",
        summary="summary",
        tags=("law",),
        created_at=None,
        updated_at=None,
        message_count=2,
    )


def _build_plan(case: str) -> QueryExecutionPlan:
    selection = MagicMock()
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
        provider="claude",
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
        provider="claude",
        title="Output Contract",
        messages=[
            Message(id=f"{conversation_id}-user", role="user", text="hello"),
            Message(id=f"{conversation_id}-assistant", role="assistant", text="world"),
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
        asyncio.run(_apply_modifiers(env, results, params, repo))

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
        asyncio.run(_delete_conversations(env, results, params, repo))

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
def test_output_summary_list_structured_formats_roundtrip(case) -> None:
    repo = MagicMock()
    repo.get_message_counts_batch = AsyncMock(return_value=build_message_counts(case.summaries))
    env = _make_env(repo=repo)
    summaries = [build_conversation_summary(spec) for spec in case.summaries]
    params = {"output_format": case.output_format, "fields": _fields_arg(case.selected_fields)}

    with patch("click.echo") as mock_echo:
        asyncio.run(_output_summary_list(env, summaries, params, repo))

    rendered = mock_echo.call_args[0][0]
    expected = _structured_rows(case)
    if case.output_format == "json":
        assert json.loads(rendered) == expected
    else:
        assert yaml.safe_load(rendered) == expected


@settings(max_examples=40, deadline=None)
@given(case=summary_output_case_strategy())
def test_output_summary_list_csv_roundtrip(case) -> None:
    repo = MagicMock()
    repo.get_message_counts_batch = AsyncMock(return_value=build_message_counts(case.summaries))
    env = _make_env(repo=repo)
    summaries = [build_conversation_summary(spec) for spec in case.summaries]

    with patch("click.echo") as mock_echo:
        asyncio.run(_output_summary_list(env, summaries, {"output_format": "csv"}, repo))

    rendered = mock_echo.call_args[0][0]
    parsed = list(csv.DictReader(io.StringIO(rendered)))
    assert parsed == _csv_rows(case)


@settings(max_examples=40, deadline=None)
@given(case=summary_output_case_strategy())
def test_output_summary_list_plaintext_emits_one_row_per_summary(case) -> None:
    repo = MagicMock()
    repo.get_message_counts_batch = AsyncMock(return_value=build_message_counts(case.summaries))
    env = _make_env(repo=repo)
    env.ui.plain = True
    summaries = [build_conversation_summary(spec) for spec in case.summaries]

    with patch("click.echo") as mock_echo:
        asyncio.run(_output_summary_list(env, summaries, {"output_format": "text"}, repo))

    assert mock_echo.call_count == len(case.summaries)
    rendered = "\n".join(call.args[0] for call in mock_echo.call_args_list if call.args)
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


@settings(max_examples=40, deadline=None)
@given(case=summary_stats_case_strategy())
def test_output_stats_by_summaries_contract(case) -> None:
    env, buffer = _make_recording_env()
    summaries = [build_conversation_summary(spec) for spec in case.summaries]
    msg_counts = build_message_counts(case.summaries)

    _output_stats_by_summaries(env, summaries, msg_counts, case.dimension)

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


def test_output_results_no_results_contract() -> None:
    env = _make_env()

    with (
        patch("polylogue.cli.query._no_results", side_effect=SystemExit(2)) as mock_no_results,
        patch("polylogue.cli.query_output._render_conversation_rich") as mock_render,
        patch("polylogue.cli.query_output._send_output") as mock_send,
        patch("polylogue.cli.query_output.format_conversation") as mock_format,
        pytest.raises(SystemExit) as exc_info,
    ):
        _output_results(env, [], {})

    assert exc_info.value.code == 2
    mock_no_results.assert_called_once_with(env, {})
    mock_render.assert_not_called()
    mock_send.assert_not_called()
    mock_format.assert_not_called()


def test_output_results_single_markdown_stdout_rich_contract() -> None:
    env = _make_env()
    env.ui.plain = False
    conversation = _sample_output_conversation()

    with (
        patch("polylogue.cli.query_output._render_conversation_rich") as mock_render,
        patch("polylogue.cli.query_output._send_output") as mock_send,
        patch("polylogue.cli.query_output.format_conversation") as mock_format,
    ):
        _output_results(
            env,
            [conversation],
            {"output_format": "markdown", "output": "stdout", "list_mode": False},
        )

    mock_render.assert_called_once_with(env, conversation)
    mock_send.assert_not_called()
    mock_format.assert_not_called()


def test_output_results_single_nonrich_contract() -> None:
    env = _make_env()
    conversation = _sample_output_conversation()

    with (
        patch("polylogue.cli.query_output.format_conversation", return_value="<html>ok</html>") as mock_format,
        patch("polylogue.cli.query_output._send_output") as mock_send,
    ):
        _output_results(
            env,
            [conversation],
            {"output_format": "html", "output": "stdout,browser", "fields": "id,title"},
        )

    mock_format.assert_called_once_with(conversation, "html", "id,title")
    mock_send.assert_called_once_with(
        env,
        "<html>ok</html>",
        ["stdout", "browser"],
        "html",
        conversation,
    )


def test_output_results_multi_list_contract() -> None:
    env = _make_env()
    conversations = [
        _sample_output_conversation("conv-output-1"),
        _sample_output_conversation("conv-output-2"),
    ]

    with (
        patch("polylogue.cli.query_output._format_list", return_value="formatted-list") as mock_format_list,
        patch("polylogue.cli.query_output._send_output") as mock_send,
    ):
        _output_results(
            env,
            conversations,
            {"output_format": "json", "output": "stdout", "fields": "id,provider"},
        )

    mock_format_list.assert_called_once_with(conversations, "json", "id,provider")
    mock_send.assert_called_once_with(env, "formatted-list", ["stdout"], "json", None)


def test_render_conversation_rich_contract() -> None:
    env, buffer = _make_recording_env()
    conversation = Conversation(
        id="conv-rich",
        provider="claude",
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
    assert "claude" in rendered
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

    projected = _project_query_results([conversation], plan)

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
        patch("polylogue.cli.query._output_summary_list", new_callable=AsyncMock) as mock_output_summary_list,
        patch("polylogue.cli.query._output_stats_sql", new_callable=AsyncMock) as mock_output_stats_sql,
        patch("polylogue.cli.query._output_stats_by_summaries") as mock_output_stats_by_summaries,
        patch("polylogue.cli.query._output_stats_by") as mock_output_stats_by,
        patch("polylogue.cli.query._apply_modifiers", new_callable=AsyncMock) as mock_apply_modifiers,
        patch("polylogue.cli.query._delete_conversations", new_callable=AsyncMock) as mock_delete_conversations,
        patch("polylogue.cli.query._open_result") as mock_open_result,
        patch("polylogue.cli.query._output_results") as mock_output_results,
        patch("polylogue.cli.query.resolve_stream_target", new_callable=AsyncMock, return_value="conv-stream") as mock_stream_target,
        patch("polylogue.cli.query.stream_conversation", new_callable=AsyncMock) as mock_stream_conversation,
    ):
        repo.get_message_counts_batch = AsyncMock(return_value={str(summary.id): 2})
        asyncio.run(_async_execute_query(env, {"limit": 7}))

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
        repo.get_message_counts_batch.assert_awaited_once_with([str(summary.id)])
        mock_output_stats_by_summaries.assert_called_once()
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
        patch("polylogue.cli.query._output_stats_by") as mock_output_stats_by,
        patch("polylogue.cli.query._output_stats_by_summaries") as mock_output_stats_by_summaries,
    ):
        asyncio.run(_async_execute_query(env, {}))

    filter_chain.list.assert_awaited_once()
    filter_chain.list_summaries.assert_not_called()
    mock_output_stats_by.assert_called_once()
    mock_output_stats_by_summaries.assert_not_called()


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
        patch("polylogue.cli.query._no_results", side_effect=SystemExit(2)) as mock_no_results,
        pytest.raises(SystemExit) as exc_info,
    ):
        asyncio.run(_async_execute_query(env, {}))

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
        asyncio.run(_async_execute_query(env, {}))

    assert exc_info.value.code == 1
    assert [call.args[0] for call in mock_echo.call_args_list if call.args] == [
        "Error: Cannot parse date: 'not-a-date'",
        "Hint: use ISO format (2025-01-15), relative ('yesterday', 'last week'), or month (2025-01)",
    ]
    assert all(call.kwargs.get("err") is True for call in mock_echo.call_args_list)


def test_async_execute_query_show_projects_results_before_output_contract() -> None:
    env = _make_env(repo=MagicMock(), config=MagicMock())
    selection = MagicMock()
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
        patch("polylogue.cli.query._output_results") as mock_output_results,
    ):
        asyncio.run(_async_execute_query(env, {}))

    projected_results = mock_output_results.call_args.args[1]
    assert [message.id for message in projected_results[0].messages] == ["m-user", "m-assistant"]


@pytest.mark.parametrize(
    ("params", "expected_lines"),
    [
        (
            {"provider": "claude", "limit": 5},
            [
                "No conversations matched filters:",
                "  provider: claude",
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
        _no_results(env, params)

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

    transformed = _apply_transform([conversation], transform)

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
    repo.aggregate_message_stats = AsyncMock(
        return_value={
            "total": 9,
            "user": 4,
            "assistant": 5,
            "words_approx": 42,
            "attachments": 2,
            "min_sort_key": 1704067200,
            "max_sort_key": 1704153600,
            "providers": {"claude": 2, "chatgpt": 1},
        }
    )
    summary_specs = (
        ConversationSummarySpec(
            conversation_id="conv-a",
            provider="claude",
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
    filter_chain.describe.return_value = ["provider=claude", "tag=law"]
    filter_chain.can_use_summaries.return_value = True
    filter_chain.list_summaries = AsyncMock(return_value=summaries)

    await _output_stats_sql(env, filter_chain, repo)

    filter_chain.list_summaries.assert_awaited_once()
    filter_chain.count.assert_not_called()
    repo.aggregate_message_stats.assert_awaited_once_with(["conv-a", "conv-b"])
    printed = [call.args[0] for call in env.ui.console.print.call_args_list if call.args]
    assert printed == [
        "\nConversations: 2\n",
        "Messages: 9 total (4 user, 5 assistant)",
        "Words: ~42",
        "Providers: claude (2), chatgpt (1)",
        "Attachments: 2",
        "Date range: 2024-01-01 to 2024-01-02",
    ]


@pytest.mark.asyncio
async def test_output_stats_sql_empty_paths_contract() -> None:
    env = _make_env()
    repo = MagicMock()
    repo.aggregate_message_stats = AsyncMock()

    filtered = MagicMock()
    filtered.describe.return_value = ["provider=claude"]
    filtered.can_use_summaries.return_value = True
    filtered.list_summaries = AsyncMock(return_value=[])
    await _output_stats_sql(env, filtered, repo)
    env.ui.console.print.assert_called_once_with("No conversations matched.")
    repo.aggregate_message_stats.assert_not_called()

    env = _make_env()
    unfiltered = MagicMock()
    unfiltered.describe.return_value = []
    unfiltered.count = AsyncMock(return_value=0)
    unfiltered.can_use_summaries.return_value = False
    await _output_stats_sql(env, unfiltered, repo)
    env.ui.console.print.assert_called_once_with("No conversations in archive.")
    repo.aggregate_message_stats.assert_not_called()

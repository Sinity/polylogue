from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.action_event.action_events import ActionEvent
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.archive.viewport.enums import ToolCategory
from polylogue.cli import query_output, query_semantic, query_stats
from polylogue.cli.query_actions import apply_modifiers, apply_transform, delete_conversations, resolve_stream_target
from polylogue.cli.query_contracts import QueryDeliveryTarget, QueryMutationSpec, QueryOutputSpec
from polylogue.cli.shared.types import AppEnv
from polylogue.lib.models import ConversationSummary
from polylogue.lib.roles import Role
from polylogue.lib.search_hits import ConversationSearchHit
from polylogue.storage.action_events.artifacts import ActionEventArtifactState
from polylogue.types import ConversationId, Provider
from tests.infra.builders import make_conv, make_msg


def _env(*, plain: bool = True) -> AppEnv:
    ui = MagicMock()
    ui.plain = plain
    ui.console = MagicMock()
    ui.confirm = MagicMock(return_value=False)
    return AppEnv(ui=ui)


def _output_spec(output_format: str = "markdown") -> QueryOutputSpec:
    return QueryOutputSpec(
        output_format=output_format,
        destinations=(QueryDeliveryTarget.parse("stdout"),),
        fields=None,
        dialogue_only=False,
        message_roles=(),
        transform=None,
        list_mode=False,
        print_path=False,
    )


def _mutation(*, dry_run: bool = False, force: bool = False, add_tags: tuple[str, ...] = ()) -> QueryMutationSpec:
    return QueryMutationSpec(
        set_meta=(("priority", "high"),),
        add_tags=add_tags,
        delete_matched=False,
        dry_run=dry_run,
        force=force,
    )


def _summary(
    *,
    conversation_id: str = "conv-1",
    provider: Provider = Provider.CLAUDE_CODE,
    title: str = "Archive Retry",
    updated_at: datetime | None = None,
) -> ConversationSummary:
    return ConversationSummary(
        id=ConversationId(conversation_id),
        provider=provider,
        title=title,
        updated_at=updated_at or datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc),
        message_count=2,
    )


def _action(
    *,
    kind: ToolCategory = ToolCategory.SHELL,
    tool_name: str = "read_file",
    search_text: str = "shell read_file /tmp/demo.py",
    affected_paths: tuple[str, ...] = ("/tmp/demo.py",),
) -> ActionEvent:
    return ActionEvent(
        event_id="event-1",
        message_id="message-1",
        timestamp=datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc),
        sequence_index=0,
        kind=kind,
        tool_name=tool_name,
        tool_id=None,
        provider=Provider.CLAUDE_CODE,
        affected_paths=affected_paths,
        cwd_path="/tmp",
        branch_names=(),
        command="cat demo.py",
        query=None,
        url=None,
        output_text="output",
        search_text=search_text,
        raw={},
    )


@pytest.mark.asyncio
async def test_query_actions_cover_error_branches_and_abort_paths() -> None:
    repo = SimpleNamespace(resolve_id=AsyncMock(side_effect=[None, None, "conv-123"]))
    filter_chain = MagicMock()
    filter_chain.list_summaries = AsyncMock(return_value=[])
    filter_chain.sort.return_value.limit.return_value.list_summaries = AsyncMock(return_value=[])

    with patch("click.echo") as echo:
        with pytest.raises(SystemExit) as missing_id:
            await resolve_stream_target(repo, filter_chain, ConversationQuerySpec(conversation_id="missing"))
        with pytest.raises(SystemExit) as latest_missing:
            await resolve_stream_target(repo, filter_chain, ConversationQuerySpec(latest=True))
        with pytest.raises(SystemExit) as filtered_missing:
            await resolve_stream_target(
                repo,
                filter_chain,
                ConversationQuerySpec(providers=(Provider.CLAUDE_CODE,)),
            )
        query_selection = cast(
            ConversationQuerySpec,
            SimpleNamespace(
                conversation_id=None,
                latest=False,
                query_terms=("missing",),
                has_filters=lambda: False,
            ),
        )
        with pytest.raises(SystemExit) as query_missing:
            await resolve_stream_target(repo, filter_chain, query_selection)
        with pytest.raises(SystemExit) as missing_selection:
            await resolve_stream_target(repo, filter_chain, ConversationQuerySpec())

    assert missing_id.value.code == 2
    assert latest_missing.value.code == 2
    assert filtered_missing.value.code == 2
    assert query_missing.value.code == 2
    assert missing_selection.value.code == 1
    echoed = [call.args[0] for call in echo.call_args_list if call.args]
    assert "No conversations matched." in echoed
    assert "No conversations matched filters." in echoed
    assert any("Hint: use" in line for line in echoed)

    env = _env()
    await apply_modifiers(env, [], _mutation())
    console_print = cast(MagicMock, env.ui.console.print)
    assert console_print.call_args_list[0].args[0] == "No conversations matched."

    env = _env()
    convs = [make_conv(id=f"conv-{index}", provider="claude-code") for index in range(12)]
    await apply_modifiers(env, convs, _mutation(force=False, add_tags=("review",)))
    console_print = cast(MagicMock, env.ui.console.print)
    assert console_print.call_args_list[-1].args[0] == "Aborted."

    env = _env()
    await delete_conversations(env, [], _mutation())
    console_print = cast(MagicMock, env.ui.console.print)
    assert console_print.call_args_list[0].args[0] == "No conversations matched."

    env = _env()
    same_day = [
        make_conv(
            id=f"conv-{index}",
            provider="claude-code",
            updated_at=datetime(2026, 4, 23, 10, index, tzinfo=timezone.utc),
        )
        for index in range(2)
    ]
    await delete_conversations(env, same_day, _mutation(force=False))
    console_print = cast(MagicMock, env.ui.console.print)
    assert console_print.call_args_list[-1].args[0] == "Aborted."

    success_selection = cast(
        ConversationQuerySpec,
        SimpleNamespace(
            conversation_id=None,
            latest=False,
            query_terms=("conv-123",),
            has_filters=lambda: False,
        ),
    )
    success = await resolve_stream_target(
        repo,
        filter_chain,
        success_selection,
    )
    assert success == "conv-123"

    env = _env()
    confirm = cast(MagicMock, env.ui.confirm)
    confirm.return_value = False
    many_conversations = [make_conv(id=f"conv-{index}", provider="claude-code") for index in range(11)]
    await delete_conversations(env, many_conversations, _mutation(force=False))
    console_print = cast(MagicMock, env.ui.console.print)
    assert console_print.call_args_list[-1].args[0] == "Aborted."


def test_apply_transform_covers_all_supported_variants() -> None:
    conversation = make_conv(
        id="conv-tools",
        provider="claude-code",
        messages=[
            make_msg(role="assistant", text="Answer"),
            make_msg(role="assistant", text="Thinking", is_thinking=True),
        ],
    )

    assert len(apply_transform([conversation], "strip-tools")) == 1
    assert len(apply_transform([conversation], "strip-thinking")) == 1
    assert len(apply_transform([conversation], "strip-all")) == 1


def test_query_semantic_matches_none_blocked_text_and_referenced_path() -> None:
    action = _action()

    assert (
        query_semantic.action_matches_slice(action, query_semantic.SemanticStatsSlice(action_terms=("none",))) is False
    )
    assert (
        query_semantic.action_matches_slice(
            action,
            query_semantic.SemanticStatsSlice(excluded_action_terms=("shell",)),
        )
        is False
    )
    assert (
        query_semantic.action_matches_slice(
            action,
            query_semantic.SemanticStatsSlice(action_text_terms=("missing",)),
        )
        is False
    )
    assert (
        query_semantic.action_matches_slice(
            action,
            query_semantic.SemanticStatsSlice(tool_terms=("none",)),
        )
        is False
    )
    assert query_semantic.referenced_path_matches_slice(action, ("demo.py",)) is True
    assert query_semantic.referenced_path_matches_slice(action, ("missing.py",)) is False


@pytest.mark.asyncio
async def test_query_semantic_helpers_cover_hydration_empty_and_plain_rendering() -> None:
    assert (
        await query_semantic._load_semantic_stats_conversations(
            SimpleNamespace(
                get_conversations_batch=AsyncMock(return_value=[]),
                get_messages_batch=AsyncMock(return_value={}),
                get_attachments_batch=lambda ids: {},
            ),
            [],
        )
        == []
    )

    with patch(
        "polylogue.storage.hydrators.conversation_from_records",
        side_effect=lambda record, messages, attachments: make_conv(id=str(record.conversation_id), messages=[]),
    ):
        conversations = await query_semantic._load_semantic_stats_conversations(
            SimpleNamespace(
                get_conversations_batch=AsyncMock(return_value=[SimpleNamespace(conversation_id="conv-1")]),
                get_messages_batch=AsyncMock(return_value={"conv-1": []}),
                get_attachments_batch=lambda ids: {},
            ),
            ["conv-1", "missing"],
        )

    assert [str(conv.id) for conv in conversations] == ["conv-1"]

    env = _env()
    repo = SimpleNamespace(
        get_action_event_artifact_state=AsyncMock(
            return_value=ActionEventArtifactState(
                source_conversations=2,
                materialized_conversations=2,
                materialized_rows=2,
                fts_rows=2,
            )
        ),
        get_action_events_batch=AsyncMock(return_value={"conv-1": (_action(),), "conv-2": ()}),
    )
    await query_semantic.output_stats_by_semantic_ids(env, ["conv-1", "conv-2"], repo, "action")

    console_print = cast(MagicMock, env.ui.console.print)
    printed = [call.args[0] for call in console_print.call_args_list if call.args]
    assert printed[0] == "\nMatched: 2 conversations (by action)\n"
    assert "multiple action groups" in printed[-1]

    with patch("polylogue.cli.query_semantic.emit_no_results") as emit_no_results:
        await query_semantic.output_stats_by_semantic_ids(env, [], repo, "action")
    emit_no_results.assert_called_once()

    with patch("polylogue.cli.query_semantic.output_stats_by_semantic_ids", new_callable=AsyncMock) as output_ids:
        await query_semantic.output_stats_by_semantic_query(env, ["conv-1"], repo, "action")
    output_ids.assert_awaited_once()

    with pytest.raises(ValueError, match="Unsupported semantic stats dimension"):
        await query_semantic.output_stats_by_semantic_ids(env, ["conv-1"], repo, "provider")


@pytest.mark.asyncio
async def test_query_stats_helpers_cover_structured_sql_and_profile_paths() -> None:
    with patch("click.echo") as echo:
        assert (
            query_stats.emit_structured_stats(
                output_format="yaml",
                dimension="provider",
                rows=[{"group": "claude-code", "conversations": 1, "messages": 2}],
                summary={"group": "TOTAL", "conversations": 1, "messages": 2},
            )
            is True
        )
        assert (
            query_stats.emit_structured_stats(
                output_format="csv",
                dimension="provider",
                rows=[{"group": "claude-code", "conversations": 1, "messages": 2}],
                summary={"group": "TOTAL", "conversations": 1, "messages": 2},
            )
            is True
        )
    assert echo.call_count == 2
    with pytest.raises(TypeError, match="must be int"):
        query_stats._count_value({"messages": "two"}, "messages")

    env = _env()
    filter_chain = SimpleNamespace(
        describe=lambda: ["provider=claude-code"],
        can_use_summaries=lambda: False,
        count=AsyncMock(return_value=0),
    )
    repo = SimpleNamespace()
    with patch("polylogue.cli.query_stats.emit_no_results") as emit_no_results:
        await query_stats.output_stats_sql(env, cast(Any, filter_chain), repo, output_format="json")
    emit_no_results.assert_called_once()

    env = _env()
    archive_filter = SimpleNamespace(describe=lambda: [], can_use_summaries=lambda: False)
    archive_repo = SimpleNamespace(
        get_archive_stats=AsyncMock(
            return_value=SimpleNamespace(
                total_conversations=4,
                embedded_conversations=3,
                embedded_messages=9,
                embedding_coverage=75.0,
                pending_embedding_conversations=1,
                stale_embedding_messages=2,
                messages_missing_embedding_provenance=1,
                embedding_readiness_status="pending",
                retrieval_ready=False,
            )
        ),
        aggregate_message_stats=AsyncMock(
            return_value={
                "total": 12,
                "user": 5,
                "assistant": 7,
                "words_approx": 123,
                "attachment_refs": 2,
                "distinct_attachments": 1,
                "providers": {"claude-code": 4},
                "min_sort_key": 1713830400,
                "max_sort_key": 1713916800,
            }
        ),
    )
    await query_stats.output_stats_sql(env, cast(Any, archive_filter), archive_repo, output_format="text")
    console_print = cast(MagicMock, env.ui.console.print)
    sql_lines = [call.args[0] for call in console_print.call_args_list if call.args]
    assert any("Embeddings: 3/4 convs, 9 msgs (75.0%), pending 1, stale 2" in line for line in sql_lines)
    assert any("Date range: 2024-04-23 to 2024-04-24" in line for line in sql_lines)

    env = _env()
    profile_repo = SimpleNamespace(
        get_session_profiles_batch=AsyncMock(return_value={}),
        get_many=AsyncMock(return_value=[make_conv(id="conv-1", provider="claude-code")]),
    )
    profile = SimpleNamespace(
        repo_names=("repo-a", "repo-b"), auto_tags=("kind:implementation",), work_events=[1], message_count=3
    )
    with (
        patch("polylogue.archive.session.session_profile.build_session_profile", return_value=profile),
        patch("polylogue.cli.query_stats._emit_grouped_stats_table") as emit_table,
        patch("polylogue.cli.query_stats.emit_no_results"),
    ):
        await query_stats.output_stats_by_profile_ids(env, ["conv-1"], profile_repo, "repo")
        await query_stats.output_stats_by_profile_ids(env, ["conv-1"], profile_repo, "work-kind")
        await query_stats.output_stats_by_profile_summaries(env, [_summary()], profile_repo, "repo")
        await query_stats.output_stats_by_profile_query(env, ["conv-1"], profile_repo, "work-kind")

    repo_call = emit_table.call_args_list[0].kwargs
    work_kind_call = emit_table.call_args_list[1].kwargs
    assert repo_call["multi_membership"] is True
    assert repo_call["note"] == "Note: conversations may appear in multiple repo groups."
    assert work_kind_call["multi_membership"] is False
    assert work_kind_call["note"] is None

    with pytest.raises(ValueError, match="Unsupported profile stats dimension"):
        await query_stats.output_stats_by_profile_ids(env, ["conv-1"], profile_repo, "provider")


@pytest.mark.asyncio
async def test_query_output_helpers_cover_stream_dates_headers_and_rich_lists(tmp_path: Path) -> None:
    class _DateLike:
        def strftime(self, fmt: str) -> str:
            assert fmt == "%Y-%m-%d %H:%M"
            return "2026-04-23 12:00"

        def __str__(self) -> str:
            return "2026-04-23 12:00"

    date_text, date_value = query_output._stream_date_parts(_DateLike())
    assert (date_text, date_value) == ("2026-04-23 12:00", "2026-04-23 12:00")

    message = make_msg(
        id="message-1",
        role=Role.USER,
        text="hello",
        timestamp=datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc),
    )
    assert query_output.render_stream_message(make_msg(text=None), "plaintext") == ""
    assert query_output.render_stream_message(make_msg(text=None), "markdown") == ""
    payload = json.loads(query_output.render_stream_message(message, "json-lines"))
    assert payload["timestamp"] == "2026-04-23T12:00:00+00:00"

    header = query_output.render_stream_header(
        conversation_id="conv-1",
        title="Archive Retry",
        provider="claude-code",
        display_date=_DateLike(),
        output_format="markdown",
        dialogue_only=False,
        message_roles=(Role.USER, Role.SYSTEM),
        message_limit=3,
        stats={"total_messages": 4, "role_user_messages": 1, "role_system_messages": 1},
    )
    assert "_Showing 2 selected-role messages (limit: 3) of 4 total_" in header

    env = _env(plain=False)
    repo = SimpleNamespace(get_message_counts_batch=AsyncMock(return_value={"conv-1": 2}))
    hit = ConversationSearchHit(
        summary=_summary(),
        rank=1,
        retrieval_lane="fts",
        match_surface="message",
        snippet="lock retry",
        message_id="message-1",
    )
    await query_output.output_search_hits(env, [hit], _output_spec("markdown"), repo=repo)

    env = _env(plain=False)
    repo = SimpleNamespace(get_message_counts_batch=AsyncMock(return_value={"conv-1": 2}))
    await query_output.output_summary_list(env, [_summary()], _output_spec("markdown"), repo=repo)

    env = _env(plain=True)
    plain_repo = SimpleNamespace(get_message_counts_batch=AsyncMock(return_value={"conv-1": 2}))
    await query_output.output_search_hits(env, [hit], _output_spec("markdown"), repo=plain_repo)
    await query_output.output_summary_list(env, [_summary()], _output_spec("markdown"), repo=None)

    with (
        patch("polylogue.cli.shared.helpers.load_effective_config", side_effect=RuntimeError("boom")),
        patch("polylogue.paths.render_root", return_value=Path("/tmp/missing-render-root")),
        patch("click.echo") as echo,
    ):
        with pytest.raises(SystemExit) as missing_root:
            query_output.open_result(_env(), [make_conv(id="conv-open", provider="claude-code")], _output_spec("html"))
    assert missing_root.value.code == 1
    assert any("No rendered outputs found." in call.args[0] for call in echo.call_args_list if call.args)

    render_root = tmp_path / "render-root"
    render_root.mkdir()
    with (
        patch(
            "polylogue.cli.shared.helpers.load_effective_config", return_value=SimpleNamespace(render_root=render_root)
        ),
        patch("polylogue.paths.render_root", return_value=render_root),
        patch(
            "polylogue.paths.sanitize.conversation_render_root",
            return_value=Path("/tmp/render-root/claude-code/conv-open"),
        ),
        patch("polylogue.cli.shared.helpers.latest_render_path", return_value=None),
        patch("click.echo") as echo,
    ):
        with pytest.raises(SystemExit) as missing_render:
            query_output.open_result(_env(), [make_conv(id="conv-open", provider="claude-code")], _output_spec("html"))
    assert missing_render.value.code == 1
    assert any(
        "No rendered output found for this conversation." in call.args[0] for call in echo.call_args_list if call.args
    )

    with patch("sys.stdout.write") as write, patch("sys.stdout.flush") as flush:
        query_output.write_message_streaming(make_msg(text=None), "plaintext")
        assert query_output.render_stream_message(message, "unknown") == ""
    write.assert_not_called()
    flush.assert_not_called()

"""Concrete CLI query side-effect tests.

Routing/output algebra lives in ``test_query_exec_laws.py``. This file keeps
only the concrete seams where direct example-driven side-effect tests remain the
clearest specification.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.cli.query import QueryAction, QueryRoute
from polylogue.cli.types import AppEnv
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.services import build_runtime_services

pytestmark = pytest.mark.query_routing


def _make_msg(id: str, role: str, text: str, *, timestamp=None, provider_meta=None) -> Message:
    return Message(id=id, role=role, text=text, timestamp=timestamp, provider_meta=provider_meta)


def _make_conv(
    id: str = "conv-1",
    provider: str = "claude-ai",
    title: str = "Test Conversation",
    messages: list[Message] | None = None,
) -> Conversation:
    return Conversation(
        id=id,
        provider=provider,
        title=title,
        messages=MessageCollection(messages=messages or [_make_msg("m1", "user", "Hello"), _make_msg("m2", "assistant", "Hi")]),
    )


def _make_summary(id: str = "conv-1") -> ConversationSummary:
    return ConversationSummary(
        id=id,
        provider="claude-ai",
        title="Test",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        metadata={"tags": [], "summary": "A test conversation"},
    )


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


def _make_params(**overrides) -> dict:
    params = {
        "conv_id": None,
        "query": (),
        "contains": (),
        "exclude_text": (),
        "retrieval_lane": None,
        "provider": None,
        "exclude_provider": None,
        "tag": None,
        "exclude_tag": None,
        "title": None,
        "path_terms": (),
        "action": (),
        "exclude_action": (),
        "action_sequence": None,
        "action_text": (),
        "similar_text": None,
        "has_type": (),
        "since": None,
        "until": None,
        "latest": False,
        "sort": None,
        "reverse": False,
        "limit": None,
        "sample": None,
        "count_only": False,
        "list_mode": False,
        "stream": False,
        "output_format": None,
        "output": None,
        "transform": None,
        "dialogue_only": False,
        "stats_only": False,
        "stats_by": None,
        "set_meta": None,
        "add_tag": None,
        "delete_matched": False,
        "force": False,
        "dry_run": False,
        "open_result": False,
        "fields": None,
    }
    params.update(overrides)
    return params


@pytest.mark.parametrize(
    ("param_overrides", "resolved_id", "expect_exit", "expect_warning"),
    [
        ({"latest": True}, "latest-conv-id", None, False),
        ({"conv_id": "abc"}, "full-conv-id-12345", None, False),
        ({"latest": True, "transform": "strip-tools"}, "latest-conv-id", None, True),
        ({}, None, 1, False),
    ],
    ids=["latest", "conv-id", "warning", "missing-target"],
)
def test_execute_query_stream_target_resolution_contract(param_overrides, resolved_id, expect_exit, expect_warning) -> None:
    from polylogue.cli.query import execute_query

    mock_repo = MagicMock()
    mock_filter = MagicMock()
    mock_repo.resolve_id = AsyncMock(return_value="full-conv-id-12345")
    mock_filter.list_summaries = AsyncMock(return_value=[_make_summary("latest-conv-id")])

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.lib.filters.ConversationFilter", return_value=mock_filter),
        patch("polylogue.cli.query_output.stream_conversation", new_callable=AsyncMock) as mock_stream,
        patch("click.echo") as mock_echo,
    ):
        env = _make_env(repo=mock_repo, config=MagicMock())
        params = _make_params(stream=True, **param_overrides)

        if expect_exit is not None:
            with pytest.raises(SystemExit) as exc_info:
                execute_query(env, params)
            assert exc_info.value.code == expect_exit
            mock_stream.assert_not_called()
        else:
            execute_query(env, params)
            mock_stream.assert_called_once()
            assert mock_stream.call_args.args[2] == resolved_id
            warnings = [call.args[0] for call in mock_echo.call_args_list if call.args and "Warning" in call.args[0]]
            assert bool(warnings) is expect_warning


@pytest.mark.asyncio
async def test_async_execute_query_errors_for_similar_without_vector_support() -> None:
    from polylogue.cli.query import async_execute_query

    env = _make_env(repo=MagicMock(), config=MagicMock())

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("click.echo") as mock_echo,
    ):
        with pytest.raises(SystemExit) as exc_info:
            await async_execute_query(env, _make_params(similar_text="sqlite locking regression"))

    assert exc_info.value.code == 1
    mock_echo.assert_called_once()
    assert "requires vector search support" in mock_echo.call_args.args[0]


@pytest.mark.asyncio
async def test_async_execute_query_errors_for_similar_without_embeddings() -> None:
    from polylogue.cli.query import async_execute_query

    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(
        return_value=SimpleNamespace(embedded_messages=0, embedded_conversations=0)
    )
    env = _make_env(repo=repo, config=MagicMock())

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
        patch("click.echo") as mock_echo,
    ):
        with pytest.raises(SystemExit) as exc_info:
            await async_execute_query(env, _make_params(similar_text="sqlite locking regression"))

    assert exc_info.value.code == 1
    mock_echo.assert_called_once()
    assert "requires existing embeddings" in mock_echo.call_args.args[0]


@pytest.mark.asyncio
async def test_async_execute_query_reports_non_date_query_spec_errors() -> None:
    from polylogue.cli.query import (
        QueryAction,
        QueryExecutionPlan,
        QueryMutationSpec,
        QueryOutputSpec,
        async_execute_query,
    )
    from polylogue.lib.query_spec import QuerySpecError

    env = _make_env(repo=MagicMock(), config=MagicMock())
    selection = MagicMock()
    selection.similar_text = None
    selection.build_filter.side_effect = QuerySpecError("action", "bogus")
    plan = QueryExecutionPlan(
        selection=selection,
        action=QueryAction.SHOW,
        output=QueryOutputSpec("markdown", ("stdout",), None, False, None, False),
        mutation=QueryMutationSpec((), (), False, False, False),
    )

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("click.echo") as mock_echo,
    ):
        with pytest.raises(SystemExit) as exc_info:
            await async_execute_query(env, _make_params(action=("bogus",)))

    assert exc_info.value.code == 1
    mock_echo.assert_called_once_with("Error: invalid action: 'bogus'", err=True)


@pytest.mark.asyncio
async def test_query_plan_filters_ordered_action_sequence() -> None:
    from polylogue.lib.query_spec import ConversationQuerySpec

    matching = Conversation(
        id="conv-sequence-match",
        provider="claude-code",
        title="Matching sequence",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Read", "tool_input": {"file_path": "/tmp/a.py"}, "semantic_type": "file_read"},
                    ],
                ),
                Message(
                    id="m2",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Edit", "tool_input": {"file_path": "/tmp/a.py"}, "semantic_type": "file_edit"},
                    ],
                ),
                Message(
                    id="m3",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Bash", "tool_input": {"command": "pytest -q"}, "semantic_type": "shell"},
                    ],
                ),
            ]
        ),
    )
    non_matching = Conversation(
        id="conv-sequence-miss",
        provider="claude-code",
        title="Non matching sequence",
        messages=MessageCollection(
            messages=[
                Message(
                    id="x1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Bash", "tool_input": {"command": "pytest -q"}, "semantic_type": "shell"},
                    ],
                ),
                Message(
                    id="x2",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Edit", "tool_input": {"file_path": "/tmp/a.py"}, "semantic_type": "file_edit"},
                    ],
                ),
            ]
        ),
    )

    repo = MagicMock()
    repo.list_by_query = AsyncMock(return_value=[matching, non_matching])
    plan = ConversationQuerySpec(action_sequence=("file_read", "file_edit", "shell")).to_plan()

    results = await plan.list(repo)

    assert [conversation.id for conversation in results] == ["conv-sequence-match"]


@pytest.mark.asyncio
async def test_query_plan_filters_action_text_terms() -> None:
    from polylogue.lib.query_spec import ConversationQuerySpec

    matching = Conversation(
        id="conv-action-text-match",
        provider="claude-code",
        title="Action text match",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Bash", "tool_input": {"command": "pytest -q tests/unit/core/test_semantic_facts.py"}, "semantic_type": "shell"},
                    ],
                ),
            ]
        ),
    )
    non_matching = Conversation(
        id="conv-action-text-miss",
        provider="claude-code",
        title="Action text miss",
        messages=MessageCollection(
            messages=[
                Message(
                    id="x1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Bash", "tool_input": {"command": "ruff check polylogue/lib/action_events.py"}, "semantic_type": "shell"},
                    ],
                ),
            ]
        ),
    )

    repo = MagicMock()
    repo.list_by_query = AsyncMock(return_value=[matching, non_matching])
    plan = ConversationQuerySpec(action_text_terms=("pytest -q", "semantic_facts.py")).to_plan()

    results = await plan.list(repo)

    assert [conversation.id for conversation in results] == ["conv-action-text-match"]


@pytest.mark.asyncio
async def test_query_plan_batches_post_filter_candidate_fetches() -> None:
    from polylogue.lib.query_spec import ConversationQuerySpec

    matching = Conversation(
        id="conv-batched-match",
        provider="claude-code",
        title="Batched match",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Read", "tool_input": {"file_path": "/tmp/a.py"}, "semantic_type": "file_read"},
                        {"type": "tool_use", "tool_name": "Edit", "tool_input": {"file_path": "/tmp/a.py"}, "semantic_type": "file_edit"},
                    ],
                ),
            ]
        ),
    )

    repo = MagicMock()
    repo.list_by_query = AsyncMock(return_value=[matching])
    plan = ConversationQuerySpec(action_sequence=("file_read", "file_edit"), limit=50).to_plan()

    results = await plan.list(repo)

    assert [conversation.id for conversation in results] == ["conv-batched-match"]
    request = repo.list_by_query.await_args.args[0]
    assert request.limit == 100
    assert request.offset == 0


@pytest.mark.asyncio
async def test_query_plan_action_retrieval_lane_matches_tool_command_text() -> None:
    from polylogue.lib.query_spec import ConversationQuerySpec

    matching = Conversation(
        id="conv-actions-lane-match",
        provider="claude-code",
        title="Action lane match",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Bash", "tool_input": {"command": "pytest -q tests/unit/core/test_semantic_facts.py"}, "semantic_type": "shell"},
                    ],
                ),
            ]
        ),
    )
    repo = MagicMock()
    repo.search = AsyncMock(return_value=[])
    repo.search_actions = AsyncMock(return_value=[matching])
    plan = ConversationQuerySpec(query_terms=("pytest", "semantic_facts.py"), retrieval_lane="actions", limit=10).to_plan()

    results = await plan.list(repo)

    assert [conversation.id for conversation in results] == ["conv-actions-lane-match"]
    repo.search_actions.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_plan_action_retrieval_lane_falls_back_when_action_read_model_unready() -> None:
    from polylogue.lib.query_spec import ConversationQuerySpec

    matching = Conversation(
        id="conv-actions-lane-fallback",
        provider="claude-code",
        title="Action lane fallback",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Bash", "tool_input": {"command": "pytest -q tests/unit/core/test_semantic_facts.py"}, "semantic_type": "shell"},
                    ],
                ),
            ]
        ),
    )
    repo = MagicMock()
    repo.get_action_event_read_model_status = AsyncMock(return_value={"ready": False})
    repo.search = AsyncMock(return_value=[])
    repo.search_actions = AsyncMock(return_value=[])
    repo.list_by_query = AsyncMock(return_value=[matching])
    plan = ConversationQuerySpec(query_terms=("pytest", "semantic_facts.py"), retrieval_lane="actions", limit=10).to_plan()

    results = await plan.list(repo)

    assert [conversation.id for conversation in results] == ["conv-actions-lane-fallback"]
    repo.search_actions.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_plan_hybrid_retrieval_lane_combines_text_and_action_hits() -> None:
    from polylogue.lib.query_spec import ConversationQuerySpec

    text_hit = Conversation(
        id="conv-text-hit",
        provider="claude-code",
        title="Text hit",
        messages=MessageCollection(messages=[Message(id="t1", role="assistant", provider="claude-code", text="pytest failure in semantic facts test")]),
    )
    action_hit = Conversation(
        id="conv-action-hit",
        provider="claude-code",
        title="Action hit",
        messages=MessageCollection(
            messages=[
                Message(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Bash", "tool_input": {"command": "pytest -q tests/unit/core/test_semantic_facts.py"}, "semantic_type": "shell"},
                    ],
                ),
            ]
        ),
    )

    repo = MagicMock()
    repo.search = AsyncMock(return_value=[text_hit])
    repo.search_actions = AsyncMock(return_value=[action_hit])
    repo.search_similar = AsyncMock(return_value=[])
    plan = ConversationQuerySpec(query_terms=("pytest", "semantic_facts.py"), retrieval_lane="hybrid", limit=10).to_plan()

    results = await plan.list(repo)

    assert {conversation.id for conversation in results} == {"conv-text-hit", "conv-action-hit"}
    repo.search_actions.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_plan_path_filters_fall_back_to_full_list_when_action_read_model_unready() -> None:
    from polylogue.lib.query_spec import ConversationQuerySpec

    matching = Conversation(
        id="conv-summary-path-match",
        provider="claude-code",
        title="Summary path match",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Read", "tool_input": {"file_path": "/tmp/a.py"}, "semantic_type": "file_read"},
                    ],
                ),
            ]
        ),
    )
    non_matching = Conversation(
        id="conv-summary-path-miss",
        provider="claude-code",
        title="Summary path miss",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m2",
                    role="assistant",
                    provider="claude-code",
                    content_blocks=[
                        {"type": "tool_use", "tool_name": "Read", "tool_input": {"file_path": "/tmp/b.py"}, "semantic_type": "file_read"},
                    ],
                ),
            ]
        ),
    )

    repo = MagicMock()
    repo.get_action_event_read_model_status = AsyncMock(return_value={"ready": False})
    repo.list_by_query = AsyncMock(return_value=[matching, non_matching])
    plan = ConversationQuerySpec(path_terms=("/tmp/a.py",), limit=10).to_plan()

    results = await plan.list(repo)

    assert [conversation.id for conversation in results] == ["conv-summary-path-match"]


@pytest.mark.asyncio
async def test_async_execute_query_uses_action_event_stats_lane_for_semantic_stats() -> None:
    from polylogue.cli.query import async_execute_query

    repo = MagicMock()
    repo.get_action_event_read_model_status = AsyncMock(return_value={"ready": True})
    repo.queries.list_conversations = AsyncMock(return_value=[SimpleNamespace(conversation_id="conv-semantic-1")])
    env = _make_env(repo=repo, config=MagicMock())

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query_output.output_stats_by_semantic_query", new_callable=AsyncMock) as mock_output,
        patch("polylogue.cli.query_output._output_stats_by") as mock_fallback,
    ):
        await async_execute_query(
            env,
            _make_params(
                stats_by="tool",
                provider="claude-code",
                since="2026-01-01",
                action=("search",),
                limit=20,
            ),
        )

    repo.queries.list_conversations.assert_awaited_once()
    mock_output.assert_awaited_once()
    mock_fallback.assert_not_called()


@pytest.mark.asyncio
async def test_async_execute_query_uses_session_product_stats_lane_for_project_stats() -> None:
    from polylogue.cli.query import async_execute_query

    repo = MagicMock()
    env = _make_env(repo=repo, config=MagicMock())

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        patch("polylogue.cli.query_output.output_stats_by_profile_summaries", new_callable=AsyncMock) as mock_output,
        patch("polylogue.cli.query_output._output_stats_by") as mock_fallback,
        patch("polylogue.lib.filters.ConversationFilter.list_summaries", new=AsyncMock(return_value=[_make_summary("conv-semantic-1")])),
        patch("polylogue.lib.filters.ConversationFilter.can_use_summaries", return_value=True),
    ):
        await async_execute_query(
            env,
            _make_params(
                stats_by="project",
                provider="claude-code",
                since="2026-01-01",
                limit=20,
            ),
        )

    mock_output.assert_awaited_once()
    mock_fallback.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output_format", "dialogue_only", "limit", "expected_fragments"),
    [
        ("plaintext", False, None, ["[USER]", "Hello", "[ASSISTANT]", "Hi"]),
        ("markdown", True, 1, ["# Test Title", "_Showing 1 dialogue messages (limit: 1) of 2 total_", "## User", "---", "_Streamed 1 messages_"]),
        ("json-lines", False, None, ['"type": "header"', '"type": "message"', '"type": "footer"']),
    ],
    ids=["plaintext", "markdown", "json-lines"],
)
async def test_stream_conversation_output_contract(output_format: str, dialogue_only: bool, limit: int | None, expected_fragments: list[str]) -> None:
    from polylogue.cli.query_output import stream_conversation

    repo = MagicMock()
    repo.queries.get_conversation = AsyncMock(return_value=SimpleNamespace(title="Test Title"))
    repo.queries.get_conversation_stats = AsyncMock(return_value={"dialogue_messages": 1, "total_messages": 2})

    async def _iter_messages(*_args, **_kwargs):
        messages = [_make_msg("m1", "user", "Hello"), _make_msg("m2", "assistant", "Hi")]
        if limit is not None:
            messages[:] = messages[:limit]
        for message in messages:
            yield message

    repo.iter_messages = _iter_messages
    env = _make_env(repo=repo, config=MagicMock())
    stdout = StringIO()

    with patch("sys.stdout", stdout):
        count = await stream_conversation(env, repo, "conv-1", output_format=output_format, dialogue_only=dialogue_only, message_limit=limit)

    output = stdout.getvalue()
    assert count == (1 if limit == 1 else 2)
    for fragment in expected_fragments:
        assert fragment in output


@pytest.mark.asyncio
async def test_stream_conversation_errors_for_missing_conversation() -> None:
    from polylogue.cli.query_output import stream_conversation

    repo = MagicMock()
    repo.queries.get_conversation = AsyncMock(return_value=None)
    env = _make_env(repo=repo, config=MagicMock())

    with patch("click.echo") as mock_echo, pytest.raises(SystemExit) as exc_info:
        await stream_conversation(env, repo, "missing")

    assert exc_info.value.code == 1
    mock_echo.assert_called_once_with("Conversation not found: missing", err=True)


@pytest.mark.parametrize(
    ("content", "output_format", "conv", "expected_in_file"),
    [
        ("<html><body>Test</body></html>", "html", None, "<html><body>Test</body></html>"),
        ("Plain text", "markdown", None, "<pre>Plain text</pre>"),
        ("ignored", "markdown", _make_conv(), "Test Conversation"),
    ],
    ids=["html-pass-through", "wrapped-non-html", "conversation-html"],
)
def test_open_in_browser_contract(content: str, output_format: str, conv: Conversation | None, expected_in_file: str, tmp_path: Path) -> None:
    from polylogue.cli.query_output import _open_in_browser

    env = _make_env(config=MagicMock())
    created_file = tmp_path / "output.html"

    class _TempFile:
        name = str(created_file)
        def __enter__(self):
            self.handle = created_file.open("w", encoding="utf-8")
            return self.handle
        def __exit__(self, exc_type, exc, tb):
            self.handle.close()
            return False

    with (
        patch("tempfile.NamedTemporaryFile", return_value=_TempFile()),
        patch("webbrowser.open") as mock_open,
    ):
        _open_in_browser(env, content, output_format, conv)

    assert expected_in_file in created_file.read_text(encoding="utf-8")
    mock_open.assert_called_once()
    env.ui.console.print.assert_called_once()


@pytest.mark.parametrize(
    ("side_effects", "expect_console", "expect_echo"),
    [
        ([None], True, False),
        ([FileNotFoundError(), FileNotFoundError(), FileNotFoundError(), FileNotFoundError()], False, True),
    ],
    ids=["success", "failure"],
)
def test_copy_to_clipboard_contract(side_effects, expect_console: bool, expect_echo: bool) -> None:
    from polylogue.cli.query_output import _copy_to_clipboard

    env = _make_env(config=MagicMock())

    with patch("subprocess.run", side_effect=side_effects), patch("click.echo") as mock_echo:
        _copy_to_clipboard(env, "hello")

    assert env.ui.console.print.called is expect_console
    assert mock_echo.called is expect_echo


@pytest.mark.parametrize(
    ("results", "render_root_exists", "html_exists", "latest_exists", "expected_exit", "expected_open_name"),
    [
        ([], True, False, False, 2, None),
        ([_make_conv(id="conv-output-1234")], False, False, False, 1, None),
        ([_make_conv(id="conv-output-1234")], True, True, False, None, "conversation.html"),
        ([_make_conv(id="conv-output-1234")], True, False, True, None, "fallback.html"),
    ],
    ids=["no-results", "no-render-root", "specific-render", "latest-fallback"],
)
def test_open_result_contract(results, render_root_exists: bool, html_exists: bool, latest_exists: bool, expected_exit: int | None, expected_open_name: str | None, tmp_path: Path) -> None:
    from polylogue.cli.query_output import _open_result

    render_root = tmp_path / "rendered"
    if render_root_exists:
        render_root.mkdir()
    specific_dir = render_root / "conv-outp"
    if html_exists:
        specific_dir.mkdir(parents=True, exist_ok=True)
        (specific_dir / "conversation.html").write_text("<html></html>", encoding="utf-8")
    fallback = render_root / "fallback.html"
    if latest_exists:
        fallback.write_text("<html></html>", encoding="utf-8")

    env = _make_env(config=MagicMock(render_root=render_root))
    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock(render_root=render_root)),
        patch("polylogue.cli.helpers.latest_render_path", return_value=fallback if latest_exists else None),
        patch("webbrowser.open") as mock_open,
        patch("click.echo") as mock_echo,
    ):
        if expected_exit is not None:
            with pytest.raises(SystemExit) as exc_info:
                _open_result(env, results, {})
            assert exc_info.value.code == expected_exit
            mock_open.assert_not_called()
        else:
            _open_result(env, results, {})
            opened = mock_open.call_args.args[0]
            assert expected_open_name in opened
            env.ui.console.print.assert_called_once()
            mock_echo.assert_not_called()


# ---------------------------------------------------------------------------
# Merged from test_query.py (2026-03-15)
# ---------------------------------------------------------------------------

# Note: test_execute_query_stream_target_resolution_contract and related streaming
# tests from test_query.py are already in this file above (lines 101-140).
# The parametrize decorator was missing in the original test_query_exec.py and
# is now added via merge from test_query.py.


# ---------------------------------------------------------------------------
# Merged from test_query_plan.py (2026-03-15)
# ---------------------------------------------------------------------------


class TestBuildQueryExecutionPlan:
    def test_delete_without_filters_raises(self) -> None:
        from polylogue.cli.query import QueryPlanError, build_query_execution_plan
        with pytest.raises(QueryPlanError, match="delete requires at least one filter"):
            build_query_execution_plan({"delete_matched": True, "query": ()})

    @pytest.mark.parametrize(
        ("params", "expected_action"),
        [
            ({"count_only": True, "query": ()}, QueryAction.COUNT),
            ({"stream": True, "query": ("abc",)}, QueryAction.STREAM),
            ({"stats_only": True, "query": ()}, QueryAction.STATS),
            ({"stats_by": "provider", "query": ()}, QueryAction.STATS_BY),
            ({"stats_by": "action", "query": ()}, QueryAction.STATS_BY),
            ({"stats_by": "tool", "query": ()}, QueryAction.STATS_BY),
            ({"stats_by": "project", "query": ()}, QueryAction.STATS_BY),
            ({"stats_by": "work-kind", "query": ()}, QueryAction.STATS_BY),
            ({"add_tag": ["x"], "query": ()}, QueryAction.MODIFY),
            ({"delete_matched": True, "provider": "claude-ai", "query": ()}, QueryAction.DELETE),
            ({"open_result": True, "query": ("abc",)}, QueryAction.OPEN),
            ({"query": ("abc",)}, QueryAction.SHOW),
        ],
    )
    def test_action_selection(self, params: dict[str, object], expected_action: QueryAction) -> None:
        from polylogue.cli.query import build_query_execution_plan
        plan = build_query_execution_plan(params)
        assert plan.action == expected_action

    def test_stream_format_converts_json_to_json_lines(self) -> None:
        from polylogue.cli.query import build_query_execution_plan
        plan = build_query_execution_plan({"stream": True, "output_format": "json", "query": ("abc",)})
        assert plan.output.stream_format() == "json-lines"

    def test_summary_list_preference_requires_plain_listing_shape(self) -> None:
        from polylogue.cli.query import build_query_execution_plan
        plan = build_query_execution_plan({"list_mode": True, "query": ("abc",)})
        assert plan.prefers_summary_list() is True

        transformed = build_query_execution_plan({"list_mode": True, "transform": "strip-tools", "query": ("abc",)})
        assert transformed.prefers_summary_list() is False

    def test_mutation_fields_are_normalized(self) -> None:
        from polylogue.cli.query import build_query_execution_plan
        plan = build_query_execution_plan(
            {
                "set_meta": [("priority", 3)],
                "add_tag": ["todo", "review"],
                "force": True,
                "dry_run": True,
                "provider": "claude-ai",
                "query": (),
            }
        )
        assert plan.mutation.set_meta == (("priority", "3"),)
        assert plan.mutation.add_tags == ("todo", "review")
        assert plan.mutation.force is True
        assert plan.mutation.dry_run is True

    @pytest.mark.parametrize(
        ("params", "can_use_summaries", "expected_route"),
        [
            ({"count_only": True, "query": ()}, False, QueryRoute.COUNT),
            ({"list_mode": True, "query": ("abc",)}, True, QueryRoute.SUMMARY_LIST),
            ({"list_mode": True, "query": ("abc",)}, False, QueryRoute.SHOW),
            ({"stream": True, "query": ("abc",)}, False, QueryRoute.STREAM),
            ({"stats_only": True, "query": ()}, True, QueryRoute.STATS_SQL),
            ({"stats_by": "provider", "query": ()}, True, QueryRoute.SUMMARY_STATS),
            ({"stats_by": "provider", "query": ()}, False, QueryRoute.STATS_BY),
            ({"stats_by": "action", "query": ()}, True, QueryRoute.STATS_BY),
            ({"stats_by": "action", "query": ()}, False, QueryRoute.STATS_BY),
            ({"stats_by": "tool", "query": ()}, True, QueryRoute.STATS_BY),
            ({"stats_by": "tool", "query": ()}, False, QueryRoute.STATS_BY),
            ({"stats_by": "project", "query": ()}, True, QueryRoute.STATS_BY),
            ({"stats_by": "work-kind", "query": ()}, True, QueryRoute.STATS_BY),
            (
                {"set_meta": [("priority", "1")], "query": ("abc",)},
                True,
                QueryRoute.SUMMARY_MODIFY,
            ),
            (
                {"set_meta": [("priority", "1")], "query": ("abc",)},
                False,
                QueryRoute.MODIFY,
            ),
            (
                {"delete_matched": True, "provider": "claude-ai", "query": ()},
                True,
                QueryRoute.SUMMARY_DELETE,
            ),
            (
                {"delete_matched": True, "provider": "claude-ai", "query": ()},
                False,
                QueryRoute.DELETE,
            ),
            ({"open_result": True, "query": ("abc",)}, False, QueryRoute.OPEN),
        ],
    )
    def test_route_resolution(self, params: dict[str, object], can_use_summaries: bool, expected_route: QueryRoute) -> None:
        from polylogue.cli.query import build_query_execution_plan, resolve_query_route
        plan = build_query_execution_plan(params)
        assert resolve_query_route(plan, can_use_summaries=can_use_summaries) == expected_route

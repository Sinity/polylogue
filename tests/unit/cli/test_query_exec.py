"""Concrete CLI query side-effect tests.

Routing/output algebra lives in ``test_query_exec_laws.py``. This file keeps
only the concrete seams where direct example-driven side-effect tests remain the
clearest specification.
"""

from __future__ import annotations

import subprocess
import tempfile
import webbrowser
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest

from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.query_plan import QueryAction, QueryRoute
from polylogue.cli.types import AppEnv
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.services import build_runtime_services


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
    return AppEnv(ui=ui, services=build_runtime_services(config=config, repository=repo))


def _make_params(**overrides) -> dict:
    params = {
        "conv_id": None,
        "query": (),
        "contains": (),
        "exclude_text": (),
        "provider": None,
        "exclude_provider": None,
        "tag": None,
        "exclude_tag": None,
        "title": None,
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
    mock_filter.sort.return_value.limit.return_value.list_summaries = AsyncMock(return_value=[_make_summary("latest-conv-id")])

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
    repo.get_conversation = AsyncMock(return_value=SimpleNamespace(title="Test Title"))
    repo.get_conversation_stats = AsyncMock(return_value={"dialogue_messages": 1, "total_messages": 2})

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
    repo.get_conversation = AsyncMock(return_value=None)
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
        from polylogue.cli.query_plan import QueryPlanError, build_query_execution_plan
        with pytest.raises(QueryPlanError, match="--delete requires at least one filter"):
            build_query_execution_plan({"delete_matched": True, "query": ()})

    @pytest.mark.parametrize(
        ("params", "expected_action"),
        [
            ({"count_only": True, "query": ()}, QueryAction.COUNT),
            ({"stream": True, "query": ("abc",)}, QueryAction.STREAM),
            ({"stats_only": True, "query": ()}, QueryAction.STATS),
            ({"stats_by": "provider", "query": ()}, QueryAction.STATS_BY),
            ({"add_tag": ["x"], "query": ()}, QueryAction.MODIFY),
            ({"delete_matched": True, "provider": "claude-ai", "query": ()}, QueryAction.DELETE),
            ({"open_result": True, "query": ("abc",)}, QueryAction.OPEN),
            ({"query": ("abc",)}, QueryAction.SHOW),
        ],
    )
    def test_action_selection(self, params: dict[str, object], expected_action: QueryAction) -> None:
        from polylogue.cli.query_plan import build_query_execution_plan
        plan = build_query_execution_plan(params)
        assert plan.action == expected_action

    def test_stream_format_converts_json_to_json_lines(self) -> None:
        from polylogue.cli.query_plan import build_query_execution_plan
        plan = build_query_execution_plan({"stream": True, "output_format": "json", "query": ("abc",)})
        assert plan.output.stream_format() == "json-lines"

    def test_summary_list_preference_requires_plain_listing_shape(self) -> None:
        from polylogue.cli.query_plan import build_query_execution_plan
        plan = build_query_execution_plan({"list_mode": True, "query": ("abc",)})
        assert plan.prefers_summary_list() is True

        transformed = build_query_execution_plan({"list_mode": True, "transform": "strip-tools", "query": ("abc",)})
        assert transformed.prefers_summary_list() is False

    def test_mutation_fields_are_normalized(self) -> None:
        from polylogue.cli.query_plan import build_query_execution_plan
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
        from polylogue.cli.query_plan import build_query_execution_plan, resolve_query_route
        plan = build_query_execution_plan(params)
        assert resolve_query_route(plan, can_use_summaries=can_use_summaries) == expected_route

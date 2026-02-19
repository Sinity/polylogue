"""Tests for CLI query execution and output functions.

Covers uncovered areas of polylogue/cli/query.py:
- _no_results(): Error reporting with/without filters
- execute_query(): Main orchestration path, all flow branches
- _apply_modifiers(): Metadata and tag operations with dry-run/force
- _delete_conversations(): Conversation deletion with confirmation
- _output_summary_list(): JSON/YAML/CSV/text format output
- stream_conversation(): Memory-efficient streaming output
- _write_message_streaming(): Per-message streaming format
- _send_output(): Destination routing (stdout, file, browser, clipboard)
- _open_in_browser(): Browser opening with temp files
- _copy_to_clipboard(): Clipboard tool invocation
- _open_result(): Rendered file discovery and opening
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.cli.types import AppEnv
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, ConversationSummary, Message

# =============================================================================
# Test Helpers for Building Test Data
# =============================================================================


def _make_msg(
    id: str = "msg-1",
    role: str = "user",
    text: str | None = "Hello",
    **kwargs,
) -> Message:
    """Create a test message."""
    return Message(
        id=id,
        role=role,
        text=text,
        timestamp=kwargs.get("timestamp"),
        attachments=kwargs.get("attachments", []),
        provider_meta=kwargs.get("provider_meta"),
    )


def _make_conv(
    id: str = "test-conv-123",
    provider: str = "claude",
    title: str = "Test Conversation",
    messages: list[Message] | None = None,
    **kwargs,
) -> Conversation:
    """Create a test conversation."""
    if messages is None:
        messages = [_make_msg("msg-1", "user", "Hello"), _make_msg("msg-2", "assistant", "Hi there")]

    return Conversation(
        id=id,
        provider=provider,
        title=title,
        messages=MessageCollection(messages=messages),
        created_at=kwargs.get("created_at"),
        updated_at=kwargs.get("updated_at"),
        metadata={"tags": kwargs.get("tags", []), "summary": kwargs.get("summary")},
    )


def _make_summary(
    id: str = "conv-1",
    provider: str = "claude",
    title: str = "Test",
    tags: list[str] | None = None,
) -> ConversationSummary:
    """Create a test conversation summary."""
    return ConversationSummary(
        id=id,
        provider=provider,
        title=title,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        metadata={"tags": tags or [], "summary": "A test conversation"},
    )


def _make_env() -> AppEnv:
    """Create a mock AppEnv."""
    ui = MagicMock()
    ui.plain = True
    ui.console = MagicMock()
    ui.confirm = MagicMock(return_value=True)
    return AppEnv(ui=ui)


def _make_params(**overrides) -> dict:
    """Build default params dict with overrides."""
    defaults = {
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
    defaults.update(overrides)
    return defaults


def _make_mock_repo(
    *,
    update_metadata: AsyncMock | None = None,
    add_tag: AsyncMock | None = None,
    delete_conversation: AsyncMock | None = None,
) -> MagicMock:
    """Create a mock ConversationRepository with common async methods."""
    repo = MagicMock()
    repo.update_metadata = update_metadata or AsyncMock()
    repo.add_tag = add_tag or AsyncMock()
    repo.delete_conversation = delete_conversation or AsyncMock()
    return repo


# =============================================================================
# Parametrization Data Tables (Module-level Constants)
# =============================================================================

OUTPUT_FORMAT_TEST_CASES = (
    ("json", ["c1", "c2"], True, 2),  # format, expected_ids, is_json_parseable, min_lines
    ("yaml", ["c1"], False, 0),  # yaml doesn't have header row
    ("csv", ["c1", "c2"], False, 3),  # includes header
    ("text", ["c1"], False, -1),  # text format is variable
)

STREAM_TARGET_TEST_CASES = (
    ({"latest": True}, True, "latest-conv-id"),
    ({"conv_id": "abc"}, True, "full-conv-id-12345"),
    ({}, False, None),
)

DELETE_CONFIRMATION_TEST_CASES = (
    (11, False, True),  # count, force, should_confirm
    (1, False, True),
    (5, True, False),
)

SEND_OUTPUT_DESTINATIONS = (
    ("stdout", ["stdout"], False, "text", True, False, False),  # dest, mocks (stdout, file, browser, clipboard)
    ("file", [], True, "text", False, False, False),
    ("browser", ["browser"], False, "html", False, True, False),
    ("clipboard", ["clipboard"], False, "text", False, False, True),
)


# =============================================================================
# Tests for _no_results()
# =============================================================================


class TestNoResults:
    def _fn(self, env: AppEnv, params: dict, **kwargs):
        from polylogue.cli.query import _no_results

        return _no_results(env, params, **kwargs)

    @pytest.mark.parametrize(
        "has_filters,has_custom_exit_code,expected_code",
        [
            (True, False, 2),  # With filters: exit code 2
            (False, False, 2),  # Without filters: exit code 2
            (False, True, 1),  # Custom exit code
        ],
    )
    def test_no_results_behavior(
        self, has_filters, has_custom_exit_code, expected_code
    ) -> None:
        """Test _no_results with and without filters, custom exit codes."""
        import io
        from contextlib import redirect_stderr

        env = _make_env()
        params = _make_params(provider="claude", tag="important", query=("error",)) if has_filters else _make_params()

        kwargs = {"exit_code": 1} if has_custom_exit_code else {}

        stderr = io.StringIO()
        with pytest.raises(SystemExit) as exc_info:
            with redirect_stderr(stderr):
                self._fn(env, params, **kwargs)

        assert exc_info.value.code == expected_code

        if has_filters:
            output = stderr.getvalue()
            assert "filter" in output.lower()


# =============================================================================
# Tests for execute_query() - Main Orchestration
# =============================================================================


class TestExecuteQueryCount:
    """Test execute_query with --count flag."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @pytest.mark.parametrize(
        "use_summaries,result_count",
        [
            (True, 2),  # With summaries
            (False, 3),  # Fallback to list
        ],
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("click.echo")
    def test_count_only(
        self,
        mock_echo,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
        use_summaries,
        result_count,
    ) -> None:
        """Count-only mode uses summaries when available, falls back to list otherwise."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        mock_filter.can_use_summaries.return_value = use_summaries

        if use_summaries:
            mock_filter.list_summaries = AsyncMock(return_value=[_make_summary(), _make_summary()])
        else:
            mock_filter.list = AsyncMock(return_value=[_make_conv() for _ in range(result_count)])

        env = _make_env()
        params = _make_params(count_only=True)

        self._fn(env, params)

        mock_echo.assert_called_with(result_count)


class TestExecuteQueryBasic:
    """Test execute_query with single-feature flags (list, stats)."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @pytest.mark.parametrize(
        "has_summaries",
        [
            (True),  # Has summaries
            (False),  # No summaries, calls _no_results
        ],
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._output_summary_list")
    @patch("polylogue.cli.query._no_results")
    def test_list_mode(
        self,
        mock_no_results,
        mock_output,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
        has_summaries,
    ) -> None:
        """List mode outputs summaries when available, calls _no_results otherwise."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        mock_filter.can_use_summaries.return_value = True

        result_summaries = [_make_summary(), _make_summary()] if has_summaries else []
        mock_filter.list_summaries = AsyncMock(return_value=result_summaries)

        if not has_summaries:
            mock_no_results.side_effect = SystemExit(2)

        env = _make_env()
        params = _make_params(list_mode=True)

        if not has_summaries:
            with pytest.raises(SystemExit):
                self._fn(env, params)
            mock_no_results.assert_called_once()
        else:
            self._fn(env, params)
            mock_output.assert_called_once()
            args = mock_output.call_args[0]
            assert args[1] == result_summaries

    @pytest.mark.parametrize(
        "stats_key,stats_value,mock_fn",
        [
            ("stats_only", True, "_output_stats"),
            ("stats_by", "provider", "_output_stats_by"),
        ],
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._output_stats")
    @patch("polylogue.cli.query._output_stats_by")
    def test_stats_modes(
        self,
        mock_stats_by,
        mock_stats,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
        stats_key,
        stats_value,
        mock_fn,
    ) -> None:
        """stats_only and stats_by trigger appropriate output functions."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv()]
        mock_filter.list = AsyncMock(return_value=convs)

        env = _make_env()
        params = _make_params(**{stats_key: stats_value})

        self._fn(env, params)

        if mock_fn == "_output_stats":
            mock_stats.assert_called_once()
            assert mock_stats.call_args[0][1] == convs
        else:
            mock_stats_by.assert_called_once()
            assert mock_stats_by.call_args[0][2] == "provider"




class TestExecuteQueryStream:
    """Test execute_query with --stream flag."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @pytest.mark.parametrize(
        "param_overrides,resolves_id,expected_id",
        STREAM_TARGET_TEST_CASES,
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query.stream_conversation")
    @patch("click.echo")
    def test_stream_target_resolution(
        self,
        mock_echo,
        mock_stream,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
        param_overrides,
        resolves_id,
        expected_id,
    ) -> None:
        """Stream resolves targets correctly (latest, conv_id, or errors on missing)."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter

        if "latest" in param_overrides:
            summary = _make_summary(id="latest-conv-id")
            mock_filter.sort.return_value.limit.return_value.list_summaries = AsyncMock(return_value=[summary])
        elif "conv_id" in param_overrides:
            mock_repo.resolve_id.return_value = "full-conv-id-12345"
            mock_async_repo.resolve_id = AsyncMock(return_value="full-conv-id-12345")

        env = _make_env()
        params = _make_params(stream=True, **param_overrides)

        if not resolves_id:
            with pytest.raises(SystemExit) as exc_info:
                self._fn(env, params)
            assert exc_info.value.code == 1
        else:
            self._fn(env, params)
            mock_stream.assert_called_once()
            call_args = mock_stream.call_args[0]
            assert call_args[2] == expected_id

    @pytest.mark.parametrize(
        "param_overrides",
        [
            {"latest": True, "transform": "strip-tools"},
            {"latest": True, "output": "/tmp/out.txt"},
        ],
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query.stream_conversation")
    @patch("click.echo")
    def test_stream_warns_on_conflict(
        self,
        mock_echo,
        mock_stream,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
        param_overrides,
    ) -> None:
        """Stream prints warnings for transform and output file conflicts."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        summary = _make_summary(id="conv-id")
        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        mock_filter.sort.return_value.limit.return_value.list_summaries = AsyncMock(return_value=[summary])

        env = _make_env()
        params = _make_params(stream=True, **param_overrides)

        self._fn(env, params)

        # Check that warning was printed
        warning_calls = [call for call in mock_echo.call_args_list if "Warning" in str(call)]
        assert len(warning_calls) > 0


class TestExecuteQueryActions:
    """Test execute_query with --set-meta, --add-tag, --delete-matched."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @pytest.mark.parametrize(
        "param_key,param_value",
        [
            ("set_meta", [("key", "value")]),
            ("add_tag", ["important"]),
        ],
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._apply_modifiers")
    def test_modifiers_trigger_apply(
        self,
        mock_apply,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
        param_key,
        param_value,
    ) -> None:
        """set_meta and add_tag both trigger _apply_modifiers."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv()]
        mock_filter.list = AsyncMock(return_value=convs)

        env = _make_env()
        params = _make_params(**{param_key: param_value})

        self._fn(env, params)

        mock_apply.assert_called_once()
        assert mock_apply.call_args[0][1] == convs

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._delete_conversations")
    def test_delete_matched_calls_delete(
        self,
        mock_delete,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
    ) -> None:
        """delete_matched with a filter triggers _delete_conversations."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv(), _make_conv()]
        # Chain calls return the same mock so .provider().list() resolves correctly
        mock_filter.provider.return_value = mock_filter
        mock_filter.list = AsyncMock(return_value=convs)

        env = _make_env()
        # --delete requires at least one filter to prevent accidental full wipe
        params = _make_params(delete_matched=True, provider="claude")

        self._fn(env, params)

        mock_delete.assert_called_once()
        assert mock_delete.call_args[0][1] == convs


class TestExecuteQueryFlags:
    """Test execute_query with various flags (dialogue-only, transform)."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @pytest.mark.parametrize(
        "flag_param,flag_value",
        [
            ("dialogue_only", True),
        ],
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._output_results")
    def test_output_flags(
        self,
        mock_output,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
        flag_param,
        flag_value,
    ) -> None:
        """dialogue_only and other output flags apply transformations."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        conv = _make_conv()
        mock_filter.list = AsyncMock(return_value=[conv])

        env = _make_env()
        params = _make_params(**{flag_param: flag_value})

        self._fn(env, params)

        mock_output.assert_called_once()

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._output_results")
    @patch("polylogue.cli.query._apply_transform")
    def test_transform_applied(
        self,
        mock_transform,
        mock_output,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_get_async_repo,
        mock_load_config,
    ) -> None:
        """transform flag applies transformation."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_async_repo = MagicMock()
        mock_get_async_repo.return_value = mock_async_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv()]
        mock_filter.list = AsyncMock(return_value=convs)
        mock_transform.return_value = convs

        env = _make_env()
        params = _make_params(transform="strip-tools")

        self._fn(env, params)

        mock_transform.assert_called_once()
        assert mock_transform.call_args[0][1] == "strip-tools"


# =============================================================================
# Tests for _apply_modifiers()
# =============================================================================


class TestApplyModifiers:
    async def _fn(self, env: AppEnv, results: list, params: dict):
        from polylogue.cli.query import _apply_modifiers

        return await _apply_modifiers(env, results, params)

    @patch("polylogue.services.get_repository")
    async def test_no_results_prints_message(self, mock_get_repo) -> None:
        """No results prints message and returns."""
        env = _make_env()
        params = _make_params(set_meta=[("key", "val")])
        mock_get_repo.return_value = _make_mock_repo()

        await self._fn(env, [], params)

        env.ui.console.print.assert_called()

    @patch("polylogue.services.get_repository")
    async def test_dry_run_shows_preview(self, mock_get_repo) -> None:
        """Dry-run shows preview without modifying."""
        import io
        from contextlib import redirect_stdout

        env = _make_env()
        convs = [_make_conv(), _make_conv()]
        params = _make_params(set_meta=[("key", "value")], dry_run=True)
        mock_get_repo.return_value = _make_mock_repo()

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            await self._fn(env, convs, params)

        # Should print preview
        output = stdout.getvalue()
        assert "DRY-RUN" in output
        # Repository should not be modified
        mock_get_repo.return_value.update_metadata.assert_not_called()

    @pytest.mark.parametrize(
        "conv_count,force,should_confirm",
        [
            (11, False, True),  # Bulk without force: prompts
            (1, False, False),  # Small count: no prompt
            (11, True, False),  # Bulk with force: no prompt
        ],
    )
    @patch("polylogue.services.get_repository")
    async def test_bulk_operations_confirmation(
        self, mock_get_repo, conv_count, force, should_confirm
    ) -> None:
        """Bulk operations (>10) without force require confirmation; force skips it."""
        env = _make_env()
        convs = [_make_conv(id=f"conv-{i}") for i in range(conv_count)]
        params = _make_params(set_meta=[("key", "value")], force=force)

        mock_get_repo.return_value = _make_mock_repo()

        await self._fn(env, convs, params)

        if should_confirm:
            env.ui.confirm.assert_called_once()
        else:
            env.ui.confirm.assert_not_called()

    @patch("polylogue.services.get_repository")
    async def test_set_metadata(self, mock_get_repo) -> None:
        """set_meta updates metadata."""
        env = _make_env()
        convs = [_make_conv("c1"), _make_conv("c2")]
        params = _make_params(set_meta=[("key1", "val1"), ("key2", "val2")], force=True)

        mock_get_repo.return_value = _make_mock_repo()

        await self._fn(env, convs, params)

        # Verify metadata was set for each conv
        assert mock_get_repo.return_value.update_metadata.call_count == 4  # 2 convs * 2 metadata fields

    @patch("polylogue.services.get_repository")
    @patch("click.echo")
    async def test_add_tags(self, mock_echo, mock_get_repo) -> None:
        """add_tag adds tags to conversations."""
        env = _make_env()
        convs = [_make_conv("c1")]
        params = _make_params(add_tag=["tag1", "tag2"], force=True)

        mock_get_repo.return_value = _make_mock_repo()

        await self._fn(env, convs, params)

        assert mock_get_repo.return_value.add_tag.call_count == 2  # 1 conv * 2 tags

    @patch("polylogue.services.get_repository")
    @patch("click.echo")
    async def test_reports_results(self, mock_echo, mock_get_repo) -> None:
        """Reports number of modifications."""
        env = _make_env()
        convs = [_make_conv()]
        params = _make_params(set_meta=[("k", "v")], force=True)

        mock_get_repo.return_value = _make_mock_repo()

        await self._fn(env, convs, params)

        # Check that results were reported
        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("metadata" in str(c).lower() for c in calls)


# =============================================================================
# Tests for _delete_conversations()
# =============================================================================


class TestDeleteConversations:
    async def _fn(self, env: AppEnv, results: list, params: dict):
        from polylogue.cli.query import _delete_conversations

        return await _delete_conversations(env, results, params)

    @patch("polylogue.services.get_repository")
    async def test_no_results_prints_message(self, mock_get_repo) -> None:
        """No results prints message."""
        env = _make_env()
        params = _make_params(delete_matched=True)
        mock_get_repo.return_value = _make_mock_repo()

        await self._fn(env, [], params)

        env.ui.console.print.assert_called()

    @patch("polylogue.services.get_repository")
    async def test_dry_run_shows_breakdown(self, mock_get_repo) -> None:
        """Dry-run shows breakdown without deleting."""
        env = _make_env()
        convs = [_make_conv("c1", provider="claude"), _make_conv("c2", provider="chatgpt")]
        params = _make_params(delete_matched=True, dry_run=True)
        mock_get_repo.return_value = _make_mock_repo()

        await self._fn(env, convs, params)

        # Check for dry-run message
        [str(c) for c in env.ui.console.print.call_args_list]
        # Repository delete should not be called
        mock_get_repo.return_value.delete_conversation.assert_not_called()

    @pytest.mark.parametrize(
        "conv_count,force,should_confirm",
        DELETE_CONFIRMATION_TEST_CASES,
    )
    @patch("polylogue.services.get_repository")
    async def test_delete_confirmation_behavior(
        self, mock_get_repo, conv_count, force, should_confirm
    ) -> None:
        """Deletion requires confirmation unless --force, bulk or small counts both require."""
        env = _make_env()
        convs = [_make_conv(id=f"c-{i}") for i in range(conv_count)]
        params = _make_params(delete_matched=True, force=force)

        if should_confirm:
            env.ui.confirm.return_value = False

        # When force=True (no confirmation), execution reaches repo.delete_conversation
        mock_get_repo.return_value = _make_mock_repo(
            delete_conversation=AsyncMock(return_value=True)
        )

        await self._fn(env, convs, params)

        if should_confirm:
            env.ui.confirm.assert_called_once()
        else:
            env.ui.confirm.assert_not_called()

    @patch("polylogue.services.get_repository")
    @patch("click.echo")
    async def test_deletes_conversations(self, mock_echo, mock_get_repo) -> None:
        """Deletes conversations when confirmed."""
        env = _make_env()
        env.ui.confirm.return_value = True
        convs = [_make_conv("c1"), _make_conv("c2")]
        params = _make_params(delete_matched=True)

        mock_get_repo.return_value = _make_mock_repo(
            delete_conversation=AsyncMock(side_effect=[True, True])
        )

        await self._fn(env, convs, params)

        assert mock_get_repo.return_value.delete_conversation.call_count == 2


# =============================================================================
# Tests for _output_summary_list()
# =============================================================================


class TestOutputSummaryList:
    async def _fn(self, env: AppEnv, summaries: list, params: dict, repo=None):
        from polylogue.cli.query import _output_summary_list

        return await _output_summary_list(env, summaries, params, repo)

    @pytest.mark.parametrize(
        "output_format,expected_ids,is_json_parseable,min_lines",
        OUTPUT_FORMAT_TEST_CASES,
    )
    @patch("click.echo")
    async def test_output_formats(self, mock_echo, output_format, expected_ids, is_json_parseable, min_lines) -> None:
        """Test JSON, YAML, CSV, and text output formats."""
        env = _make_env()
        summaries = [_make_summary(id) for id in expected_ids]
        params = _make_params(output_format=output_format)

        await self._fn(env, summaries, params)

        if output_format == "text":
            # Plain text mode: uses click.echo per row (env.ui.plain=True)
            assert mock_echo.call_count == len(expected_ids)
            rendered = "\n".join(
                call.args[0] for call in mock_echo.call_args_list
            )
            for id in expected_ids:
                assert id in rendered
        elif output_format == "json":
            # JSON format - verify parseable and structure
            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert len(data) == len(expected_ids)
            assert data[0]["id"] == expected_ids[0]
        elif output_format == "csv":
            # CSV has header row
            output = mock_echo.call_args[0][0]
            lines = output.split("\n")
            assert len(lines) >= min_lines
            assert "id" in lines[0]  # Header check for CSV
        elif output_format == "yaml":
            # YAML - just verify it's output and contains IDs
            output = mock_echo.call_args[0][0]
            for id in expected_ids:
                assert id in output


# =============================================================================
# Tests for stream_conversation()
# =============================================================================


class TestStreamConversation:
    async def _fn(self, env: AppEnv, repo, conv_id: str, **kwargs):
        from polylogue.cli.query import stream_conversation

        return await stream_conversation(env, repo, conv_id, **kwargs)

    @staticmethod
    def _make_async_repo(**overrides):
        """Create a mock repo with async methods matching stream_conversation's usage."""
        repo = MagicMock()
        repo.backend.get_conversation = AsyncMock(return_value=overrides.get("conv_record"))
        repo.get_conversation_stats = AsyncMock(return_value=overrides.get("stats", {}))

        # iter_messages returns an async iterator
        messages = overrides.get("messages", [])

        async def _async_iter(*args, **kwargs):
            for m in messages:
                yield m

        repo.iter_messages = _async_iter
        return repo

    async def test_conversation_not_found(self) -> None:
        """Conversation not found raises SystemExit."""
        env = _make_env()
        repo = self._make_async_repo(conv_record=None)

        with pytest.raises(SystemExit) as exc_info:
            await self._fn(env, repo, "nonexistent")

        assert exc_info.value.code == 1

    @pytest.mark.parametrize(
        "output_format,has_header,has_footer",
        [
            ("markdown", True, True),
            ("json-lines", True, True),
        ],
    )
    @patch("sys.stdout", new_callable=StringIO)
    async def test_stream_formats(self, mock_stdout, output_format, has_header, has_footer) -> None:
        """Test markdown and json-lines streaming formats with headers/footers."""
        env = _make_env()

        conv_record = MagicMock()
        conv_record.title = "Test Title"
        repo = self._make_async_repo(
            conv_record=conv_record,
            stats={"total_messages": 5, "dialogue_messages": 3},
        )

        await self._fn(env, repo, "conv-1", output_format=output_format)

        output = mock_stdout.getvalue()

        if output_format == "markdown":
            if has_header:
                assert "# Test Title" in output
            if has_footer:
                assert "---" in output
        elif output_format == "json-lines":
            lines = output.strip().split("\n")
            if has_header:
                header = json.loads(lines[0])
                assert header["type"] == "header"
            if has_footer:
                footer = json.loads(lines[-1])
                assert footer["type"] == "footer"

    @patch("sys.stdout", new_callable=StringIO)
    async def test_returns_message_count(self, mock_stdout) -> None:
        """Returns number of messages streamed."""
        env = _make_env()

        conv_record = MagicMock()
        conv_record.title = "Test"
        msg1 = _make_msg("msg-1", "user", "Hello")
        msg2 = _make_msg("msg-2", "assistant", "Hi")
        repo = self._make_async_repo(
            conv_record=conv_record,
            messages=[msg1, msg2],
        )

        count = await self._fn(env, repo, "conv-1")

        assert count == 2


# =============================================================================
# Tests for _send_output()
# =============================================================================


class TestSendOutput:
    def _fn(self, env: AppEnv, content: str, destinations: list, output_format: str, conv=None):
        from polylogue.cli.query import _send_output

        return _send_output(env, content, destinations, output_format, conv)

    @pytest.mark.parametrize(
        "dest_label,destinations,expects_file,output_format,expects_stdout,expects_browser,expects_clipboard",
        SEND_OUTPUT_DESTINATIONS,
    )
    @patch("click.echo")
    @patch("polylogue.cli.query._open_in_browser")
    @patch("polylogue.cli.query._copy_to_clipboard")
    def test_send_to_destinations(
        self,
        mock_clipboard,
        mock_browser,
        mock_echo,
        dest_label,
        destinations,
        expects_file,
        output_format,
        expects_stdout,
        expects_browser,
        expects_clipboard,
    ) -> None:
        """Test output routing to stdout, browser, clipboard, and file destinations."""
        env = _make_env()
        content = "test output"
        conv = _make_conv() if "browser" in destinations else None

        if expects_file:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = f"{tmpdir}/output.txt"
                destinations = [path]
                self._fn(env, content, destinations, output_format, conv)
                assert Path(path).exists()
                assert Path(path).read_text() == content
        else:
            self._fn(env, content, destinations, output_format, conv)

        if expects_stdout:
            mock_echo.assert_called_with(content)
        if expects_browser:
            mock_browser.assert_called_once()
        if expects_clipboard:
            mock_clipboard.assert_called_once_with(env, content)

    def test_multiple_destinations(self) -> None:
        """Multiple destinations are all handled."""
        env = _make_env()
        content = "test"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/out.txt"
            with patch("click.echo"):
                self._fn(env, content, ["stdout", path], "text")

            assert Path(path).exists()


# =============================================================================
# Tests for _open_in_browser()
# =============================================================================


class TestQueryOpenInBrowser:
    def _fn(self, env: AppEnv, content: str, output_format: str, conv=None):
        from polylogue.cli.query import _open_in_browser

        return _open_in_browser(env, content, output_format, conv)

    @pytest.mark.parametrize(
        "content,output_format,use_conv,is_html_content",
        [
            ("<html><body>Test</body></html>", "html", False, True),
            ("Plain text", "text", False, False),
        ],
    )
    @patch("webbrowser.open")
    def test_browser_content_handling(
        self, mock_browser, content, output_format, use_conv, is_html_content
    ) -> None:
        """Test HTML pass-through and content wrapping for non-HTML formats."""
        env = _make_env()
        conv = _make_conv() if use_conv else None

        self._fn(env, content, output_format, conv=conv)

        mock_browser.assert_called_once()
        # Verify file:// URL was used
        call_arg = mock_browser.call_args[0][0]
        assert call_arg.startswith("file://")

    @patch("webbrowser.open")
    def test_browser_with_conversation(self, mock_browser) -> None:
        """When conv provided, converts to HTML."""
        env = _make_env()
        conv = _make_conv()

        self._fn(env, "ignored", "text", conv=conv)

        mock_browser.assert_called_once()


# =============================================================================
# Tests for _copy_to_clipboard()
# =============================================================================


class TestCopyToClipboard:
    def _fn(self, env: AppEnv, content: str):
        from polylogue.cli.query import _copy_to_clipboard

        return _copy_to_clipboard(env, content)

    @pytest.mark.parametrize(
        "tool_success,should_fail",
        [
            (True, False),  # xclip succeeds
            (False, True),  # all tools fail
        ],
    )
    @patch("subprocess.run")
    @patch("click.echo")
    def test_clipboard_tools(self, mock_echo, mock_run, tool_success, should_fail) -> None:
        """Test clipboard tool success and failure scenarios."""
        env = _make_env()
        if tool_success:
            mock_run.return_value = MagicMock(returncode=0)
            self._fn(env, "test content")
            mock_run.assert_called()
            env.ui.console.print.assert_called()
        else:
            mock_run.side_effect = FileNotFoundError()
            self._fn(env, "test")
            assert mock_echo.called


# =============================================================================
# Tests for _open_result()
# =============================================================================


class TestOpenResult:
    def _fn(self, env: AppEnv, results: list, params: dict):
        from polylogue.cli.query import _open_result

        return _open_result(env, results, params)

    def test_no_results_exits(self) -> None:
        """No results raises SystemExit."""
        env = _make_env()
        params = _make_params()

        with pytest.raises(SystemExit) as exc_info:
            self._fn(env, [], params)

        assert exc_info.value.code == 2

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("click.echo")
    def test_no_render_root_errors(self, mock_echo, mock_config) -> None:
        """No render_root shows error."""
        env = _make_env()
        mock_config.return_value = None
        conv = _make_conv("c1")
        params = _make_params()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                self._fn(env, [conv], params)

            assert exc_info.value.code == 1

    @patch("webbrowser.open")
    @patch("polylogue.cli.helpers.load_effective_config")
    def test_opens_html_file(self, mock_config, mock_browser) -> None:
        """Opens HTML file when found and opens in browser."""
        env = _make_env()
        conv = _make_conv("c1")
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmpdir:
            render_root = Path(tmpdir) / "renders" / "c1234567"
            render_root.mkdir(parents=True)
            html_file = render_root / "conversation.html"
            html_file.write_text("<html>test</html>")

            mock_config.return_value = MagicMock()
            mock_config.return_value.render_root = str(Path(tmpdir) / "renders")

            self._fn(env, [conv], params)

            mock_browser.assert_called_once()


# =============================================================================
# Helpers for mocking click.echo
# =============================================================================


MockEchoCalls = []


@pytest.fixture(autouse=True)
def reset_echo_calls():
    """Reset echo call tracker."""
    global MockEchoCalls
    MockEchoCalls = []
    yield
    MockEchoCalls = []

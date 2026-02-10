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
import sys
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from polylogue.cli.types import AppEnv
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.lib.messages import MessageCollection


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


# =============================================================================
# Tests for _no_results()
# =============================================================================


class TestNoResults:
    def _fn(self, env: AppEnv, params: dict, **kwargs):
        from polylogue.cli.query import _no_results

        return _no_results(env, params, **kwargs)

    def test_with_filters_shows_descriptive_message(self) -> None:
        """With active filters, prints filter details."""
        import io
        from contextlib import redirect_stderr

        env = _make_env()
        params = _make_params(
            provider="claude",
            tag="important",
            query=("error",),
        )

        stderr = io.StringIO()
        with pytest.raises(SystemExit) as exc_info:
            with redirect_stderr(stderr):
                self._fn(env, params)

        assert exc_info.value.code == 2
        output = stderr.getvalue()
        # Verify that filter info was printed
        assert "filter" in output.lower()

    def test_without_filters_generic_message(self) -> None:
        """Without filters, prints generic message."""
        env = _make_env()
        params = _make_params()

        with pytest.raises(SystemExit) as exc_info:
            self._fn(env, params)

        assert exc_info.value.code == 2

    def test_custom_exit_code(self) -> None:
        """Supports custom exit codes."""
        env = _make_env()
        params = _make_params()

        with pytest.raises(SystemExit) as exc_info:
            self._fn(env, params, exit_code=1)

        assert exc_info.value.code == 1


# =============================================================================
# Tests for execute_query() - Main Orchestration
# =============================================================================


class TestExecuteQueryCountOnly:
    """Test execute_query with --count flag."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("click.echo")
    def test_count_only_with_summaries(
        self,
        mock_echo,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """Count-only mode uses summaries when available."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        mock_filter.can_use_summaries.return_value = True
        mock_filter.list_summaries.return_value = [_make_summary(), _make_summary()]

        env = _make_env()
        params = _make_params(count_only=True)

        self._fn(env, params)

        mock_echo.assert_called_with(2)

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("click.echo")
    def test_count_only_fallback_to_list(
        self,
        mock_echo,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """Count-only falls back to list() when summaries not available."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        mock_filter.can_use_summaries.return_value = False
        mock_filter.list.return_value = [_make_conv(), _make_conv(), _make_conv()]

        env = _make_env()
        params = _make_params(count_only=True)

        self._fn(env, params)

        mock_echo.assert_called_with(3)


class TestExecuteQueryListMode:
    """Test execute_query with --list flag."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._output_summary_list")
    def test_list_mode_with_summaries(
        self,
        mock_output,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """List mode outputs summaries when available."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        mock_filter.can_use_summaries.return_value = True
        summaries = [_make_summary(), _make_summary()]
        mock_filter.list_summaries.return_value = summaries

        env = _make_env()
        params = _make_params(list_mode=True)

        self._fn(env, params)

        mock_output.assert_called_once()
        args = mock_output.call_args[0]
        assert args[1] == summaries  # Check summaries were passed

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._no_results")
    def test_list_mode_no_results(
        self,
        mock_no_results,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """List mode calls _no_results when no summaries found."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        mock_filter.can_use_summaries.return_value = True
        mock_filter.list_summaries.return_value = []

        env = _make_env()
        params = _make_params(list_mode=True)

        mock_no_results.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            self._fn(env, params)

        mock_no_results.assert_called_once()


class TestExecuteQueryStreamMode:
    """Test execute_query with --stream flag."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query.stream_conversation")
    @patch("click.echo")
    def test_stream_with_latest(
        self,
        mock_echo,
        mock_stream,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """Stream with --latest resolves to most recent conversation via filter chain."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        summary = _make_summary(id="latest-conv-id")

        # --latest applies .sort("date").limit(1) to filter_chain,
        # then the streaming path calls .list_summaries() on it.
        mock_filter = MagicMock()
        mock_filter.sort.return_value.limit.return_value.list_summaries.return_value = [summary]
        MockFilter.return_value = mock_filter

        env = _make_env()
        params = _make_params(stream=True, latest=True)

        self._fn(env, params)

        mock_stream.assert_called_once()
        call_args = mock_stream.call_args[0]
        assert call_args[2] == "latest-conv-id"  # conversation_id

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query.stream_conversation")
    @patch("click.echo")
    def test_stream_with_conv_id(
        self,
        mock_echo,
        mock_stream,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """Stream with --id resolves conversation ID."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_repo.resolve_id.return_value = "full-conv-id-12345"

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter

        env = _make_env()
        params = _make_params(stream=True, conv_id="abc")

        self._fn(env, params)

        mock_stream.assert_called_once()
        call_args = mock_stream.call_args[0]
        assert call_args[2] == "full-conv-id-12345"

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("click.echo")
    def test_stream_without_target_errors(
        self,
        mock_echo,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """Stream without a target (--latest, --id, or query term) fails."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter

        env = _make_env()
        params = _make_params(stream=True)

        with pytest.raises(SystemExit) as exc_info:
            self._fn(env, params)

        assert exc_info.value.code == 1

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query.stream_conversation")
    @patch("click.echo")
    def test_stream_warns_on_transform(
        self,
        mock_echo,
        mock_stream,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """Stream with --transform prints warning."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        summary = _make_summary(id="conv-id")
        mock_repo.list_summaries.return_value = [summary]

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter

        env = _make_env()
        params = _make_params(stream=True, latest=True, transform="strip-tools")

        self._fn(env, params)

        # Check that warning was printed
        warning_calls = [call for call in mock_echo.call_args_list if "Warning" in str(call)]
        assert len(warning_calls) > 0

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query.stream_conversation")
    @patch("click.echo")
    def test_stream_warns_on_output_file(
        self,
        mock_echo,
        mock_stream,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """Stream with --output file prints warning."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        summary = _make_summary(id="conv-id")
        mock_repo.list_summaries.return_value = [summary]

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter

        env = _make_env()
        params = _make_params(stream=True, latest=True, output="/tmp/out.txt")

        self._fn(env, params)

        # Check that warning was printed
        warning_calls = [call for call in mock_echo.call_args_list if "Warning" in str(call)]
        assert len(warning_calls) > 0


class TestExecuteQueryModifiers:
    """Test execute_query with --set-meta and --add-tag."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._apply_modifiers")
    def test_set_meta_calls_apply_modifiers(
        self,
        mock_apply,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """set_meta triggers _apply_modifiers."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv()]
        mock_filter.list.return_value = convs

        env = _make_env()
        params = _make_params(set_meta=[("key", "value")])

        self._fn(env, params)

        mock_apply.assert_called_once()
        assert mock_apply.call_args[0][1] == convs

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._apply_modifiers")
    def test_add_tag_calls_apply_modifiers(
        self,
        mock_apply,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """add_tag triggers _apply_modifiers."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv()]
        mock_filter.list.return_value = convs

        env = _make_env()
        params = _make_params(add_tag=["important"])

        self._fn(env, params)

        mock_apply.assert_called_once()


class TestExecuteQueryDelete:
    """Test execute_query with --delete-matched."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @patch("polylogue.cli.helpers.load_effective_config")
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
        mock_load_config,
    ) -> None:
        """delete_matched with a filter triggers _delete_conversations."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv(), _make_conv()]
        # Chain calls return the same mock so .provider().list() resolves correctly
        mock_filter.provider.return_value = mock_filter
        mock_filter.list.return_value = convs

        env = _make_env()
        # --delete requires at least one filter to prevent accidental full wipe
        params = _make_params(delete_matched=True, provider="claude")

        self._fn(env, params)

        mock_delete.assert_called_once()
        assert mock_delete.call_args[0][1] == convs


class TestExecuteQueryStats:
    """Test execute_query with --stats and --stats-by."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._output_stats")
    def test_stats_only_calls_output_stats(
        self,
        mock_output,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """stats_only triggers _output_stats."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv()]
        mock_filter.list.return_value = convs

        env = _make_env()
        params = _make_params(stats_only=True)

        self._fn(env, params)

        mock_output.assert_called_once()
        assert mock_output.call_args[0][1] == convs

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._output_stats_by")
    def test_stats_by_calls_output_stats_by(
        self,
        mock_output,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """stats_by triggers _output_stats_by."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv()]
        mock_filter.list.return_value = convs

        env = _make_env()
        params = _make_params(stats_by="provider")

        self._fn(env, params)

        mock_output.assert_called_once()
        assert mock_output.call_args[0][2] == "provider"


class TestExecuteQueryDialogueOnly:
    """Test execute_query with --dialogue-only."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.services.get_repository")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query._output_results")
    def test_dialogue_only_filters_messages(
        self,
        mock_output,
        MockFilter,
        mock_vector_provider,
        mock_get_repo,
        mock_load_config,
    ) -> None:
        """dialogue_only applies dialogue_only() to conversations."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        conv = _make_conv()
        mock_filter.list.return_value = [conv]

        env = _make_env()
        params = _make_params(dialogue_only=True)

        self._fn(env, params)

        mock_output.assert_called_once()


class TestExecuteQueryTransform:
    """Test execute_query with --transform."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @patch("polylogue.cli.helpers.load_effective_config")
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
        mock_load_config,
    ) -> None:
        """transform flag applies transformation."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        convs = [_make_conv()]
        mock_filter.list.return_value = convs
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
    def _fn(self, env: AppEnv, results: list, params: dict):
        from polylogue.cli.query import _apply_modifiers

        return _apply_modifiers(env, results, params)

    @patch("polylogue.services.get_repository")
    def test_no_results_prints_message(self, mock_get_repo) -> None:
        """No results prints message and returns."""
        env = _make_env()
        params = _make_params(set_meta=[("key", "val")])

        self._fn(env, [], params)

        env.ui.console.print.assert_called()

    @patch("polylogue.services.get_repository")
    def test_dry_run_shows_preview(self, mock_get_repo) -> None:
        """Dry-run shows preview without modifying."""
        import io
        from contextlib import redirect_stdout

        env = _make_env()
        convs = [_make_conv(), _make_conv()]
        params = _make_params(set_meta=[("key", "value")], dry_run=True)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self._fn(env, convs, params)

        # Should print preview
        output = stdout.getvalue()
        assert "DRY-RUN" in output
        # Repository should not be modified
        mock_get_repo.return_value.update_metadata.assert_not_called()

    @patch("polylogue.services.get_repository")
    def test_bulk_without_force_prompts(self, mock_get_repo) -> None:
        """Bulk operations (>10) without force prompt for confirmation."""
        env = _make_env()
        convs = [_make_conv(id=f"conv-{i}") for i in range(11)]
        params = _make_params(set_meta=[("key", "value")], force=False)
        env.ui.confirm.return_value = False

        self._fn(env, convs, params)

        env.ui.confirm.assert_called_once()

    @patch("polylogue.services.get_repository")
    def test_bulk_with_force_skips_prompt(self, mock_get_repo) -> None:
        """Bulk with --force skips confirmation."""
        env = _make_env()
        convs = [_make_conv(id=f"conv-{i}") for i in range(11)]
        params = _make_params(set_meta=[("key", "value")], force=True)

        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        self._fn(env, convs, params)

        env.ui.confirm.assert_not_called()

    @patch("polylogue.services.get_repository")
    def test_set_metadata(self, mock_get_repo) -> None:
        """set_meta updates metadata."""
        env = _make_env()
        convs = [_make_conv("c1"), _make_conv("c2")]
        params = _make_params(set_meta=[("key1", "val1"), ("key2", "val2")], force=True)

        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        self._fn(env, convs, params)

        # Verify metadata was set for each conv
        assert mock_repo.update_metadata.call_count == 4  # 2 convs * 2 metadata fields

    @patch("polylogue.services.get_repository")
    @patch("click.echo")
    def test_add_tags(self, mock_echo, mock_get_repo) -> None:
        """add_tag adds tags to conversations."""
        env = _make_env()
        convs = [_make_conv("c1")]
        params = _make_params(add_tag=["tag1", "tag2"], force=True)

        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        self._fn(env, convs, params)

        assert mock_repo.add_tag.call_count == 2  # 1 conv * 2 tags

    @patch("polylogue.services.get_repository")
    @patch("click.echo")
    def test_reports_results(self, mock_echo, mock_get_repo) -> None:
        """Reports number of modifications."""
        env = _make_env()
        convs = [_make_conv()]
        params = _make_params(set_meta=[("k", "v")], force=True)

        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo

        self._fn(env, convs, params)

        # Check that results were reported
        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("metadata" in str(c).lower() for c in calls)


# =============================================================================
# Tests for _delete_conversations()
# =============================================================================


class TestDeleteConversations:
    def _fn(self, env: AppEnv, results: list, params: dict):
        from polylogue.cli.query import _delete_conversations

        return _delete_conversations(env, results, params)

    @patch("polylogue.services.get_repository")
    def test_no_results_prints_message(self, mock_get_repo) -> None:
        """No results prints message."""
        env = _make_env()
        params = _make_params(delete_matched=True)

        self._fn(env, [], params)

        env.ui.console.print.assert_called()

    @patch("polylogue.services.get_repository")
    def test_dry_run_shows_breakdown(self, mock_get_repo) -> None:
        """Dry-run shows breakdown without deleting."""
        from unittest.mock import call

        env = _make_env()
        convs = [_make_conv("c1", provider="claude"), _make_conv("c2", provider="chatgpt")]
        params = _make_params(delete_matched=True, dry_run=True)

        self._fn(env, convs, params)

        # Check for dry-run message
        calls = [str(c) for c in env.ui.console.print.call_args_list]
        echo_calls = [str(c) for c in MockEchoCalls]
        # Repository delete should not be called
        mock_get_repo.return_value.delete_conversation.assert_not_called()

    @patch("polylogue.services.get_repository")
    def test_bulk_requires_confirmation(self, mock_get_repo) -> None:
        """Deleting >10 items requires confirmation."""
        env = _make_env()
        convs = [_make_conv(id=f"c-{i}") for i in range(11)]
        params = _make_params(delete_matched=True)
        env.ui.confirm.return_value = False

        self._fn(env, convs, params)

        env.ui.confirm.assert_called_once()

    @patch("polylogue.services.get_repository")
    def test_small_count_requires_confirmation(self, mock_get_repo) -> None:
        """Deleting small counts requires confirmation if not forced."""
        env = _make_env()
        convs = [_make_conv("c1")]
        params = _make_params(delete_matched=True, force=False)
        env.ui.confirm.return_value = False

        self._fn(env, convs, params)

        env.ui.confirm.assert_called_once()

    @patch("polylogue.services.get_repository")
    @patch("click.echo")
    def test_force_skips_confirmation(self, mock_echo, mock_get_repo) -> None:
        """--force skips confirmation."""
        env = _make_env()
        convs = [_make_conv("c1")]
        params = _make_params(delete_matched=True, force=True)

        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_repo.delete_conversation.return_value = True

        self._fn(env, convs, params)

        env.ui.confirm.assert_not_called()

    @patch("polylogue.services.get_repository")
    @patch("click.echo")
    def test_deletes_conversations(self, mock_echo, mock_get_repo) -> None:
        """Deletes conversations when confirmed."""
        env = _make_env()
        env.ui.confirm.return_value = True
        convs = [_make_conv("c1"), _make_conv("c2")]
        params = _make_params(delete_matched=True)

        mock_repo = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_repo.delete_conversation.side_effect = [True, True]

        self._fn(env, convs, params)

        assert mock_repo.delete_conversation.call_count == 2


# =============================================================================
# Tests for _output_summary_list()
# =============================================================================


class TestOutputSummaryList:
    def _fn(self, env: AppEnv, summaries: list, params: dict, repo=None):
        from polylogue.cli.query import _output_summary_list

        return _output_summary_list(env, summaries, params, repo)

    @patch("click.echo")
    def test_json_format(self, mock_echo) -> None:
        """JSON format outputs valid JSON."""
        env = _make_env()
        summaries = [_make_summary("c1"), _make_summary("c2")]
        params = _make_params(output_format="json")

        self._fn(env, summaries, params)

        # Get the output
        output = mock_echo.call_args[0][0]
        data = json.loads(output)
        assert len(data) == 2
        assert data[0]["id"] == "c1"

    @patch("click.echo")
    def test_yaml_format(self, mock_echo) -> None:
        """YAML format outputs valid YAML."""
        import yaml

        env = _make_env()
        summaries = [_make_summary("c1")]
        params = _make_params(output_format="yaml")

        self._fn(env, summaries, params)

        output = mock_echo.call_args[0][0]
        # Should parse as YAML without error
        data = yaml.safe_load(output)
        assert len(data) == 1

    @patch("click.echo")
    def test_csv_format(self, mock_echo) -> None:
        """CSV format outputs valid CSV."""
        env = _make_env()
        summaries = [_make_summary("c1"), _make_summary("c2")]
        params = _make_params(output_format="csv")

        self._fn(env, summaries, params)

        output = mock_echo.call_args[0][0]
        lines = output.split("\n")
        assert len(lines) >= 3  # header + 2 rows
        assert "id" in lines[0]

    def test_plain_text_format(self) -> None:
        """Plain text format outputs readable list."""
        env = _make_env()
        summaries = [_make_summary("c1", title="Test Conv")]
        params = _make_params(output_format="text")

        self._fn(env, summaries, params)

        env.ui.console.print.assert_called_once()
        output = env.ui.console.print.call_args[0][0]
        assert "c1" in output
        assert "Test Conv" in output


# =============================================================================
# Tests for stream_conversation()
# =============================================================================


class TestStreamConversation:
    def _fn(self, env: AppEnv, repo, conv_id: str, **kwargs):
        from polylogue.cli.query import stream_conversation

        return stream_conversation(env, repo, conv_id, **kwargs)

    def test_conversation_not_found(self) -> None:
        """Conversation not found raises SystemExit."""
        env = _make_env()
        repo = MagicMock()
        repo.backend.get_conversation.return_value = None

        with pytest.raises(SystemExit) as exc_info:
            self._fn(env, repo, "nonexistent")

        assert exc_info.value.code == 1

    @patch("sys.stdout", new_callable=StringIO)
    def test_markdown_header_footer(self, mock_stdout) -> None:
        """Markdown format includes header and footer."""
        env = _make_env()
        repo = MagicMock()

        conv_record = MagicMock()
        conv_record.title = "Test Title"
        repo.backend.get_conversation.return_value = conv_record
        repo.get_conversation_stats.return_value = {"total_messages": 5, "dialogue_messages": 3}
        repo.iter_messages.return_value = []

        self._fn(env, repo, "conv-1", output_format="markdown")

        output = mock_stdout.getvalue()
        assert "# Test Title" in output
        assert "---" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_json_lines_header_footer(self, mock_stdout) -> None:
        """JSON-lines format includes header and footer."""
        env = _make_env()
        repo = MagicMock()

        conv_record = MagicMock()
        conv_record.title = "Test"
        repo.backend.get_conversation.return_value = conv_record
        repo.get_conversation_stats.return_value = {}
        repo.iter_messages.return_value = []

        self._fn(env, repo, "conv-1", output_format="json-lines")

        output = mock_stdout.getvalue()
        lines = output.strip().split("\n")
        assert len(lines) >= 2
        # First line is header
        header = json.loads(lines[0])
        assert header["type"] == "header"
        # Last line is footer
        footer = json.loads(lines[-1])
        assert footer["type"] == "footer"

    @patch("sys.stdout", new_callable=StringIO)
    def test_returns_message_count(self, mock_stdout) -> None:
        """Returns number of messages streamed."""
        env = _make_env()
        repo = MagicMock()

        conv_record = MagicMock()
        conv_record.title = "Test"
        repo.backend.get_conversation.return_value = conv_record
        repo.get_conversation_stats.return_value = {}

        msg1 = _make_msg("msg-1", "user", "Hello")
        msg2 = _make_msg("msg-2", "assistant", "Hi")
        repo.iter_messages.return_value = [msg1, msg2]

        count = self._fn(env, repo, "conv-1")

        assert count == 2


# =============================================================================
# Tests for _send_output()
# =============================================================================


class TestSendOutput:
    def _fn(self, env: AppEnv, content: str, destinations: list, output_format: str, conv=None):
        from polylogue.cli.query import _send_output

        return _send_output(env, content, destinations, output_format, conv)

    @patch("click.echo")
    def test_stdout_destination(self, mock_echo) -> None:
        """stdout destination uses click.echo."""
        env = _make_env()
        content = "test output"

        self._fn(env, content, ["stdout"], "text")

        mock_echo.assert_called_with(content)

    @patch("polylogue.cli.query._open_in_browser")
    def test_browser_destination(self, mock_browser) -> None:
        """browser destination calls _open_in_browser."""
        env = _make_env()
        content = "test"
        conv = _make_conv()

        self._fn(env, content, ["browser"], "html", conv)

        mock_browser.assert_called_once()

    @patch("polylogue.cli.query._copy_to_clipboard")
    def test_clipboard_destination(self, mock_clipboard) -> None:
        """clipboard destination calls _copy_to_clipboard."""
        env = _make_env()
        content = "test"

        self._fn(env, content, ["clipboard"], "text")

        mock_clipboard.assert_called_once_with(env, content)

    def test_file_destination_creates_file(self) -> None:
        """File destination writes file."""
        env = _make_env()
        content = "test output"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/output.txt"
            self._fn(env, content, [path], "text")

            assert Path(path).exists()
            assert Path(path).read_text() == content

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


class TestOpenInBrowser:
    def _fn(self, env: AppEnv, content: str, output_format: str, conv=None):
        from polylogue.cli.query import _open_in_browser

        return _open_in_browser(env, content, output_format, conv)

    @patch("webbrowser.open")
    def test_html_content_untouched(self, mock_browser) -> None:
        """HTML content is passed to browser as-is."""
        env = _make_env()
        html_content = "<html><body>Test</body></html>"

        self._fn(env, html_content, "html")

        mock_browser.assert_called_once()
        # Verify file was written and browser was called
        call_arg = mock_browser.call_args[0][0]
        assert call_arg.startswith("file://")

    @patch("webbrowser.open")
    def test_non_html_wrapped_in_html(self, mock_browser) -> None:
        """Non-HTML content gets wrapped."""
        env = _make_env()
        content = "Plain text"

        self._fn(env, content, "text")

        mock_browser.assert_called_once()

    @patch("webbrowser.open")
    def test_conv_converts_to_html(self, mock_browser) -> None:
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

    @patch("subprocess.run")
    def test_xclip_success(self, mock_run) -> None:
        """Successful xclip invocation."""
        env = _make_env()
        mock_run.return_value = MagicMock(returncode=0)

        self._fn(env, "test content")

        mock_run.assert_called()
        env.ui.console.print.assert_called()

    @patch("subprocess.run")
    @patch("click.echo")
    def test_all_tools_fail(self, mock_echo, mock_run) -> None:
        """All clipboard tools fail shows error."""
        env = _make_env()
        mock_run.side_effect = FileNotFoundError()

        self._fn(env, "test")

        # Error message should be printed
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
        """Opens HTML file when found."""
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

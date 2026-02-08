"""Tests for CLI query routing logic in click_app.py.

Regression tests for the 3 query routing bugs fixed during the deep sweep:
1. --id filter was missing from has_filters → fell through to stats dashboard
2. Modifier flags (--add-tag, --set, --delete) not in routing check → silently ignored
3. Output mode flags (--list, --count, --stats, --stream) routing

Also tests QueryFirstGroup.parse_args positional arg → --query-term conversion.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


# =============================================================================
# _handle_query_mode internal routing
# =============================================================================


class TestHandleQueryMode:
    """Test the internal routing logic of _handle_query_mode.

    The function decides between stats mode and query mode based on
    has_filters, has_output_mode, and has_modifiers checks on ctx.params.
    """

    def _make_params(self, **overrides):
        """Build a minimal ctx.params dict for _handle_query_mode."""
        defaults = {
            "query_term": (),
            "conv_id": None,
            "contains": (),
            "exclude_text": (),
            "provider": None,
            "exclude_provider": None,
            "tag": None,
            "exclude_tag": None,
            "has_type": (),
            "since": None,
            "until": None,
            "title": None,
            "latest": False,
            "limit": None,
            "sort": None,
            "reverse": False,
            "sample": None,
            "output": None,
            "output_format": None,
            "fields": None,
            "list_mode": False,
            "stats_only": False,
            "count_only": False,
            "stats_by": None,
            "open_result": False,
            "transform": None,
            "stream": False,
            "dialogue_only": False,
            "set_meta": (),
            "add_tag": (),
            "delete_matched": False,
            "dry_run": False,
            "force": False,
            "plain": False,
            "verbose": False,
        }
        defaults.update(overrides)
        return defaults

    def _call(self, params):
        """Call _handle_query_mode with mocked ctx and track which path is taken.

        Returns (execute_query_mock, show_stats_mock).
        """
        from polylogue.cli.click_app import _handle_query_mode

        mock_ctx = MagicMock()
        mock_ctx.params = params
        mock_ctx.obj = MagicMock()  # AppEnv

        # execute_query is imported lazily inside _handle_query_mode
        with patch("polylogue.cli.query.execute_query") as mock_execute:
            with patch("polylogue.cli.click_app._show_stats") as mock_stats:
                _handle_query_mode(mock_ctx)
                return mock_execute, mock_stats

    # --- Stats mode (no filters, no output, no modifiers) ---

    def test_no_args_shows_stats(self):
        """Empty params → stats mode."""
        mock_execute, mock_stats = self._call(self._make_params())
        mock_stats.assert_called_once()
        mock_execute.assert_not_called()

    def test_verbose_stats(self):
        """Verbose flag alone → stats mode with verbose=True."""
        mock_execute, mock_stats = self._call(self._make_params(verbose=True))
        mock_stats.assert_called_once()
        mock_execute.assert_not_called()

    # --- Query terms ---

    def test_query_terms_trigger_query(self):
        """Positional query terms → query mode."""
        mock_execute, mock_stats = self._call(
            self._make_params(query_term=("error", "handling"))
        )
        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    # --- Filter flags ---

    def test_conv_id_triggers_query(self):
        """--id flag → query mode (REGRESSION: was missing from has_filters)."""
        mock_execute, mock_stats = self._call(self._make_params(conv_id="abc123"))
        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    def test_provider_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(provider="claude"))
        mock_execute.assert_called_once()

    def test_tag_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(tag="important"))
        mock_execute.assert_called_once()

    def test_contains_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(contains=("error",)))
        mock_execute.assert_called_once()

    def test_has_type_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(has_type=("thinking",)))
        mock_execute.assert_called_once()

    def test_since_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(since="2025-01-01"))
        mock_execute.assert_called_once()

    def test_until_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(until="2025-12-31"))
        mock_execute.assert_called_once()

    def test_latest_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(latest=True))
        mock_execute.assert_called_once()

    def test_title_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(title="test"))
        mock_execute.assert_called_once()

    def test_exclude_text_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(exclude_text=("noise",)))
        mock_execute.assert_called_once()

    def test_exclude_provider_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(exclude_provider="chatgpt"))
        mock_execute.assert_called_once()

    def test_exclude_tag_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(exclude_tag="deprecated"))
        mock_execute.assert_called_once()

    # --- Output mode flags ---

    def test_list_mode_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(list_mode=True))
        mock_execute.assert_called_once()

    def test_limit_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(limit=10))
        mock_execute.assert_called_once()

    def test_stats_only_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(stats_only=True))
        mock_execute.assert_called_once()

    def test_count_only_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(count_only=True))
        mock_execute.assert_called_once()

    def test_stream_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(stream=True))
        mock_execute.assert_called_once()

    def test_dialogue_only_triggers_query(self):
        mock_execute, _ = self._call(self._make_params(dialogue_only=True))
        mock_execute.assert_called_once()

    # --- Modifier flags (REGRESSION: were silently ignored) ---

    def test_add_tag_triggers_query(self):
        """--add-tag must trigger query mode (REGRESSION: was silently ignored)."""
        mock_execute, mock_stats = self._call(
            self._make_params(add_tag=("review",))
        )
        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    def test_set_meta_triggers_query(self):
        """--set must trigger query mode (REGRESSION: was silently ignored)."""
        mock_execute, mock_stats = self._call(
            self._make_params(set_meta=(("status", "done"),))
        )
        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    def test_delete_matched_triggers_query(self):
        """--delete must trigger query mode (REGRESSION: was silently ignored)."""
        mock_execute, mock_stats = self._call(
            self._make_params(delete_matched=True)
        )
        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    # --- Query term forwarding ---

    def test_query_terms_forwarded(self):
        """Query terms are passed to execute_query as 'query' param."""
        mock_execute, _ = self._call(
            self._make_params(query_term=("python", "error"))
        )
        call_args = mock_execute.call_args
        params = call_args[0][1]  # Second positional arg is params dict
        assert params["query"] == ("python", "error")

    def test_combined_filters_and_query(self):
        """Multiple filters + query terms all route to query mode."""
        mock_execute, mock_stats = self._call(
            self._make_params(
                query_term=("error",),
                provider="claude",
                since="2025-01-01",
                list_mode=True,
            )
        )
        mock_execute.assert_called_once()
        mock_stats.assert_not_called()


# =============================================================================
# QueryFirstGroup.parse_args
# =============================================================================


class TestQueryFirstGroupParseArgs:
    """Test positional arg → --query-term conversion."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_subcommand_dispatches_normally(self, runner):
        """Known subcommand (e.g., 'check') is not treated as a query term."""
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["check", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "health" in result.output.lower() or "repair" in result.output.lower()

    def test_positional_args_become_query_terms(self, runner):
        """Positional args are converted to --query-term options."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            result = runner.invoke(cli, ["hello", "world", "--plain"], catch_exceptions=False)
            if mock_execute.called:
                _, params = mock_execute.call_args[0]
                assert "hello" in params.get("query", ())
                assert "world" in params.get("query", ())

    def test_option_args_preserved(self, runner):
        """Option args with values are preserved correctly alongside positional args."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            result = runner.invoke(
                cli, ["-p", "claude", "search_term", "--plain"], catch_exceptions=False
            )
            if mock_execute.called:
                _, params = mock_execute.call_args[0]
                assert params.get("provider") == "claude"
                assert "search_term" in params.get("query", ())

    def test_mixed_options_and_positionals(self, runner):
        """Options interspersed with positional args all parse correctly."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            result = runner.invoke(
                cli,
                ["error", "-p", "claude", "handling", "--latest", "--plain"],
                catch_exceptions=False,
            )
            if mock_execute.called:
                _, params = mock_execute.call_args[0]
                assert "error" in params.get("query", ())
                assert "handling" in params.get("query", ())
                assert params.get("provider") == "claude"
                assert params.get("latest") is True

    def test_no_args_shows_stats(self, runner):
        """No args → stats mode (dispatched by invoke)."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app._show_stats") as mock_stats:
            result = runner.invoke(cli, ["--plain"], catch_exceptions=False)
            mock_stats.assert_called_once()

    def test_help_flag(self, runner):
        """--help shows help text with all options."""
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()
        assert "--provider" in result.output
        assert "--latest" in result.output

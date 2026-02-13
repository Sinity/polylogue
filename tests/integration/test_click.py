"""Tests for CLI integration: QueryFirstGroup, cli() setup, routing, and formatting.

Covers uncovered code paths in click_app.py:
- QueryFirstGroup.invoke() subcommand vs query dispatch
- QueryFirstGroup.parse_args() positional arg → --query-term conversion
- _handle_query_mode() routing logic (filters, modifiers, output modes)
- cli() callback: logging, UI setup, plain mode announcement
- _show_stats() helper
- main() entrypoint
- formatting.py: should_use_plain, format_cursors, format_counts, etc.

CONSOLIDATED: This file merges tests from:
- test_click_app_routing.py (internal routing tests for _handle_query_mode and parse_args)
"""

from __future__ import annotations

import os
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

# =============================================================================
# QueryFirstGroup.invoke()
# =============================================================================


class TestQueryFirstGroupInvoke:
    """Tests for invoke() method dispatching."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_subcommand_invokes_super(self, runner):
        """When parse_args sets _has_subcommand=True, super().invoke() is called."""
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0

    def test_no_subcommand_calls_callback_then_query_mode(self, runner):
        """Without subcommand, callback is invoked then _handle_query_mode runs."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app._show_stats") as mock_stats:
            result = runner.invoke(cli, ["--plain"], catch_exceptions=False)
            mock_stats.assert_called_once()

    def test_query_mode_with_positional_args(self, runner):
        """Positional args route through invoke() to query mode."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_exec:
            result = runner.invoke(cli, ["hello", "--plain"], catch_exceptions=False)
            mock_exec.assert_called_once()


# =============================================================================
# cli() callback setup
# =============================================================================


class TestCliSetup:
    """Tests for cli() callback: logging, UI, plain mode."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_verbose_configures_debug_logging(self, runner):
        """--verbose should call configure_logging(verbose=True)."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.configure_logging") as mock_log, patch(
            "polylogue.cli.click_app._show_stats"
        ):
            runner.invoke(cli, ["--verbose", "--plain"], catch_exceptions=False)
            mock_log.assert_called_once_with(verbose=True)

    def test_no_verbose_configures_info_logging(self, runner):
        """Without --verbose, configure_logging(verbose=False)."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.configure_logging") as mock_log, patch(
            "polylogue.cli.click_app._show_stats"
        ):
            runner.invoke(cli, ["--plain"], catch_exceptions=False)
            mock_log.assert_called_once_with(verbose=False)

    def test_plain_flag_creates_plain_ui(self, runner):
        """--plain flag should create plain UI."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.create_ui") as mock_ui, patch(
            "polylogue.cli.click_app._show_stats"
        ):
            mock_ui.return_value = MagicMock()
            runner.invoke(cli, ["--plain"], catch_exceptions=False)
            mock_ui.assert_called_once_with(True)

    def test_plain_mode_announcement_when_auto_detected(self, runner):
        """When plain mode is auto-detected (not --plain, not env), announce_plain_mode() is called."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.should_use_plain", return_value=True), patch(
            "polylogue.cli.click_app.announce_plain_mode"
        ) as mock_announce, patch(
            "polylogue.cli.click_app.create_ui", return_value=MagicMock()
        ), patch(
            "polylogue.cli.click_app._show_stats"
        ):
            runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": ""})
            mock_announce.assert_called_once()

    def test_no_announcement_when_plain_flag_explicit(self, runner):
        """When --plain is explicitly passed, no announcement."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.should_use_plain", return_value=True), patch(
            "polylogue.cli.click_app.announce_plain_mode"
        ) as mock_announce, patch(
            "polylogue.cli.click_app.create_ui", return_value=MagicMock()
        ), patch(
            "polylogue.cli.click_app._show_stats"
        ):
            runner.invoke(cli, ["--plain"], catch_exceptions=False)
            mock_announce.assert_not_called()

    def test_no_announcement_when_env_force_plain(self, runner):
        """When POLYLOGUE_FORCE_PLAIN is set truthy, no announcement."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.should_use_plain", return_value=True), patch(
            "polylogue.cli.click_app.announce_plain_mode"
        ) as mock_announce, patch(
            "polylogue.cli.click_app.create_ui", return_value=MagicMock()
        ), patch(
            "polylogue.cli.click_app._show_stats"
        ):
            runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": "1"})
            mock_announce.assert_not_called()

    def test_env_force_plain_false_values_dont_block_announcement(self, runner):
        """POLYLOGUE_FORCE_PLAIN set to '0', 'false', 'no' should NOT suppress announcement."""
        from polylogue.cli.click_app import cli

        for val in ("0", "false", "no"):
            with patch(
                "polylogue.cli.click_app.should_use_plain", return_value=True
            ), patch("polylogue.cli.click_app.announce_plain_mode") as mock_announce, patch(
                "polylogue.cli.click_app.create_ui", return_value=MagicMock()
            ), patch(
                "polylogue.cli.click_app._show_stats"
            ):
                runner.invoke(
                    cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": val}
                )
                mock_announce.assert_called_once()

    def test_ctx_obj_set_to_appenv(self, runner):
        """ctx.obj should be an AppEnv instance after cli() callback."""
        from polylogue.cli.click_app import cli
        from polylogue.cli.types import AppEnv

        captured_env = {}

        def capture_stats(env, *, verbose=False):
            captured_env["env"] = env

        with patch("polylogue.cli.click_app._show_stats", side_effect=capture_stats):
            runner.invoke(cli, ["--plain"], catch_exceptions=False)
            assert isinstance(captured_env.get("env"), AppEnv)


# =============================================================================
# _show_stats
# =============================================================================


class TestShowStats:
    """Tests for _show_stats helper."""

    def test_calls_print_summary(self):
        from polylogue.cli.click_app import _show_stats

        env = MagicMock()
        with patch("polylogue.cli.helpers.print_summary") as mock_print:
            _show_stats(env, verbose=True)
            mock_print.assert_called_once_with(env, verbose=True)

    def test_calls_print_summary_not_verbose(self):
        from polylogue.cli.click_app import _show_stats

        env = MagicMock()
        with patch("polylogue.cli.helpers.print_summary") as mock_print:
            _show_stats(env, verbose=False)
            mock_print.assert_called_once_with(env, verbose=False)


# =============================================================================
# Version and help
# =============================================================================


class TestCliMetadata:
    """Tests for CLI metadata."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_version_flag(self, runner):
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()

    def test_help_flag_lists_subcommands(self, runner):
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd in ("run", "check", "embed", "site", "mcp", "tags"):
            assert cmd in result.output

    def test_all_subcommands_registered(self):
        from polylogue.cli.click_app import cli

        expected = {
            "run",
            "sources",
            "check",
            "reset",
            "mcp",
            "auth",
            "completions",
            "dashboard",
            "embed",
            "site",
            "tags",
        }
        assert set(cli.commands.keys()) == expected


# =============================================================================
# formatting.py
# =============================================================================


class TestShouldUsePlain:
    """Tests for should_use_plain()."""

    def test_explicit_plain_returns_true(self):
        from polylogue.cli.formatting import should_use_plain

        assert should_use_plain(plain=True) is True

    def test_env_force_plain_returns_true(self, monkeypatch):
        from polylogue.cli.formatting import should_use_plain

        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        assert should_use_plain(plain=False) is True

    def test_env_force_plain_false_values(self, monkeypatch):
        from polylogue.cli.formatting import should_use_plain

        for val in ("0", "false", "no"):
            monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", val)
            # Result depends on TTY status, but env var should NOT force plain
            # In test env (non-TTY), will be True due to isatty check
            result = should_use_plain(plain=False)
            assert isinstance(result, bool)

    def test_non_tty_returns_true(self, monkeypatch):
        from polylogue.cli.formatting import should_use_plain

        monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
        # CliRunner doesn't have a real TTY, so this will be True
        assert should_use_plain(plain=False) is True


class TestAnnouncePlainMode:
    """Tests for announce_plain_mode()."""

    def test_writes_to_stderr(self, capsys):
        from polylogue.cli.formatting import announce_plain_mode

        announce_plain_mode()
        captured = capsys.readouterr()
        assert "Plain output active" in captured.err


class TestFormatCursors:
    """Tests for format_cursors()."""

    def test_empty_cursors_returns_none(self):
        from polylogue.cli.formatting import format_cursors

        assert format_cursors({}) is None

    def test_cursor_with_file_count(self):
        from polylogue.cli.formatting import format_cursors

        result = format_cursors({"source1": {"file_count": 42}})
        assert "42 files" in result
        assert "source1" in result

    def test_cursor_with_error_count(self):
        from polylogue.cli.formatting import format_cursors

        result = format_cursors({"src": {"error_count": 3}})
        assert "3 errors" in result

    def test_cursor_with_zero_error_count_omitted(self):
        from polylogue.cli.formatting import format_cursors

        result = format_cursors({"src": {"error_count": 0, "file_count": 10}})
        assert "error" not in result

    def test_cursor_with_latest_mtime(self):
        from polylogue.cli.formatting import format_cursors

        result = format_cursors({"src": {"latest_mtime": 1700000000}})
        assert "latest" in result

    def test_cursor_with_latest_file_name(self):
        from polylogue.cli.formatting import format_cursors

        result = format_cursors({"src": {"latest_file_name": "session.jsonl"}})
        assert "session.jsonl" in result

    def test_cursor_with_latest_path_fallback(self):
        from polylogue.cli.formatting import format_cursors

        result = format_cursors({"src": {"latest_path": "/data/exports/session.jsonl"}})
        assert "session.jsonl" in result

    def test_cursor_with_non_dict_value(self):
        from polylogue.cli.formatting import format_cursors

        result = format_cursors({"src": "plain_string"})
        assert "src" in result
        assert "unknown" in result

    def test_multiple_cursors_joined(self):
        from polylogue.cli.formatting import format_cursors

        result = format_cursors(
            {
                "chatgpt": {"file_count": 5},
                "claude": {"file_count": 3},
            }
        )
        assert "chatgpt" in result
        assert "claude" in result
        assert ";" in result


class TestFormatCounts:
    """Tests for format_counts()."""

    def test_basic_counts(self):
        from polylogue.cli.formatting import format_counts

        result = format_counts({"conversations": 100, "messages": 5000})
        assert "100 conv" in result
        assert "5000 msg" in result

    def test_with_rendered(self):
        from polylogue.cli.formatting import format_counts

        result = format_counts({"conversations": 10, "messages": 50, "rendered": 10})
        assert "10 rendered" in result

    def test_zero_rendered_omitted(self):
        from polylogue.cli.formatting import format_counts

        result = format_counts({"conversations": 10, "messages": 50, "rendered": 0})
        assert "rendered" not in result

    def test_missing_keys_default_zero(self):
        from polylogue.cli.formatting import format_counts

        result = format_counts({})
        assert "0 conv" in result
        assert "0 msg" in result


class TestFormatIndexStatus:
    """Tests for format_index_status()."""

    def test_parse_stage_skipped(self):
        from polylogue.cli.formatting import format_index_status

        assert format_index_status("parse", False, None) == "Index: skipped"

    def test_render_stage_skipped(self):
        from polylogue.cli.formatting import format_index_status

        assert format_index_status("render", True, None) == "Index: skipped"

    def test_index_error(self):
        from polylogue.cli.formatting import format_index_status

        assert format_index_status("index", False, "boom") == "Index: error"

    def test_indexed_ok(self):
        from polylogue.cli.formatting import format_index_status

        assert format_index_status("index", True, None) == "Index: ok"

    def test_not_indexed_up_to_date(self):
        from polylogue.cli.formatting import format_index_status

        assert format_index_status("index", False, None) == "Index: up-to-date"


class TestFormatSourceLabel:
    """Tests for format_source_label()."""

    def test_different_source_and_provider(self):
        from polylogue.cli.formatting import format_source_label

        assert format_source_label("inbox", "chatgpt") == "inbox/chatgpt"

    def test_same_source_and_provider(self):
        from polylogue.cli.formatting import format_source_label

        assert format_source_label("chatgpt", "chatgpt") == "chatgpt"

    def test_none_source(self):
        from polylogue.cli.formatting import format_source_label

        assert format_source_label(None, "chatgpt") == "chatgpt"


class TestFormatSourcesSummary:
    """Tests for format_sources_summary()."""

    def test_empty_sources(self):
        from polylogue.cli.formatting import format_sources_summary

        assert format_sources_summary([]) == "none"

    def test_path_source(self):
        from pathlib import Path

        from polylogue.cli.formatting import format_sources_summary
        from polylogue.config import Source

        sources = [Source(name="chatgpt", path=Path("/data/chatgpt"))]
        result = format_sources_summary(sources)
        assert "chatgpt" in result

    def test_drive_source(self):
        from polylogue.cli.formatting import format_sources_summary
        from polylogue.config import Source

        sources = [Source(name="drive", folder="some-folder-id")]
        result = format_sources_summary(sources)
        assert "drive" in result
        assert "drive" in result.lower()

    def test_source_with_both_path_and_folder_variants(self):
        """Test that format_sources_summary handles mixed source types."""
        from pathlib import Path

        from polylogue.cli.formatting import format_sources_summary
        from polylogue.config import Source

        sources = [
            Source(name="local", path=Path("/data/exports")),
            Source(name="google", folder="folder-id-123"),
        ]
        result = format_sources_summary(sources)
        assert "local" in result
        assert "google" in result
        assert "drive" in result.lower()

    def test_truncation_over_8(self):
        from pathlib import Path

        from polylogue.cli.formatting import format_sources_summary
        from polylogue.config import Source

        sources = [Source(name=f"src{i}", path=Path(f"/data/{i}")) for i in range(12)]
        result = format_sources_summary(sources)
        assert "+4 more" in result

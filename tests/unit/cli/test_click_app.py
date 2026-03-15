"""Focused tests for click_app routing and setup internals."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestHandleQueryMode:
    def _make_params(self, **overrides):
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
            "filter_has_tool_use": False,
            "filter_has_thinking": False,
            "filter_has_file_ops": False,
            "filter_has_git_ops": False,
            "filter_has_subagent": False,
            "min_messages": None,
            "max_messages": None,
            "min_words": None,
        }
        defaults.update(overrides)
        return defaults

    def _call(self, params):
        from polylogue.cli.click_app import _handle_query_mode

        mock_ctx = MagicMock()
        mock_ctx.params = params
        mock_ctx.obj = MagicMock()

        with patch("polylogue.cli.query.execute_query") as mock_execute, patch(
            "polylogue.cli.click_app._show_stats"
        ) as mock_stats:
            _handle_query_mode(mock_ctx)
            return mock_execute, mock_stats

    def test_no_args_shows_stats(self):
        mock_execute, mock_stats = self._call(self._make_params())
        mock_stats.assert_called_once()
        mock_execute.assert_not_called()

    def test_verbose_stats(self):
        mock_execute, mock_stats = self._call(self._make_params(verbose=True))
        mock_stats.assert_called_once()
        mock_execute.assert_not_called()

    def test_query_terms_trigger_query(self):
        mock_execute, mock_stats = self._call(self._make_params(query_term=("error", "handling")))
        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    def test_filter_flags_trigger_query(self):
        for params in (
            self._make_params(conv_id="abc123"),
            self._make_params(provider="claude"),
            self._make_params(tag="important"),
            self._make_params(contains=("error",)),
            self._make_params(has_type=("thinking",)),
            self._make_params(since="2025-01-01"),
            self._make_params(until="2025-12-31"),
            self._make_params(latest=True),
            self._make_params(title="test"),
            self._make_params(exclude_text=("noise",)),
            self._make_params(exclude_provider="chatgpt"),
            self._make_params(exclude_tag="deprecated"),
            self._make_params(filter_has_tool_use=True),
            self._make_params(min_messages=10),
        ):
            mock_execute, mock_stats = self._call(params)
            mock_execute.assert_called_once()
            mock_stats.assert_not_called()

    def test_output_mode_flags_trigger_query(self):
        for params in (
            self._make_params(list_mode=True),
            self._make_params(limit=10),
            self._make_params(stats_only=True),
            self._make_params(count_only=True),
            self._make_params(stream=True),
            self._make_params(dialogue_only=True),
            self._make_params(stats_by="provider"),
        ):
            mock_execute, _ = self._call(params)
            mock_execute.assert_called_once()

    def test_modifier_flags_trigger_query(self):
        for params in (
            self._make_params(add_tag=("review",)),
            self._make_params(set_meta=(("status", "done"),)),
            self._make_params(delete_matched=True),
        ):
            mock_execute, mock_stats = self._call(params)
            mock_execute.assert_called_once()
            mock_stats.assert_not_called()

    def test_query_terms_forwarded(self):
        mock_execute, _ = self._call(self._make_params(query_term=("python", "error")))
        params = mock_execute.call_args[0][1]
        assert params["query"] == ("python", "error")


class TestQueryFirstGroupParseArgs:
    def test_subcommand_dispatches_normally(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["check", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "health" in result.output.lower() or "repair" in result.output.lower()

    def test_positional_args_become_query_terms(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            cli_runner.invoke(cli, ["hello", "world", "--plain"], catch_exceptions=False)
        _, params = mock_execute.call_args[0]
        assert set(params.get("query", ())) == {"hello", "world"}

    def test_option_args_preserved(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            cli_runner.invoke(cli, ["-p", "claude", "search_term", "--plain"], catch_exceptions=False)
        _, params = mock_execute.call_args[0]
        assert params.get("provider") == "claude"
        assert "search_term" in params.get("query", ())

    def test_mixed_options_and_positionals(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            cli_runner.invoke(
                cli,
                ["error", "-p", "claude", "handling", "--latest", "--plain"],
                catch_exceptions=False,
            )
        _, params = mock_execute.call_args[0]
        assert params.get("provider") == "claude"
        assert params.get("latest") is True
        assert set(params.get("query", ())) == {"error", "handling"}

    def test_no_args_shows_stats(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app._show_stats") as mock_stats:
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_stats.assert_called_once()

    def test_help_flag(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()
        assert "--provider" in result.output
        assert "--latest" in result.output


class TestQueryFirstGroupInvoke:
    def test_subcommand_invokes_super(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0

    def test_no_subcommand_calls_stats_path(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app._show_stats") as mock_stats:
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_stats.assert_called_once()

    def test_query_mode_with_positional_args(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_exec:
            cli_runner.invoke(cli, ["hello", "--plain"], catch_exceptions=False)
        mock_exec.assert_called_once()


class TestCliSetup:
    def test_verbose_configures_debug_logging(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.configure_logging") as mock_log, patch(
            "polylogue.cli.click_app._show_stats"
        ):
            cli_runner.invoke(cli, ["--verbose", "--plain"], catch_exceptions=False)
        mock_log.assert_called_once_with(verbose=True)

    def test_no_verbose_configures_info_logging(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.configure_logging") as mock_log, patch(
            "polylogue.cli.click_app._show_stats"
        ):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_log.assert_called_once_with(verbose=False)

    def test_plain_flag_creates_plain_ui(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.create_ui") as mock_ui, patch(
            "polylogue.cli.click_app._show_stats"
        ):
            mock_ui.return_value = MagicMock()
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_ui.assert_called_once_with(True)

    def test_plain_mode_announcement_when_auto_detected(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.should_use_plain", return_value=True), patch(
            "polylogue.cli.click_app.announce_plain_mode"
        ) as mock_announce, patch(
            "polylogue.cli.click_app.create_ui", return_value=MagicMock()
        ), patch("polylogue.cli.click_app._show_stats"):
            cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": ""})
        mock_announce.assert_called_once()

    def test_no_announcement_when_plain_flag_explicit(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.should_use_plain", return_value=True), patch(
            "polylogue.cli.click_app.announce_plain_mode"
        ) as mock_announce, patch(
            "polylogue.cli.click_app.create_ui", return_value=MagicMock()
        ), patch("polylogue.cli.click_app._show_stats"):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_announce.assert_not_called()

    def test_no_announcement_when_env_force_plain(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.should_use_plain", return_value=True), patch(
            "polylogue.cli.click_app.announce_plain_mode"
        ) as mock_announce, patch(
            "polylogue.cli.click_app.create_ui", return_value=MagicMock()
        ), patch("polylogue.cli.click_app._show_stats"):
            cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": "1"})
        mock_announce.assert_not_called()

    def test_env_force_plain_false_values_dont_block_announcement(self, cli_runner):
        from polylogue.cli.click_app import cli

        for value in ("0", "false", "no"):
            with patch("polylogue.cli.click_app.should_use_plain", return_value=True), patch(
                "polylogue.cli.click_app.announce_plain_mode"
            ) as mock_announce, patch(
                "polylogue.cli.click_app.create_ui", return_value=MagicMock()
            ), patch("polylogue.cli.click_app._show_stats"):
                cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": value})
            mock_announce.assert_called_once()

    def test_ctx_obj_set_to_appenv(self, cli_runner):
        from polylogue.cli.click_app import cli
        from polylogue.cli.types import AppEnv

        captured_env = {}

        def capture_stats(env, *, verbose=False):
            captured_env["env"] = env

        with patch("polylogue.cli.click_app._show_stats", side_effect=capture_stats):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        assert isinstance(captured_env.get("env"), AppEnv)


class TestShowStats:
    def test_calls_print_summary_verbose(self):
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


class TestCliMetadata:
    def test_version_flag(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()

    def test_help_flag_lists_subcommands(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for command in ("run", "check", "embed", "site", "mcp", "tags"):
            assert command in result.output

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
            "demo",
            "embed",
            "qa",
            "schema",
            "site",
            "tags",
        }
        assert set(cli.commands.keys()) == expected

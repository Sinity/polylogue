"""Focused tests for click_app routing and setup internals."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli
from tests.infra.cli_subprocess import run_cli

QueryParams = dict[str, object]
CliWorkspace = dict[str, Path]


class TestHandleQueryMode:
    def _make_params(self, **overrides: object) -> QueryParams:
        defaults: QueryParams = {
            "conv_id": None,
            "contains": (),
            "exclude_text": (),
            "retrieval_lane": None,
            "provider": None,
            "exclude_provider": None,
            "tag": None,
            "exclude_tag": None,
            "has_type": (),
            "since": None,
            "until": None,
            "title": None,
            "referenced_path": (),
            "action": (),
            "exclude_action": (),
            "action_sequence": None,
            "action_text": (),
            "tool": (),
            "exclude_tool": (),
            "similar_text": None,
            "latest": False,
            "limit": None,
            "sort": None,
            "reverse": False,
            "sample": None,
            "output": None,
            "output_format": None,
            "transform": None,
            "stream": False,
            "dialogue_only": False,
            "message_role": (),
            "set_meta": (),
            "add_tag": (),
            "tail": False,
            "plain": False,
            "verbose": False,
            "filter_has_tool_use": False,
            "filter_has_thinking": False,
            "min_messages": None,
            "max_messages": None,
            "min_words": None,
        }
        defaults.update(overrides)
        return defaults

    def _call(self, params: QueryParams) -> tuple[MagicMock, MagicMock]:
        from polylogue.cli.click_app import _handle_query_mode

        mock_ctx = MagicMock()
        mock_ctx.params = params
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {}

        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app._show_stats") as mock_stats,
        ):
            _handle_query_mode(mock_ctx)
            return mock_execute, mock_stats

    def test_no_args_shows_stats(self) -> None:
        mock_execute, mock_stats = self._call(self._make_params())
        mock_stats.assert_called_once()
        mock_execute.assert_not_called()

    def test_verbose_stats(self) -> None:
        mock_execute, mock_stats = self._call(self._make_params(verbose=True))
        mock_stats.assert_called_once()
        mock_execute.assert_not_called()

    def test_query_terms_trigger_query(self) -> None:
        mock_ctx = MagicMock()
        mock_ctx.params = self._make_params()
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {"polylogue_query_terms": ("error", "handling")}

        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app._show_stats") as mock_stats,
        ):
            from polylogue.cli.click_app import _handle_query_mode

            _handle_query_mode(mock_ctx)

        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    def test_filter_flags_trigger_query(self) -> None:
        for params in (
            self._make_params(conv_id="abc123"),
            self._make_params(provider="claude-ai"),
            self._make_params(tag="important"),
            self._make_params(contains=("error",)),
            self._make_params(has_type=("thinking",)),
            self._make_params(since="2025-01-01"),
            self._make_params(until="2025-12-31"),
            self._make_params(latest=True),
            self._make_params(title="test"),
            self._make_params(referenced_path=("/workspace/polylogue/README.md",)),
            self._make_params(action=("search",)),
            self._make_params(exclude_action=("git",)),
            self._make_params(action_sequence="file_read,file_edit,shell"),
            self._make_params(action_text=("pytest -q",)),
            self._make_params(retrieval_lane="actions", contains=("pytest",)),
            self._make_params(tool=("grep",)),
            self._make_params(exclude_tool=("bash",)),
            self._make_params(similar_text="sqlite locking bug"),
            self._make_params(exclude_text=("noise",)),
            self._make_params(exclude_provider="chatgpt"),
            self._make_params(exclude_tag="deprecated"),
            self._make_params(filter_has_tool_use=True),
            self._make_params(min_messages=10),
        ):
            mock_execute, mock_stats = self._call(params)
            mock_execute.assert_called_once()
            mock_stats.assert_not_called()

    def test_output_mode_flags_trigger_query(self) -> None:
        for params in (
            self._make_params(limit=10),
            self._make_params(stream=True),
            self._make_params(dialogue_only=True),
            self._make_params(message_role=("user",)),
        ):
            mock_execute, _ = self._call(params)
            mock_execute.assert_called_once()

    def test_modifier_flags_trigger_query(self) -> None:
        for params in (
            self._make_params(add_tag=("review",)),
            self._make_params(set_meta=(("status", "done"),)),
        ):
            mock_execute, mock_stats = self._call(params)
            mock_execute.assert_called_once()
            mock_stats.assert_not_called()

    def test_query_terms_forwarded(self) -> None:
        mock_ctx = MagicMock()
        mock_ctx.params = self._make_params()
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {"polylogue_query_terms": ("python", "error")}

        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app._show_stats"),
        ):
            from polylogue.cli.click_app import _handle_query_mode

            _handle_query_mode(mock_ctx)

        request = mock_execute.call_args[0][1]
        assert request.query_params()["query"] == ("python", "error")


def test_messages_verb_forwards_parent_request_and_projection_options(cli_runner: CliRunner) -> None:
    with patch("polylogue.cli.messages.run_messages") as mock_run_messages:
        result = cli_runner.invoke(
            click_cli,
            [
                "--plain",
                "messages",
                "conv-1",
                "--message-role",
                "user",
                "--message-type",
                "summary",
                "--limit",
                "2",
                "--offset",
                "1",
                "--no-code-blocks",
                "--no-tool-calls",
                "--no-tool-outputs",
                "--no-file-reads",
                "--prose-only",
                "-f",
                "json",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert mock_run_messages.call_args.kwargs == {
        "conversation_id": "conv-1",
        "message_role": ("user",),
        "message_type": "summary",
        "limit": 2,
        "offset": 1,
        "no_code_blocks": True,
        "no_tool_calls": True,
        "no_tool_outputs": True,
        "no_file_reads": True,
        "prose_only": True,
        "output_format": "json",
    }


def test_messages_verb_requires_id(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(click_cli, ["--plain", "messages"])

    assert result.exit_code != 0
    assert "messages requires a conversation ID" in result.output


def test_raw_verb_forwards_pagination_and_format(cli_runner: CliRunner) -> None:
    with patch("polylogue.cli.messages.run_raw") as mock_run_raw:
        result = cli_runner.invoke(
            click_cli,
            ["--plain", "raw", "conv-1", "--limit", "3", "--offset", "2", "-f", "yaml"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert mock_run_raw.call_args.kwargs == {
        "conversation_id": "conv-1",
        "limit": 3,
        "offset": 2,
        "output_format": "yaml",
    }


def test_raw_verb_requires_id(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(click_cli, ["--plain", "raw"])

    assert result.exit_code != 0
    assert "raw requires a conversation ID" in result.output


class TestQueryFirstGroupParseArgs:
    def test_subcommand_dispatches_normally(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["doctor", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "health" in result.output.lower() or "repair" in result.output.lower()

    def test_positional_args_become_query_terms(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            cli_runner.invoke(cli, ["hello", "world", "--plain"], catch_exceptions=False)
        request = mock_execute.call_args[0][1]
        assert set(request.query_params().get("query", ())) == {"hello", "world"}

    def test_query_option_before_bare_word_stays_query_mode(self, cli_runner: CliRunner) -> None:
        """Filter options followed by a bare word (not a subcommand name) stay in query mode."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            cli_runner.invoke(cli, ["-p", "claude-ai", "my_search", "--plain"], catch_exceptions=False)
        params = mock_execute.call_args[0][1].query_params()
        assert params.get("provider") == "claude-ai"
        assert params.get("query") == ("my_search",)

    def test_filter_option_before_subcommand_routes_to_subcommand(self, cli_runner: CliRunner) -> None:
        """Filter options followed by a known subcommand route to that subcommand."""
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--plain", "-p", "claude-ai", "insights", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "insights" in result.output.lower()

    def test_option_args_preserved(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            cli_runner.invoke(cli, ["-p", "claude-ai", "search_term", "--plain"], catch_exceptions=False)
        params = mock_execute.call_args[0][1].query_params()
        assert params.get("provider") == "claude-ai"
        assert "search_term" in params.get("query", ())

    def test_mixed_options_and_positionals(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            cli_runner.invoke(
                cli,
                ["error", "-p", "claude-ai", "handling", "--latest", "--plain"],
                catch_exceptions=False,
            )
        params = mock_execute.call_args[0][1].query_params()
        assert params.get("provider") == "claude-ai"
        assert params.get("latest") is True
        assert set(params.get("query", ())) == {"error", "handling"}

    def test_no_args_shows_stats(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app._show_stats") as mock_stats:
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_stats.assert_called_once()

    def test_help_flag(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()
        assert "insights" in result.output
        assert "--provider" in result.output
        assert "--latest" in result.output
        assert "--tail" in result.output
        assert "Subcommands:" not in result.output
        assert "polylogue --provider claude-code --since 2026-01-01 stats --by repo --format json" in result.output
        assert "polylogue --tail --provider claude-code --latest list" in result.output
        assert "polylogue stats --by repo --provider claude-code --since 2026-01-01 --format json" not in result.output

    def test_root_query_option_after_verb_gets_specific_usage_error(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["stats", "--by", "provider", "--since", "2026-01-01"], catch_exceptions=False)
        assert result.exit_code == 2
        assert "Query filters and root output flags must appear before the verb." in result.output
        assert "Move --since before `stats`." in result.output

    def test_root_filter_after_verb_gets_specific_usage_error(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(
            cli, ["stats", "--by", "provider", "--provider", "claude-ai"], catch_exceptions=False
        )
        assert result.exit_code == 2
        assert "Query filters and root output flags must appear before the verb." in result.output
        assert "Move --provider before `stats`." in result.output


class TestQueryFirstGroupInvoke:
    def test_subcommand_invokes_super(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0

    def test_no_subcommand_calls_stats_path(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app._show_stats") as mock_stats:
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_stats.assert_called_once()

    def test_stats_by_subcommand_preserves_grouped_stats_mode(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            result = cli_runner.invoke(cli, ["--plain", "stats", "--by", "provider"], catch_exceptions=False)

        assert result.exit_code == 0
        params = mock_execute.call_args[0][1].query_params()
        assert params["stats_by"] == "provider"
        assert params["stats_only"] is False

    def test_query_mode_with_positional_args(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            cli_runner.invoke(cli, ["hello", "--plain"], catch_exceptions=False)
        mock_exec.assert_called_once()


class TestCliSetup:
    def test_verbose_configures_debug_logging(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.configure_logging") as mock_log,
            patch("polylogue.cli.click_app._show_stats"),
        ):
            cli_runner.invoke(cli, ["--verbose", "--plain"], catch_exceptions=False)
        mock_log.assert_called_once_with(verbose=True)

    def test_no_verbose_configures_info_logging(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.configure_logging") as mock_log,
            patch("polylogue.cli.click_app._show_stats"),
        ):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_log.assert_called_once_with(verbose=False)

    def test_plain_flag_creates_plain_ui(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.create_ui") as mock_ui, patch("polylogue.cli.click_app._show_stats"):
            mock_ui.return_value = MagicMock()
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_ui.assert_called_once_with(True)

    def test_plain_mode_auto_detection_does_not_announce(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": ""})
        assert "Plain output active" not in result.output

    def test_no_announcement_when_plain_flag_explicit(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        assert "Plain output active" not in result.output

    def test_no_announcement_when_env_force_plain(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": "1"})
        assert "Plain output active" not in result.output

    def test_no_announcement_when_json_requested(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, ["list", "--format", "json"], catch_exceptions=False)
        assert "Plain output active" not in result.output

    def test_env_force_plain_false_values_still_do_not_announce(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        for value in ("0", "false", "no"):
            with (
                patch("polylogue.cli.click_app.should_use_plain", return_value=True),
                patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
                patch("polylogue.cli.click_app._show_stats"),
            ):
                result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": value})
            assert "Plain output active" not in result.output

    def test_ctx_obj_set_to_appenv(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli
        from polylogue.cli.shared.types import AppEnv

        captured_env: dict[str, object] = {}

        def capture_stats(env: object, *, verbose: bool = False) -> None:
            captured_env["env"] = env

        with patch("polylogue.cli.click_app._show_stats", side_effect=capture_stats):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        assert isinstance(captured_env.get("env"), AppEnv)


class TestShowStats:
    def test_calls_print_summary_verbose(self) -> None:
        from polylogue.cli.click_app import _show_stats

        env = MagicMock()
        with patch("polylogue.cli.shared.helpers.print_summary") as mock_print:
            _show_stats(env, verbose=True)
        mock_print.assert_called_once_with(env, verbose=True)

    def test_calls_print_summary_not_verbose(self) -> None:
        from polylogue.cli.click_app import _show_stats

        env = MagicMock()
        with patch("polylogue.cli.shared.helpers.print_summary") as mock_print:
            _show_stats(env, verbose=False)
        mock_print.assert_called_once_with(env, verbose=False)


class TestCliMetadata:
    def test_version_flag(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()

    def test_help_flag_lists_subcommands(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for command in ("run", "doctor", "tags", "list", "count", "stats"):
            assert command in result.output
        assert "mcp" not in click_cli.commands
        assert "watch" not in click_cli.commands
        assert "browser-capture" not in click_cli.commands
        assert result.output.count("Commands:") == 1

    def test_all_subcommands_registered(self) -> None:
        from polylogue.cli.click_app import cli

        expected = {
            "run",
            "doctor",
            "reset",
            "auth",
            "completions",
            "dashboard",
            "neighbors",
            "export",
            "resume",
            "insights",
            "schema",
            "tags",
            "diagnostics",
            # Query verbs
            "list",
            "count",
            "stats",
            "open",
            "show",
            "bulk-export",
            "delete",
            "messages",
            "raw",
        }
        assert set(cli.commands.keys()) == expected


# ---------------------------------------------------------------------------
# Merged from test_cli_subprocess.py (2026-03-15)
# ---------------------------------------------------------------------------


def test_run_cli_honors_explicit_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUTANT_UNDER_TEST", raising=False)
    monkeypatch.delenv("PY_IGNORE_IMPORTMISMATCH", raising=False)
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"], cwd=tmp_path)

    command = mock_run.call_args.args[0]
    kwargs = mock_run.call_args.kwargs

    assert kwargs["cwd"] == tmp_path
    assert command[:4] == ["uv", "run", "--project", str(Path(__file__).parents[3])]


def test_run_cli_defaults_cwd_to_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUTANT_UNDER_TEST", raising=False)
    monkeypatch.delenv("PY_IGNORE_IMPORTMISMATCH", raising=False)
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"])

    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == Path(__file__).parents[3]


def test_run_cli_uses_python_bootstrap_under_mutmut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    monkeypatch.setenv("PY_IGNORE_IMPORTMISMATCH", "1")
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"], cwd=tmp_path)

    command = mock_run.call_args.args[0]
    kwargs = mock_run.call_args.kwargs
    env = kwargs["env"]

    assert kwargs["cwd"] == tmp_path
    assert "python" in Path(command[0]).name
    assert command[1] == "-c"
    assert "ensure_config_loaded" in command[2]
    assert command[3:] == ["--help"]
    assert env["MUTANT_UNDER_TEST"] == "stats"
    assert env["PY_IGNORE_IMPORTMISMATCH"] == "1"


# ---------------------------------------------------------------------------
# Merged from test_command_surfaces.py (2026-03-15)
# ---------------------------------------------------------------------------


class TestDashboardCommand:
    def test_dashboard_launches_app(self, cli_runner: CliRunner, cli_workspace: CliWorkspace) -> None:
        with patch("polylogue.ui.tui.app.PolylogueApp") as mock_app_cls:
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            result = cli_runner.invoke(click_cli, ["--plain", "dashboard"])
        assert result.exit_code == 0
        mock_app.run.assert_called_once()

    def test_dashboard_creates_app_with_repository(
        self,
        cli_runner: CliRunner,
        cli_workspace: CliWorkspace,
    ) -> None:
        with patch("polylogue.ui.tui.app.PolylogueApp") as mock_app_cls:
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            result = cli_runner.invoke(click_cli, ["--plain", "dashboard"])
        assert result.exit_code == 0
        kwargs = mock_app_cls.call_args.kwargs
        assert kwargs["repository"] is not None


class TestCompletionsCommand:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_completion_generates_script(self, cli_runner: CliRunner, shell: str) -> None:
        result = cli_runner.invoke(click_cli, ["completions", "--shell", shell])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower() or "complete" in result.output.lower()

    def test_shell_option_is_required(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(click_cli, ["completions"])
        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "required" in result.output.lower()

    def test_invalid_shell_rejected(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "powershell"])
        assert result.exit_code != 0
        assert "invalid value" in result.output.lower() or "choice" in result.output.lower()


class TestMcpServerImport:
    def test_serve_stdio_can_be_imported(self) -> None:
        try:
            from polylogue.mcp.server import serve_stdio

            assert callable(serve_stdio)
        except ImportError:
            pytest.skip("MCP dependencies not installed")

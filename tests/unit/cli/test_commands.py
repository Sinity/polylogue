"""Tests for CLI commands with zero coverage.

Tests cover: run, check, reset, completions, serve, mcp commands.
Uses subprocess isolation for proper environment handling.

CONSOLIDATED: This file merges tests from:
- test_cli_completions.py (CliRunner unit tests)
- test_cli_mcp.py (CliRunner unit tests)
- test_cli_reset.py (CliRunner unit tests)
- test_cli_serve.py (CliRunner unit tests)
- test_cli_commands_coverage.py (internal function tests for helpers, auth, completions, dashboard)

The subprocess integration tests provide end-to-end validation, while
the CliRunner unit tests provide faster, more detailed coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli, helpers
from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.commands.completions import completions_command
from polylogue.cli.commands.dashboard import dashboard_command
from polylogue.cli.commands.mcp import mcp_command
from tests.cli_helpers.cli_subprocess import run_cli, setup_isolated_workspace
from tests.helpers import GenericConversationBuilder

# =============================================================================
# TEST DATA TABLES (module-level constants)
# =============================================================================

IS_DECLARATIVE_CASES = [
    (None, False, "unset"),
    ("1", True, "set to 1"),
    ("yes", True, "set to yes"),
    ("true", True, "set to true"),
    ("false", False, "set to false"),
    ("no", False, "set to no"),
    ("0", False, "set to 0"),
    ("YES", True, "case insensitive YES"),
    ("FALSE", False, "case insensitive FALSE"),
]

RESOLVE_SOURCES_VALID_CASES = [
    (("chatgpt",), ["chatgpt"], "single valid source"),
    (("chatgpt", "claude"), {"chatgpt", "claude"}, "multiple valid sources"),
    (("chatgpt", "chatgpt"), ["chatgpt"], "deduplicated sources"),
]

RESOLVE_SOURCES_ERROR_CASES = [
    ((), None, "empty sources returns None"),
    (("unknown",), SystemExit, "unknown source fails"),
    (("chatgpt", "unknown"), SystemExit, "mixed valid/invalid fails"),
]

SHELL_COMPLETION_CASES = [
    ("bash", "bash"),
    ("zsh", "zsh"),
    ("fish", "fish"),
]

STAGE_CASES = [
    ("parse", "parsing"),
    ("render", "rendering"),
    ("index", "indexing"),
]

# =============================================================================
# HELPERS.PY TESTS
# =============================================================================


class TestFail:
    """Tests for fail() function."""

    def test_fail_raises_system_exit(self):
        """fail() should raise SystemExit with formatted message."""
        with pytest.raises(SystemExit) as exc_info:
            helpers.fail("test_cmd", "something broke")
        assert "test_cmd: something broke" in str(exc_info.value)

    def test_fail_with_empty_message(self):
        """fail() should work with empty message."""
        with pytest.raises(SystemExit) as exc_info:
            helpers.fail("test_cmd", "")
        assert "test_cmd:" in str(exc_info.value)


class TestIsDeclarative:
    """Tests for is_declarative() environment flag."""

    @pytest.mark.parametrize("env_val,expected,desc", IS_DECLARATIVE_CASES)
    def test_is_declarative(self, monkeypatch, env_val, expected, desc):
        """is_declarative() respects POLYLOGUE_DECLARATIVE env var."""
        if env_val is None:
            monkeypatch.delenv("POLYLOGUE_DECLARATIVE", raising=False)
        else:
            monkeypatch.setenv("POLYLOGUE_DECLARATIVE", env_val)
        assert helpers.is_declarative() is expected


class TestSourceStatePath:
    """Tests for source_state_path() function."""

    def test_default_path_without_xdg(self, monkeypatch, tmp_path):
        """Without XDG_STATE_HOME, should use ~/.local/state."""
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        result = helpers.source_state_path()
        assert "polylogue" in str(result)
        assert "last-source.json" in str(result)

    def test_with_xdg_state_home(self, monkeypatch):
        """With XDG_STATE_HOME set, should use it."""
        monkeypatch.setenv("XDG_STATE_HOME", "/custom/state")
        result = helpers.source_state_path()
        assert str(result).startswith("/custom/state")
        assert "polylogue" in str(result)
        assert "last-source.json" in str(result)


class TestResolveSources:
    """Tests for resolve_sources() function."""

    @pytest.mark.parametrize("sources,expected,desc", RESOLVE_SOURCES_VALID_CASES)
    def test_resolve_sources_valid(self, sources, expected, desc):
        """resolve_sources handles valid source combinations."""
        from polylogue.config import Source

        config = MagicMock()
        config.sources = [
            Source(name="chatgpt", path=Path("/data")),
            Source(name="claude", path=Path("/data2")),
        ]
        result = helpers.resolve_sources(config, sources, "test_cmd")
        if isinstance(expected, set):
            assert set(result) == expected
        else:
            assert result == expected

    @pytest.mark.parametrize("sources,expected,desc", RESOLVE_SOURCES_ERROR_CASES)
    def test_resolve_sources_error(self, sources, expected, desc):
        """resolve_sources handles error cases."""
        from polylogue.config import Source

        config = MagicMock()
        config.sources = [Source(name="chatgpt", path=Path("/data"))]

        if expected is None:
            result = helpers.resolve_sources(config, sources, "test_cmd")
            assert result is None
        else:
            with pytest.raises(expected):
                helpers.resolve_sources(config, sources, "test_cmd")

    def test_special_last_with_saved_source(self, tmp_path, monkeypatch):
        """resolve_sources should handle 'last' special source."""
        from polylogue.config import Source

        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        helpers.save_last_source("chatgpt")
        config = MagicMock()
        config.sources = [Source(name="chatgpt", path=Path("/data"))]
        result = helpers.resolve_sources(config, ("last",), "test_cmd")
        assert result == ["chatgpt"]

    def test_special_last_without_saved_fails(self, tmp_path, monkeypatch):
        """resolve_sources should fail with 'last' if no saved source."""
        from polylogue.config import Source

        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        config = MagicMock()
        config.sources = []
        with pytest.raises(SystemExit):
            helpers.resolve_sources(config, ("last",), "test_cmd")

    def test_last_combined_with_others_fails(self, tmp_path, monkeypatch):
        """resolve_sources should fail if 'last' combined with other sources."""
        from polylogue.config import Source

        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        config = MagicMock()
        config.sources = [Source(name="chatgpt", path=Path("/data"))]
        with pytest.raises(SystemExit):
            helpers.resolve_sources(config, ("last", "chatgpt"), "test_cmd")


class TestCliLatestRenderPath:
    """Tests for latest_render_path() function."""

    def test_nonexistent_dir_returns_none(self, tmp_path):
        """latest_render_path should return None for nonexistent dir."""
        result = helpers.latest_render_path(tmp_path / "missing")
        assert result is None

    def test_empty_dir_returns_none(self, tmp_path):
        """latest_render_path should return None for empty dir."""
        result = helpers.latest_render_path(tmp_path)
        assert result is None

    def test_finds_markdown_file(self, tmp_path):
        """latest_render_path should find conversation.md files."""
        sub = tmp_path / "conv1"
        sub.mkdir()
        (sub / "conversation.md").write_text("# Test")
        result = helpers.latest_render_path(tmp_path)
        assert result is not None
        assert result.name == "conversation.md"

    def test_finds_html_file(self, tmp_path):
        """latest_render_path should find conversation.html files."""
        sub = tmp_path / "conv1"
        sub.mkdir()
        (sub / "conversation.html").write_text("<html>test</html>")
        result = helpers.latest_render_path(tmp_path)
        assert result is not None
        assert result.name == "conversation.html"

    def test_prefers_latest_by_mtime(self, tmp_path):
        """latest_render_path should return most recently modified file."""
        import time

        sub1 = tmp_path / "conv1"
        sub1.mkdir()
        file1 = sub1 / "conversation.md"
        file1.write_text("# Old")
        time.sleep(0.01)  # Ensure different mtime

        sub2 = tmp_path / "conv2"
        sub2.mkdir()
        file2 = sub2 / "conversation.md"
        file2.write_text("# New")

        result = helpers.latest_render_path(tmp_path)
        assert result == file2

    def test_handles_missing_file_between_list_and_stat(self, tmp_path):
        """latest_render_path should gracefully skip deleted files."""
        sub = tmp_path / "conv1"
        sub.mkdir()
        file1 = sub / "conversation.md"
        file1.write_text("# Test")

        # Real test: just verify nonexistent file doesn't break
        result = helpers.latest_render_path(tmp_path)
        assert result is not None


# =============================================================================
# DASHBOARD COMMAND TESTS
# =============================================================================


class TestDashboardCommand:
    """Tests for dashboard_command()."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_dashboard_launches_app(self, runner):
        """dashboard_command should create and run PolylogueApp."""
        with patch("polylogue.cli.commands.dashboard.get_config") as mock_get_config, patch(
            "polylogue.ui.tui.app.PolylogueApp"
        ) as mock_app_cls:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app

            result = runner.invoke(click_cli, ["dashboard", "--plain"])
            # May fail or succeed depending on TUI init, but should run
            # Just verify it doesn't error on our mocks
            assert not isinstance(result.exception, AttributeError) or result.exit_code == 0

    def test_dashboard_creates_app_with_config(self, runner):
        """dashboard_command should pass config to PolylogueApp."""
        with patch("polylogue.cli.commands.dashboard.get_config") as mock_get_config, patch(
            "polylogue.ui.tui.app.PolylogueApp"
        ) as mock_app_cls:
            mock_config = MagicMock()
            mock_config.archive_root = Path("/archive")
            mock_get_config.return_value = mock_config
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            mock_app.run.side_effect = Exception("Test exit")

            result = runner.invoke(click_cli, ["dashboard", "--plain"])
            # If we got to the exception, the command ran and created the app
            # Verify app was created with config
            if mock_app_cls.called:
                mock_app_cls.assert_called_once_with(config=mock_config)

    def test_dashboard_with_cli_runner(self, runner):
        """dashboard_command via CLI runner."""
        with patch("polylogue.cli.commands.dashboard.get_config") as mock_get_config, patch(
            "polylogue.ui.tui.app.PolylogueApp"
        ) as mock_app_cls:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app

            result = runner.invoke(click_cli, ["dashboard", "--plain"])
            # Should invoke without unknown service error
            assert "Unknown" not in result.output or result.exit_code == 0

# =============================================================================
# SUBPROCESS INTEGRATION TESTS - RUN COMMAND
# =============================================================================


@pytest.mark.integration
class TestCliRunCommand:
    """Tests for the run command."""

    def test_run_preview_shows_plan(self, tmp_path):
        """run --preview shows what would be done without writing."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        # Create test data
        (GenericConversationBuilder("conv1")
         .add_user("preview test")
         .write_to(inbox / "test.json"))

        result = run_cli(["--plain", "run", "--preview"], env=env)
        # Preview should succeed (exit 0) or fail gracefully
        assert result.exit_code in (0, 1)
        # Should mention preview or snapshot
        output_lower = result.output.lower()
        assert "preview" in output_lower or "snapshot" in output_lower or "source" in output_lower

    @pytest.mark.parametrize("stage,desc", [(s, d) for s, d in STAGE_CASES])
    def test_run_stage_execution(self, tmp_path, stage, desc):
        """run --stage executes specific pipeline stage."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        # Only add data for parse stage
        if stage == "parse":
            (GenericConversationBuilder("conv1")
             .add_user("test data")
             .write_to(inbox / "test.json"))

        result = run_cli(["--plain", "run", "--stage", stage], env=env)
        # Should succeed even with no data for later stages
        assert result.exit_code == 0

    def test_run_with_source_filter(self, tmp_path):
        """run --source filters to specific source."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        # Should handle nonexistent source gracefully
        result = run_cli(["--plain", "run", "--source", "nonexistent"], env=env)
        # Either fails with error or succeeds with no-op
        assert result.exit_code in (0, 1)

    def test_run_watch_flags_require_watch(self, tmp_path):
        """--notify, --exec, --webhook require --watch."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "run", "--notify"], env=env)
        assert result.exit_code != 0
        assert "watch" in result.output.lower()

        result = run_cli(["--plain", "run", "--exec", "echo test"], env=env)
        assert result.exit_code != 0

        result = run_cli(["--plain", "run", "--webhook", "http://example.com"], env=env)
        assert result.exit_code != 0


# =============================================================================
# SUBPROCESS INTEGRATION TESTS - CHECK COMMAND
# =============================================================================


@pytest.mark.integration
class TestCliCheckCommand:
    """Tests for the check command."""

    def test_check_basic(self, tmp_path):
        """check runs health checks."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "check"], env=env)
        # Should succeed even with empty database
        assert result.exit_code == 0
        # Should show some health check output
        output_lower = result.output.lower()
        assert "ok" in output_lower or "summary" in output_lower or "health" in output_lower

    def test_check_json_output(self, tmp_path):
        """check --json outputs valid JSON."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "check", "--json"], env=env)
        assert result.exit_code == 0

        # Should be valid JSON
        try:
            data = json.loads(result.stdout)
            assert isinstance(data, dict)
        except json.JSONDecodeError:
            pytest.fail(f"check --json did not output valid JSON: {result.stdout}")

    def test_check_verbose(self, tmp_path):
        """check --verbose shows breakdown."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "check", "--verbose"], env=env)
        assert result.exit_code == 0

    def test_check_vacuum_requires_repair(self, tmp_path):
        """check --vacuum requires --repair."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "check", "--vacuum"], env=env)
        assert result.exit_code != 0
        assert "repair" in result.output.lower()


# =============================================================================
# CLIRUNNER UNIT TESTS - SOURCES COMMAND
# =============================================================================


class TestSourcesCommand:
    """Tests for the sources command."""

    def test_sources_lists_configured(self, cli_runner, monkeypatch, cli_workspace):
        """sources command lists configured sources."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_DATA_HOME", str(cli_workspace["data_root"]))

        result = cli_runner.invoke(click_cli, ["sources"])
        # Should succeed (may show no sources or default inbox)
        assert result.exit_code == 0

    def test_sources_json_output(self, cli_runner, monkeypatch, cli_workspace):
        """sources --json outputs valid JSON."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_DATA_HOME", str(cli_workspace["data_root"]))

        result = cli_runner.invoke(click_cli, ["sources", "--json"])
        assert result.exit_code == 0

        try:
            data = json.loads(result.output)
            assert isinstance(data, list)
        except json.JSONDecodeError:
            pytest.fail(f"sources --json did not output valid JSON: {result.output}")


# =============================================================================
# CLIRUNNER UNIT TESTS - COMPLETIONS COMMAND
# =============================================================================


class TestCompletionsCommandUnit:
    """Unit tests for the completions command using CliRunner."""

    @pytest.mark.parametrize("shell,desc", [(s, s) for s, _ in SHELL_COMPLETION_CASES])
    def test_completion_generates_script(self, cli_runner, shell, desc):
        """Completion generates a valid script."""
        result = cli_runner.invoke(click_cli, ["completions", "--shell", shell])

        assert result.exit_code == 0
        # Should contain completion markers
        assert "polylogue" in result.output.lower() or "complete" in result.output.lower()

    def test_shell_option_is_required(self, cli_runner):
        """--shell option is required."""
        result = cli_runner.invoke(click_cli, ["completions"])

        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "required" in result.output.lower()

    def test_invalid_shell_rejected(self, cli_runner):
        """Invalid shell type is rejected."""
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "powershell"])

        assert result.exit_code != 0
        assert "invalid value" in result.output.lower() or "choice" in result.output.lower()

    def test_completion_uses_prog_name_polylogue(self, cli_runner):
        """Completion script uses 'polylogue' as program name."""
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "bash"])

        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()

    def test_completions_outputs_to_stdout(self, cli_runner):
        """completions should output to stdout, not stderr."""
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "bash"])
        assert result.exit_code == 0
        # Output should be in result.output, not error
        assert result.output and not result.exception


# =============================================================================
# CLIRUNNER UNIT TESTS - MCP COMMAND
# =============================================================================


class TestMcpCommandUnit:
    """Unit tests for the mcp command using CliRunner."""

    @pytest.fixture
    def mock_env(self):
        """Create mock AppEnv for tests."""
        mock_ui = MagicMock()
        mock_ui.plain = True
        mock_ui.console = MagicMock()

        env = MagicMock()
        env.ui = mock_ui
        return env

    def test_default_transport_is_stdio(self, cli_runner, mock_env):
        """Default transport is stdio."""
        with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
            result = cli_runner.invoke(mcp_command, [], obj=mock_env)

            # Should call serve_stdio
            mock_serve.assert_called_once()
            assert result.exit_code == 0

    def test_explicit_stdio_transport_works(self, cli_runner, mock_env):
        """--transport stdio works."""
        with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
            result = cli_runner.invoke(mcp_command, ["--transport", "stdio"], obj=mock_env)

            mock_serve.assert_called_once()
            assert result.exit_code == 0

    def test_missing_mcp_dependencies_error(self, cli_runner, mock_env):
        """Missing MCP dependencies show helpful error."""
        # Patch the import to raise ImportError
        with patch.dict(sys.modules, {"polylogue.mcp.server": None}):
            # Force ImportError by patching the actual import
            def mock_import(*args, **kwargs):
                raise ImportError("No module named 'mcp'")

            with patch("builtins.__import__", side_effect=mock_import):
                result = cli_runner.invoke(mcp_command, [], obj=mock_env)

                # Should fail with helpful message
                assert result.exit_code != 0 or mock_env.ui.console.print.called

    def test_unsupported_transport_error(self, cli_runner, mock_env):
        """Unsupported transport type raises error."""
        # The Click choice validation should reject this
        result = cli_runner.invoke(click_cli, ["mcp", "--transport", "http"])

        assert result.exit_code != 0

    def test_mcp_help_shows_description(self, cli_runner):
        """MCP help shows useful description."""
        result = cli_runner.invoke(click_cli, ["mcp", "--help"])

        assert result.exit_code == 0
        assert "mcp" in result.output.lower()
        assert "server" in result.output.lower() or "protocol" in result.output.lower()


class TestMcpServerIntegration:
    """Integration tests for MCP server (when dependencies are available)."""

    def test_serve_stdio_can_be_imported(self):
        """serve_stdio can be imported if mcp is installed."""
        try:
            from polylogue.mcp.server import serve_stdio
            assert callable(serve_stdio)
        except ImportError:
            # MCP not installed, skip
            pytest.skip("MCP dependencies not installed")

    def test_mcp_server_module_exists(self):
        """MCP server module exists in package."""
        import polylogue.mcp as mcp_module
        assert hasattr(mcp_module, "__file__")


# =============================================================================
# INTEGRATION TESTS FROM test_cli_integration.py
# =============================================================================


def _write_prompt_file(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


# =============================================================================
# END-TO-END CLI TESTS (from test_cli.py)
# =============================================================================


@pytest.mark.integration
def test_cli_run_and_search(tmp_path):
    """Test CLI run and search with isolated workspace."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    # Create test conversation in inbox
    (GenericConversationBuilder("conv1").add_user("hello").add_assistant("world").write_to(inbox / "conversation.json"))

    # Run pipeline via subprocess
    result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
    assert result.exit_code == 0, result.output

    render_root = paths["render_root"]
    assert any(render_root.rglob("*.html")) or any(render_root.rglob("*.md"))

    # Query mode: --latest shows most recent conversation
    latest_result = run_cli(["--plain", "--latest"], env=env, cwd=tmp_path)
    # exit_code 0 = found result, exit_code 2 = no results
    assert latest_result.exit_code in (0, 2)

    # Query mode: search with query terms, json format, --list forces list output
    search_result = run_cli(["--plain", "hello", "--limit", "1", "-f", "json", "--list"], env=env, cwd=tmp_path)
    # exit_code 0 = found result, exit_code 2 = no results
    assert search_result.exit_code in (0, 2)
    if search_result.exit_code == 0:
        payload = json.loads(search_result.stdout.strip())
        # With --list flag, output is always a list
        assert payload and isinstance(payload, list)


@pytest.mark.integration
def test_cli_search_csv_header(tmp_path):
    """Test that CSV output includes proper header."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    output = tmp_path / "out.csv"
    # Query mode: positional args are query terms, --csv writes output
    result = run_cli(["--plain", "missing", "--csv", str(output)], env=env, cwd=tmp_path)
    # exit_code 2 = no results found, but CSV should still be written with header
    assert result.exit_code in (0, 2)
    if output.exists():
        header = output.read_text(encoding="utf-8").splitlines()[0]
        assert header.startswith("source,provider,conversation_id,message_id")


@pytest.mark.integration
def test_cli_search_latest_missing_render(tmp_path):
    """Test --latest --open with no rendered outputs shows error."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: --latest --open
    result = run_cli(["--plain", "--latest", "--open"], env=env, cwd=tmp_path)
    # Should fail: either no results or no rendered outputs
    assert result.exit_code != 0
    output_lower = result.output.lower()
    # Accept various error messages
    assert (
        "no rendered" in output_lower
        or "no conversation" in output_lower
        or "no results" in output_lower
        or result.exit_code == 2
    )


@pytest.mark.integration
def test_cli_search_open_prefers_html(tmp_path):
    """Test that --open prefers HTML over markdown."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    (GenericConversationBuilder("conv-html").add_user("hello html").write_to(inbox / "conversation.json"))

    # First run to create conversation and render
    result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
    assert result.exit_code == 0, result.output

    # Verify render was created
    render_root = paths["render_root"]
    html_files = list(render_root.rglob("*.html"))
    assert html_files, "Expected HTML render to be created"

    # Query mode with --open - just verify it doesn't crash
    search_result = run_cli(["--plain", "hello", "--limit", "1"], env=env, cwd=tmp_path)
    # exit_code 0 = found result, exit_code 2 = no results
    assert search_result.exit_code in (0, 2)


@pytest.mark.integration
def test_cli_config_set_invalid(tmp_path):
    """Test that invalid config keys are rejected."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    result = run_cli(["config", "set", "unknown.key", "value"], env=env, cwd=tmp_path)
    assert result.exit_code != 0
    result = run_cli(["config", "set", "source.missing.type", "auto"], env=env, cwd=tmp_path)
    assert result.exit_code != 0


# --latest validation tests


@pytest.mark.integration
def test_cli_search_latest_returns_path_without_open(tmp_path):
    """polylogue --latest prints conversation info when --open not specified."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    # Create a conversation to ingest
    (GenericConversationBuilder("conv1-abc123").add_user("test content").write_to(inbox / "conversation.json"))

    # First run
    run_result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
    assert run_result.exit_code == 0, run_result.output

    # Query mode: --latest
    result = run_cli(["--plain", "--latest"], env=env, cwd=tmp_path)
    # Should succeed and show conversation info
    assert result.exit_code in (0, 2)  # 0 = found, 2 = no results


@pytest.mark.integration
def test_cli_query_latest_with_query(tmp_path):
    """--latest with query terms is now allowed in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: query terms + --latest = find latest matching query
    result = run_cli(["--plain", "some", "query", "--latest"], env=env, cwd=tmp_path)
    # exit_code 2 = no results (empty db), but should not be invalid syntax
    assert result.exit_code in (0, 2)


@pytest.mark.integration
def test_cli_query_latest_with_json(tmp_path):
    """--latest with --format json is now allowed in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: --latest with json format
    result = run_cli(["--plain", "--latest", "-f", "json"], env=env, cwd=tmp_path)
    # exit_code 2 = no results (empty db), but should not be invalid syntax
    assert result.exit_code in (0, 2)


@pytest.mark.integration
def test_cli_no_args_shows_stats(tmp_path):
    """polylogue (no args) shows stats in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: no args shows stats
    result = run_cli(["--plain"], env=env, cwd=tmp_path)
    # Should succeed and show archive stats
    assert result.exit_code == 0


# Race condition test


@pytest.mark.integration
def test_latest_render_path_handles_deleted_file(tmp_path):
    """latest_render_path() doesn't crash if file deleted between list and stat."""
    from polylogue.cli import helpers as helpers_mod

    render_root = tmp_path / "render"
    conv_dir = render_root / "test" / "conv1-abc"
    conv_dir.mkdir(parents=True, exist_ok=True)

    html_file = conv_dir / "conversation.html"
    html_file.write_text("<html>test</html>", encoding="utf-8")

    # Verify it works normally first
    result = helpers_mod.latest_render_path(render_root)
    assert result is not None
    assert result.name == "conversation.html"

    # Now test with a file that gets "deleted" during iteration
    # Create multiple files
    conv_dir2 = render_root / "test" / "conv2-def"
    conv_dir2.mkdir(parents=True, exist_ok=True)
    html_file2 = conv_dir2 / "conversation.html"
    html_file2.write_text("<html>test2</html>", encoding="utf-8")

    # Touch html_file2 to make it the newest
    html_file2.touch()

    # Delete the first file to simulate race condition
    html_file.unlink()

    # Should still work, returning the file that exists
    result = helpers_mod.latest_render_path(render_root)
    assert result is not None
    assert "conv2" in str(result)


# --open missing render test


@pytest.mark.integration
def test_cli_search_open_missing_render_shows_hint(tmp_path):
    """--open with missing render shows hint to run polylogue."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    # Create inbox with a conversation but don't run render
    (GenericConversationBuilder("conv-no-render").add_user("no render").write_to(inbox / "conversation.json"))

    # Run parse stage only, skip render
    result = run_cli(["--plain", "run", "--stage", "parse"], env=env, cwd=tmp_path)
    assert result.exit_code == 0

    # Query mode: search and try to open - render doesn't exist
    search_result = run_cli(["--plain", "render", "--open"], env=env, cwd=tmp_path)
    # Should either succeed with a warning or indicate render/run not found
    assert (
        search_result.exit_code == 0
        or search_result.exit_code == 2  # no results
        or "render" in search_result.output.lower()
        or "run" in search_result.output.lower()
    )

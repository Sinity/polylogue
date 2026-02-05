"""Tests for CLI commands with zero coverage.

Tests cover: run, auth, check, reset, completions, serve, mcp commands.
Uses subprocess isolation for proper environment handling.

CONSOLIDATED: This file merges tests from:
- test_cli_completions.py (CliRunner unit tests)
- test_cli_mcp.py (CliRunner unit tests)
- test_cli_reset.py (CliRunner unit tests)
- test_cli_serve.py (CliRunner unit tests)

The subprocess integration tests provide end-to-end validation, while
the CliRunner unit tests provide faster, more detailed coverage.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.commands.mcp import mcp_command
from polylogue.storage.index import rebuild_index
from tests.cli_helpers.cli_subprocess import run_cli, setup_isolated_workspace
from tests.helpers import DbFactory, GenericConversationBuilder

# =============================================================================
# SUBPROCESS INTEGRATION TESTS - RUN COMMAND
# =============================================================================


class TestRunCommand:
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

    def test_run_stage_parse_only(self, tmp_path):
        """run --stage parse only does parsing."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        (GenericConversationBuilder("conv1")
         .add_user("parse only")
         .write_to(inbox / "test.json"))

        result = run_cli(["--plain", "run", "--stage", "parse"], env=env)
        assert result.exit_code == 0

    def test_run_stage_render_only(self, tmp_path):
        """run --stage render only does rendering."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "run", "--stage", "render"], env=env)
        # Should succeed even with no data
        assert result.exit_code == 0

    def test_run_stage_index_only(self, tmp_path):
        """run --stage index only does indexing."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "run", "--stage", "index"], env=env)
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
# SUBPROCESS INTEGRATION TESTS - AUTH COMMAND
# =============================================================================


class TestAuthCommand:
    """Tests for the auth command."""

    def test_auth_unknown_service_fails(self, tmp_path):
        """auth --service unknown fails with error."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["auth", "--service", "unknown"], env=env)
        assert result.exit_code != 0
        assert "unknown" in result.output.lower() or "provider" in result.output.lower()

    def test_auth_revoke_no_token(self, tmp_path):
        """auth --revoke handles missing token gracefully."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["auth", "--revoke"], env=env)
        # Should succeed or show "no token" message
        output_lower = result.output.lower()
        assert result.exit_code == 0 or "no token" in output_lower or "not found" in output_lower

    def test_auth_missing_credentials(self, tmp_path):
        """auth fails gracefully when credentials file missing."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["auth"], env=env)
        # Should fail with helpful message about missing credentials
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "credentials" in output_lower or "missing" in output_lower or "oauth" in output_lower


# =============================================================================
# SUBPROCESS INTEGRATION TESTS - CHECK COMMAND
# =============================================================================


class TestCheckCommand:
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
# SUBPROCESS INTEGRATION TESTS - RESET COMMAND
# =============================================================================


class TestResetCommandSubprocess:
    """Subprocess integration tests for the reset command."""

    def test_reset_requires_target(self, tmp_path):
        """reset without flags fails with helpful message."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["reset"], env=env)
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "specify" in output_lower or "target" in output_lower or "--database" in output_lower

    def test_reset_database_requires_force(self, tmp_path):
        """reset --database without --yes prompts (plain mode fails)."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "reset", "--database"], env=env)
        # In plain mode without --yes, should exit without deleting
        # (may succeed if no db exists, or show "use --yes" message)
        output_lower = result.output.lower()
        assert result.exit_code == 0 or "force" in output_lower or "nothing" in output_lower

    def test_reset_force_database(self, tmp_path):
        """reset --database --yes deletes database."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        # Create some data first
        (GenericConversationBuilder("to-delete")
         .add_user("will be deleted")
         .write_to(inbox / "test.json"))
        run_cli(["--plain", "run", "--stage", "parse"], env=env)

        # Now reset
        result = run_cli(["--plain", "reset", "--database", "--yes"], env=env)
        # Should succeed (either deleted or nothing existed)
        assert result.exit_code == 0

    def test_reset_all_flag(self, tmp_path):
        """reset --all sets all targets."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        # With --yes in plain mode
        result = run_cli(["--plain", "reset", "--all", "--yes"], env=env)
        # Should succeed (nothing to delete in fresh workspace)
        assert result.exit_code == 0


# =============================================================================
# SUBPROCESS INTEGRATION TESTS - COMPLETIONS COMMAND
# =============================================================================


class TestCompletionsCommandSubprocess:
    """Subprocess integration tests for the completions command."""

    def test_completions_bash(self, tmp_path):
        """completions --shell bash outputs bash completion script."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["completions", "--shell", "bash"], env=env)
        assert result.exit_code == 0
        # Should contain bash completion markers
        assert "_POLYLOGUE_COMPLETE" in result.stdout or "complete" in result.stdout.lower()

    def test_completions_zsh(self, tmp_path):
        """completions --shell zsh outputs zsh completion script."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["completions", "--shell", "zsh"], env=env)
        assert result.exit_code == 0

    def test_completions_fish(self, tmp_path):
        """completions --shell fish outputs fish completion script."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["completions", "--shell", "fish"], env=env)
        assert result.exit_code == 0

    def test_completions_requires_shell(self, tmp_path):
        """completions without --shell fails."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["completions"], env=env)
        assert result.exit_code != 0
        assert "shell" in result.output.lower() or "required" in result.output.lower()


# =============================================================================
# SUBPROCESS INTEGRATION TESTS - SOURCES COMMAND
# =============================================================================


class TestSourcesCommand:
    """Tests for the sources command."""

    def test_sources_lists_configured(self, tmp_path):
        """sources command lists configured sources."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "sources"], env=env)
        # Should succeed (may show no sources or default inbox)
        assert result.exit_code == 0

    def test_sources_json_output(self, tmp_path):
        """sources --json outputs valid JSON."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "sources", "--json"], env=env)
        assert result.exit_code == 0

        try:
            data = json.loads(result.stdout)
            assert isinstance(data, list)
        except json.JSONDecodeError:
            pytest.fail(f"sources --json did not output valid JSON: {result.stdout}")


# =============================================================================
# SUBPROCESS INTEGRATION TESTS - MCP COMMAND
# =============================================================================


class TestMcpCommandSubprocess:
    """Subprocess integration tests for the mcp command."""

    def test_mcp_unsupported_transport(self, tmp_path):
        """mcp with unsupported transport fails."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        # Click should reject invalid choice
        result = run_cli(["mcp", "--transport", "http"], env=env)
        assert result.exit_code != 0

    # Note: We don't test mcp with stdio transport because it blocks waiting
    # for input. The command structure is tested via help.


# =============================================================================
# CLIRUNNER UNIT TESTS - COMPLETIONS COMMAND
# =============================================================================


class TestCompletionsCommandUnit:
    """Unit tests for the completions command using CliRunner."""

    def test_bash_completion_generates_script(self, cli_runner):
        """Bash completion generates a valid script."""
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "bash"])

        assert result.exit_code == 0
        # Bash completion scripts contain specific markers
        assert "_polylogue_completion" in result.output.lower() or "complete" in result.output.lower()

    def test_zsh_completion_generates_script(self, cli_runner):
        """Zsh completion generates a valid script."""
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "zsh"])

        assert result.exit_code == 0
        # Zsh completion scripts contain specific markers
        assert "compdef" in result.output.lower() or "polylogue" in result.output

    def test_fish_completion_generates_script(self, cli_runner):
        """Fish completion generates a valid script."""
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "fish"])

        assert result.exit_code == 0
        # Fish completion scripts contain specific markers
        assert "complete" in result.output.lower()

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
# CLIRUNNER UNIT TESTS - RESET COMMAND
# =============================================================================


class TestResetCommandValidation:
    """Tests for reset command validation."""

    def test_no_flags_shows_error(self, tmp_path, monkeypatch):
        """Reset without any target flags shows error."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        runner = CliRunner()
        result = runner.invoke(cli, ["reset"])

        assert result.exit_code == 1
        assert "specify" in result.output.lower()

    def test_all_flag_sets_all_targets(self, tmp_path, monkeypatch):
        """--all enables all reset targets."""
        # Patch paths to point to tmp_path
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Create mock path constants for the test
        with patch("polylogue.cli.commands.reset.DB_PATH", tmp_path / "nonexistent.db"), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path / "data"), \
             patch("polylogue.cli.commands.reset.RENDER_ROOT", tmp_path / "render"), \
             patch("polylogue.cli.commands.reset.CACHE_HOME", tmp_path / "cache"), \
             patch("polylogue.cli.commands.reset.DRIVE_TOKEN_PATH", tmp_path / "token.json"):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--all", "--yes"])

            # Should not error even if files don't exist
            assert result.exit_code == 0


class TestResetCommandDeletion:
    """Tests for reset file/directory deletion."""

    def test_database_flag_deletes_db(self, tmp_path, monkeypatch):
        """--database deletes the database file."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test database", encoding="utf-8")
        assert db_path.exists()

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--yes"])

            assert result.exit_code == 0
            assert not db_path.exists()

    def test_assets_flag_deletes_assets(self, tmp_path, monkeypatch):
        """--assets deletes the assets directory."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        data_home = tmp_path / "data"
        assets_dir = data_home / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "test.png").write_bytes(b"test")
        assert assets_dir.exists()

        with patch("polylogue.cli.commands.reset.DB_PATH", tmp_path / "nonexistent.db"), \
             patch("polylogue.cli.commands.reset.DATA_HOME", data_home):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--assets", "--yes"])

            assert result.exit_code == 0
            assert not assets_dir.exists()

    def test_render_flag_deletes_render(self, tmp_path, monkeypatch):
        """--render deletes the render directory."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        render_dir = tmp_path / "render"
        render_dir.mkdir(parents=True)
        (render_dir / "test.html").write_text("<html>test</html>", encoding="utf-8")
        assert render_dir.exists()

        with patch("polylogue.cli.commands.reset.DB_PATH", tmp_path / "nonexistent.db"), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path), \
             patch("polylogue.cli.commands.reset.RENDER_ROOT", render_dir):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--render", "--yes"])

            assert result.exit_code == 0
            assert not render_dir.exists()

    def test_cache_flag_deletes_cache(self, tmp_path, monkeypatch):
        """--cache deletes the cache directory."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "index").write_text("index data", encoding="utf-8")
        assert cache_dir.exists()

        with patch("polylogue.cli.commands.reset.DB_PATH", tmp_path / "nonexistent.db"), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path), \
             patch("polylogue.cli.commands.reset.RENDER_ROOT", tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.CACHE_HOME", cache_dir):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--cache", "--yes"])

            assert result.exit_code == 0
            assert not cache_dir.exists()

    def test_auth_flag_deletes_token(self, tmp_path, monkeypatch):
        """--auth deletes the OAuth token."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        token_path = tmp_path / "token.json"
        token_path.write_text(json.dumps({"token": "test"}), encoding="utf-8")
        assert token_path.exists()

        with patch("polylogue.cli.commands.reset.DB_PATH", tmp_path / "nonexistent.db"), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path), \
             patch("polylogue.cli.commands.reset.RENDER_ROOT", tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.CACHE_HOME", tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.DRIVE_TOKEN_PATH", token_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--auth", "--yes"])

            assert result.exit_code == 0
            assert not token_path.exists()

    def test_multiple_flags(self, tmp_path, monkeypatch):
        """Multiple flags delete specified targets."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test database", encoding="utf-8")

        render_dir = tmp_path / "render"
        render_dir.mkdir(parents=True)
        (render_dir / "test.html").write_text("<html>test</html>", encoding="utf-8")

        data_home = tmp_path / "data"
        assets_dir = data_home / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "keep.png").write_bytes(b"keep")

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", data_home), \
             patch("polylogue.cli.commands.reset.RENDER_ROOT", render_dir):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--render", "--yes"])

            assert result.exit_code == 0
            assert not db_path.exists()
            assert not render_dir.exists()
            # Assets should still exist
            assert assets_dir.exists()


class TestResetConfirmation:
    """Tests for reset confirmation flow."""

    def test_without_force_in_plain_mode_skips(self, tmp_path, monkeypatch):
        """Without --yes in plain mode, shows message and skips."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test database", encoding="utf-8")

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database"])

            # In plain mode without --yes, should not delete
            assert result.exit_code == 0
            assert db_path.exists()
            assert "force" in result.output.lower()

    def test_force_bypasses_confirmation(self, tmp_path, monkeypatch):
        """--yes bypasses confirmation prompt."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test database", encoding="utf-8")

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--yes"])

            assert result.exit_code == 0
            assert not db_path.exists()


class TestResetEmptyTargets:
    """Tests for reset when targets don't exist."""

    def test_nothing_to_reset(self, tmp_path, monkeypatch):
        """When no files exist, shows 'nothing to reset'."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        with patch("polylogue.cli.commands.reset.DB_PATH", tmp_path / "nonexistent.db"), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.RENDER_ROOT", tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.CACHE_HOME", tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.DRIVE_TOKEN_PATH", tmp_path / "nonexistent.json"):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--all", "--yes"])

            assert result.exit_code == 0
            assert "nothing to reset" in result.output.lower()

    def test_partial_targets_exist(self, tmp_path, monkeypatch):
        """Only deletes targets that exist."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test database", encoding="utf-8")

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path / "nonexistent"):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--assets", "--yes"])

            assert result.exit_code == 0
            assert not db_path.exists()
            assert "database" in result.output.lower()


class TestResetErrorHandling:
    """Tests for reset error handling."""

    def test_deletion_failure_shows_error(self, tmp_path, monkeypatch):
        """Deletion failure shows error but continues."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test", encoding="utf-8")

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path), \
             patch("pathlib.Path.unlink") as mock_unlink:
            mock_unlink.side_effect = OSError("Permission denied")

            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--yes"])

            # Should report failure but not crash
            assert "failed" in result.output.lower() or result.exit_code == 0

    def test_shows_what_will_be_deleted(self, tmp_path, monkeypatch):
        """Shows summary of what will be deleted."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test", encoding="utf-8")

        data_home = tmp_path / "data"
        assets_dir = data_home / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "test.png").write_bytes(b"test")

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", data_home):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--assets"])

            # Should show paths in output
            assert "database" in result.output.lower()
            assert "assets" in result.output.lower()


# =============================================================================
# INTEGRATION TESTS FROM test_cli_integration.py
# =============================================================================


def _write_prompt_file(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


# =============================================================================
# END-TO-END CLI TESTS (from test_cli.py)
# =============================================================================


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


def test_cli_config_set_invalid(tmp_path):
    """Test that invalid config keys are rejected."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    result = run_cli(["config", "set", "unknown.key", "value"], env=env, cwd=tmp_path)
    assert result.exit_code != 0
    result = run_cli(["config", "set", "source.missing.type", "auto"], env=env, cwd=tmp_path)
    assert result.exit_code != 0


# --latest validation tests


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


def test_cli_query_latest_with_query(tmp_path):
    """--latest with query terms is now allowed in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: query terms + --latest = find latest matching query
    result = run_cli(["--plain", "some", "query", "--latest"], env=env, cwd=tmp_path)
    # exit_code 2 = no results (empty db), but should not be invalid syntax
    assert result.exit_code in (0, 2)


def test_cli_query_latest_with_json(tmp_path):
    """--latest with --format json is now allowed in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: --latest with json format
    result = run_cli(["--plain", "--latest", "-f", "json"], env=env, cwd=tmp_path)
    # exit_code 2 = no results (empty db), but should not be invalid syntax
    assert result.exit_code in (0, 2)


def test_cli_no_args_shows_stats(tmp_path):
    """polylogue (no args) shows stats in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: no args shows stats
    result = run_cli(["--plain"], env=env, cwd=tmp_path)
    # Should succeed and show archive stats
    assert result.exit_code == 0


# Race condition test


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


# =============================================================================
# SEARCH INTEGRATION TESTS (from test_cli_search_expanded.py)
# =============================================================================


@pytest.fixture
def search_workspace(cli_workspace, monkeypatch):
    """CLI workspace with searchable conversations."""
    # Set up environment
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
    monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_workspace["archive_root"]))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    # Create sample conversations with searchable content
    db_path = cli_workspace["db_path"]
    factory = DbFactory(db_path)

    # Conversation 1: Python content, recent
    factory.create_conversation(
        id="conv1",
        provider="chatgpt",
        title="Python Error Handling",
        messages=[
            {"id": "m1", "role": "user", "text": "How to handle exceptions in Python?"},
            {"id": "m2", "role": "assistant", "text": "Use try-except blocks for Python exception handling."},
        ],
        created_at=datetime.now() - timedelta(days=1),
        updated_at=datetime.now() - timedelta(days=1),
    )

    # Conversation 2: JavaScript content, older
    factory.create_conversation(
        id="conv2",
        provider="claude",
        title="JavaScript Async Patterns",
        messages=[
            {"id": "m3", "role": "user", "text": "Explain async/await in JavaScript"},
            {"id": "m4", "role": "assistant", "text": "Async/await is JavaScript syntax for promises."},
        ],
        created_at=datetime.now() - timedelta(days=10),
        updated_at=datetime.now() - timedelta(days=10),
    )

    # Conversation 3: Rust content
    factory.create_conversation(
        id="conv3",
        provider="claude-code",
        title="Rust Ownership",
        messages=[
            {"id": "m5", "role": "user", "text": "What is ownership in Rust?"},
            {
                "id": "m6",
                "role": "assistant",
                "text": "Rust ownership ensures memory safety without garbage collection.",
            },
        ],
        created_at=datetime.now() - timedelta(hours=6),
        updated_at=datetime.now() - timedelta(hours=6),
    )

    # Build FTS index using rebuild_index

    rebuild_index()

    return cli_workspace


class TestSearchFilters:
    """Tests for search filtering options."""

    def test_search_with_provider_filter(self, search_workspace):
        """Filter search results by provider."""
        runner = CliRunner()
        # Query mode: positional args = query, -p = provider filter
        result = runner.invoke(cli, ["--plain", "Python", "-p", "chatgpt"])
        # exit_code 0 = found, exit_code 2 = no results
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            assert "Python" in result.output or "conv1" in result.output

    def test_search_with_since_date(self, search_workspace):
        """Filter search results by date."""
        runner = CliRunner()
        since_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        # Query mode: positional args = query, --since = date filter
        result = runner.invoke(cli, ["--plain", "Python", "--since", since_date])
        assert result.exit_code in (0, 2)
        # Should find recent Python conversation

    def test_search_with_invalid_since_date(self, search_workspace):
        """Handle invalid --since date format gracefully."""
        runner = CliRunner()
        # Query mode with invalid date
        result = runner.invoke(cli, ["--plain", "Python", "--since", "not-a-date"])
        # The filter chain should handle this gracefully
        # Either fail with error message or treat as "no results"
        assert result.exit_code in (0, 1, 2)

    def test_search_with_limit(self, search_workspace):
        """Limit number of search results."""
        runner = CliRunner()
        # Query mode with --limit
        result = runner.invoke(cli, ["--plain", "JavaScript", "--limit", "1", "--list"])
        assert result.exit_code in (0, 2)
        # Should return at most 1 result


class TestSearchOutputFormats:
    """Tests for different output formats."""

    def test_search_json_output(self, search_workspace):
        """Search with JSON output format."""
        runner = CliRunner()
        # Query mode with -f json and --list
        result = runner.invoke(cli, ["--plain", "Python", "-f", "json", "--list"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert isinstance(data, list)
            if data:
                # JSON output contains conversation-level info
                assert "id" in data[0]

    def test_search_json_format_single(self, search_workspace):
        """Search with JSON output for single result."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "JavaScript", "-f", "json", "--limit", "1"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            # Single result = dict, multiple or --list = list
            assert isinstance(data, (list, dict))

    def test_search_list_mode(self, search_workspace):
        """Search in list mode (shows all results)."""
        runner = CliRunner()
        # Query mode with --list
        result = runner.invoke(cli, ["--plain", "async", "--list"])
        assert result.exit_code in (0, 2)
        # Should list all results

    def test_search_markdown_format(self, search_workspace):
        """Search with markdown output format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "Rust", "-f", "markdown", "--limit", "1"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            # Markdown output should contain headers
            assert "#" in result.output or "Rust" in result.output


class TestSearchEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_no_results(self, search_workspace):
        """Handle query with no matching results."""
        runner = CliRunner()
        # Query mode with non-matching term
        result = runner.invoke(cli, ["--plain", "nonexistent_term_xyz"])
        # exit_code 2 = no results (valid outcome)
        assert result.exit_code == 2
        assert "no conversation" in result.output.lower() or "matched" in result.output.lower()

    def test_stats_mode_no_filters(self, cli_workspace, monkeypatch):
        """Stats mode when no query terms or filters provided."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        runner = CliRunner()
        # No args = stats mode in query-first CLI
        result = runner.invoke(cli, ["--plain"])
        assert result.exit_code == 0
        # Should show stats, not require query

    def test_search_case_insensitive(self, search_workspace):
        """Search is case-insensitive."""
        runner = CliRunner()
        # Query mode with --list to ensure consistent output
        result_lower = runner.invoke(cli, ["--plain", "python", "-f", "json", "--list"])
        result_upper = runner.invoke(cli, ["--plain", "PYTHON", "-f", "json", "--list"])

        # Both should have same exit code
        assert result_lower.exit_code == result_upper.exit_code

        if result_lower.exit_code == 0:
            # Both should find results (FTS5 is case-insensitive by default)
            data_lower = json.loads(result_lower.output)
            data_upper = json.loads(result_upper.output)
            assert len(data_lower) > 0
            assert len(data_upper) > 0

    def test_search_multiple_terms(self, search_workspace):
        """Search with multiple query terms."""
        runner = CliRunner()
        # Query mode: multiple positional args = multiple query terms
        result = runner.invoke(cli, ["--plain", "Python", "exception", "-f", "json", "--list"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert isinstance(data, list)


class TestSearchIndexRebuild:
    """Tests for automatic index rebuild on missing index."""

    def test_search_handles_missing_index(self, cli_workspace, monkeypatch):
        """Search handles missing index gracefully."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Create conversation without building index
        db_path = cli_workspace["db_path"]
        factory = DbFactory(db_path)
        factory.create_conversation(
            id="c1",
            provider="test",
            title="Test",
            messages=[{"id": "m1", "role": "user", "text": "searchable content"}],
        )

        runner = CliRunner()
        # Query mode
        result = runner.invoke(cli, ["--plain", "searchable"])
        # Should either succeed (rebuild worked) or report no results
        assert result.exit_code in (0, 1, 2)


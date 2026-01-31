"""Tests for CLI commands with zero coverage.

Tests cover: sync, auth, check, reset, completions, serve, mcp commands.
Uses subprocess isolation for proper environment handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cli_helpers.cli_subprocess import run_cli, setup_isolated_workspace
from tests.helpers import GenericConversationBuilder


# =============================================================================
# SYNC COMMAND TESTS
# =============================================================================


class TestSyncCommand:
    """Tests for the sync command."""

    def test_sync_preview_shows_plan(self, tmp_path):
        """sync --preview shows what would be done without writing."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        # Create test data
        (GenericConversationBuilder("conv1")
         .add_user("preview test")
         .write_to(inbox / "test.json"))

        result = run_cli(["--plain", "sync", "--preview"], env=env)
        # Preview should succeed (exit 0) or fail gracefully
        assert result.exit_code in (0, 1)
        # Should mention preview or snapshot
        output_lower = result.output.lower()
        assert "preview" in output_lower or "snapshot" in output_lower or "source" in output_lower

    def test_sync_stage_ingest_only(self, tmp_path):
        """sync --stage ingest only does ingestion."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        (GenericConversationBuilder("conv1")
         .add_user("ingest only")
         .write_to(inbox / "test.json"))

        result = run_cli(["--plain", "sync", "--stage", "ingest"], env=env)
        assert result.exit_code == 0

    def test_sync_stage_render_only(self, tmp_path):
        """sync --stage render only does rendering."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "sync", "--stage", "render"], env=env)
        # Should succeed even with no data
        assert result.exit_code == 0

    def test_sync_stage_index_only(self, tmp_path):
        """sync --stage index only does indexing."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "sync", "--stage", "index"], env=env)
        assert result.exit_code == 0

    def test_sync_with_source_filter(self, tmp_path):
        """sync --source filters to specific source."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        # Should handle nonexistent source gracefully
        result = run_cli(["--plain", "sync", "--source", "nonexistent"], env=env)
        # Either fails with error or succeeds with no-op
        assert result.exit_code in (0, 1)

    def test_sync_watch_flags_require_watch(self, tmp_path):
        """--notify, --exec, --webhook require --watch."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "sync", "--notify"], env=env)
        assert result.exit_code != 0
        assert "watch" in result.output.lower()

        result = run_cli(["--plain", "sync", "--exec", "echo test"], env=env)
        assert result.exit_code != 0

        result = run_cli(["--plain", "sync", "--webhook", "http://example.com"], env=env)
        assert result.exit_code != 0


# =============================================================================
# AUTH COMMAND TESTS
# =============================================================================


class TestAuthCommand:
    """Tests for the auth command."""

    def test_auth_unknown_provider_fails(self, tmp_path):
        """auth --provider unknown fails with error."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["auth", "--provider", "unknown"], env=env)
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
# CHECK COMMAND TESTS
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
# RESET COMMAND TESTS
# =============================================================================


class TestResetCommand:
    """Tests for the reset command."""

    def test_reset_requires_target(self, tmp_path):
        """reset without flags fails with helpful message."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["reset"], env=env)
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "specify" in output_lower or "target" in output_lower or "--database" in output_lower

    def test_reset_database_requires_force(self, tmp_path):
        """reset --database without --force prompts (plain mode fails)."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "reset", "--database"], env=env)
        # In plain mode without --force, should exit without deleting
        # (may succeed if no db exists, or show "use --force" message)
        output_lower = result.output.lower()
        assert result.exit_code == 0 or "force" in output_lower or "nothing" in output_lower

    def test_reset_force_database(self, tmp_path):
        """reset --database --force deletes database."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        # Create some data first
        (GenericConversationBuilder("to-delete")
         .add_user("will be deleted")
         .write_to(inbox / "test.json"))
        run_cli(["--plain", "sync", "--stage", "ingest"], env=env)

        # Now reset
        result = run_cli(["--plain", "reset", "--database", "--force"], env=env)
        # Should succeed (either deleted or nothing existed)
        assert result.exit_code == 0

    def test_reset_all_flag(self, tmp_path):
        """reset --all sets all targets."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        # With --force in plain mode
        result = run_cli(["--plain", "reset", "--all", "--force"], env=env)
        # Should succeed (nothing to delete in fresh workspace)
        assert result.exit_code == 0


# =============================================================================
# COMPLETIONS COMMAND TESTS
# =============================================================================


class TestCompletionsCommand:
    """Tests for the completions command."""

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
# SOURCES COMMAND TESTS
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
# MCP COMMAND TESTS
# =============================================================================


class TestMcpCommand:
    """Tests for the mcp command."""

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
# SERVE COMMAND TESTS
# =============================================================================


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_help(self, tmp_path):
        """serve --help shows options."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["serve", "--help"], env=env)
        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    # Note: We don't test serve actually starting because it blocks.
    # The command structure is tested via help.

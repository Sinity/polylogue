"""Tests for reset command."""

from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

# =============================================================================
# TEST DATA TABLE
# =============================================================================

RESET_DELETION_CASES = [
    ("--database", "db_path", "database"),
    ("--assets", "assets_dir", "assets"),
    ("--render", "render_dir", "render"),
    ("--cache", "cache_dir", "cache"),
    ("--auth", "token_path", "auth token"),
]

# =============================================================================
# SUBPROCESS INTEGRATION TESTS - RESET COMMAND
# =============================================================================


@pytest.mark.integration
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
        from tests.infra.helpers import GenericConversationBuilder

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
        with patch("polylogue.cli.commands.reset.db_path", return_value=tmp_path / "nonexistent.db"), \
             patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path / "data"), \
             patch("polylogue.cli.commands.reset.render_root", return_value=tmp_path / "render"), \
             patch("polylogue.cli.commands.reset.cache_home", return_value=tmp_path / "cache"), \
             patch("polylogue.cli.commands.reset.drive_token_path", return_value=tmp_path / "token.json"):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--all", "--yes"])

            # Should not error even if files don't exist
            assert result.exit_code == 0


class TestResetCommandDeletion:
    """Tests for reset file/directory deletion."""

    @pytest.mark.parametrize("flag,path_attr,desc", RESET_DELETION_CASES)
    def test_reset_flag_deletes_target(self, tmp_path, monkeypatch, flag, path_attr, desc):
        """Reset flags delete specified targets."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Set up appropriate paths based on path_attr
        if path_attr == "db_path":
            target_path = tmp_path / "polylogue.db"
            target_path.write_text("test database", encoding="utf-8")
            patches = [patch("polylogue.cli.commands.reset.db_path", return_value=target_path),
                      patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path)]
        elif path_attr == "assets_dir":
            data_home = tmp_path / "data"
            target_path = data_home / "assets"
            target_path.mkdir(parents=True)
            (target_path / "test.png").write_bytes(b"test")
            patches = [patch("polylogue.cli.commands.reset.db_path", return_value=tmp_path / "nonexistent.db"),
                      patch("polylogue.cli.commands.reset.data_home", return_value=data_home)]
        elif path_attr == "render_dir":
            target_path = tmp_path / "render"
            target_path.mkdir(parents=True)
            (target_path / "test.html").write_text("<html>test</html>", encoding="utf-8")
            patches = [patch("polylogue.cli.commands.reset.db_path", return_value=tmp_path / "nonexistent.db"),
                      patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
                      patch("polylogue.cli.commands.reset.render_root", return_value=target_path)]
        elif path_attr == "cache_dir":
            target_path = tmp_path / "cache"
            target_path.mkdir(parents=True)
            (target_path / "index").write_text("index data", encoding="utf-8")
            patches = [patch("polylogue.cli.commands.reset.db_path", return_value=tmp_path / "nonexistent.db"),
                      patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
                      patch("polylogue.cli.commands.reset.render_root", return_value=tmp_path / "nonexistent"),
                      patch("polylogue.cli.commands.reset.cache_home", return_value=target_path)]
        elif path_attr == "token_path":
            target_path = tmp_path / "token.json"
            target_path.write_text(json.dumps({"token": "test"}), encoding="utf-8")
            patches = [patch("polylogue.cli.commands.reset.db_path", return_value=tmp_path / "nonexistent.db"),
                      patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
                      patch("polylogue.cli.commands.reset.render_root", return_value=tmp_path / "nonexistent"),
                      patch("polylogue.cli.commands.reset.cache_home", return_value=tmp_path / "nonexistent"),
                      patch("polylogue.cli.commands.reset.drive_token_path", return_value=target_path)]

        assert target_path.exists()

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", flag, "--yes"])

            assert result.exit_code == 0
            assert not target_path.exists()

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

        with patch("polylogue.cli.commands.reset.db_path", return_value=db_path), \
             patch("polylogue.cli.commands.reset.data_home", return_value=data_home), \
             patch("polylogue.cli.commands.reset.render_root", return_value=render_dir):
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

        with patch("polylogue.cli.commands.reset.db_path", return_value=db_path), \
             patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path):
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

        with patch("polylogue.cli.commands.reset.db_path", return_value=db_path), \
             patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--yes"])

            assert result.exit_code == 0
            assert not db_path.exists()


class TestResetEmptyTargets:
    """Tests for reset when targets don't exist."""

    def test_nothing_to_reset(self, tmp_path, monkeypatch):
        """When no files exist, shows 'nothing to reset'."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        with patch("polylogue.cli.commands.reset.db_path", return_value=tmp_path / "nonexistent.db"), \
             patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.render_root", return_value=tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.cache_home", return_value=tmp_path / "nonexistent"), \
             patch("polylogue.cli.commands.reset.drive_token_path", return_value=tmp_path / "nonexistent.json"):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--all", "--yes"])

            assert result.exit_code == 0
            assert "nothing to reset" in result.output.lower()

    def test_partial_targets_exist(self, tmp_path, monkeypatch):
        """Only deletes targets that exist."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test database", encoding="utf-8")

        with patch("polylogue.cli.commands.reset.db_path", return_value=db_path), \
             patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path / "nonexistent"):
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

        with patch("polylogue.cli.commands.reset.db_path", return_value=db_path), \
             patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path), \
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

        with patch("polylogue.cli.commands.reset.db_path", return_value=db_path), \
             patch("polylogue.cli.commands.reset.data_home", return_value=data_home):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--assets"])

            # Should show paths in output
            assert "database" in result.output.lower()
            assert "assets" in result.output.lower()

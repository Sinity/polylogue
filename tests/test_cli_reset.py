"""Tests for polylogue.cli.commands.reset module.

Coverage targets:
- reset_command: flag validation, target selection, confirmation flow
- File/directory deletion for each target type
- --force bypasses confirmation
- --all sets all flags
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil

import pytest
from click.testing import CliRunner

from polylogue.cli import cli


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
            result = runner.invoke(cli, ["reset", "--all", "--force"])

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
            result = runner.invoke(cli, ["reset", "--database", "--force"])

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
            result = runner.invoke(cli, ["reset", "--assets", "--force"])

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
            result = runner.invoke(cli, ["reset", "--render", "--force"])

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
            result = runner.invoke(cli, ["reset", "--cache", "--force"])

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
            result = runner.invoke(cli, ["reset", "--auth", "--force"])

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
            result = runner.invoke(cli, ["reset", "--database", "--render", "--force"])

            assert result.exit_code == 0
            assert not db_path.exists()
            assert not render_dir.exists()
            # Assets should still exist
            assert assets_dir.exists()


class TestResetConfirmation:
    """Tests for reset confirmation flow."""

    def test_without_force_in_plain_mode_skips(self, tmp_path, monkeypatch):
        """Without --force in plain mode, shows message and skips."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test database", encoding="utf-8")

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database"])

            # In plain mode without --force, should not delete
            assert result.exit_code == 0
            assert db_path.exists()
            assert "force" in result.output.lower()

    def test_force_bypasses_confirmation(self, tmp_path, monkeypatch):
        """--force bypasses confirmation prompt."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        db_path = tmp_path / "polylogue.db"
        db_path.write_text("test database", encoding="utf-8")

        with patch("polylogue.cli.commands.reset.DB_PATH", db_path), \
             patch("polylogue.cli.commands.reset.DATA_HOME", tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["reset", "--database", "--force"])

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
            result = runner.invoke(cli, ["reset", "--all", "--force"])

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
            result = runner.invoke(cli, ["reset", "--database", "--assets", "--force"])

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
            result = runner.invoke(cli, ["reset", "--database", "--force"])

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

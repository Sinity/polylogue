"""Tests for cli/commands/browse.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.commands.browse import browse


class TestBrowseCommand:
    """Tests for the browse command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_browse_calls_run_browser(self, runner):
        """run_browser() is invoked when TUI is available."""
        with patch("polylogue.tui.app.run_browser") as mock_run:
            result = runner.invoke(browse)

        mock_run.assert_called_once()
        assert result.exit_code == 0

    def test_browse_passes_provider_filter(self, runner):
        """--provider flag is forwarded to run_browser."""
        with patch("polylogue.tui.app.run_browser") as mock_run:
            result = runner.invoke(browse, ["--provider", "claude"])

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("provider") == "claude"

    def test_browse_passes_db_path(self, runner, tmp_path):
        """--db-path flag is forwarded to run_browser."""
        db_path = tmp_path / "custom.db"

        with patch("polylogue.tui.app.run_browser") as mock_run:
            result = runner.invoke(browse, ["--db-path", str(db_path)])

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("db_path") == db_path

    def test_browse_keyboard_interrupt_clean_exit(self, runner):
        """Ctrl+C results in clean exit without traceback."""
        with patch("polylogue.tui.app.run_browser", side_effect=KeyboardInterrupt()):
            result = runner.invoke(browse)

        # Should exit cleanly (exit code 0 because KeyboardInterrupt is caught)
        assert result.exit_code == 0
        assert "Traceback" not in result.output

    def test_browse_exception_shows_error(self, runner):
        """Generic exception is displayed to user."""
        with patch("polylogue.tui.app.run_browser", side_effect=RuntimeError("Database corrupted")):
            result = runner.invoke(browse)

        assert "Error running browser" in result.output
        assert "Database corrupted" in result.output

    def test_browse_exception_raises_abort(self, runner):
        """Generic exception raises click.Abort (non-zero exit)."""
        with patch("polylogue.tui.app.run_browser", side_effect=RuntimeError("Test error")):
            result = runner.invoke(browse)

        # click.Abort results in non-zero exit
        assert result.exit_code != 0

    def test_browse_no_provider_filter_default(self, runner):
        """Default provider filter is None."""
        with patch("polylogue.tui.app.run_browser") as mock_run:
            result = runner.invoke(browse)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("provider") is None

    def test_browse_no_db_path_default(self, runner):
        """Default db_path is None."""
        with patch("polylogue.tui.app.run_browser") as mock_run:
            result = runner.invoke(browse)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("db_path") is None

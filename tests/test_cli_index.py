"""Tests for cli/commands/index.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.types import AppEnv
from polylogue.config import ConfigError
from polylogue.ingestion import DriveError


class TestIndexCommand:
    """Tests for the index command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_ui(self):
        """Create a mock UI object."""
        ui = MagicMock()
        ui.plain = True
        ui.console = MagicMock()
        return ui

    @pytest.fixture
    def mock_env(self, mock_ui, tmp_path):
        """Create a mock AppEnv."""
        return AppEnv(ui=mock_ui, config_path=tmp_path / "config.json")

    @patch("polylogue.cli.commands.index.create_config")
    @patch("polylogue.cli.commands.index.run_sources")
    def test_index_command_loads_config(self, mock_run_sources, mock_create_config, cli_workspace, runner):
        """create_config() is called with config path."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock successful run result
        mock_result = MagicMock()
        mock_result.indexed = True
        mock_result.index_error = None
        mock_result.duration_ms = 100
        mock_run_sources.return_value = mock_result

        result = runner.invoke(cli, ["index"], catch_exceptions=False)

        mock_create_config.assert_called_once()

    @patch("polylogue.cli.commands.index.create_config")
    def test_index_config_error_fails(self, mock_create_config, cli_workspace, runner):
        """ConfigError causes fail() to be called."""
        mock_create_config.side_effect = ConfigError("Config file not found")

        result = runner.invoke(cli, ["index"])

        assert result.exit_code != 0
        assert "Config file not found" in result.output

    @patch("polylogue.cli.commands.index.create_config")
    @patch("polylogue.cli.commands.index.run_sources")
    def test_index_calls_run_sources_index_stage(self, mock_run_sources, mock_create_config, cli_workspace, runner):
        """run_sources() is called with stage='index'."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        mock_result = MagicMock()
        mock_result.indexed = True
        mock_result.index_error = None
        mock_result.duration_ms = 100
        mock_run_sources.return_value = mock_result

        result = runner.invoke(cli, ["index"], catch_exceptions=False)

        mock_run_sources.assert_called_once()
        call_kwargs = mock_run_sources.call_args.kwargs
        assert call_kwargs.get("stage") == "index"

    @patch("polylogue.cli.commands.index.create_config")
    @patch("polylogue.cli.commands.index.run_sources")
    def test_index_drive_error_fails(self, mock_run_sources, mock_create_config, cli_workspace, runner):
        """DriveError causes fail() to be called."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config
        mock_run_sources.side_effect = DriveError("Drive authentication failed")

        result = runner.invoke(cli, ["index"])

        assert result.exit_code != 0
        assert "Drive authentication failed" in result.output

    @patch("polylogue.cli.commands.index.create_config")
    @patch("polylogue.cli.commands.index.run_sources")
    @patch("polylogue.cli.commands.index.format_index_status")
    def test_index_displays_status(self, mock_format_status, mock_run_sources, mock_create_config, cli_workspace, runner):
        """format_index_status() output is displayed."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        mock_result = MagicMock()
        mock_result.indexed = True
        mock_result.index_error = None
        mock_result.duration_ms = 100
        mock_run_sources.return_value = mock_result

        mock_format_status.return_value = "Index: OK (updated)"

        result = runner.invoke(cli, ["index"], catch_exceptions=False)

        mock_format_status.assert_called_once_with("index", True, None)

    @patch("polylogue.cli.commands.index.create_config")
    @patch("polylogue.cli.commands.index.run_sources")
    def test_index_displays_duration(self, mock_run_sources, mock_create_config, cli_workspace, runner):
        """Duration is shown in output."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        mock_result = MagicMock()
        mock_result.indexed = True
        mock_result.index_error = None
        mock_result.duration_ms = 1234
        mock_run_sources.return_value = mock_result

        result = runner.invoke(cli, ["index"], catch_exceptions=False)

        assert "Duration: 1234ms" in result.output

    @patch("polylogue.cli.commands.index.create_config")
    @patch("polylogue.cli.commands.index.run_sources")
    def test_index_error_shown_when_present(self, mock_run_sources, mock_create_config, cli_workspace, runner):
        """Index error is displayed when present."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        mock_result = MagicMock()
        mock_result.indexed = False
        mock_result.index_error = "FTS5 table corrupted"
        mock_result.duration_ms = 50
        mock_run_sources.return_value = mock_result

        result = runner.invoke(cli, ["index"], catch_exceptions=False)

        assert "Index error" in result.output
        assert "FTS5 table corrupted" in result.output

    @patch("polylogue.cli.commands.index.create_config")
    @patch("polylogue.cli.commands.index.run_sources")
    def test_index_error_shows_hint(self, mock_run_sources, mock_create_config, cli_workspace, runner):
        """Error shows hint to run polylogue index."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        mock_result = MagicMock()
        mock_result.indexed = False
        mock_result.index_error = "Something went wrong"
        mock_result.duration_ms = 50
        mock_run_sources.return_value = mock_result

        result = runner.invoke(cli, ["index"], catch_exceptions=False)

        assert "polylogue index" in result.output


class TestIndexCommandWithConfig:
    """Tests for index command with --config option."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @patch("polylogue.cli.commands.index.create_config")
    @patch("polylogue.cli.commands.index.run_sources")
    def test_index_custom_config_path(self, mock_run_sources, mock_create_config, cli_workspace, runner, tmp_path):
        """Custom --config path is passed to create_config."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        mock_result = MagicMock()
        mock_result.indexed = True
        mock_result.index_error = None
        mock_result.duration_ms = 100
        mock_run_sources.return_value = mock_result

        custom_config = tmp_path / "custom_config.json"
        custom_config.write_text('{"version": 2}')

        result = runner.invoke(cli, ["index", "--config", str(custom_config)], catch_exceptions=False)

        # create_config should be called with the custom path
        mock_create_config.assert_called_once()
        call_args = mock_create_config.call_args[0]
        assert call_args[0] == custom_config

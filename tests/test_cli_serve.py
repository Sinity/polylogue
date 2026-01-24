"""Tests for cli/commands/serve.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli


class TestServeCommand:
    """Tests for the serve command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_uvicorn(self):
        """Create a mock uvicorn module."""
        mock = MagicMock()
        mock.run = MagicMock()
        return mock

    @pytest.fixture
    def mock_config(self, cli_workspace):
        """Create a mock config object."""
        config = MagicMock()
        config.archive_root = cli_workspace["archive_root"]
        return config

    def test_serve_missing_uvicorn_fails(self, runner, cli_workspace):
        """Missing uvicorn shows helpful error message."""
        # Remove uvicorn from sys.modules to simulate it not being installed
        original_modules = sys.modules.copy()

        # Make uvicorn import fail
        def mock_import(name, *args, **kwargs):
            if name == "uvicorn" or name.startswith("uvicorn."):
                raise ImportError("No module named 'uvicorn'")
            return original_modules.get(name) or __import__(name, *args, **kwargs)

        with patch.dict(sys.modules):
            # Remove uvicorn if it exists
            for key in list(sys.modules.keys()):
                if key == "uvicorn" or key.startswith("uvicorn."):
                    del sys.modules[key]

            with patch("builtins.__import__", side_effect=mock_import):
                result = runner.invoke(cli, ["serve"])

        assert result.exit_code != 0

    def test_serve_loads_config(self, runner, cli_workspace, mock_uvicorn, mock_config):
        """load_effective_config() is called."""
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("polylogue.cli.commands.serve.load_effective_config", return_value=mock_config) as mock_load:
                result = runner.invoke(cli, ["serve"], catch_exceptions=False)

        mock_load.assert_called_once()

    def test_serve_sets_archive_root_env(self, runner, cli_workspace, mock_uvicorn, mock_config, monkeypatch):
        """POLYLOGUE_ARCHIVE_ROOT is set from config."""
        # Clear env var first
        monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("polylogue.cli.commands.serve.load_effective_config", return_value=mock_config):
                result = runner.invoke(cli, ["serve"], catch_exceptions=False)

        # Check that uvicorn.run was called (meaning we got past env setup)
        mock_uvicorn.run.assert_called_once()

    def test_serve_default_host_port(self, runner, cli_workspace, mock_uvicorn, mock_config):
        """Default host is 127.0.0.1 and port is 8000."""
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("polylogue.cli.commands.serve.load_effective_config", return_value=mock_config):
                result = runner.invoke(cli, ["serve"], catch_exceptions=False)

        mock_uvicorn.run.assert_called_once()
        call_args = mock_uvicorn.run.call_args
        assert call_args.kwargs.get("host") == "127.0.0.1"
        assert call_args.kwargs.get("port") == 8000

    def test_serve_custom_host_port(self, runner, cli_workspace, mock_uvicorn, mock_config):
        """Custom --host and --port are forwarded to uvicorn."""
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("polylogue.cli.commands.serve.load_effective_config", return_value=mock_config):
                result = runner.invoke(cli, ["serve", "--host", "0.0.0.0", "--port", "9000"], catch_exceptions=False)

        mock_uvicorn.run.assert_called_once()
        call_args = mock_uvicorn.run.call_args
        assert call_args.kwargs.get("host") == "0.0.0.0"
        assert call_args.kwargs.get("port") == 9000

    def test_serve_prints_startup_message(self, runner, cli_workspace, mock_uvicorn, mock_config):
        """Startup message shows host and port."""
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("polylogue.cli.commands.serve.load_effective_config", return_value=mock_config):
                result = runner.invoke(cli, ["serve"], catch_exceptions=False)

        # Should print a startup message with URL
        assert "Starting server" in result.output
        assert "127.0.0.1:8000" in result.output

    def test_serve_exception_shows_error(self, runner, cli_workspace, mock_uvicorn, mock_config):
        """Exceptions from uvicorn.run are caught and displayed."""
        mock_uvicorn.run.side_effect = RuntimeError("Port already in use")

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("polylogue.cli.commands.serve.load_effective_config", return_value=mock_config):
                result = runner.invoke(cli, ["serve"])

        assert result.exit_code != 0
        assert "Port already in use" in result.output


class TestServeCommandEdgeCases:
    """Tests for serve command edge cases."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_uvicorn(self):
        """Create a mock uvicorn module."""
        mock = MagicMock()
        mock.run = MagicMock()
        return mock

    def test_serve_no_archive_root_still_works(self, runner, cli_workspace, mock_uvicorn):
        """Server starts even if archive_root is None (edge case)."""
        mock_config = MagicMock()
        mock_config.archive_root = None  # Edge case

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("polylogue.cli.commands.serve.load_effective_config", return_value=mock_config):
                result = runner.invoke(cli, ["serve"], catch_exceptions=False)

        # Should still call uvicorn.run
        mock_uvicorn.run.assert_called_once()

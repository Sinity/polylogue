"""Focused tests for non-query CLI command surfaces."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.click_app import mcp_command


class TestDashboardCommand:
    def test_dashboard_launches_app(self, cli_runner, cli_workspace) -> None:
        with patch("polylogue.ui.tui.app.PolylogueApp") as mock_app_cls:
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            result = cli_runner.invoke(click_cli, ["--plain", "dashboard"])
        assert result.exit_code == 0
        mock_app.run.assert_called_once()

    def test_dashboard_creates_app_with_config(self, cli_runner, cli_workspace) -> None:
        with patch("polylogue.ui.tui.app.PolylogueApp") as mock_app_cls:
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            result = cli_runner.invoke(click_cli, ["--plain", "dashboard"])
        assert result.exit_code == 0
        kwargs = mock_app_cls.call_args.kwargs
        assert kwargs["config"].archive_root == cli_workspace["archive_root"]
        assert kwargs["repository"] is not None


class TestSourcesCommand:
    def test_sources_lists_configured(self, cli_runner, monkeypatch, cli_workspace) -> None:
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_DATA_HOME", str(cli_workspace["data_root"]))
        result = cli_runner.invoke(click_cli, ["sources"])
        assert result.exit_code == 0

    def test_sources_json_output(self, cli_runner, monkeypatch, cli_workspace) -> None:
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_DATA_HOME", str(cli_workspace["data_root"]))
        result = cli_runner.invoke(click_cli, ["sources", "--json"])
        assert result.exit_code == 0
        assert isinstance(json.loads(result.output), list)


class TestCompletionsCommand:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_completion_generates_script(self, cli_runner, shell: str) -> None:
        result = cli_runner.invoke(click_cli, ["completions", "--shell", shell])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower() or "complete" in result.output.lower()

    def test_shell_option_is_required(self, cli_runner) -> None:
        result = cli_runner.invoke(click_cli, ["completions"])
        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "required" in result.output.lower()

    def test_invalid_shell_rejected(self, cli_runner) -> None:
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "powershell"])
        assert result.exit_code != 0
        assert "invalid value" in result.output.lower() or "choice" in result.output.lower()


class TestMcpCommandUnit:
    @pytest.fixture
    def mock_env(self):
        mock_ui = MagicMock()
        mock_ui.plain = True
        mock_ui.console = MagicMock()
        env = MagicMock()
        env.ui = mock_ui
        return env

    def test_default_transport_is_stdio(self, cli_runner, mock_env) -> None:
        with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
            result = cli_runner.invoke(mcp_command, [], obj=mock_env)
        mock_serve.assert_called_once()
        assert result.exit_code == 0

    def test_explicit_stdio_transport_works(self, cli_runner, mock_env) -> None:
        with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
            result = cli_runner.invoke(mcp_command, ["--transport", "stdio"], obj=mock_env)
        mock_serve.assert_called_once()
        assert result.exit_code == 0

    def test_missing_mcp_dependencies_error(self, cli_runner, mock_env) -> None:
        with patch.dict(sys.modules, {"polylogue.mcp.server": None}):
            def mock_import(*args, **kwargs):
                raise ImportError("No module named 'mcp'")

            with patch("builtins.__import__", side_effect=mock_import):
                result = cli_runner.invoke(mcp_command, [], obj=mock_env)
        assert result.exit_code != 0 or mock_env.ui.console.print.called

    def test_unsupported_transport_error(self, cli_runner, mock_env) -> None:
        result = cli_runner.invoke(click_cli, ["mcp", "--transport", "http"])
        assert result.exit_code != 0

    def test_mcp_help_shows_description(self, cli_runner) -> None:
        result = cli_runner.invoke(click_cli, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "mcp" in result.output.lower()
        assert "server" in result.output.lower() or "protocol" in result.output.lower()


class TestMcpServerImport:
    def test_serve_stdio_can_be_imported(self) -> None:
        try:
            from polylogue.mcp.server import serve_stdio
            assert callable(serve_stdio)
        except ImportError:
            pytest.skip("MCP dependencies not installed")

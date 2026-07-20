from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import tomllib
from click.testing import CliRunner

from polylogue.mcp.cli import main
from polylogue.mcp.declarations.models import MCPCapabilities


def _config_stub(*, write: bool = False, judge: bool = False, maintenance: bool = False) -> SimpleNamespace:
    return SimpleNamespace(mcp_write_enabled=write, mcp_judge_enabled=judge, mcp_maintenance_enabled=maintenance)


def test_polylogue_mcp_runs_stdio_server_read_only_by_default() -> None:
    runner = CliRunner()

    with (
        patch("polylogue.mcp.server.serve_stdio") as mock_serve,
        patch("polylogue.config.load_polylogue_config", return_value=_config_stub()),
    ):
        result = runner.invoke(main, [])

    assert result.exit_code == 0
    mock_serve.assert_called_once_with(capabilities=MCPCapabilities())


def test_polylogue_mcp_resolves_capabilities_from_config_independently() -> None:
    """No ladder (polylogue-800m): write/judge/maintenance are independent flags."""
    runner = CliRunner()

    with (
        patch("polylogue.mcp.server.serve_stdio") as mock_serve,
        patch(
            "polylogue.config.load_polylogue_config",
            return_value=_config_stub(write=True, judge=False, maintenance=True),
        ),
    ):
        result = runner.invoke(main, [])

    assert result.exit_code == 0
    mock_serve.assert_called_once_with(capabilities=MCPCapabilities(write=True, judge=False, maintenance=True))


def test_polylogue_mcp_handles_missing_dependency() -> None:
    runner = CliRunner()
    real_import = __import__

    def raise_mcp_import_error(name: str, *args: Any, **kwargs: Any) -> object:
        if name == "polylogue.mcp.server":
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=raise_mcp_import_error):
        result = runner.invoke(main, [])

    assert result.exit_code == 1
    assert "MCP dependencies not installed" in result.stderr
    assert "Install the base polylogue package" in result.stderr


def test_polylogue_mcp_has_no_role_flag() -> None:
    """polylogue-800m: the role ladder is retired; there is no --role flag."""
    runner = CliRunner()

    result = runner.invoke(main, ["--role", "admin"])

    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_polylogue_mcp_console_script_is_declared() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["project"]["scripts"]["polylogue-mcp"] == "polylogue.mcp.cli:main"

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import tomllib
from click.testing import CliRunner

from polylogue.mcp.cli import main


def test_polylogue_mcp_runs_stdio_server_with_default_role() -> None:
    runner = CliRunner()

    with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
        result = runner.invoke(main, [])

    assert result.exit_code == 0
    mock_serve.assert_called_once_with(role="read")


def test_polylogue_mcp_accepts_role() -> None:
    runner = CliRunner()

    with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
        result = runner.invoke(main, ["--role", "admin"])

    assert result.exit_code == 0
    mock_serve.assert_called_once_with(role="admin")


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


def test_polylogue_mcp_rejects_unknown_role() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--role", "owner"])

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_polylogue_mcp_console_script_is_declared() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["project"]["scripts"]["polylogue-mcp"] == "polylogue.mcp.cli:main"

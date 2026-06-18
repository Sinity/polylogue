"""CLI output contract tests — JSON/plain/error envelope behavior (#807)."""

from __future__ import annotations

import json

from click.testing import CliRunner


def _invoke(*args: str) -> tuple[int, str]:
    from polylogue.cli.click_app import cli

    runner = CliRunner()
    result = runner.invoke(cli, list(args))
    return result.exit_code, result.output


def test_ops_status_help_plain_output() -> None:
    """ops status --help produces non-empty plain text output."""
    exit_code, output = _invoke("ops", "status", "--help")
    assert exit_code == 0
    assert "status" in output.lower()
    assert len(output.strip()) > 50


def test_config_show_json_output() -> None:
    """``polylogue config --format json`` produces valid JSON."""
    exit_code, output = _invoke("config", "--format", "json")
    assert exit_code == 0
    data = json.loads(output.strip())
    assert isinstance(data, dict)
    assert "archive_root" in data or "daemon_host" in data


def test_config_show_toml_output() -> None:
    """``polylogue config`` (default TOML) produces TOML-like output."""
    exit_code, output = _invoke("config")
    assert exit_code == 0
    assert "[" in output  # TOML section headers


def test_import_missing_path_error() -> None:
    """import with nonexistent path returns non-zero and error message."""
    exit_code, output = _invoke("import", "/nonexistent/path/12345")
    assert exit_code != 0

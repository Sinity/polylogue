"""Tests for polylogue ingest truthfulness (#869)."""

from __future__ import annotations


def test_ingest_command_registered() -> None:
    """ingest command must be available in the CLI group."""
    from polylogue.cli.click_app import cli

    commands = {name for name in cli.commands if not name.startswith("_")}
    assert "ingest" in commands, "ingest command not registered"


def test_ingest_help_includes_inbox_info() -> None:
    """ingest --help should document that files are staged for daemon processing."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "daemon" in result.output.lower() or "polylogued" in result.output.lower(), (
        "ingest help should reference the daemon"
    )

from __future__ import annotations

from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli


def test_inspect_command_is_rejected(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["inspect"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_config_group_lists_subcommands(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["config"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "set" in result.output
    assert "show" in result.output
    assert "edit" in result.output


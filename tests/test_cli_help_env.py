from __future__ import annotations

import json

from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli


def test_help_topic_outputs_details(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["help", "sync"])
    assert result.exit_code == 0
    assert "Synchronize provider archives" in result.output


def test_help_examples_flag(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["help", "--examples"])
    assert result.exit_code == 0
    assert "EXAMPLES" in result.output
    assert "render" in result.output


def test_help_unknown_command_reports_error(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["help", "nope"])
    assert result.exit_code == 1
    assert "Unknown command" in result.output


def test_help_lists_command_descriptions(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["help"])
    assert result.exit_code == 0
    assert "sync" in result.output
    assert "Synchronize provider archives" in result.output


def test_config_show_json(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["config", "show", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "outputs" in parsed
    assert "statePath" in parsed
    assert "auth" in parsed
    assert "credentialPath" in parsed["auth"]


def test_completions_emits_script(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["completions", "--shell", "bash"])
    assert result.exit_code == 0
    assert "polylogue" in result.output
    assert "complete -F" in result.output


def test_zsh_completions_include_compdef(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["completions", "--shell", "zsh"])
    assert result.exit_code == 0
    assert "#compdef polylogue" in result.output
    assert "compdef _polylogue_complete polylogue" in result.output


def test_fish_completions_include_descriptions(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["completions", "--shell", "fish"])
    assert result.exit_code == 0
    assert "Synchronize provider archives" in result.output


def test_complete_top_level(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["_complete", "--shell", "zsh", "--cword", "1", "polylogue", ""])
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert "browse" in values
    assert "config" in values
    assert "doctor" in values
    assert "import" in values
    assert "verify" in values


def test_complete_sync_provider(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["_complete", "--shell", "zsh", "--cword", "2", "polylogue", "sync", ""])
    assert result.exit_code == 0
    assert "drive" in result.output


def test_complete_browse_subcommands_include_recent_additions(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["_complete", "--shell", "zsh", "--cword", "2", "polylogue", "browse", ""])
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert "metrics" in values
    assert "timeline" in values
    assert "analytics" in values
    assert "inbox" in values


def test_complete_doctor_subcommands_include_restore(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["_complete", "--shell", "zsh", "--cword", "2", "polylogue", "doctor", ""])
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert "restore" in values


def test_complete_config_subcommands_include_edit(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(click_cli, ["_complete", "--shell", "zsh", "--cword", "2", "polylogue", "config", ""])
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert "edit" in values


def test_complete_attachments_subcommands(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(
        click_cli,
        ["_complete", "--shell", "zsh", "--cword", "3", "polylogue", "doctor", "attachments", ""],
    )
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert "stats" in values
    assert "extract" in values


def test_complete_prefs_subcommands(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(
        click_cli,
        ["_complete", "--shell", "zsh", "--cword", "3", "polylogue", "config", "prefs", ""],
    )
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert "list" in values
    assert "set" in values
    assert "clear" in values


def test_complete_browse_stats_sort_values(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(
        click_cli,
        ["_complete", "--shell", "zsh", "--cword", "4", "--", "polylogue", "browse", "stats", "--sort", ""],
    )
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert "tokens" in values
    assert "recent" in values


def test_complete_sync_html_values(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(
        click_cli,
        ["_complete", "--shell", "zsh", "--cword", "4", "--", "polylogue", "sync", "drive", "--html", ""],
    )
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert "on" in values
    assert "off" in values
    assert "auto" in values


def test_complete_render_path_values_trigger_path_mode(state_env) -> None:
    runner = CliRunner()
    result = runner.invoke(
        click_cli,
        ["_complete", "--shell", "zsh", "--cword", "2", "polylogue", "render", ""],
    )
    assert result.exit_code == 0
    values = [line.split(":", 1)[0] for line in result.output.strip().splitlines() if line.strip()]
    assert values
    assert values[0] == "__PATH__"

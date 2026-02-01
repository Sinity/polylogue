"""Tests for polylogue completions command.

Coverage targets:
- completions_command: Shell completion generation
- --shell: Support for bash, zsh, fish
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.completions import completions_command


@pytest.fixture
def runner():
    """CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_root_command():
    """Create a mock Click command for completion testing."""
    import click

    @click.command()
    def root_cmd():
        pass

    return root_cmd


class TestCompletionsCommand:
    """Tests for the completions command."""

    def test_bash_completion_generates_script(self, runner):
        """Bash completion generates a valid script."""
        # Need to invoke through parent context
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["completions", "--shell", "bash"])

        assert result.exit_code == 0
        # Bash completion scripts contain specific markers
        assert "_polylogue_completion" in result.output.lower() or "complete" in result.output.lower()

    def test_zsh_completion_generates_script(self, runner):
        """Zsh completion generates a valid script."""
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["completions", "--shell", "zsh"])

        assert result.exit_code == 0
        # Zsh completion scripts contain specific markers
        assert "compdef" in result.output.lower() or "polylogue" in result.output

    def test_fish_completion_generates_script(self, runner):
        """Fish completion generates a valid script."""
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["completions", "--shell", "fish"])

        assert result.exit_code == 0
        # Fish completion scripts contain specific markers
        assert "complete" in result.output.lower()

    def test_shell_option_is_required(self, runner):
        """--shell option is required."""
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["completions"])

        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "required" in result.output.lower()

    def test_invalid_shell_rejected(self, runner):
        """Invalid shell type is rejected."""
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["completions", "--shell", "powershell"])

        assert result.exit_code != 0
        assert "invalid value" in result.output.lower() or "choice" in result.output.lower()

    def test_completion_uses_prog_name_polylogue(self, runner):
        """Completion script uses 'polylogue' as program name."""
        from polylogue.cli.click_app import cli

        result = runner.invoke(cli, ["completions", "--shell", "bash"])

        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()

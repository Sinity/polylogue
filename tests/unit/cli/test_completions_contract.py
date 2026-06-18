"""Contracts for the ``polylogue ops completions`` shell-integration surface.

These tests pin the shape of the completions subcommand: each supported
shell produces non-empty, shell-syntax output that mentions the program
name, exits 0 on success, and exits non-zero (without traceback) on an
unsupported shell choice.

Companion to ``test_plain_output_contract.py`` and ``test_help_contract.py``
under #1060.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli

pytestmark = pytest.mark.contract

TRACEBACK_SENTINEL = "Traceback (most recent call last)"

# Shells whose completion scripts polylogue ships. Drives the parametrized
# matrix and is asserted as a stable contract; adding a shell here is a
# deliberate behavior change.
SUPPORTED_SHELLS = ("bash", "zsh", "fish")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCompletionsPerShell:
    """Each supported shell produces non-empty, well-formed completion output."""

    @pytest.mark.parametrize("shell", SUPPORTED_SHELLS)
    def test_completions_for_shell_succeeds(
        self,
        shell: str,
        runner: CliRunner,
    ) -> None:
        result = runner.invoke(cli, ["ops", "completions", "--shell", shell])
        assert result.exit_code == 0, f"completions --shell {shell} exited {result.exit_code}: {result.output!r}"
        assert TRACEBACK_SENTINEL not in result.output
        assert result.stdout.strip(), f"empty completion script for shell={shell}"
        # The script always invokes our binary name; otherwise the shell would
        # not know which completion handler to attach.
        assert "polylogue" in result.stdout, (
            f"completion script for shell={shell} does not mention 'polylogue': {result.stdout!r}"
        )

    @pytest.mark.parametrize("shell", SUPPORTED_SHELLS)
    def test_completions_uses_recognizable_shell_syntax(
        self,
        shell: str,
        runner: CliRunner,
    ) -> None:
        """Output for each shell contains syntax markers specific to that shell."""
        result = runner.invoke(cli, ["ops", "completions", "--shell", shell])
        assert result.exit_code == 0
        body = result.stdout
        if shell == "bash":
            # Click bash completion uses bash function syntax and `complete -o`.
            assert "()" in body and ("complete" in body or "compgen" in body), (
                f"bash completion lacks bash markers: {body!r}"
            )
        elif shell == "zsh":
            # Click zsh completion registers via compdef.
            assert "compdef" in body or "_polylogue" in body, f"zsh completion lacks zsh markers: {body!r}"
        elif shell == "fish":
            # Click fish completion uses `function` blocks and `complete -c`.
            assert "function" in body and "complete" in body, f"fish completion lacks fish markers: {body!r}"


class TestCompletionsErrorPaths:
    """Unsupported shell choices fail cleanly without tracebacks."""

    def test_unknown_shell_exits_non_zero(
        self,
        runner: CliRunner,
    ) -> None:
        result = runner.invoke(cli, ["ops", "completions", "--shell", "ksh"])
        assert result.exit_code != 0
        assert TRACEBACK_SENTINEL not in result.output

    def test_missing_shell_flag_exits_non_zero(self, runner: CliRunner) -> None:
        """``polylogue ops completions`` without ``--shell`` is a Click usage error."""
        result = runner.invoke(cli, ["ops", "completions"])
        assert result.exit_code != 0
        assert TRACEBACK_SENTINEL not in result.output

    def test_missing_shell_error_goes_to_stderr(self, runner: CliRunner) -> None:
        """Click usage errors land on stderr, not stdout."""
        result = runner.invoke(cli, ["ops", "completions"])
        # The error message is on stderr; stdout stays empty.
        assert "Missing option" in result.stderr or "Error" in result.stderr, (
            f"expected Click usage error on stderr, got stderr={result.stderr!r}"
        )
        assert "Missing option" not in result.stdout

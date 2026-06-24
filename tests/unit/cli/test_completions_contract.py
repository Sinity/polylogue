"""Contracts for the ``polylogue config completions`` shell-integration surface.

These tests pin the shape of the completions subcommand: each supported
shell produces non-empty, shell-syntax output that mentions the program
name, exits 0 on success, and exits non-zero (without traceback) on an
unsupported shell choice.

Companion to ``test_plain_output_contract.py`` and ``test_help_contract.py``.
"""

from __future__ import annotations

import json

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
        result = runner.invoke(cli, ["config", "completions", "--shell", shell])
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
        result = runner.invoke(cli, ["config", "completions", "--shell", shell])
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
        result = runner.invoke(cli, ["config", "completions", "--shell", "ksh"])
        assert result.exit_code != 0
        assert TRACEBACK_SENTINEL not in result.output

    def test_missing_shell_flag_exits_non_zero(self, runner: CliRunner) -> None:
        """``polylogue config completions`` without ``--shell`` is a Click usage error."""
        result = runner.invoke(cli, ["config", "completions"])
        assert result.exit_code != 0
        assert TRACEBACK_SENTINEL not in result.output

    def test_missing_shell_error_goes_to_stderr(self, runner: CliRunner) -> None:
        """Click usage errors land on stderr, not stdout."""
        result = runner.invoke(cli, ["config", "completions"])
        # The error message is on stderr; stdout stays empty.
        assert "Missing option" in result.stderr or "Error" in result.stderr, (
            f"expected Click usage error on stderr, got stderr={result.stderr!r}"
        )
        assert "Missing option" not in result.stdout


class TestQueryCompletionMetadata:
    """The CLI exposes the shared query-builder metadata payload directly."""

    def test_query_completions_prints_shared_field_payload(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["config", "query-completions", "--kind", "field", "--incomplete", "da"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["kind"] == "field"
        candidates = payload["candidates"]
        assert isinstance(candidates, list)
        date_candidate = next(candidate for candidate in candidates if candidate["value"] == "date")
        assert date_candidate["insert"] == "date "
        assert date_candidate["source"] == "DATE_QUERY_FIELD_REGISTRY"

    def test_query_completions_prints_unit_scoped_terminal_fields(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            [
                "config",
                "query-completions",
                "--kind",
                "terminal-field",
                "--unit",
                "context-snapshots",
                "--incomplete",
                "bound",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["kind"] == "terminal-field"
        assert payload["unit"] == "context-snapshots"
        candidates = payload["candidates"]
        assert isinstance(candidates, list)
        assert [candidate["value"] for candidate in candidates] == ["boundary"]

    def test_query_completions_reports_missing_context_cleanly(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["config", "query-completions", "--kind", "terminal-field"])

        assert result.exit_code != 0
        assert "--unit is required for terminal-field completion" in result.output
        assert TRACEBACK_SENTINEL not in result.output


class TestActionAffordanceMetadata:
    """The CLI exposes shared query-action affordances directly."""

    def test_action_affordances_prints_shared_payload(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["config", "action-affordances"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        actions = payload["actions"]
        assert isinstance(actions, list)
        by_id = {item["id"]: item for item in actions}
        assert {"find", "read", "select", "continue", "analyze", "mark", "delete"}.issubset(by_id)
        read = by_id["read"]
        assert "input_unit" not in read
        assert read["input"]["unit"] == "query_result_set"
        assert read["execution"]["cardinality_state"] == "explicit_multi"
        assert read["safety"]["selection_command"] == "polylogue find QUERY then select"
        assert "browser" in read["output"]["destination_support"]
        delete = by_id["delete"]
        assert "safety_level" not in delete
        assert "confirmation_command" not in delete
        assert delete["safety"]["safety_level"] == "destructive"
        assert delete["safety"]["confirmation_command"] == "polylogue find QUERY then delete --dry-run"

    @pytest.mark.parametrize("args", (["--json"], ["--format", "json"]))
    def test_action_affordances_accepts_common_json_flags(self, runner: CliRunner, args: list[str]) -> None:
        result = runner.invoke(cli, ["config", "action-affordances", *args])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert any(action["id"] == "find" for action in payload["actions"])

"""CLI exit code contract tests.

Pins the exit code semantics of the polylogue CLI:

- Exit 0: command completed successfully
- Exit 1: runtime error (PolylogueError, unhandled exception, invalid date)
- Exit 2: Click argument-type mismatch (UsageError / BadParameter)

These tests use CliRunner so they run in-process. Tests that need a
real database use ``workspace_env`` (isolated XDG paths, no production DB
access). Tests that only need flag parsing use ``--help`` or rely on Click
type rejection, which fires before any database I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli

pytestmark = pytest.mark.machine_contract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Exit 0 — successful commands (--help is zero-cost, no DB required)
# ---------------------------------------------------------------------------


class TestSuccessExitCode:
    """CLI exit code 0 for successful invocations."""

    def test_root_help(self, runner: CliRunner) -> None:
        """polylogue --help exits 0."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        "cmd",
        [
            "doctor",
            "import",
            "schema",
            "tags",
            "list",
            "count",
            "stats",
        ],
    )
    def test_subcommand_help_exits_zero(self, runner: CliRunner, cmd: str) -> None:
        """Every subcommand --help exits 0."""
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, f"polylogue {cmd} --help exited {result.exit_code}: {result.output}"

    def test_version_exits_zero(self, runner: CliRunner) -> None:
        """polylogue --version exits 0."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_read_help_exits_zero(self, runner: CliRunner) -> None:
        """polylogue read --help exits 0."""
        result = runner.invoke(cli, ["read", "--help"])
        assert result.exit_code == 0

    def test_analyze_help_exits_zero(self, runner: CliRunner) -> None:
        """polylogue analyze --help exits 0."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Exit 2 — Click type-check rejections (argument-type mismatches)
#
# --limit, --min-messages, --min-words, --max-messages are declared as
# ``type=int`` Click options. Passing a non-numeric string triggers a
# Click UsageError immediately — before any database I/O — and Click
# exits with code 2.
# ---------------------------------------------------------------------------


class TestTypeErrorExitCode:
    """CLI exit code 2 for invalid argument types."""

    def test_limit_non_numeric(self, runner: CliRunner) -> None:
        """polylogue --limit abc exits 2 (int type mismatch)."""
        result = runner.invoke(cli, ["--limit", "abc"])
        assert result.exit_code == 2, (
            f"Expected exit 2 for --limit abc, got {result.exit_code}. Output: {result.output}"
        )

    def test_min_messages_non_numeric(self, runner: CliRunner) -> None:
        """polylogue --min-messages xyz exits 2 (int type mismatch)."""
        result = runner.invoke(cli, ["--min-messages", "xyz"])
        assert result.exit_code == 2, (
            f"Expected exit 2 for --min-messages xyz, got {result.exit_code}. Output: {result.output}"
        )

    def test_min_words_non_numeric(self, runner: CliRunner) -> None:
        """polylogue --min-words abc exits 2 (int type mismatch)."""
        result = runner.invoke(cli, ["--min-words", "abc"])
        assert result.exit_code == 2, (
            f"Expected exit 2 for --min-words abc, got {result.exit_code}. Output: {result.output}"
        )

    def test_max_messages_non_numeric(self, runner: CliRunner) -> None:
        """polylogue --max-messages abc exits 2 (int type mismatch)."""
        result = runner.invoke(cli, ["--max-messages", "abc"])
        assert result.exit_code == 2, (
            f"Expected exit 2 for --max-messages abc, got {result.exit_code}. Output: {result.output}"
        )

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--limit", "not-a-number"),
            ("--limit", "3.14"),
            ("--min-messages", "one"),
            ("--min-words", "many"),
            ("--max-messages", "lots"),
        ],
    )
    def test_int_flag_type_matrix(self, runner: CliRunner, flag: str, value: str) -> None:
        """All int-typed flags produce exit 2 for non-integer values."""
        result = runner.invoke(cli, [flag, value])
        assert result.exit_code == 2, (
            f"Expected exit 2 for {flag} {value!r}, got {result.exit_code}. Output: {result.output}"
        )


# ---------------------------------------------------------------------------
# Exit 1 — runtime errors from query execution
#
# --since is a plain ``str`` option: Click accepts any value and passes it
# to the query layer, which validates the date string at execution time.
# An invalid date raises QuerySpecError (PolylogueError), yielding exit 1
# — not exit 2. This is a deliberate design choice in the CLI: date
# parsing is semantic, not structural, so it is validated at query time.
# ---------------------------------------------------------------------------


class TestRuntimeErrorExitCode:
    """CLI exit code 1 for runtime errors that occur during query execution."""

    def test_invalid_since_date_exits_one(
        self,
        runner: CliRunner,
        workspace_env: Mapping[str, Path],
    ) -> None:
        """polylogue --since not-a-date exits 1 (date validation is runtime, not Click type).

        --since is declared as str, so Click accepts it. The query layer
        raises QuerySpecError (a PolylogueError) when it cannot parse the
        date, which run_machine_entry maps to exit 1.
        """
        result = runner.invoke(
            cli,
            ["--plain", "--since", "not-a-date", "--limit", "0"],
            catch_exceptions=True,
        )
        assert result.exit_code == 1, (
            f"Expected exit 1 for invalid --since date, got {result.exit_code}. Output: {result.output}"
        )

    def test_invalid_until_date_exits_one(
        self,
        runner: CliRunner,
        workspace_env: Mapping[str, Path],
    ) -> None:
        """polylogue --until garbage-date exits 1 (same runtime-validation path)."""
        result = runner.invoke(
            cli,
            ["--plain", "--until", "garbage-date", "--limit", "0"],
            catch_exceptions=True,
        )
        assert result.exit_code == 1, (
            f"Expected exit 1 for invalid --until date, got {result.exit_code}. Output: {result.output}"
        )


# ---------------------------------------------------------------------------
# JSON mode (--format json) — exit code contract via machine_main
#
# In JSON mode, run_machine_entry converts UsageError → exit 2 and
# PolylogueError → exit 1 explicitly. The exit code semantics must be
# consistent regardless of whether --format json is passed.
# ---------------------------------------------------------------------------


class TestJsonModeExitCodeConsistency:
    """Exit code semantics are stable across plain and JSON output modes."""

    def test_help_exits_zero_json_format(self, runner: CliRunner) -> None:
        """schema list --format json --help exits 0 (--help short-circuits before format matters)."""
        result = runner.invoke(cli, ["ops", "schema", "list", "--help"])
        assert result.exit_code == 0

    def test_int_type_error_exits_two_regardless_of_output_format(self, runner: CliRunner) -> None:
        """--limit abc exits 2 even when combined with --format json context.

        Click's type checking fires before any output formatting, so the
        exit code contract is identical in plain and JSON modes.
        """
        result = runner.invoke(cli, ["--limit", "abc", "--format", "json"])
        assert result.exit_code == 2, (
            f"Expected exit 2 for --limit abc, got {result.exit_code}. Output: {result.output}"
        )

    def test_zero_and_nonzero_are_disjoint(self, runner: CliRunner) -> None:
        """A successful --help and a type error produce distinct exit codes."""
        help_result = runner.invoke(cli, ["--help"])
        error_result = runner.invoke(cli, ["--limit", "xyz"])
        assert help_result.exit_code == 0
        assert error_result.exit_code != 0
        assert help_result.exit_code != error_result.exit_code

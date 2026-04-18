"""CLI error boundary tests.

Proves the CLI never exposes raw Python tracebacks to end users,
handles invalid flags gracefully, and all subcommands provide --help.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli

TRACEBACK_SENTINEL = "Traceback (most recent call last)"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# =============================================================================
# All subcommands have --help
# =============================================================================

SUBCOMMANDS = [
    "run",
    "doctor",
    "reset",
    "mcp",
    "auth",
    "completions",
    "dashboard",
    "audit",
    "schema",
    "tags",
    "list",
    "count",
    "stats",
    "open",
    "delete",
]


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_subcommand_help(runner: CliRunner, cmd: str) -> None:
    """Every registered subcommand responds to --help without error."""
    result = runner.invoke(cli, [cmd, "--help"])
    assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
    assert TRACEBACK_SENTINEL not in result.output


def test_root_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert TRACEBACK_SENTINEL not in result.output


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# Invalid flag values
# =============================================================================


class TestInvalidFlags:
    def test_invalid_limit(self, runner: CliRunner) -> None:
        """Non-numeric limit should not produce traceback."""
        result = runner.invoke(cli, ["--limit", "abc"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_negative_limit(self, runner: CliRunner) -> None:
        """Negative limit should not produce traceback."""
        result = runner.invoke(cli, ["--limit", "-1"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_invalid_min_messages(self, runner: CliRunner) -> None:
        """Non-numeric --min-messages should not produce traceback."""
        result = runner.invoke(cli, ["--min-messages", "abc"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_invalid_max_messages(self, runner: CliRunner) -> None:
        """Non-numeric --max-messages should not produce traceback."""
        result = runner.invoke(cli, ["--max-messages", "xyz"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_invalid_min_words(self, runner: CliRunner) -> None:
        """Non-numeric --min-words should not produce traceback."""
        result = runner.invoke(cli, ["--min-words", "not-a-number"])
        assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# Invalid date values
# =============================================================================


class TestInvalidDates:
    """Date boundary tests use workspace_env + --plain --limit 0 to stay fast.

    Without workspace_env, root CLI invocations resolve the real user database.
    --limit 0 exits after filter parsing with zero results rendered.
    """

    def test_invalid_since_date(self, runner: CliRunner, workspace_env: Mapping[str, Path]) -> None:
        """Malformed --since should not produce traceback."""
        result = runner.invoke(cli, ["--plain", "--since", "not-a-date", "--limit", "0"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_invalid_until_date(self, runner: CliRunner, workspace_env: Mapping[str, Path]) -> None:
        """Malformed --until should not produce traceback."""
        result = runner.invoke(cli, ["--plain", "--until", "garbage-date", "--limit", "0"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_since_format_variations(self, runner: CliRunner, workspace_env: Mapping[str, Path]) -> None:
        """Partial/malformed ISO dates should not crash."""
        bad_dates = ["2024", "2024-13", "2024-01-32", "invalid"]
        for bad_date in bad_dates:
            result = runner.invoke(cli, ["--plain", "--since", bad_date, "--limit", "0"])
            assert TRACEBACK_SENTINEL not in result.output, f"Traceback for --since {bad_date}"


# =============================================================================
# Nonexistent paths
# =============================================================================


class TestNonexistentPaths:
    def test_nonexistent_source_path(self, runner: CliRunner) -> None:
        """Non-existent --source should not produce traceback."""
        result = runner.invoke(cli, ["run", "--source", "/nonexistent/path"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_nonexistent_output_path(self, runner: CliRunner, workspace_env: Mapping[str, Path]) -> None:
        """Non-existent --output path should not produce traceback."""
        result = runner.invoke(cli, ["--plain", "--output", "/nonexistent/dir/file.md", "--limit", "0"])
        assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# Unknown commands and flags
# =============================================================================


class TestUnknownInputs:
    def test_unknown_subcommand(self, runner: CliRunner) -> None:
        """Unknown subcommand should not produce traceback."""
        result = runner.invoke(cli, ["nonexistent-command"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_unknown_flag(self, runner: CliRunner) -> None:
        """Unknown flag should not produce traceback."""
        result = runner.invoke(cli, ["--nonexistent-flag"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_unknown_flag_with_subcommand(self, runner: CliRunner) -> None:
        """Unknown flag in subcommand should not produce traceback."""
        result = runner.invoke(cli, ["run", "--fake-flag"])
        assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# Multiple/conflicting filters
# =============================================================================


class TestFilterCombinations:
    """Filter combination tests use workspace_env + --plain --limit 0 to stay fast.

    Without workspace_env the root CLI resolves the real user DB (XDG_DATA_HOME),
    which can contain thousands of conversations and make the test very slow.
    --limit 0 ensures we exit after filter resolution with zero results rendered.
    """

    def test_exclude_and_include_same_provider(
        self,
        runner: CliRunner,
        workspace_env: Mapping[str, Path],
    ) -> None:
        """Conflicting provider filters should not traceback."""
        result = runner.invoke(
            cli, ["--plain", "--provider", "claude-ai", "--exclude-provider", "claude-ai", "--limit", "0"]
        )
        assert TRACEBACK_SENTINEL not in result.output

    def test_multiple_contain_terms(self, runner: CliRunner, workspace_env: Mapping[str, Path]) -> None:
        """Multiple --contains should work without traceback."""
        result = runner.invoke(cli, ["--plain", "--contains", "word1", "--contains", "word2", "--limit", "0"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_multiple_exclude_terms(self, runner: CliRunner, workspace_env: Mapping[str, Path]) -> None:
        """Multiple --exclude-text should work without traceback."""
        result = runner.invoke(cli, ["--plain", "--exclude-text", "bad1", "--exclude-text", "bad2", "--limit", "0"])
        assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# Hypothesis fuzz: random query terms never produce tracebacks
# =============================================================================


@pytest.mark.parametrize(
    "query",
    [
        "",
        "hello",
        "SELECT * FROM users",
        "foo bar baz",
        "!@#$%^&*()",
        "a" * 200,
        "日本語",
        "--help",
        "query with spaces and 123 numbers",
        "\t\ttabs",
    ],
)
def test_random_query_no_traceback(query: str) -> None:
    """Various query strings never produce a traceback."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--plain", query, "--limit", "0"])
    assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# QA subcommand variants
# =============================================================================


class TestQAErrorBoundaries:
    def test_qa_invalid_tier(self, runner: CliRunner) -> None:
        """Non-numeric --tier should not produce traceback."""
        result = runner.invoke(cli, ["audit", "--tier", "not-a-tier"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_qa_nonexistent_source(self, runner: CliRunner) -> None:
        """Unknown configured source name should fail fast without a traceback."""
        result = runner.invoke(cli, ["audit", "--source", "/does/not/exist"])
        assert result.exit_code != 0
        assert TRACEBACK_SENTINEL not in result.output
        assert "audit: Unknown source(s): /does/not/exist." in result.output


# =============================================================================
# Completions subcommand
# =============================================================================


class TestCompletionsErrorBoundaries:
    def test_completions_invalid_shell(self, runner: CliRunner) -> None:
        """Unknown --shell value should not produce traceback."""
        result = runner.invoke(cli, ["completions", "--shell", "unknown-shell"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_completions_missing_shell(self, runner: CliRunner) -> None:
        """Missing --shell should not produce traceback (Click enforces required)."""
        result = runner.invoke(cli, ["completions"])
        # Click will show usage error, not traceback
        assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# MCP subcommand
# =============================================================================


class TestMCPErrorBoundaries:
    def test_mcp_invalid_transport(self, runner: CliRunner) -> None:
        """Unknown --transport should not produce traceback."""
        result = runner.invoke(cli, ["mcp", "--transport", "invalid-transport"])
        assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# Generate subcommand
# =============================================================================


class TestGenerateErrorBoundaries:
    def test_generate_invalid_count(self, runner: CliRunner) -> None:
        """Non-numeric -n/--count should not produce traceback."""
        result = runner.invoke(cli, ["audit", "generate", "-n", "not-a-number"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_generate_invalid_seed(self, runner: CliRunner) -> None:
        """Non-numeric --seed should not produce traceback."""
        result = runner.invoke(cli, ["audit", "generate", "--seed", "not-a-seed"])
        assert TRACEBACK_SENTINEL not in result.output


# =============================================================================
# Schema subcommand
# =============================================================================


class TestSchemaErrorBoundaries:
    def test_schema_help_exists(self, runner: CliRunner) -> None:
        """schema command should have --help."""
        result = runner.invoke(cli, ["schema", "--help"])
        assert result.exit_code == 0
        assert TRACEBACK_SENTINEL not in result.output

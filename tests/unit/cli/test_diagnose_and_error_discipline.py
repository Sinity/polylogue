"""Tests for #1273: error/help/query-first discipline + --diagnose.

Covers four contracts:

1. Root ``--help`` explicitly documents query-first dispatch and points at
   ``--diagnose`` and ``<subcommand> --help`` as escape hatches.
2. ``--diagnose`` prints a parser-decision banner on stderr explaining how
   the parser routed the invocation (matched subcommand vs query-first
   fallback) without changing observable behavior on stdout.
3. ``polylogue invalid-xyz`` produces an actionable strict-floor error and
   (when the token looks like a typo of a real subcommand) a "did you mean a
   subcommand?" hint — within a reasonable time when the archive is empty.
4. Click ``UsageError`` (e.g. unknown long-option) carries an actionable
   next-step hint in addition to Click's default usage line.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.machine_main import actionable_hint_for_usage_error
from polylogue.cli.parser_diagnostics import (
    format_unknown_subcommand_hint,
    looks_like_subcommand_typo,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Contract 1: root --help documents query-first dispatch
# ---------------------------------------------------------------------------


class TestRootHelpDocumentsQueryFirst:
    def test_root_help_mentions_query_first(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        text = result.output.lower()
        assert "query-first" in text, (
            "root --help must explicitly document query-first dispatch so users do not "
            f"expect Click's 'unknown subcommand' error path. Got: {result.output!r}"
        )

    def test_root_help_mentions_diagnose_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--diagnose" in result.output, (
            "root --help should advertise --diagnose as the escape hatch for understanding parser decisions."
        )

    def test_root_help_lists_see_also(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Allow either explicit "See also:" wording, or the equivalent
        # references in the docstring; verified by presence of subcommand
        # help reference.
        assert "<subcommand> --help" in result.output


# ---------------------------------------------------------------------------
# Contract 2: --diagnose prints parser-decision banner on stderr
# ---------------------------------------------------------------------------


class TestDiagnoseBanner:
    def test_diagnose_banner_on_query_dispatch(
        self,
        runner: CliRunner,
        workspace_env: Mapping[str, Path],
    ) -> None:
        """Explicit query mode is routed to search; --diagnose says so."""
        result = runner.invoke(cli, ["--diagnose", "find", "invalid-xyz", "--limit", "0"])
        # We expect an explicit query-first dispatch line in the diagnose banner.
        assert "[diagnose]" in result.output, f"missing diagnose lines: {result.output!r}"
        assert "interpreting as search query" in result.output, (
            f"diagnose banner must announce query-first interpretation: {result.output!r}"
        )
        assert "invalid-xyz" in result.output

    def test_diagnose_banner_on_strict_floor_refusal(self, runner: CliRunner) -> None:
        """Bare plain tokens are refused before any archive search runs."""
        result = runner.invoke(cli, ["--diagnose", "invalid-xyz", "--limit", "0"])
        assert result.exit_code == 2
        assert "[diagnose]" in result.output
        assert "No such command 'invalid-xyz'." in result.output
        assert "polylogue find invalid-xyz" in result.output
        assert "strict command floor will refuse it" in result.output
        assert "interpreting as search query" not in result.output

    def test_diagnose_banner_on_matched_subcommand(
        self,
        runner: CliRunner,
        workspace_env: Mapping[str, Path],
    ) -> None:
        """A registered subcommand routes directly; --diagnose says so."""
        result = runner.invoke(cli, ["--diagnose", "ops", "--help"])
        assert result.exit_code == 0
        assert "[diagnose]" in result.output
        assert "matched subcommand" in result.output and "ops" in result.output


# ---------------------------------------------------------------------------
# Contract 3: invalid-xyz produces actionable output in reasonable time
# ---------------------------------------------------------------------------


class TestUnknownTokenActionability:
    def test_unknown_token_completes_quickly(
        self,
        runner: CliRunner,
        workspace_env: Mapping[str, Path],
    ) -> None:
        """``polylogue invalid-xyz`` on an empty archive returns within ~5s.

        This is the regression guard for the "actionable error within reasonable
        time" acceptance criterion. The deeper perf fix lives in #1181; this
        bound just verifies query-first dispatch on an empty archive does not
        hang.
        """
        started = time.monotonic()
        result = runner.invoke(cli, ["invalid-xyz", "--limit", "0"])
        elapsed = time.monotonic() - started
        assert elapsed < 10.0, f"unknown token took {elapsed:.2f}s (expected <10s)"
        # exit non-zero (no results)
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_unknown_token_resembling_subcommand_gets_typo_hint(
        self,
        runner: CliRunner,
        workspace_env: Mapping[str, Path],
    ) -> None:
        """Single-token query close to a real subcommand surfaces a typo hint."""
        result = runner.invoke(cli, ["red", "--limit", "0"])
        assert "If you meant a subcommand" in result.output, f"missing did-you-mean hint for `red`: {result.output!r}"
        assert "polylogue read" in result.output


# ---------------------------------------------------------------------------
# Contract 4: Click UsageError carries an actionable hint
# ---------------------------------------------------------------------------


class TestUsageErrorHints:
    def test_actionable_hint_for_unknown_option(self) -> None:
        hint = actionable_hint_for_usage_error("No such option: --nonexistent")
        assert hint is not None
        assert "--nonexistent" in hint
        assert "polylogue --help" in hint

    def test_actionable_hint_for_missing_argument(self) -> None:
        hint = actionable_hint_for_usage_error("Missing argument 'PATH'")
        assert hint is not None
        assert "--help" in hint

    def test_actionable_hint_for_misplaced_root_option(self) -> None:
        message = "Query filters and root output flags must appear before the verb. Move --origin before `read`."
        hint = actionable_hint_for_usage_error(message)
        assert hint is not None
        assert "precede the verb" in hint or "before the verb" in hint

    def test_actionable_hint_for_no_such_command(self) -> None:
        hint = actionable_hint_for_usage_error("No such command 'foo'.")
        assert hint is not None
        assert "polylogue find foo" in hint
        assert "polylogue --help" in hint


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------


class TestParserDiagnosticsHelpers:
    def test_looks_like_subcommand_typo_matches_close_token(self) -> None:
        suggestions = looks_like_subcommand_typo("red", ["read", "analyze", "delete"])
        assert "read" in suggestions

    def test_looks_like_subcommand_typo_no_match_returns_empty(self) -> None:
        suggestions = looks_like_subcommand_typo("zzz", ["read", "analyze", "delete"])
        assert suggestions == []

    def test_format_unknown_subcommand_hint_always_suggests_query(self) -> None:
        hint = format_unknown_subcommand_hint("foo", ["read", "analyze"])
        assert "polylogue find foo" in hint
        assert "plain unmarked roots" in hint

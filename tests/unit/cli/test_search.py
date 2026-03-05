"""Tests for CLI search functionality."""

from __future__ import annotations

import json

from click.testing import CliRunner
import pytest

from polylogue.cli import cli
from tests.infra.helpers import DbFactory

# =============================================================================
# TEST DATA TABLES (module-level constants)
# =============================================================================

SEARCH_FILTER_CASES = [
    ("provider", ["Python", "-p", "chatgpt"], 0, None),
    ("since_valid", ["Python", "--since", "__DYNAMIC_DATE__"], 0, None),
    ("since_invalid", ["Python", "--since", "not-a-date"], 1, "date"),
    ("limit_list", ["JavaScript", "--limit", "1", "--list"], 0, None),
]

SEARCH_FORMAT_CASES = [
    ("json_list", ["Python", "-f", "json", "--list"], "json_list"),
    ("json_single", ["JavaScript", "-f", "json", "--limit", "1"], "json_single"),
    ("list_mode", ["async", "--list"], "plain_list"),
    ("markdown", ["Rust", "-f", "markdown", "--limit", "1"], "markdown"),
]


class TestSearchQueryContracts:
    """Matrix coverage for search filters and output formats."""

    @pytest.mark.parametrize(
        "case_id,args,expected_exit,error_hint",
        SEARCH_FILTER_CASES,
    )
    def test_filter_contract(self, search_workspace, case_id, args, expected_exit, error_hint):
        """Filter flags produce expected status codes and validation behavior."""
        from datetime import datetime, timedelta

        runner = CliRunner()
        resolved_args = list(args)
        if "__DYNAMIC_DATE__" in resolved_args:
            idx = resolved_args.index("__DYNAMIC_DATE__")
            resolved_args[idx] = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        result = runner.invoke(cli, ["--plain", *resolved_args])
        assert result.exit_code == expected_exit, case_id
        if error_hint:
            assert error_hint in result.output.lower(), case_id

    @pytest.mark.parametrize(
        "case_id,args,expectation",
        SEARCH_FORMAT_CASES,
    )
    def test_output_contract(self, search_workspace, case_id, args, expectation):
        """Output format combinations produce parseable and mode-consistent output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", *args])
        assert result.exit_code == 0, case_id

        if expectation == "json_list":
            data = json.loads(result.output)
            assert isinstance(data, list), case_id
            assert data and "id" in data[0], case_id
        elif expectation == "json_single":
            data = json.loads(result.output)
            assert isinstance(data, (list, dict)), case_id
        elif expectation == "plain_list":
            assert result.output.strip(), case_id
        elif expectation == "markdown":
            assert "#" in result.output or "Rust" in result.output, case_id


class TestSearchEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_no_results(self, search_workspace):
        """Handle query with no matching results."""
        runner = CliRunner()
        # Query mode with non-matching term
        result = runner.invoke(cli, ["--plain", "nonexistent_term_xyz"])
        # exit_code 2 = no results (valid outcome)
        assert result.exit_code == 2
        assert "no conversation" in result.output.lower() or "matched" in result.output.lower()

    def test_stats_mode_no_filters(self, cli_workspace, monkeypatch):
        """Stats mode when no query terms or filters provided."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        runner = CliRunner()
        # No args = stats mode in query-first CLI
        result = runner.invoke(cli, ["--plain"])
        assert result.exit_code == 0
        # Should show stats, not require query

    def test_search_case_insensitive(self, search_workspace):
        """Search is case-insensitive."""
        runner = CliRunner()
        # Query mode with --list to ensure consistent output
        result_lower = runner.invoke(cli, ["--plain", "python", "-f", "json", "--list"])
        result_upper = runner.invoke(cli, ["--plain", "PYTHON", "-f", "json", "--list"])

        # Both should have same exit code
        assert result_lower.exit_code == result_upper.exit_code

        if result_lower.exit_code == 0:
            # Both should find results (FTS5 is case-insensitive by default)
            data_lower = json.loads(result_lower.output)
            data_upper = json.loads(result_upper.output)
            assert len(data_lower) > 0
            assert len(data_upper) > 0

    def test_search_multiple_terms(self, search_workspace):
        """Search with multiple query terms."""
        runner = CliRunner()
        # Query mode: multiple positional args = multiple query terms
        result = runner.invoke(cli, ["--plain", "Python", "exception", "-f", "json", "--list"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert isinstance(data, list)


class TestSearchIndexRebuild:
    """Tests for automatic index rebuild on missing index."""

    def test_search_handles_missing_index(self, cli_workspace, monkeypatch):
        """Search handles missing index gracefully."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Create conversation without building index
        db_path = cli_workspace["db_path"]
        factory = DbFactory(db_path)
        factory.create_conversation(
            id="c1",
            provider="test",
            title="Test",
            messages=[{"id": "m1", "role": "user", "text": "searchable content"}],
        )

        runner = CliRunner()
        # Query mode
        result = runner.invoke(cli, ["--plain", "searchable"])
        # Should either succeed (rebuild worked) or report no results.
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            assert "searchable" in result.output.lower() or "c1" in result.output
        else:
            assert "no conversation" in result.output.lower() or "matched" in result.output.lower()

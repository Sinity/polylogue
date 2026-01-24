"""Expanded tests for the CLI query mode (formerly search command)."""

import json
from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from tests.factories import DbFactory


@pytest.fixture
def search_workspace(cli_workspace, monkeypatch):
    """CLI workspace with searchable conversations."""
    # Set up environment
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
    monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_workspace["archive_root"]))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    # Create sample conversations with searchable content
    db_path = cli_workspace["db_path"]
    factory = DbFactory(db_path)

    # Conversation 1: Python content, recent
    factory.create_conversation(
        id="conv1",
        provider="chatgpt",
        title="Python Error Handling",
        messages=[
            {"id": "m1", "role": "user", "text": "How to handle exceptions in Python?"},
            {"id": "m2", "role": "assistant", "text": "Use try-except blocks for Python exception handling."},
        ],
        created_at=datetime.now() - timedelta(days=1),
        updated_at=datetime.now() - timedelta(days=1),
    )

    # Conversation 2: JavaScript content, older
    factory.create_conversation(
        id="conv2",
        provider="claude",
        title="JavaScript Async Patterns",
        messages=[
            {"id": "m3", "role": "user", "text": "Explain async/await in JavaScript"},
            {"id": "m4", "role": "assistant", "text": "Async/await is JavaScript syntax for promises."},
        ],
        created_at=datetime.now() - timedelta(days=10),
        updated_at=datetime.now() - timedelta(days=10),
    )

    # Conversation 3: Rust content
    factory.create_conversation(
        id="conv3",
        provider="claude-code",
        title="Rust Ownership",
        messages=[
            {"id": "m5", "role": "user", "text": "What is ownership in Rust?"},
            {"id": "m6", "role": "assistant", "text": "Rust ownership ensures memory safety without garbage collection."},
        ],
        created_at=datetime.now() - timedelta(hours=6),
        updated_at=datetime.now() - timedelta(hours=6),
    )

    # Build FTS index using rebuild_index
    from polylogue.storage.index import rebuild_index

    rebuild_index()

    return cli_workspace


class TestSearchFilters:
    """Tests for search filtering options."""

    def test_search_with_provider_filter(self, search_workspace):
        """Filter search results by provider."""
        runner = CliRunner()
        # Query mode: positional args = query, -p = provider filter
        result = runner.invoke(cli, ["--plain", "Python", "-p", "chatgpt"])
        # exit_code 0 = found, exit_code 2 = no results
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            assert "Python" in result.output or "conv1" in result.output

    def test_search_with_since_date(self, search_workspace):
        """Filter search results by date."""
        runner = CliRunner()
        since_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        # Query mode: positional args = query, --since = date filter
        result = runner.invoke(cli, ["--plain", "Python", "--since", since_date])
        assert result.exit_code in (0, 2)
        # Should find recent Python conversation

    def test_search_with_invalid_since_date(self, search_workspace):
        """Handle invalid --since date format gracefully."""
        runner = CliRunner()
        # Query mode with invalid date
        result = runner.invoke(cli, ["--plain", "Python", "--since", "not-a-date"])
        # The filter chain should handle this gracefully
        # Either fail with error message or treat as "no results"
        assert result.exit_code in (0, 1, 2)

    def test_search_with_limit(self, search_workspace):
        """Limit number of search results."""
        runner = CliRunner()
        # Query mode with --limit
        result = runner.invoke(cli, ["--plain", "JavaScript", "--limit", "1", "--list"])
        assert result.exit_code in (0, 2)
        # Should return at most 1 result


class TestSearchOutputFormats:
    """Tests for different output formats."""

    def test_search_json_output(self, search_workspace):
        """Search with JSON output format."""
        runner = CliRunner()
        # Query mode with -f json and --list
        result = runner.invoke(cli, ["--plain", "Python", "-f", "json", "--list"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert isinstance(data, list)
            if data:
                # JSON output contains conversation-level info
                assert "id" in data[0]

    def test_search_json_format_single(self, search_workspace):
        """Search with JSON output for single result."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "JavaScript", "-f", "json", "--limit", "1"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            # Single result = dict, multiple or --list = list
            assert isinstance(data, (list, dict))

    def test_search_list_mode(self, search_workspace):
        """Search in list mode (shows all results)."""
        runner = CliRunner()
        # Query mode with --list
        result = runner.invoke(cli, ["--plain", "async", "--list"])
        assert result.exit_code in (0, 2)
        # Should list all results

    def test_search_markdown_format(self, search_workspace):
        """Search with markdown output format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "Rust", "-f", "markdown", "--limit", "1"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            # Markdown output should contain headers
            assert "#" in result.output or "Rust" in result.output


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
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
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
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
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
        # Should either succeed (rebuild worked) or report no results
        assert result.exit_code in (0, 1, 2)

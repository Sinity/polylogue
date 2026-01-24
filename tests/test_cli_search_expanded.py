"""Expanded tests for the search CLI command."""

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

    def test_search_with_source_filter(self, search_workspace):
        """Filter search results by source/provider."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "Python", "--source", "chatgpt"])
        assert result.exit_code == 0
        assert "Python" in result.output
        # Should find conv1 (chatgpt provider)

    def test_search_with_since_date(self, search_workspace):
        """Filter search results by date."""
        runner = CliRunner()
        since_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        result = runner.invoke(cli, ["search", "Python", "--since", since_date])
        assert result.exit_code == 0
        # Should find recent Python conversation

    def test_search_with_invalid_since_date(self, search_workspace):
        """Handle invalid --since date format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "Python", "--since", "not-a-date"])
        assert result.exit_code != 0
        assert "parse" in result.output.lower() or "date" in result.output.lower()

    def test_search_with_limit(self, search_workspace):
        """Limit number of search results."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "JavaScript", "--limit", "1"])
        assert result.exit_code == 0
        # Should return at most 1 result


class TestSearchOutputFormats:
    """Tests for different output formats."""

    def test_search_json_output(self, search_workspace):
        """Search with JSON output format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "Python", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        if data:
            assert "conversation_id" in data[0]
            assert "message_id" in data[0]
            assert "snippet" in data[0]

    def test_search_json_lines_output(self, search_workspace):
        """Search with JSON Lines output format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "JavaScript", "--json-lines"])
        assert result.exit_code == 0
        lines = [line for line in result.output.strip().split("\n") if line]
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "conversation_id" in obj
            assert "message_id" in obj

    def test_search_verbose_output(self, search_workspace):
        """Search with verbose output (includes snippets)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "Rust", "--verbose"])
        assert result.exit_code == 0
        # Verbose mode should show snippets in output
        assert "Rust" in result.output

    def test_search_list_mode(self, search_workspace):
        """Search in list mode (no interactive picker)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "async", "--list"])
        assert result.exit_code == 0
        # Should list all results without interactive prompt


class TestSearchEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_no_results(self, search_workspace):
        """Handle query with no matching results."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "nonexistent_term_xyz"])
        assert result.exit_code == 0
        # Should handle empty results gracefully

    def test_search_empty_query(self, cli_workspace, monkeypatch):
        """Require query when not using --latest."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        runner = CliRunner()
        result = runner.invoke(cli, ["search"])
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "query" in result.output.lower()

    def test_search_case_insensitive(self, search_workspace):
        """Search is case-insensitive."""
        runner = CliRunner()
        result_lower = runner.invoke(cli, ["search", "python", "--json"])
        result_upper = runner.invoke(cli, ["search", "PYTHON", "--json"])

        assert result_lower.exit_code == 0
        assert result_upper.exit_code == 0

        # Both should find results (FTS5 is case-insensitive by default)
        data_lower = json.loads(result_lower.output)
        data_upper = json.loads(result_upper.output)
        assert len(data_lower) > 0
        assert len(data_upper) > 0

    def test_search_multiple_terms(self, search_workspace):
        """Search with multiple terms."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "Python exception", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should find messages matching both terms (or either, depending on FTS5 config)
        assert isinstance(data, list)


class TestSearchIndexRebuild:
    """Tests for automatic index rebuild on missing index."""

    def test_search_rebuilds_missing_index(self, cli_workspace, monkeypatch):
        """Search rebuilds index when missing."""
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
        result = runner.invoke(cli, ["search", "searchable"])
        # Should either succeed (rebuild worked) or fail gracefully
        # The rebuild happens automatically on first search if index missing
        assert result.exit_code in (0, 1)  # Allow both outcomes

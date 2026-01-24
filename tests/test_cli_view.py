"""Tests for the view CLI command."""

import json
from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from tests.factories import DbFactory


@pytest.fixture
def populated_workspace(cli_workspace, monkeypatch):
    """CLI workspace with sample conversations for testing."""
    # Set up environment
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
    monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_workspace["archive_root"]))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    # Create sample conversations with DbFactory
    db_path = cli_workspace["db_path"]
    factory = DbFactory(db_path)

    # Conversation 1: Recent, chatgpt
    factory.create_conversation(
        id="conv1",
        provider="chatgpt",
        title="Python Testing Guide",
        messages=[
            {"id": "m1", "role": "user", "text": "How do I test Python code?"},
            {"id": "m2", "role": "assistant", "text": "Use pytest for Python testing."},
        ],
        created_at=datetime.now() - timedelta(days=1),
        updated_at=datetime.now() - timedelta(days=1),
    )

    # Conversation 2: Older, claude
    factory.create_conversation(
        id="conv2",
        provider="claude",
        title="JavaScript Frameworks",
        messages=[
            {"id": "m5", "role": "user", "text": "What are popular JS frameworks?"},
            {"id": "m6", "role": "assistant", "text": "React, Vue, Angular are popular."},
        ],
        created_at=datetime.now() - timedelta(days=10),
        updated_at=datetime.now() - timedelta(days=10),
    )

    return cli_workspace


class TestViewList:
    """Tests for listing conversations."""

    def test_list_all_conversations(self, populated_workspace):
        """List all conversations without filters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view"])
        assert result.exit_code == 0
        assert "conv1" in result.output
        assert "conv2" in result.output

    def test_list_with_limit(self, populated_workspace):
        """List conversations with limit."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "--limit", "1"])
        assert result.exit_code == 0

    def test_list_filter_by_provider(self, populated_workspace):
        """Filter conversations by provider."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "--provider", "chatgpt"])
        assert result.exit_code == 0
        assert "conv1" in result.output

    def test_list_with_since_date(self, populated_workspace):
        """Filter conversations updated since date."""
        runner = CliRunner()
        since_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        result = runner.invoke(cli, ["view", "--since", since_date])
        assert result.exit_code == 0
        assert "conv1" in result.output

    def test_list_with_until_date(self, populated_workspace):
        """Filter conversations updated until date."""
        runner = CliRunner()
        until_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        result = runner.invoke(cli, ["view", "--until", until_date])
        assert result.exit_code == 0
        assert "conv2" in result.output

    def test_list_empty_results(self, cli_workspace, monkeypatch):
        """Handle empty conversation list gracefully."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        runner = CliRunner()
        result = runner.invoke(cli, ["view"])
        assert result.exit_code == 0
        assert "No conversations found" in result.output


class TestViewSingle:
    """Tests for viewing a single conversation."""

    def test_view_single_conversation(self, populated_workspace):
        """View a single conversation by ID."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1"])
        assert result.exit_code == 0
        assert "pytest" in result.output

    def test_view_nonexistent_conversation(self, populated_workspace):
        """Fail gracefully when conversation not found."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "nonexistent"])
        assert result.exit_code != 0


class TestProjections:
    """Tests for different projection types."""

    def test_projection_full(self, populated_workspace):
        """Apply 'full' projection (no filtering)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "-p", "full"])
        assert result.exit_code == 0
        assert "pytest" in result.output

    def test_projection_dialogue(self, populated_workspace):
        """Apply 'dialogue' projection."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "-p", "dialogue"])
        assert result.exit_code == 0

    def test_projection_clean(self, populated_workspace):
        """Apply 'clean' projection (default)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1"])
        assert result.exit_code == 0
        assert "pytest" in result.output

    def test_projection_pairs(self, populated_workspace):
        """Apply 'pairs' projection."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "-p", "pairs"])
        assert result.exit_code == 0
        assert "Turn" in result.output

    def test_projection_user(self, populated_workspace):
        """Apply 'user' projection."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "-p", "user"])
        assert result.exit_code == 0

    def test_projection_assistant(self, populated_workspace):
        """Apply 'assistant' projection."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "-p", "assistant"])
        assert result.exit_code == 0

    def test_projection_stats(self, populated_workspace):
        """Apply 'stats' projection."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "-p", "stats"])
        assert result.exit_code == 0
        assert "message_count" in result.output


class TestOutputFormats:
    """Tests for different output formats."""

    def test_output_text_default(self, populated_workspace):
        """Default text output format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1"])
        assert result.exit_code == 0
        assert "pytest" in result.output

    def test_output_json(self, populated_workspace):
        """JSON output format for single conversation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "id" in data
        assert data["id"] == "conv1"

    def test_output_json_multiple(self, populated_workspace):
        """JSON output format for multiple conversations."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_output_json_lines(self, populated_workspace):
        """JSON Lines output format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "--json-lines"])
        assert result.exit_code == 0
        lines = [line for line in result.output.strip().split("\n") if line]
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "id" in obj

    def test_output_list_mode(self, populated_workspace):
        """List mode output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "--list"])
        assert result.exit_code == 0
        assert "conv1" in result.output

    def test_output_verbose(self, populated_workspace):
        """Verbose output includes metadata."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "--verbose"])
        assert result.exit_code == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_projection(self, populated_workspace):
        """Reject invalid projection type."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "conv1", "-p", "invalid"])
        assert result.exit_code != 0

    def test_invalid_since_date(self, populated_workspace):
        """Handle invalid --since date format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["view", "--since", "not-a-date"])
        assert result.exit_code != 0

    def test_combined_filters(self, populated_workspace):
        """Combine multiple filters."""
        runner = CliRunner()
        since_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        result = runner.invoke(cli, ["view", "--provider", "chatgpt", "--since", since_date, "-p", "stats"])
        assert result.exit_code == 0

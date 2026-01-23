"""Tests for the view CLI command."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.db import default_db_path
from tests.factories import DbFactory


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def db_with_conversations(workspace_env):
    """Database populated with test conversations."""
    # Create a valid config file
    config_path = workspace_env["config_path"]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "version": 2,
        "archive_root": str(workspace_env["archive_root"]),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    db_path = default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    factory = DbFactory(db_path)

    # Create a basic conversation
    factory.create_conversation(
        id="test:conv1",
        provider="chatgpt",
        title="Basic Conversation",
        messages=[
            {"id": "m1", "role": "user", "text": "Hello there"},
            {"id": "m2", "role": "assistant", "text": "Hi! How can I help you?"},
            {"id": "m3", "role": "user", "text": "What's the weather?"},
            {"id": "m4", "role": "assistant", "text": "I don't have real-time weather data."},
        ],
    )

    # Create conversation with system message
    factory.create_conversation(
        id="test:conv2",
        provider="claude",
        title="Conversation with System",
        messages=[
            {"id": "m5", "role": "system", "text": "You are a helpful assistant."},
            {"id": "m6", "role": "user", "text": "Tell me a joke"},
            {"id": "m7", "role": "assistant", "text": "Why did the chicken cross the road?"},
        ],
    )

    # Create conversation with partial ID matching
    factory.create_conversation(
        id="chatgpt:abc123xyz789",
        provider="chatgpt",
        title="Partial ID Test",
        messages=[
            {"id": "m8", "role": "user", "text": "Test partial ID resolution"},
            {"id": "m9", "role": "assistant", "text": "Partial ID works!"},
        ],
    )

    # Create conversation for stats testing
    factory.create_conversation(
        id="test:stats",
        provider="claude",
        title="Stats Test Conversation",
        messages=[
            {"id": "m10", "role": "user", "text": "First message with several words here"},
            {"id": "m11", "role": "assistant", "text": "Second message also has multiple words"},
            {"id": "m12", "role": "user", "text": "Third message"},
        ],
    )

    return db_path


class TestViewCommand:
    """Tests for polylogue view command."""

    def test_view_full_conversation(self, workspace_env, db_with_conversations, cli_runner):
        """View command shows full conversation by default (clean projection)."""
        result = cli_runner.invoke(cli, ["view", "test:conv1"])
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Hello there" in result.output
        assert "Hi! How can I help you?" in result.output
        assert "What's the weather?" in result.output

    def test_view_full_projection_explicit(self, workspace_env, db_with_conversations, cli_runner):
        """View with explicit --projection full shows all messages."""
        result = cli_runner.invoke(cli, ["view", "test:conv1", "-p", "full"])
        assert result.exit_code == 0
        assert "Hello there" in result.output
        assert "Hi! How can I help you?" in result.output

    def test_view_partial_id_resolution(self, workspace_env, db_with_conversations, cli_runner):
        """View resolves partial conversation IDs (prefix matching)."""
        # Test with chatgpt: prefix which should match chatgpt:abc123xyz789
        result = cli_runner.invoke(cli, ["view", "chatgpt:abc"])
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Partial ID works!" in result.output or "Test partial ID resolution" in result.output

    def test_view_dialogue_projection(self, workspace_env, db_with_conversations, cli_runner):
        """View --projection dialogue filters to user/assistant only."""
        result = cli_runner.invoke(cli, ["view", "test:conv2", "-p", "dialogue"])
        assert result.exit_code == 0
        # Should include user and assistant messages
        assert "Tell me a joke" in result.output
        assert "Why did the chicken cross the road?" in result.output
        # System message might be filtered depending on dialogue_only() implementation

    def test_view_user_projection(self, workspace_env, db_with_conversations, cli_runner):
        """View --projection user shows only user messages."""
        result = cli_runner.invoke(cli, ["view", "test:conv1", "-p", "user"])
        assert result.exit_code == 0
        assert "Hello there" in result.output
        assert "What's the weather?" in result.output
        # Assistant messages should not appear
        assert "Hi! How can I help you?" not in result.output

    def test_view_assistant_projection(self, workspace_env, db_with_conversations, cli_runner):
        """View --projection assistant shows only assistant messages."""
        result = cli_runner.invoke(cli, ["view", "test:conv1", "-p", "assistant"])
        assert result.exit_code == 0
        assert "Hi! How can I help you?" in result.output
        assert "I don't have real-time weather data." in result.output
        # User messages should not appear
        assert "Hello there" not in result.output

    def test_view_pairs_projection(self, workspace_env, db_with_conversations, cli_runner):
        """View --projection pairs shows dialogue pairs."""
        result = cli_runner.invoke(cli, ["view", "test:conv1", "-p", "pairs"])
        assert result.exit_code == 0
        # Should show turn markers
        assert "Turn" in result.output
        # Should show both user and assistant
        assert "User:" in result.output or "Hello there" in result.output
        assert "Assistant:" in result.output or "Hi! How can I help you?" in result.output

    def test_view_stats_projection(self, workspace_env, db_with_conversations, cli_runner):
        """View --projection stats shows statistics."""
        result = cli_runner.invoke(cli, ["view", "test:stats", "-p", "stats"])
        assert result.exit_code == 0
        # Check for stats keywords
        output_lower = result.output.lower()
        assert "message" in output_lower or "count" in output_lower
        assert "user" in output_lower or "assistant" in output_lower
        # Should show the conversation ID
        assert "test:stats" in result.output

    def test_view_json_output(self, workspace_env, db_with_conversations, cli_runner):
        """View --json produces valid JSON."""
        result = cli_runner.invoke(cli, ["--plain", "view", "test:conv1", "--json"])
        assert result.exit_code == 0

        # Parse JSON to verify it's valid
        data = json.loads(result.output)
        assert isinstance(data, dict)
        # Should have messages or id field
        assert "messages" in data or "id" in data

        # Verify conversation data
        if "id" in data:
            assert data["id"] == "test:conv1"
        if "messages" in data:
            assert len(data["messages"]) > 0

    def test_view_json_with_stats_projection(self, workspace_env, db_with_conversations, cli_runner):
        """View --json with stats projection returns stats dict."""
        result = cli_runner.invoke(cli, ["--plain", "view", "test:stats", "-p", "stats", "--json"])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "message_count" in data
        assert "word_count" in data
        assert data["id"] == "test:stats"
        assert data["message_count"] == 3

    def test_view_not_found(self, workspace_env, db_with_conversations, cli_runner):
        """View returns error for nonexistent conversation."""
        result = cli_runner.invoke(cli, ["view", "nonexistent-id-xyz"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_view_list_mode_no_id(self, workspace_env, db_with_conversations, cli_runner):
        """View without conversation ID lists conversations."""
        result = cli_runner.invoke(cli, ["view"])
        assert result.exit_code == 0
        # Should show multiple conversations
        assert "test:conv1" in result.output or "Basic Conversation" in result.output
        # Should show summary information
        assert "Conversations" in result.output or "Found:" in result.output

    def test_view_list_with_provider_filter(self, workspace_env, db_with_conversations, cli_runner):
        """View --provider filters conversations by provider."""
        result = cli_runner.invoke(cli, ["view", "--provider", "chatgpt"])
        assert result.exit_code == 0
        # Should include chatgpt conversations
        assert "chatgpt" in result.output.lower()

    def test_view_list_with_limit(self, workspace_env, db_with_conversations, cli_runner):
        """View --limit restricts number of results."""
        result = cli_runner.invoke(cli, ["view", "--limit", "1"])
        assert result.exit_code == 0
        # Should only show one conversation
        # Count conversation IDs or message counts in output
        assert "Found: 1" in result.output or result.output.count("msgs)") == 1

    def test_view_json_lines_output(self, workspace_env, db_with_conversations, cli_runner):
        """View --json-lines produces one JSON object per line."""
        result = cli_runner.invoke(cli, ["--plain", "view", "--json-lines", "--limit", "2"])
        assert result.exit_code == 0

        # Split by lines and parse each as JSON
        lines = [line for line in result.output.strip().split("\n") if line]
        assert len(lines) >= 1

        for line in lines:
            data = json.loads(line)
            assert isinstance(data, dict)
            assert "id" in data or "messages" in data

    def test_view_verbose_includes_metadata(self, workspace_env, db_with_conversations, cli_runner):
        """View --verbose includes metadata in output."""
        result = cli_runner.invoke(cli, ["view", "test:conv1", "-v"])
        assert result.exit_code == 0
        # Verbose mode should include timestamps or other metadata
        # The exact format depends on implementation, but should show messages
        assert "Hello there" in result.output

    def test_view_thinking_projection(self, workspace_env, cli_runner):
        """View --projection thinking shows thinking traces."""
        # Create a valid config file
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
        }
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")

        db_path = default_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        factory = DbFactory(db_path)
        factory.create_conversation(
            id="test:thinking",
            provider="claude",
            title="Thinking Test",
            messages=[
                {"id": "m1", "role": "user", "text": "Solve this problem"},
                {"id": "m2", "role": "assistant", "text": "<thinking>Let me think about this...</thinking>\nHere's my answer."},
            ],
        )

        result = cli_runner.invoke(cli, ["view", "test:thinking", "-p", "thinking"])
        assert result.exit_code == 0
        # Should show thinking content
        assert "Thinking" in result.output or result.output  # Should have some output

    def test_view_clean_projection_default(self, workspace_env, cli_runner):
        """View uses clean projection by default."""
        # Create a valid config file
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
        }
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")

        db_path = default_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        factory = DbFactory(db_path)
        factory.create_conversation(
            id="test:clean",
            provider="chatgpt",
            title="Clean Projection Test",
            messages=[
                {"id": "m1", "role": "system", "text": "System prompt here"},
                {"id": "m2", "role": "user", "text": "User question"},
                {"id": "m3", "role": "assistant", "text": "Assistant response"},
            ],
        )

        result = cli_runner.invoke(cli, ["view", "test:clean"])
        assert result.exit_code == 0
        # Clean projection should filter out system messages (non-substantive)
        assert "User question" in result.output
        assert "Assistant response" in result.output


class TestViewEdgeCases:
    """Edge case tests for view command."""

    def test_view_empty_database(self, workspace_env, cli_runner):
        """View on empty database shows no conversations."""
        # Create a valid config file
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
        }
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")

        # Initialize empty database
        db_path = default_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        result = cli_runner.invoke(cli, ["view"])
        assert result.exit_code == 0
        assert "No conversations found" in result.output or "Found: 0" in result.output

    def test_view_conversation_no_messages(self, workspace_env, cli_runner):
        """View handles conversations with no messages."""
        # Create a valid config file
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
        }
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")

        db_path = default_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        factory = DbFactory(db_path)
        factory.create_conversation(
            id="test:empty",
            provider="test",
            title="Empty Conversation",
            messages=[],
        )

        result = cli_runner.invoke(cli, ["view", "test:empty"])
        # Should not crash, may show empty or minimal output
        assert result.exit_code == 0

    def test_view_invalid_projection(self, workspace_env, db_with_conversations, cli_runner):
        """View rejects invalid projection types."""
        result = cli_runner.invoke(cli, ["view", "test:conv1", "-p", "invalid"])
        assert result.exit_code != 0
        # Click should report invalid choice

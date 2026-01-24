"""Integration tests for the Polylogue facade (high-level library API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.config import Source


@pytest.fixture
def sample_chatgpt_file(tmp_path):
    """Create a sample ChatGPT export file."""
    export = {
        "title": "Python Help",
        "mapping": {
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": ["msg_1"],
            },
            "msg_1": {
                "id": "msg_1",
                "message": {
                    "id": "msg_1",
                    "author": {"role": "user"},
                    "create_time": 1234567890.0,
                    "content": {"content_type": "text", "parts": ["How do I use Python?"]},
                },
                "parent": "root",
                "children": ["msg_2"],
            },
            "msg_2": {
                "id": "msg_2",
                "message": {
                    "id": "msg_2",
                    "author": {"role": "assistant"},
                    "create_time": 1234567891.0,
                    "content": {
                        "content_type": "text",
                        "parts": ["Python is a high-level programming language. You can start by installing Python and running simple scripts."],
                    },
                },
                "parent": "msg_1",
                "children": [],
            },
        },
    }

    file_path = tmp_path / "chatgpt_export.json"
    file_path.write_text(json.dumps(export), encoding="utf-8")
    return file_path


@pytest.fixture
def sample_claude_file(tmp_path):
    """Create a sample Claude AI export file."""
    # Claude AI exports are JSONL with chat_messages array
    export = {
        "uuid": "claude_1",
        "name": "Code Review",
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T12:05:00Z",
        "chat_messages": [
            {
                "uuid": "claude_1_msg_1",
                "text": "Can you review my code?",
                "sender": "human",
                "created_at": "2024-01-01T12:00:00Z",
            },
            {
                "uuid": "claude_1_msg_2",
                "text": "I'd be happy to review your code. Please share the code you'd like me to look at.",
                "sender": "assistant",
                "created_at": "2024-01-01T12:01:00Z",
            },
        ],
    }

    file_path = tmp_path / "claude_export.jsonl"
    file_path.write_text(json.dumps(export), encoding="utf-8")
    return file_path


class TestPolylogueInitialization:
    """Test Polylogue initialization."""

    def test_init_with_defaults(self, workspace_env):
        """Test initialization with default paths."""
        archive = Polylogue()
        assert archive is not None
        assert archive.archive_root is not None
        assert archive.config is not None
        assert archive.repository is not None

    def test_init_with_custom_archive_root(self, tmp_path):
        """Test initialization with custom archive root."""
        custom_root = tmp_path / "custom_archive"
        archive = Polylogue(archive_root=custom_root)
        assert archive.archive_root == custom_root

    def test_init_with_expanduser(self):
        """Test initialization with ~ in path."""
        archive = Polylogue(archive_root="~/test_polylogue")
        assert "~" not in str(archive.archive_root)
        assert archive.archive_root.is_absolute()

    def test_repr(self, tmp_path):
        """Test string representation."""
        archive = Polylogue(archive_root=tmp_path / "archive")
        repr_str = repr(archive)
        assert "Polylogue" in repr_str
        assert "archive" in repr_str


class TestPolylogueIngest:
    """Test ingestion functionality."""

    def test_ingest_chatgpt_file(self, workspace_env, sample_chatgpt_file):
        """Test ingesting a ChatGPT export file."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        result = archive.ingest_file(sample_chatgpt_file)

        # Verify counts
        assert result.counts["conversations"] > 0
        assert result.counts["messages"] > 0

        # Verify we can retrieve the conversation
        conversations = archive.list_conversations(provider="chatgpt")
        assert len(conversations) > 0
        assert conversations[0].title == "Python Help"

    def test_ingest_claude_file(self, workspace_env, sample_claude_file):
        """Test ingesting a Claude AI export file."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        result = archive.ingest_file(sample_claude_file)

        # Verify counts
        assert result.counts["conversations"] > 0
        assert result.counts["messages"] > 0

        # Verify we can retrieve the conversation
        conversations = archive.list_conversations(provider="claude")
        assert len(conversations) > 0

    def test_ingest_duplicate_is_idempotent(self, workspace_env, sample_chatgpt_file):
        """Test that re-ingesting the same file skips duplicates."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # First ingest
        result1 = archive.ingest_file(sample_chatgpt_file)
        assert result1.counts["conversations"] > 0

        # Second ingest (should skip)
        result2 = archive.ingest_file(sample_chatgpt_file)
        assert result2.counts["skipped_conversations"] > 0
        assert result2.counts["conversations"] == 0  # No new conversations

    def test_ingest_with_custom_source_name(self, workspace_env, sample_chatgpt_file):
        """Test ingesting with a custom source name."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        result = archive.ingest_file(sample_chatgpt_file, source_name="my_custom_source")
        assert result.counts["conversations"] > 0

    def test_ingest_sources(self, workspace_env, sample_chatgpt_file, sample_claude_file):
        """Test ingesting multiple sources."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        sources = [
            Source(name="chatgpt", path=sample_chatgpt_file),
            Source(name="claude", path=sample_claude_file),
        ]

        result = archive.ingest_sources(sources=sources, download_assets=False)

        # Verify both sources were ingested
        assert result.counts["conversations"] >= 2

        # Verify we can list conversations from both providers
        chatgpt_convs = archive.list_conversations(provider="chatgpt")
        claude_convs = archive.list_conversations(provider="claude")
        assert len(chatgpt_convs) > 0
        assert len(claude_convs) > 0


class TestPolylogueQuery:
    """Test query functionality."""

    def test_get_conversation_by_full_id(self, workspace_env, sample_chatgpt_file):
        """Test getting a conversation by full ID."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest and get conversation ID
        archive.ingest_file(sample_chatgpt_file)
        conversations = archive.list_conversations(provider="chatgpt")
        conv_id = conversations[0].id

        # Retrieve by full ID
        conv = archive.get_conversation(conv_id)
        assert conv is not None
        assert conv.id == conv_id
        assert conv.title == "Python Help"

    def test_get_conversation_by_partial_id(self, workspace_env, sample_chatgpt_file):
        """Test getting a conversation by partial ID (prefix match)."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest and get conversation ID
        archive.ingest_file(sample_chatgpt_file)
        conversations = archive.list_conversations(provider="chatgpt")
        conv_id = conversations[0].id

        # Retrieve by partial ID (first 8 characters)
        partial_id = conv_id[:8]
        conv = archive.get_conversation(partial_id)
        assert conv is not None
        assert conv.id == conv_id

    def test_get_conversation_nonexistent(self, workspace_env):
        """Test getting a non-existent conversation returns None."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        conv = archive.get_conversation("nonexistent_id")
        assert conv is None

    def test_list_conversations_all(self, workspace_env, sample_chatgpt_file, sample_claude_file):
        """Test listing all conversations."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest both files
        archive.ingest_file(sample_chatgpt_file)
        archive.ingest_file(sample_claude_file)

        # List all
        all_convs = archive.list_conversations()
        assert len(all_convs) >= 2

    def test_list_conversations_filter_by_provider(self, workspace_env, sample_chatgpt_file, sample_claude_file):
        """Test filtering conversations by provider."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest both files
        archive.ingest_file(sample_chatgpt_file)
        archive.ingest_file(sample_claude_file)

        # Filter by provider
        chatgpt_convs = archive.list_conversations(provider="chatgpt")
        claude_convs = archive.list_conversations(provider="claude")

        assert len(chatgpt_convs) > 0
        assert len(claude_convs) > 0
        assert all(c.provider == "chatgpt" for c in chatgpt_convs)
        assert all(c.provider == "claude" for c in claude_convs)

    def test_list_conversations_with_limit(self, workspace_env, sample_chatgpt_file):
        """Test limiting the number of conversations returned."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        archive.ingest_file(sample_chatgpt_file)

        # List with limit
        convs = archive.list_conversations(limit=1)
        assert len(convs) == 1


class TestPolylogueSemanticProjections:
    """Test semantic projections on retrieved conversations."""

    def test_conversation_substantive_only(self, workspace_env, sample_chatgpt_file):
        """Test getting substantive messages only."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest
        archive.ingest_file(sample_chatgpt_file)
        conversations = archive.list_conversations()
        conv = conversations[0]

        # Get substantive messages
        substantive_conv = conv.substantive_only()
        substantive = substantive_conv.messages
        assert len(substantive) > 0
        assert all(msg.is_substantive for msg in substantive)

    def test_conversation_iter_pairs(self, workspace_env, sample_chatgpt_file):
        """Test iterating user/assistant pairs."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest
        archive.ingest_file(sample_chatgpt_file)
        conversations = archive.list_conversations()
        conv = conversations[0]

        # Get pairs
        pairs = list(conv.iter_pairs())
        assert len(pairs) > 0
        for pair in pairs:
            assert pair.user.role == "user"
            assert pair.assistant.role == "assistant"

    def test_conversation_without_noise(self, workspace_env, sample_chatgpt_file):
        """Test filtering out noise messages."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest
        archive.ingest_file(sample_chatgpt_file)
        conversations = archive.list_conversations()
        conv = conversations[0]

        # Get without noise
        clean = list(conv.without_noise())
        assert len(clean) > 0


class TestPolylogueSearch:
    """Test search functionality."""

    def test_search_basic(self, workspace_env, sample_chatgpt_file):
        """Test basic search functionality."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest
        archive.ingest_file(sample_chatgpt_file)

        # Rebuild index to ensure search works
        archive.rebuild_index()

        # Search for Python
        results = archive.search("Python")
        assert results is not None
        # Note: Results might be empty if FTS indexing is async

    def test_search_with_limit(self, workspace_env, sample_chatgpt_file):
        """Test search with result limit."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # Ingest
        archive.ingest_file(sample_chatgpt_file)

        # Rebuild index
        archive.rebuild_index()

        # Search with limit
        results = archive.search("Python", limit=5)
        assert results is not None
        assert len(results.hits) <= 5


class TestPolylogueEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_archive(self, workspace_env):
        """Test operations on an empty archive."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        # List should return empty
        convs = archive.list_conversations()
        assert len(convs) == 0

        # Get should return None
        conv = archive.get_conversation("nonexistent")
        assert conv is None

    def test_ingest_nonexistent_file(self, workspace_env, tmp_path):
        """Test ingesting a non-existent file."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["state_root"] / "polylogue.db",
        )

        nonexistent = tmp_path / "does_not_exist.json"

        # Should handle gracefully with empty counts
        result = archive.ingest_file(nonexistent)
        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0

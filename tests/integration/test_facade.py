"""Integration tests for high-level Polylogue facade workflows."""

from __future__ import annotations

import json

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
                        "parts": [
                            "Python is a high-level programming language. You can start by installing Python and running simple scripts."
                        ],
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


class TestPolylogueParsing:
    """Test ingestion functionality."""

    @pytest.mark.asyncio
    async def test_ingest_chatgpt_file(self, workspace_env, sample_chatgpt_file):
        """Test ingesting a ChatGPT export file."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        result = await archive.parse_file(sample_chatgpt_file)

        # Verify counts
        assert result.counts["conversations"] > 0
        assert result.counts["messages"] > 0

        # Verify we can retrieve the conversation
        conversations = await archive.list_conversations(provider="chatgpt")
        assert len(conversations) > 0
        assert conversations[0].title == "Python Help"

    @pytest.mark.asyncio
    async def test_ingest_claude_file(self, workspace_env, sample_claude_file):
        """Test ingesting a Claude AI export file."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        result = await archive.parse_file(sample_claude_file)

        # Verify counts
        assert result.counts["conversations"] > 0
        assert result.counts["messages"] > 0

        # Verify we can retrieve the conversation
        conversations = await archive.list_conversations(provider="claude-ai")
        assert len(conversations) > 0

    @pytest.mark.asyncio
    async def test_ingest_duplicate_is_idempotent(self, workspace_env, sample_chatgpt_file):
        """Test that re-ingesting the same file skips duplicates.

        With stage-based ingestion (acquire → parse), duplicates are skipped
        at the acquire stage (raw_id already exists), so the parse stage
        is never called and returns an empty result.
        """
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # First ingest
        result1 = await archive.parse_file(sample_chatgpt_file)
        first_count = result1.counts["conversations"]
        assert first_count > 0

        # Second ingest (acquire stage skips duplicate raw_id)
        result2 = await archive.parse_file(sample_chatgpt_file)
        # With stage architecture, no parsing happens when acquire skips
        assert result2.counts["conversations"] == 0

        # Verify DB still has same number of conversations (idempotent)
        conversations = await archive.list_conversations()
        assert len(conversations) == first_count

    @pytest.mark.asyncio
    async def test_ingest_with_custom_source_name(self, workspace_env, sample_chatgpt_file):
        """Test ingesting with a custom source name."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        result = await archive.parse_file(sample_chatgpt_file, source_name="my_custom_source")
        assert result.counts["conversations"] > 0

    @pytest.mark.asyncio
    async def test_parse_sources(self, workspace_env, sample_chatgpt_file, sample_claude_file):
        """Test ingesting multiple sources."""
        db_path = workspace_env["data_root"] / "polylogue.db"

        # Initialize database with WAL mode before concurrent ingestion
        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            conn.execute("SELECT 1").fetchone()

        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=db_path,
        )

        sources = [
            Source(name="chatgpt", path=sample_chatgpt_file),
            Source(name="claude-ai", path=sample_claude_file),
        ]

        result = await archive.parse_sources(sources=sources, download_assets=False)

        # Verify both sources were ingested
        assert result.counts["conversations"] >= 2

        # Verify we can list conversations from both providers
        chatgpt_convs = await archive.list_conversations(provider="chatgpt")
        claude_convs = await archive.list_conversations(provider="claude-ai")
        assert len(chatgpt_convs) > 0
        assert len(claude_convs) > 0


class TestPolylogueQuery:
    """Test query functionality."""

    @pytest.mark.asyncio
    async def test_get_conversation_by_full_id(self, workspace_env, sample_chatgpt_file):
        """Test getting a conversation by full ID."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest and get conversation ID
        await archive.parse_file(sample_chatgpt_file)
        conversations = await archive.list_conversations(provider="chatgpt")
        conv_id = conversations[0].id

        # Retrieve by full ID
        conv = await archive.get_conversation(conv_id)
        assert conv is not None
        assert conv.id == conv_id
        assert conv.title == "Python Help"

    @pytest.mark.asyncio
    async def test_get_conversation_by_partial_id(self, workspace_env, sample_chatgpt_file):
        """Test getting a conversation by partial ID (prefix match)."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest and get conversation ID
        await archive.parse_file(sample_chatgpt_file)
        conversations = await archive.list_conversations(provider="chatgpt")
        conv_id = conversations[0].id

        # Retrieve by partial ID (first 8 characters)
        partial_id = conv_id[:8]
        conv = await archive.get_conversation(partial_id)
        assert conv is not None
        assert conv.id == conv_id

    @pytest.mark.asyncio
    async def test_get_conversation_nonexistent(self, workspace_env):
        """Test getting a non-existent conversation returns None."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        conv = await archive.get_conversation("nonexistent_id")
        assert conv is None

    @pytest.mark.asyncio
    async def test_list_conversations_all(self, workspace_env, sample_chatgpt_file, sample_claude_file):
        """Test listing all conversations."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest both files
        await archive.parse_file(sample_chatgpt_file)
        await archive.parse_file(sample_claude_file)

        # List all
        all_convs = await archive.list_conversations()
        assert len(all_convs) >= 2

    @pytest.mark.asyncio
    async def test_list_conversations_filter_by_provider(self, workspace_env, sample_chatgpt_file, sample_claude_file):
        """Test filtering conversations by provider."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest both files
        await archive.parse_file(sample_chatgpt_file)
        await archive.parse_file(sample_claude_file)

        # Filter by provider
        chatgpt_convs = await archive.list_conversations(provider="chatgpt")
        claude_convs = await archive.list_conversations(provider="claude-ai")

        assert len(chatgpt_convs) > 0
        assert len(claude_convs) > 0
        assert all(c.provider == "chatgpt" for c in chatgpt_convs)
        assert all(c.provider == "claude-ai" for c in claude_convs)

    @pytest.mark.asyncio
    async def test_list_conversations_with_limit(self, workspace_env, sample_chatgpt_file):
        """Test limiting the number of conversations returned."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        await archive.parse_file(sample_chatgpt_file)

        # List with limit
        convs = await archive.list_conversations(limit=1)
        assert len(convs) == 1


class TestPolylogueSemanticProjections:
    """Test semantic projections on retrieved conversations."""

    @pytest.mark.asyncio
    async def test_conversation_substantive_only(self, workspace_env, sample_chatgpt_file):
        """Test getting substantive messages only."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest
        await archive.parse_file(sample_chatgpt_file)
        conversations = await archive.list_conversations()
        conv = conversations[0]

        # Get substantive messages
        substantive_conv = conv.substantive_only()
        substantive = substantive_conv.messages
        assert len(substantive) > 0
        assert all(msg.is_substantive for msg in substantive)

    @pytest.mark.asyncio
    async def test_conversation_iter_pairs(self, workspace_env, sample_chatgpt_file):
        """Test iterating user/assistant pairs."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest
        await archive.parse_file(sample_chatgpt_file)
        conversations = await archive.list_conversations()
        conv = conversations[0]

        # Get pairs
        pairs = list(conv.iter_pairs())
        assert len(pairs) > 0
        for pair in pairs:
            assert pair.user.role == "user"
            assert pair.assistant.role == "assistant"

    @pytest.mark.asyncio
    async def test_conversation_without_noise(self, workspace_env, sample_chatgpt_file):
        """Test filtering out noise messages."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest
        await archive.parse_file(sample_chatgpt_file)
        conversations = await archive.list_conversations()
        conv = conversations[0]

        # Get without noise
        clean = list(conv.without_noise())
        assert len(clean) > 0


class TestPolylogueSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_basic(self, workspace_env, sample_chatgpt_file):
        """Test basic search functionality."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest
        await archive.parse_file(sample_chatgpt_file)

        # Rebuild index to ensure search works
        await archive.rebuild_index()

        # Search for Python
        results = await archive.search("Python")
        assert results is not None
        assert results.hits
        first_hit = results.hits[0]
        assert first_hit.conversation_id
        assert first_hit.title
        assert first_hit.conversation_path.name == "conversation.md"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, workspace_env, sample_chatgpt_file):
        """Test search with result limit."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # Ingest
        await archive.parse_file(sample_chatgpt_file)

        # Rebuild index
        await archive.rebuild_index()

        # Search with limit
        results = await archive.search("Python", limit=5)
        assert results is not None
        assert len(results.hits) <= 5
        if results.hits:
            assert all(hit.provider_name for hit in results.hits)


class TestPolylogueEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_archive(self, workspace_env):
        """Test operations on an empty archive."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        # List should return empty
        convs = await archive.list_conversations()
        assert len(convs) == 0

        # Get should return None
        conv = await archive.get_conversation("nonexistent")
        assert conv is None

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self, workspace_env, tmp_path):
        """Test ingesting a non-existent file."""
        archive = Polylogue(
            archive_root=workspace_env["archive_root"],
            db_path=workspace_env["data_root"] / "polylogue.db",
        )

        nonexistent = tmp_path / "does_not_exist.json"

        # Should handle gracefully with empty counts
        result = await archive.parse_file(nonexistent)
        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0


# MERGED FROM test_facade_helpers_coverage.py

# ============================================================================
# ARCHIVE STATS TESTS
# ============================================================================

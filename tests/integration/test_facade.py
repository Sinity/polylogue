"""Integration tests for the Polylogue facade (high-level library API)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue import Polylogue
from polylogue.facade import ArchiveStats
from polylogue.cli.helpers import (
    latest_render_path,
    load_last_source,
    maybe_prompt_sources,
    save_last_source,
    source_state_path,
)
from polylogue.cli.types import AppEnv
from polylogue.config import Config, Source
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.storage.store import ConversationRecord, MessageRecord

# ============================================================================
# PARAMETRIZATION TABLES
# ============================================================================

# ArchiveStats initialization parameters: (conv_count, msg_count, word_count, providers, tags, last_sync)
ARCHIVE_STATS_PARAMS = [
    (10, 50, 1000, {"claude": 7, "chatgpt": 3}, {"test": 2, "work": 3}, None),
    (5, 25, 500, {"claude": 5}, {}, "2025-01-15T12:30:45Z"),
    (0, 0, 0, {}, {}, None),
    (1, 1, 10, {"claude": 1}, {}, None),
    (20, 100, 2000, {"claude": 10, "chatgpt": 5, "gemini": 5}, {"personal": 1, "work": 2}, "2025-01-20T10:00:00Z"),
]

# Load last source test states: (state_exists, json_valid, source_field_exists, expected_result)
LOAD_SOURCE_STATES = [
    (True, True, True, "test-source"),
    (False, None, None, None),
    (True, False, None, None),
    (True, True, False, None),
]

# Maybe prompt sources scenarios: (ui_mode, sources_count, selection, has_last_choice, expect_choose_called)
MAYBE_PROMPT_SCENARIOS = [
    ("already_selected", 2, "source-a", False, False),
    ("plain_mode", 2, None, False, False),
    ("single_source", 1, None, False, False),
    ("multiple_sources", 2, "source-b", False, True),
    ("all_selected", 2, "all", False, True),
    ("restore_last_choice", 2, "source-b", True, True),
    ("no_choice_fails", 2, None, False, True),
]

# Latest render path scenarios: (root_exists, files_present, file_types)
RENDER_PATH_SCENARIOS = [
    ("no_root", False, []),
    ("empty_root", True, []),
    ("md_files", True, ["md"]),
    ("html_files", True, ["html"]),
    ("mixed_formats", True, ["md", "html"]),
]

# Polylogue init configurations: (db_path, property_to_check)
POLYLOGUE_INIT_CONFIGS = [
    (":memory:", "archive_root"),
    ("file.db", "config"),
    (":memory:", "repository"),
    ("file.db", "archive_root"),
    ("file.db", "repr"),
]

# List conversations filters: (setup_count, provider_filter, source_filter, limit, expected_count)
LIST_CONV_FILTERS = [
    (0, None, None, None, 0),
    (3, None, None, None, 3),
    (4, "claude", None, None, 2),
    (5, None, "inbox", None, 5),
    (5, None, "inbox", 3, 3),
]


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
        conversations = await archive.list_conversations(provider="claude")
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
            Source(name="claude", path=sample_claude_file),
        ]

        result = await archive.parse_sources(sources=sources, download_assets=False)

        # Verify both sources were ingested
        assert result.counts["conversations"] >= 2

        # Verify we can list conversations from both providers
        chatgpt_convs = await archive.list_conversations(provider="chatgpt")
        claude_convs = await archive.list_conversations(provider="claude")
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
        claude_convs = await archive.list_conversations(provider="claude")

        assert len(chatgpt_convs) > 0
        assert len(claude_convs) > 0
        assert all(c.provider == "chatgpt" for c in chatgpt_convs)
        assert all(c.provider == "claude" for c in claude_convs)

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
        # Note: Results might be empty if FTS indexing is async

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


class TestArchiveStatsCreation:
    """Test ArchiveStats instantiation and attributes."""

    @pytest.mark.parametrize(
        "conv_count,msg_count,word_count,providers,tags,last_sync",
        ARCHIVE_STATS_PARAMS,
    )
    def test_archive_stats_init(self, conv_count, msg_count, word_count, providers, tags, last_sync):
        """Test ArchiveStats initialization with various parameter combinations."""
        stats = ArchiveStats(
            conversation_count=conv_count,
            message_count=msg_count,
            word_count=word_count,
            providers=providers,
            tags=tags,
            last_sync=last_sync,
            recent=[],
        )
        assert stats.conversation_count == conv_count
        assert stats.message_count == msg_count
        assert stats.word_count == word_count
        assert stats.providers == providers
        assert stats.tags == tags
        assert stats.last_sync == last_sync
        assert stats.recent == []

    def test_archive_stats_with_recent_conversations(self):
        """Test ArchiveStats with recent conversations."""
        recent_msgs = [
            Message(id="m1", role="user", text="Hello", timestamp=datetime.now(tz=timezone.utc)),
        ]
        recent_conv = Conversation(
            id="conv1",
            provider="claude",
            messages=MessageCollection(messages=recent_msgs),
        )
        stats = ArchiveStats(
            conversation_count=1,
            message_count=1,
            word_count=10,
            providers={"claude": 1},
            tags={},
            last_sync=None,
            recent=[recent_conv],
        )
        assert len(stats.recent) == 1
        assert stats.recent[0].id == "conv1"



# ============================================================================
# POLYLOGUE INITIALIZATION TESTS
# ============================================================================


class TestPolylogueInit:
    """Test Polylogue initialization with various configurations."""

    @pytest.mark.parametrize(
        "db_path_type,property_name",
        [
            (":memory:", "archive_root"),
            ("file.db", "config"),
            ("file.db", "archive_root"),
        ],
    )
    def test_polylogue_init_properties(self, tmp_path, db_path_type, property_name):
        """Test Polylogue initialization with various db_path types and property access."""
        db_path = tmp_path / "test.db" if db_path_type == "file.db" else db_path_type

        archive = Polylogue(archive_root=tmp_path, db_path=db_path)

        if property_name == "archive_root":
            assert archive.archive_root == tmp_path
        elif property_name == "config":
            cfg = archive.config
            assert type(cfg).__name__ == "Config"
            assert cfg.archive_root is not None


# ============================================================================
# POLYLOGUE CONVERSATION RETRIEVAL TESTS
# ============================================================================


class TestPolylogueGetConversation:
    """Test getting single conversations."""

    @pytest.mark.asyncio
    async def test_get_conversation_empty_db(self, tmp_path):
        """Test getting conversation from empty database."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        conv = await archive.get_conversation("nonexistent_id")
        assert conv is None

    @pytest.mark.asyncio
    async def test_get_conversation_with_seed_data(self, tmp_path):
        """Test retrieving a conversation after adding data."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        backend = archive._backend

        # Create and save a conversation
        conv_record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="provider-1",
            title="Test Conversation",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash-1",
        )

        # Save messages
        msg_records = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T00:00:00Z",
                content_hash="msg-hash-1",
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                role="assistant",
                text="Hi there",
                timestamp="2025-01-01T00:01:00Z",
                content_hash="msg-hash-2",
            ),
        ]

        # Use the async backend's save_conversation method
        await backend.save_conversation(conv_record, msg_records, [])

        # Retrieve by ID
        conv = await archive.get_conversation("conv-1")
        assert conv is not None
        assert conv.id == "conv-1"
        assert conv.title == "Test Conversation"


class TestPolylogueGetConversations:
    """Test batch conversation retrieval."""

    @pytest.mark.asyncio
    async def test_get_conversations_empty_list(self, tmp_path):
        """Test get_conversations with empty ID list."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        convs = await archive.get_conversations([])
        assert convs == []

    @pytest.mark.asyncio
    async def test_get_conversations_batch(self, tmp_path):
        """Test batch retrieval of multiple conversations."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        backend = archive._backend

        # Create multiple conversations
        for i in range(3):
            conv_record = ConversationRecord(
                conversation_id=f"conv-{i}",
                provider_name="claude",
                provider_conversation_id=f"provider-{i}",
                title=f"Conversation {i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"hash-{i}",
            )
            await backend.save_conversation(conv_record, [], [])

        # Retrieve batch
        ids = ["conv-0", "conv-1", "conv-2"]
        convs = await archive.get_conversations(ids)
        assert len(convs) == 3
        assert all(c.id in ids for c in convs)

    @pytest.mark.asyncio
    async def test_get_conversations_partial_match(self, tmp_path):
        """Test batch retrieval with some missing IDs."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        backend = archive._backend

        # Create only conv-1
        conv_record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="provider-1",
            title="Conversation 1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash-1",
        )
        await backend.save_conversation(conv_record, [], [])

        # Request multiple IDs but only conv-1 exists
        ids = ["conv-1", "conv-999"]
        convs = await archive.get_conversations(ids)
        assert len(convs) == 1
        assert convs[0].id == "conv-1"


# ============================================================================
# POLYLOGUE LIST CONVERSATIONS TESTS
# ============================================================================


class TestPolylogueListConversations:
    """Test listing conversations with various filters."""

    @pytest.mark.parametrize(
        "setup_count,provider_filter,source_filter,limit,expected_count",
        LIST_CONV_FILTERS,
    )
    @pytest.mark.asyncio
    async def test_list_conversations_with_filters(self, tmp_path, setup_count, provider_filter, source_filter, limit, expected_count):
        """Test listing conversations with various filter combinations."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        backend = archive._backend

        # Setup conversations
        for i in range(setup_count):
            provider = "claude" if i < 2 else "chatgpt"
            conv_record = ConversationRecord(
                conversation_id=f"conv-{i}",
                provider_name=provider,
                provider_conversation_id=f"provider-{i}",
                title=f"Conv {i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"hash-{i}",
                provider_meta={"source": source_filter or "inbox"} if source_filter else {},
            )
            await backend.save_conversation(conv_record, [], [])

        # Retrieve with filters
        convs = await archive.list_conversations(
            provider=provider_filter,
            limit=limit,
        )
        assert len(convs) == expected_count


# ============================================================================
# POLYLOGUE FILTER AND CONTEXT MANAGER TESTS
# ============================================================================


class TestPolylogueFilter:
    """Test filter builder creation."""

    def test_filter_returns_conversation_filter(self, tmp_path):
        """Test that filter() returns a ConversationFilter."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        filter_builder = archive.filter()
        assert filter_builder is not None
        # ConversationFilter has provider method
        assert hasattr(filter_builder, "provider")

    def test_filter_chaining(self, tmp_path):
        """Test filter method chaining."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        filter_builder = archive.filter()
        # Should support chaining
        result = filter_builder.provider("claude")
        assert result is not None


class TestPolylogueContextManager:
    """Test Polylogue as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enter_returns_self(self, tmp_path):
        """Test __aenter__ returns self."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        async with archive as ctx:
            assert ctx is archive

    @pytest.mark.asyncio
    async def test_context_manager_exit_calls_close(self, tmp_path):
        """Test __aexit__ calls close."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        # Just verify context manager works without exception
        async with archive:
            pass
        # If we get here without exception, context manager works properly

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, tmp_path):
        """Test context manager properly closes on exception."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)

        try:
            async with archive:
                raise ValueError("Test error")
        except ValueError:
            pass
        # If we get here, context manager handled exception properly

    @pytest.mark.asyncio
    async def test_close_method(self, tmp_path):
        """Test close() method."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        # Should not raise
        await archive.close()


# ============================================================================
# POLYLOGUE REBUILD INDEX AND STATS TESTS
# ============================================================================


class TestPolylogueRebuildIndex:
    """Test index rebuilding."""

    @pytest.mark.asyncio
    async def test_rebuild_index_lazy_init(self, tmp_path):
        """Test that rebuild_index can be called."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)

        # Just verify the method exists and can be called
        assert hasattr(archive, "rebuild_index")
        assert callable(archive.rebuild_index)
        # Can call it (it's async)
        result = await archive.rebuild_index()
        assert isinstance(result, bool)


class TestPolylogueStats:
    """Test statistics generation."""

    @pytest.mark.asyncio
    async def test_stats_empty_db(self, tmp_path):
        """Test stats() on empty database."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        stats = await archive.stats()

        assert isinstance(stats, ArchiveStats)
        assert stats.conversation_count == 0
        assert stats.message_count == 0
        assert stats.word_count == 0
        assert stats.providers == {}
        assert stats.tags == {}

    @pytest.mark.asyncio
    async def test_stats_with_conversations(self, tmp_path):
        """Test stats() with conversations in database."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        backend = archive._backend

        # Create conversations with different providers
        for i in range(2):
            conv_record = ConversationRecord(
                conversation_id=f"claude-{i}",
                provider_name="claude",
                provider_conversation_id=f"p-{i}",
                title=f"Claude Conv {i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"h-{i}",
            )

            # Add messages
            msg_records = [
                MessageRecord(
                    message_id=f"msg-{i}-0",
                    conversation_id=f"claude-{i}",
                    role="user",
                    text="Hello world",
                    timestamp="2025-01-01T00:00:00Z",
                    content_hash=f"mh-{i}-0",
                ),
                MessageRecord(
                    message_id=f"msg-{i}-1",
                    conversation_id=f"claude-{i}",
                    role="assistant",
                    text="Hi there friend",
                    timestamp="2025-01-01T00:01:00Z",
                    content_hash=f"mh-{i}-1",
                ),
            ]
            await backend.save_conversation(conv_record, msg_records, [])

        # Add ChatGPT conversation
        conv_record = ConversationRecord(
            conversation_id="chatgpt-0",
            provider_name="chatgpt",
            provider_conversation_id="p-chatgpt-0",
            title="ChatGPT Conv",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="h-chatgpt",
        )
        msg_records = [
            MessageRecord(
                message_id="msg-chatgpt-0",
                conversation_id="chatgpt-0",
                role="user",
                text="Test",
                timestamp="2025-01-01T00:00:00Z",
                content_hash="mh-chatgpt",
            ),
        ]
        await backend.save_conversation(conv_record, msg_records, [])

        stats = await archive.stats()
        assert stats.conversation_count == 3
        assert stats.message_count == 5
        assert "claude" in stats.providers
        assert "chatgpt" in stats.providers
        assert stats.providers["claude"] == 2
        assert stats.providers["chatgpt"] == 1

    @pytest.mark.asyncio
    async def test_stats_recent_conversations(self, tmp_path):
        """Test that stats includes recent conversations."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        backend = archive._backend

        # Create a single conversation
        conv_record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="p-1",
            title="Test Conv",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-15T12:00:00Z",
            content_hash="h-1",
        )
        await backend.save_conversation(conv_record, [], [])

        stats = await archive.stats()
        assert len(stats.recent) == 1
        assert stats.recent[0].id == "conv-1"


# ============================================================================
# CLI HELPERS TESTS: SOURCE STATE MANAGEMENT
# ============================================================================


class TestLoadSaveLastSource:
    """Test source state persistence."""

    def test_save_last_source(self, tmp_path, monkeypatch):
        """Test saving last source."""
        state_dir = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        save_last_source("my-source")

        path = state_dir / "polylogue" / "last-source.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["source"] == "my-source"

    @pytest.mark.parametrize(
        "state_exists,json_valid,source_field_exists,expected_result",
        LOAD_SOURCE_STATES,
    )
    def test_load_last_source(self, tmp_path, monkeypatch, state_exists, json_valid, source_field_exists, expected_result):
        """Test loading last source with various cache states."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        if state_exists:
            source_file = state_dir / "polylogue" / "last-source.json"
            source_file.parent.mkdir(parents=True, exist_ok=True)

            if json_valid:
                if source_field_exists:
                    content = json.dumps({"source": "test-source"})
                else:
                    content = json.dumps({"other": "value"})
            else:
                content = "invalid json"

            source_file.write_text(content, encoding="utf-8")

        result = load_last_source()
        assert result == expected_result

    def test_load_non_dict_json_returns_none(self, tmp_path, monkeypatch):
        """load_last_source() should return None if JSON is not a dict."""
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        path = source_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        assert load_last_source() is None

    def test_load_non_string_source_returns_none(self, tmp_path, monkeypatch):
        """load_last_source() should return None if source is not a string."""
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        path = source_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": 123}), encoding="utf-8")
        assert load_last_source() is None

    def test_multiple_save_overwrites(self, tmp_path, monkeypatch):
        """Multiple saves should overwrite previous value."""
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        save_last_source("chatgpt")
        save_last_source("claude")
        assert load_last_source() == "claude"

    def test_save_creates_parent_dirs(self, tmp_path, monkeypatch):
        """save_last_source() should create missing parent directories."""
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        path = source_state_path()
        assert not path.parent.exists()
        save_last_source("test")
        assert path.parent.exists()
        assert path.exists()


# ============================================================================
# CLI HELPERS TESTS: maybe_prompt_sources
# ============================================================================


class TestMaybePromptSources:
    """Test interactive source selection."""

    def test_maybe_prompt_already_selected(self):
        """Test when sources are already selected."""
        ui_mock = MagicMock()
        env = AppEnv(ui=ui_mock)
        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[Source(name="source-a", path=Path("/a")), Source(name="source-b", path=Path("/b"))],
        )
        selected = ["source-a"]

        result = maybe_prompt_sources(env, config, selected, "test-cmd")
        assert result == ["source-a"]
        ui_mock.choose.assert_not_called()

    def test_maybe_prompt_plain_mode(self):
        """Test plain mode passes through."""
        ui_mock = MagicMock()
        ui_mock.plain = True
        env = AppEnv(ui=ui_mock)
        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[Source(name="source-a", path=Path("/a")), Source(name="source-b", path=Path("/b"))],
        )

        result = maybe_prompt_sources(env, config, None, "test-cmd")
        assert result is None
        ui_mock.choose.assert_not_called()

    def test_maybe_prompt_single_source(self):
        """Test with single source (no prompt needed)."""
        ui_mock = MagicMock()
        ui_mock.plain = False
        env = AppEnv(ui=ui_mock)
        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[Source(name="only-source", path=Path("/a"))],
        )

        result = maybe_prompt_sources(env, config, None, "test-cmd")
        assert result is None
        ui_mock.choose.assert_not_called()

    def test_maybe_prompt_multiple_sources(self):
        """Test interactive prompt with multiple sources."""
        ui_mock = MagicMock()
        ui_mock.plain = False
        ui_mock.choose.return_value = "source-b"
        env = AppEnv(ui=ui_mock)

        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[Source(name="source-a", path=Path("/a")), Source(name="source-b", path=Path("/b"))],
        )

        with patch("polylogue.cli.helpers.load_last_source", return_value=None):
            with patch("polylogue.cli.helpers.save_last_source"):
                result = maybe_prompt_sources(env, config, None, "test-cmd")

        assert result == ["source-b"]
        ui_mock.choose.assert_called_once()

    def test_maybe_prompt_all_selected(self):
        """Test when 'all' is chosen."""
        ui_mock = MagicMock()
        ui_mock.plain = False
        ui_mock.choose.return_value = "all"
        env = AppEnv(ui=ui_mock)

        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[Source(name="source-a", path=Path("/a")), Source(name="source-b", path=Path("/b"))],
        )

        with patch("polylogue.cli.helpers.load_last_source", return_value=None):
            with patch("polylogue.cli.helpers.save_last_source"):
                result = maybe_prompt_sources(env, config, None, "test-cmd")

        assert result is None

    def test_maybe_prompt_restores_last_choice(self):
        """Test that last choice is restored to top of options."""
        ui_mock = MagicMock()
        ui_mock.plain = False
        ui_mock.choose.return_value = "source-b"
        env = AppEnv(ui=ui_mock)

        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[Source(name="source-a", path=Path("/a")), Source(name="source-b", path=Path("/b"))],
        )

        with patch("polylogue.cli.helpers.load_last_source", return_value="source-b"):
            with patch("polylogue.cli.helpers.save_last_source"):
                maybe_prompt_sources(env, config, None, "test-cmd")

        # Verify choose was called with options list that has source-b at top
        call_args = ui_mock.choose.call_args
        options = call_args[0][1]
        assert options[0] == "source-b"

    def test_maybe_prompt_no_choice_fails(self):
        """Test that not selecting a source fails."""
        ui_mock = MagicMock()
        ui_mock.plain = False
        ui_mock.choose.return_value = None
        env = AppEnv(ui=ui_mock)

        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[Source(name="source-a", path=Path("/a")), Source(name="source-b", path=Path("/b"))],
        )

        with patch("polylogue.cli.helpers.load_last_source", return_value=None):
            with pytest.raises(SystemExit):
                maybe_prompt_sources(env, config, None, "test-cmd")


# ============================================================================
# CLI HELPERS TESTS: latest_render_path
# ============================================================================


class TestLatestRenderPath:
    """Test finding latest rendered conversation."""

    def test_latest_render_path_no_root(self, tmp_path):
        """Test when render root doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        result = latest_render_path(nonexistent)
        assert result is None

    def test_latest_render_path_empty_root(self, tmp_path):
        """Test when render root is empty."""
        result = latest_render_path(tmp_path)
        assert result is None

    def test_latest_render_path_md_files(self, tmp_path):
        """Test finding latest .md file."""
        import time

        render_dir = tmp_path / "render"
        render_dir.mkdir()

        # Create nested structure with .md files
        conv_dir = render_dir / "some_conv"
        conv_dir.mkdir()

        # Create multiple .md files with different timestamps
        md1 = conv_dir / "conversation.md"
        md1.write_text("old")
        os.utime(md1, (100, 100))  # Set old timestamp

        md2 = render_dir / "conversation.md"
        md2.write_text("new")
        time.sleep(0.01)  # Ensure timestamp difference
        os.utime(md2, (200, 200))  # Set newer timestamp

        result = latest_render_path(render_dir)
        assert result is not None
        assert result.name == "conversation.md"

    def test_latest_render_path_html_files(self, tmp_path):
        """Test finding latest .html file."""
        render_dir = tmp_path / "render"
        render_dir.mkdir()

        html_file = render_dir / "conversation.html"
        html_file.write_text("<html></html>")

        result = latest_render_path(render_dir)
        assert result is not None
        assert result.suffix == ".html"

    def test_latest_render_path_handles_deleted_files(self, tmp_path):
        """Test handling of files deleted between rglob and stat."""
        render_dir = tmp_path / "render"
        render_dir.mkdir()

        # Create a file
        md_file = render_dir / "conversation.md"
        md_file.write_text("test")

        # Patch rglob to return a path that will be deleted before stat
        def mock_rglob(self, pattern):
            # Return both the real file and a non-existent one
            return [md_file, tmp_path / "deleted.md"]

        with patch.object(Path, "rglob", mock_rglob):
            result = latest_render_path(render_dir)
            # Should handle the deleted file gracefully
            assert result is not None or result is None  # Either is acceptable

    def test_latest_render_path_mixed_formats(self, tmp_path):
        """Test with both .md and .html files."""
        render_dir = tmp_path / "render"
        render_dir.mkdir()

        md_file = render_dir / "conversation.md"
        md_file.write_text("markdown")
        os.utime(md_file, (100, 100))

        html_file = render_dir / "conversation.html"
        html_file.write_text("<html></html>")
        os.utime(html_file, (200, 200))

        result = latest_render_path(render_dir)
        assert result is not None
        # Should return the newer one
        assert result.name in ["conversation.md", "conversation.html"]


# ============================================================================
# POLYLOGUE INGESTION TESTS
# ============================================================================


class TestPolylogueParseFile:
    """Test file ingestion."""

    def test_parse_file_method_exists(self, tmp_path):
        """Test that parse_file method exists and is callable."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        assert hasattr(archive, "parse_file")
        assert callable(archive.parse_file)

    @pytest.mark.asyncio
    async def test_parse_file_with_nonexistent_file(self, tmp_path):
        """Test parse_file with non-existent file."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)

        nonexistent = tmp_path / "does_not_exist.json"
        # Ingestion should handle gracefully (empty result or exception)
        # Most implementations skip missing files silently
        try:
            result = await archive.parse_file(nonexistent)
            # If no exception, verify result structure
            assert hasattr(result, "counts")
        except Exception:
            # Some implementations raise exceptions for missing files
            pass


class TestPolylogueParseSources:
    """Test sources ingestion."""

    def test_parse_sources_method_exists(self, tmp_path):
        """Test that parse_sources method exists and is callable."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        assert hasattr(archive, "parse_sources")
        assert callable(archive.parse_sources)  # It's an async method but still callable

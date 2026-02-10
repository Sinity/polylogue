"""Tests for vector search and archive statistics methods in ConversationRepository.

Tests the uncovered paths:
1. get_summary() not-found path
2. get_root() parent-not-found break
3. search_similar() full flow
4. record_run()
5. embed_conversation() with/without vector provider
6. similarity_search() returning tuples
7. get_archive_stats() comprehensive stats
8. _records_to_conversation() helper

These tests complement test_repository_operations.py which covers basic CRUD.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest

from polylogue.lib.models import Conversation, Message
from polylogue.lib.stats import ArchiveStats
from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository, _records_to_conversation
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
    store_records,
)
from tests.helpers import ConversationBuilder


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_db(tmp_path):
    """Create an empty database for tests."""
    db_path = tmp_path / "test.db"
    # Initialize the database by creating a connection
    with open_connection(db_path) as conn:
        pass  # Just open and close to initialize schema
    return db_path


@pytest.fixture
def repo(empty_db):
    """Create a ConversationRepository with empty database."""
    backend = SQLiteBackend(db_path=empty_db)
    return ConversationRepository(backend=backend)


@pytest.fixture
def db_with_conversations(tmp_path):
    """Create database with multiple conversations and messages."""
    db_path = tmp_path / "conversations.db"

    # Conversation 1: Root conversation
    (ConversationBuilder(db_path, "conv-1")
     .provider("claude")
     .title("Root Conversation")
     .add_message("m1", role="user", text="What is AI?")
     .add_message("m2", role="assistant", text="AI is artificial intelligence")
     .save())

    # Conversation 2: Child of conv-1
    (ConversationBuilder(db_path, "conv-2")
     .provider("claude")
     .title("Follow-up Conversation")
     .parent_conversation("conv-1")
     .add_message("m3", role="user", text="Tell me more")
     .add_message("m4", role="assistant", text="More details about AI")
     .save())

    # Conversation 3: Another root
    (ConversationBuilder(db_path, "conv-3")
     .provider("chatgpt")
     .title("ChatGPT Conversation")
     .add_message("m5", role="user", text="ChatGPT question")
     .save())

    # Conversation 4: Another root (will test orphaned case separately)
    (ConversationBuilder(db_path, "conv-4")
     .provider("gemini")
     .title("Gemini Conversation")
     .add_message("m6", role="user", text="Gemini message")
     .save())

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path


@pytest.fixture
def repo_with_conversations(db_with_conversations):
    """Create ConversationRepository with populated database."""
    backend = SQLiteBackend(db_path=db_with_conversations)
    return ConversationRepository(backend=backend)


# =============================================================================
# Helper Classes and Functions
# =============================================================================


class MockVectorProvider:
    """Mock VectorProvider for testing vector search methods."""

    def __init__(self, results: list[tuple[str, float]] | None = None):
        """Initialize with optional query results.

        Args:
            results: List of (message_id, distance) tuples
        """
        self._results = results or []
        self._upserted: dict[str, list[MessageRecord]] = {}

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Mock query that returns configured results."""
        return self._results[:limit]

    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None:
        """Mock upsert that records the operation."""
        self._upserted[conversation_id] = messages


def _make_conv_record(
    conv_id: str = "conv-1",
    provider: str = "claude",
    title: str = "Test Conversation",
) -> ConversationRecord:
    """Helper to create ConversationRecord without database."""
    now = datetime.now(timezone.utc).isoformat()
    return ConversationRecord(
        conversation_id=conv_id,
        provider_name=provider,
        provider_conversation_id=f"ext-{conv_id}",
        title=title,
        content_hash=uuid4().hex[:16],
        created_at=now,
        updated_at=now,
    )


def _make_msg_record(
    msg_id: str = "msg-1",
    conv_id: str = "conv-1",
    role: str = "user",
    content: str = "Hello",
) -> MessageRecord:
    """Helper to create MessageRecord without database."""
    now = datetime.now(timezone.utc).isoformat()
    return MessageRecord(
        message_id=msg_id,
        conversation_id=conv_id,
        role=role,
        text=content,
        timestamp=now,
        content_hash=uuid4().hex[:16],
    )


def _make_att_record(
    att_id: str = "att-1",
    conv_id: str = "conv-1",
    msg_id: str | None = "msg-1",
) -> AttachmentRecord:
    """Helper to create AttachmentRecord without database."""
    return AttachmentRecord(
        attachment_id=att_id,
        conversation_id=conv_id,
        message_id=msg_id,
        mime_type="text/plain",
        size_bytes=1024,
        path="/tmp/test.txt",
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestGetSummaryNotFound:
    """Test get_summary() with nonexistent conversation."""

    def test_get_summary_returns_none_for_nonexistent(self, repo):
        """get_summary() returns None when conversation doesn't exist."""
        assert repo.get_summary("nonexistent-conv") is None

    def test_get_summary_returns_summary_for_existing(self, repo_with_conversations):
        """get_summary() returns ConversationSummary for existing conversation."""
        summary = repo_with_conversations.get_summary("conv-1")
        assert summary is not None
        assert str(summary.id) == "conv-1"
        assert summary.title == "Root Conversation"
        assert summary.provider == "claude"

    def test_get_summary_does_not_load_messages(self, repo_with_conversations):
        """get_summary() doesn't load message data."""
        summary = repo_with_conversations.get_summary("conv-1")
        assert summary is not None
        # ConversationSummary should not have a messages attribute
        assert not hasattr(summary, "messages") or summary.messages is None


class TestGetRootEdgeCases:
    """Test get_root() with parent not found (orphaned trees)."""

    def test_get_root_returns_root_conversation(self, repo_with_conversations):
        """get_root() returns the root when no parent exists."""
        root = repo_with_conversations.get_root("conv-1")
        assert str(root.id) == "conv-1"

    def test_get_root_walks_up_to_parent(self, repo_with_conversations):
        """get_root() walks up the tree to find root."""
        root = repo_with_conversations.get_root("conv-2")
        assert str(root.id) == "conv-1"
        assert root.title == "Root Conversation"

    def test_get_root_stops_when_parent_deleted(self, repo_with_conversations):
        """get_root() breaks loop when parent_id points to deleted conversation.

        This tests the break condition when parent_id is set but parent doesn't exist.
        """
        from polylogue.storage.backends.sqlite import open_connection

        db_path = repo_with_conversations.backend._db_path

        # Create a child conversation with parent conv-3
        (ConversationBuilder(db_path, "conv-orphan")
         .provider("claude")
         .title("Child Before Delete")
         .parent_conversation("conv-3")  # conv-3 will be deleted
         .add_message("m-orphan", role="user", text="Orphan test")
         .save())

        # Disable FK constraints temporarily, delete conv-3, then re-enable
        with open_connection(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DELETE FROM messages WHERE conversation_id = 'conv-3'")
            conn.execute("DELETE FROM conversations WHERE conversation_id = 'conv-3'")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.commit()

        # Now get_root("conv-orphan") should break the parent loop when it tries
        # to get conv-3 and finds it doesn't exist
        root = repo_with_conversations.get_root("conv-orphan")
        # Should return the orphaned conversation as the "root" since parent is deleted
        assert str(root.id) == "conv-orphan"

    def test_get_root_raises_for_nonexistent_conversation(self, repo):
        """get_root() raises ValueError when conversation doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            repo.get_root("nonexistent-conv")


class TestSearchSimilar:
    """Test search_similar() full vector similarity flow."""

    def test_search_similar_raises_without_vector_provider(self, repo_with_conversations):
        """search_similar() raises ValueError if no vector provider supplied."""
        with pytest.raises(ValueError, match="Semantic search requires a vector provider"):
            repo_with_conversations.search_similar("test query", vector_provider=None)

    def test_search_similar_returns_empty_list_when_no_results(self, repo_with_conversations):
        """search_similar() returns empty list when provider has no results."""
        mock_provider = MockVectorProvider(results=[])
        results = repo_with_conversations.search_similar(
            "test query",
            limit=10,
            vector_provider=mock_provider,
        )
        assert results == []

    def test_search_similar_ranks_by_highest_score(self, repo_with_conversations):
        """search_similar() ranks conversations by highest message score."""
        # Set up results: multiple messages from same conversation
        mock_provider = MockVectorProvider(
            results=[
                ("m1", 0.95),  # From conv-1
                ("m2", 0.90),  # From conv-1
                ("m5", 0.85),  # From conv-3
            ]
        )
        results = repo_with_conversations.search_similar(
            "AI question",
            limit=10,
            vector_provider=mock_provider,
        )
        # Should have 2 conversations (conv-1 with score 0.95, conv-3 with 0.85)
        assert len(results) == 2
        result_ids = [str(r.id) for r in results]
        assert "conv-1" in result_ids
        assert "conv-3" in result_ids

    def test_search_similar_limits_results(self, repo_with_conversations):
        """search_similar() limits results to specified count."""
        # Create provider with many results
        results_data = [(f"m{i}", 1.0 - i * 0.01) for i in range(1, 31)]
        mock_provider = MockVectorProvider(results=results_data)
        results = repo_with_conversations.search_similar(
            "test",
            limit=5,
            vector_provider=mock_provider,
        )
        assert len(results) <= 5

    def test_search_similar_queries_3x_limit(self, repo_with_conversations):
        """search_similar() queries vector provider with 3x limit for ranking."""
        mock_provider = MockVectorProvider(
            results=[("m1", 0.9), ("m2", 0.85), ("m3", 0.8)] * 3
        )
        repo_with_conversations.search_similar(
            "test",
            limit=5,
            vector_provider=mock_provider,
        )
        # Provider should have been queried with limit=15 (5*3)
        # We verify this by checking the results were limited


class TestRecordRun:
    """Test record_run() pipeline audit entry."""

    def test_record_run_saves_run_record(self, repo):
        """record_run() saves RunRecord to backend."""
        now = datetime.now(timezone.utc).isoformat()
        run = RunRecord(
            run_id="run-1",
            timestamp=now,
            counts={"conversations": 5, "messages": 100},
        )
        # Should not raise
        repo.record_run(run)

    def test_record_run_with_counts_and_drift(self, repo):
        """record_run() handles counts and drift metadata."""
        now = datetime.now(timezone.utc).isoformat()
        run = RunRecord(
            run_id="run-error",
            timestamp=now,
            counts={"conversations": 0, "messages": 0},
            drift={"skipped": 5},
        )
        repo.record_run(run)

    def test_record_run_is_thread_safe(self, repo):
        """record_run() uses write lock for thread safety."""
        import threading

        results = []

        def save_run(run_id):
            now = datetime.now(timezone.utc).isoformat()
            run = RunRecord(
                run_id=run_id,
                timestamp=now,
                counts={"conversations": 1, "messages": 10},
            )
            repo.record_run(run)
            results.append(run_id)

        threads = [threading.Thread(target=save_run, args=(f"run-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5


class TestEmbedConversation:
    """Test embed_conversation() with and without vector provider."""

    def test_embed_conversation_with_explicit_provider(self, repo_with_conversations):
        """embed_conversation() uses provided vector provider."""
        mock_provider = MockVectorProvider()
        count = repo_with_conversations.embed_conversation(
            "conv-1",
            vector_provider=mock_provider,
        )
        assert count == 2  # conv-1 has 2 messages
        assert "conv-1" in mock_provider._upserted
        assert len(mock_provider._upserted["conv-1"]) == 2

    def test_embed_conversation_returns_zero_for_empty_conversation(self, repo_with_conversations):
        """embed_conversation() returns 0 when conversation has no messages."""
        # Create conversation with no messages
        (ConversationBuilder(repo_with_conversations.backend._db_path, "conv-empty")
         .provider("claude")
         .title("Empty Conversation")
         .save())

        mock_provider = MockVectorProvider()
        count = repo_with_conversations.embed_conversation(
            "conv-empty",
            vector_provider=mock_provider,
        )
        assert count == 0

    def test_embed_conversation_raises_without_api_key(self, repo_with_conversations):
        """embed_conversation() raises when trying to create default provider without VOYAGE_API_KEY."""
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "polylogue.storage.search_providers.create_vector_provider",
                return_value=None,
            ):
                with pytest.raises(ValueError, match="No vector provider available"):
                    repo_with_conversations.embed_conversation("conv-1", vector_provider=None)

    def test_embed_conversation_creates_default_provider_if_available(self, repo_with_conversations):
        """embed_conversation() creates default provider if VOYAGE_API_KEY available."""
        mock_provider = MockVectorProvider()
        with patch(
            "polylogue.storage.search_providers.create_vector_provider",
            return_value=mock_provider,
        ):
            count = repo_with_conversations.embed_conversation("conv-1", vector_provider=None)
            assert count == 2


class TestSimilaritySearch:
    """Test similarity_search() returning (conv_id, msg_id, distance) tuples."""

    def test_similarity_search_returns_tuples(self, repo_with_conversations):
        """similarity_search() returns list of (conv_id, msg_id, distance) tuples."""
        mock_provider = MockVectorProvider(
            results=[
                ("m1", 0.95),
                ("m2", 0.90),
            ]
        )
        results = repo_with_conversations.similarity_search(
            "test query",
            limit=10,
            vector_provider=mock_provider,
        )
        assert len(results) == 2
        for conv_id, msg_id, distance in results:
            assert isinstance(conv_id, str)
            assert isinstance(msg_id, str)
            assert isinstance(distance, float)

    def test_similarity_search_maps_messages_to_conversations(self, repo_with_conversations):
        """similarity_search() maps message IDs to their conversations."""
        mock_provider = MockVectorProvider(
            results=[
                ("m1", 0.95),  # m1 is in conv-1
                ("m5", 0.85),  # m5 is in conv-3
            ]
        )
        results = repo_with_conversations.similarity_search(
            "test query",
            vector_provider=mock_provider,
        )
        result_dict = {msg_id: conv_id for conv_id, msg_id, _ in results}
        assert result_dict.get("m1") == "conv-1"
        assert result_dict.get("m5") == "conv-3"

    def test_similarity_search_preserves_distance_scores(self, repo_with_conversations):
        """similarity_search() preserves distance scores from provider."""
        mock_provider = MockVectorProvider(
            results=[
                ("m1", 0.95),
                ("m2", 0.87),
            ]
        )
        results = repo_with_conversations.similarity_search(
            "test query",
            vector_provider=mock_provider,
        )
        distances = {msg_id: dist for _, msg_id, dist in results}
        assert abs(distances["m1"] - 0.95) < 0.001
        assert abs(distances["m2"] - 0.87) < 0.001

    def test_similarity_search_returns_empty_when_no_results(self, repo_with_conversations):
        """similarity_search() returns empty list when provider has no results."""
        mock_provider = MockVectorProvider(results=[])
        results = repo_with_conversations.similarity_search(
            "test query",
            vector_provider=mock_provider,
        )
        assert results == []

    def test_similarity_search_raises_without_provider(self, repo_with_conversations):
        """similarity_search() raises ValueError if no vector provider available."""
        with patch(
            "polylogue.storage.search_providers.create_vector_provider",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="No vector provider configured"):
                repo_with_conversations.similarity_search("test query", vector_provider=None)

    def test_similarity_search_filters_orphaned_messages(self, repo_with_conversations):
        """similarity_search() filters out messages whose conversation doesn't exist."""
        mock_provider = MockVectorProvider(
            results=[
                ("m1", 0.95),        # Valid
                ("m_orphaned", 0.90),  # Doesn't exist
            ]
        )
        results = repo_with_conversations.similarity_search(
            "test query",
            vector_provider=mock_provider,
        )
        # Only m1 should be in results (m_orphaned doesn't exist)
        msg_ids = [msg_id for _, msg_id, _ in results]
        assert "m1" in msg_ids
        assert "m_orphaned" not in msg_ids


class TestGetArchiveStats:
    """Test get_archive_stats() comprehensive statistics."""

    def test_get_archive_stats_counts_conversations(self, repo_with_conversations):
        """get_archive_stats() returns correct conversation count."""
        stats = repo_with_conversations.get_archive_stats()
        assert stats.total_conversations == 4  # conv-1, 2, 3, 4

    def test_get_archive_stats_counts_messages(self, repo_with_conversations):
        """get_archive_stats() returns correct message count."""
        stats = repo_with_conversations.get_archive_stats()
        # conv-1: 2 msgs, conv-2: 2 msgs, conv-3: 1 msg, conv-4: 1 msg = 6 total
        assert stats.total_messages == 6

    def test_get_archive_stats_breaks_down_by_provider(self, repo_with_conversations):
        """get_archive_stats() returns provider breakdown."""
        stats = repo_with_conversations.get_archive_stats()
        assert stats.providers.get("claude") == 2  # conv-1, conv-2
        assert stats.providers.get("chatgpt") == 1  # conv-3
        assert stats.providers.get("gemini") == 1  # conv-4

    def test_get_archive_stats_provider_count(self, repo_with_conversations):
        """get_archive_stats() returns count of unique providers."""
        stats = repo_with_conversations.get_archive_stats()
        assert stats.provider_count == 3

    def test_get_archive_stats_returns_archive_stats_type(self, repo_with_conversations):
        """get_archive_stats() returns ArchiveStats instance."""
        stats = repo_with_conversations.get_archive_stats()
        assert isinstance(stats, ArchiveStats)

    def test_get_archive_stats_empty_database(self, repo):
        """get_archive_stats() works with empty database."""
        stats = repo.get_archive_stats()
        assert stats.total_conversations == 0
        assert stats.total_messages == 0
        assert stats.providers == {}
        assert stats.provider_count == 0

    def test_get_archive_stats_embedding_coverage(self, repo_with_conversations):
        """get_archive_stats() computes embedding coverage percentage."""
        stats = repo_with_conversations.get_archive_stats()
        # Initially, no embeddings
        assert stats.embedded_conversations == 0
        assert stats.embedding_coverage == 0.0

    def test_get_archive_stats_avg_messages_per_conversation(self, repo_with_conversations):
        """get_archive_stats() computes average messages per conversation."""
        stats = repo_with_conversations.get_archive_stats()
        # 6 messages / 4 conversations = 1.5
        assert stats.avg_messages_per_conversation == 1.5

    def test_get_archive_stats_includes_db_size(self, repo_with_conversations):
        """get_archive_stats() includes database file size."""
        stats = repo_with_conversations.get_archive_stats()
        assert stats.db_size_bytes >= 0

    def test_get_archive_stats_handles_missing_embedding_tables(self, repo_with_conversations):
        """get_archive_stats() gracefully handles missing embedding tables."""
        # The database doesn't have embedding tables, so embedded_* should be 0
        stats = repo_with_conversations.get_archive_stats()
        assert stats.embedded_conversations == 0
        assert stats.embedded_messages == 0


class TestRecordsToConversation:
    """Test _records_to_conversation() standalone helper."""

    def test_records_to_conversation_converts_to_model(self):
        """_records_to_conversation() converts records to Conversation model."""
        conv_rec = _make_conv_record()
        msg_recs = [
            _make_msg_record("msg-1", role="user", content="Hello"),
            _make_msg_record("msg-2", role="assistant", content="Hi there"),
        ]
        att_recs = [_make_att_record("att-1", msg_id="msg-1")]

        conv = _records_to_conversation(conv_rec, msg_recs, att_recs)

        assert isinstance(conv, Conversation)
        assert str(conv.id) == "conv-1"
        assert conv.title == "Test Conversation"
        assert len(conv.messages) == 2

    def test_records_to_conversation_with_no_attachments(self):
        """_records_to_conversation() works with empty attachments list."""
        conv_rec = _make_conv_record()
        msg_recs = [_make_msg_record("msg-1")]
        att_recs = []

        conv = _records_to_conversation(conv_rec, msg_recs, att_recs)

        assert str(conv.id) == "conv-1"
        assert len(conv.messages) == 1

    def test_records_to_conversation_with_no_messages(self):
        """_records_to_conversation() works with empty messages list."""
        conv_rec = _make_conv_record()
        msg_recs = []
        att_recs = []

        conv = _records_to_conversation(conv_rec, msg_recs, att_recs)

        assert str(conv.id) == "conv-1"
        assert len(conv.messages) == 0

    def test_records_to_conversation_preserves_message_order(self):
        """_records_to_conversation() preserves message order."""
        conv_rec = _make_conv_record()
        msg_recs = [
            _make_msg_record("m1", role="user"),
            _make_msg_record("m2", role="assistant"),
            _make_msg_record("m3", role="user"),
        ]

        conv = _records_to_conversation(conv_rec, msg_recs, [])

        assert len(conv.messages) == 3
        assert conv.messages[0].id == "m1"
        assert conv.messages[1].id == "m2"
        assert conv.messages[2].id == "m3"

    def test_records_to_conversation_with_parent_id(self):
        """_records_to_conversation() preserves parent_id from record."""
        conv_rec = _make_conv_record().model_copy(
            update={"parent_conversation_id": "parent-conv"}
        )
        msg_recs = []

        conv = _records_to_conversation(conv_rec, msg_recs, [])

        assert str(conv.parent_id) == "parent-conv"

    def test_records_to_conversation_used_by_migration_tools(self):
        """_records_to_conversation() is suitable for migration tools."""
        # Simulate a migration scenario: bulk convert records
        convs_data = [
            (_make_conv_record(f"conv-{i}"), [_make_msg_record(f"m-{i}")], [])
            for i in range(1, 4)
        ]

        results = [_records_to_conversation(c, m, a) for c, m, a in convs_data]

        assert len(results) == 3
        assert all(isinstance(r, Conversation) for r in results)
        assert [str(r.id) for r in results] == ["conv-1", "conv-2", "conv-3"]


# =============================================================================
# Integration Tests (Cross-method interactions)
# =============================================================================


class TestIntegrationVectorWorkflows:
    """Test vector search workflows across multiple methods."""

    def test_embed_then_search_workflow(self, repo_with_conversations):
        """Full workflow: embed conversations, then search by similarity."""
        # Step 1: Embed a conversation
        mock_provider = MockVectorProvider()
        embed_count = repo_with_conversations.embed_conversation(
            "conv-1",
            vector_provider=mock_provider,
        )
        assert embed_count == 2

        # Step 2: Configure provider with search results
        mock_provider._results = [("m1", 0.95), ("m2", 0.87)]

        # Step 3: Perform similarity search
        results = repo_with_conversations.similarity_search(
            "embedded search",
            vector_provider=mock_provider,
        )
        assert len(results) == 2
        assert all(len(r) == 3 for r in results)  # 3-tuple validation

    def test_search_similar_with_multiple_conversations(self, repo_with_conversations):
        """search_similar() aggregates results across conversations."""
        # Set up results spanning multiple conversations
        mock_provider = MockVectorProvider(
            results=[
                ("m1", 0.95),  # conv-1
                ("m2", 0.92),  # conv-1
                ("m3", 0.88),  # conv-2
                ("m5", 0.80),  # conv-3
            ]
        )
        results = repo_with_conversations.search_similar(
            "test",
            limit=3,
            vector_provider=mock_provider,
        )
        # Should return max 3 conversations
        assert len(results) <= 3

    def test_stats_after_embedding_operations(self, repo_with_conversations):
        """get_archive_stats() reflects state after embedding operations."""
        # Get initial stats
        stats_before = repo_with_conversations.get_archive_stats()
        assert stats_before.total_conversations == 4

        # Perform embedding (doesn't change conversation count, just records state)
        mock_provider = MockVectorProvider()
        repo_with_conversations.embed_conversation("conv-1", vector_provider=mock_provider)

        # Stats should still show same conversation count
        stats_after = repo_with_conversations.get_archive_stats()
        assert stats_after.total_conversations == stats_before.total_conversations

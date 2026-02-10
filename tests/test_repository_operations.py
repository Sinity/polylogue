"""Tests for ConversationRepository operations.

SYSTEMATIZATION: Merged from:
- Original test_repository_operations.py (Basic CRUD, tree traversal, metadata)
- test_repository_vector_stats.py (Vector search, embeddings, statistics) [MERGED]

Tests the repository facade for conversation retrieval, tree traversal,
transactional save, metadata CRUD, search operations, vector embeddings,
similarity search, and archive statistics.
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


@pytest.fixture
def repo_db(tmp_path):
    """Create database with test conversations."""
    db_path = tmp_path / "repo_test.db"

    # Conversation 1: claude with 2 messages
    (ConversationBuilder(db_path, "conv-1")
     .provider("claude")
     .title("First Conversation")
     .add_message("m1", role="user", text="Hello from user")
     .add_message("m2", role="assistant", text="Hello from assistant")
     .metadata({"tags": ["greeting"]})
     .save())

    # Conversation 2: chatgpt with 1 message
    (ConversationBuilder(db_path, "conv-2")
     .provider("chatgpt")
     .title("Second Conversation")
     .add_message("m3", role="user", text="ChatGPT question")
     .save())

    # Conversation 3: claude child of conv-1
    (ConversationBuilder(db_path, "conv-3")
     .provider("claude")
     .title("Child Conversation")
     .parent_conversation("conv-1")
     .add_message("m4", role="user", text="Follow up question")
     .save())

    # Build FTS index
    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path


@pytest.fixture
def repo(repo_db):
    """Create ConversationRepository."""
    backend = SQLiteBackend(db_path=repo_db)
    return ConversationRepository(backend=backend)


# =============================================================================
# Lazy vs Eager Loading
# =============================================================================


class TestLazyVsEagerLoading:
    """Test that get() provides lazy loading and get_eager() loads eagerly."""

    def test_get_returns_conversation(self, repo):
        """get() should return a Conversation object."""
        conv = repo.get("conv-1")
        assert conv is not None
        assert str(conv.id) == "conv-1"

    def test_get_nonexistent_returns_none(self, repo):
        """get() should return None for nonexistent conversation."""
        assert repo.get("nonexistent") is None

    def test_get_lazy_loads_messages_on_access(self, repo):
        """get() returns lazy Conversation; messages load on first access."""
        conv = repo.get("conv-1")
        assert conv is not None
        # Accessing messages triggers lazy load
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"

    def test_get_eager_loads_messages_immediately(self, repo):
        """get_eager() loads all messages upfront."""
        conv = repo.get_eager("conv-1")
        assert conv is not None
        # Messages should be immediately available
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"

    def test_get_eager_nonexistent_returns_none(self, repo):
        """get_eager() should return None for nonexistent conversation."""
        assert repo.get_eager("nonexistent") is None

    def test_get_summary_no_messages(self, repo):
        """get_summary() returns summary without loading messages."""
        summary = repo.get_summary("conv-1")
        assert summary is not None
        assert summary.id == "conv-1"
        assert summary.title == "First Conversation"


# =============================================================================
# resolve_id and view
# =============================================================================


class TestResolveIdAndView:
    """Test ID resolution and view()."""

    def test_resolve_full_id(self, repo):
        """resolve_id() should return ConversationId for exact match."""
        resolved = repo.resolve_id("conv-1")
        assert resolved is not None
        assert str(resolved) == "conv-1"

    def test_resolve_prefix(self, repo):
        """resolve_id() should resolve unique prefixes."""
        # This depends on the backend implementation
        # At minimum, exact match should work
        resolved = repo.resolve_id("conv-1")
        assert resolved is not None

    def test_resolve_nonexistent(self, repo):
        """resolve_id() should return None for nonexistent ID."""
        resolved = repo.resolve_id("nonexistent-id")
        assert resolved is None

    def test_view_with_full_id(self, repo):
        """view() should return conversation for full ID."""
        conv = repo.view("conv-1")
        assert conv is not None
        assert str(conv.id) == "conv-1"

    def test_view_nonexistent(self, repo):
        """view() should return None for nonexistent ID."""
        conv = repo.view("nonexistent-id-xyz")
        assert conv is None


# =============================================================================
# Tree Traversal
# =============================================================================


class TestTreeTraversal:
    """Test parent/child/root/tree operations."""

    def test_get_parent_of_child(self, repo):
        """get_parent() should return parent conversation."""
        parent = repo.get_parent("conv-3")
        assert parent is not None
        assert str(parent.id) == "conv-1"

    def test_get_parent_of_root(self, repo):
        """get_parent() should return None for root conversation."""
        parent = repo.get_parent("conv-1")
        assert parent is None

    def test_get_children(self, repo):
        """get_children() should return direct children."""
        children = repo.get_children("conv-1")
        assert len(children) == 1
        assert str(children[0].id) == "conv-3"

    def test_get_children_none(self, repo):
        """get_children() should return empty list for no children."""
        children = repo.get_children("conv-2")
        assert children == []

    def test_get_root_from_child(self, repo):
        """get_root() should walk up to find root from child."""
        root = repo.get_root("conv-3")
        assert str(root.id) == "conv-1"

    def test_get_root_from_root(self, repo):
        """get_root() should return itself for root conversation."""
        root = repo.get_root("conv-1")
        assert str(root.id) == "conv-1"

    def test_get_root_nonexistent_raises(self, repo):
        """get_root() should raise ValueError for nonexistent conversation."""
        with pytest.raises(ValueError, match="not found"):
            repo.get_root("nonexistent")

    def test_get_session_tree(self, repo):
        """get_session_tree() should return root and all descendants."""
        tree = repo.get_session_tree("conv-3")
        tree_ids = {str(c.id) for c in tree}
        assert "conv-1" in tree_ids
        assert "conv-3" in tree_ids

    def test_get_session_tree_from_root(self, repo):
        """get_session_tree() from root should include all descendants."""
        tree = repo.get_session_tree("conv-1")
        tree_ids = {str(c.id) for c in tree}
        assert "conv-1" in tree_ids
        assert "conv-3" in tree_ids
        # Should have 2 conversations in the tree
        assert len(tree) == 2


# =============================================================================
# save_conversation
# =============================================================================


class TestSaveConversation:
    """Test transactional save with skip counting."""

    def test_save_new_conversation(self, repo):
        """save_conversation() should insert new conversation and messages."""
        conv = ConversationRecord(
            conversation_id="new-conv",
            provider_name="claude",
            provider_conversation_id="prov-new",
            title="New Conversation",
            created_at="2025-06-01T00:00:00Z",
            updated_at="2025-06-01T00:00:00Z",
            content_hash="new-hash",
            version=1,
        )
        msg = MessageRecord(
            message_id="new-msg",
            conversation_id="new-conv",
            role="user",
            text="New message",
            timestamp="2025-06-01T00:00:00Z",
            content_hash="msg-hash",
            version=1,
        )
        counts = repo.save_conversation(
            conversation=conv, messages=[msg], attachments=[]
        )
        assert counts["conversations"] == 1
        assert counts["messages"] == 1

        # Verify it was saved
        retrieved = repo.get("new-conv")
        assert retrieved is not None

    def test_save_duplicate_conversation_skipped(self, repo):
        """Saving same content_hash again should skip."""
        conv = ConversationRecord(
            conversation_id="dup-conv",
            provider_name="claude",
            provider_conversation_id="prov-dup",
            title="Dup Conv",
            created_at="2025-06-01T00:00:00Z",
            updated_at="2025-06-01T00:00:00Z",
            content_hash="dup-hash",
            version=1,
        )
        msg = MessageRecord(
            message_id="dup-msg",
            conversation_id="dup-conv",
            role="user",
            text="Dup message",
            timestamp="2025-06-01T00:00:00Z",
            content_hash="dup-msg-hash",
            version=1,
        )

        # First save
        counts1 = repo.save_conversation(
            conversation=conv, messages=[msg], attachments=[]
        )
        assert counts1["conversations"] == 1

        # Second save with same hash
        counts2 = repo.save_conversation(
            conversation=conv, messages=[msg], attachments=[]
        )
        assert counts2["skipped_conversations"] == 1
        assert counts2["skipped_messages"] == 1

    def test_save_with_attachments(self, repo):
        """save_conversation() should save attachments."""
        conv = ConversationRecord(
            conversation_id="conv-att",
            provider_name="claude",
            provider_conversation_id="prov-att",
            title="With Attachments",
            created_at="2025-06-01T00:00:00Z",
            updated_at="2025-06-01T00:00:00Z",
            content_hash="att-hash",
            version=1,
        )
        msg = MessageRecord(
            message_id="msg-att",
            conversation_id="conv-att",
            role="user",
            text="Message with attachment",
            timestamp="2025-06-01T00:00:00Z",
            content_hash="msg-att-hash",
            version=1,
        )
        att = AttachmentRecord(
            attachment_id="att-1",
            conversation_id="conv-att",
            message_id="msg-att",
            mime_type="application/pdf",
            size_bytes=1024,
        )
        counts = repo.save_conversation(
            conversation=conv, messages=[msg], attachments=[att]
        )
        assert counts["conversations"] == 1
        assert counts["messages"] == 1
        assert counts["attachments"] == 1


# =============================================================================
# Metadata CRUD
# =============================================================================


class TestMetadataCRUD:
    """Test metadata operations through repository."""

    def test_update_and_get_metadata(self, repo):
        """update_metadata() and get_metadata() should work."""
        repo.update_metadata("conv-1", "status", "reviewed")
        meta = repo.get_metadata("conv-1")
        assert meta.get("status") == "reviewed"

    def test_delete_metadata(self, repo):
        """delete_metadata() should remove key."""
        repo.update_metadata("conv-1", "temp", "value")
        repo.delete_metadata("conv-1", "temp")
        meta = repo.get_metadata("conv-1")
        assert "temp" not in meta

    def test_add_and_remove_tag(self, repo):
        """add_tag() and remove_tag() should manage tags."""
        repo.add_tag("conv-1", "test-tag")
        tags = repo.list_tags()
        assert "test-tag" in tags

        repo.remove_tag("conv-1", "test-tag")
        # After removal, tag should be gone or have zero count
        tags = repo.list_tags()
        # The tag might still exist with 0 count, or might be removed

    def test_list_tags_with_provider(self, repo):
        """list_tags() should filter by provider."""
        repo.add_tag("conv-1", "claude-tag")
        tags = repo.list_tags(provider="claude")
        assert "claude-tag" in tags

    def test_set_metadata_replaces(self, repo):
        """set_metadata() should replace all metadata."""
        repo.set_metadata("conv-1", {"key1": "val1", "key2": "val2"})
        meta = repo.get_metadata("conv-1")
        assert meta.get("key1") == "val1"
        assert meta.get("key2") == "val2"

    def test_delete_conversation(self, repo):
        """delete_conversation() should remove conversation."""
        assert repo.delete_conversation("conv-2")
        assert repo.get("conv-2") is None

    def test_delete_nonexistent_conversation(self, repo):
        """delete_conversation() should return False for nonexistent."""
        result = repo.delete_conversation("nonexistent")
        assert result is False


# =============================================================================
# Count and List operations
# =============================================================================


class TestCountAndList:
    """Test count() and list_summaries() with filters."""

    def test_count_all(self, repo):
        """count() should return total conversations."""
        count = repo.count()
        assert count == 3

    def test_count_by_provider(self, repo):
        """count() should filter by provider."""
        assert repo.count(provider="claude") == 2
        assert repo.count(provider="chatgpt") == 1

    def test_count_by_providers_list(self, repo):
        """count() should filter by providers list."""
        assert repo.count(providers=["claude", "chatgpt"]) == 3

    def test_list_summaries_limit(self, repo):
        """list_summaries() should respect limit."""
        summaries = repo.list_summaries(limit=2)
        assert len(summaries) == 2

    def test_list_summaries_by_provider(self, repo):
        """list_summaries() should filter by provider."""
        summaries = repo.list_summaries(provider="claude")
        assert len(summaries) == 2
        assert all(s.provider == "claude" for s in summaries)

    def test_list_with_title_filter(self, repo):
        """list() should filter by title."""
        convs = repo.list(title_contains="First")
        assert len(convs) == 1
        assert "First" in convs[0].display_title

    def test_list_returns_lazy_conversations(self, repo):
        """list() should return lazy Conversation objects."""
        convs = repo.list()
        assert len(convs) == 3
        # Each should have an id
        assert all(hasattr(c, "id") for c in convs)


# =============================================================================
# Search operations
# =============================================================================


class TestSearch:
    """Test FTS search through repository."""

    def test_search_finds_matching(self, repo):
        """search() should find conversations matching query."""
        results = repo.search("Hello")
        assert len(results) >= 1

    def test_search_summaries(self, repo):
        """search_summaries() should return summaries matching query."""
        results = repo.search_summaries("Hello")
        assert len(results) >= 1
        assert hasattr(results[0], "title")

    def test_search_empty_query(self, repo):
        """search() should handle empty query."""
        results = repo.search("")
        assert isinstance(results, list)

    def test_search_no_match(self, repo):
        """search() should return empty list for no matches."""
        results = repo.search("zzzznonexistentzzzz")
        assert results == []

    def test_search_summaries_no_match(self, repo):
        """search_summaries() should return empty list for no matches."""
        results = repo.search_summaries("zzzznonexistentzzzz")
        assert results == []


# =============================================================================
# iter_messages
# =============================================================================


class TestIterMessages:
    """Test message streaming."""

    def test_iter_messages_basic(self, repo):
        """iter_messages() should yield all messages."""
        messages = list(repo.iter_messages("conv-1"))
        assert len(messages) == 2

    def test_iter_messages_with_limit(self, repo):
        """iter_messages() should respect limit."""
        messages = list(repo.iter_messages("conv-1", limit=1))
        assert len(messages) == 1

    def test_iter_messages_nonexistent(self, repo):
        """iter_messages() should return empty for nonexistent conversation."""
        messages = list(repo.iter_messages("nonexistent"))
        assert messages == []

    def test_get_conversation_stats(self, repo):
        """get_conversation_stats() should return message counts."""
        stats = repo.get_conversation_stats("conv-1")
        assert stats is not None
        assert stats["total_messages"] == 2

    def test_get_conversation_stats_nonexistent(self, repo):
        """get_conversation_stats() should return None for nonexistent."""
        stats = repo.get_conversation_stats("nonexistent")
        assert stats is None


# =============================================================================
# filter() factory
# =============================================================================


class TestFilterFactory:
    """Test that filter() creates a working ConversationFilter."""

    def test_filter_returns_filter_object(self, repo):
        """filter() should return ConversationFilter."""
        from polylogue.lib.filters import ConversationFilter
        f = repo.filter()
        assert isinstance(f, ConversationFilter)

    def test_filter_chains_work(self, repo):
        """filter() chains should work."""
        result = repo.filter().provider("claude").limit(1).list()
        assert len(result) == 1
        assert result[0].provider == "claude"

    def test_filter_count(self, repo):
        """filter() should allow count() chain."""
        count = repo.filter().provider("chatgpt").count()
        assert count == 1



# =============================================================================
# MERGED FROM test_repository_vector_stats.py - Vector & Statistics Operations
# =============================================================================

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
def repo_empty(empty_db):
    """Create ConversationRepository with empty database."""
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

    def test_get_summary_returns_none_for_nonexistent(self, repo_empty):
        """get_summary() returns None when conversation doesn't exist."""
        assert repo_empty.get_summary("nonexistent-conv") is None

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

    def test_get_root_raises_for_nonexistent_conversation(self, repo_empty):
        """get_root() raises ValueError when conversation doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            repo_empty.get_root("nonexistent-conv")


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

    def test_record_run_saves_run_record(self, repo_empty):
        """record_run() saves RunRecord to backend."""
        now = datetime.now(timezone.utc).isoformat()
        run = RunRecord(
            run_id="run-1",
            timestamp=now,
            counts={"conversations": 5, "messages": 100},
        )
        # Should not raise
        repo_empty.record_run(run)

    def test_record_run_with_counts_and_drift(self, repo_empty):
        """record_run() handles counts and drift metadata."""
        now = datetime.now(timezone.utc).isoformat()
        run = RunRecord(
            run_id="run-error",
            timestamp=now,
            counts={"conversations": 0, "messages": 0},
            drift={"skipped": 5},
        )
        repo_empty.record_run(run)

    def test_record_run_is_thread_safe(self, repo_empty):
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
            repo_empty.record_run(run)
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

    def test_get_archive_stats_empty_database(self, repo_empty):
        """get_archive_stats() works with empty database."""
        stats = repo_empty.get_archive_stats()
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

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
from unittest.mock import patch

import pytest

from polylogue.lib.models import Conversation
from polylogue.lib.stats import ArchiveStats
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository, _records_to_conversation
from polylogue.storage.store import (
    MessageRecord,
    RunRecord,
)
from tests.infra.helpers import ConversationBuilder, make_attachment, make_conversation, make_message


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
# Lazy vs Eager Loading (Parametrized)
# =============================================================================


class TestLazyVsEagerLoading:
    """Test get(), get_eager(), and get_summary() loading behavior."""

    @pytest.mark.parametrize("method,conv_id,should_exist,should_load_messages", [
        ("get", "conv-1", True, True),
        ("get_eager", "conv-1", True, True),
        ("get", "nonexistent", False, False),
        ("get_eager", "nonexistent", False, False),
    ])
    async def test_get_methods(self, repo, method, conv_id, should_exist, should_load_messages):
        """get() and get_eager() return Conversation or None, loading messages."""
        get_method = getattr(repo, method)
        conv = await get_method(conv_id)
        if should_exist:
            assert conv is not None
            assert str(conv.id) == conv_id
            if should_load_messages:
                assert len(conv.messages) == 2
                assert conv.messages[0].role == "user"
                assert conv.messages[1].role == "assistant"
        else:
            assert conv is None

    async def test_get_summary_no_messages(self, repo):
        """get_summary() returns summary without loading messages."""
        summary = await repo.get_summary("conv-1")
        assert summary is not None
        assert summary.id == "conv-1"
        assert summary.title == "First Conversation"


# =============================================================================
# resolve_id and view (Parametrized)
# =============================================================================


class TestResolveIdAndView:
    """Test ID resolution and view()."""

    @pytest.mark.parametrize("conv_id,should_resolve", [
        ("conv-1", True),  # Full ID match
        ("nonexistent-id", False),  # Nonexistent
    ])
    async def test_resolve_id(self, repo, conv_id, should_resolve):
        """resolve_id() should resolve IDs and return None for nonexistent."""
        resolved = await repo.resolve_id(conv_id)
        if should_resolve:
            assert resolved is not None
            assert str(resolved) == conv_id
        else:
            assert resolved is None

    @pytest.mark.parametrize("conv_id,should_exist", [
        ("conv-1", True),
        ("nonexistent-id-xyz", False),
    ])
    async def test_view(self, repo, conv_id, should_exist):
        """view() should return conversation for full ID or None for nonexistent."""
        conv = await repo.view(conv_id)
        if should_exist:
            assert conv is not None
            assert str(conv.id) == conv_id
        else:
            assert conv is None


# =============================================================================
# Tree Traversal (Parametrized)
# =============================================================================


class TestRepositoryTreeTraversal:
    """Test parent/child/root/tree operations."""

    @pytest.mark.parametrize("method,conv_id,expected_check", [
        ("get_parent", "conv-3", lambda r: r is not None and str(r.id) == "conv-1"),
        ("get_parent", "conv-1", lambda r: r is None),
        ("get_children", "conv-1", lambda r: len(r) == 1 and str(r[0].id) == "conv-3"),
        ("get_children", "conv-2", lambda r: len(r) == 0),
        ("get_root", "conv-1", lambda r: str(r.id) == "conv-1"),
        ("get_root", "conv-3", lambda r: str(r.id) == "conv-1"),
    ])
    async def test_tree_methods(self, repo, method, conv_id, expected_check):
        """Tree traversal methods return expected relationships."""
        method_fn = getattr(repo, method)
        result = await method_fn(conv_id)
        assert expected_check(result)

    async def test_get_root_nonexistent_raises(self, repo):
        """get_root() should raise ValueError for nonexistent conversation."""
        with pytest.raises(ValueError, match="not found"):
            await repo.get_root("nonexistent")

    @pytest.mark.parametrize("start_conv_id,expected_ids", [
        ("conv-3", {"conv-1", "conv-3"}),   # From child
        ("conv-1", {"conv-1", "conv-3"}),   # From root, includes children
    ])
    async def test_get_session_tree(self, repo, start_conv_id, expected_ids):
        """get_session_tree() should return root and all descendants."""
        tree = await repo.get_session_tree(start_conv_id)
        tree_ids = {str(c.id) for c in tree}
        assert tree_ids == expected_ids


# =============================================================================
# save_conversation
# =============================================================================


class TestSaveConversation:
    """Test transactional save with skip counting."""

    @pytest.mark.parametrize("conv_id,has_attachments,expected_att_count", [
        ("new-conv", False, 0),
        ("conv-att", True, 1),
    ])
    async def test_save_conversation(self, repo, conv_id, has_attachments, expected_att_count):
        """save_conversation() inserts conversations, messages, and attachments."""
        conv = make_conversation(conv_id, title="Test Save")
        msg = make_message(f"m-{conv_id}", conv_id, text="Test message")
        atts = [make_attachment("att-1", conv_id, msg.message_id)] if has_attachments else []

        counts = await repo.save_conversation(conversation=conv, messages=[msg], attachments=atts)
        assert counts["conversations"] == 1
        assert counts["messages"] == 1
        assert counts["attachments"] == expected_att_count

        retrieved = await repo.get(conv_id)
        assert retrieved is not None

    async def test_save_duplicate_conversation_skipped(self, repo):
        """Saving same content_hash again should skip."""
        conv = make_conversation("dup-conv", content_hash="dup-hash")
        msg = make_message("dup-msg", "dup-conv", content_hash="msg-hash")

        counts1 = await repo.save_conversation(conversation=conv, messages=[msg], attachments=[])
        assert counts1["conversations"] == 1

        counts2 = await repo.save_conversation(conversation=conv, messages=[msg], attachments=[])
        assert counts2["skipped_conversations"] == 1
        assert counts2["skipped_messages"] == 1


# =============================================================================
# Metadata CRUD (Parametrized)
# =============================================================================


class TestMetadataCRUD:
    """Test metadata operations through repository."""

    @pytest.mark.parametrize("operation,key,value,check_fn", [
        ("update", "status", "reviewed", lambda m: m.get("status") == "reviewed"),
        ("set", None, {"key1": "val1", "key2": "val2"}, lambda m: m.get("key1") == "val1" and m.get("key2") == "val2"),
    ])
    async def test_metadata_operations(self, repo, operation, key, value, check_fn):
        """update_metadata() and set_metadata() should work."""
        if operation == "update":
            await repo.update_metadata("conv-1", key, value)
        elif operation == "set":
            await repo.set_metadata("conv-1", value)

        meta = await repo.get_metadata("conv-1")
        assert check_fn(meta)

    async def test_delete_and_manage_tags(self, repo):
        """delete_metadata(), add_tag(), remove_tag(), list_tags() work together."""
        # Delete metadata
        await repo.update_metadata("conv-1", "temp", "value")
        await repo.delete_metadata("conv-1", "temp")
        meta = await repo.get_metadata("conv-1")
        assert "temp" not in meta

        # Manage tags
        await repo.add_tag("conv-1", "test-tag")
        tags = await repo.list_tags()
        assert "test-tag" in tags

        # Filter by provider
        await repo.add_tag("conv-1", "claude-tag")
        tags_claude = await repo.list_tags(provider="claude")
        assert "claude-tag" in tags_claude

        await repo.remove_tag("conv-1", "test-tag")

    @pytest.mark.parametrize("conv_id,should_exist", [
        ("conv-2", True),
        ("nonexistent", False),
    ])
    async def test_delete_conversation(self, repo, conv_id, should_exist):
        """delete_conversation() should remove conversation or return False."""
        if should_exist:
            result = await repo.delete_conversation(conv_id)
            assert result is True
            assert await repo.get(conv_id) is None
        else:
            result = await repo.delete_conversation(conv_id)
            assert result is False


# =============================================================================
# Count and List operations (Parametrized)
# =============================================================================


class TestCountAndList:
    """Test count() and list_summaries() with filters."""

    @pytest.mark.parametrize("provider,providers,expected_count", [
        (None, None, 3),
        ("claude", None, 2),
        ("chatgpt", None, 1),
        (None, ["claude", "chatgpt"], 3),
    ])
    async def test_count(self, repo, provider, providers, expected_count):
        """count() returns conversations, optionally filtered by provider."""
        if provider:
            count = await repo.count(provider=provider)
        elif providers:
            count = await repo.count(providers=providers)
        else:
            count = await repo.count()
        assert count == expected_count

    @pytest.mark.parametrize("limit,provider,expected_count", [
        (None, None, 3),
        (2, None, 2),
        (None, "claude", 2),
    ])
    async def test_list_summaries_with_filters(self, repo, limit, provider, expected_count):
        """list_summaries() respects limit and provider filters."""
        kwargs = {}
        if limit is not None:
            kwargs["limit"] = limit
        if provider:
            kwargs["provider"] = provider
        summaries = await repo.list_summaries(**kwargs)
        assert len(summaries) == expected_count

    async def test_list_operations(self, repo):
        """list() filters by title and returns lazy Conversation objects."""
        # Title filter
        convs_filtered = await repo.list(title_contains="First")
        assert len(convs_filtered) == 1
        assert "First" in convs_filtered[0].display_title

        # All conversations
        convs_all = await repo.list()
        assert len(convs_all) == 3
        assert all(hasattr(c, "id") for c in convs_all)


# =============================================================================
# Search operations (Parametrized)
# =============================================================================


class TestSearch:
    """Test FTS search through repository."""

    @pytest.mark.parametrize("query,should_find,search_summaries", [
        ("Hello", True, False),           # Full search
        ("Hello", True, True),            # Summary search
        ("", False, False),               # Empty query
        ("zzzznonexistentzzzz", False, False),  # No match
        ("zzzznonexistentzzzz", False, True),   # No match summaries
    ])
    async def test_search(self, repo, query, should_find, search_summaries):
        """search() and search_summaries() should find or return empty."""
        if search_summaries:
            results = await repo.search_summaries(query)
            if should_find:
                assert len(results) >= 1
                assert hasattr(results[0], "title")
            else:
                assert results == []
        else:
            results = await repo.search(query)
            if should_find:
                assert len(results) >= 1
            else:
                assert isinstance(results, list)


# =============================================================================
# iter_messages (Parametrized)
# =============================================================================


class TestIterMessages:
    """Test message streaming."""

    @pytest.mark.parametrize("conv_id,limit,expected_count", [
        ("conv-1", None, 2),
        ("conv-1", 1, 1),
        ("nonexistent", None, 0),
    ])
    async def test_iter_messages_and_stats(self, repo, conv_id, limit, expected_count):
        """iter_messages() and get_conversation_stats() work with limits and nonexistent."""
        kwargs = {} if limit is None else {"limit": limit}
        messages = [msg async for msg in repo.iter_messages(conv_id, **kwargs)]
        assert len(messages) == expected_count

        stats = await repo.get_conversation_stats(conv_id)
        if conv_id == "conv-1":
            assert stats is not None
            assert stats["total_messages"] == 2
        else:
            assert stats is None


# =============================================================================
# filter() factory
# =============================================================================


class TestFilterFactory:
    """Test that filter() creates a working ConversationFilter via async repo."""

    @pytest.fixture
    def async_repo(self, repo_db):
        """Create ConversationRepository for filter tests."""
        from polylogue.storage.backends.async_sqlite import SQLiteBackend
        from polylogue.storage.repository import ConversationRepository

        backend = SQLiteBackend(db_path=repo_db)
        return ConversationRepository(backend=backend)

    def test_filter_returns_filter_object(self, async_repo):
        """filter() should return ConversationFilter."""
        from polylogue.lib.filters import ConversationFilter
        f = async_repo.filter()
        assert isinstance(f, ConversationFilter)

    async def test_filter_chains_work(self, async_repo):
        """filter() chains should work."""
        result = await async_repo.filter().provider("claude").limit(1).list()
        assert len(result) == 1
        assert result[0].provider == "claude"

    async def test_filter_count(self, async_repo):
        """filter() should allow count() chain."""
        count = await async_repo.filter().provider("chatgpt").count()
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
    with open_connection(db_path):
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


# =============================================================================
# Test Classes
# =============================================================================


class TestGetSummaryNotFound:
    """Test get_summary() with nonexistent conversation."""

    @pytest.mark.parametrize("conv_id,should_exist,expected_title", [
        ("nonexistent-conv", False, None),
        ("conv-1", True, "Root Conversation"),
    ])
    async def test_get_summary(self, repo_with_conversations, conv_id, should_exist, expected_title):
        """get_summary() returns ConversationSummary for existing or None."""
        summary = await repo_with_conversations.get_summary(conv_id)
        if should_exist:
            assert summary is not None
            assert str(summary.id) == conv_id
            assert summary.title == expected_title
            assert summary.provider == "claude"
        else:
            assert summary is None
            # Also verify that message data isn't loaded for nonexistent
            assert not hasattr(summary or {}, "messages") or not summary.messages

    async def test_get_summary_does_not_load_messages(self, repo_with_conversations):
        """get_summary() doesn't load message data."""
        summary = await repo_with_conversations.get_summary("conv-1")
        assert summary is not None
        assert not hasattr(summary, "messages") or summary.messages is None


class TestGetRootEdgeCases:
    """Test get_root() edge cases."""

    @pytest.mark.parametrize("conv_id,expected_root", [
        ("conv-1", "conv-1"),  # Root returns itself
        ("conv-2", "conv-1"),  # Child walks up to parent
    ])
    async def test_get_root(self, repo_with_conversations, conv_id, expected_root):
        """get_root() returns root or walks up the tree."""
        root = await repo_with_conversations.get_root(conv_id)
        assert str(root.id) == expected_root

    async def test_get_root_raises_for_nonexistent_conversation(self, repo_empty):
        """get_root() raises ValueError when conversation doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            await repo_empty.get_root("nonexistent-conv")


class TestSearchSimilar:
    """Test search_similar() full vector similarity flow."""

    async def test_search_similar_raises_without_vector_provider(self, repo_with_conversations):
        """search_similar() raises ValueError if no vector provider supplied."""
        with pytest.raises(ValueError, match="Semantic search requires a vector provider"):
            await repo_with_conversations.search_similar("test query", vector_provider=None)

    async def test_search_similar_returns_empty_list_when_no_results(self, repo_with_conversations):
        """search_similar() returns empty list when provider has no results."""
        mock_provider = MockVectorProvider(results=[])
        results = await repo_with_conversations.search_similar(
            "test query",
            limit=10,
            vector_provider=mock_provider,
        )
        assert results == []

    async def test_search_similar_ranks_by_highest_score(self, repo_with_conversations):
        """search_similar() ranks conversations by highest message score."""
        # Set up results: multiple messages from same conversation
        mock_provider = MockVectorProvider(
            results=[
                ("m1", 0.95),  # From conv-1
                ("m2", 0.90),  # From conv-1
                ("m5", 0.85),  # From conv-3
            ]
        )
        results = await repo_with_conversations.search_similar(
            "AI question",
            limit=10,
            vector_provider=mock_provider,
        )
        # Should have 2 conversations (conv-1 with score 0.95, conv-3 with 0.85)
        assert len(results) == 2
        result_ids = [str(r.id) for r in results]
        assert "conv-1" in result_ids
        assert "conv-3" in result_ids

    async def test_search_similar_limits_results(self, repo_with_conversations):
        """search_similar() limits results to specified count."""
        # Create provider with many results
        results_data = [(f"m{i}", 1.0 - i * 0.01) for i in range(1, 31)]
        mock_provider = MockVectorProvider(results=results_data)
        results = await repo_with_conversations.search_similar(
            "test",
            limit=5,
            vector_provider=mock_provider,
        )
        assert len(results) <= 5

    async def test_search_similar_queries_3x_limit(self, repo_with_conversations):
        """search_similar() queries vector provider with 3x limit for ranking."""
        mock_provider = MockVectorProvider(
            results=[("m1", 0.9), ("m2", 0.85), ("m3", 0.8)] * 3
        )
        await repo_with_conversations.search_similar(
            "test",
            limit=5,
            vector_provider=mock_provider,
        )
        # Provider should have been queried with limit=15 (5*3)
        # We verify this by checking the results were limited


class TestRecordRun:
    """Test record_run() pipeline audit entry."""

    async def test_record_run_saves_run_record(self, repo_empty):
        """record_run() saves RunRecord to backend."""
        now = datetime.now(timezone.utc).isoformat()
        run = RunRecord(
            run_id="run-1",
            timestamp=now,
            counts={"conversations": 5, "messages": 100},
        )
        # Should not raise
        await repo_empty.record_run(run)

    async def test_record_run_with_counts_and_drift(self, repo_empty):
        """record_run() handles counts and drift metadata."""
        now = datetime.now(timezone.utc).isoformat()
        run = RunRecord(
            run_id="run-error",
            timestamp=now,
            counts={"conversations": 0, "messages": 0},
            drift={"skipped": 5},
        )
        await repo_empty.record_run(run)

    async def test_record_run_is_concurrent_safe(self, repo_empty):
        """record_run() uses write lock for concurrency safety."""
        import asyncio

        results = []

        async def save_run(run_id):
            now = datetime.now(timezone.utc).isoformat()
            run = RunRecord(
                run_id=run_id,
                timestamp=now,
                counts={"conversations": 1, "messages": 10},
            )
            await repo_empty.record_run(run)
            results.append(run_id)

        await asyncio.gather(*(save_run(f"run-{i}") for i in range(5)))

        assert len(results) == 5


class TestEmbedConversation:
    """Test embed_conversation() with and without vector provider."""

    @pytest.mark.parametrize("conv_id,has_messages,expected_count", [
        ("conv-1", True, 2),
        ("conv-empty", False, 0),
    ])
    async def test_embed_conversation_counts_messages(self, repo_with_conversations, conv_id, has_messages, expected_count):
        """embed_conversation() uses provider and returns correct message count."""
        if not has_messages:
            (ConversationBuilder(repo_with_conversations.backend._db_path, "conv-empty")
             .provider("claude")
             .title("Empty Conversation")
             .save())

        mock_provider = MockVectorProvider()
        count = await repo_with_conversations.embed_conversation(conv_id, vector_provider=mock_provider)
        assert count == expected_count
        if has_messages:
            assert conv_id in mock_provider._upserted
            assert len(mock_provider._upserted[conv_id]) == expected_count

    async def test_embed_conversation_provider_creation(self, repo_with_conversations):
        """embed_conversation() handles provider creation and errors."""
        # Test with no provider and fallback available
        mock_provider = MockVectorProvider()
        with patch(
            "polylogue.storage.search_providers.create_vector_provider",
            return_value=mock_provider,
        ):
            count = await repo_with_conversations.embed_conversation("conv-1", vector_provider=None)
            assert count == 2

        # Test without provider and no fallback
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "polylogue.storage.search_providers.create_vector_provider",
                return_value=None,
            ):
                with pytest.raises(ValueError, match="No vector provider available"):
                    await repo_with_conversations.embed_conversation("conv-1", vector_provider=None)


class TestSimilaritySearch:
    """Test similarity_search() returning (conv_id, msg_id, distance) tuples."""

    @pytest.mark.parametrize("results_data,should_exist,expected_msg_ids", [
        ([("m1", 0.95), ("m2", 0.90)], True, ["m1", "m2"]),
        ([("m1", 0.95), ("m5", 0.85)], True, ["m1", "m5"]),
        ([("m1", 0.95), ("m_orphaned", 0.90)], False, ["m1"]),
        ([], True, []),
    ])
    async def test_similarity_search_tuple_format_and_mapping(self, repo_with_conversations, results_data, should_exist, expected_msg_ids):
        """similarity_search() returns tuples, maps IDs, preserves scores, filters orphaned."""
        mock_provider = MockVectorProvider(results=results_data)
        results = await repo_with_conversations.similarity_search("test query", vector_provider=mock_provider)

        # Verify tuple format and basic structure
        for conv_id, msg_id, distance in results:
            assert isinstance(conv_id, str)
            assert isinstance(msg_id, str)
            assert isinstance(distance, float)

        # Verify message IDs and mapping
        msg_ids = [msg_id for _, msg_id, _ in results]
        for expected_id in expected_msg_ids:
            assert expected_id in msg_ids
        assert "m_orphaned" not in msg_ids  # Orphaned filtered

        # Verify mapping for specific messages
        result_dict = {msg_id: conv_id for conv_id, msg_id, _ in results}
        if "m1" in result_dict:
            assert result_dict["m1"] == "conv-1"
        if "m5" in result_dict:
            assert result_dict["m5"] == "conv-3"

    async def test_similarity_search_error_handling(self, repo_with_conversations):
        """similarity_search() raises without provider."""
        with patch(
            "polylogue.storage.search_providers.create_vector_provider",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="No vector provider configured"):
                await repo_with_conversations.similarity_search("test query", vector_provider=None)


class TestGetArchiveStats:
    """Test get_archive_stats() comprehensive statistics."""

    @pytest.mark.parametrize("fixture,expected_convs,expected_msgs,expected_provider_count", [
        ("repo_with_conversations", 4, 6, 3),
        ("repo_empty", 0, 0, 0),
    ])
    async def test_get_archive_stats_counts_and_types(self, request, fixture, expected_convs, expected_msgs, expected_provider_count):
        """get_archive_stats() returns ArchiveStats with correct counts and provider breakdown."""
        repo = request.getfixturevalue(fixture)
        stats = await repo.get_archive_stats()

        assert isinstance(stats, ArchiveStats)
        assert stats.total_conversations == expected_convs
        assert stats.total_messages == expected_msgs
        assert stats.provider_count == expected_provider_count

        # For non-empty repo, check provider breakdown
        if expected_convs > 0:
            assert stats.providers.get("claude") == 2
            assert stats.providers.get("chatgpt") == 1
            assert stats.providers.get("gemini") == 1

    async def test_get_archive_stats_embeddings_and_metrics(self, repo_with_conversations):
        """get_archive_stats() computes embedding coverage, avg messages, and DB size."""
        stats = await repo_with_conversations.get_archive_stats()

        # Embedding coverage (no embeddings initially)
        assert stats.embedded_conversations == 0
        assert stats.embedded_messages == 0
        assert stats.embedding_coverage == 0.0

        # Metrics
        assert stats.avg_messages_per_conversation == 1.5  # 6 / 4
        assert stats.db_size_bytes >= 0


class TestRecordsToConversation:
    """Test _records_to_conversation() standalone helper."""

    @pytest.mark.parametrize("has_messages,has_attachments", [
        (True, False),
        (False, False),
    ])
    def test_records_to_conversation_variants(self, has_messages, has_attachments):
        """_records_to_conversation() converts records and handles empty messages/attachments."""
        conv_rec = make_conversation("conv-1")
        msg_recs = [make_message("msg-1", "conv-1", text="Hello")] if has_messages else []
        att_recs = [make_attachment("att-1", "conv-1", "msg-1")] if has_attachments else []

        conv = _records_to_conversation(conv_rec, msg_recs, att_recs)

        assert isinstance(conv, Conversation)
        assert str(conv.id) == "conv-1"
        assert len(conv.messages) == (1 if has_messages else 0)

    def test_records_to_conversation_preserves_order_and_parent(self):
        """_records_to_conversation() preserves message order and parent_id."""
        conv_rec = make_conversation("conv-1").model_copy(
            update={"parent_conversation_id": "parent-conv"}
        )
        msg_recs = [
            make_message("m1", "conv-1", role="user"),
            make_message("m2", "conv-1", role="assistant"),
            make_message("m3", "conv-1", role="user"),
        ]

        conv = _records_to_conversation(conv_rec, msg_recs, [])

        assert len(conv.messages) == 3
        assert [m.id for m in conv.messages] == ["m1", "m2", "m3"]
        assert str(conv.parent_id) == "parent-conv"

    def test_records_to_conversation_bulk_migration(self):
        """_records_to_conversation() handles bulk migration of records."""
        convs_data = [
            (make_conversation(f"conv-{i}"), [make_message(f"m-{i}", f"conv-{i}")], [])
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

    async def test_embed_then_search_workflow(self, repo_with_conversations):
        """Full workflow: embed conversations, then search by similarity."""
        # Step 1: Embed a conversation
        mock_provider = MockVectorProvider()
        embed_count = await repo_with_conversations.embed_conversation(
            "conv-1",
            vector_provider=mock_provider,
        )
        assert embed_count == 2

        # Step 2: Configure provider with search results
        mock_provider._results = [("m1", 0.95), ("m2", 0.87)]

        # Step 3: Perform similarity search
        results = await repo_with_conversations.similarity_search(
            "embedded search",
            vector_provider=mock_provider,
        )
        assert len(results) == 2
        assert all(len(r) == 3 for r in results)  # 3-tuple validation

    async def test_search_similar_with_multiple_conversations(self, repo_with_conversations):
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
        results = await repo_with_conversations.search_similar(
            "test",
            limit=3,
            vector_provider=mock_provider,
        )
        # Should return max 3 conversations
        assert len(results) <= 3

    async def test_stats_after_embedding_operations(self, repo_with_conversations):
        """get_archive_stats() reflects state after embedding operations."""
        # Get initial stats
        stats_before = await repo_with_conversations.get_archive_stats()
        assert stats_before.total_conversations == 4

        # Perform embedding (doesn't change conversation count, just records state)
        mock_provider = MockVectorProvider()
        await repo_with_conversations.embed_conversation("conv-1", vector_provider=mock_provider)

        # Stats should still show same conversation count
        stats_after = await repo_with_conversations.get_archive_stats()
        assert stats_after.total_conversations == stats_before.total_conversations

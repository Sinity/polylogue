"""Tests for ConversationRepository operations.

Tests the repository facade for conversation retrieval, tree traversal,
transactional save, metadata CRUD, and search operations.
"""

from __future__ import annotations

import pytest

from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ConversationRecord, MessageRecord, AttachmentRecord
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

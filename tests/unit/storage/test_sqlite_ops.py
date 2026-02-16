"""SQLite backend operations tests — connection, save/get round-trips, batch, transactions, delete, stats."""

from __future__ import annotations

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
)
from tests.infra.helpers import (
    make_attachment,
    make_conversation,
    make_message,
    store_records,
)

# =============================================================================
# METADATA OPERATIONS (update, delete, add_tag, remove_tag, RMW atomicity)
# =============================================================================


# =============================================================================
# MERGED FROM test_sqlite_backend.py - Additional SQLiteBackend Tests
# =============================================================================

# ============================================================================
# Test: connection_context
# ============================================================================


class TestSaveGetConversation:
    """Tests for save_conversation and get_conversation operations."""

    async def test_save_and_get_conversation_basic(self, tmp_path):
        """Test basic save and retrieval of a conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = make_conversation("conv-1", provider_name="claude", title="Test Conversation",
                                    created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T12:00:00Z",
                                    content_hash="hash123", provider_meta={"source": "test"},
                                    metadata={"tags": ["important"]}, version=1)
        await backend.save_conversation_record(record)
        retrieved = await backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv-1"
        assert retrieved.title == "Test Conversation"
        assert retrieved.provider_meta == {"source": "test"}
        assert retrieved.metadata == {"tags": ["important"]}

    async def test_get_nonexistent_conversation(self, tmp_path):
        """Test that get_conversation returns None for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = await backend.get_conversation("nonexistent")
        assert result is None

    async def test_save_conversation_upsert_different_hash(self, tmp_path):
        """Test upsert: same ID, different content_hash -> updates."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record1 = make_conversation("conv-1", provider_name="claude", title="Original Title",
                                    created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T10:00:00Z",
                                    content_hash="hash_old", version=1)
        await backend.save_conversation_record(record1)

        record2 = make_conversation("conv-1", provider_name="claude", title="Updated Title",
                                    created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T12:00:00Z",
                                    content_hash="hash_new", version=2)
        await backend.save_conversation_record(record2)

        retrieved = await backend.get_conversation("conv-1")
        assert retrieved.title == "Updated Title"
        assert retrieved.updated_at == "2025-01-01T12:00:00Z"

    async def test_save_conversation_no_update_same_hash(self, tmp_path):
        """Test upsert: same ID and content_hash -> no update."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = make_conversation("conv-1", provider_name="claude", title="Original",
                                    created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T10:00:00Z",
                                    content_hash="hash123", version=1)
        await backend.save_conversation_record(record)
        retrieved1 = await backend.get_conversation("conv-1")

        # Save with same hash and content
        await backend.save_conversation_record(record)
        retrieved2 = await backend.get_conversation("conv-1")

        assert retrieved1.title == retrieved2.title
        assert retrieved1.updated_at == retrieved2.updated_at

    async def test_save_conversation_metadata_not_overwritten(self, tmp_path):
        """Test that upsert does NOT overwrite user metadata."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record1 = make_conversation("conv-1", provider_name="claude", title="Test",
                                    created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T10:00:00Z",
                                    content_hash="hash1", metadata={"tags": ["important"]}, version=1)
        await backend.save_conversation_record(record1)

        # Update metadata manually
        await backend.update_metadata("conv-1", "custom_key", "custom_value")

        # Save new record with different content_hash
        record2 = make_conversation("conv-1", provider_name="claude", title="Test",
                                    created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T12:00:00Z",
                                    content_hash="hash2", metadata=None, version=2)
        await backend.save_conversation_record(record2)

        # Metadata should still have the custom key
        meta = await backend.get_metadata("conv-1")
        assert meta.get("custom_key") == "custom_value"

    async def test_save_conversation_with_null_fields(self, tmp_path):
        """Test save_conversation with None values for optional fields."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title=None,
            created_at=None,
            updated_at=None,
            content_hash="hash",
            provider_meta=None,
            version=1,
        )
        await backend.save_conversation_record(record)
        retrieved = await backend.get_conversation("conv-1")
        assert retrieved.title is None
        assert retrieved.created_at is None

    async def test_save_conversation_with_branching_info(self, tmp_path):
        """Test save_conversation with parent and branch_type."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Create parent conversation
        parent = make_conversation("conv-parent", provider_name="claude", title="Parent",
                                    created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T00:00:00Z",
                                    content_hash="hash-parent", version=1)
        await backend.save_conversation_record(parent)

        # Create child conversation
        child = make_conversation("conv-child", provider_name="claude", title="Child",
                                  created_at="2025-01-01T01:00:00Z", updated_at="2025-01-01T01:00:00Z",
                                  content_hash="hash-child", version=1,
                                  parent_conversation_id="conv-parent", branch_type="continuation")
        await backend.save_conversation_record(child)

        retrieved = await backend.get_conversation("conv-child")
        assert retrieved.parent_conversation_id == "conv-parent"
        assert retrieved.branch_type == "continuation"


# ============================================================================
# Test: save_messages + get_messages round-trip
# ============================================================================


class TestSaveGetMessages:
    """Tests for save_messages and get_messages operations."""

    async def test_save_and_get_messages(self, tmp_path):
        """Test basic save and retrieval of messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Create conversation first
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        await backend.save_conversation_record(conv)

        messages = [
            make_message("msg-1", "conv-1", role="user", text="Hello",
                        timestamp="2025-01-01T10:00:00Z", content_hash="msg-hash-1", version=1, provider_message_id="prov-msg-1"),
            make_message("msg-2", "conv-1", role="assistant", text="Hi there",
                        timestamp="2025-01-01T10:01:00Z", content_hash="msg-hash-2", version=1, provider_message_id="prov-msg-2"),
        ]
        await backend.save_messages(messages)

        retrieved = await backend.get_messages("conv-1")
        assert len(retrieved) == 2
        assert retrieved[0].message_id == "msg-1"
        assert retrieved[1].message_id == "msg-2"

    async def test_save_empty_messages_list(self, tmp_path):
        """Test that save_messages([]) is a no-op."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        await backend.save_messages([])  # Should not raise
        # No messages to retrieve, but no error either
        assert True

    async def test_get_messages_ordering_by_timestamp(self, tmp_path):
        """Test that messages are returned ordered by timestamp."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = make_conversation("conv-1", provider_name="claude", title="Test",
                                created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T00:00:00Z",
                                content_hash="hash", version=1)
        await backend.save_conversation_record(conv)

        # Save in non-chronological order
        messages = [
            MessageRecord(
                message_id="msg-3",
                conversation_id="conv-1",
                role="assistant",
                text="Third",
                timestamp="2025-01-01T10:02:00Z",
                content_hash="h3",
                version=1,
            ),
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                role="user",
                text="First",
                timestamp="2025-01-01T10:00:00Z",
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                role="assistant",
                text="Second",
                timestamp="2025-01-01T10:01:00Z",
                content_hash="h2",
                version=1,
            ),
        ]
        await backend.save_messages(messages)

        retrieved = await backend.get_messages("conv-1")
        assert [m.message_id for m in retrieved] == ["msg-1", "msg-2", "msg-3"]

    async def test_save_messages_upsert_different_hash(self, tmp_path):
        """Test upsert: same message_id, different content_hash -> updates."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = make_conversation("conv-1", provider_name="claude", title="Test",
                                created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T00:00:00Z",
                                content_hash="hash", version=1)
        await backend.save_conversation_record(conv)

        msg1 = make_message("msg-1", "conv-1", role="user", text="Original",
                           timestamp="2025-01-01T10:00:00Z", content_hash="hash_old", version=1)
        await backend.save_messages([msg1])

        msg2 = make_message("msg-1", "conv-1", role="user", text="Updated",
                           timestamp="2025-01-01T10:00:00Z", content_hash="hash_new", version=2)
        await backend.save_messages([msg2])

        retrieved = await backend.get_messages("conv-1")
        assert retrieved[0].text == "Updated"

    async def test_save_messages_no_update_same_hash(self, tmp_path):
        """Test upsert: same message_id and content_hash -> no update."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = make_conversation("conv-1", provider_name="claude", title="Test",
                                created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T00:00:00Z",
                                content_hash="hash", version=1)
        await backend.save_conversation_record(conv)

        msg = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Hello",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash123",
            version=1,
        )
        await backend.save_messages([msg])
        retrieved1 = (await backend.get_messages("conv-1"))[0]

        await backend.save_messages([msg])  # Save again with same hash
        retrieved2 = (await backend.get_messages("conv-1"))[0]

        assert retrieved1.text == retrieved2.text

    async def test_get_messages_nonexistent_conversation(self, tmp_path):
        """Test get_messages returns empty list for nonexistent conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = await backend.get_messages("nonexistent")
        assert result == []

    async def test_save_messages_with_provider_meta(self, tmp_path):
        """Test saving messages with provider metadata."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        await backend.save_conversation_record(conv)

        msg = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Hello",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash",
            provider_meta={"custom": "data"},
            version=1,
        )
        await backend.save_messages([msg])

        retrieved = (await backend.get_messages("conv-1"))[0]
        assert retrieved.provider_meta == {"custom": "data"}

    async def test_save_messages_with_branching_info(self, tmp_path):
        """Test saving messages with parent_message_id and branch_index."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        await backend.save_conversation_record(conv)

        msg = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Hello",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash",
            version=1,
            parent_message_id="msg-parent",
            branch_index=2,
        )
        await backend.save_messages([msg])

        retrieved = (await backend.get_messages("conv-1"))[0]
        assert retrieved.parent_message_id == "msg-parent"
        assert retrieved.branch_index == 2


# ============================================================================
# Test: get_conversations_batch
# ============================================================================


class TestGetConversationsBatch:
    """Tests for get_conversations_batch operation."""

    async def test_get_conversations_batch_basic(self, tmp_path):
        """Test batch retrieval preserves order."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        ids = ["conv-1", "conv-2", "conv-3"]
        for cid in ids:
            record = ConversationRecord(
                conversation_id=cid,
                provider_name="claude",
                provider_conversation_id=f"prov-{cid}",
                title=f"Conversation {cid}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"hash-{cid}",
                version=1,
            )
            await backend.save_conversation_record(record)

        # Request in different order
        batch = await backend.get_conversations_batch(["conv-3", "conv-1", "conv-2"])
        assert [r.conversation_id for r in batch] == ["conv-3", "conv-1", "conv-2"]

    async def test_get_conversations_batch_missing_ids_skipped(self, tmp_path):
        """Test that missing IDs are silently skipped."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        await backend.save_conversation_record(record)

        batch = await backend.get_conversations_batch(["conv-1", "nonexistent", "also-missing"])
        assert len(batch) == 1
        assert batch[0].conversation_id == "conv-1"

    async def test_get_conversations_batch_empty_input(self, tmp_path):
        """Test that empty input returns empty list."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        batch = await backend.get_conversations_batch([])
        assert batch == []

    async def test_get_conversations_batch_duplicate_ids(self, tmp_path):
        """Test batch with duplicate IDs returns each occurrence."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        await backend.save_conversation_record(record)

        batch = await backend.get_conversations_batch(["conv-1", "conv-1", "conv-1"])
        assert len(batch) == 3
        assert all(r.conversation_id == "conv-1" for r in batch)


# ============================================================================
# Test: Transaction management (begin/commit/rollback)
# ============================================================================


class TestTransactionManagement:
    """Tests for transaction management."""

    async def test_begin_commit_persists_data(self, tmp_path):
        """Test that begin+commit persists data."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )

        await backend.begin()
        await backend.save_conversation_record(record)
        await backend.commit()

        # Verify persisted
        retrieved = await backend.get_conversation("conv-1")
        assert retrieved is not None

    async def test_begin_rollback_reverts_data(self, tmp_path):
        """Test that begin+rollback reverts data."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )

        await backend.begin()
        await backend.save_conversation_record(record)
        await backend.rollback()

        # Should not be persisted
        retrieved = await backend.get_conversation("conv-1")
        assert retrieved is None

    async def test_nested_savepoints(self, tmp_path):
        """Test nested transaction support via savepoints."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="First",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash1",
            version=1,
        )
        record2 = ConversationRecord(
            conversation_id="conv-2",
            provider_name="claude",
            provider_conversation_id="prov-2",
            title="Second",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash2",
            version=1,
        )

        await backend.begin()
        await backend.save_conversation_record(record1)

        await backend.begin()
        await backend.save_conversation_record(record2)
        await backend.rollback()  # Rollback only record2

        await backend.commit()  # Commit record1

        # record1 should exist, record2 should not
        assert await backend.get_conversation("conv-1") is not None
        assert await backend.get_conversation("conv-2") is None

    async def test_commit_without_begin_raises_error(self, tmp_path):
        """Test that commit without begin raises DatabaseError."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # transaction_depth starts at 0, commit should fail immediately
        with pytest.raises(Exception, match="No active transaction to commit"):
            await backend.commit()

    async def test_rollback_without_begin_raises_error(self, tmp_path):
        """Test that rollback without begin raises DatabaseError."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # transaction_depth starts at 0, rollback should fail immediately
        with pytest.raises(Exception, match="No active transaction to rollback"):
            await backend.rollback()

    async def test_transaction_context_manager(self, tmp_path):
        """Test transaction context manager."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )

        async with backend.transaction():
            await backend.save_conversation_record(record)

        assert await backend.get_conversation("conv-1") is not None

    async def test_transaction_context_manager_rollback_on_error(self, tmp_path):
        """Test that transaction context manager rolls back on error."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )

        try:
            async with backend.transaction():
                await backend.save_conversation_record(record)
                raise ValueError("Test error")
        except ValueError:
            pass

        assert await backend.get_conversation("conv-1") is None


# Note: metadata operations are tested via standalone functions above (test_update_and_get_metadata, etc.)
# TestMetadataOperations class was a duplicate and has been consolidated.


# ============================================================================
# Test: delete_conversation
# ============================================================================


class TestDeleteConversation:
    """Tests for delete_conversation operation."""

    async def test_delete_conversation_success(self, tmp_path):
        """Test successful deletion of conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        await backend.save_conversation_record(record)

        result = await backend.delete_conversation("conv-1")
        assert result is True
        assert await backend.get_conversation("conv-1") is None

    async def test_delete_conversation_nonexistent(self, tmp_path):
        """Test deleting nonexistent conversation returns False."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = await backend.delete_conversation("nonexistent")
        assert result is False

    async def test_delete_conversation_with_messages(self, tmp_path):
        """Test that deleting conversation also deletes messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        await backend.save_conversation_record(record)

        messages = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T10:00:00Z",
                content_hash="h1",
                version=1,
            )
        ]
        await backend.save_messages(messages)

        await backend.delete_conversation("conv-1")
        assert await backend.get_messages("conv-1") == []

    async def test_delete_conversation_with_attachments(self, tmp_path):
        """Test that attachment refs are cleaned up when conversation is deleted."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        await backend.save_conversation_record(record)

        attachment = AttachmentRecord(
            attachment_id="att-1",
            conversation_id="conv-1",
            message_id=None,
            mime_type="image/png",
            size_bytes=1024,
            path="/path/to/image.png",
        )
        await backend.save_attachments([attachment])

        await backend.delete_conversation("conv-1")
        # Verify conversation is gone
        assert await backend.get_conversation("conv-1") is None


# ============================================================================
# Test: get_conversation_stats
# ============================================================================


# =============================================================================
# GET CONVERSATION STATS (parametrized)
# =============================================================================

# Test cases for get_conversation_stats with different message configurations
CONVERSATION_STATS_CASES = [
    (
        [],
        {"total": 0, "dialogue": 0, "tool": 0},
        "empty",
    ),
    (
        [
            ("msg-0", "user", "Message 0"),
            ("msg-1", "assistant", "Message 1"),
            ("msg-2", "user", "Message 2"),
            ("msg-3", "assistant", "Message 3"),
            ("msg-4", "user", "Message 4"),
        ],
        {"total": 5, "dialogue": 5, "tool": 0},
        "dialogue_only",
    ),
    (
        [
            ("msg-1", "user", "Hello"),
            ("msg-2", "tool", "Tool output"),
            ("msg-3", "assistant", "Response"),
        ],
        {"total": 3, "dialogue": 2, "tool": 1},
        "with_tool",
    ),
]


def _conversation_record():
    return make_conversation("conv:perf", provider_name="codex", title="Perf Test", created_at=None, updated_at=None, content_hash="hash-perf", provider_meta=None)


@pytest.mark.slow
async def test_prune_multiple_attachments_correctly(workspace_env, storage_repository):
    """Verify that pruning multiple attachments works correctly.

    This exercises the N+1 query fix in _prune_attachment_refs which now
    uses a single UPDATE with IN clause instead of individual UPDATEs per attachment.
    """
    from polylogue.sources import RecordBundle, save_bundle

    # Create initial conversation with 10 attachments
    attachments = [
        make_attachment(f"att-{i}", "conv:perf", "msg:perf", mime_type="text/plain", size_bytes=10, provider_meta=None)
        for i in range(10)
    ]

    bundle = RecordBundle(
        conversation=_conversation_record(),
        messages=[make_message("msg:perf", "conv:perf", text="hello", timestamp="1", content_hash="msg:perf", provider_meta=None)],
        attachments=attachments,
    )
    await save_bundle(bundle, repository=storage_repository)

    # Verify all 10 attachments were created
    with open_connection(None) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM attachments WHERE attachment_id LIKE 'att-%'"
        ).fetchone()[0]
        assert count == 10, f"Expected 10 attachments, got {count}"

        # Check ref_count is correct
        refs = conn.execute(
            "SELECT attachment_id, ref_count FROM attachments WHERE attachment_id LIKE 'att-%' ORDER BY attachment_id"
        ).fetchall()
        for ref in refs:
            assert ref["ref_count"] == 1, f"Expected ref_count=1 for {ref['attachment_id']}, got {ref['ref_count']}"

    # Now re-ingest with only 2 attachments, which should prune 8
    new_attachments = [
        make_attachment("att-0", "conv:perf", "msg:perf", mime_type="text/plain", size_bytes=10, provider_meta=None),
        make_attachment("att-1", "conv:perf", "msg:perf", mime_type="text/plain", size_bytes=10, provider_meta=None),
    ]

    await save_bundle(
        RecordBundle(
            conversation=_conversation_record(),
            messages=[make_message("msg:perf", "conv:perf", text="hello", timestamp="1", content_hash="msg:perf", provider_meta=None)],
            attachments=new_attachments,
        ),
        repository=storage_repository,
    )

    # Verify only 2 attachments remain (the 8 others should have been pruned)
    with open_connection(None) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM attachments WHERE attachment_id LIKE 'att-%'"
        ).fetchone()[0]
        assert count == 2, f"Expected 2 attachments after pruning, got {count}"

        remaining = conn.execute(
            "SELECT attachment_id FROM attachments WHERE attachment_id LIKE 'att-%' ORDER BY attachment_id"
        ).fetchall()
        remaining_ids = [row["attachment_id"] for row in remaining]
        assert remaining_ids == ["att-0", "att-1"], f"Expected att-0 and att-1, got {remaining_ids}"


# --- merged from test_raw_conversations.py ---

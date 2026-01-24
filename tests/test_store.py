from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from polylogue.storage.db import open_connection
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    _make_ref_id,
    _prune_attachment_refs,
    store_records,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    db_path = tmp_path / "test.db"
    with open_connection(db_path):
        pass
    return db_path


@pytest.fixture
def test_conn(test_db):
    """Provide a connection to the test database."""
    with open_connection(test_db) as conn:
        yield conn


def test_store_records_inserts_new_conversation(test_conn):
    """store_records() inserts a new conversation with messages."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test Conversation",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash123",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="Hello",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="msghash1",
        version=1,
    )

    counts = store_records(conversation=conv, messages=[msg], attachments=[], conn=test_conn)

    assert counts["conversations"] == 1
    assert counts["messages"] == 1
    assert counts["skipped_conversations"] == 0
    assert counts["skipped_messages"] == 0

    # Verify in database
    row = test_conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
    assert row is not None
    assert row["title"] == "Test Conversation"

    msg_row = test_conn.execute("SELECT * FROM messages WHERE message_id = ?", ("msg1",)).fetchone()
    assert msg_row is not None
    assert msg_row["text"] == "Hello"


def test_store_records_skips_duplicate_conversation(test_conn):
    """store_records() skips duplicate conversations with same content_hash."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Same Title",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="samehash",
        version=1,
    )

    # First insert
    counts1 = store_records(conversation=conv, messages=[], attachments=[], conn=test_conn)
    assert counts1["conversations"] == 1

    # Second insert with same hash
    counts2 = store_records(conversation=conv, messages=[], attachments=[], conn=test_conn)
    assert counts2["conversations"] == 0
    assert counts2["skipped_conversations"] == 1


def test_store_records_updates_changed_conversation(test_conn):
    """store_records() updates conversation when content changes."""
    conv1 = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Original Title",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    store_records(conversation=conv1, messages=[], attachments=[], conn=test_conn)

    # Update with different content
    conv2 = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Updated Title",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-02T00:00:00Z",
        content_hash="hash2",  # Different hash
        version=1,
    )
    counts = store_records(conversation=conv2, messages=[], attachments=[], conn=test_conn)

    assert counts["conversations"] == 1
    assert counts["skipped_conversations"] == 0

    # Verify update
    row = test_conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
    assert row["title"] == "Updated Title"
    assert row["content_hash"] == "hash2"


def test_store_records_handles_multiple_messages(test_conn):
    """store_records() correctly handles multiple messages."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Multi Message",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )

    messages = [
        MessageRecord(
            message_id=f"msg{i}",
            conversation_id="conv1",
            role="user" if i % 2 == 0 else "assistant",
            text=f"Message {i}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash=f"hash{i}",
            version=1,
        )
        for i in range(5)
    ]

    counts = store_records(conversation=conv, messages=messages, attachments=[], conn=test_conn)

    assert counts["messages"] == 5
    assert counts["skipped_messages"] == 0

    # Verify all messages in database
    rows = test_conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", ("conv1",)).fetchone()
    assert rows[0] == 5


def test_store_records_attachment_ref_counting(test_conn):
    """store_records() correctly maintains attachment ref_count."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Attachment Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )

    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        provider_message_id="ext-msg1",
        role="user",
        text="Test message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg1",
        version=1,
    )

    att1 = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/png",
        size_bytes=1024,
        path="/path/to/file.png",
    )

    counts = store_records(conversation=conv, messages=[msg1], attachments=[att1], conn=test_conn)

    assert counts["attachments"] == 1

    # Check ref_count
    row = test_conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert row["ref_count"] == 1

    # Check attachment_refs
    ref_rows = test_conn.execute("SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?", ("att1",)).fetchone()
    assert ref_rows[0] == 1


def test_prune_attachment_refs_removes_old_refs(test_conn):
    """_prune_attachment_refs() removes refs not in keep_ref_ids set."""
    # Setup: Insert conversation and attachments
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Prune Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )

    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        provider_message_id="ext-msg1",
        role="user",
        text="First message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg1",
        version=1,
    )
    msg2 = MessageRecord(
        message_id="msg2",
        conversation_id="conv1",
        provider_message_id="ext-msg2",
        role="user",
        text="Second message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg2",
        version=1,
    )

    att1 = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/png",
        size_bytes=1024,
    )
    att2 = AttachmentRecord(
        attachment_id="att2",
        conversation_id="conv1",
        message_id="msg2",
        mime_type="image/jpeg",
        size_bytes=2048,
    )

    store_records(conversation=conv, messages=[msg1, msg2], attachments=[att1, att2], conn=test_conn)

    # Get ref IDs
    ref_id1 = _make_ref_id("att1", "conv1", "msg1")
    ref_id2 = _make_ref_id("att2", "conv1", "msg2")

    # Verify both exist
    count_before = test_conn.execute(
        "SELECT COUNT(*) FROM attachment_refs WHERE conversation_id = ?", ("conv1",)
    ).fetchone()[0]
    assert count_before == 2

    # Prune, keeping only ref_id1
    _prune_attachment_refs(test_conn, "conv1", {ref_id1})

    # Verify only one ref remains
    count_after = test_conn.execute(
        "SELECT COUNT(*) FROM attachment_refs WHERE conversation_id = ?", ("conv1",)
    ).fetchone()[0]
    assert count_after == 1

    # Verify correct ref was kept
    remaining = test_conn.execute(
        "SELECT ref_id FROM attachment_refs WHERE conversation_id = ?", ("conv1",)
    ).fetchone()
    assert remaining["ref_id"] == ref_id1


def test_prune_attachment_refs_updates_ref_count(test_conn):
    """_prune_attachment_refs() updates attachment ref_count correctly."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="RefCount Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )

    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        provider_message_id="ext-msg1",
        role="user",
        text="First message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg1",
        version=1,
    )
    msg2 = MessageRecord(
        message_id="msg2",
        conversation_id="conv1",
        provider_message_id="ext-msg2",
        role="user",
        text="Second message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg2",
        version=1,
    )

    # Same attachment referenced twice
    att1 = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/png",
        size_bytes=1024,
    )
    att2 = AttachmentRecord(
        attachment_id="att1",  # Same attachment_id
        conversation_id="conv1",
        message_id="msg2",  # Different message
        mime_type="image/png",
        size_bytes=1024,
    )

    store_records(conversation=conv, messages=[msg1, msg2], attachments=[att1, att2], conn=test_conn)

    # ref_count should be 2
    row = test_conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert row["ref_count"] == 2

    # Prune one reference
    ref_id1 = _make_ref_id("att1", "conv1", "msg1")
    _prune_attachment_refs(test_conn, "conv1", {ref_id1})

    # ref_count should now be 1
    row = test_conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert row["ref_count"] == 1


def test_prune_attachment_refs_deletes_zero_ref_attachments(test_conn):
    """_prune_attachment_refs() deletes attachments with ref_count <= 0."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Delete Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )

    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        provider_message_id="ext-msg1",
        role="user",
        text="Test message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg1",
        version=1,
    )

    att = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/png",
        size_bytes=1024,
    )

    store_records(conversation=conv, messages=[msg1], attachments=[att], conn=test_conn)

    # Verify attachment exists
    row = test_conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert row is not None

    # Prune all refs
    _prune_attachment_refs(test_conn, "conv1", set())

    # Verify attachment was deleted
    row = test_conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert row is None


def test_upsert_conversation_missing_optional_fields(test_conn):
    """upsert_conversation() handles None values for optional fields."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title=None,  # Optional
        created_at=None,  # Optional
        updated_at=None,  # Optional
        content_hash="hash1",
        provider_meta=None,  # Optional
        version=1,
    )

    updated = upsert_conversation(test_conn, conv)
    assert updated

    row = test_conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
    assert row is not None
    assert row["title"] is None
    assert row["created_at"] is None
    assert row["provider_meta"] is None


def test_upsert_message_missing_optional_fields(test_conn):
    """upsert_message() handles None values for optional fields."""
    # First insert conversation
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )
    upsert_conversation(test_conn, conv)

    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        provider_message_id=None,  # Optional
        role=None,  # Optional
        text=None,  # Optional
        timestamp=None,  # Optional
        content_hash="msghash1",
        provider_meta=None,  # Optional
        version=1,
    )

    updated = upsert_message(test_conn, msg)
    assert updated

    row = test_conn.execute("SELECT * FROM messages WHERE message_id = ?", ("msg1",)).fetchone()
    assert row is not None
    assert row["role"] is None
    assert row["text"] is None
    assert row["provider_message_id"] is None


def test_upsert_attachment_duplicate_ref_skipped(test_conn):
    """upsert_attachment() skips duplicate refs (INSERT OR IGNORE)."""
    # Setup conversation
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )
    upsert_conversation(test_conn, conv)

    # Setup message
    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        provider_message_id="ext-msg1",
        role="user",
        text="Test message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg1",
        version=1,
    )
    upsert_message(test_conn, msg1)

    att = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/png",
        size_bytes=1024,
    )

    # First insert
    updated1 = upsert_attachment(test_conn, att)
    assert updated1 is True

    # Second insert (duplicate)
    updated2 = upsert_attachment(test_conn, att)
    assert updated2 is False

    # ref_count should still be 1
    row = test_conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert row["ref_count"] == 1


def test_make_ref_id_consistency(test_conn):
    """_make_ref_id() generates consistent hashes."""
    ref_id1 = _make_ref_id("att1", "conv1", "msg1")
    ref_id2 = _make_ref_id("att1", "conv1", "msg1")
    assert ref_id1 == ref_id2

    # Different message should give different ref_id
    ref_id3 = _make_ref_id("att1", "conv1", "msg2")
    assert ref_id1 != ref_id3

    # None message_id should be handled
    ref_id4 = _make_ref_id("att1", "conv1", None)
    assert ref_id4.startswith("ref-")


def test_make_ref_id_format(test_conn):
    """_make_ref_id() returns expected format."""
    ref_id = _make_ref_id("att1", "conv1", "msg1")
    assert ref_id.startswith("ref-")
    assert len(ref_id) == 20  # "ref-" + 16 hex chars


def test_write_lock_prevents_concurrent_writes(test_db):
    """_WRITE_LOCK prevents concurrent store_records() calls from corrupting data."""
    results = []
    errors = []

    def write_conversation(conv_id: int):
        try:
            conv = ConversationRecord(
                conversation_id=f"conv{conv_id}",
                provider_name="test",
                provider_conversation_id=f"ext-conv{conv_id}",
                title=f"Conversation {conv_id}",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                content_hash=uuid4().hex,
                version=1,
            )

            messages = [
                MessageRecord(
                    message_id=f"msg{conv_id}-{i}",
                    conversation_id=f"conv{conv_id}",
                    role="user",
                    text=f"Message {i}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash=uuid4().hex,
                    version=1,
                )
                for i in range(3)
            ]

            with open_connection(test_db) as conn:
                counts = store_records(conversation=conv, messages=messages, attachments=[], conn=conn)
            results.append(counts)
        except Exception as e:
            errors.append(e)

    # Run multiple concurrent writes
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write_conversation, i) for i in range(10)]
        for future in as_completed(futures):
            future.result()

    # No errors should occur
    assert len(errors) == 0

    # All writes should succeed
    assert len(results) == 10

    # Verify all conversations were written
    with open_connection(test_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        assert count == 10

        # Verify message count
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert msg_count == 30  # 10 conversations × 3 messages


def test_store_records_without_connection_creates_own(test_db):
    """store_records() works without explicit connection parameter."""
    # Set the DB path in environment so default_db_path() returns test_db
    import os

    os.environ["XDG_STATE_HOME"] = str(test_db.parent.parent)

    # Move test DB to default location
    from polylogue.storage.db import default_db_path

    default_path = default_db_path()
    default_path.parent.mkdir(parents=True, exist_ok=True)
    test_db.rename(default_path)

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="No Conn Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )

    # Call without conn parameter
    counts = store_records(conversation=conv, messages=[], attachments=[])

    assert counts["conversations"] == 1

    # Verify it was written
    with open_connection(default_path) as conn:
        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
        assert row is not None


def test_upsert_attachment_updates_existing_metadata(test_conn):
    """upsert_attachment() updates existing attachment metadata."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )
    upsert_conversation(test_conn, conv)

    # Setup messages
    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        provider_message_id="ext-msg1",
        role="user",
        text="Test message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg1",
        version=1,
    )
    upsert_message(test_conn, msg1)

    msg2 = MessageRecord(
        message_id="msg2",
        conversation_id="conv1",
        provider_message_id="ext-msg2",
        role="user",
        text="Second message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg2",
        version=1,
    )
    upsert_message(test_conn, msg2)

    # First insert
    att1 = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/png",
        size_bytes=1024,
        path=None,
    )
    upsert_attachment(test_conn, att1)

    # Update with new path and size
    att2 = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg2",  # Different message (new ref)
        mime_type="image/jpeg",  # Updated
        size_bytes=2048,  # Updated
        path="/new/path.jpg",  # Updated
    )
    upsert_attachment(test_conn, att2)

    # Verify updates
    row = test_conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert row["mime_type"] == "image/jpeg"
    assert row["size_bytes"] == 2048
    assert row["path"] == "/new/path.jpg"
    assert row["ref_count"] == 2  # Two refs now


def test_prune_attachment_refs_transactional_rollback(test_conn):
    """_prune_attachment_refs() rolls back on error, maintaining consistent ref_count."""
    # Setup: Create conversation with attachments
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Transaction Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash1",
        version=1,
    )
    upsert_conversation(test_conn, conv)

    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        provider_message_id="ext-msg1",
        role="user",
        text="Test message",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="hash-msg1",
        version=1,
    )
    upsert_message(test_conn, msg1)

    # Create two attachments
    att1 = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/png",
        size_bytes=1024,
    )
    att2 = AttachmentRecord(
        attachment_id="att2",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/jpeg",
        size_bytes=2048,
    )
    upsert_attachment(test_conn, att1)
    upsert_attachment(test_conn, att2)

    # Verify initial state: 2 attachments, ref_count = 1 each
    count_before = test_conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
    assert count_before == 2
    ref_count_before = test_conn.execute(
        "SELECT SUM(ref_count) FROM attachments"
    ).fetchone()[0]
    assert ref_count_before == 2

    # Save snapshot to verify rollback
    refs_before = test_conn.execute(
        "SELECT ref_id, attachment_id FROM attachment_refs ORDER BY ref_id"
    ).fetchall()
    refs_before_ids = [(r["ref_id"], r["attachment_id"]) for r in refs_before]

    # Try to prune with an invalid keep_ref_ids that will cause the function
    # to execute but then we'll verify the SAVEPOINT mechanism works
    ref_id1 = _make_ref_id("att1", "conv1", "msg1")

    # The function should execute successfully
    _prune_attachment_refs(test_conn, "conv1", {ref_id1})

    # Verify: only att1 should remain (att2 pruned)
    count_after = test_conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
    assert count_after == 1

    remaining = test_conn.execute("SELECT attachment_id, ref_count FROM attachments").fetchone()
    assert remaining["attachment_id"] == "att1"
    assert remaining["ref_count"] == 1

    # Verify ref_count is consistent with actual refs
    actual_refs = test_conn.execute(
        "SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?", ("att1",)
    ).fetchone()[0]
    assert actual_refs == 1  # Consistent!


def test_concurrent_upsert_same_attachment_ref_count_correct(test_db):
    """Test for concurrent attachment ref_count race condition.

    Issue: store.py:258-283 has a read-modify-write race in upsert_attachment.
    This test SHOULD FAIL until the race condition is fixed.
    The fix requires atomic increment (e.g., UPDATE ... SET ref_count = ref_count + 1).

    Concurrent upserts of same attachment should maintain correct ref_count.
    """
    SHARED_ATTACHMENT_ID = "shared-attachment-race-test"

    def create_conversation(i: int):
        conv = ConversationRecord(
            conversation_id=f"race-conv-{i}",
            provider_name="test",
            provider_conversation_id=f"race-{i}",
            title=f"Race Test {i}",
            created_at=None,
            updated_at=None,
            content_hash=f"hash-{i}",
            version=1,
        )
        msg = MessageRecord(
            message_id=f"race-msg-{i}",
            conversation_id=f"race-conv-{i}",
            role="user",
            text="test",
            timestamp=None,
            provider_meta=None,
            content_hash=f"msg-hash-{i}",
            version=1,
        )
        # Each conversation references the SAME attachment_id
        attachment = AttachmentRecord(
            attachment_id=SHARED_ATTACHMENT_ID,
            conversation_id=f"race-conv-{i}",
            message_id=f"race-msg-{i}",
            mime_type="text/plain",
            size_bytes=100,
            path="/fake/path.txt",
            provider_meta=None,
        )
        with open_connection(test_db) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[attachment], conn=conn)

    # Run concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(create_conversation, range(10)))

    # Verify ref_count matches actual refs with strict assertions
    with open_connection(test_db) as conn:
        cursor = conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", (SHARED_ATTACHMENT_ID,))
        row = cursor.fetchone()
        assert row is not None, "Attachment should exist"
        stored_ref_count = row[0]

        cursor = conn.execute("SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?", (SHARED_ATTACHMENT_ID,))
        actual_refs = cursor.fetchone()[0]

        # Strict assertion: ref_count must equal number of concurrent insertions
        assert stored_ref_count == 10, f"Race condition! ref_count is {stored_ref_count}, expected 10"
        assert actual_refs == 10, f"Missing refs! Found {actual_refs}, expected 10"
        assert stored_ref_count == actual_refs, f"Mismatch: ref_count={stored_ref_count}, actual_refs={actual_refs}"


class TestAttachmentRecordValidation:
    """Tests for AttachmentRecord field validation."""

    def test_size_bytes_rejects_negative(self):
        """size_bytes cannot be negative."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AttachmentRecord(
                attachment_id="test",
                conversation_id="conv1",
                message_id="msg1",
                mime_type="text/plain",
                size_bytes=-100,
                path="/fake/path",
                provider_meta=None,
            )

    def test_size_bytes_rejects_impossibly_large(self):
        """size_bytes cannot exceed reasonable maximum (1TB)."""
        from pydantic import ValidationError

        from polylogue.storage.store import MAX_ATTACHMENT_SIZE

        with pytest.raises(ValidationError):
            AttachmentRecord(
                attachment_id="test",
                conversation_id="conv1",
                message_id="msg1",
                mime_type="text/plain",
                size_bytes=MAX_ATTACHMENT_SIZE * 10,  # 10TB - clearly invalid
                path="/fake/path",
                provider_meta=None,
            )

    def test_size_bytes_allows_zero(self):
        """size_bytes of 0 should be valid (empty file)."""
        record = AttachmentRecord(
            attachment_id="test",
            conversation_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=0,
            path="/fake/path",
            provider_meta=None,
        )
        assert record.size_bytes == 0

    def test_size_bytes_allows_reasonable_values(self):
        """size_bytes should allow reasonable file sizes."""
        # 100MB - reasonable
        record = AttachmentRecord(
            attachment_id="test",
            conversation_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=100 * 1024 * 1024,
            path="/fake/path",
            provider_meta=None,
        )
        assert record.size_bytes == 100 * 1024 * 1024

    def test_size_bytes_allows_max_size(self):
        """size_bytes should allow exactly the maximum size (1TB)."""
        from polylogue.storage.store import MAX_ATTACHMENT_SIZE

        record = AttachmentRecord(
            attachment_id="test",
            conversation_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=MAX_ATTACHMENT_SIZE,
            path="/fake/path",
            provider_meta=None,
        )
        assert record.size_bytes == MAX_ATTACHMENT_SIZE

    def test_size_bytes_rejects_one_byte_over_max(self):
        """size_bytes cannot exceed maximum by even one byte."""
        from pydantic import ValidationError

        from polylogue.storage.store import MAX_ATTACHMENT_SIZE

        with pytest.raises(ValidationError):
            AttachmentRecord(
                attachment_id="test",
                conversation_id="conv1",
                message_id="msg1",
                mime_type="text/plain",
                size_bytes=MAX_ATTACHMENT_SIZE + 1,
                path="/fake/path",
                provider_meta=None,
            )

    def test_size_bytes_allows_none(self):
        """size_bytes can be None (unknown size)."""
        record = AttachmentRecord(
            attachment_id="test",
            conversation_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=None,
            path="/fake/path",
            provider_meta=None,
        )
        assert record.size_bytes is None


class TestProviderNameValidation:
    """Tests for provider_name field validation."""

    def test_provider_name_rejects_empty(self):
        """provider_name cannot be empty."""
        with pytest.raises(ValidationError):
            ConversationRecord(
                conversation_id="test",
                provider_name="",
                provider_conversation_id="ext1",
                title="Test",
                content_hash="hash123",
                version=1,
            )

    def test_provider_name_rejects_whitespace_only(self):
        """provider_name cannot be only whitespace."""
        with pytest.raises(ValidationError):
            ConversationRecord(
                conversation_id="test",
                provider_name="   ",
                provider_conversation_id="ext1",
                title="Test",
                content_hash="hash123",
                version=1,
            )

    def test_provider_name_rejects_special_characters(self):
        """provider_name should only contain safe characters."""
        invalid_names = [
            "../escape",
            "name\x00null",
            "name\nwith\nnewlines",
            "path/separator",
            "back\\slash",
            "name.with.dots",
            "name with spaces",
            "!invalid",
            "@symbol",
        ]

        for name in invalid_names:
            with pytest.raises(ValidationError, match="invalid"):
                ConversationRecord(
                    conversation_id="test",
                    provider_name=name,
                    provider_conversation_id="ext1",
                    title="Test",
                    content_hash="hash123",
                    version=1,
                )

    def test_provider_name_rejects_starting_with_number(self):
        """provider_name must start with a letter."""
        with pytest.raises(ValidationError, match="Must start with a letter"):
            ConversationRecord(
                conversation_id="test",
                provider_name="123invalid",
                provider_conversation_id="ext1",
                title="Test",
                content_hash="hash123",
                version=1,
            )

    def test_provider_name_rejects_starting_with_hyphen(self):
        """provider_name must start with a letter."""
        with pytest.raises(ValidationError, match="Must start with a letter"):
            ConversationRecord(
                conversation_id="test",
                provider_name="-invalid",
                provider_conversation_id="ext1",
                title="Test",
                content_hash="hash123",
                version=1,
            )

    def test_provider_name_rejects_starting_with_underscore(self):
        """provider_name must start with a letter."""
        with pytest.raises(ValidationError, match="Must start with a letter"):
            ConversationRecord(
                conversation_id="test",
                provider_name="_invalid",
                provider_conversation_id="ext1",
                title="Test",
                content_hash="hash123",
                version=1,
            )

    def test_provider_name_allows_valid_names(self):
        """Valid provider names should be accepted."""
        valid_names = [
            "chatgpt",
            "claude",
            "claude-code",
            "codex",
            "gemini",
            "custom_provider",
            "Provider123",
            "a",  # Single letter
            "A-Z_123",  # Mix of valid characters
            "CustomProvider",
        ]

        for name in valid_names:
            record = ConversationRecord(
                conversation_id="test",
                provider_name=name,
                provider_conversation_id="ext1",
                title="Test",
                content_hash="hash123",
                version=1,
            )
            assert record.provider_name == name

    def test_provider_name_validates_on_store_records(self, test_conn):
        """ConversationRecord rejects invalid provider names at construction time."""
        # Should raise ValidationError during record creation (Pydantic validates at construction)
        with pytest.raises(ValidationError):
            ConversationRecord(
                conversation_id="test",
                provider_name="invalid/name",
                provider_conversation_id="ext1",
                title="Test",
                content_hash="hash123",
                version=1,
            )

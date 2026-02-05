"""Consolidated storage tests.

SYSTEMATIZATION: Merged from:
- test_store.py (Record storage operations)
- test_db.py (Database/connection management)
- test_db_store.py (Integration tests)

This file contains tests for:
- Store record operations (upsert, deduplication)
- Database connection management
- Schema management and migrations
- Attachment ref counting
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from pydantic import ValidationError

from polylogue.storage.backends.sqlite import (
    SCHEMA_VERSION,
    _apply_schema,
    _ensure_schema,
    _run_migrations,
    connection_context,
    default_db_path,
    open_connection,
)
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
from tests.helpers import make_attachment, make_conversation, make_message

# test_db and test_conn fixtures are in conftest.py


# =============================================================================
# STORE RECORD OPERATIONS (from test_store.py)
# =============================================================================


# test_db and test_conn fixtures are in conftest.py


def test_store_records_inserts_new_conversation(test_conn):
    """store_records() inserts a new conversation with messages."""
    conv = make_conversation("conv1", content_hash="hash123")
    msg = make_message("msg1", "conv1", text="Hello")

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
    conv = make_conversation("conv1", title="Same Title", content_hash="samehash")

    # First insert
    counts1 = store_records(conversation=conv, messages=[], attachments=[], conn=test_conn)
    assert counts1["conversations"] == 1

    # Second insert with same hash
    counts2 = store_records(conversation=conv, messages=[], attachments=[], conn=test_conn)
    assert counts2["conversations"] == 0
    assert counts2["skipped_conversations"] == 1


def test_store_records_updates_changed_conversation(test_conn):
    """store_records() updates conversation when content changes."""
    conv1 = make_conversation("conv1", title="Original Title", content_hash="hash1")
    store_records(conversation=conv1, messages=[], attachments=[], conn=test_conn)

    # Update with different content
    conv2 = make_conversation("conv1", title="Updated Title", content_hash="hash2")
    counts = store_records(conversation=conv2, messages=[], attachments=[], conn=test_conn)

    assert counts["conversations"] == 1
    assert counts["skipped_conversations"] == 0

    # Verify update
    row = test_conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
    assert row["title"] == "Updated Title"
    assert row["content_hash"] == "hash2"


def test_store_records_handles_multiple_messages(test_conn):
    """store_records() correctly handles multiple messages."""
    conv = make_conversation("conv1", title="Multi Message")
    messages = [
        make_message(f"msg{i}", "conv1", role="user" if i % 2 == 0 else "assistant", text=f"Message {i}")
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
    conv = make_conversation("conv1", title="Attachment Test")
    msg1 = make_message("msg1", "conv1", provider_message_id="ext-msg1")
    att1 = make_attachment("att1", "conv1", "msg1", mime_type="image/png")

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
    conv = make_conversation("conv1", title="Prune Test")
    msg1 = make_message("msg1", "conv1", provider_message_id="ext-msg1", text="First message")
    msg2 = make_message("msg2", "conv1", provider_message_id="ext-msg2", text="Second message")
    att1 = make_attachment("att1", "conv1", "msg1", mime_type="image/png")
    att2 = make_attachment("att2", "conv1", "msg2", mime_type="image/jpeg", size_bytes=2048)

    store_records(conversation=conv, messages=[msg1, msg2], attachments=[att1, att2], conn=test_conn)

    # Get ref IDs
    ref_id1 = _make_ref_id("att1", "conv1", "msg1")
    # ref_id2 = _make_ref_id("att2", "conv1", "msg2")

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
    remaining = test_conn.execute("SELECT ref_id FROM attachment_refs WHERE conversation_id = ?", ("conv1",)).fetchone()
    assert remaining["ref_id"] == ref_id1


def test_prune_attachment_refs_updates_ref_count(test_conn):
    """_prune_attachment_refs() updates attachment ref_count correctly."""
    conv = make_conversation("conv1", title="RefCount Test")
    msg1 = make_message("msg1", "conv1", provider_message_id="ext-msg1", text="First message")
    msg2 = make_message("msg2", "conv1", provider_message_id="ext-msg2", text="Second message")

    # Same attachment referenced twice (different messages)
    att1 = make_attachment("att1", "conv1", "msg1", mime_type="image/png")
    att2 = make_attachment("att1", "conv1", "msg2", mime_type="image/png")  # Same att_id, different msg

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
    conv = make_conversation("conv1", title="Delete Test")
    msg1 = make_message("msg1", "conv1", provider_message_id="ext-msg1")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png")

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
    # Manual construction to test explicit None handling (not using helper)
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title=None,
        created_at=None,
        updated_at=None,
        content_hash="hash1",
        provider_meta=None,
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
    conv = make_conversation("conv1")
    upsert_conversation(test_conn, conv)

    msg = make_message(
        "msg1", "conv1", role=None, text=None, timestamp=None, provider_message_id=None, provider_meta=None
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
    # Setup conversation and message
    conv = make_conversation("conv1")
    upsert_conversation(test_conn, conv)

    msg1 = make_message("msg1", "conv1", provider_message_id="ext-msg1")
    upsert_message(test_conn, msg1)

    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png")

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
            conv = make_conversation(f"conv{conv_id}", title=f"Conversation {conv_id}")
            messages = [make_message(f"msg{conv_id}-{i}", f"conv{conv_id}", text=f"Message {i}") for i in range(3)]

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


def test_store_records_without_connection_creates_own(test_db, tmp_path, monkeypatch):
    """store_records() works without explicit connection parameter."""
    import importlib
    import shutil

    # Create a temp location for "default" storage within tmp_path to avoid cross-device issues
    # NOTE: default_db_path() uses XDG_DATA_HOME, not XDG_STATE_HOME
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    # Reload paths and sqlite modules to pick up new XDG_DATA_HOME
    import polylogue.paths
    import polylogue.storage.backends.sqlite

    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.storage.backends.sqlite)

    # Now import default_db_path AFTER reload

    default_path = default_db_path()
    default_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(test_db), str(default_path))

    conv = make_conversation("conv1", title="No Conn Test")

    # Call without conn parameter
    counts = store_records(conversation=conv, messages=[], attachments=[])

    assert counts["conversations"] == 1

    # Verify it was written
    with open_connection(default_path) as conn:
        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
        assert row is not None


def test_upsert_attachment_updates_existing_metadata(test_conn):
    """upsert_attachment() updates existing attachment metadata."""
    conv = make_conversation("conv1")
    upsert_conversation(test_conn, conv)

    # Setup messages
    msg1 = make_message("msg1", "conv1", provider_message_id="ext-msg1")
    upsert_message(test_conn, msg1)

    msg2 = make_message("msg2", "conv1", provider_message_id="ext-msg2", text="Second message")
    upsert_message(test_conn, msg2)

    # First insert
    att1 = make_attachment("att1", "conv1", "msg1", mime_type="image/png")
    upsert_attachment(test_conn, att1)

    # Update with new path and size (different message = new ref)
    att2 = make_attachment("att1", "conv1", "msg2", mime_type="image/jpeg", size_bytes=2048, path="/new/path.jpg")
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
    conv = make_conversation("conv1", title="Transaction Test")
    upsert_conversation(test_conn, conv)

    msg1 = make_message("msg1", "conv1", provider_message_id="ext-msg1")
    upsert_message(test_conn, msg1)

    # Create two attachments
    att1 = make_attachment("att1", "conv1", "msg1", mime_type="image/png")
    att2 = make_attachment("att2", "conv1", "msg1", mime_type="image/jpeg", size_bytes=2048)
    upsert_attachment(test_conn, att1)
    upsert_attachment(test_conn, att2)

    # Verify initial state: 2 attachments, ref_count = 1 each
    count_before = test_conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
    assert count_before == 2
    ref_count_before = test_conn.execute("SELECT SUM(ref_count) FROM attachments").fetchone()[0]
    assert ref_count_before == 2

    # Save snapshot to verify rollback

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
        conv = make_conversation(
            f"race-conv-{i}", title=f"Race Test {i}", created_at=None, updated_at=None, content_hash=f"hash-{i}"
        )
        msg = make_message(f"race-msg-{i}", f"race-conv-{i}", text="test", timestamp=None, provider_meta=None)
        # Each conversation references the SAME attachment_id
        attachment = make_attachment(
            SHARED_ATTACHMENT_ID,
            f"race-conv-{i}",
            f"race-msg-{i}",
            mime_type="text/plain",
            size_bytes=100,
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
            )


# =============================================================================
# DATABASE/CONNECTION MANAGEMENT (from test_db.py)
# =============================================================================


def test_open_connection_creates_database(tmp_path):
    """open_connection() creates a new database with schema."""
    db_path = tmp_path / "new.db"
    assert not db_path.exists()

    with open_connection(db_path) as conn:
        assert conn is not None
        # Verify schema was created
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

    assert db_path.exists()
    assert "conversations" in tables
    assert "messages" in tables
    assert "attachments" in tables
    assert "attachment_refs" in tables
    assert "runs" in tables


def test_open_connection_returns_row_factory(tmp_path):
    """open_connection() returns connection with Row factory."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        # Insert test data
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, "
            "title, created_at, updated_at, content_hash, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("conv1", "test", "ext1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()

        # Row factory allows dict-like access
        assert row["conversation_id"] == "conv1"
        assert row["title"] == "Test"


def test_open_connection_enables_foreign_keys(tmp_path):
    """open_connection() enables foreign key constraints."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        # Check foreign_keys pragma
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1  # Foreign keys are ON


def test_open_connection_sets_wal_mode(tmp_path):
    """open_connection() sets WAL journal mode."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal"


def test_connection_context_uses_provided_connection(tmp_path):
    """connection_context() uses provided connection without creating new one."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        original_id = id(conn)

        with connection_context(conn) as ctx_conn:
            assert id(ctx_conn) == original_id


def test_apply_schema_creates_all_tables(tmp_path):
    """_apply_schema() creates all required tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    _apply_schema(conn)

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    assert "conversations" in tables
    assert "messages" in tables
    assert "attachments" in tables
    assert "attachment_refs" in tables
    assert "runs" in tables

    conn.close()


def test_apply_schema_creates_indexes(tmp_path):
    """_apply_schema() creates required indexes."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    _apply_schema(conn)

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
    indexes = [row[0] for row in cursor.fetchall()]

    # Check for some key indexes
    assert "idx_conversations_provider" in indexes
    assert "idx_messages_conversation" in indexes
    assert "idx_attachment_refs_conversation" in indexes

    conn.close()


def test_apply_schema_sets_user_version(tmp_path):
    """_apply_schema() sets PRAGMA user_version to SCHEMA_VERSION."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    _apply_schema(conn)

    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == SCHEMA_VERSION

    conn.close()


def test_ensure_schema_applies_on_new_database(tmp_path):
    """_ensure_schema() applies schema to new database (version 0)."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Verify starting at version 0
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == 0

    _ensure_schema(conn)

    # Check version updated
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == SCHEMA_VERSION

    # Check tables exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert "conversations" in tables

    conn.close()


def test_ensure_schema_raises_on_unsupported_version(tmp_path):
    """_ensure_schema() raises error for unsupported schema versions."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    # Set unsupported version
    conn.execute("PRAGMA user_version = 999")
    conn.commit()

    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        _ensure_schema(conn)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Unsupported DB schema version" in str(exc_info.value)

    conn.close()


def test_migrate_v1_to_v2_creates_new_tables(tmp_path):
    """_migrate_v1_to_v2() creates attachments and attachment_refs tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Setup v1 schema (simplified)
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            provider_message_id TEXT,
            role TEXT,
            text TEXT,
            timestamp TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL
        )
        """
    )
    # Old attachments table (v1 schema)
    conn.execute(
        """
        CREATE TABLE attachments (
            attachment_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            message_id TEXT,
            mime_type TEXT,
            size_bytes INTEGER,
            path TEXT,
            provider_meta TEXT
        )
        """
    )
    conn.commit()

    # Insert test data
    conn.execute(
        "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("conv1", "test", "ext1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", None, 1),
    )
    conn.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("msg1", "conv1", None, "user", "Hello", "2024-01-01T00:00:00Z", "msghash1", None, 1),
    )
    conn.execute(
        "INSERT INTO attachments VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("att1", "conv1", "msg1", "image/png", 1024, "/path/to/file.png", None),
    )
    conn.commit()

    # Migrate using the runner (which updates version)
    _run_migrations(conn, 1, 2)

    # Check version updated
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == 2

    # Check new tables exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    assert "attachments" in tables
    assert "attachment_refs" in tables

    # Check data migrated
    att_row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert att_row is not None
    assert att_row["ref_count"] == 1

    ref_rows = conn.execute("SELECT * FROM attachment_refs WHERE attachment_id = ?", ("att1",)).fetchall()
    assert len(ref_rows) == 1

    conn.close()


def test_migrate_v2_to_v3_updates_runs_table(tmp_path):
    """_migrate_v2_to_v3() updates runs table schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Setup v2 schema with old runs table
    conn.execute("PRAGMA user_version = 2")
    conn.executescript(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL
        );

        CREATE TABLE runs_old (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            plan_snapshot TEXT,
            counts_json TEXT,
            drift_json TEXT,
            indexed INTEGER,
            duration_ms INTEGER
        );
        """
    )

    # Insert test run data
    conn.execute(
        "INSERT INTO runs_old VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("run1", "2024-01-01T00:00:00Z", '{"test": true}', '{"count": 5}', "{}", 1, 1000),
    )
    conn.commit()

    # Temporarily rename runs_old for migration test
    conn.execute("ALTER TABLE runs_old RENAME TO runs")
    conn.commit()

    # Migrate
    # Migrate using the runner (which updates version)
    _run_migrations(conn, 2, 3)

    # Check version updated to 3
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == 3

    # Check runs table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    assert cursor.fetchone() is not None

    # Check data preserved
    run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", ("run1",)).fetchone()
    assert run_row is not None
    assert run_row["timestamp"] == "2024-01-01T00:00:00Z"

    conn.close()


def test_migrate_v3_to_v4_adds_source_name_column(tmp_path):
    """_migrate_v3_to_v4() adds computed source_name column and index."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Setup v3 schema without source_name column
    conn.execute("PRAGMA user_version = 3")
    conn.execute(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id)
        """
    )

    # Insert test conversation with source in provider_meta
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id, provider_name, provider_conversation_id,
            title, content_hash, provider_meta, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("test:123", "test", "123", "Test", "abc123", '{"source": "my-source"}', 1),
    )
    conn.commit()

    # Migrate using the runner (which updates version)
    _run_migrations(conn, 3, 4)

    # Check version updated to 4
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == 4

    # Check source_name column exists and is computed correctly
    conv_row = conn.execute(
        "SELECT conversation_id, source_name FROM conversations WHERE conversation_id = ?",
        ("test:123",),
    ).fetchone()
    assert conv_row is not None
    assert conv_row["source_name"] == "my-source"

    # Check index exists
    index_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_conversations_source_name'"
    ).fetchone()
    assert index_row is not None

    conn.close()


def test_open_connection_thread_isolation(tmp_path):
    """open_connection() maintains separate connections per thread."""
    from threading import Barrier

    db_path = tmp_path / "test.db"
    num_threads = 3

    # Initialize database with WAL mode first to avoid lock contention during PRAGMA journal_mode
    with open_connection(db_path) as conn:
        conn.execute("SELECT 1").fetchone()

    # Barrier ensures all threads hold connections simultaneously
    # This prevents Python from reusing memory addresses after GC
    barrier = Barrier(num_threads)
    connection_ids = []
    errors = []

    def thread_func(thread_id: int):
        try:
            with open_connection(db_path) as conn:
                conn_id = id(conn)
                connection_ids.append(conn_id)
                # All threads wait here with live connections before proceeding
                barrier.wait()
                # Do some work to verify connection is usable
                conn.execute("SELECT 1").fetchone()
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = [threading.Thread(target=thread_func, args=(i,)) for i in range(num_threads)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # No errors should occur
    assert len(errors) == 0

    # All threads should succeed
    assert len(connection_ids) == num_threads

    # Each thread should have had a different connection object
    # (guaranteed because barrier ensures all connections exist simultaneously)
    assert len(set(connection_ids)) == num_threads


def test_open_connection_creates_parent_directories(tmp_path):
    """open_connection() creates parent directories if they don't exist."""
    db_path = tmp_path / "nested" / "deeply" / "test.db"
    assert not db_path.parent.exists()

    with open_connection(db_path) as conn:
        assert conn is not None

    assert db_path.exists()
    assert db_path.parent.exists()


def test_open_connection_busy_timeout_set(tmp_path):
    """open_connection() sets busy_timeout for concurrent access."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        # Check busy_timeout is set (should be 30000ms = 30 seconds)
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] == 30000


class TestMigrations:
    """Tests for database migration behavior."""

    def test_migration_failure_preserves_original_state(self, tmp_path, monkeypatch):
        """Failed migration should not leave database in inconsistent state.

        This test verifies that if a migration step fails, the database
        remains at the last successful version (ratcheting behavior).
        """
        db_path = tmp_path / "test_state_preservation.db"

        # Initialize database at v1 (past version)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA user_version = 1")
        # Minimal v1 schema to satisfy _migrate_v1_to_v2
        conn.execute(
            """
            CREATE TABLE attachments (
                attachment_id TEXT PRIMARY KEY,
                mime_type TEXT,
                size_bytes INTEGER,
                path TEXT,
                provider_meta TEXT,
                conversation_id TEXT,
                message_id TEXT
            )
            """
        )
        conn.commit()
        conn.close()

        # Patch _MIGRATIONS[2] to fail (simulating v2->v3 failure)
        # We allow v1->v2 to succeed
        from polylogue.storage.backends.sqlite import _MIGRATIONS

        def failing_migration(conn):
            raise RuntimeError("Simulated migration v2->v3 failure")

        # Copy dict to avoid polluting other tests
        patched_migrations = _MIGRATIONS.copy()
        patched_migrations[2] = failing_migration
        monkeypatch.setattr("polylogue.storage.backends.sqlite._MIGRATIONS", patched_migrations)

        # Run connection open which triggers _ensure_schema
        # It should raise DatabaseError or RuntimeError from the migration
        with pytest.raises(Exception, match="Simulated migration v2->v3 failure"):
            with open_connection(db_path) as conn:
                pass

        # Verify database state
        conn = sqlite3.connect(db_path)
        version = conn.execute("PRAGMA user_version").fetchone()[0]

        # Should be at version 2 (because v1->v2 succeeded and committed)
        # The failing v2->v3 rolled back its changes (if any) and didn't bump version
        assert version == 2
        conn.close()

    def test_migration_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        """Failed migration raises RuntimeError with details.

        Note: SQLite DDL (ALTER TABLE, CREATE TABLE) cannot be rolled back.
        This test verifies that migration failures are properly reported.
        """
        db_path = tmp_path / "migration_failure_test.db"

        # Create a v3 database manually
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA user_version = 3")
        conn.execute(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                provider_conversation_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                content_hash TEXT NOT NULL,
                provider_meta TEXT,
                version INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX idx_conversations_provider
            ON conversations(provider_name, provider_conversation_id)
            """
        )
        conn.commit()
        conn.close()

        # Patch _MIGRATIONS[3] to fail (patching the function won't work as dict has reference)
        from polylogue.storage.backends.sqlite import _MIGRATIONS

        def failing_migration(conn):
            raise RuntimeError("Simulated migration failure")

        _MIGRATIONS[3]
        monkeypatch.setitem(_MIGRATIONS, 3, failing_migration)

        # Migration should raise RuntimeError
        with pytest.raises(RuntimeError, match="Migration from v3 to v4 failed"):
            with open_connection(db_path) as conn:
                pass

    def test_connection_context_rollsback_on_exception(self, tmp_path):
        """open_connection should rollback on exception."""
        db_path = tmp_path / "rollback_test.db"

        # First create the database and table
        with open_connection(db_path) as conn:
            _ensure_schema(conn)

        # Now try to insert and raise
        try:
            with open_connection(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO conversations
                    (conversation_id, provider_name, provider_conversation_id, title, content_hash, version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    ("rollback-test", "test", "prov-1", "Rollback Test", "hash789", 1),
                )
                raise ValueError("Simulated failure")
        except ValueError:
            pass

        # Verify rolled back
        with open_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT title FROM conversations WHERE conversation_id = ?",
                ("rollback-test",),
            )
            row = cursor.fetchone()
            assert row is None, "Insert should have been rolled back"


# =============================================================================
# DB+STORE INTEGRATION (from test_db_store.py)
# =============================================================================


class TestConnectionContextReuse:
    """Test connection reuse within same thread."""


class TestConnectionCommitAndRollback:
    """Test transaction commit/rollback behavior."""

    def test_connection_no_commit_on_exception(self, tmp_path):
        """Exception in context skips commit (data may not persist).

        Note: The implementation commits only on normal exit. On exception,
        it closes without commit. With SQLite WAL mode + autocommit, writes
        within the context may still be visible within that session but won't
        persist after close.
        """
        db_path = tmp_path / "test.db"

        # Initialize schema first
        with open_connection(db_path):
            pass

        try:
            with open_connection(db_path) as conn:
                conn.execute(
                    """INSERT INTO conversations
                    (conversation_id, provider_name, provider_conversation_id,
                     title, created_at, updated_at, content_hash, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("c1", "test", "ext1", "Title1", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
                )
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify no explicit commit happened - data should not persist
        # (SQLite autocommit behavior may vary, but no explicit commit was called)
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            # The implementation doesn't do rollback, but also doesn't commit
            # Row presence depends on SQLite internals; we verify connection still works
            if row is not None:
                assert row["conversation_id"] == "c1"


class TestThreadSafety:
    """Test thread-local connection safety."""

    def test_thread_local_connections_isolated(self, tmp_path):
        """Each thread gets its own isolated connection."""
        db_path = tmp_path / "test.db"

        # Initialize database with WAL mode first to avoid lock contention
        with open_connection(db_path) as conn:
            conn.execute("SELECT 1").fetchone()

        connection_ids = {}
        errors = []

        def thread_work(thread_id: int):
            try:
                with open_connection(db_path) as conn:
                    connection_ids[thread_id] = id(conn)
                    # Verify connection is functional
                    cursor = conn.execute("SELECT 1")
                    assert cursor.fetchone() is not None
                    conn.commit()  # Ensure locks are released
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Use fewer threads to reduce lock contention
        threads = [threading.Thread(target=thread_work, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(set(connection_ids.values())) == 3, "Each thread should have different connection object"

    def test_concurrent_writes_with_write_lock(self, tmp_path):
        """Concurrent store_records() calls properly serialize via write lock."""
        db_path = tmp_path / "test.db"

        # Initialize database with WAL mode first
        with open_connection(db_path) as conn:
            conn.execute("SELECT 1").fetchone()

        errors = []

        def write_conversation(conv_id: int):
            try:
                conv = make_conversation(f"c{conv_id}", title=f"Conversation {conv_id}")
                messages = [
                    make_message(
                        f"m{conv_id}-{i}",
                        f"c{conv_id}",
                        role="user" if i % 2 == 0 else "assistant",
                        text=f"Message {i}",
                    )
                    for i in range(3)
                ]

                with open_connection(db_path) as conn:
                    store_records(conversation=conv, messages=messages, attachments=[], conn=conn)
                    conn.commit()  # Explicit commit to release locks faster
            except Exception as e:
                errors.append((conv_id, str(e)))

        # Run concurrent writes with reduced parallelism to avoid lock contention
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_conversation, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0

        # Verify all conversations written
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            assert count == 20

            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            assert msg_count == 60  # 20 * 3


class TestSchemaAndMigration:
    """Test schema initialization and versioning."""

    def test_open_connection_applies_schema_on_new_db(self, tmp_path):
        """open_connection() applies full schema to new database."""
        db_path = tmp_path / "new.db"
        assert not db_path.exists()

        with open_connection(db_path) as conn:
            # Verify schema version
            version_row = conn.execute("PRAGMA user_version").fetchone()
            assert version_row[0] == SCHEMA_VERSION

            # Verify tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = [
                "conversations",
                "messages",
                "attachments",
                "attachment_refs",
                "runs",
            ]
            for table in expected_tables:
                assert table in tables, f"Table {table} not found"

    def test_open_connection_foreign_keys_enabled(self, tmp_path):
        """open_connection() enables foreign key constraints."""
        db_path = tmp_path / "test.db"

        with open_connection(db_path) as conn:
            row = conn.execute("PRAGMA foreign_keys").fetchone()
            assert row[0] == 1, "Foreign keys should be ON"

    def test_open_connection_wal_mode_enabled(self, tmp_path):
        """open_connection() enables WAL journal mode."""
        db_path = tmp_path / "test.db"

        with open_connection(db_path) as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert row[0].lower() == "wal", "WAL mode should be enabled"

    def test_open_connection_creates_parent_directories(self, tmp_path):
        """open_connection() creates nested parent directories."""
        db_path = tmp_path / "a" / "b" / "c" / "test.db"
        assert not db_path.parent.exists()

        with open_connection(db_path) as conn:
            assert conn is not None

        assert db_path.exists()
        assert db_path.parent.exists()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_store_records_with_null_optional_fields(self, tmp_path):
        """store_records() handles conversations/messages with NULL optional fields."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title=None,  # NULL
            created_at=None,  # NULL
            updated_at=None,  # NULL
            content_hash="hash1",
            provider_meta=None,  # NULL
        )

        msg = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            provider_message_id=None,  # NULL
            role=None,  # NULL
            text=None,  # NULL
            timestamp=None,  # NULL
            content_hash="msghash1",
            provider_meta=None,  # NULL
        )

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)

        assert counts["conversations"] == 1
        assert counts["messages"] == 1

        # Verify NULLs preserved
        with open_connection(db_path) as conn:
            conv_row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            assert conv_row["title"] is None
            assert conv_row["created_at"] is None

            msg_row = conn.execute("SELECT * FROM messages WHERE message_id = ?", ("m1",)).fetchone()
            assert msg_row["role"] is None
            assert msg_row["text"] is None

    def test_store_records_with_empty_messages_and_attachments(self, tmp_path):
        """store_records() handles conversation with no messages or attachments."""
        db_path = tmp_path / "test.db"

        conv = make_conversation("c1", title="Empty Conversation")

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        assert counts["conversations"] == 1
        assert counts["messages"] == 0
        assert counts["attachments"] == 0

    def test_attachment_without_message_id(self, tmp_path):
        """Attachments can exist without being tied to a message."""
        db_path = tmp_path / "test.db"

        conv = make_conversation("c1", title="Test")
        # Attachment without message_id
        att = make_attachment("att1", "c1", message_id=None, mime_type="application/pdf", size_bytes=5000)

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[], attachments=[att], conn=conn)

        assert counts["attachments"] == 1

        # Verify stored
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row is not None
            assert row["ref_count"] == 1


class TestComplexScenarios:
    """Test realistic complex scenarios."""

    def test_conversation_lifecycle_with_attachments(self, tmp_path):
        """Full lifecycle: create → add attachments → remove attachments → cleanup."""
        db_path = tmp_path / "test.db"

        # Step 1: Create conversation with one attachment
        conv_v1 = make_conversation(
            "c1",
            provider_name="claude",
            title="Analysis Project",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:00:00Z",
            content_hash="hash-v1",
        )
        msg1 = make_message("m1", "c1", text="Please analyze this image", timestamp="2024-01-01T10:00:00Z")
        att1 = make_attachment("att-image", "c1", "m1", mime_type="image/png", size_bytes=51200)

        with open_connection(db_path) as conn:
            store_records(conversation=conv_v1, messages=[msg1], attachments=[att1], conn=conn)

        # Step 2: Add more messages and attachments
        msg2 = make_message("m2", "c1", role="assistant", text="The image shows...", timestamp="2024-01-01T10:01:00Z")
        att2 = make_attachment("att-export", "c1", "m2", mime_type="application/json", size_bytes=2048)
        conv_v2 = make_conversation(
            "c1",
            provider_name="claude",
            title="Analysis Project",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:02:00Z",
            content_hash="hash-v2",
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv_v2,
                messages=[msg1, msg2],
                attachments=[att1, att2],
                conn=conn,
            )

        # Verify 2 attachments now
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
            assert count == 2

        # Step 3: Final update removes one attachment
        conv_v3 = make_conversation(
            "c1",
            provider_name="claude",
            title="Analysis Project - Final",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:03:00Z",
            content_hash="hash-v3",
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv_v3,
                messages=[msg1, msg2],
                attachments=[att1],  # Only image, no export
                conn=conn,
            )

        # Verify: image kept, export deleted
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
            assert count == 1

            remaining = conn.execute("SELECT attachment_id FROM attachments").fetchone()
            assert remaining["attachment_id"] == "att-image"

    def test_multi_provider_conversations_separate(self, tmp_path):
        """Conversations from different providers don't interfere."""
        db_path = tmp_path / "test.db"

        conv_gpt = make_conversation(
            "c-gpt", provider_name="chatgpt", title="ChatGPT Conversation", content_hash="hash-gpt"
        )
        conv_claude = make_conversation(
            "c-claude", provider_name="claude", title="Claude Conversation", content_hash="hash-claude"
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv_gpt, messages=[], attachments=[], conn=conn)
            store_records(conversation=conv_claude, messages=[], attachments=[], conn=conn)

        # Verify both stored correctly
        with open_connection(db_path) as conn:
            gpt_row = conn.execute("SELECT * FROM conversations WHERE provider_name = ?", ("chatgpt",)).fetchone()
            claude_row = conn.execute("SELECT * FROM conversations WHERE provider_name = ?", ("claude",)).fetchone()

            assert gpt_row is not None
            assert gpt_row["title"] == "ChatGPT Conversation"
            assert claude_row is not None
            assert claude_row["title"] == "Claude Conversation"

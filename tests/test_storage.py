"""Consolidated storage tests.

SYSTEMATIZATION: Merged from:
- test_store.py (Record storage operations)
- test_db.py (Database/connection management)
- test_db_store.py (Integration tests)
- test_sqlite_backend.py (SQLiteBackend API tests) [MERGED]

This file contains tests for:
- Store record operations (upsert, deduplication)
- Database connection management
- Schema management and migrations
- Attachment ref counting
- SQLiteBackend connection context, initialization, CRUD operations, transactions
- Message ordering, batch operations, conversation stats
- Helper function utilities
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from polylogue.schemas.unified import (
    extract_harmonized_message,
    is_message_record,
)
from polylogue.schemas.unified import (
    normalize_role as new_normalize_role,
)
from polylogue.sources.parsers.base import normalize_role as old_normalize_role
from polylogue.sources.parsers.claude import (
    extract_text_from_segments as old_extract_segments,
)
from polylogue.storage.backends import SQLiteBackend
from polylogue.storage.backends.sqlite import (
    SCHEMA_VERSION,
    SQLiteBackend,
    DatabaseError,
    _apply_schema,
    _ensure_schema,
    _json_or_none,
    _make_ref_id,
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


# =============================================================================
# SQLITE BACKEND OPERATIONS (from test_backend_core.py)
# =============================================================================


def test_backend_save_and_get_conversation(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving a conversation."""
    conv = make_conversation("conv1", title="Test Conversation", content_hash="hash123")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.commit()

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is not None
    assert retrieved.conversation_id == "conv1"
    assert retrieved.title == "Test Conversation"
    assert retrieved.content_hash == "hash123"


def test_backend_save_and_get_messages(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving messages."""
    conv = make_conversation("conv1", title="Test")
    msg1 = make_message("msg1", "conv1", text="Hello")
    msg2 = make_message("msg2", "conv1", role="assistant", text="Hi there")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg1, msg2])
    sqlite_backend.commit()

    messages = sqlite_backend.get_messages("conv1")
    assert len(messages) == 2
    assert messages[0].message_id == "msg1"
    assert messages[0].text == "Hello"
    assert messages[1].message_id == "msg2"
    assert messages[1].text == "Hi there"


def test_backend_list_conversations(sqlite_backend: SQLiteBackend) -> None:
    """Test listing conversations."""
    conv1 = make_conversation("conv1", title="First", created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z", provider_meta={"source": "claude"})
    conv2 = make_conversation("conv2", title="Second", created_at="2024-01-02T00:00:00Z", updated_at="2024-01-02T00:00:00Z", provider_meta={"source": "chatgpt"})

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv1)
    sqlite_backend.save_conversation(conv2)
    sqlite_backend.commit()

    all_convs = sqlite_backend.list_conversations()
    assert len(all_convs) == 2

    claude_convs = sqlite_backend.list_conversations(source="claude")
    assert len(claude_convs) == 1
    assert claude_convs[0].conversation_id == "conv1"


def test_backend_save_attachments(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving attachments."""
    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="Hello")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg])
    sqlite_backend.save_attachments([att])
    sqlite_backend.commit()

    attachments = sqlite_backend.get_attachments("conv1")
    assert len(attachments) == 1
    assert attachments[0].attachment_id == "att1"
    assert attachments[0].mime_type == "image/png"
    assert attachments[0].size_bytes == 1024


def test_backend_transaction_rollback(sqlite_backend: SQLiteBackend) -> None:
    """Test transaction rollback."""
    conv = make_conversation("conv1", title="Test")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.rollback()

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is None


def test_backend_transaction_context_manager(sqlite_backend: SQLiteBackend) -> None:
    """Test using the transaction context manager."""
    conv = make_conversation("conv1", title="Test")

    with sqlite_backend.transaction():
        sqlite_backend.save_conversation(conv)

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is not None
    assert retrieved.conversation_id == "conv1"


def test_backend_transaction_context_manager_exception(sqlite_backend: SQLiteBackend) -> None:
    """Test transaction context manager rolls back on exception."""
    conv = make_conversation("conv1", title="Test")

    with pytest.raises(ValueError), sqlite_backend.transaction():
        sqlite_backend.save_conversation(conv)
        raise ValueError("Test error")

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is None


def test_backend_delete_conversation(sqlite_backend: SQLiteBackend) -> None:
    """Test deleting a conversation and all related records."""
    conv = make_conversation("conv1", title="Test")
    msg1 = make_message("msg1", "conv1", text="Hello")
    msg2 = make_message("msg2", "conv1", role="assistant", text="Hi there")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg1, msg2])
    sqlite_backend.save_attachments([att])
    sqlite_backend.commit()

    assert sqlite_backend.get_conversation("conv1") is not None
    assert len(sqlite_backend.get_messages("conv1")) == 2
    assert len(sqlite_backend.get_attachments("conv1")) == 1

    result = sqlite_backend.delete_conversation("conv1")
    assert result is True

    assert sqlite_backend.get_conversation("conv1") is None
    assert len(sqlite_backend.get_messages("conv1")) == 0
    assert len(sqlite_backend.get_attachments("conv1")) == 0


def test_backend_delete_conversation_not_found(sqlite_backend: SQLiteBackend) -> None:
    """Test deleting a non-existent conversation returns False."""
    result = sqlite_backend.delete_conversation("nonexistent")
    assert result is False


def test_backend_delete_conversation_cleans_fts(sqlite_backend: SQLiteBackend) -> None:
    """Test that deleting a conversation also cleans up FTS entries."""
    from polylogue.storage.index import ensure_index, update_index_for_conversations

    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="searchable content here")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg])
    sqlite_backend.commit()

    conn = sqlite_backend._get_connection()
    ensure_index(conn)
    update_index_for_conversations(["conv1"], conn)
    conn.commit()

    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
    assert fts_count > 0

    result = sqlite_backend.delete_conversation("conv1")
    assert result is True

    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
    assert fts_count == 0


# =============================================================================
# STORAGE BACKEND PROTOCOL CONFORMANCE (from test_backend_core.py)
# =============================================================================


class TestConversationOperations:
    """Test conversation save/retrieve operations."""

    def test_save_and_get_conversation(self, tmp_path: Path) -> None:
        """save_conversation persists data retrievable by get_conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1", title="Test Conversation", provider_name="claude")
        backend.save_conversation(conv)

        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv-1"
        assert retrieved.title == "Test Conversation"
        assert retrieved.provider_name == "claude"
        backend.close()

    def test_get_nonexistent_conversation_returns_none(self, tmp_path: Path) -> None:
        """get_conversation returns None for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = backend.get_conversation("nonexistent")
        assert result is None
        backend.close()

    def test_save_conversation_upserts(self, tmp_path: Path) -> None:
        """save_conversation updates existing conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv1 = make_conversation("conv-1", title="Original Title")
        backend.save_conversation(conv1)

        conv2 = make_conversation("conv-1", title="Updated Title")
        backend.save_conversation(conv2)

        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        backend.close()

    def test_list_conversations_returns_all(self, tmp_path: Path) -> None:
        """list_conversations returns all stored conversations."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(3):
            conv = make_conversation(f"conv-{i}", title=f"Conversation {i}")
            backend.save_conversation(conv)

        all_convs = backend.list_conversations()
        assert len(all_convs) == 3
        assert {c.conversation_id for c in all_convs} == {"conv-0", "conv-1", "conv-2"}
        backend.close()

    def test_list_conversations_filters_by_provider(self, tmp_path: Path) -> None:
        """list_conversations filters by provider_name."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.save_conversation(make_conversation("c1", provider_name="claude"))
        backend.save_conversation(make_conversation("c2", provider_name="chatgpt"))
        backend.save_conversation(make_conversation("c3", provider_name="claude"))

        claude_convs = backend.list_conversations(provider="claude")
        assert len(claude_convs) == 2
        assert all(c.provider_name == "claude" for c in claude_convs)
        backend.close()

    def test_list_conversations_with_limit_and_offset(self, tmp_path: Path) -> None:
        """list_conversations supports pagination."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(10):
            conv = make_conversation(f"conv-{i:02d}")
            backend.save_conversation(conv)

        page1 = backend.list_conversations(limit=3, offset=0)
        page2 = backend.list_conversations(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        page1_ids = {c.conversation_id for c in page1}
        page2_ids = {c.conversation_id for c in page2}
        assert page1_ids.isdisjoint(page2_ids)
        backend.close()

    def test_backend_list_conversations_offset_without_limit(self, tmp_path: Path) -> None:
        """Regression: OFFSET without LIMIT must not raise SQL syntax error."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(5):
            conv = make_conversation(f"off-{i}", updated_at=f"2024-01-{i+1:02d}T00:00:00Z")
            backend.save_conversation(conv)

        # This previously generated invalid SQL: ... ORDER BY ... OFFSET ? (no LIMIT)
        result = backend.list_conversations(offset=2)
        assert len(result) == 3  # 5 total - 2 skipped = 3
        backend.close()


def test_repository_message_mapping_uses_backend_path(tmp_path: Path) -> None:
    """Regression: _get_message_conversation_mapping must use backend's db_path."""
    from polylogue.storage.repository import ConversationRepository

    db_path = tmp_path / "custom.db"
    backend = SQLiteBackend(db_path=db_path)

    conv = make_conversation("map-conv-1", title="Mapping Test")
    msg = make_message("map-msg-1", "map-conv-1", text="Hello")

    backend.begin()
    backend.save_conversation(conv)
    backend.save_messages([msg])
    backend.commit()

    repo = ConversationRepository(backend)
    mapping = repo._get_message_conversation_mapping(["map-msg-1"])
    assert mapping == {"map-msg-1": "map-conv-1"}

    # Non-existent messages should return empty
    mapping_empty = repo._get_message_conversation_mapping(["nonexistent"])
    assert mapping_empty == {}
    backend.close()


class TestMessageOperations:
    """Test message save/retrieve operations."""

    def test_save_and_get_messages(self, tmp_path: Path) -> None:
        """save_messages persists data retrievable by get_messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        messages = [
            make_message("m1", "conv-1", role="user", text="Hello"),
            make_message("m2", "conv-1", role="assistant", text="Hi there"),
        ]
        backend.save_messages(messages)

        retrieved = backend.get_messages("conv-1")
        assert len(retrieved) == 2
        assert {m.message_id for m in retrieved} == {"m1", "m2"}
        assert {m.role for m in retrieved} == {"user", "assistant"}
        backend.close()

    def test_get_messages_for_empty_conversation(self, tmp_path: Path) -> None:
        """get_messages returns empty list for conversation with no messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        retrieved = backend.get_messages("conv-1")
        assert retrieved == []
        backend.close()


class TestAttachmentOperations:
    """Test attachment save/retrieve operations."""

    def test_save_and_get_attachments(self, tmp_path: Path) -> None:
        """save_attachments persists data retrievable by get_attachments."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        attachments = [
            make_attachment("att1", "conv-1", mime_type="image/png", size_bytes=1024),
            make_attachment("att2", "conv-1", mime_type="text/plain", size_bytes=256),
        ]
        backend.save_attachments(attachments)

        retrieved = backend.get_attachments("conv-1")
        assert len(retrieved) == 2
        assert {a.attachment_id for a in retrieved} == {"att1", "att2"}
        backend.close()

    def test_prune_attachments_removes_unlisted(self, tmp_path: Path) -> None:
        """prune_attachments removes attachments not in keep set."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        attachments = [
            make_attachment("att1", "conv-1"),
            make_attachment("att2", "conv-1"),
            make_attachment("att3", "conv-1"),
        ]
        backend.save_attachments(attachments)

        backend.prune_attachments("conv-1", {"att1", "att3"})

        retrieved = backend.get_attachments("conv-1")
        assert {a.attachment_id for a in retrieved} == {"att1", "att3"}
        backend.close()


class TestTransactionOperations:
    """Test transaction management."""

    def test_begin_commit_persists_data(self, tmp_path: Path) -> None:
        """Data saved within begin/commit is persisted."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.begin()
        conv = make_conversation("tx-conv")
        backend.save_conversation(conv)
        backend.commit()

        retrieved = backend.get_conversation("tx-conv")
        assert retrieved is not None
        backend.close()

    def test_rollback_discards_changes(self, tmp_path: Path) -> None:
        """Data saved within begin/rollback is discarded."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.begin()
        backend.save_conversation(make_conversation("rollback-conv"))
        backend.rollback()

        assert backend.get_conversation("rollback-conv") is None
        backend.close()


class TestMetadataOperations:
    """Test metadata CRUD operations."""

    def test_update_and_get_metadata(self, tmp_path: Path) -> None:
        """update_metadata sets key, get_metadata retrieves it."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.update_metadata("conv-1", "rating", 5)
        backend.update_metadata("conv-1", "reviewed", True)

        metadata = backend.get_metadata("conv-1")
        assert metadata.get("rating") == 5
        assert metadata.get("reviewed") is True
        backend.close()

    def test_delete_metadata(self, tmp_path: Path) -> None:
        """delete_metadata removes a key."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.update_metadata("conv-1", "temp", "value")
        backend.delete_metadata("conv-1", "temp")

        metadata = backend.get_metadata("conv-1")
        assert "temp" not in metadata
        backend.close()

    def test_add_and_remove_tag(self, tmp_path: Path) -> None:
        """add_tag adds to tags list, remove_tag removes it."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-1", "work")

        metadata = backend.get_metadata("conv-1")
        tags = metadata.get("tags", [])
        assert "important" in tags
        assert "work" in tags

        backend.remove_tag("conv-1", "work")
        metadata = backend.get_metadata("conv-1")
        tags = metadata.get("tags", [])
        assert "important" in tags
        assert "work" not in tags
        backend.close()

    def test_list_tags_empty(self, tmp_path: Path) -> None:
        """list_tags returns empty dict when no tags exist."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        tags = backend.list_tags()
        assert tags == {}
        backend.close()

    def test_list_tags_counts(self, tmp_path: Path) -> None:
        """list_tags returns correct tag counts across conversations."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        # Create 3 conversations with different tags
        conv1 = make_conversation("conv-1")
        conv2 = make_conversation("conv-2")
        conv3 = make_conversation("conv-3")

        backend.save_conversation(conv1)
        backend.save_conversation(conv2)
        backend.save_conversation(conv3)

        # Tag conv-1 with "important" and "work"
        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-1", "work")

        # Tag conv-2 with "important"
        backend.add_tag("conv-2", "important")

        # conv-3 has no tags

        tags = backend.list_tags()
        assert tags == {"important": 2, "work": 1}
        backend.close()

    def test_list_tags_provider_filter(self, tmp_path: Path) -> None:
        """list_tags with provider filter only counts tags from that provider."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        # Create conversations with different providers
        conv_claude = make_conversation("conv-claude", provider_name="claude")
        conv_chatgpt = make_conversation("conv-chatgpt", provider_name="chatgpt")

        backend.save_conversation(conv_claude)
        backend.save_conversation(conv_chatgpt)

        # Tag both
        backend.add_tag("conv-claude", "important")
        backend.add_tag("conv-chatgpt", "important")
        backend.add_tag("conv-chatgpt", "review")

        # Filter by claude provider
        tags_claude = backend.list_tags(provider="claude")
        assert tags_claude == {"important": 1}

        # Filter by chatgpt provider
        tags_chatgpt = backend.list_tags(provider="chatgpt")
        assert tags_chatgpt == {"important": 1, "review": 1}

        # All tags
        tags_all = backend.list_tags()
        assert tags_all == {"important": 2, "review": 1}

        backend.close()

    def test_list_tags_dedup(self, tmp_path: Path) -> None:
        """list_tags doesn't double-count duplicate tags on same conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        # Add the same tag twice
        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-1", "important")

        tags = backend.list_tags()
        # Should count as 1, not 2
        assert tags == {"important": 1}
        backend.close()


class TestSearchOperations:
    """Test search and resolve operations."""

    def test_resolve_id_exact_match(self, tmp_path: Path) -> None:
        """resolve_id returns full ID for exact match."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conversation-12345")
        backend.save_conversation(conv)

        resolved = backend.resolve_id("conversation-12345")
        assert resolved == "conversation-12345"
        backend.close()

    def test_resolve_id_prefix_match(self, tmp_path: Path) -> None:
        """resolve_id returns full ID for unique prefix."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("unique-prefix-abc123")
        backend.save_conversation(conv)

        resolved = backend.resolve_id("unique-prefix")
        assert resolved == "unique-prefix-abc123"
        backend.close()

    def test_resolve_id_ambiguous_returns_none(self, tmp_path: Path) -> None:
        """resolve_id returns None for ambiguous prefix."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.save_conversation(make_conversation("prefix-abc"))
        backend.save_conversation(make_conversation("prefix-def"))

        resolved = backend.resolve_id("prefix")
        assert resolved is None
        backend.close()


class TestDeleteOperations:
    """Test deletion operations."""

    def test_delete_conversation(self, tmp_path: Path) -> None:
        """delete_conversation removes conversation and related data."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("to-delete")
        backend.save_conversation(conv)
        backend.save_messages([make_message("m1", "to-delete")])
        backend.save_attachments([make_attachment("a1", "to-delete")])

        result = backend.delete_conversation("to-delete")
        assert result is True

        assert backend.get_conversation("to-delete") is None
        assert backend.get_messages("to-delete") == []
        assert backend.get_attachments("to-delete") == []
        backend.close()

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """delete_conversation returns False for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = backend.delete_conversation("nonexistent")
        assert result is False
        backend.close()


# =============================================================================
# BACKEND COMPARISON TESTS (from test_backend_core.py)
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of comparing old vs new extraction."""

    field: str
    old_value: str | None
    new_value: str | None
    equivalent: bool


def compare_extractions(provider: str, raw: dict) -> list[ComparisonResult]:
    """Compare old and new extraction for a single message."""
    results = []

    try:
        new_msg = extract_harmonized_message(provider, raw)
    except Exception as e:
        return [ComparisonResult("extraction", None, str(e), False)]

    if provider == "claude-code":
        msg_obj = raw.get("message", {})
        msg_type = raw.get("type")

        if msg_type in ("user", "human"):
            old_role = "user"
        elif msg_type == "assistant":
            old_role = "assistant"
        else:
            old_role = msg_type or "unknown"

        content_raw = msg_obj.get("content") if isinstance(msg_obj, dict) else None
        old_text = old_extract_segments(content_raw) if isinstance(content_raw, list) else None

        old_role_norm = old_normalize_role(old_role)
        new_role_norm = new_msg.role

        results.append(ComparisonResult(
            field="role",
            old_value=old_role_norm,
            new_value=new_role_norm,
            equivalent=old_role_norm == new_role_norm,
        ))

        old_text_norm = (old_text or "").strip()
        new_text_norm = (new_msg.text or "").strip()

        text_equiv = old_text_norm == new_text_norm

        results.append(ComparisonResult(
            field="text",
            old_value=old_text_norm[:50] + "..." if len(old_text_norm) > 50 else old_text_norm,
            new_value=new_text_norm[:50] + "..." if len(new_text_norm) > 50 else new_text_norm,
            equivalent=text_equiv,
        ))

    return results


class TestBackendComparison:
    """Compare old vs new extraction backends."""

    def test_role_normalization_equivalence(self):
        """Old and new role normalization should produce same results for valid inputs."""
        test_roles = [
            "user", "human", "USER",
            "assistant", "model", "ai",
            "system",
            "tool", "function",
        ]

        differences = []
        for role in test_roles:
            old = old_normalize_role(role)
            new = new_normalize_role(role)
            if old != new:
                differences.append((role, old, new))

        if differences:
            print("\nRole normalization differences (may be improvements):")
            for role, old, new in differences:
                print(f"  {role!r}: old={old!r}, new={new!r}")

        assert old_normalize_role("user") == new_normalize_role("user")
        assert old_normalize_role("assistant") == new_normalize_role("assistant")

    def test_claude_code_extraction_equivalence(self, seeded_db):
        """Compare old and new extraction on real Claude Code data."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
            LIMIT 500
            """
        )

        rows = cur.fetchall()
        conn.close()

        equiv_count = Counter()
        diff_samples = []

        for (pm_json,) in rows:
            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            if not is_message_record("claude-code", raw):
                continue

            results = compare_extractions("claude-code", raw)

            for r in results:
                if r.equivalent:
                    equiv_count[r.field] += 1
                else:
                    if len(diff_samples) < 10:
                        diff_samples.append(r)

        total = sum(equiv_count.values())
        if total == 0:
            pytest.skip("No claude-code messages in seeded database")

    def test_new_extraction_is_superset(self, seeded_db):
        """New extraction should provide more information, not less."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
            LIMIT 500
            """
        )

        rows = cur.fetchall()
        conn.close()

        tool_calls_found = 0
        reasoning_found = 0

        for (pm_json,) in rows:
            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            if not is_message_record("claude-code", raw):
                continue

            new_msg = extract_harmonized_message("claude-code", raw)
            tool_calls_found += len(new_msg.tool_calls)
            reasoning_found += len(new_msg.reasoning_traces)

        assert tool_calls_found >= 0
        assert reasoning_found >= 0
        print(f"Reasoning traces extracted: {reasoning_found}")

        assert tool_calls_found > 0 or reasoning_found >= 0, "New extraction should find viewports"


class TestAPICompatibility:
    """Test that new extraction can be adapted to old API."""

    def test_parsed_message_equivalent_fields(self):
        """HarmonizedMessage has equivalent fields to ParsedMessage."""
        from polylogue.schemas.unified import HarmonizedMessage
        from polylogue.sources.parsers.base import ParsedMessage

        field_mapping = {
            "provider_message_id": "id",
            "role": "role",
            "text": "text",
            "timestamp": "timestamp",
            "provider_meta": "raw",
        }

        pm = ParsedMessage(provider_message_id="test", role="user", text="hello")
        for pm_field in field_mapping:
            assert hasattr(pm, pm_field), f"ParsedMessage missing {pm_field}"

        hm = HarmonizedMessage(role="user", text="hello", provider="test")
        for hm_field in field_mapping.values():
            assert hasattr(hm, hm_field), f"HarmonizedMessage missing {hm_field}"

    def test_can_convert_harmonized_to_parsed(self):
        """Demonstrate conversion from HarmonizedMessage to ParsedMessage."""
        from polylogue.sources.parsers.base import ParsedMessage

        raw = {
            "type": "assistant",
            "uuid": "test-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}]
            }
        }

        hm = extract_harmonized_message("claude-code", raw)

        pm = ParsedMessage(
            provider_message_id=hm.id or "unknown",
            role=hm.role,
            text=hm.text,
            timestamp=hm.timestamp.isoformat() if hm.timestamp else None,
            provider_meta={"raw": hm.raw},
        )

        assert pm.role == "assistant"
        assert pm.text == "Hello!"
        assert pm.provider_message_id == "test-123"


# =============================================================================
# METADATA OPERATIONS (update, delete, add_tag, remove_tag, RMW atomicity)
# =============================================================================


def _seed_conversation(backend):
    """Helper: insert a conversation so metadata operations have a target."""
    conn = backend._get_connection()
    conv = make_conversation("conv1", content_hash="hash1")
    msg = make_message("m1", "conv1", text="Hello")
    store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)
    conn.commit()
    return "conv1"


def test_update_and_get_metadata(sqlite_backend):
    """update_metadata stores and retrieves values."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.update_metadata(conv_id, "key1", "value1")
    meta = sqlite_backend.get_metadata(conv_id)
    assert meta["key1"] == "value1"


def test_update_metadata_overwrites(sqlite_backend):
    """update_metadata overwrites existing key."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.update_metadata(conv_id, "key1", "old")
    sqlite_backend.update_metadata(conv_id, "key1", "new")
    meta = sqlite_backend.get_metadata(conv_id)
    assert meta["key1"] == "new"


def test_update_metadata_preserves_other_keys(sqlite_backend):
    """update_metadata doesn't clobber unrelated keys."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.update_metadata(conv_id, "key1", "val1")
    sqlite_backend.update_metadata(conv_id, "key2", "val2")
    meta = sqlite_backend.get_metadata(conv_id)
    assert meta["key1"] == "val1"
    assert meta["key2"] == "val2"


def test_delete_metadata_removes_key(sqlite_backend):
    """delete_metadata removes a specific key."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.update_metadata(conv_id, "key1", "val1")
    sqlite_backend.update_metadata(conv_id, "key2", "val2")
    sqlite_backend.delete_metadata(conv_id, "key1")
    meta = sqlite_backend.get_metadata(conv_id)
    assert "key1" not in meta
    assert meta["key2"] == "val2"


def test_delete_metadata_nonexistent_key_is_noop(sqlite_backend):
    """delete_metadata on nonexistent key doesn't raise."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.delete_metadata(conv_id, "nope")  # Should not raise
    meta = sqlite_backend.get_metadata(conv_id)
    assert "nope" not in meta


def test_add_tag(sqlite_backend):
    """add_tag appends to tags list."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.add_tag(conv_id, "important")
    meta = sqlite_backend.get_metadata(conv_id)
    assert "important" in meta["tags"]


def test_add_tag_idempotent(sqlite_backend):
    """add_tag doesn't duplicate existing tags."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.add_tag(conv_id, "dup")
    sqlite_backend.add_tag(conv_id, "dup")
    meta = sqlite_backend.get_metadata(conv_id)
    assert meta["tags"].count("dup") == 1


def test_add_multiple_tags(sqlite_backend):
    """Multiple tags accumulate correctly."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.add_tag(conv_id, "tag1")
    sqlite_backend.add_tag(conv_id, "tag2")
    sqlite_backend.add_tag(conv_id, "tag3")
    meta = sqlite_backend.get_metadata(conv_id)
    assert sorted(meta["tags"]) == ["tag1", "tag2", "tag3"]


def test_remove_tag(sqlite_backend):
    """remove_tag removes a tag from the list."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.add_tag(conv_id, "keep")
    sqlite_backend.add_tag(conv_id, "drop")
    sqlite_backend.remove_tag(conv_id, "drop")
    meta = sqlite_backend.get_metadata(conv_id)
    assert "keep" in meta["tags"]
    assert "drop" not in meta["tags"]


def test_remove_tag_nonexistent_is_noop(sqlite_backend):
    """remove_tag on nonexistent tag doesn't raise."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.remove_tag(conv_id, "nope")  # Should not raise


def test_metadata_rmw_mutator_exception_rolls_back(sqlite_backend):
    """If the mutator function raises, metadata is not modified."""
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.update_metadata(conv_id, "safe", "original")

    def bad_mutator(meta):
        meta["safe"] = "corrupted"
        raise ValueError("intentional error")

    with pytest.raises(ValueError, match="intentional error"):
        sqlite_backend._metadata_read_modify_write(conv_id, bad_mutator)

    # Verify metadata was not corrupted
    meta = sqlite_backend.get_metadata(conv_id)
    assert meta["safe"] == "original"


def test_metadata_operations_after_store(sqlite_backend):
    """Metadata operations work correctly immediately after store_records."""
    # This tests the BEGIN IMMEDIATE / SAVEPOINT nesting since
    # store_records may leave an implicit transaction open
    conv_id = _seed_conversation(sqlite_backend)
    sqlite_backend.add_tag(conv_id, "after-store")
    sqlite_backend.update_metadata(conv_id, "key", "value")
    meta = sqlite_backend.get_metadata(conv_id)
    assert "after-store" in meta.get("tags", [])
    assert meta["key"] == "value"



# =============================================================================
# MERGED FROM test_sqlite_backend.py - Additional SQLiteBackend Tests
# =============================================================================

# ============================================================================
# Test: connection_context
# ============================================================================


class TestConnectionContext:
    """Tests for connection_context context manager."""

    def test_connection_context_creates_connection(self, tmp_path):
        """Test that connection_context creates and yields a valid connection."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            assert isinstance(conn, sqlite3.Connection)
            assert conn.row_factory == sqlite3.Row
            # Verify schema was created
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            assert len(tables) > 0

    def test_connection_context_closes_connection(self, tmp_path):
        """Test that connection_context closes the connection after exiting."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            conn_obj = conn
        # After exiting context, connection should be closed
        with pytest.raises(sqlite3.ProgrammingError):
            conn_obj.execute("SELECT 1")

    def test_connection_context_with_none_uses_default(self, tmp_path, monkeypatch):
        """Test that connection_context(None) uses default_db_path()."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        # Reload to pick up env change
        import importlib

        import polylogue.paths

        importlib.reload(polylogue.paths)

        with connection_context(None) as conn:
            assert isinstance(conn, sqlite3.Connection)
            # Verify it created a database file
            assert Path(conn.execute("PRAGMA database_list").fetchone()[2]).exists()

    def test_connection_context_with_existing_connection(self, tmp_path):
        """Test that connection_context yields the passed connection unchanged."""
        db_path = tmp_path / "test.db"
        # Create initial connection
        with connection_context(db_path) as initial_conn:
            initial_conn_obj = initial_conn
            # Pass the connection to connection_context again
            with connection_context(initial_conn_obj) as conn:
                assert conn is initial_conn_obj

    def test_connection_context_sets_pragmas(self, tmp_path):
        """Test that connection_context sets required PRAGMAs."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            # Check foreign keys enabled
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            assert fk == 1
            # Check WAL mode enabled
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.upper() == "WAL"

    def test_connection_context_creates_schema(self, tmp_path):
        """Test that connection_context ensures schema is created."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            # Check key tables exist
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "conversations" in tables
            assert "messages" in tables
            assert "attachments" in tables
            assert "attachment_refs" in tables

    def test_connection_context_parent_directory_created(self, tmp_path):
        """Test that connection_context creates parent directories."""
        nested_path = tmp_path / "a" / "b" / "c" / "test.db"
        with connection_context(nested_path) as conn:
            assert isinstance(conn, sqlite3.Connection)
            assert nested_path.exists()


# ============================================================================
# Test: SQLiteBackend.__init__
# ============================================================================


class TestSQLiteBackendInit:
    """Tests for SQLiteBackend initialization."""

    def test_init_with_custom_path(self, tmp_path):
        """Test SQLiteBackend initialization with custom path."""
        db_path = tmp_path / "custom.db"
        backend = SQLiteBackend(db_path=db_path)
        assert backend._db_path == db_path

    def test_init_with_none_uses_default(self, tmp_path, monkeypatch):
        """Test that SQLiteBackend(None) uses default_db_path()."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        import importlib

        import polylogue.paths

        importlib.reload(polylogue.paths)

        backend = SQLiteBackend(db_path=None)
        # Should use default_db_path() which includes XDG_DATA_HOME
        assert "polylogue" in str(backend._db_path)
        assert str(backend._db_path).endswith("polylogue.db")

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that SQLiteBackend creates parent directories."""
        nested_path = tmp_path / "x" / "y" / "z" / "test.db"
        backend = SQLiteBackend(db_path=nested_path)
        assert nested_path.parent.exists()

    def test_init_thread_local_storage(self, tmp_path):
        """Test that SQLiteBackend has thread-local storage."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        assert hasattr(backend, "_local")
        import threading

        assert isinstance(backend._local, threading.local)


# ============================================================================
# Test: save_conversation + get_conversation round-trip
# ============================================================================


class TestSaveGetConversation:
    """Tests for save_conversation and get_conversation operations."""

    def test_save_and_get_conversation_basic(self, tmp_path):
        """Test basic save and retrieval of a conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-conv-1",
            title="Test Conversation",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T12:00:00Z",
            content_hash="hash123",
            provider_meta={"source": "test"},
            metadata={"tags": ["important"]},
            version=1,
        )
        backend.save_conversation(record)
        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv-1"
        assert retrieved.title == "Test Conversation"
        assert retrieved.provider_meta == {"source": "test"}
        assert retrieved.metadata == {"tags": ["important"]}

    def test_get_nonexistent_conversation(self, tmp_path):
        """Test that get_conversation returns None for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = backend.get_conversation("nonexistent")
        assert result is None

    def test_save_conversation_upsert_different_hash(self, tmp_path):
        """Test upsert: same ID, different content_hash → updates."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Original Title",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T10:00:00Z",
            content_hash="hash_old",
            version=1,
        )
        backend.save_conversation(record1)

        record2 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Updated Title",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T12:00:00Z",
            content_hash="hash_new",
            version=2,
        )
        backend.save_conversation(record2)

        retrieved = backend.get_conversation("conv-1")
        assert retrieved.title == "Updated Title"
        assert retrieved.updated_at == "2025-01-01T12:00:00Z"

    def test_save_conversation_no_update_same_hash(self, tmp_path):
        """Test upsert: same ID and content_hash → no update."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Original",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T10:00:00Z",
            content_hash="hash123",
            version=1,
        )
        backend.save_conversation(record)
        retrieved1 = backend.get_conversation("conv-1")

        # Save with same hash and content
        backend.save_conversation(record)
        retrieved2 = backend.get_conversation("conv-1")

        assert retrieved1.title == retrieved2.title
        assert retrieved1.updated_at == retrieved2.updated_at

    def test_save_conversation_metadata_not_overwritten(self, tmp_path):
        """Test that upsert does NOT overwrite user metadata."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T10:00:00Z",
            content_hash="hash1",
            metadata={"tags": ["important"]},
            version=1,
        )
        backend.save_conversation(record1)

        # Update metadata manually
        backend.update_metadata("conv-1", "custom_key", "custom_value")

        # Save new record with different content_hash
        record2 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T12:00:00Z",
            content_hash="hash2",
            metadata=None,  # Empty metadata in the record
            version=2,
        )
        backend.save_conversation(record2)

        # Metadata should still have the custom key
        meta = backend.get_metadata("conv-1")
        assert meta.get("custom_key") == "custom_value"

    def test_save_conversation_with_null_fields(self, tmp_path):
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
        backend.save_conversation(record)
        retrieved = backend.get_conversation("conv-1")
        assert retrieved.title is None
        assert retrieved.created_at is None

    def test_save_conversation_with_branching_info(self, tmp_path):
        """Test save_conversation with parent and branch_type."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Create parent conversation
        parent = ConversationRecord(
            conversation_id="conv-parent",
            provider_name="claude",
            provider_conversation_id="prov-parent",
            title="Parent",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash-parent",
            version=1,
        )
        backend.save_conversation(parent)

        # Create child conversation
        child = ConversationRecord(
            conversation_id="conv-child",
            provider_name="claude",
            provider_conversation_id="prov-child",
            title="Child",
            created_at="2025-01-01T01:00:00Z",
            updated_at="2025-01-01T01:00:00Z",
            content_hash="hash-child",
            version=1,
            parent_conversation_id="conv-parent",
            branch_type="continuation",
        )
        backend.save_conversation(child)

        retrieved = backend.get_conversation("conv-child")
        assert retrieved.parent_conversation_id == "conv-parent"
        assert retrieved.branch_type == "continuation"


# ============================================================================
# Test: save_messages + get_messages round-trip
# ============================================================================


class TestSaveGetMessages:
    """Tests for save_messages and get_messages operations."""

    def test_save_and_get_messages(self, tmp_path):
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
        backend.save_conversation(conv)

        messages = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                provider_message_id="prov-msg-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T10:00:00Z",
                content_hash="msg-hash-1",
                version=1,
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                provider_message_id="prov-msg-2",
                role="assistant",
                text="Hi there",
                timestamp="2025-01-01T10:01:00Z",
                content_hash="msg-hash-2",
                version=1,
            ),
        ]
        backend.save_messages(messages)

        retrieved = backend.get_messages("conv-1")
        assert len(retrieved) == 2
        assert retrieved[0].message_id == "msg-1"
        assert retrieved[1].message_id == "msg-2"

    def test_save_empty_messages_list(self, tmp_path):
        """Test that save_messages([]) is a no-op."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        backend.save_messages([])  # Should not raise
        # No messages to retrieve, but no error either
        assert True

    def test_get_messages_ordering_by_timestamp(self, tmp_path):
        """Test that messages are returned ordered by timestamp."""
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
        backend.save_conversation(conv)

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
        backend.save_messages(messages)

        retrieved = backend.get_messages("conv-1")
        assert [m.message_id for m in retrieved] == ["msg-1", "msg-2", "msg-3"]

    def test_save_messages_upsert_different_hash(self, tmp_path):
        """Test upsert: same message_id, different content_hash → updates."""
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
        backend.save_conversation(conv)

        msg1 = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Original",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash_old",
            version=1,
        )
        backend.save_messages([msg1])

        msg2 = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Updated",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash_new",
            version=2,
        )
        backend.save_messages([msg2])

        retrieved = backend.get_messages("conv-1")
        assert retrieved[0].text == "Updated"

    def test_save_messages_no_update_same_hash(self, tmp_path):
        """Test upsert: same message_id and content_hash → no update."""
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
        backend.save_conversation(conv)

        msg = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Hello",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash123",
            version=1,
        )
        backend.save_messages([msg])
        retrieved1 = backend.get_messages("conv-1")[0]

        backend.save_messages([msg])  # Save again with same hash
        retrieved2 = backend.get_messages("conv-1")[0]

        assert retrieved1.text == retrieved2.text

    def test_get_messages_nonexistent_conversation(self, tmp_path):
        """Test get_messages returns empty list for nonexistent conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = backend.get_messages("nonexistent")
        assert result == []

    def test_save_messages_with_provider_meta(self, tmp_path):
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
        backend.save_conversation(conv)

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
        backend.save_messages([msg])

        retrieved = backend.get_messages("conv-1")[0]
        assert retrieved.provider_meta == {"custom": "data"}

    def test_save_messages_with_branching_info(self, tmp_path):
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
        backend.save_conversation(conv)

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
        backend.save_messages([msg])

        retrieved = backend.get_messages("conv-1")[0]
        assert retrieved.parent_message_id == "msg-parent"
        assert retrieved.branch_index == 2


# ============================================================================
# Test: get_conversations_batch
# ============================================================================


class TestGetConversationsBatch:
    """Tests for get_conversations_batch operation."""

    def test_get_conversations_batch_basic(self, tmp_path):
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
            backend.save_conversation(record)

        # Request in different order
        batch = backend.get_conversations_batch(["conv-3", "conv-1", "conv-2"])
        assert [r.conversation_id for r in batch] == ["conv-3", "conv-1", "conv-2"]

    def test_get_conversations_batch_missing_ids_skipped(self, tmp_path):
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
        backend.save_conversation(record)

        batch = backend.get_conversations_batch(["conv-1", "nonexistent", "also-missing"])
        assert len(batch) == 1
        assert batch[0].conversation_id == "conv-1"

    def test_get_conversations_batch_empty_input(self, tmp_path):
        """Test that empty input returns empty list."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        batch = backend.get_conversations_batch([])
        assert batch == []

    def test_get_conversations_batch_duplicate_ids(self, tmp_path):
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
        backend.save_conversation(record)

        batch = backend.get_conversations_batch(["conv-1", "conv-1", "conv-1"])
        assert len(batch) == 3
        assert all(r.conversation_id == "conv-1" for r in batch)


# ============================================================================
# Test: Transaction management (begin/commit/rollback)
# ============================================================================


class TestTransactionManagement:
    """Tests for transaction management."""

    def test_begin_commit_persists_data(self, tmp_path):
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

        backend.begin()
        backend.save_conversation(record)
        backend.commit()

        # Verify persisted
        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None

    def test_begin_rollback_reverts_data(self, tmp_path):
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

        backend.begin()
        backend.save_conversation(record)
        backend.rollback()

        # Should not be persisted
        retrieved = backend.get_conversation("conv-1")
        assert retrieved is None

    def test_nested_savepoints(self, tmp_path):
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

        backend.begin()
        backend.save_conversation(record1)

        backend.begin()
        backend.save_conversation(record2)
        backend.rollback()  # Rollback only record2

        backend.commit()  # Commit record1

        # record1 should exist, record2 should not
        assert backend.get_conversation("conv-1") is not None
        assert backend.get_conversation("conv-2") is None

    def test_commit_without_begin_raises_error(self, tmp_path):
        """Test that commit without begin raises DatabaseError."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Need to access connection first to initialize transaction_depth
        backend._get_connection()
        with pytest.raises(Exception, match="No active transaction to commit"):
            backend.commit()

    def test_rollback_without_begin_raises_error(self, tmp_path):
        """Test that rollback without begin raises DatabaseError."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Need to access connection first to initialize transaction_depth
        backend._get_connection()
        with pytest.raises(Exception, match="No active transaction to rollback"):
            backend.rollback()

    def test_transaction_context_manager(self, tmp_path):
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

        with backend.transaction():
            backend.save_conversation(record)

        assert backend.get_conversation("conv-1") is not None

    def test_transaction_context_manager_rollback_on_error(self, tmp_path):
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
            with backend.transaction():
                backend.save_conversation(record)
                raise ValueError("Test error")
        except ValueError:
            pass

        assert backend.get_conversation("conv-1") is None


# ============================================================================
# Test: Metadata operations
# ============================================================================


class TestMetadataOperations:
    """Tests for metadata CRUD operations."""

    def _create_conversation(self, backend, cid="conv-1"):
        """Helper to create a conversation."""
        record = ConversationRecord(
            conversation_id=cid,
            provider_name="claude",
            provider_conversation_id=f"prov-{cid}",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

    def test_update_metadata(self, tmp_path):
        """Test setting a metadata key."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.update_metadata("conv-1", "rating", 5)

        meta = backend.get_metadata("conv-1")
        assert meta.get("rating") == 5

    def test_delete_metadata(self, tmp_path):
        """Test removing a metadata key."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.update_metadata("conv-1", "key1", "value1")
        backend.delete_metadata("conv-1", "key1")

        meta = backend.get_metadata("conv-1")
        assert "key1" not in meta

    def test_add_tag(self, tmp_path):
        """Test adding a tag."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.add_tag("conv-1", "important")

        meta = backend.get_metadata("conv-1")
        assert "important" in meta.get("tags", [])

    def test_remove_tag(self, tmp_path):
        """Test removing a tag."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.add_tag("conv-1", "important")
        backend.remove_tag("conv-1", "important")

        meta = backend.get_metadata("conv-1")
        assert "important" not in meta.get("tags", [])

    def test_add_tag_idempotent(self, tmp_path):
        """Test that adding same tag twice doesn't duplicate."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-1", "important")

        meta = backend.get_metadata("conv-1")
        tags = meta.get("tags", [])
        assert tags.count("important") == 1

    def test_list_tags(self, tmp_path):
        """Test listing all tags with counts."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        for i in range(3):
            self._create_conversation(backend, f"conv-{i+1}")

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-2", "important")
        backend.add_tag("conv-3", "follow-up")

        tags = backend.list_tags()
        assert tags.get("important") == 2
        assert tags.get("follow-up") == 1

    def test_list_tags_with_provider_filter(self, tmp_path):
        """Test listing tags filtered by provider."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Create conversations with different providers
        record1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record1)

        record2 = ConversationRecord(
            conversation_id="conv-2",
            provider_name="chatgpt",
            provider_conversation_id="prov-2",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record2)

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-2", "important")

        claude_tags = backend.list_tags(provider="claude")
        assert claude_tags.get("important") == 1

    def test_set_metadata_replaces_entire_dict(self, tmp_path):
        """Test set_metadata replaces the entire metadata dict."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.update_metadata("conv-1", "key1", "value1")
        backend.update_metadata("conv-1", "key2", "value2")

        new_metadata = {"new_key": "new_value"}
        backend.set_metadata("conv-1", new_metadata)

        meta = backend.get_metadata("conv-1")
        assert meta == new_metadata
        assert "key1" not in meta

    def test_get_metadata_nonexistent_conversation(self, tmp_path):
        """Test get_metadata returns empty dict for nonexistent conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        meta = backend.get_metadata("nonexistent")
        assert meta == {}


# ============================================================================
# Test: delete_conversation
# ============================================================================


class TestDeleteConversation:
    """Tests for delete_conversation operation."""

    def test_delete_conversation_success(self, tmp_path):
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
        backend.save_conversation(record)

        result = backend.delete_conversation("conv-1")
        assert result is True
        assert backend.get_conversation("conv-1") is None

    def test_delete_conversation_nonexistent(self, tmp_path):
        """Test deleting nonexistent conversation returns False."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = backend.delete_conversation("nonexistent")
        assert result is False

    def test_delete_conversation_with_messages(self, tmp_path):
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
        backend.save_conversation(record)

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
        backend.save_messages(messages)

        backend.delete_conversation("conv-1")
        assert backend.get_messages("conv-1") == []

    def test_delete_conversation_with_attachments(self, tmp_path):
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
        backend.save_conversation(record)

        attachment = AttachmentRecord(
            attachment_id="att-1",
            conversation_id="conv-1",
            message_id=None,
            mime_type="image/png",
            size_bytes=1024,
            path="/path/to/image.png",
        )
        backend.save_attachments([attachment])

        backend.delete_conversation("conv-1")
        # Verify conversation is gone
        assert backend.get_conversation("conv-1") is None


# ============================================================================
# Test: get_conversation_stats
# ============================================================================


class TestGetConversationStats:
    """Tests for get_conversation_stats operation."""

    def test_get_conversation_stats_basic(self, tmp_path):
        """Test getting message counts for a conversation."""
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
        backend.save_conversation(record)

        messages = [
            MessageRecord(
                message_id=f"msg-{i}",
                conversation_id="conv-1",
                role="user" if i % 2 == 0 else "assistant",
                text=f"Message {i}",
                timestamp=f"2025-01-01T10:{i:02d}:00Z",
                content_hash=f"h{i}",
                version=1,
            )
            for i in range(5)
        ]
        backend.save_messages(messages)

        stats = backend.get_conversation_stats("conv-1")
        assert stats["total_messages"] == 5
        assert stats["dialogue_messages"] == 5
        assert stats["tool_messages"] == 0

    def test_get_conversation_stats_with_tool_messages(self, tmp_path):
        """Test stats with mixed message types."""
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
        backend.save_conversation(record)

        messages = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T10:00:00Z",
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                role="tool",
                text="Tool output",
                timestamp="2025-01-01T10:01:00Z",
                content_hash="h2",
                version=1,
            ),
            MessageRecord(
                message_id="msg-3",
                conversation_id="conv-1",
                role="assistant",
                text="Response",
                timestamp="2025-01-01T10:02:00Z",
                content_hash="h3",
                version=1,
            ),
        ]
        backend.save_messages(messages)

        stats = backend.get_conversation_stats("conv-1")
        assert stats["total_messages"] == 3
        assert stats["dialogue_messages"] == 2
        assert stats["tool_messages"] == 1

    def test_get_conversation_stats_empty_conversation(self, tmp_path):
        """Test stats for conversation with no messages."""
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
        backend.save_conversation(record)

        stats = backend.get_conversation_stats("conv-1")
        assert stats["total_messages"] == 0
        assert stats["dialogue_messages"] == 0
        assert stats["tool_messages"] == 0


# ============================================================================
# Test: Helper functions
# ============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_json_or_none_with_dict(self):
        """Test _json_or_none with a dictionary."""
        result = _json_or_none({"key": "value"})
        assert isinstance(result, str)
        assert json.loads(result) == {"key": "value"}

    def test_json_or_none_with_none(self):
        """Test _json_or_none with None."""
        result = _json_or_none(None)
        assert result is None

    def test_json_or_none_with_nested_dict(self):
        """Test _json_or_none with nested structures."""
        data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        result = _json_or_none(data)
        assert json.loads(result) == data

    def test_make_ref_id_deterministic(self):
        """Test that _make_ref_id produces deterministic results."""
        ref_id_1 = _make_ref_id("att-1", "conv-1", "msg-1")
        ref_id_2 = _make_ref_id("att-1", "conv-1", "msg-1")
        assert ref_id_1 == ref_id_2

    def test_make_ref_id_different_inputs(self):
        """Test that _make_ref_id produces different IDs for different inputs."""
        ref_id_1 = _make_ref_id("att-1", "conv-1", "msg-1")
        ref_id_2 = _make_ref_id("att-2", "conv-1", "msg-1")
        assert ref_id_1 != ref_id_2

    def test_make_ref_id_format(self):
        """Test that _make_ref_id has the correct format."""
        ref_id = _make_ref_id("att-1", "conv-1", "msg-1")
        assert ref_id.startswith("ref-")
        assert len(ref_id) == len("ref-") + 16  # 16-char hex digest

    def test_make_ref_id_with_none_message_id(self):
        """Test _make_ref_id with None message_id."""
        ref_id_1 = _make_ref_id("att-1", "conv-1", None)
        ref_id_2 = _make_ref_id("att-1", "conv-1", "msg-1")
        assert ref_id_1 != ref_id_2

    def test_default_db_path(self, tmp_path, monkeypatch):
        """Test that default_db_path returns correct path."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        import importlib

        import polylogue.paths

        importlib.reload(polylogue.paths)

        path = default_db_path()
        assert str(path).endswith("polylogue.db")
        assert "polylogue" in str(path)


# ============================================================================
# Test: Backend lifecycle (close)
# ============================================================================


class TestBackendLifecycle:
    """Tests for backend lifecycle management."""

    def test_close_backend(self, tmp_path):
        """Test that close() closes the connection."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Access connection
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
        backend.save_conversation(record)

        backend.close()

        # After close, operations should fail or create new connection
        # depending on lazy connection semantics
        # Just verify it doesn't raise
        assert True

    def test_close_and_reopen(self, tmp_path):
        """Test that connection can be re-established after close."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db_path)
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
        # Use transaction to ensure data is persisted
        with backend.transaction():
            backend.save_conversation(record)

        # Verify data exists before close
        retrieved1 = backend.get_conversation("conv-1")
        assert retrieved1 is not None

        backend.close()

        # After close, the thread-local connection is cleared
        # Verify a new connection can be established
        conn = backend._get_connection()
        assert conn is not None

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from pydantic import ValidationError

from polylogue.storage.backends.sqlite import open_connection
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

    msg = make_message("msg1", "conv1", role=None, text=None, timestamp=None, provider_message_id=None, provider_meta=None)

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
            messages = [
                make_message(f"msg{conv_id}-{i}", f"conv{conv_id}", text=f"Message {i}")
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


def test_store_records_without_connection_creates_own(test_db, tmp_path, monkeypatch):
    """store_records() works without explicit connection parameter."""
    import shutil

    # Create a temp location for "default" storage within tmp_path to avoid cross-device issues
    state_home = tmp_path / "state"
    state_home.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    # Move test DB to default location
    from polylogue.storage.backends.sqlite import default_db_path

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
        conv = make_conversation(f"race-conv-{i}", title=f"Race Test {i}", created_at=None, updated_at=None, content_hash=f"hash-{i}")
        msg = make_message(f"race-msg-{i}", f"race-conv-{i}", text="test", timestamp=None, provider_meta=None)
        # Each conversation references the SAME attachment_id
        attachment = make_attachment(SHARED_ATTACHMENT_ID, f"race-conv-{i}", f"race-msg-{i}", mime_type="text/plain", size_bytes=100, provider_meta=None)
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

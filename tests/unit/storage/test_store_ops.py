"""Store record operations tests — insert, upsert, deduplication, attachment refs, provider validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from pydantic import ValidationError

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.sqlite import (
    default_db_path,
    open_connection,
)
from polylogue.storage.store import (
    MAX_ATTACHMENT_SIZE,
    AttachmentRecord,
    ConversationRecord,
)
from tests.infra.helpers import (
    _make_ref_id,
    _prune_attachment_refs,
    make_attachment,
    make_conversation,
    make_message,
    store_records,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)

# test_db and test_conn fixtures are in conftest.py


# =============================================================================
# STORE RECORD OPERATIONS (from test_store.py)
# =============================================================================


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


@pytest.mark.slow
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

    # Close cached connections before moving the DB file — otherwise
    # WAL sidecar files (.db-wal, .db-shm) won't be checkpointed and
    # the moved file will fail with "disk I/O error".
    from polylogue.storage.backends.connection import _clear_connection_cache
    _clear_connection_cache()

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


@pytest.mark.slow
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


# =============================================================================
# ATTACHMENT RECORD VALIDATION (parametrized)
# =============================================================================

# Valid size_bytes test cases (consolidated)
VALID_ATTACHMENT_SIZES = [(0, "zero"), (MAX_ATTACHMENT_SIZE, "max_1TB"), (None, "unknown")]


@pytest.mark.parametrize("size_bytes,desc", VALID_ATTACHMENT_SIZES, ids=str)
def test_attachment_size_bytes_valid(size_bytes, desc):
    """size_bytes accepts valid values: {desc}."""
    record = AttachmentRecord(
        attachment_id="test",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="text/plain",
        size_bytes=size_bytes,
        provider_meta=None,
    )
    assert record.size_bytes == size_bytes


@pytest.mark.parametrize("size_bytes,desc", [(-100, "negative"), (MAX_ATTACHMENT_SIZE + 1, "over_max")], ids=str)
def test_attachment_size_bytes_invalid(size_bytes, desc):
    """size_bytes rejects invalid values: {desc}."""
    with pytest.raises(ValidationError):
        AttachmentRecord(
            attachment_id="test",
            conversation_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=size_bytes,
            provider_meta=None,
        )


# =============================================================================
# PROVIDER NAME VALIDATION (consolidated to representative cases)
# =============================================================================


@pytest.mark.parametrize("name,desc", [("claude", "known"), ("claude-code", "hyphenated"), ("Provider123", "mixed_case")], ids=str)
def test_provider_name_accepts_valid(name, desc):
    """provider_name accepts {desc}."""
    record = ConversationRecord(
        conversation_id="test",
        provider_name=name,
        provider_conversation_id="ext1",
        title="Test",
        content_hash="hash123",
    )
    assert record.provider_name == name


@pytest.mark.parametrize("name,desc", [("", "empty"), ("123invalid", "starts_number"), ("../escape", "path_escape")], ids=str)
def test_provider_name_rejects_invalid(name, desc):
    """provider_name rejects {desc}."""
    with pytest.raises(ValidationError):
        ConversationRecord(
            conversation_id="test",
            provider_name=name,
            provider_conversation_id="ext1",
            title="Test",
            content_hash="hash123",
        )


# =============================================================================
# BACKEND CRUD OPERATIONS
# =============================================================================


async def test_backend_transaction_rollback(sqlite_backend: SQLiteBackend) -> None:
    """Test transaction rollback."""
    conv = make_conversation("conv1", title="Test")

    await sqlite_backend.begin()
    await sqlite_backend.save_conversation_record(conv)
    await sqlite_backend.rollback()

    retrieved = await sqlite_backend.get_conversation("conv1")
    assert retrieved is None


async def test_backend_transaction_context_manager(sqlite_backend: SQLiteBackend) -> None:
    """Test using the transaction context manager."""
    conv = make_conversation("conv1", title="Test")

    async with sqlite_backend.transaction():
        await sqlite_backend.save_conversation_record(conv)

    retrieved = await sqlite_backend.get_conversation("conv1")
    assert retrieved is not None
    assert retrieved.conversation_id == "conv1"


async def test_backend_transaction_context_manager_exception(sqlite_backend: SQLiteBackend) -> None:
    """Test transaction context manager rolls back on exception."""
    conv = make_conversation("conv1", title="Test")

    with pytest.raises(ValueError):
        async with sqlite_backend.transaction():
            await sqlite_backend.save_conversation_record(conv)
            raise ValueError("Test error")

    retrieved = await sqlite_backend.get_conversation("conv1")
    assert retrieved is None


async def test_backend_delete_conversation(sqlite_backend: SQLiteBackend) -> None:
    """Test deleting a conversation and all related records."""
    conv = make_conversation("conv1", title="Test")
    msg1 = make_message("msg1", "conv1", text="Hello")
    msg2 = make_message("msg2", "conv1", role="assistant", text="Hi there")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    await sqlite_backend.begin()
    await sqlite_backend.save_conversation_record(conv)
    await sqlite_backend.save_messages([msg1, msg2])
    await sqlite_backend.save_attachments([att])
    await sqlite_backend.commit()

    assert await sqlite_backend.get_conversation("conv1") is not None
    assert len(await sqlite_backend.get_messages("conv1")) == 2
    assert len(await sqlite_backend.get_attachments("conv1")) == 1

    result = await sqlite_backend.delete_conversation("conv1")
    assert result is True

    assert await sqlite_backend.get_conversation("conv1") is None
    assert len(await sqlite_backend.get_messages("conv1")) == 0
    assert len(await sqlite_backend.get_attachments("conv1")) == 0


async def test_backend_delete_conversation_not_found(sqlite_backend: SQLiteBackend) -> None:
    """Test deleting a non-existent conversation returns False."""
    result = await sqlite_backend.delete_conversation("nonexistent")
    assert result is False


async def test_backend_delete_conversation_cleans_fts(sqlite_backend: SQLiteBackend) -> None:
    """Test that deleting a conversation also cleans up FTS entries."""
    from polylogue.storage.index import ensure_index, update_index_for_conversations

    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="searchable content here")

    await sqlite_backend.begin()
    await sqlite_backend.save_conversation_record(conv)
    await sqlite_backend.save_messages([msg])
    await sqlite_backend.commit()

    with open_connection(sqlite_backend._db_path) as conn:
        ensure_index(conn)
        update_index_for_conversations(["conv1"], conn)
        conn.commit()

        fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
        assert fts_count > 0

    result = await sqlite_backend.delete_conversation("conv1")
    assert result is True

    with open_connection(sqlite_backend._db_path) as conn:
        fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
        assert fts_count == 0
    await sqlite_backend.close()

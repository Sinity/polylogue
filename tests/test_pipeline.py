"""Tests for pipeline modules: ids, ingest, and runner orchestration."""

from __future__ import annotations

import pytest

from polylogue.importers.base import ParsedAttachment, ParsedConversation, ParsedMessage
from polylogue.pipeline.ids import (
    conversation_content_hash,
    conversation_id,
    message_content_hash,
    message_id,
)
from polylogue.pipeline.ingest import prepare_ingest
from polylogue.storage.backends.sqlite import open_connection

# ============================================================================
# Test conversation_content_hash
# ============================================================================


def test_conversation_content_hash_deterministic():
    """conversation_content_hash() returns the same hash for identical content."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test Conversation",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Hello, world!",
                timestamp="2024-01-01T00:00:00Z",
            ),
            ParsedMessage(
                provider_message_id="msg-2",
                role="assistant",
                text="Hi there!",
                timestamp="2024-01-01T00:00:01Z",
            ),
        ],
        attachments=[],
    )

    hash1 = conversation_content_hash(convo)
    hash2 = conversation_content_hash(convo)

    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hex digest


def test_conversation_content_hash_changes_on_title_edit():
    """Changing conversation title changes the hash."""
    base_convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Original Title",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    hash_original = conversation_content_hash(base_convo)

    # Create same conversation with different title
    modified_convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Modified Title",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    hash_modified = conversation_content_hash(modified_convo)

    assert hash_original != hash_modified


def test_conversation_content_hash_changes_on_message_text_edit():
    """Changing message text changes the hash."""
    convo1 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Original text",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    hash1 = conversation_content_hash(convo1)

    convo2 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Modified text",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    hash2 = conversation_content_hash(convo2)

    assert hash1 != hash2


def test_conversation_content_hash_changes_on_message_reorder():
    """Reordering messages changes the hash."""
    convo1 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="First",
                timestamp="2024-01-01T00:00:00Z",
            ),
            ParsedMessage(
                provider_message_id="msg-2",
                role="assistant",
                text="Second",
                timestamp="2024-01-01T00:00:01Z",
            ),
        ],
        attachments=[],
    )

    hash1 = conversation_content_hash(convo1)

    convo2 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-2",
                role="assistant",
                text="Second",
                timestamp="2024-01-01T00:00:01Z",
            ),
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="First",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    hash2 = conversation_content_hash(convo2)

    assert hash1 != hash2


def test_conversation_content_hash_includes_attachments():
    """Changing attachments changes the hash."""
    convo1 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    hash1 = conversation_content_hash(convo1)

    convo2 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="msg-1",
                name="document.pdf",
                mime_type="application/pdf",
                size_bytes=1024,
            ),
        ],
    )

    hash2 = conversation_content_hash(convo2)

    assert hash1 != hash2


def test_conversation_content_hash_with_missing_message_ids():
    """Hash handles messages without provider_message_id (uses fallback idx)."""
    convo1 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="",  # Empty ID
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    hash1 = conversation_content_hash(convo1)
    assert isinstance(hash1, str)
    assert len(hash1) == 64


# ============================================================================
# Test message_content_hash
# ============================================================================


def test_message_content_hash_deterministic():
    """message_content_hash() is deterministic."""
    msg = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="Hello, world!",
        timestamp="2024-01-01T00:00:00Z",
    )

    hash1 = message_content_hash(msg, "msg-1")
    hash2 = message_content_hash(msg, "msg-1")

    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64


def test_message_content_hash_different_messages():
    """Different messages produce different hashes."""
    msg1 = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="Hello",
        timestamp="2024-01-01T00:00:00Z",
    )

    msg2 = ParsedMessage(
        provider_message_id="msg-2",
        role="assistant",
        text="Hi there",
        timestamp="2024-01-01T00:00:01Z",
    )

    hash1 = message_content_hash(msg1, "msg-1")
    hash2 = message_content_hash(msg2, "msg-2")

    assert hash1 != hash2


def test_message_content_hash_changes_on_text_edit():
    """Changing message text changes hash."""
    msg1 = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="Original",
        timestamp="2024-01-01T00:00:00Z",
    )

    msg2 = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="Modified",
        timestamp="2024-01-01T00:00:00Z",
    )

    hash1 = message_content_hash(msg1, "msg-1")
    hash2 = message_content_hash(msg2, "msg-1")

    assert hash1 != hash2


def test_message_content_hash_changes_on_role_edit():
    """Changing message role changes hash."""
    msg1 = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="Hello",
        timestamp="2024-01-01T00:00:00Z",
    )

    msg2 = ParsedMessage(
        provider_message_id="msg-1",
        role="assistant",
        text="Hello",
        timestamp="2024-01-01T00:00:00Z",
    )

    hash1 = message_content_hash(msg1, "msg-1")
    hash2 = message_content_hash(msg2, "msg-1")

    assert hash1 != hash2


# ============================================================================
# Test ID generation functions
# ============================================================================


def test_conversation_id_format():
    """conversation_id() formats correctly."""
    cid = conversation_id("chatgpt", "ext-conv-123")

    assert cid == "chatgpt:ext-conv-123"
    assert ":" in cid


def test_message_id_format():
    """message_id() formats correctly."""
    mid = message_id("chatgpt:ext-conv-123", "ext-msg-456")

    assert mid == "chatgpt:ext-conv-123:ext-msg-456"
    assert mid.count(":") == 2


# ============================================================================
# Test prepare_ingest
# ============================================================================


# test_db and test_conn fixtures are in conftest.py


@pytest.fixture
def test_repository(test_db):
    """Provide a repository pointing to the test database."""
    from polylogue.storage.backends.sqlite import SQLiteBackend
    from polylogue.storage.repository import StorageRepository
    backend = SQLiteBackend(db_path=test_db)
    return StorageRepository(backend=backend)


def test_prepare_ingest_new_conversation(test_conn, test_repository, tmp_path):
    """prepare_ingest() marks new conversation for insertion."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="new-conv-1",
        title="New Conversation",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    cid, counts, changed = prepare_ingest(
        convo,
        "test-source",
        archive_root=tmp_path / "archive",
        conn=test_conn,
        repository=test_repository,
    )

    # New conversation should be inserted
    assert counts["conversations"] == 1
    assert counts["messages"] == 1
    assert counts["skipped_conversations"] == 0
    assert changed is False  # First insert not considered a "change"
    assert cid == "test:new-conv-1"


def test_prepare_ingest_unchanged_conversation_skips(test_conn, test_repository, tmp_path):
    """prepare_ingest() skips conversation with unchanged content_hash."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test Conversation",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    # First ingest
    cid1, counts1, changed1 = prepare_ingest(
        convo,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    assert counts1["conversations"] == 1

    # Re-ingest same conversation (same content)
    cid2, counts2, changed2 = prepare_ingest(
        convo,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    # Same conversation ID and content should be skipped
    assert cid2 == cid1
    # Second insert should skip (same hash)
    assert counts2["conversations"] == 0
    assert changed2 is False


def test_prepare_ingest_detects_changed_content(test_conn, test_repository, tmp_path):
    """prepare_ingest() marks conversation as changed when content differs."""
    convo_v1 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test Conversation",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Original text",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    # First ingest
    cid1, _, _ = prepare_ingest(
        convo_v1,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    # Same conversation but with modified message
    convo_v2 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test Conversation",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Modified text",  # Changed!
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    # Re-ingest with different content
    cid2, _, changed = prepare_ingest(
        convo_v2,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    # Same conversation ID but content changed
    assert cid2 == cid1
    assert changed is True


def test_prepare_ingest_creates_message_records(test_conn, test_repository, tmp_path):
    """prepare_ingest() creates message records with proper IDs."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
            ParsedMessage(
                provider_message_id="msg-2",
                role="assistant",
                text="Hi there",
                timestamp="2024-01-01T00:00:01Z",
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    cid, counts, _ = prepare_ingest(
        convo,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    assert counts["conversations"] == 1
    assert counts["messages"] == 2
    # Verify messages are in DB
    rows = test_conn.execute(
        "SELECT message_id, role FROM messages WHERE conversation_id = ?",
        (cid,),
    ).fetchall()
    assert len(rows) == 2


def test_prepare_ingest_handles_empty_provider_message_ids(test_conn, test_repository, tmp_path):
    """prepare_ingest() generates IDs for messages with empty provider_message_id."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="",  # Empty - will use fallback
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            ),
            ParsedMessage(
                provider_message_id="msg-explicit",  # Explicit ID
                role="assistant",
                text="Hi",
                timestamp="2024-01-01T00:00:01Z",
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    cid, counts, _ = prepare_ingest(
        convo,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    assert counts["messages"] == 2
    # Both messages should be created (one with fallback, one with explicit ID)
    rows = test_conn.execute(
        "SELECT message_id FROM messages WHERE conversation_id = ? ORDER BY rowid",
        (cid,),
    ).fetchall()
    assert len(rows) == 2
    # First one should have a fallback ID (contains provider_message_id as 'msg-1')
    # Second one should contain the explicit ID
    provider_ids = [
        test_conn.execute(
            "SELECT provider_message_id FROM messages WHERE message_id = ?", (row["message_id"],)
        ).fetchone()["provider_message_id"]
        for row in rows
    ]
    assert "msg-1" in provider_ids  # Fallback
    assert "msg-explicit" in provider_ids


def test_prepare_ingest_stores_source_metadata(test_conn, test_repository, tmp_path):
    """prepare_ingest() stores source name in provider_meta."""
    convo = ParsedConversation(
        provider_name="chatgpt",
        provider_conversation_id="ext-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role="user",
                text="Hi",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    cid, _, _ = prepare_ingest(
        convo,
        "my-export",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    # Verify source is recorded
    row = test_conn.execute(
        "SELECT provider_meta FROM conversations WHERE conversation_id = ?",
        (cid,),
    ).fetchone()
    assert row is not None
    import json

    meta = json.loads(row["provider_meta"]) if isinstance(row["provider_meta"], str) else row["provider_meta"]
    assert meta.get("source") == "my-export"


def test_prepare_ingest_multiple_messages_get_unique_hashes(test_conn, test_repository, tmp_path):
    """prepare_ingest() creates unique content hashes for each message."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role="user",
                text="First",
                timestamp="2024-01-01T00:00:00Z",
            ),
            ParsedMessage(
                provider_message_id="m2",
                role="assistant",
                text="Second",
                timestamp="2024-01-01T00:00:01Z",
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    cid, _, _ = prepare_ingest(
        convo,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    # Verify each message has a unique content hash
    rows = test_conn.execute(
        "SELECT content_hash FROM messages WHERE conversation_id = ? ORDER BY provider_message_id",
        (cid,),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["content_hash"] != rows[1]["content_hash"]


def test_prepare_ingest_with_attachments(test_conn, test_repository, tmp_path):
    """prepare_ingest() creates attachment records."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role="user",
                text="See attachment",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="m1",
                name="document.pdf",
                mime_type="application/pdf",
                size_bytes=2048,
            ),
        ],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    cid, counts, _ = prepare_ingest(
        convo,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    assert counts["attachments"] == 1
    # Verify attachment ref in DB (attachments use attachment_refs table)
    att_refs = test_conn.execute(
        "SELECT ar.*, a.mime_type FROM attachment_refs ar "
        "JOIN attachments a ON ar.attachment_id = a.attachment_id "
        "WHERE ar.conversation_id = ?",
        (cid,),
    ).fetchall()
    assert len(att_refs) == 1
    assert att_refs[0]["mime_type"] == "application/pdf"


def test_prepare_ingest_returns_correct_tuple_structure(test_conn, test_repository, tmp_path):
    """prepare_ingest() returns (conversation_id, counts_dict, changed_bool)."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role="user",
                text="Hi",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    result = prepare_ingest(
        convo,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    assert isinstance(result, tuple)
    assert len(result) == 3

    cid, counts, changed = result

    assert isinstance(cid, str)
    assert isinstance(counts, dict)
    assert isinstance(changed, bool)

    # Verify counts structure
    required_keys = {
        "conversations",
        "messages",
        "attachments",
        "skipped_conversations",
        "skipped_messages",
        "skipped_attachments",
    }
    assert set(counts.keys()) >= required_keys

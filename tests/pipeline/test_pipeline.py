"""Tests for pipeline modules: ids, ingest, and runner orchestration.

Note: Most hashing property tests have been moved to test_properties.py, which
provides systematic coverage via Hypothesis. This file retains edge case tests,
validation tests, and integration tests for prepare_records.

Consolidated from:
- test_pipeline.py (original)
- test_pipeline_ids.py (merged)
"""

from __future__ import annotations

import pytest

from polylogue.assets import asset_path
from polylogue.pipeline.ids import (
    attachment_content_id,
    conversation_content_hash,
    conversation_id,
    message_content_hash,
    move_attachment_to_archive,
)
from polylogue.pipeline.prepare import prepare_records
from polylogue.sources.parsers.base import ParsedAttachment, ParsedConversation, ParsedMessage

# ============================================================================
# Attachment file handling tests (from test_pipeline_ids.py)
# ============================================================================


def test_attachment_content_id_moves_file_into_assets(tmp_path):
    """attachment_content_id moves file to archive and returns digest."""
    archive_root = tmp_path / "archive"
    uploads = tmp_path / "uploads"
    archive_root.mkdir()
    uploads.mkdir()
    source_file = uploads / "note.txt"
    source_file.write_text("hello world", encoding="utf-8")

    attachment = ParsedAttachment(
        provider_attachment_id="file-1",
        message_provider_id="msg-1",
        name="note.txt",
        mime_type="text/plain",
        size_bytes=11,
        path=str(source_file),  # Must set path for file to be moved
        provider_meta={},
    )

    # attachment_content_id returns (digest, updated_meta, updated_path) without mutation
    digest, updated_meta, updated_path = attachment_content_id("chatgpt", attachment, archive_root=archive_root)
    target = asset_path(archive_root, digest)

    assert digest
    assert updated_path == str(target)  # returned path, not mutated attachment.path
    assert updated_meta is not None and "sha256" in updated_meta
    assert not source_file.exists()
    assert target.exists()


class TestAttachmentPathMove:
    """Tests for move_attachment_to_archive error handling."""

    def test_move_attachment_raises_on_missing_source(self, tmp_path):
        """Moving non-existent attachment should raise FileNotFoundError."""
        missing_source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "archive" / "dest.txt"

        with pytest.raises(FileNotFoundError):
            move_attachment_to_archive(missing_source, dest)

    def test_move_attachment_raises_on_permission_error(self, tmp_path, monkeypatch):
        """Move failure due to permissions should raise PermissionError."""
        import shutil

        source = tmp_path / "source.txt"
        source.write_text("content")
        dest = tmp_path / "archive" / "dest.txt"

        def failing_move(*args, **kwargs):
            raise PermissionError("Access denied")

        monkeypatch.setattr(shutil, "move", failing_move)

        with pytest.raises(PermissionError):
            move_attachment_to_archive(source, dest)

    def test_move_attachment_creates_parent_dirs(self, tmp_path):
        """Move should create parent directories if needed."""
        source = tmp_path / "source.txt"
        source.write_text("content")
        dest = tmp_path / "deep" / "nested" / "archive" / "dest.txt"

        move_attachment_to_archive(source, dest)

        assert dest.exists()
        assert dest.read_text() == "content"
        assert not source.exists()  # Original should be gone


# ============================================================================
# conversation_id validation tests (from test_pipeline_ids.py)
# ============================================================================


class TestConversationIdValidation:
    """Tests for conversation_id input validation."""

    def test_rejects_empty_provider(self):
        """Empty provider_name MUST be rejected."""
        with pytest.raises(ValueError, match="provider"):
            conversation_id("", "conv-123")

    def test_rejects_empty_provider_conversation_id(self):
        """Empty provider_conversation_id MUST be rejected."""
        with pytest.raises(ValueError, match="conversation"):
            conversation_id("chatgpt", "")


# ============================================================================
# Hash edge case tests (from test_pipeline_ids.py)
# ============================================================================


def test_conversation_content_hash_with_missing_message_ids():
    """Hash handles messages without provider_message_id (uses fallback idx).

    This edge case tests the fallback behavior when provider_message_id is empty,
    which cannot be easily generated via Hypothesis strategies.
    """
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


def test_message_hash_none_vs_empty_timestamp_distinguishable():
    """None vs empty timestamp MUST produce different hashes.

    This test verifies hash collision prevention for edge cases.
    """
    msg_with_none = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="hello",
        timestamp=None,
    )
    msg_with_empty = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="hello",
        timestamp="",
    )

    hash_with_none = message_content_hash(msg_with_none, "msg-1")
    hash_with_empty = message_content_hash(msg_with_empty, "msg-1")

    assert hash_with_none != hash_with_empty, "Hash collision: None and empty string produce same hash!"


def test_message_hash_empty_text_is_deterministic():
    """Empty text should still produce valid, deterministic hash."""
    msg = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="",
        timestamp="2024-01-01",
    )
    hash1 = message_content_hash(msg, "msg-1")
    hash2 = message_content_hash(msg, "msg-1")
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex


def test_message_hash_different_provider_id_produces_different_hash():
    """Different fallback_id should produce different hash."""
    msg = ParsedMessage(
        provider_message_id="msg-1",
        role="user",
        text="hello",
        timestamp="2024-01-01",
    )
    hash1 = message_content_hash(msg, "msg-1")
    hash2 = message_content_hash(msg, "msg-2")
    assert hash1 != hash2


def test_conversation_hash_empty_messages_is_valid():
    """Conversation with no messages should still produce valid hash."""
    convo = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Empty Conv",
        created_at=None,
        updated_at=None,
        messages=[],
    )
    hash_result = conversation_content_hash(convo)
    assert len(hash_result) == 64


def test_conversation_hash_timestamps_affect_hash():
    """Different created_at/updated_at should produce different hashes."""
    msg = ParsedMessage(
        provider_message_id="m1",
        role="user",
        text="hi",
        timestamp=None,
    )
    convo1 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01",
        updated_at=None,
        messages=[msg],
    )
    convo2 = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-02",
        updated_at=None,
        messages=[msg],
    )

    hash1 = conversation_content_hash(convo1)
    hash2 = conversation_content_hash(convo2)

    assert hash1 != hash2


# ============================================================================
# Test prepare_records
# ============================================================================


# test_db and test_conn fixtures are in conftest.py


@pytest.fixture
def test_repository(test_db):
    """Provide a repository pointing to the test database."""
    from polylogue.storage.backends.sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository
    backend = SQLiteBackend(db_path=test_db)
    return ConversationRepository(backend=backend)


def test_prepare_records_new_conversation(test_conn, test_repository, tmp_path):
    """prepare_records() marks new conversation for insertion."""
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

    cid, counts, changed = prepare_records(
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


def test_prepare_records_unchanged_conversation_skips(test_conn, test_repository, tmp_path):
    """prepare_records() skips conversation with unchanged content_hash."""
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
    cid1, counts1, changed1 = prepare_records(
        convo,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    assert counts1["conversations"] == 1

    # Re-ingest same conversation (same content)
    cid2, counts2, changed2 = prepare_records(
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


def test_prepare_records_detects_changed_content(test_conn, test_repository, tmp_path):
    """prepare_records() marks conversation as changed when content differs."""
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
    cid1, _, _ = prepare_records(
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
    cid2, _, changed = prepare_records(
        convo_v2,
        "test-source",
        archive_root=archive_root,
        conn=test_conn,
        repository=test_repository,
    )

    # Same conversation ID but content changed
    assert cid2 == cid1
    assert changed is True


def test_prepare_records_creates_message_records(test_conn, test_repository, tmp_path):
    """prepare_records() creates message records with proper IDs."""
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

    cid, counts, _ = prepare_records(
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


def test_prepare_records_handles_empty_provider_message_ids(test_conn, test_repository, tmp_path):
    """prepare_records() generates IDs for messages with empty provider_message_id."""
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

    cid, counts, _ = prepare_records(
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


def test_prepare_records_stores_source_metadata(test_conn, test_repository, tmp_path):
    """prepare_records() stores source name in provider_meta."""
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

    cid, _, _ = prepare_records(
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


def test_prepare_records_multiple_messages_get_unique_hashes(test_conn, test_repository, tmp_path):
    """prepare_records() creates unique content hashes for each message."""
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

    cid, _, _ = prepare_records(
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


def test_prepare_records_with_attachments(test_conn, test_repository, tmp_path):
    """prepare_records() creates attachment records."""
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

    cid, counts, _ = prepare_records(
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


def test_prepare_records_returns_correct_tuple_structure(test_conn, test_repository, tmp_path):
    """prepare_records() returns (conversation_id, counts_dict, changed_bool)."""
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

    result = prepare_records(
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

"""Focused prepare_records integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from polylogue.pipeline.prepare import prepare_records
from polylogue.sources.parsers.base import ParsedAttachment, ParsedConversation, ParsedMessage


@pytest.fixture
async def async_backend(test_db):
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

    backend = SQLiteBackend(db_path=test_db)
    yield backend
    await backend.close()


@pytest.fixture
async def test_repository(async_backend):
    from polylogue.storage.repository import ConversationRepository

    return ConversationRepository(backend=async_backend)


async def test_prepare_records_new_conversation(async_backend, test_repository, tmp_path):
    conversation = ParsedConversation(
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
            )
        ],
        attachments=[],
    )

    conversation_id, counts, changed = await prepare_records(
        conversation,
        "test-source",
        archive_root=tmp_path / "archive",
        backend=async_backend,
        repository=test_repository,
    )

    assert counts["conversations"] == 1
    assert counts["messages"] == 1
    assert counts["skipped_conversations"] == 0
    assert changed is False
    assert conversation_id == "unknown:new-conv-1"


async def test_prepare_records_unchanged_conversation_skips(async_backend, test_repository, tmp_path):
    conversation = ParsedConversation(
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
            )
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    first_id, first_counts, _ = await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )
    second_id, second_counts, changed = await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )

    assert first_counts["conversations"] == 1
    assert second_id == first_id
    assert second_counts["conversations"] == 0
    assert changed is False


async def test_prepare_records_detects_changed_content(async_backend, test_repository, tmp_path):
    original = ParsedConversation(
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
            )
        ],
        attachments=[],
    )
    modified = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test Conversation",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="Modified text",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    first_id, _, _ = await prepare_records(
        original,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )
    second_id, _, changed = await prepare_records(
        modified,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )

    assert second_id == first_id
    assert changed is True


async def test_prepare_records_creates_message_records(async_backend, test_repository, tmp_path):
    conversation = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="msg-1", role="user", text="Hello", timestamp="2024-01-01T00:00:00Z"),
            ParsedMessage(provider_message_id="msg-2", role="assistant", text="Hi there", timestamp="2024-01-01T00:00:01Z"),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    conversation_id, counts, _ = await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )

    assert counts["conversations"] == 1
    assert counts["messages"] == 2
    async with async_backend.connection() as conn:
        rows = await (await conn.execute("SELECT message_id, role FROM messages WHERE conversation_id = ?", (conversation_id,))).fetchall()
    assert len(rows) == 2


async def test_prepare_records_handles_empty_provider_message_ids(async_backend, test_repository, tmp_path):
    conversation = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="", role="user", text="Hello", timestamp="2024-01-01T00:00:00Z"),
            ParsedMessage(provider_message_id="msg-explicit", role="assistant", text="Hi", timestamp="2024-01-01T00:00:01Z"),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    conversation_id, counts, _ = await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )

    assert counts["messages"] == 2
    async with async_backend.connection() as conn:
        rows = await (await conn.execute(
            "SELECT provider_message_id FROM messages WHERE conversation_id = ? ORDER BY rowid",
            (conversation_id,),
        )).fetchall()
    provider_ids = [row["provider_message_id"] for row in rows]
    assert "msg-1" in provider_ids
    assert "msg-explicit" in provider_ids


async def test_prepare_records_stores_source_metadata(async_backend, test_repository, tmp_path):
    import json

    conversation = ParsedConversation(
        provider_name="chatgpt",
        provider_conversation_id="ext-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="m1", role="user", text="Hi", timestamp="2024-01-01T00:00:00Z")
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    conversation_id, _, _ = await prepare_records(
        conversation,
        "my-export",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )

    async with async_backend.connection() as conn:
        row = await (await conn.execute(
            "SELECT provider_meta FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )).fetchone()
    assert row is not None
    meta = json.loads(row["provider_meta"]) if isinstance(row["provider_meta"], str) else row["provider_meta"]
    assert meta.get("source") == "my-export"


async def test_prepare_records_multiple_messages_get_unique_hashes(async_backend, test_repository, tmp_path):
    conversation = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="m1", role="user", text="First", timestamp="2024-01-01T00:00:00Z"),
            ParsedMessage(provider_message_id="m2", role="assistant", text="Second", timestamp="2024-01-01T00:00:01Z"),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    conversation_id, _, _ = await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )

    async with async_backend.connection() as conn:
        rows = await (await conn.execute(
            "SELECT content_hash FROM messages WHERE conversation_id = ? ORDER BY provider_message_id",
            (conversation_id,),
        )).fetchall()
    assert len(rows) == 2
    assert rows[0]["content_hash"] != rows[1]["content_hash"]


async def test_prepare_records_with_attachments(async_backend, test_repository, tmp_path):
    conversation = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="m1", role="user", text="See attachment", timestamp="2024-01-01T00:00:00Z")
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="m1",
                name="document.pdf",
                mime_type="application/pdf",
                size_bytes=2048,
            )
        ],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    conversation_id, counts, _ = await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    )

    assert counts["attachments"] == 1
    async with async_backend.connection() as conn:
        attachment_refs = await (await conn.execute(
            "SELECT ar.*, a.mime_type FROM attachment_refs ar "
            "JOIN attachments a ON ar.attachment_id = a.attachment_id "
            "WHERE ar.conversation_id = ?",
            (conversation_id,),
        )).fetchall()
    assert len(attachment_refs) == 1
    assert attachment_refs[0]["mime_type"] == "application/pdf"


async def test_prepare_records_rolls_back_attachment_move_on_save_failure(async_backend, test_repository, tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source_file = uploads / "document.pdf"
    source_file.write_bytes(b"pdf bytes")

    conversation = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-rollback",
        title="Rollback",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="m1", role="user", text="See attachment", timestamp="2024-01-01T00:00:00Z")
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="m1",
                name="document.pdf",
                mime_type="application/pdf",
                size_bytes=len(b"pdf bytes"),
                path=str(source_file),
            )
        ],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    with patch("polylogue.pipeline.prepare.save_bundle", new=AsyncMock(side_effect=RuntimeError("save failed"))):
        with pytest.raises(RuntimeError, match="save failed"):
            await prepare_records(
                conversation,
                "test-source",
                archive_root=archive_root,
                backend=async_backend,
                repository=test_repository,
            )

    assert source_file.exists()
    assert not any(path.is_file() for path in (archive_root / "assets").rglob("*"))


async def test_prepare_records_returns_correct_tuple_structure(async_backend, test_repository, tmp_path):
    conversation = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="m1", role="user", text="Hi", timestamp="2024-01-01T00:00:00Z")
        ],
        attachments=[],
    )

    result = await prepare_records(
        conversation,
        "test-source",
        archive_root=tmp_path / "archive",
        backend=async_backend,
        repository=test_repository,
    )

    assert isinstance(result, tuple)
    assert len(result) == 3
    conversation_id, counts, changed = result
    assert isinstance(conversation_id, str)
    assert isinstance(counts, dict)
    assert isinstance(changed, bool)
    required_keys = {
        "conversations",
        "messages",
        "attachments",
        "skipped_conversations",
        "skipped_messages",
        "skipped_attachments",
    }
    assert set(counts.keys()) >= required_keys

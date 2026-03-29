"""Focused prepare_records integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.pipeline.prepare import prepare_records
from polylogue.pipeline.services.validation import ValidationService
from polylogue.sources.parsers.base import ParsedAttachment, ParsedConversation, ParsedMessage


def _prepare_fields(result):
    return result.conversation_id, result.counts, result.content_changed


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

    conversation_id, counts, changed = _prepare_fields(await prepare_records(
        conversation,
        "test-source",
        archive_root=tmp_path / "archive",
        backend=async_backend,
        repository=test_repository,
    ))

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

    first_id, first_counts, _ = _prepare_fields(await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))
    second_id, second_counts, changed = _prepare_fields(await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))

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

    first_id, _, _ = _prepare_fields(await prepare_records(
        original,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))
    second_id, _, changed = _prepare_fields(await prepare_records(
        modified,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))

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

    conversation_id, counts, _ = _prepare_fields(await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))

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

    conversation_id, counts, _ = _prepare_fields(await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))

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

    conversation_id, _, _ = _prepare_fields(await prepare_records(
        conversation,
        "my-export",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))

    async with async_backend.connection() as conn:
        row = await (await conn.execute(
            "SELECT provider_meta FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )).fetchone()
    assert row is not None
    meta = json.loads(row["provider_meta"]) if isinstance(row["provider_meta"], str) else row["provider_meta"]
    assert meta.get("source") == "my-export"


async def test_prepare_records_backfills_structured_blocks_from_provider_meta(async_backend, test_repository, tmp_path):
    conversation = ParsedConversation(
        provider_name="gemini",
        provider_conversation_id="gemini-structured-1",
        title="Structured Gemini",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="assistant",
                text=None,
                timestamp="2024-01-01T00:00:00Z",
                provider_meta={
                    "content_blocks": [
                        {"type": "text", "text": "inline"},
                        {"type": "code", "text": "print('ok')", "language": "python"},
                        {"type": "tool_result", "text": "ok"},
                    ]
                },
            ),
            ParsedMessage(
                provider_message_id="msg-2",
                role="assistant",
                text=None,
                timestamp="2024-01-01T00:00:01Z",
                provider_meta={
                    "content_blocks": [{"type": "thinking", "text": "reasoning"}],
                    "reasoning_traces": [{"text": "reasoning", "token_count": 12, "provider": "gemini"}],
                },
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    conversation_id, counts, _ = _prepare_fields(await prepare_records(
        conversation,
        "drive-export",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))

    assert counts["messages"] == 2
    async with async_backend.connection() as conn:
        block_rows = await (
            await conn.execute(
                """
                SELECT m.provider_message_id, cb.type, cb.text
                FROM content_blocks cb
                JOIN messages m ON m.message_id = cb.message_id
                WHERE cb.conversation_id = ?
                ORDER BY m.provider_message_id, cb.block_index
                """,
                (conversation_id,),
            )
        ).fetchall()
        message_rows = await (
            await conn.execute(
                "SELECT provider_message_id, text, has_thinking FROM messages WHERE conversation_id = ? ORDER BY provider_message_id",
                (conversation_id,),
            )
        ).fetchall()

    assert [(row["provider_message_id"], row["type"], row["text"]) for row in block_rows] == [
        ("msg-1", "text", "inline"),
        ("msg-1", "code", "print('ok')"),
        ("msg-1", "tool_result", "ok"),
        ("msg-2", "thinking", "reasoning"),
    ]
    assert [(row["provider_message_id"], row["text"], row["has_thinking"]) for row in message_rows] == [
        ("msg-1", "inline\nprint('ok')\nok", 0),
        ("msg-2", "reasoning", 1),
    ]


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

    conversation_id, _, _ = _prepare_fields(await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))

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

    conversation_id, counts, _ = _prepare_fields(await prepare_records(
        conversation,
        "test-source",
        archive_root=archive_root,
        backend=async_backend,
        repository=test_repository,
    ))

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


async def test_prepare_records_returns_typed_result(async_backend, test_repository, tmp_path):
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

    assert isinstance(result.conversation_id, str)
    assert isinstance(result.counts, dict)
    assert isinstance(result.content_changed, bool)
    required_keys = {
        "conversations",
        "messages",
        "attachments",
        "skipped_conversations",
        "skipped_messages",
        "skipped_attachments",
    }
    assert set(result.counts.keys()) >= required_keys


# =====================================================================
# Merged from test_validation_service.py (record preparation/validation)
# =====================================================================


class TestValidationService:
    def test_validation_default_mode_is_strict(self, monkeypatch):
        monkeypatch.delenv("POLYLOGUE_SCHEMA_VALIDATION", raising=False)
        service = ValidationService(backend=MagicMock())
        assert service._schema_validation_mode() == "strict"

    async def test_validation_uses_all_record_samples_by_default(self, monkeypatch):
        from polylogue.schemas import ValidationResult
        from polylogue.storage.blob_store import get_blob_store

        raw_content = (
            b'{"type":"session_meta"}\n'
            b'{"type":"response_item","payload":{"type":"message"}}\n'
            b'{"record_type":"state"}'
        )
        blob_store = get_blob_store()
        raw_id, blob_size = blob_store.write_from_bytes(raw_content)

        raw_record = MagicMock(
            raw_id=raw_id,
            raw_content=raw_content,  # Keep for backwards compatibility in mocks
            provider_name="codex",
            source_path="/tmp/session.jsonl",
            blob_size=blob_size,
        )
        service = ValidationService(backend=MagicMock())
        service.repository = MagicMock()
        service.repository.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        service.repository.mark_raw_validated = AsyncMock()
        service.repository.mark_raw_parsed = AsyncMock()

        class _CapturingValidator:
            provider = "codex"

            def __init__(self):
                self.max_samples_seen = None

            def validation_samples(self, payload, max_samples=None):
                self.max_samples_seen = max_samples
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        capturing = _CapturingValidator()
        monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_payload", lambda *args, **kwargs: capturing)

        result = await service.validate_raw_ids(raw_ids=[raw_id])

        assert result.parseable_raw_ids == [raw_id]
        assert capturing.max_samples_seen is None

    async def test_validation_strict_detects_malformed_jsonl_beyond_large_prefix(self, monkeypatch):
        from polylogue.schemas import ValidationResult
        from polylogue.storage.blob_store import get_blob_store

        raw_content = (b'{"type":"session_meta"}\n' * 1024) + b"not json at all\n"
        blob_store = get_blob_store()
        raw_id, blob_size = blob_store.write_from_bytes(raw_content)

        raw_record = MagicMock(
            raw_id=raw_id,
            raw_content=raw_content,  # Keep for backwards compatibility in mocks
            provider_name="codex",
            source_path="/tmp/session.jsonl",
            blob_size=blob_size,
        )
        service = ValidationService(backend=MagicMock())
        service.repository = MagicMock()
        service.repository.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        service.repository.mark_raw_validated = AsyncMock()
        service.repository.mark_raw_parsed = AsyncMock()

        class _AlwaysValidValidator:
            provider = "codex"

            def validation_samples(self, payload, max_samples=16):
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "strict")
        monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_payload", lambda *args, **kwargs: _AlwaysValidValidator())

        result = await service.validate_raw_ids(raw_ids=[raw_id])

        assert result.counts["invalid"] == 1
        assert result.parseable_raw_ids == []
        kwargs = service.repository.mark_raw_validated.await_args.kwargs
        assert kwargs["status"] == "failed"
        assert "Malformed JSONL lines" in (kwargs.get("error") or "")

    async def test_validation_progress_callback_reports_counts(self, monkeypatch):
        from polylogue.schemas import ValidationResult
        from polylogue.storage.blob_store import get_blob_store

        blob_store = get_blob_store()
        raw_id_1, blob_size_1 = blob_store.write_from_bytes(b'{"id":"1","mapping":{}}')
        raw_id_2, blob_size_2 = blob_store.write_from_bytes(b'{"id":"2","mapping":{}}')

        raw_records = [
            MagicMock(raw_id=raw_id_1, raw_content=b'{"id":"1","mapping":{}}', provider_name="chatgpt", source_path="/tmp/a.json", blob_size=blob_size_1),
            MagicMock(raw_id=raw_id_2, raw_content=b'{"id":"2","mapping":{}}', provider_name="chatgpt", source_path="/tmp/b.json", blob_size=blob_size_2),
        ]
        service = ValidationService(backend=MagicMock())
        service.repository = MagicMock()
        service.repository.get_raw_conversations_batch = AsyncMock(return_value=raw_records)
        service.repository.mark_raw_validated = AsyncMock()
        callback = MagicMock()

        class _AlwaysValidValidator:
            provider = "chatgpt"

            def validation_samples(self, payload, max_samples=16):
                return [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_payload", lambda *args, **kwargs: _AlwaysValidValidator())
        await service.validate_raw_ids(raw_ids=[raw_id_1, raw_id_2], progress_callback=callback)

        callback.assert_any_call(0, desc="Validating: 0/2 raw")
        callback.assert_any_call(1, desc="Validating: 1/2 raw")
        callback.assert_any_call(1, desc="Validating: 2/2 raw")

    async def test_validation_persists_payload_provider_from_decoded_payload(self, monkeypatch):
        from polylogue.schemas import ValidationResult
        from polylogue.storage.blob_store import get_blob_store

        raw_content = b'[{"id":"conv-1","mapping":{}}]'
        blob_store = get_blob_store()
        raw_id, blob_size = blob_store.write_from_bytes(raw_content)

        raw_record = MagicMock(
            raw_id=raw_id,
            raw_content=raw_content,  # Keep for backwards compatibility in mocks
            provider_name="inbox-source",
            source_path="/tmp/conversations.json",
            payload_provider=None,
            blob_size=blob_size,
        )
        service = ValidationService(backend=MagicMock())
        service.repository = MagicMock()
        service.repository.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        service.repository.mark_raw_validated = AsyncMock()
        service.repository.mark_raw_parsed = AsyncMock()

        class _AlwaysValidValidator:
            provider = "chatgpt"

            def validation_samples(self, payload, max_samples=16):
                return [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_payload", lambda *args, **kwargs: _AlwaysValidValidator())
        result = await service.validate_raw_ids(raw_ids=[raw_id])

        assert result.parseable_raw_ids == [raw_id]
        kwargs = service.repository.mark_raw_validated.await_args.kwargs
        assert kwargs["provider"] == "chatgpt"
        assert kwargs["payload_provider"] == "chatgpt"

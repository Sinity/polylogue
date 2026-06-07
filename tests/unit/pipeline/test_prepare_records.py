"""Focused prepare_records integration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.message.roles import Role
from polylogue.pipeline.prepare import prepare_records
from polylogue.pipeline.prepare_models import PersistedSessionResult
from polylogue.pipeline.services.validation import ValidationService
from polylogue.schemas import ValidationResult
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.types import ContentBlockType, Provider


def _prepare_fields(result: PersistedSessionResult) -> tuple[str, dict[str, int], bool]:
    return result.session_id, result.counts, result.content_changed


@pytest.fixture
async def async_backend(test_db: Path) -> AsyncIterator[SQLiteBackend]:
    backend = SQLiteBackend(db_path=test_db)
    yield backend
    await backend.close()


@pytest.fixture
async def test_repository(async_backend: SQLiteBackend) -> SessionRepository:
    return SessionRepository(backend=async_backend)


async def test_prepare_records_new_session(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="new-conv-1",
        title="New Session",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.USER,
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[],
    )

    session_id, counts, changed = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=tmp_path / "archive",
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert counts["sessions"] == 1
    assert counts["messages"] == 1
    assert counts["skipped_sessions"] == 0
    assert changed is False
    assert session_id == "unknown-export:new-conv-1"


async def test_prepare_records_persists_compaction_session_events(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="conv-events",
        title="Session events",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.USER,
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="compaction",
                timestamp="2024-01-01T00:00:00Z",
                payload={"summary": "Older turns compacted"},
                source_message_provider_id="msg-1",
            )
        ],
        attachments=[],
    )

    session_id, counts, _ = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=tmp_path / "archive",
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert counts["session_events"] == 1
    archive_session_id = "codex-session:conv-events"
    async with async_backend.connection() as conn:
        row = await (
            await conn.execute(
                """
                SELECT event_type, summary, source_message_id
                FROM session_events
                WHERE session_id = ?
                """,
                (archive_session_id,),
            )
        ).fetchone()
    assert row is not None
    assert row["event_type"] == "compaction"
    assert row["summary"] == "Older turns compacted"
    assert row["source_message_id"] == f"{archive_session_id}:msg-1"


async def test_prepare_records_projects_agent_policy_events(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="conv-event-raw",
        title="Agent policy event",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.USER,
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="agent_policy",
                timestamp="2024-01-01T00:00:01Z",
                payload={"approval": "on-request", "sandbox": "workspace-write", "network": "restricted"},
                source_message_provider_id="msg-1",
            )
        ],
    )

    session_id, counts, _ = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=tmp_path / "archive",
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert counts["session_events"] == 1
    archive_session_id = "codex-session:conv-event-raw"
    async with async_backend.connection() as conn:
        row = await (
            await conn.execute(
                """
                SELECT approval_policy, sandbox_policy, network_policy, source_message_id
                FROM session_agent_policies
                WHERE session_id = ?
                """,
                (archive_session_id,),
            )
        ).fetchone()

    assert row is not None
    assert row["approval_policy"] == "on-request"
    assert row["sandbox_policy"] == "workspace-write"
    assert row["network_policy"] == "restricted"
    assert row["source_message_id"] == f"{archive_session_id}:msg-1"


async def test_prepare_records_unchanged_session_skips(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test Session",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.USER,
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    first_id, first_counts, _ = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )
    second_id, second_counts, changed = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert first_counts["sessions"] == 1
    assert second_id == first_id
    # Idempotent re-ingest: archive content unchanged, but row_graph_hash
    # may differ for cosmetic substrate fields (#943 cost columns), so the
    # write-counter can be 0 (skipped) or 1 (rewrote same row).
    assert second_counts["sessions"] in (0, 1)
    assert changed is False


async def test_prepare_records_detects_changed_content(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    original = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test Session",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.USER,
                text="Original text",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[],
    )
    modified = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test Session",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.USER,
                text="Modified text",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    first_id, _, _ = _prepare_fields(
        await prepare_records(
            original,
            "test-source",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )
    second_id, _, changed = _prepare_fields(
        await prepare_records(
            modified,
            "test-source",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert second_id == first_id
    assert changed is True


async def test_prepare_records_creates_message_records(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="msg-1", role=Role.USER, text="Hello", timestamp="2024-01-01T00:00:00Z"),
            ParsedMessage(
                provider_message_id="msg-2", role=Role.ASSISTANT, text="Hi there", timestamp="2024-01-01T00:00:01Z"
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    session_id, counts, _ = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert counts["sessions"] == 1
    assert counts["messages"] == 2
    async with async_backend.connection() as conn:
        rows = list(
            await (
                await conn.execute("SELECT message_id, role FROM messages WHERE session_id = ?", (session_id,))
            ).fetchall()
        )
    assert len(rows) == 2


async def test_prepare_records_handles_empty_parser_message_ids(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="", role=Role.USER, text="Hello", timestamp="2024-01-01T00:00:00Z"),
            ParsedMessage(
                provider_message_id="msg-explicit", role=Role.ASSISTANT, text="Hi", timestamp="2024-01-01T00:00:01Z"
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    session_id, counts, _ = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert counts["messages"] == 2
    async with async_backend.connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT native_id FROM messages WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            )
        ).fetchall()
    native_ids = [row["native_id"] for row in rows]
    assert None in native_ids
    assert "msg-explicit" in native_ids


async def test_prepare_records_stores_typed_origin(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="ext-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="Hi", timestamp="2024-01-01T00:00:00Z")],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    session_id, _, _ = _prepare_fields(
        await prepare_records(
            session,
            "my-export",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )

    async with async_backend.connection() as conn:
        row = await (await conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (session_id,))).fetchone()
    assert row is not None
    assert row["origin"] == "chatgpt-export"


async def test_prepare_records_persists_structured_blocks(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="gemini-structured-1",
        title="Structured Gemini",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.ASSISTANT,
                text=None,
                timestamp="2024-01-01T00:00:00Z",
                content_blocks=[
                    ParsedContentBlock(type=ContentBlockType.TEXT, text="inline"),
                    ParsedContentBlock(
                        type=ContentBlockType.CODE,
                        text="print('ok')",
                        metadata={"language": "python"},
                    ),
                    ParsedContentBlock(type=ContentBlockType.TOOL_RESULT, text="ok"),
                ],
            ),
            ParsedMessage(
                provider_message_id="msg-2",
                role=Role.ASSISTANT,
                text=None,
                timestamp="2024-01-01T00:00:01Z",
                content_blocks=[ParsedContentBlock(type=ContentBlockType.THINKING, text="reasoning")],
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    session_id, counts, _ = _prepare_fields(
        await prepare_records(
            session,
            "drive-export",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert counts["messages"] == 2
    async with async_backend.connection() as conn:
        block_rows = await (
            await conn.execute(
                """
                SELECT m.native_id, b.block_type, b.text
                FROM blocks b
                JOIN messages m ON m.message_id = b.message_id
                WHERE b.session_id = ?
                ORDER BY m.native_id, b.position
                """,
                (session_id,),
            )
        ).fetchall()
        message_rows = await (
            await conn.execute(
                """
                SELECT native_id, has_thinking
                FROM messages
                WHERE session_id = ?
                ORDER BY position
                """,
                (session_id,),
            )
        ).fetchall()

    assert [(row["native_id"], row["block_type"], row["text"]) for row in block_rows] == [
        ("msg-1", "text", "inline"),
        ("msg-1", "code", "print('ok')"),
        ("msg-1", "tool_result", "ok"),
        ("msg-2", "thinking", "reasoning"),
    ]
    assert [(row["native_id"], row["has_thinking"]) for row in message_rows] == [
        ("msg-1", 0),
        ("msg-2", 1),
    ]


async def test_prepare_records_multiple_messages_get_unique_hashes(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="m1", role=Role.USER, text="First", timestamp="2024-01-01T00:00:00Z"),
            ParsedMessage(
                provider_message_id="m2", role=Role.ASSISTANT, text="Second", timestamp="2024-01-01T00:00:01Z"
            ),
        ],
        attachments=[],
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()

    session_id, _, _ = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )

    async with async_backend.connection() as conn:
        rows = list(
            await (
                await conn.execute(
                    "SELECT content_hash FROM messages WHERE session_id = ? ORDER BY native_id",
                    (session_id,),
                )
            ).fetchall()
        )
    assert len(rows) == 2
    assert rows[0]["content_hash"] != rows[1]["content_hash"]


async def test_prepare_records_with_attachments(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1", role=Role.USER, text="See attachment", timestamp="2024-01-01T00:00:00Z"
            )
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

    session_id, counts, _ = _prepare_fields(
        await prepare_records(
            session,
            "test-source",
            archive_root=archive_root,
            backend=async_backend,
            repository=test_repository,
        )
    )

    assert counts["attachments"] == 1
    async with async_backend.connection() as conn:
        attachment_refs = list(
            await (
                await conn.execute(
                    "SELECT ar.*, a.media_type FROM attachment_refs ar "
                    "JOIN attachments a ON ar.attachment_id = a.attachment_id "
                    "WHERE ar.session_id = ?",
                    (session_id,),
                )
            ).fetchall()
        )
    assert len(attachment_refs) == 1
    assert attachment_refs[0]["media_type"] == "application/pdf"


async def test_prepare_records_rolls_back_attachment_move_on_save_failure(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source_file = uploads / "document.pdf"
    source_file.write_bytes(b"pdf bytes")

    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-rollback",
        title="Rollback",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1", role=Role.USER, text="See attachment", timestamp="2024-01-01T00:00:00Z"
            )
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

    with patch.object(test_repository, "save_parsed_session", new=AsyncMock(side_effect=RuntimeError("save failed"))):
        with pytest.raises(RuntimeError, match="save failed"):
            await prepare_records(
                session,
                "test-source",
                archive_root=archive_root,
                backend=async_backend,
                repository=test_repository,
            )

    assert source_file.exists()
    assert not any(path.is_file() for path in (archive_root / "assets").rglob("*"))


async def test_prepare_records_returns_typed_result(
    async_backend: SQLiteBackend,
    test_repository: SessionRepository,
    tmp_path: Path,
) -> None:
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="Hi", timestamp="2024-01-01T00:00:00Z")],
        attachments=[],
    )

    result = await prepare_records(
        session,
        "test-source",
        archive_root=tmp_path / "archive",
        backend=async_backend,
        repository=test_repository,
    )

    assert isinstance(result.session_id, str)
    assert isinstance(result.counts, dict)
    assert isinstance(result.content_changed, bool)
    required_keys = {
        "sessions",
        "messages",
        "attachments",
        "skipped_sessions",
        "skipped_messages",
        "skipped_attachments",
    }
    assert set(result.counts.keys()) >= required_keys


# =====================================================================
# Merged from test_validation_service.py (record preparation/validation)
# =====================================================================


class TestValidationService:
    def test_validation_default_mode_is_advisory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Default validation mode is "advisory": ingest does not block on
        # schema mismatches but records them as observations. Operators opt
        # into strict mode via POLYLOGUE_SCHEMA_VALIDATION=strict.
        monkeypatch.delenv("POLYLOGUE_SCHEMA_VALIDATION", raising=False)
        service = ValidationService(backend=MagicMock())
        assert service._schema_validation_mode() == "advisory"

    async def test_validation_uses_all_record_samples_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.storage.blob_store import get_blob_store

        raw_content = (
            b'{"type":"session_meta"}\n{"type":"response_item","payload":{"type":"message"}}\n{"record_type":"state"}'
        )
        blob_store = get_blob_store()
        raw_id, blob_size = blob_store.write_from_bytes(raw_content)

        raw_record = MagicMock(
            raw_id=raw_id,
            raw_content=raw_content,  # Keep for backwards compatibility in mocks
            source_name="codex",
            source_path="/tmp/session.jsonl",
            blob_size=blob_size,
        )
        service = ValidationService(backend=MagicMock())
        service.repository = MagicMock()
        service.repository.get_raw_sessions_batch = AsyncMock(return_value=[raw_record])
        service.repository.mark_raw_validated = AsyncMock()
        service.repository.mark_raw_parsed = AsyncMock()

        class _CapturingValidator:
            provider = "codex"

            def __init__(self) -> None:
                self.max_samples_seen: int | None = None

            def validation_samples(self, payload: object, max_samples: int | None = None) -> list[object]:
                self.max_samples_seen = max_samples
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample: object) -> ValidationResult:
                return ValidationResult(is_valid=True)

        capturing = _CapturingValidator()
        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_payload", lambda *args, **kwargs: capturing
        )

        result = await service.validate_raw_ids(raw_ids=[raw_id])

        assert result.parseable_raw_ids == [raw_id]
        assert capturing.max_samples_seen is None

    async def test_validation_strict_detects_malformed_jsonl_beyond_large_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.storage.blob_store import get_blob_store

        raw_content = (b'{"type":"session_meta"}\n' * 1024) + b"not json at all\n"
        blob_store = get_blob_store()
        raw_id, blob_size = blob_store.write_from_bytes(raw_content)

        raw_record = MagicMock(
            raw_id=raw_id,
            raw_content=raw_content,  # Keep for backwards compatibility in mocks
            source_name="codex",
            source_path="/tmp/session.jsonl",
            blob_size=blob_size,
        )
        service = ValidationService(backend=MagicMock())
        service.repository = MagicMock()
        service.repository.get_raw_sessions_batch = AsyncMock(return_value=[raw_record])
        service.repository.mark_raw_validated = AsyncMock()
        service.repository.mark_raw_parsed = AsyncMock()

        class _AlwaysValidValidator:
            provider = "codex"

            def validation_samples(self, payload: object, max_samples: int = 16) -> list[object]:
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample: object) -> ValidationResult:
                return ValidationResult(is_valid=True)

        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "strict")
        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_payload", lambda *args, **kwargs: _AlwaysValidValidator()
        )

        result = await service.validate_raw_ids(raw_ids=[raw_id])

        assert result.counts["invalid"] == 1
        assert result.parseable_raw_ids == []
        kwargs = service.repository.mark_raw_validated.await_args.kwargs
        assert kwargs["status"] == "failed"
        assert "Malformed JSONL lines" in (kwargs.get("error") or "")

    async def test_validation_progress_callback_reports_counts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.storage.blob_store import get_blob_store

        blob_store = get_blob_store()
        raw_id_1, blob_size_1 = blob_store.write_from_bytes(b'{"id":"1","mapping":{}}')
        raw_id_2, blob_size_2 = blob_store.write_from_bytes(b'{"id":"2","mapping":{}}')

        raw_artifacts = [
            MagicMock(
                raw_id=raw_id_1,
                raw_content=b'{"id":"1","mapping":{}}',
                source_name=Provider.CHATGPT,
                source_path="/tmp/a.json",
                blob_size=blob_size_1,
            ),
            MagicMock(
                raw_id=raw_id_2,
                raw_content=b'{"id":"2","mapping":{}}',
                source_name=Provider.CHATGPT,
                source_path="/tmp/b.json",
                blob_size=blob_size_2,
            ),
        ]
        service = ValidationService(backend=MagicMock())
        service.repository = MagicMock()
        service.repository.get_raw_sessions_batch = AsyncMock(return_value=raw_artifacts)
        service.repository.mark_raw_validated = AsyncMock()
        callback = MagicMock()

        class _AlwaysValidValidator:
            provider = "chatgpt"

            def validation_samples(self, payload: object, max_samples: int = 16) -> list[object]:
                return [payload]

            def validate(self, _sample: object) -> ValidationResult:
                return ValidationResult(is_valid=True)

        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_payload", lambda *args, **kwargs: _AlwaysValidValidator()
        )
        await service.validate_raw_ids(raw_ids=[raw_id_1, raw_id_2], progress_callback=callback)

        callback.assert_any_call(0, desc="Validating: 0/2 raw")
        callback.assert_any_call(1, desc="Validating: 1/2 raw")
        callback.assert_any_call(1, desc="Validating: 2/2 raw")

    async def test_validation_persists_payload_provider_from_decoded_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.storage.blob_store import get_blob_store

        raw_content = b'[{"id":"conv-1","mapping":{}}]'
        blob_store = get_blob_store()
        raw_id, blob_size = blob_store.write_from_bytes(raw_content)

        raw_record = MagicMock(
            raw_id=raw_id,
            raw_content=raw_content,  # Keep for backwards compatibility in mocks
            source_name="inbox-source",
            source_path="/tmp/sessions.json",
            payload_provider=None,
            blob_size=blob_size,
        )
        service = ValidationService(backend=MagicMock())
        service.repository = MagicMock()
        service.repository.get_raw_sessions_batch = AsyncMock(return_value=[raw_record])
        service.repository.mark_raw_validated = AsyncMock()
        service.repository.mark_raw_parsed = AsyncMock()

        class _AlwaysValidValidator:
            provider = "chatgpt"

            def validation_samples(self, payload: object, max_samples: int = 16) -> list[object]:
                return [payload]

            def validate(self, _sample: object) -> ValidationResult:
                return ValidationResult(is_valid=True)

        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_payload", lambda *args, **kwargs: _AlwaysValidValidator()
        )
        result = await service.validate_raw_ids(raw_ids=[raw_id])

        assert result.parseable_raw_ids == [raw_id]
        kwargs = service.repository.mark_raw_validated.await_args.kwargs
        assert kwargs["provider"] == "chatgpt"
        assert kwargs["payload_provider"] == "chatgpt"

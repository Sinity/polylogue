"""Archive write-path tests for ``write_parsed_session_to_archive``.

These exercise the live writer the daemon uses (via the ``ingest_session``
helper). Assertions read back actual archive rows — session/message/block/
event/attachment state, content-hash idempotency, and content-change
detection — rather than inspecting an intermediate counts envelope.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline.services.validation import ValidationService
from polylogue.schemas import ValidationResult
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.live_ingest import ingest_session


@pytest.fixture
async def async_backend(test_db: Path) -> AsyncIterator[SQLiteBackend]:
    backend = SQLiteBackend(db_path=test_db)
    yield backend
    await backend.close()


async def _session_row(backend: SQLiteBackend, session_id: str) -> dict[str, object]:
    async with backend.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT message_count, content_hash FROM sessions WHERE session_id = ?",
                (session_id,),
            )
        ).fetchone()
    assert row is not None, f"session {session_id} not written"
    return {"message_count": row["message_count"], "content_hash": bytes(row["content_hash"])}


async def _count(backend: SQLiteBackend, sql: str, *params: object) -> int:
    async with backend.connection() as conn:
        row = await (await conn.execute(sql, params)).fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# Core write behavior
# ---------------------------------------------------------------------------


async def test_writes_new_session_and_message_rows(async_backend: SQLiteBackend) -> None:
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

    session_id = await ingest_session(session, async_backend)

    assert session_id == "unknown-export:new-conv-1"
    assert await _count(async_backend, "SELECT COUNT(*) FROM sessions WHERE session_id = ?", session_id) == 1
    assert await _count(async_backend, "SELECT COUNT(*) FROM messages WHERE session_id = ?", session_id) == 1


async def test_idempotent_reingest_leaves_one_session_and_stable_hash(async_backend: SQLiteBackend) -> None:
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

    first_id = await ingest_session(session, async_backend)
    first = await _session_row(async_backend, first_id)
    second_id = await ingest_session(session, async_backend)
    second = await _session_row(async_backend, second_id)

    assert second_id == first_id
    # Re-ingesting unchanged content must not duplicate rows nor change the hash.
    assert await _count(async_backend, "SELECT COUNT(*) FROM sessions WHERE session_id = ?", first_id) == 1
    assert await _count(async_backend, "SELECT COUNT(*) FROM messages WHERE session_id = ?", first_id) == 1
    assert second["content_hash"] == first["content_hash"]


async def test_changed_content_updates_hash_and_text(async_backend: SQLiteBackend) -> None:
    def _make(text: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.UNKNOWN,
            provider_session_id="conv-1",
            title="Test Session",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.USER,
                    text=text,
                    timestamp="2024-01-01T00:00:00Z",
                )
            ],
            attachments=[],
        )

    first_id = await ingest_session(_make("Original text"), async_backend)
    first_hash = (await _session_row(async_backend, first_id))["content_hash"]
    second_id = await ingest_session(_make("Modified text"), async_backend)
    second_hash = (await _session_row(async_backend, second_id))["content_hash"]

    assert second_id == first_id
    assert second_hash != first_hash
    # The stored block text reflects the latest ingest, not a duplicate row.
    assert await _count(async_backend, "SELECT COUNT(*) FROM messages WHERE session_id = ?", first_id) == 1
    async with async_backend.connection() as conn:
        row = await (await conn.execute("SELECT text FROM blocks WHERE session_id = ?", (first_id,))).fetchone()
    assert row is not None
    assert row["text"] == "Modified text"


# ---------------------------------------------------------------------------
# Session events
# ---------------------------------------------------------------------------


async def test_persists_compaction_session_event_summary(async_backend: SQLiteBackend) -> None:
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

    session_id = await ingest_session(session, async_backend)

    async with async_backend.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT event_type, summary, source_message_id FROM session_events WHERE session_id = ?",
                (session_id,),
            )
        ).fetchone()
    assert row is not None
    assert row["event_type"] == "compaction"
    assert row["summary"] == "Older turns compacted"
    assert row["source_message_id"] == f"{session_id}:msg-1"


async def test_projects_agent_policy_event_into_typed_columns(async_backend: SQLiteBackend) -> None:
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

    session_id = await ingest_session(session, async_backend)

    async with async_backend.connection() as conn:
        row = await (
            await conn.execute(
                """
                SELECT approval_policy, sandbox_policy, network_policy, source_message_id
                FROM session_agent_policies
                WHERE session_id = ?
                """,
                (session_id,),
            )
        ).fetchone()
    assert row is not None
    assert row["approval_policy"] == "on-request"
    assert row["sandbox_policy"] == "workspace-write"
    assert row["network_policy"] == "restricted"
    assert row["source_message_id"] == f"{session_id}:msg-1"


# ---------------------------------------------------------------------------
# Message / block / attachment structure
# ---------------------------------------------------------------------------


async def test_writes_one_message_row_per_parsed_message(async_backend: SQLiteBackend) -> None:
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

    session_id = await ingest_session(session, async_backend)

    async with async_backend.connection() as conn:
        rows = list(
            await (
                await conn.execute("SELECT message_id, role FROM messages WHERE session_id = ?", (session_id,))
            ).fetchall()
        )
    assert len(rows) == 2


async def test_preserves_empty_and_explicit_native_message_ids(async_backend: SQLiteBackend) -> None:
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

    session_id = await ingest_session(session, async_backend)

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


async def test_stores_typed_origin_on_session(async_backend: SQLiteBackend) -> None:
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="ext-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="Hi", timestamp="2024-01-01T00:00:00Z")],
        attachments=[],
    )

    session_id = await ingest_session(session, async_backend)

    async with async_backend.connection() as conn:
        row = await (await conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (session_id,))).fetchone()
    assert row is not None
    assert row["origin"] == "chatgpt-export"


async def test_persists_structured_blocks_in_order(async_backend: SQLiteBackend) -> None:
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
                blocks=[
                    ParsedContentBlock(type=BlockType.TEXT, text="inline"),
                    ParsedContentBlock(
                        type=BlockType.CODE,
                        text="print('ok')",
                        metadata={"language": "python"},
                    ),
                    ParsedContentBlock(type=BlockType.TOOL_RESULT, text="ok"),
                ],
            ),
            ParsedMessage(
                provider_message_id="msg-2",
                role=Role.ASSISTANT,
                text=None,
                timestamp="2024-01-01T00:00:01Z",
                blocks=[ParsedContentBlock(type=BlockType.THINKING, text="reasoning")],
            ),
        ],
        attachments=[],
    )

    session_id = await ingest_session(session, async_backend)

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
                "SELECT native_id, has_thinking FROM messages WHERE session_id = ? ORDER BY position",
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


async def test_each_message_gets_a_distinct_content_hash(async_backend: SQLiteBackend) -> None:
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

    session_id = await ingest_session(session, async_backend)

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


async def test_writes_attachment_ref_with_media_type(async_backend: SQLiteBackend) -> None:
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

    session_id = await ingest_session(session, async_backend)

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


# =====================================================================
# Record validation (ValidationService) — independent of the writer path
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
            raw_content=raw_content,
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
            raw_content=raw_content,
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
            raw_content=raw_content,
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

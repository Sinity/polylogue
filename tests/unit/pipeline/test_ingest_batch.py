"""Focused tests for sync ingest-batch DB writes."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeAlias, cast
from unittest.mock import AsyncMock

import aiosqlite
import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_INGEST_FLAG
from polylogue.archive.message.roles import Role
from polylogue.config import Config
from polylogue.core.enums import ArtifactSupportStatus, BlockType, Origin, Provider
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.types import SessionId
from polylogue.daemon.status import RawFailureSample, raw_failure_info_for_root
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.services import ingest_worker as ingest_worker_mod
from polylogue.pipeline.services.ingest_batch import (
    _build_batch_memory_observation,
    _drain_ready_session_entries,
    _failed_raw_state_update,
    _IngestBatchSummary,
    _persist_batch_raw_state_updates,
    _RawIngestOutcome,
    _successful_raw_state_update,
    _topo_sort_session_entries,
    _unattributed_batch_elapsed_s,
)
from polylogue.pipeline.services.ingest_batch._observations import _build_parse_batch_observation
from polylogue.pipeline.services.ingest_worker import (
    IngestRecordResult,
    SessionWritePayload,
)
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.sinex.models import PublicationMode
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.sources.revision_backfill import PreparedRevisionReplayResult
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.repository import SessionRepository
from polylogue.storage.search.cache import get_cache_stats
from polylogue.storage.search.runtime import search_messages
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    upsert_raw_artifact,
    write_source_raw_session,
)
from polylogue.storage.sqlite.archive_tiers.write import _attachment_id
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection
from polylogue.storage.sqlite.write_lease import UnleasedWriteError, arm_write_lease_enforcement, write_lease
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.index_writer import (
    fixture_index_mutation_scope,
    write_fixture_index_session,
    write_fixture_ingest_payload,
)
from tests.infra.live_ingest import prepared_live_convergence_owner

BlockSpec: TypeAlias = tuple[str, ParsedContentBlock]
AttachmentRefSpec: TypeAlias = tuple[str, str]
_write_session = ingest_batch_core._write_session


def _float_value(value: object) -> float:
    if not isinstance(value, (float, int, str)):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def test_worker_normalization_preserves_raw_row_archive_origin() -> None:
    parsed = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="a6216592-1a36-4748-9198-64b8db6ec05b",
        title="Session",
        messages=[],
    )

    normalized = ingest_worker_mod._normalized_session(
        parsed,
        fallback_timestamp=None,
    )

    assert str(make_session_id("claude-code-session", normalized.provider_session_id)) == (
        "claude-code-session:a6216592-1a36-4748-9198-64b8db6ec05b"
    )
    assert normalized.source_name == Provider.CLAUDE_CODE
    assert parsed.source_name == Provider.CLAUDE_CODE


def test_worker_normalization_replaces_malformed_session_timestamp_with_message_evidence() -> None:
    parsed = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="malformed-session-time",
        created_at="not-a-timestamp",
        updated_at="also-not-a-timestamp",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="evidence",
                timestamp="2026-06-01T12:00:00Z",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="evidence")],
            )
        ],
    )

    normalized = ingest_worker_mod._normalized_session(parsed, fallback_timestamp="2020-01-01T00:00:00Z")

    assert normalized.created_at == "2026-06-01T12:00:00+00:00"
    assert normalized.updated_at == "2026-06-01T12:00:00+00:00"


def test_stale_observation_repair_derives_created_time_from_session_event(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    conn = ingest_batch_core._open_sync_connection(archive_root / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="stale-event-observation",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="summary",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="summary")],
                )
            ],
        )
        session_id = write_fixture_index_session(conn, session)
        candidate = session.model_copy(
            update={
                "session_events": [
                    ParsedSessionEvent(
                        event_type="hermes_llm_request_span",
                        timestamp="2026-07-01T00:00:00Z",
                    )
                ]
            }
        )
        payload = SessionWritePayload(
            session_id=session_id,
            content_hash="ignored",
            parsed_session=candidate,
            fallback_timestamp="2020-01-01T00:00:00Z",
        )

        with fixture_index_mutation_scope(conn):
            ingest_batch_core._retain_stale_revision_observations(conn, payload)
        row = conn.execute("SELECT created_at_ms FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    finally:
        conn.close()

    assert row[0] == 1_782_864_000_000


def test_parse_batch_observation_reports_unsupported_write_mode() -> None:
    summary = _IngestBatchSummary()
    summary.attachment_owner_resolutions.append(
        {
            "raw_id": "raw-owner-test",
            "session_id": "gemini:owner-test",
            "attachment_id": "attachment-owner-test",
            "reason": "owner_ambiguous",
        }
    )

    observation = _build_parse_batch_observation(
        batch_summary=summary,
        elapsed_s=0.25,
        raw_state_update_elapsed_s=0.0,
        rss_start_mb=None,
        rss_end_mb=None,
        peak_rss_self_start_mb=None,
        peak_rss_self_end_mb=None,
        peak_rss_children_mb=None,
    )

    assert observation["primary_ingest_store"] == "archive_file_set"
    assert observation["archive_primary_write"] is False
    assert observation["archive_write_mode"] == "unsupported"
    assert "archive_sync_target" not in observation
    assert "archive_sync_elapsed_ms" not in observation
    assert observation["attachment_owner_resolutions"] == [
        {
            "raw_id": "raw-owner-test",
            "session_id": "gemini:owner-test",
            "attachment_id": "attachment-owner-test",
            "reason": "owner_ambiguous",
        }
    ]


def test_batch_writer_carries_typed_attachment_owner_resolution(tmp_path: Path) -> None:
    """The normal acquisition writer does not discard unresolved-owner evidence."""
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    conn = ingest_batch_core._open_sync_connection(archive_root / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.GEMINI,
            provider_session_id="batch-owner-receipt",
            messages=[
                ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same"),
                ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same"),
            ],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="batch-ambiguous",
                    message_position=0,
                    name="ambiguous.txt",
                    mime_type="text/plain",
                )
            ],
        )
        payload = SessionWritePayload(
            session_id=str(make_session_id(session.source_name, session.provider_session_id)),
            content_hash=str(session_content_hash(session)),
            parsed_session=session,
            raw_id="raw-batch-owner-receipt",
        )
        summary = _IngestBatchSummary()

        assert ingest_batch_core._write_session_entry(conn, "raw-batch-owner-receipt", payload, summary=summary)
        assert summary.attachment_owner_resolutions == [
            {
                "raw_id": "raw-batch-owner-receipt",
                "session_id": payload.session_id,
                "attachment_id": _attachment_id(payload.session_id, session.attachments[0]),
                "reason": "owner_ambiguous",
            }
        ]
    finally:
        conn.close()


def test_hash_unchanged_batch_write_still_reports_owner_resolutions(tmp_path: Path) -> None:
    """Replaying an unchanged session reports the same unresolved owners as its write.

    Anti-vacuity: the hash-unchanged return in ``_write_session`` used to skip
    the writer and report nothing, so replaying an orphan population reported
    zero ``attachment_owner_resolutions``.
    """
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    conn = ingest_batch_core._open_sync_connection(archive_root / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.GEMINI,
            provider_session_id="batch-unchanged-owner",
            messages=[
                ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same"),
                ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same"),
            ],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="batch-unchanged-ambiguous",
                    message_position=0,
                    name="ambiguous.txt",
                    mime_type="text/plain",
                )
            ],
        )
        payload = SessionWritePayload(
            session_id=str(make_session_id(session.source_name, session.provider_session_id)),
            content_hash=str(session_content_hash(session)),
            parsed_session=session,
            raw_id="raw-batch-unchanged-owner",
        )
        first = _IngestBatchSummary()
        assert ingest_batch_core._write_session_entry(conn, "raw-batch-unchanged-owner", payload, summary=first)
        conn.commit()
        replay = _IngestBatchSummary()
        ingest_batch_core._write_session_entry(conn, "raw-batch-unchanged-owner", payload, summary=replay)

        assert first.attachment_owner_resolutions
        assert replay.attachment_owner_resolutions == first.attachment_owner_resolutions
    finally:
        conn.close()


def test_sync_index_connection_ensures_runtime_indexes(tmp_path: Path) -> None:
    conn = ingest_batch_core._open_sync_connection(tmp_path / "archive" / "index.db")
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_messages_active_leaf'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None


def test_sync_ingest_index_publication_refuses_a_foreign_archive_lease(tmp_path: Path) -> None:
    """Index generations cannot borrow writer authority from another archive.

    Anti-vacuity: removing the explicit ``archive_root`` from
    ``_open_sync_connection`` lets this real ingest publication opener create
    a connection to the target tier under the owner archive's lease.
    """
    owner_root = tmp_path / "owner"
    target_root = tmp_path / "target"
    owner_root.mkdir()
    bootstrap_archive_root(target_root)

    with (
        arm_write_lease_enforcement(),
        write_lease("test.owner", archive_root=owner_root),
        pytest.raises(UnleasedWriteError, match="outside the archive"),
    ):
        ingest_batch_core._open_sync_connection(target_root / "index.db", archive_root=target_root)


def test_batch_projection_preserves_worker_disposition_fields() -> None:
    summary = _IngestBatchSummary()
    ingest_batch_core._record_outcome(
        summary,
        IngestRecordResult(
            raw_id="raw-worker-failure",
            error="schema rejected",
            outcome_code="validation_rejected",
            retryable=False,
            evidence_ref="schema_validation_strict",
            remediation="repair source schema",
            diagnostic="missing required field: messages",
        ),
    )

    outcome = summary.outcomes["raw-worker-failure"]
    assert outcome.outcome_code == "validation_rejected"
    assert outcome.retryable is False
    assert outcome.evidence_ref == "schema_validation_strict"
    assert outcome.remediation == "repair source schema"
    assert outcome.diagnostic == "missing required field: messages"


class _FakeConnectionBackend:
    def __init__(self, connection: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]) -> None:
        self._connection = connection

    def connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]:
        return self._connection()


class _FakeBulkBackend:
    def __init__(self, connection: Callable[[], AbstractAsyncContextManager[None]]) -> None:
        self._connection = connection

    def bulk_connection(self) -> AbstractAsyncContextManager[None]:
        return self._connection()


class _FakeRawStateRepository:
    def __init__(self, update_raw_state: AsyncMock) -> None:
        self._update_raw_state = update_raw_state

    @property
    def source_backend(self) -> None:
        return None

    async def update_raw_state(self, raw_id: str, *, state: RawSessionStateUpdate) -> object:
        return await self._update_raw_state(raw_id, state=state)


class _FakeParsingService:
    def __init__(self, update_raw_state: AsyncMock) -> None:
        self._repository = _FakeRawStateRepository(update_raw_state)

    @property
    def repository(self) -> _FakeRawStateRepository:
        return self._repository


class _FakeRefreshConnection(aiosqlite.Connection):
    def __init__(self) -> None:
        self._connection = None
        self.commit_mock = AsyncMock()

    async def commit(self) -> None:
        await self.commit_mock()


def _session_data(
    session_id: str,
    *,
    content_hash: str,
    raw_id: str | None = None,
    parent_session_id: str | None = None,
    message_tuples: list[ParsedMessage] | None = None,
    block_tuples: list[BlockSpec] | None = None,
    action_tuples: list[ParsedSessionEvent] | None = None,
    stats_tuple: object | None = None,
    attachment_tuples: list[ParsedAttachment] | None = None,
    attachment_ref_tuples: list[AttachmentRefSpec] | None = None,
    ingest_flags: list[str] | None = None,
    provider: Provider = Provider.CODEX,
    title: str = "Session",
    append_only: bool = False,
    created_at: str = "2026-04-02T00:00:00Z",
    updated_at: str = "2026-04-02T00:00:00Z",
) -> SessionWritePayload:
    del stats_tuple
    messages = list(message_tuples or [])
    blocks_by_message: dict[str, list[ParsedContentBlock]] = {}
    for message_id, block in block_tuples or []:
        blocks_by_message.setdefault(message_id, []).append(block)
    if blocks_by_message:
        messages = [
            message.model_copy(update={"blocks": blocks_by_message.get(message.provider_message_id, [])})
            for message in messages
        ]
    attachment_message_ids = dict(attachment_ref_tuples or [])
    attachments = [
        attachment.model_copy(
            update={"message_provider_id": attachment_message_ids.get(attachment.provider_attachment_id)}
        )
        for attachment in attachment_tuples or []
    ]
    parsed = ParsedSession(
        source_name=provider,
        provider_session_id=session_id.split(":", 1)[-1],
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        parent_session_provider_id=parent_session_id.split(":", 1)[-1] if parent_session_id else None,
        messages=messages,
        attachments=attachments,
        session_events=list(action_tuples or []),
        ingest_flags=list(ingest_flags or []),
    )
    return SessionWritePayload(
        session_id=session_id,
        content_hash=sha256(content_hash.encode()).hexdigest(),
        parsed_session=parsed,
        message_count=len(messages),
        attachment_count=len(attachments),
        raw_id=raw_id,
        append_only=append_only,
    )


def _message_tuple(
    message_id: str,
    session_id: str,
    *,
    role: str,
    text: str,
    content_hash: str,
    sort_key: float | None,
) -> ParsedMessage:
    del session_id, content_hash
    return ParsedMessage(
        provider_message_id=message_id,
        role=Role.normalize(role),
        text=text,
        occurred_at_ms=int(sort_key * 1000) if sort_key is not None else None,
    )


def _block_tuple(
    *,
    block_id: str,
    message_id: str,
    session_id: str,
    block_index: int,
    text: str,
) -> BlockSpec:
    del block_id, session_id, block_index
    return (
        message_id,
        ParsedContentBlock(type=BlockType.TEXT, text=text),
    )


def _action_tuple(
    *,
    event_id: str,
    session_id: str,
    message_id: str,
    search_text: str,
) -> ParsedSessionEvent:
    del event_id, session_id, search_text
    return ParsedSessionEvent(
        event_type="compaction",
        source_message_provider_id=message_id,
        timestamp="2026-04-02T00:00:00Z",
        payload={"summary": "compaction"},
    )


def _attachment_tuple(
    attachment_id: str,
    *,
    mime_type: str = "image/png",
    inline_bytes: bytes | None = None,
    precomputed_blob: tuple[str, int] | None = None,
) -> ParsedAttachment:
    return ParsedAttachment(
        provider_attachment_id=attachment_id,
        mime_type=mime_type,
        size_bytes=len(inline_bytes) if inline_bytes is not None else 1024,
        inline_bytes=inline_bytes,
        precomputed_blob=precomputed_blob,
    )


def _attachment_ref_tuple(
    attachment_id: str,
    session_id: str,
    message_id: str,
) -> AttachmentRefSpec:
    del session_id
    return (attachment_id, message_id)


def test_topo_sort_session_entries_orders_parent_before_child() -> None:
    parent = _session_data("codex-session:parent", content_hash="hash-parent")
    child = _session_data(
        "codex-session:child",
        content_hash="hash-child",
        parent_session_id="codex-session:parent",
    )

    ordered = _topo_sort_session_entries(
        [
            ("raw-child", child),
            ("raw-parent", parent),
        ]
    )

    assert [entry[1].session_id for entry in ordered] == [
        "codex-session:parent",
        "codex-session:child",
    ]


def test_write_session_clears_missing_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        c_msg = _message_tuple(
            "msg-c",
            "codex-session:child",
            role="user",
            text="hello",
            content_hash="hash-c",
            sort_key=1.0,
        )
        child = _session_data(
            "codex-session:child",
            content_hash="hash-child",
            parent_session_id="codex-session:missing-parent",
            message_tuples=[c_msg],
        )

        write_fixture_ingest_payload(conn, child)
        conn.commit()

        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = ?",
            ("codex-session:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_session_id"] is None


def test_write_session_preserves_existing_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        p_msg = _message_tuple(
            "msg-p",
            "codex-session:parent",
            role="user",
            text="parent msg",
            content_hash="hash-p",
            sort_key=1.0,
        )
        c_msg = _message_tuple(
            "msg-c",
            "codex-session:child",
            role="user",
            text="child msg",
            content_hash="hash-c",
            sort_key=1.0,
        )
        parent = _session_data(
            "codex-session:parent",
            content_hash="hash-parent",
            message_tuples=[p_msg],
        )
        child = _session_data(
            "codex-session:child",
            content_hash="hash-child",
            parent_session_id="codex-session:parent",
            message_tuples=[c_msg],
        )

        write_fixture_ingest_payload(conn, parent)
        write_fixture_ingest_payload(conn, child)
        conn.commit()

        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = ?",
            ("codex-session:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_session_id"] == "codex-session:parent"


def test_write_session_replaces_runtime_rows_on_content_change(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        archive = _session_data(
            "codex-session:replace",
            content_hash="hash-v1",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:replace",
                    role="user",
                    text="first",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex-session:replace",
                    role="assistant",
                    text="second",
                    content_hash="msg-v1-2",
                    sort_key=2.0,
                ),
            ],
            block_tuples=[
                _block_tuple(
                    block_id="blk-msg-1-0",
                    message_id="msg-1",
                    session_id="codex-session:replace",
                    block_index=0,
                    text="alpha",
                ),
                _block_tuple(
                    block_id="blk-msg-1-1",
                    message_id="msg-1",
                    session_id="codex-session:replace",
                    block_index=1,
                    text="beta",
                ),
            ],
            stats_tuple=(SessionId("codex-session:replace"), "codex", 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            attachment_tuples=[
                _attachment_tuple("att-1"),
                _attachment_tuple("att-2", mime_type="image/jpeg"),
            ],
            attachment_ref_tuples=[
                _attachment_ref_tuple("att-1", "codex-session:replace", "msg-1"),
                _attachment_ref_tuple("att-2", "codex-session:replace", "msg-2"),
            ],
        )
        changed, counts = write_fixture_ingest_payload(conn, archive)
        assert changed is True
        assert counts["messages"] == 2

        v2 = _session_data(
            "codex-session:replace",
            content_hash="hash-v2",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:replace",
                    role="user",
                    text="first updated",
                    content_hash="msg-v2-1",
                    sort_key=1.0,
                )
            ],
            block_tuples=[
                _block_tuple(
                    block_id="blk-msg-1-0-v2",
                    message_id="msg-1",
                    session_id="codex-session:replace",
                    block_index=0,
                    text="alpha updated",
                )
            ],
            stats_tuple=(SessionId("codex-session:replace"), "codex", 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            attachment_tuples=[_attachment_tuple("att-1")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", "codex-session:replace", "msg-1")],
        )
        changed, counts = write_fixture_ingest_payload(conn, v2)
        assert changed is True
        assert counts["messages"] == 1
        conn.commit()

        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                ("codex-session:replace",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE session_id = ?",
                ("codex-session:replace",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM attachment_refs WHERE session_id = ?",
                ("codex-session:replace",),
            ).fetchone()[0]
            == 1
        )
        att1_id = _attachment_id("codex-session:replace", _attachment_tuple("att-1"))
        att2_id = _attachment_id("codex-session:replace", _attachment_tuple("att-2", mime_type="image/jpeg"))
        assert (
            conn.execute(
                """
                SELECT m.native_id
                FROM attachment_refs r
                JOIN messages m ON m.message_id = r.message_id
                WHERE r.session_id = ? AND r.attachment_id = ?
                """,
                ("codex-session:replace", att1_id),
            ).fetchone()[0]
            == "msg-1"
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM attachment_refs WHERE session_id = ? AND attachment_id = ?",
                ("codex-session:replace", att2_id),
            ).fetchone()[0]
            == 0
        )
        stats_row = conn.execute(
            "SELECT message_count FROM sessions WHERE session_id = ?",
            ("codex-session:replace",),
        ).fetchone()
        assert stats_row is not None
        assert stats_row[0] == 1


def test_write_session_append_mode_preserves_existing_messages(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        initial = _session_data(
            "codex-session:append",
            content_hash="hash-v1",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:append",
                    role="user",
                    text="first",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                )
            ],
            stats_tuple=(SessionId("codex-session:append"), "codex", 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        tail = _session_data(
            "codex-session:append",
            content_hash="hash-tail",
            message_tuples=[
                _message_tuple(
                    "msg-2",
                    "codex-session:append",
                    role="assistant",
                    text="second",
                    content_hash="msg-v2-2",
                    sort_key=2.0,
                )
            ],
            append_only=True,
        )

        changed_initial, _initial_counts = write_fixture_ingest_payload(conn, initial)
        changed_tail, tail_counts = write_fixture_ingest_payload(conn, tail)
        conn.commit()

        rows = conn.execute(
            "SELECT native_id FROM messages WHERE session_id = ? ORDER BY position",
            ("codex-session:append",),
        ).fetchall()
        stats = conn.execute(
            "SELECT message_count, word_count FROM sessions WHERE session_id = ?",
            ("codex-session:append",),
        ).fetchone()

        assert changed_initial is True
        assert changed_tail is True
        assert tail_counts["messages"] == 1
        assert [row["native_id"] for row in rows] == ["msg-1", "msg-2"]
        assert stats is not None
        assert (stats["message_count"], stats["word_count"]) == (2, 2)


def test_write_session_append_dedupes_whitespace_padded_native_id(tmp_path: Path) -> None:
    """Append dedupe must use the normalized id before INSERT OR REPLACE.

    The writer stores ``" msg-1 "`` as ``"msg-1"``. If the append gate
    compares the raw provider id, it admits the duplicate and the generated
    message id makes INSERT OR REPLACE overwrite the original message.
    """
    with open_connection(tmp_path / "index.db") as conn:
        initial = _session_data(
            "codex-session:append-whitespace-id",
            content_hash="hash-initial",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:append-whitespace-id",
                    role="user",
                    text="original message",
                    content_hash="msg-original",
                    sort_key=1.0,
                )
            ],
        )
        duplicate = _session_data(
            "codex-session:append-whitespace-id",
            content_hash="hash-append-duplicate",
            message_tuples=[
                _message_tuple(
                    " msg-1 ",
                    "codex-session:append-whitespace-id",
                    role="assistant",
                    text="replacement must not win",
                    content_hash="msg-replacement",
                    sort_key=1.0,
                )
            ],
            append_only=True,
        )

        changed_initial, _ = write_fixture_ingest_payload(conn, initial)
        changed_duplicate, counts_duplicate = write_fixture_ingest_payload(conn, duplicate)
        conn.commit()

        rows = conn.execute(
            "SELECT native_id, position FROM messages WHERE session_id = ?",
            ("codex-session:append-whitespace-id",),
        ).fetchall()
        text = conn.execute(
            """
            SELECT b.text
            FROM blocks b
            JOIN messages m ON m.message_id = b.message_id
            WHERE m.session_id = ? AND b.block_type = 'text'
            """,
            ("codex-session:append-whitespace-id",),
        ).fetchone()

        assert changed_initial is True
        assert changed_duplicate is False
        assert counts_duplicate["skipped_messages"] == 1
        assert [(row["native_id"], row["position"]) for row in rows] == [("msg-1", 0)]
        assert text is not None
        assert text["text"] == "original message"


def test_write_session_append_no_delta_refreshes_raw_link(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        initial = _session_data(
            "codex-session:append-raw-link",
            content_hash="hash-v1",
            raw_id="raw-old",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:append-raw-link",
                    role="user",
                    text="first",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                )
            ],
        )
        recapture = _session_data(
            "codex-session:append-raw-link",
            content_hash="hash-v1",
            raw_id="raw-new",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:append-raw-link",
                    role="user",
                    text="first",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                )
            ],
            append_only=True,
        )

        changed_initial, _initial_counts = write_fixture_ingest_payload(conn, initial)
        changed_recapture, recapture_counts = write_fixture_ingest_payload(conn, recapture)
        conn.commit()

        raw_id = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("codex-session:append-raw-link",),
        ).fetchone()["raw_id"]

        assert changed_initial is True
        assert changed_recapture is False
        assert recapture_counts["skipped_sessions"] == 1
        assert recapture_counts["raw_links"] == 1
        assert raw_id == "raw-new"


def test_write_session_force_write_updates_message_time(tmp_path: Path) -> None:
    """force_write with identical content updates current message time columns."""
    with open_connection(tmp_path / "index.db") as conn:
        archive = _session_data(
            "codex-session:force",
            content_hash="same-hash",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:force",
                    role="user",
                    text="hello",
                    content_hash="msg-hash",
                    sort_key=None,
                )
            ],
        )
        changed, _ = write_fixture_ingest_payload(conn, archive)
        assert changed is True

        v2 = _session_data(
            "codex-session:force",
            content_hash="same-hash",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:force",
                    role="user",
                    text="hello",
                    content_hash="msg-hash",
                    sort_key=1777636800.0,
                )
            ],
        )
        unchanged, counts = write_fixture_ingest_payload(conn, v2)
        assert unchanged is False
        assert counts["skipped_sessions"] == 1

        forced, counts = write_fixture_ingest_payload(conn, v2, force_write=True)
        assert forced is True
        assert counts["messages"] == 1
        conn.commit()

        rows = conn.execute(
            "SELECT native_id, role, occurred_at_ms FROM messages WHERE session_id = ?",
            ("codex-session:force",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["native_id"] == "msg-1"
        assert rows[0]["role"] == "user"
        assert rows[0]["occurred_at_ms"] == 1777636800000


def test_write_session_force_write_replaces_older_freshness(tmp_path: Path) -> None:
    """Raw convergence force writes may replace a newer stale index row."""
    with open_connection(tmp_path / "index.db") as conn:
        newer = _session_data(
            "codex-session:force-stale",
            content_hash="hash-newer",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:force-stale",
                    role="user",
                    text="stale index",
                    content_hash="msg-hash-1",
                    sort_key=1777636800.0,
                )
            ],
            raw_id="raw-stale",
            created_at="2026-04-02T00:00:00Z",
            updated_at="2026-04-02T00:10:00Z",
        )
        changed, _ = write_fixture_ingest_payload(conn, newer)
        assert changed is True

        older = _session_data(
            "codex-session:force-stale",
            content_hash="hash-older",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:force-stale",
                    role="user",
                    text="durable source",
                    content_hash="msg-hash-2",
                    sort_key=1777636700.0,
                )
            ],
            raw_id="raw-source",
            created_at="2026-04-02T00:00:00Z",
            updated_at="2026-04-02T00:05:00Z",
        )
        skipped, skipped_counts = write_fixture_ingest_payload(conn, older)
        assert skipped is False
        assert skipped_counts["messages"] == 0

        forced, forced_counts = write_fixture_ingest_payload(conn, older, force_write=True)
        assert forced is True
        assert forced_counts["messages"] == 1
        conn.commit()

        row = conn.execute(
            "SELECT raw_id, message_count FROM sessions WHERE session_id = ?",
            ("codex-session:force-stale",),
        ).fetchone()
        message = conn.execute(
            "SELECT native_id, occurred_at_ms FROM messages WHERE session_id = ?",
            ("codex-session:force-stale",),
        ).fetchone()
        block = conn.execute(
            "SELECT text FROM blocks WHERE session_id = ?",
            ("codex-session:force-stale",),
        ).fetchone()
        assert row["raw_id"] == "raw-source"
        assert row["message_count"] == 1
        assert block["text"] == "durable source"
        assert message["occurred_at_ms"] == 1777636700000


def test_write_session_freshness_tie_keeps_acquired_attachment(tmp_path: Path) -> None:
    """polylogue-ixry: a same-timestamp re-acquisition must not regress attachments.

    Reproduces the aistudio-drive live-fetch race found on the live archive:
    two raw acquisitions of the same conversation share message content and
    an identical content-derived `updated_at` (attachment fetch injects bytes
    into the raw JSON without touching any message timestamp), but only one
    of the two raw revisions carries the attachment's fetched bytes. Without
    the tie-break, whichever raw is written second wins outright -- which on
    the live archive was, 157/157 times, the revision that lost the fetched
    bytes.
    """
    with open_connection(tmp_path / "index.db") as conn:
        publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        fetched = _session_data(
            "aistudio-drive:tie",
            content_hash="hash-fetched",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "aistudio-drive:tie",
                    role="user",
                    text="same content",
                    content_hash="msg-hash-tie",
                    sort_key=1777636800.0,
                )
            ],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=b"real drive bytes")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", "aistudio-drive:tie", "msg-1")],
            raw_id="raw-fetched",
            provider=Provider.DRIVE,
            created_at="2026-07-18T17:46:10Z",
            updated_at="2026-07-18T17:46:10Z",
        )
        changed, counts = write_fixture_ingest_payload(conn, fetched, blob_publisher=publisher)
        assert changed is True
        assert counts["attachments"] == 1
        conn.commit()

        status_after_fetch = conn.execute(
            """
            SELECT a.acquisition_status FROM attachment_refs r
            JOIN attachments a ON a.attachment_id = r.attachment_id
            WHERE r.session_id = ?
            """,
            ("aistudio-drive:tie",),
        ).fetchone()
        assert status_after_fetch["acquisition_status"] == "acquired"

        unfetched_revision = _session_data(
            "aistudio-drive:tie",
            content_hash="hash-unfetched",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "aistudio-drive:tie",
                    role="user",
                    text="same content",
                    content_hash="msg-hash-tie",
                    sort_key=1777636800.0,
                )
            ],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=None)],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", "aistudio-drive:tie", "msg-1")],
            raw_id="raw-prefetch",
            provider=Provider.DRIVE,
            created_at="2026-07-18T17:46:10Z",
            updated_at="2026-07-18T17:46:10Z",
        )
        skipped, skipped_counts = write_fixture_ingest_payload(conn, unfetched_revision)
        conn.commit()

        assert skipped is False
        assert skipped_counts["skipped_sessions"] == 1
        row = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("aistudio-drive:tie",),
        ).fetchone()
        assert row["raw_id"] == "raw-fetched"
        status_row = conn.execute(
            """
            SELECT a.acquisition_status FROM attachment_refs r
            JOIN attachments a ON a.attachment_id = r.attachment_id
            WHERE r.session_id = ?
            """,
            ("aistudio-drive:tie",),
        ).fetchone()
        assert status_row["acquisition_status"] == "acquired"


def test_write_session_freshness_tie_allows_attachment_improvement(tmp_path: Path) -> None:
    """The tie-break only blocks regressions -- ties/improvements still write."""
    with open_connection(tmp_path / "index.db") as conn:
        unfetched = _session_data(
            "aistudio-drive:improve",
            content_hash="hash-unfetched",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "aistudio-drive:improve",
                    role="user",
                    text="same content",
                    content_hash="msg-hash-improve",
                    sort_key=1777636800.0,
                )
            ],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=None)],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", "aistudio-drive:improve", "msg-1")],
            raw_id="raw-prefetch",
            provider=Provider.DRIVE,
            created_at="2026-07-18T17:46:10Z",
            updated_at="2026-07-18T17:46:10Z",
        )
        changed, _ = write_fixture_ingest_payload(conn, unfetched)
        assert changed is True
        conn.commit()

        publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        fetched = _session_data(
            "aistudio-drive:improve",
            content_hash="hash-fetched",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "aistudio-drive:improve",
                    role="user",
                    text="same content",
                    content_hash="msg-hash-improve",
                    sort_key=1777636800.0,
                )
            ],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=b"real drive bytes")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", "aistudio-drive:improve", "msg-1")],
            raw_id="raw-fetched",
            provider=Provider.DRIVE,
            created_at="2026-07-18T17:46:10Z",
            updated_at="2026-07-18T17:46:10Z",
        )
        changed, counts = write_fixture_ingest_payload(conn, fetched, blob_publisher=publisher)
        conn.commit()

        assert changed is True
        assert counts["attachments"] == 1
        row = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("aistudio-drive:improve",),
        ).fetchone()
        assert row["raw_id"] == "raw-fetched"


def test_write_session_binds_drive_revision_lineage(tmp_path: Path) -> None:
    """polylogue-sp72: a Drive re-acquisition must get real revision lineage.

    ``iter_drive_raw_data`` backfills live-fetched attachment bytes into
    cached Drive JSON on every ingest pass; whenever the bytes change, it
    mints a brand-new ``raw_id`` for the SAME logical session. Before this
    fix, ``_write_session`` (this module's generic, non-drive-aware write
    path) never computed ``logical_source_key`` or called the revision
    governance cohort classifier for that second raw -- confirmed live: all
    157 duplicate ``aistudio-drive`` raw pairs carried
    ``revision_kind='unknown'``, ``logical_source_key=NULL``,
    ``revision_authority='quarantined'``, with no predecessor/baseline
    linkage at all (see PR #3453's docstring on
    ``_incoming_write_regresses_attachment_coverage``, the narrow tie-break
    that papered over exactly this gap without fixing it).

    This asserts the SECOND raw's ``raw_sessions`` row now carries a real
    ``logical_source_key``, a ``predecessor_raw_id`` pointing at the first
    raw, and a non-``quarantined`` ``revision_authority`` -- i.e. that the
    real byte-prefix arbitration classifier actually ran, not just that
    some metadata got stamped. This must fail against the pre-fix
    ``_write_session`` (verified by temporarily reverting the production
    change and re-running: both raws come back with
    ``logical_source_key IS NULL``).
    """
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    source_db_path = archive_root / "source.db"
    blob_publisher = ArchiveBlobPublisher(source_db_path, archive_root / "blob")

    provider_session_id = "drive-lineage-tie"
    session_id = f"aistudio-drive:{provider_session_id}"
    # The second raw's bytes are a strict superset (prefix growth) of the
    # first's, mirroring a real Drive attachment backfill appending newly
    # fetched bytes into the cached JSON -- this is what lets the byte-prefix
    # classifier prove a real predecessor/baseline relation instead of
    # quarantining both raws as an unrelated/ambiguous pair.
    first_payload = b'{"drive":"payload","messages":[{"id":"m-1","text":"hi"}]}'
    second_payload = first_payload + b"  // backfilled attachment bytes appended"

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        first_raw_id = archive.write_raw_payload(
            provider=Provider.GEMINI,
            payload=first_payload,
            source_path="drive://file-1",
            acquired_at_ms=1_767_000_000_000,
        )
        second_raw_id = archive.write_raw_payload(
            provider=Provider.GEMINI,
            payload=second_payload,
            source_path="drive://file-1",
            acquired_at_ms=1_767_000_000_500,
        )
    assert first_raw_id != second_raw_id

    message = _message_tuple(
        "m-1",
        session_id,
        role="user",
        text="hi",
        content_hash="msg-hash-drive-lineage",
        sort_key=1777636800.0,
    )

    with (
        open_connection(archive_root / "index.db") as conn,
        sqlite3.connect(str(source_db_path)) as source_conn,
    ):
        first_session = _session_data(
            session_id,
            content_hash="hash-drive-lineage-first",
            raw_id=first_raw_id,
            message_tuples=[message],
            provider=Provider.GEMINI,
            created_at="2026-07-18T17:46:10Z",
            updated_at="2026-07-18T17:46:10Z",
        )
        changed_first, _ = write_fixture_ingest_payload(
            conn, first_session, blob_publisher=blob_publisher, source_conn=source_conn
        )
        conn.commit()
        assert changed_first is True

        second_session = _session_data(
            session_id,
            content_hash="hash-drive-lineage-second",
            raw_id=second_raw_id,
            message_tuples=[message],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=b"fetched drive attachment bytes")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", session_id, "m-1")],
            provider=Provider.GEMINI,
            created_at="2026-07-18T17:46:15Z",
            updated_at="2026-07-18T17:46:15Z",
        )
        changed_second, _ = write_fixture_ingest_payload(
            conn, second_session, blob_publisher=blob_publisher, source_conn=source_conn
        )
        conn.commit()
        assert changed_second is True

    with sqlite3.connect(str(source_db_path)) as verify_conn:
        verify_conn.row_factory = sqlite3.Row
        first_row = verify_conn.execute(
            "SELECT logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (first_raw_id,),
        ).fetchone()
        second_row = verify_conn.execute(
            "SELECT logical_source_key, predecessor_raw_id, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (second_raw_id,),
        ).fetchone()

    expected_logical_source_key = f"{Origin.AISTUDIO_DRIVE.value}:{provider_session_id}"
    assert first_row["logical_source_key"] == expected_logical_source_key
    assert second_row["logical_source_key"] == expected_logical_source_key
    assert second_row["predecessor_raw_id"] == first_raw_id
    assert second_row["revision_authority"] != "quarantined"


def test_write_session_drive_lineage_proven_winner_bypasses_freshness_tie(tmp_path: Path) -> None:
    """polylogue-sp72 AC2: a governance-proven Drive revision must not need
    the #3453 freshness-tie safety net to land.

    Reproduces the exact tie shape the #3453 tie-break exists to guess at
    (identical message content/timestamps, differing attachment coverage,
    differing raw_id) -- but this time the second raw's bytes are a proven
    byte-prefix successor of the first (the same fixture shape as
    ``test_write_session_binds_drive_revision_lineage``), so
    ``classify_raw_revision_cohort_for_live_watch`` has already settled which
    raw wins before the freshness comparison ever runs. The second write here
    carries FEWER acquired attachments than the first -- exactly the shape
    ``_incoming_write_regresses_attachment_coverage`` blocks -- yet must still
    land, because governance (not the heuristic) already proved it is the
    legitimate next revision in the chain.

    The negative control (``test_write_session_freshness_tie_regression_without_lineage_still_blocks``)
    proves the tie-break safety net is untouched when no governance evidence
    is available (no ``source_conn``/``blob_publisher``): it still blocks the
    exact same regression shape.
    """
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    source_db_path = archive_root / "source.db"
    blob_publisher = ArchiveBlobPublisher(source_db_path, archive_root / "blob")

    provider_session_id = "drive-tie-proven-winner"
    session_id = f"aistudio-drive:{provider_session_id}"
    first_payload = b'{"drive":"payload","messages":[{"id":"m-1","text":"hi"}]}'
    second_payload = first_payload + b"  // backfilled attachment bytes appended"

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        first_raw_id = archive.write_raw_payload(
            provider=Provider.GEMINI,
            payload=first_payload,
            source_path="drive://file-tie",
            acquired_at_ms=1_767_000_000_000,
        )
        second_raw_id = archive.write_raw_payload(
            provider=Provider.GEMINI,
            payload=second_payload,
            source_path="drive://file-tie",
            acquired_at_ms=1_767_000_000_500,
        )
    assert first_raw_id != second_raw_id

    message = _message_tuple(
        "m-1",
        session_id,
        role="user",
        text="hi",
        content_hash="msg-hash-drive-tie",
        sort_key=1777636800.0,
    )
    tied_timestamp = "2026-07-18T17:46:10Z"

    with (
        open_connection(archive_root / "index.db") as conn,
        sqlite3.connect(str(source_db_path)) as source_conn,
    ):
        first_session = _session_data(
            session_id,
            content_hash="hash-drive-tie-first",
            raw_id=first_raw_id,
            message_tuples=[message],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=b"acquired drive attachment bytes")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", session_id, "m-1")],
            provider=Provider.GEMINI,
            created_at=tied_timestamp,
            updated_at=tied_timestamp,
        )
        changed_first, _ = write_fixture_ingest_payload(
            conn, first_session, blob_publisher=blob_publisher, source_conn=source_conn
        )
        conn.commit()
        assert changed_first is True

        # Same content-derived timestamp as the first write, but zero
        # acquired attachments -- a coverage regression the #3453 tie-break
        # exists to block, UNLESS governance already proved this raw is the
        # legitimate next revision (which it does here).
        second_session = _session_data(
            session_id,
            content_hash="hash-drive-tie-second",
            raw_id=second_raw_id,
            message_tuples=[message],
            provider=Provider.GEMINI,
            created_at=tied_timestamp,
            updated_at=tied_timestamp,
        )
        changed_second, _ = write_fixture_ingest_payload(
            conn, second_session, blob_publisher=blob_publisher, source_conn=source_conn
        )
        conn.commit()

        assert changed_second is True
        row = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert row["raw_id"] == second_raw_id


def test_write_session_freshness_tie_regression_without_lineage_still_blocks(tmp_path: Path) -> None:
    """Negative control for the test above: absent lineage governance
    evidence (no ``source_conn``, so ``_bind_drive_revision_lineage`` bails
    out before ever classifying), the #3453 tie-break safety net still
    blocks the identical regression shape -- this fix only ever bypasses the
    heuristic when governance has independently proven a real
    predecessor/successor relationship, never unconditionally.
    ``blob_publisher`` is still supplied (attachments require one to write
    inline bytes at all) -- ``source_conn`` alone is what gates lineage
    governance."""
    with open_connection(tmp_path / "index.db") as conn:
        blob_publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        session_id = "aistudio-drive:tie-no-lineage"
        tied_timestamp = "2026-07-18T17:46:10Z"
        message = _message_tuple(
            "m-1",
            session_id,
            role="user",
            text="hi",
            content_hash="msg-hash-tie-no-lineage",
            sort_key=1777636800.0,
        )
        first_session = _session_data(
            session_id,
            content_hash="hash-tie-no-lineage-first",
            raw_id="raw-no-lineage-first",
            message_tuples=[message],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=b"acquired bytes")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", session_id, "m-1")],
            provider=Provider.GEMINI,
            created_at=tied_timestamp,
            updated_at=tied_timestamp,
        )
        changed_first, _ = write_fixture_ingest_payload(conn, first_session, blob_publisher=blob_publisher)
        conn.commit()
        assert changed_first is True

        second_session = _session_data(
            session_id,
            content_hash="hash-tie-no-lineage-second",
            raw_id="raw-no-lineage-second",
            message_tuples=[message],
            provider=Provider.GEMINI,
            created_at=tied_timestamp,
            updated_at=tied_timestamp,
        )
        changed_second, counts_second = write_fixture_ingest_payload(
            conn, second_session, blob_publisher=blob_publisher
        )
        conn.commit()

        assert changed_second is False
        assert counts_second["skipped_sessions"] == 1
        row = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert row["raw_id"] == "raw-no-lineage-first"


def test_write_session_freshness_tie_with_distinct_messages_is_not_skipped(tmp_path: Path) -> None:
    """polylogue-5uoed: the attachment tie-break must not discard new content.

    Same shape as the negative control above -- tied content-derived
    freshness, different ``raw_id``, no lineage governance evidence, and an
    incoming revision carrying FEWER acquired attachments -- with one
    difference: the incoming revision also carries a message the archive does
    not hold. Deciding that tie on attachment count alone skips the write and
    the distinct message never lands, which on a fresh import is lost content.

    Anti-vacuity: drop the ``_incoming_write_carries_distinct_messages`` term
    from the caller gate in ``ingest_batch/_core.py`` and this goes red --
    ``changed_second`` comes back ``False``, ``skipped_sessions`` is 1, and
    ``m-2`` is absent from ``messages``. The negative control immediately
    above is the other half of the pair: with the SAME message set it must
    still be skipped, so this cannot be satisfied by disabling the tie-break.
    """
    with open_connection(tmp_path / "index.db") as conn:
        blob_publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        session_id = "aistudio-drive:tie-distinct-messages"
        tied_timestamp = "2026-07-18T17:46:10Z"
        first_message = _message_tuple(
            "m-1",
            session_id,
            role="user",
            text="hi",
            content_hash="msg-hash-tie-distinct-1",
            sort_key=1777636800.0,
        )
        second_message = _message_tuple(
            "m-2",
            session_id,
            role="assistant",
            text="a reply the archive has never seen",
            content_hash="msg-hash-tie-distinct-2",
            sort_key=1777636801.0,
        )
        first_session = _session_data(
            session_id,
            content_hash="hash-tie-distinct-first",
            raw_id="raw-distinct-first",
            message_tuples=[first_message],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=b"acquired bytes")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", session_id, "m-1")],
            provider=Provider.GEMINI,
            created_at=tied_timestamp,
            updated_at=tied_timestamp,
        )
        changed_first, _ = write_fixture_ingest_payload(conn, first_session, blob_publisher=blob_publisher)
        conn.commit()
        assert changed_first is True

        second_session = _session_data(
            session_id,
            content_hash="hash-tie-distinct-second",
            raw_id="raw-distinct-second",
            message_tuples=[first_message, second_message],
            provider=Provider.GEMINI,
            created_at=tied_timestamp,
            updated_at=tied_timestamp,
        )
        changed_second, counts_second = write_fixture_ingest_payload(
            conn, second_session, blob_publisher=blob_publisher
        )
        conn.commit()

        assert changed_second is True
        assert counts_second["skipped_sessions"] == 0
        native_ids = {
            str(row[0])
            for row in conn.execute(
                "SELECT native_id FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        }
        assert native_ids == {"m-1", "m-2"}


def test_write_session_freshness_tie_with_a_revised_semantic_field_is_not_skipped(tmp_path: Path) -> None:
    """A tied revision that changes only a message's model is a semantic revision.

    Same shape as the negative control -- tied freshness, fewer acquired
    attachments, the same message text -- except that ``model_name`` differs.
    Anti-vacuity: comparing the coarse lineage signature (role and blocks)
    reads the incoming message as already held, skips the replacement, and
    leaves the old model on the stored message.
    """
    with open_connection(tmp_path / "index.db") as conn:
        blob_publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        session_id = "aistudio-drive:tie-revised-model"
        tied_timestamp = "2026-07-18T17:46:10Z"
        message = _message_tuple(
            "m-1", session_id, role="assistant", text="hi", content_hash="unused", sort_key=1777636800.0
        )
        first_session = _session_data(
            session_id,
            content_hash="hash-tie-revised-first",
            raw_id="raw-revised-first",
            message_tuples=[message.model_copy(update={"model_name": "model-a"})],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=b"acquired bytes")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", session_id, "m-1")],
            provider=Provider.GEMINI,
            created_at=tied_timestamp,
            updated_at=tied_timestamp,
        )
        changed_first, _ = write_fixture_ingest_payload(conn, first_session, blob_publisher=blob_publisher)
        conn.commit()
        assert changed_first is True

        second_session = _session_data(
            session_id,
            content_hash="hash-tie-revised-second",
            raw_id="raw-revised-second",
            message_tuples=[message.model_copy(update={"model_name": "model-b"})],
            provider=Provider.GEMINI,
            created_at=tied_timestamp,
            updated_at=tied_timestamp,
        )
        changed_second, counts_second = write_fixture_ingest_payload(
            conn, second_session, blob_publisher=blob_publisher
        )
        conn.commit()

        assert changed_second is True
        assert counts_second["skipped_sessions"] == 0
        [model] = [
            row[0] for row in conn.execute("SELECT model_name FROM messages WHERE session_id = ?", (session_id,))
        ]
        assert model == "model-b"


def test_write_session_precomputed_blob_attachment_recorded_as_acquired(tmp_path: Path) -> None:
    """bd polylogue-8ac0: bytes already streamed into the blob store during
    sidecar discovery (ChatGPT ``.dat`` asset acquisition) are recorded
    ``acquired`` via ``ParsedAttachment.precomputed_blob`` -- no
    ``blob_publisher`` write happens here, since the bytes were already
    published earlier.
    """
    store = BlobStore(tmp_path / "blob")
    payload = b"chatgpt dat asset bytes"
    blob_hash, size = store.write_from_bytes(payload)

    with open_connection(tmp_path / "index.db") as conn:
        session = _session_data(
            "chatgpt-export:conv-1",
            content_hash="hash-precomputed",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "chatgpt-export:conv-1",
                    role="user",
                    text="here is a photo",
                    content_hash="msg-hash-precomputed",
                    sort_key=1777636800.0,
                )
            ],
            attachment_tuples=[_attachment_tuple("file-xyz", precomputed_blob=(blob_hash, size))],
            attachment_ref_tuples=[_attachment_ref_tuple("file-xyz", "chatgpt-export:conv-1", "msg-1")],
            raw_id="raw-chatgpt",
            provider=Provider.CHATGPT,
        )
        changed, counts = write_fixture_ingest_payload(conn, session)
        conn.commit()

        assert changed is True
        assert counts["attachments"] == 1
        row = conn.execute(
            """
            SELECT a.acquisition_status, a.blob_hash, a.byte_count
            FROM attachment_refs r JOIN attachments a ON a.attachment_id = r.attachment_id
            WHERE r.session_id = ?
            """,
            ("chatgpt-export:conv-1",),
        ).fetchone()
        assert row["acquisition_status"] == "acquired"
        assert bytes(row["blob_hash"]) == bytes.fromhex(blob_hash)
        assert row["byte_count"] == size
        assert store.read_all(blob_hash) == payload


def _excise_in_fresh_source_tier(root: Path, payload: bytes) -> Path:
    """Initialize an archive at *root* and record *payload*'s blob hash as excised."""
    from polylogue.storage.sqlite.archive_tiers.source_write import record_excised_blob_hash

    with ArchiveStore(root, initialize=True, read_only=False):
        pass
    with sqlite3.connect(root / "source.db") as source:
        record_excised_blob_hash(
            source, blob_hash=sha256(payload).digest(), reason="synthetic excision", actor="test", excised_at_ms=1
        )
    return root / "source.db"


def _excised_attachment_row(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    row: sqlite3.Row = conn.execute(
        """
        SELECT a.acquisition_status, a.blob_hash, a.byte_count
        FROM attachment_refs r JOIN attachments a ON a.attachment_id = r.attachment_id
        WHERE r.session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return row


def test_write_session_records_an_excised_inline_attachment_unavailable(tmp_path: Path) -> None:
    """An inline attachment whose bytes the flush refused as excised is not acquired.

    Anti-vacuity: drop ``refuse_excised_attachment_blobs`` from ``_write_session``
    and the attachment row is committed ``acquired`` with the excised hash,
    whose staged bytes the flush discarded, so the reference dangles.
    """
    payload = b"attachment bytes the operator excised"
    source_db = _excise_in_fresh_source_tier(tmp_path / "archive", payload)
    publisher = ArchiveBlobPublisher(source_db, tmp_path / "archive" / "blob")
    with open_connection(tmp_path / "index.db") as conn:
        session = _session_data(
            "chatgpt-export:conv-excised",
            content_hash="hash-excised-inline",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "chatgpt-export:conv-excised",
                    role="user",
                    text="see attached",
                    content_hash="msg-hash-excised-inline",
                    sort_key=1777636800.0,
                )
            ],
            attachment_tuples=[_attachment_tuple("att-1", inline_bytes=payload)],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", "chatgpt-export:conv-excised", "msg-1")],
            raw_id="raw-excised-inline",
            provider=Provider.CHATGPT,
        )
        changed, _counts = write_fixture_ingest_payload(conn, session, blob_publisher=publisher)
        conn.commit()

        assert changed is True
        row = _excised_attachment_row(conn, "chatgpt-export:conv-excised")
        assert row["acquisition_status"] == "unavailable"
        assert row["blob_hash"] is None
        assert row["byte_count"] == len(payload)
    assert not publisher.exists(sha256(payload).hexdigest())


def test_write_session_records_an_excised_precomputed_attachment_unavailable(tmp_path: Path) -> None:
    """Bytes published earlier and excised since are not recorded as acquired.

    Anti-vacuity: drop the ledger check (``source_conn``) from
    ``refuse_excised_attachment_blobs`` and the precomputed blob is recorded
    ``acquired`` under the excised hash.
    """
    payload = b"chatgpt asset bytes excised after acquisition"
    source_db = _excise_in_fresh_source_tier(tmp_path / "archive", payload)
    with open_connection(tmp_path / "archive" / "index.db") as conn, sqlite3.connect(source_db) as source_conn:
        session = _session_data(
            "chatgpt-export:conv-precomputed",
            content_hash="hash-excised-precomputed",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "chatgpt-export:conv-precomputed",
                    role="user",
                    text="here is a photo",
                    content_hash="msg-hash-excised-precomputed",
                    sort_key=1777636800.0,
                )
            ],
            attachment_tuples=[
                _attachment_tuple("file-xyz", precomputed_blob=(sha256(payload).hexdigest(), len(payload)))
            ],
            attachment_ref_tuples=[_attachment_ref_tuple("file-xyz", "chatgpt-export:conv-precomputed", "msg-1")],
            raw_id="raw-excised-precomputed",
            provider=Provider.CHATGPT,
        )
        changed, _counts = write_fixture_ingest_payload(conn, session, source_conn=source_conn)
        conn.commit()

        assert changed is True
        row = _excised_attachment_row(conn, "chatgpt-export:conv-precomputed")
        assert row["acquisition_status"] == "unavailable"
        assert row["blob_hash"] is None


def _precomputed_blob_session(blob_hash: str, size: int) -> SessionWritePayload:
    return _session_data(
        "chatgpt-export:conv-adopt",
        content_hash="hash-adopt",
        message_tuples=[
            _message_tuple(
                "msg-1",
                "chatgpt-export:conv-adopt",
                role="user",
                text="here is a file",
                content_hash="msg-hash-adopt",
                sort_key=1777636800.0,
            )
        ],
        attachment_tuples=[_attachment_tuple("file-adopt", precomputed_blob=(blob_hash, size))],
        attachment_ref_tuples=[_attachment_ref_tuple("file-adopt", "chatgpt-export:conv-adopt", "msg-1")],
        raw_id="raw-adopt",
        provider=Provider.CHATGPT,
    )


def _reservations(source_db: Path, blob_hash: str) -> list[str]:
    with sqlite3.connect(source_db) as conn:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT publication_id FROM blob_publication_reservations WHERE blob_hash = ?",
                (bytes.fromhex(blob_hash),),
            )
        ]


def test_write_session_reserves_a_worker_published_blob_until_its_reference_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parse worker's already-published attachment bytes get a publication receipt.

    The worker holds no write lease, so the bytes are GC-eligible until the
    attachment row exists. Anti-vacuity: trusting ``precomputed_blob`` without
    adopting it leaves no reservation row and no receipt for the batch to
    consume with the attachment reference.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    blob_hash, size = BlobStore(archive_root / "blob").write_from_bytes(b"spilled carrier bytes")
    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    import polylogue.storage.blob_publication as publication

    original_consume = publication.consume_blob_publication_receipt
    consumed: list[str] = []

    def consume_after_reference(source: sqlite3.Connection, publication_id: str, digest: bytes) -> None:
        assert source.execute(
            "SELECT blob_hash FROM blob_publication_reservations WHERE publication_id=?", (publication_id,)
        ).fetchone()[0] == bytes.fromhex(blob_hash)
        with sqlite3.connect(archive_root / "index.db") as committed_index:
            assert committed_index.execute(
                "SELECT acquisition_status, lower(hex(blob_hash)) FROM attachments"
            ).fetchone() == ("acquired", blob_hash)
        consumed.append(publication_id)
        original_consume(source, publication_id, digest)

    monkeypatch.setattr(publication, "consume_blob_publication_receipt", consume_after_reference)

    with open_connection(archive_root / "index.db") as conn:
        changed, _counts = write_fixture_ingest_payload(
            conn,
            _precomputed_blob_session(blob_hash, size),
            blob_publisher=publisher,
        )
        conn.commit()
        acquired = conn.execute("SELECT acquisition_status, lower(hex(blob_hash)) FROM attachments").fetchall()

    assert changed is True
    assert [tuple(row) for row in acquired] == [("acquired", blob_hash)]
    assert len(consumed) == 1
    assert _reservations(archive_root / "source.db", blob_hash) == []


def test_write_session_refuses_a_worker_published_blob_gc_reclaimed(tmp_path: Path) -> None:
    """Bytes reclaimed between the worker's publish and the writer are a retryable storage fault.

    Anti-vacuity: without the flush's presence check the attachment is
    recorded ``acquired`` against a blob that no longer exists, and a check
    made after reserving would leave an orphaned reservation behind.
    """
    from polylogue.core.storage_faults import StorageFaultKind, storage_fault_kind
    from polylogue.storage.blob_publication import AdoptedBlobEvictedError

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    store = BlobStore(archive_root / "blob")
    blob_hash, size = store.write_from_bytes(b"reclaimed carrier bytes")
    store.blob_path(blob_hash).unlink()
    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    with open_connection(tmp_path / "archive" / "index.db") as conn:
        with pytest.raises(AdoptedBlobEvictedError) as refused:
            write_fixture_ingest_payload(conn, _precomputed_blob_session(blob_hash, size), blob_publisher=publisher)
        assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == 0

    assert storage_fault_kind(refused.value) is StorageFaultKind.EVICTED
    assert refused.value.blob_hashes == (blob_hash,)
    assert _reservations(archive_root / "source.db", blob_hash) == []
    assert not publisher.has_pending


def _sidecar_matched_event(tool_use_id: str) -> ParsedSessionEvent:
    return ParsedSessionEvent(
        event_type="claude_tool_result_sidecar",
        payload={
            "acquisition_status": "matched",
            "tool_use_id": tool_use_id,
            "filename": f"{tool_use_id}.txt",
            "byte_size": 999,
            "content_hash": "irrelevant-precomputed-hash",
            "content_replaced": True,
        },
    )


def test_write_session_publishes_sidecar_blob_content_addressed(tmp_path: Path) -> None:
    """polylogue-rujy AC4: acquired sidecar text is published to the blob store, content-addressed.

    A matched+content_replaced ``claude_tool_result_sidecar`` event alongside
    a ``tool_result`` block sharing its ``tool_use_id`` is exactly what
    ``apply_tool_result_sidecars`` (sources/parsers/claude/code_parser.py)
    produces once a large sidecar file's full text has replaced a truncated
    block preview. This asserts the write path (not a test-only
    reimplementation) actually pushes those bytes through
    ``ArchiveBlobPublisher`` -- the same publisher attachments use -- and
    records the resulting hash back onto the event.
    """
    full_text = "full sidecar output " * 200
    with open_connection(tmp_path / "index.db") as conn:
        publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        session = _session_data(
            "claude-code-session:sidecar-1",
            content_hash="sidecar-hash-1",
            provider=Provider.CLAUDE_CODE,
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "claude-code-session:sidecar-1",
                    role="assistant",
                    text="ran a command",
                    content_hash="msg-hash-sidecar-1",
                    sort_key=1777636900.0,
                )
            ],
            block_tuples=[
                (
                    "msg-1",
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        outcome_unknown_reason="not_reported",
                        tool_id="toolu_1",
                        text=full_text,
                    ),
                )
            ],
            action_tuples=[_sidecar_matched_event("toolu_1")],
        )
        changed, counts = write_fixture_ingest_payload(conn, session, blob_publisher=publisher)
        conn.commit()

        assert changed is True
        assert counts["sidecar_blobs_written"] == 1
        assert counts["sidecar_blob_bytes_new"] == len(full_text.encode("utf-8"))
        assert counts["sidecar_blob_bytes_dedup"] == 0

        expected_hash = sha256(full_text.encode("utf-8")).hexdigest()
        blob_path = tmp_path / "blob" / expected_hash[:2] / expected_hash[2:]
        assert blob_path.is_file(), "sidecar bytes must land in the content-addressed blob store"
        assert blob_path.read_text() == full_text

        event_row = conn.execute(
            "SELECT payload_json FROM session_events WHERE session_id = ? AND event_type = 'claude_tool_result_sidecar'",
            ("claude-code-session:sidecar-1",),
        ).fetchone()
        assert f'"blob_hash":"{expected_hash}"' in event_row["payload_json"]

        block_row = conn.execute(
            "SELECT text FROM blocks WHERE session_id = ? AND block_type = 'tool_result'",
            ("claude-code-session:sidecar-1",),
        ).fetchone()
        assert block_row["text"] == full_text, "blob publication must not disturb the FTS-indexed block text (AC2)"


def test_write_session_counts_no_refused_sidecar_blob(tmp_path: Path) -> None:
    """Sidecar bytes the flush refuses as excised are not counted as written.

    Anti-vacuity: count sidecar blobs before the flush again and the batch
    reports a published blob the flush discarded.
    """
    full_text = "excised sidecar output " * 200
    source_db = _excise_in_fresh_source_tier(tmp_path / "archive", full_text.encode("utf-8"))
    publisher = ArchiveBlobPublisher(source_db, tmp_path / "archive" / "blob")
    with open_connection(tmp_path / "index.db") as conn:
        _changed, counts = write_fixture_ingest_payload(
            conn, _excised_sidecar_session("claude-code-session:sidecar-excised", full_text), blob_publisher=publisher
        )
        conn.commit()
    assert counts["sidecar_blobs_written"] == 0
    assert counts["sidecar_blob_bytes_new"] == 0


def _excised_sidecar_session(session_id: str, full_text: str) -> Any:
    return _session_data(
        session_id,
        content_hash=f"{session_id}-hash",
        provider=Provider.CLAUDE_CODE,
        message_tuples=[
            _message_tuple(
                "msg-1",
                session_id,
                role="assistant",
                text="ran a command",
                content_hash=f"{session_id}-msg-hash",
                sort_key=1777636900.0,
            )
        ],
        block_tuples=[
            (
                "msg-1",
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT, outcome_unknown_reason="not_reported", tool_id="toolu_1", text=full_text
                ),
            )
        ],
        action_tuples=[_sidecar_matched_event("toolu_1")],
    )


def test_write_session_dedups_identical_sidecar_blob_across_sessions(tmp_path: Path) -> None:
    """Two sessions with byte-identical acquired sidecar text share one blob (AC4 dedup).

    Mutation that would make this fail: removing the ``blob_publisher.exists``
    pre-check (or always reporting ``bytes_new``) collapses the
    new-vs-deduplicated distinction the bead asks be reported -- this test
    fails if the second write is counted as new bytes instead of a dedup hit.
    """
    full_text = "identical build log content\n" * 300
    with open_connection(tmp_path / "index.db") as conn:
        publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        first = _session_data(
            "claude-code-session:sidecar-dedup-a",
            content_hash="sidecar-dedup-hash-a",
            provider=Provider.CLAUDE_CODE,
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "claude-code-session:sidecar-dedup-a",
                    role="assistant",
                    text="ran a command",
                    content_hash="msg-hash-dedup-a",
                    sort_key=1777637000.0,
                )
            ],
            block_tuples=[
                (
                    "msg-1",
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        outcome_unknown_reason="not_reported",
                        tool_id="toolu_a",
                        text=full_text,
                    ),
                )
            ],
            action_tuples=[_sidecar_matched_event("toolu_a")],
        )
        changed_a, counts_a = write_fixture_ingest_payload(conn, first, blob_publisher=publisher)
        conn.commit()
        assert changed_a is True
        assert counts_a["sidecar_blob_bytes_new"] == len(full_text.encode("utf-8"))
        assert counts_a["sidecar_blob_bytes_dedup"] == 0

        second = _session_data(
            "claude-code-session:sidecar-dedup-b",
            content_hash="sidecar-dedup-hash-b",
            provider=Provider.CLAUDE_CODE,
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "claude-code-session:sidecar-dedup-b",
                    role="assistant",
                    text="ran the same command elsewhere",
                    content_hash="msg-hash-dedup-b",
                    sort_key=1777637100.0,
                )
            ],
            block_tuples=[
                (
                    "msg-1",
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        outcome_unknown_reason="not_reported",
                        tool_id="toolu_b",
                        text=full_text,
                    ),
                )
            ],
            action_tuples=[_sidecar_matched_event("toolu_b")],
        )
        changed_b, counts_b = write_fixture_ingest_payload(conn, second, blob_publisher=publisher)
        conn.commit()

        assert changed_b is True
        assert counts_b["sidecar_blob_bytes_new"] == 0
        assert counts_b["sidecar_blob_bytes_dedup"] == len(full_text.encode("utf-8"))


def test_write_session_skips_sidecar_blob_for_debt_events(tmp_path: Path) -> None:
    """Debt (unmatched) sidecar events never trigger a blob write -- there is no owning block/bytes."""
    with open_connection(tmp_path / "index.db") as conn:
        publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        debt_event = ParsedSessionEvent(
            event_type="claude_tool_result_sidecar",
            payload={
                "acquisition_status": "debt",
                "filename": "orphan.txt",
                "byte_size": 42,
                "reason": "no_owning_tool_result_block",
            },
        )
        session = _session_data(
            "claude-code-session:sidecar-debt",
            content_hash="sidecar-debt-hash",
            provider=Provider.CLAUDE_CODE,
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "claude-code-session:sidecar-debt",
                    role="assistant",
                    text="ran a command",
                    content_hash="msg-hash-debt",
                    sort_key=1777637200.0,
                )
            ],
            action_tuples=[debt_event],
        )
        changed, counts = write_fixture_ingest_payload(conn, session, blob_publisher=publisher)
        conn.commit()

        assert changed is True
        assert counts["sidecar_blobs_written"] == 0
        assert counts["sidecar_blob_bytes_new"] == 0
        assert not any(p.is_file() for p in (tmp_path / "blob").glob("**/*")), (
            "no blob file should be written when there is no matched sidecar block"
        )


def test_write_session_upserts_ingest_flags_when_content_is_unchanged(tmp_path: Path) -> None:
    """Parser-owned auto-tags still converge when the content hash is unchanged."""
    with open_connection(tmp_path / "index.db") as conn:
        first = _session_data(
            "codex-session:unchanged-tags",
            content_hash="same-hash",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:unchanged-tags",
                    role="user",
                    text="hello",
                    content_hash="msg-hash",
                    sort_key=1777636800.0,
                )
            ],
        )
        changed, _ = write_fixture_ingest_payload(conn, first)
        assert changed is True

        recapture = _session_data(
            "codex-session:unchanged-tags",
            content_hash="same-hash",
            ingest_flags=["capture:temporary-chat"],
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:unchanged-tags",
                    role="user",
                    text="hello",
                    content_hash="msg-hash",
                    sort_key=1777636800.0,
                )
            ],
        )
        unchanged, counts = write_fixture_ingest_payload(conn, recapture)
        conn.commit()

        tags = conn.execute(
            """
            SELECT tag, tag_source, method
            FROM session_tags
            WHERE session_id = ?
            """,
            ("codex-session:unchanged-tags",),
        ).fetchall()
        assert unchanged is False
        assert counts["skipped_sessions"] == 1
        assert [(row["tag"], row["tag_source"], row["method"]) for row in tags] == [
            ("capture:temporary-chat", "auto", "parser")
        ]


def test_write_session_refreshes_raw_link_when_content_is_unchanged(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        first = _session_data(
            "codex-session:unchanged-raw-link",
            content_hash="same-hash",
            raw_id="raw-old",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:unchanged-raw-link",
                    role="user",
                    text="hello",
                    content_hash="msg-hash",
                    sort_key=1777636800.0,
                )
            ],
        )
        recapture = _session_data(
            "codex-session:unchanged-raw-link",
            content_hash="same-hash",
            raw_id="raw-new",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:unchanged-raw-link",
                    role="user",
                    text="hello",
                    content_hash="msg-hash",
                    sort_key=1777636800.0,
                )
            ],
        )

        changed, _ = write_fixture_ingest_payload(conn, first)
        unchanged, counts = write_fixture_ingest_payload(conn, recapture)
        conn.commit()

        raw_id = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("codex-session:unchanged-raw-link",),
        ).fetchone()["raw_id"]

        assert changed is True
        assert unchanged is False
        assert counts["skipped_sessions"] == 1
        assert counts["raw_links"] == 1
        assert raw_id == "raw-new"


def test_write_session_skips_shorter_duplicate_raw_source(tmp_path: Path) -> None:
    """Duplicate source files for the same session must not replace fuller rows."""
    with open_connection(tmp_path / "index.db") as conn:
        fuller = _session_data(
            "codex-session:duplicate",
            content_hash="hash-full",
            raw_id="raw-full",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:duplicate",
                    role="user",
                    text="first",
                    content_hash="msg-1",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex-session:duplicate",
                    role="assistant",
                    text="second",
                    content_hash="msg-2",
                    sort_key=2.0,
                ),
            ],
        )
        stale = _session_data(
            "codex-session:duplicate",
            content_hash="hash-stale",
            raw_id="raw-stale",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:duplicate",
                    role="user",
                    text="first stale",
                    content_hash="msg-1-stale",
                    sort_key=1.0,
                )
            ],
        )

        changed_full, _counts_full = write_fixture_ingest_payload(conn, fuller)
        changed_stale, counts_stale = write_fixture_ingest_payload(conn, stale)
        conn.commit()

        messages = conn.execute(
            """
            SELECT b.text
            FROM messages m
            JOIN blocks b ON b.message_id = m.message_id
            WHERE m.session_id = ? AND b.block_type = 'text'
            ORDER BY m.position, b.position
            """,
            ("codex-session:duplicate",),
        ).fetchall()
        raw_id = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("codex-session:duplicate",),
        ).fetchone()["raw_id"]

        assert changed_full is True
        assert changed_stale is False
        assert counts_stale["skipped_sessions"] == 1
        assert [row["text"] for row in messages] == ["first", "second"]
        assert raw_id == "raw-full"


def test_write_session_skips_equal_count_duplicate_raw_source(tmp_path: Path) -> None:
    """Equal-count changed content is still fresher evidence and must update."""
    with open_connection(tmp_path / "index.db") as conn:
        existing = _session_data(
            "codex-session:duplicate-equal",
            content_hash="hash-existing",
            raw_id="raw-existing",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:duplicate-equal",
                    role="user",
                    text="first",
                    content_hash="msg-1",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex-session:duplicate-equal",
                    role="assistant",
                    text="second",
                    content_hash="msg-2",
                    sort_key=2.0,
                ),
            ],
        )
        duplicate = _session_data(
            "codex-session:duplicate-equal",
            content_hash="hash-duplicate-title-or-raw",
            raw_id="raw-duplicate",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:duplicate-equal",
                    role="user",
                    text="first duplicate",
                    content_hash="msg-1-duplicate",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex-session:duplicate-equal",
                    role="assistant",
                    text="second duplicate",
                    content_hash="msg-2-duplicate",
                    sort_key=2.0,
                ),
            ],
        )

        changed_existing, _counts_existing = write_fixture_ingest_payload(conn, existing)
        changed_duplicate, counts_duplicate = write_fixture_ingest_payload(conn, duplicate)
        conn.commit()

        messages = conn.execute(
            """
            SELECT b.text
            FROM messages m
            JOIN blocks b ON b.message_id = m.message_id
            WHERE m.session_id = ? AND b.block_type = 'text'
            ORDER BY m.position, b.position
            """,
            ("codex-session:duplicate-equal",),
        ).fetchall()
        raw_id = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("codex-session:duplicate-equal",),
        ).fetchone()["raw_id"]

        assert changed_existing is True
        assert changed_duplicate is True
        assert counts_duplicate["sessions"] == 1
        assert [row["text"] for row in messages] == ["first duplicate", "second duplicate"]
        assert raw_id == "raw-duplicate"


def test_write_session_dom_fallback_does_not_replace_native_source(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        native = _session_data(
            "codex-session:dom-precedence",
            content_hash="hash-native",
            raw_id="raw-native",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:dom-precedence",
                    role="user",
                    text="native first",
                    content_hash="msg-1",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex-session:dom-precedence",
                    role="assistant",
                    text="native second",
                    content_hash="msg-2",
                    sort_key=2.0,
                ),
            ],
        )
        dom_fallback = _session_data(
            "codex-session:dom-precedence",
            content_hash="hash-dom-fallback",
            raw_id="raw-dom-fallback",
            ingest_flags=[DOM_FALLBACK_INGEST_FLAG],
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:dom-precedence",
                    role="user",
                    text="fallback first",
                    content_hash="msg-1-fallback",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex-session:dom-precedence",
                    role="assistant",
                    text="fallback second",
                    content_hash="msg-2-fallback",
                    sort_key=2.0,
                ),
            ],
        )

        changed_native, _counts_native = write_fixture_ingest_payload(conn, native)
        changed_fallback, counts_fallback = write_fixture_ingest_payload(conn, dom_fallback)
        conn.commit()

        messages = conn.execute(
            """
            SELECT b.text
            FROM messages m
            JOIN blocks b ON b.message_id = m.message_id
            WHERE m.session_id = ? AND b.block_type = 'text'
            ORDER BY m.position, b.position
            """,
            ("codex-session:dom-precedence",),
        ).fetchall()
        raw_id = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("codex-session:dom-precedence",),
        ).fetchone()["raw_id"]

        assert changed_native is True
        assert changed_fallback is False
        assert counts_fallback["skipped_sessions"] == 1
        assert counts_fallback["session_events"] == 1
        assert [row["text"] for row in messages] == ["native first", "native second"]
        assert raw_id == "raw-native"
        event = conn.execute(
            """
            SELECT event_type, payload_json
            FROM session_events
            WHERE session_id = ?
            """,
            ("codex-session:dom-precedence",),
        ).fetchone()
        assert event["event_type"] == "capture_gap"
        assert "DOM browser-capture fallback" in event["payload_json"]
        assert "raw-dom-fallback" in event["payload_json"]


def test_write_session_same_content_dom_fallback_does_not_refresh_native_raw_link(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        native = _session_data(
            "codex-session:same-content-dom-precedence",
            content_hash="same-hash",
            raw_id="raw-native",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:same-content-dom-precedence",
                    role="user",
                    text="native first",
                    content_hash="msg-1",
                    sort_key=1.0,
                )
            ],
        )
        dom_fallback = _session_data(
            "codex-session:same-content-dom-precedence",
            content_hash="same-hash",
            raw_id="raw-dom-fallback",
            ingest_flags=[DOM_FALLBACK_INGEST_FLAG],
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:same-content-dom-precedence",
                    role="user",
                    text="native first",
                    content_hash="msg-1",
                    sort_key=1.0,
                )
            ],
        )

        changed_native, _counts_native = write_fixture_ingest_payload(conn, native)
        changed_fallback, counts_fallback = write_fixture_ingest_payload(conn, dom_fallback)
        conn.commit()

        raw_id = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("codex-session:same-content-dom-precedence",),
        ).fetchone()["raw_id"]
        capture_gap_count = conn.execute(
            "SELECT COUNT(*) FROM session_events WHERE session_id = ? AND event_type = 'capture_gap'",
            ("codex-session:same-content-dom-precedence",),
        ).fetchone()[0]

        assert changed_native is True
        assert changed_fallback is False
        assert counts_fallback["skipped_sessions"] == 1
        assert counts_fallback["session_events"] == 1
        assert counts_fallback["raw_links"] == 0
        assert raw_id == "raw-native"
        assert capture_gap_count == 1


def test_write_session_native_source_replaces_dom_fallback_even_when_shorter(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        dom_fallback = _session_data(
            "codex-session:native-over-dom",
            content_hash="hash-dom-fallback",
            raw_id="raw-dom-fallback",
            ingest_flags=[DOM_FALLBACK_INGEST_FLAG],
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex-session:native-over-dom",
                    role="user",
                    text="fallback first",
                    content_hash="msg-1-fallback",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex-session:native-over-dom",
                    role="assistant",
                    text="fallback second",
                    content_hash="msg-2-fallback",
                    sort_key=2.0,
                ),
            ],
        )
        native = _session_data(
            "codex-session:native-over-dom",
            content_hash="hash-native",
            raw_id="raw-native",
            message_tuples=[
                _message_tuple(
                    "msg-native",
                    "codex-session:native-over-dom",
                    role="assistant",
                    text="native body",
                    content_hash="msg-native",
                    sort_key=3.0,
                )
            ],
        )

        changed_fallback, _counts_fallback = write_fixture_ingest_payload(conn, dom_fallback)
        changed_native, counts_native = write_fixture_ingest_payload(conn, native)
        conn.commit()

        messages = conn.execute(
            """
            SELECT b.text
            FROM messages m
            JOIN blocks b ON b.message_id = m.message_id
            WHERE m.session_id = ? AND b.block_type = 'text'
            ORDER BY m.position, b.position
            """,
            ("codex-session:native-over-dom",),
        ).fetchall()
        raw_id = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            ("codex-session:native-over-dom",),
        ).fetchone()["raw_id"]

        assert changed_fallback is True
        assert changed_native is True
        assert counts_native["sessions"] == 1
        assert [row["text"] for row in messages] == ["native body"]
        assert raw_id == "raw-native"


# A genuine, non-browser-capture arrival ("export" below) always outranks a
# native browser capture, and vice versa -- regardless of arrival order and
# regardless of message count (polylogue-z1c6 + review follow-up). This
# supersedes an earlier "richer/tied content wins, ties favor native" policy:
# browser capture exists only to backfill a session before its direct/native
# provider export shows up, never to compete with or shadow that export once
# it arrives. Mirrors the equivalent matrix in
# tests/unit/storage/test_archive_tiers_archive.py for the ArchiveStore
# facade write path; this one exercises the ingest_batch worker write path.
@pytest.mark.parametrize(
    (
        "initial_kind",
        "initial_count",
        "incoming_kind",
        "incoming_count",
        "expected_title",
        "expected_count",
        "expected_raw_id",
        "incoming_is_older",
    ),
    [
        ("native", 2, "export", 2, "Ordinary export", 2, "raw-incoming", False),
        ("export", 2, "native", 2, "Ordinary export", 2, "raw-initial", False),
        ("native", 2, "export", 1, "Ordinary export", 1, "raw-incoming", False),
        ("export", 1, "native", 2, "Ordinary export", 1, "raw-initial", False),
        ("native", 2, "export", 3, "Fuller export", 3, "raw-incoming", False),
        ("export", 3, "native", 2, "Fuller export", 3, "raw-initial", False),
        ("export", 2, "native", 2, "Ordinary export", 2, "raw-initial", True),
    ],
    ids=(
        "native-before-equal",
        "native-after-equal",
        "native-before-weaker",
        "native-after-weaker",
        "fuller-export-advances",
        "fuller-export-resists-shorter-native",
        "older-native-after-equal",
    ),
)
def test_write_session_native_browser_precedence_matrix(
    tmp_path: Path,
    initial_kind: str,
    initial_count: int,
    incoming_kind: str,
    incoming_count: int,
    expected_title: str,
    expected_count: int,
    expected_raw_id: str,
    incoming_is_older: bool,
) -> None:
    session_id = f"chatgpt-export:browser-precedence-{initial_kind}-{initial_count}-{incoming_kind}-{incoming_count}"

    def payload(kind: str, message_count: int, raw_id: str, *, updated_at: str) -> SessionWritePayload:
        native = kind == "native"
        title = "Native browser" if native else ("Fuller export" if message_count == 3 else "Ordinary export")
        return _session_data(
            session_id,
            content_hash=f"{kind}-{raw_id}-{message_count}",
            raw_id=raw_id,
            provider=Provider.CHATGPT,
            title=title,
            updated_at=updated_at,
            ingest_flags=[NATIVE_BROWSER_CAPTURE_INGEST_FLAG] if native else [],
            message_tuples=[
                _message_tuple(
                    f"{kind}-{position}",
                    session_id,
                    role="user" if position == 0 else "assistant",
                    text=f"{kind} message {position}",
                    content_hash=f"{kind}-{position}",
                    sort_key=float(position),
                )
                for position in range(message_count)
            ],
        )

    with open_connection(tmp_path / "index.db") as conn:
        changed_initial, _counts_initial = write_fixture_ingest_payload(
            conn,
            payload(initial_kind, initial_count, "raw-initial", updated_at="2026-04-03T00:00:00Z"),
        )
        changed_incoming, counts_incoming = write_fixture_ingest_payload(
            conn,
            payload(
                incoming_kind,
                incoming_count,
                "raw-incoming",
                updated_at="2026-04-02T00:00:00Z" if incoming_is_older else "2026-04-03T00:00:00Z",
            ),
        )
        conn.commit()
        stored = conn.execute(
            "SELECT raw_id, title, message_count FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    assert changed_initial is True
    assert changed_incoming is (expected_raw_id == "raw-incoming")
    assert counts_incoming["skipped_sessions"] == int(expected_raw_id == "raw-initial")
    assert dict(stored) == {
        "raw_id": expected_raw_id,
        "title": expected_title,
        "message_count": expected_count,
    }


@pytest.mark.parametrize(
    ("arrivals", "expected_title", "expected_native_flag", "expected_content_changed"),
    [
        (
            (
                ("native", 2, "Native browser", "2026-04-01T00:00:00Z"),
                ("export", 3, "Fuller export", "2026-04-02T00:00:00Z"),
                ("export", 3, "Newest export", "2026-04-03T00:00:00Z"),
            ),
            "Newest export",
            False,
            (True, True, True),
        ),
        (
            (
                ("export", 2, "Newer export", "2026-04-03T00:00:00Z"),
                ("native", 2, "Older native", "2026-04-01T00:00:00Z"),
                ("native", 2, "Native update", "2026-04-02T00:00:00Z"),
            ),
            "Newer export",
            False,
            # A genuine export, once established, resists every later native
            # arrival unconditionally (polylogue-z1c6 review follow-up) --
            # neither later native arrival changes content.
            (True, False, False),
        ),
    ],
    ids=("owner-flag-is-replaced", "established-export-resists-later-native-arrivals"),
)
def test_write_session_browser_precedence_tracks_three_arrivals(
    tmp_path: Path,
    arrivals: tuple[tuple[str, int, str, str], ...],
    expected_title: str,
    expected_native_flag: bool,
    expected_content_changed: tuple[bool, bool, bool],
) -> None:
    session_id = f"chatgpt-export:browser-three-arrivals-{expected_title.lower().replace(' ', '-')}"

    def payload(kind: str, count: int, title: str, updated_at: str, raw_id: str) -> SessionWritePayload:
        return _session_data(
            session_id,
            content_hash=f"{raw_id}-{title}",
            raw_id=raw_id,
            provider=Provider.CHATGPT,
            title=title,
            updated_at=updated_at,
            ingest_flags=[NATIVE_BROWSER_CAPTURE_INGEST_FLAG] if kind == "native" else [],
            message_tuples=[
                _message_tuple(
                    f"{raw_id}-{position}",
                    session_id,
                    role="user" if position == 0 else "assistant",
                    text=f"{title} message {position}",
                    content_hash=f"{raw_id}-{position}",
                    sort_key=float(position),
                )
                for position in range(count)
            ],
        )

    with open_connection(tmp_path / "index.db") as conn:
        outcomes = [
            write_fixture_ingest_payload(conn, payload(kind, count, title, updated_at, f"raw-{index}"))
            for index, (kind, count, title, updated_at) in enumerate(arrivals)
        ]
        conn.commit()
        stored = conn.execute(
            "SELECT raw_id, title FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        native_flag = conn.execute(
            "SELECT 1 FROM session_tags WHERE session_id = ? AND tag = ?",
            (session_id, NATIVE_BROWSER_CAPTURE_INGEST_FLAG),
        ).fetchone()

    assert [changed for changed, _counts in outcomes] == list(expected_content_changed)
    owner_index = max(index for index, changed in enumerate(expected_content_changed) if changed)
    assert dict(stored) == {"raw_id": f"raw-{owner_index}", "title": expected_title}
    assert (native_flag is not None) is expected_native_flag


def test_write_session_skips_new_with_zero_messages(tmp_path: Path) -> None:
    """A new session with zero messages is skipped, not left as a manifest-only row."""
    with open_connection(tmp_path / "index.db") as conn:
        empty = _session_data(
            "codex-session:empty-manifest",
            content_hash="hash-empty",
            message_tuples=[],
        )
        changed, counts = write_fixture_ingest_payload(conn, empty)
        conn.commit()

        # Verify skipped
        assert changed is False
        assert counts["skipped_sessions"] == 1

        # Verify no row was created
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            ("codex-session:empty-manifest",),
        ).fetchone()
        assert row is None


def test_write_session_allows_existing_upsert_even_without_messages(tmp_path: Path) -> None:
    """An existing session with a changed hash and zero new messages is still upserted.
    The guard only blocks *new* sessions from being created without messages.
    Replacing existing content with empty content is a legitimate content update.
    """
    with open_connection(tmp_path / "index.db") as conn:
        msg = _message_tuple(
            "msg-1",
            "codex-session:keep",
            role="user",
            text="hello",
            content_hash="hash-msg",
            sort_key=1.0,
        )
        first = _session_data(
            "codex-session:keep",
            content_hash="hash-1",
            message_tuples=[msg],
        )
        write_fixture_ingest_payload(conn, first)
        conn.commit()

        # Same session, different hash, zero messages — should be allowed
        update = _session_data(
            "codex-session:keep",
            content_hash="hash-2",
            message_tuples=[],
        )
        changed, counts = write_fixture_ingest_payload(conn, update)
        conn.commit()

        assert changed is True
        assert counts["skipped_sessions"] == 0


def test_write_session_allows_rewrite_of_its_own_accepted_revision_head(tmp_path: Path) -> None:
    """polylogue-buq8/i415/lkos: a write for the raw ``raw_revision_heads``
    itself names as the accepted authority for this session must never be
    refused, even though the session already has a governed head.

    Live-archive reproduction (2026-07-31): 11 ``codex-session`` rows carry
    ``message_count=0`` despite 996 KB-3.3 MB of real ``event_msg``/
    ``response_item`` content in their raw bytes. Every one has
    ``raw_revision_heads.accepted_raw_id`` equal to its own
    ``sessions.raw_id`` -- ``decided_at_ms`` for the governance row postdates
    ``sessions.updated_at_ms`` by months, proving a later bookkeeping-only
    backfill (recomputing authority for a pre-governance session) recorded
    the raw as authoritative without re-running message extraction against
    it. Direct reproduction of ``parse_stream_payload`` against the exact
    live raw bytes of the flagship sample (native_id
    ``357c7da6-8703-4ba3-8f70-f5f253571c12``) confirms the current codex
    parser already extracts 1,496 real messages correctly -- the content
    loss was never a parser defect in ``sources/``. It was
    ``revision_authority_refuses_write``'s ``governed`` check refusing
    *every* write once ``raw_revision_heads`` had any row for the
    ``session_id``, with no comparison against which raw the row actually
    names -- permanently freezing whatever content the pre-governance write
    happened to leave behind, including zero messages, because the very raw
    the backfill just declared authoritative could never pass its own gate.

    Mutation that fails this: reverting the ``accepted_raw_id`` comparison
    back to a bare existence check (``governed is not None: return True``).
    """
    with open_connection(tmp_path / "index.db") as conn:
        # Simulate the historical defect: a session was written (long ago,
        # ``force_write=True`` standing in for whatever historical write
        # path/bug left this content behind) with zero messages for raw
        # "raw-accepted", then a bookkeeping-only backfill later recorded
        # that same raw as the accepted revision head without ever
        # re-writing the session's content.
        stub = _session_data(
            "codex-session:frozen-empty",
            content_hash="hash-stub-empty",
            raw_id="raw-accepted",
            message_tuples=[],
        )
        write_fixture_ingest_payload(conn, stub, force_write=True)
        conn.execute(
            "INSERT INTO raw_revision_heads (logical_source_key, session_id, accepted_raw_id, "
            "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, decided_at_ms) VALUES "
            "('codex:frozen-empty','codex-session:frozen-empty','raw-accepted','sr',?,'byte',1,0,1)",
            (b"\x09" * 32,),
        )
        conn.commit()

        real_msg = _message_tuple(
            "msg-real",
            "codex-session:frozen-empty",
            role="user",
            text="the real conversation content",
            content_hash="hash-real",
            sort_key=1.0,
        )
        corrective = _session_data(
            "codex-session:frozen-empty",
            content_hash="hash-corrected",
            raw_id="raw-accepted",
            message_tuples=[real_msg],
        )
        changed, counts = write_fixture_ingest_payload(conn, corrective)
        conn.commit()

        stored = conn.execute(
            "SELECT message_count FROM sessions WHERE session_id = ?",
            ("codex-session:frozen-empty",),
        ).fetchone()

    assert changed is True
    assert counts["skipped_sessions"] == 0
    assert stored["message_count"] == 1


def test_write_session_still_refuses_a_different_raw_than_the_accepted_head(tmp_path: Path) -> None:
    """The mirror of the fix above: a *different*, non-accepted raw for a
    governed session_id must still be refused -- the accepted-raw carve-out
    must not reopen the door to an arbitrary competing/losing raw
    overwriting the winner."""
    with open_connection(tmp_path / "index.db") as conn:
        winner = _session_data(
            "codex-session:governed",
            content_hash="hash-winner",
            raw_id="raw-winner",
            message_tuples=[
                _message_tuple(
                    "msg-winner",
                    "codex-session:governed",
                    role="user",
                    text="winning content",
                    content_hash="hash-winner-msg",
                    sort_key=1.0,
                )
            ],
        )
        write_fixture_ingest_payload(conn, winner)
        conn.execute(
            "INSERT INTO raw_revision_heads (logical_source_key, session_id, accepted_raw_id, "
            "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, decided_at_ms) VALUES "
            "('codex:governed','codex-session:governed','raw-winner','sr',?,'byte',1,0,1)",
            (b"\x0a" * 32,),
        )
        conn.commit()

        loser = _session_data(
            "codex-session:governed",
            content_hash="hash-loser",
            raw_id="raw-loser",
            message_tuples=[
                _message_tuple(
                    "msg-loser",
                    "codex-session:governed",
                    role="user",
                    text="losing content",
                    content_hash="hash-loser-msg",
                    sort_key=1.0,
                )
            ],
        )
        changed, counts = write_fixture_ingest_payload(conn, loser)
        conn.commit()

        stored = conn.execute(
            "SELECT raw_id, message_count FROM sessions WHERE session_id = ?",
            ("codex-session:governed",),
        ).fetchone()

    assert changed is False
    assert counts["skipped_sessions"] == 1
    assert dict(stored) == {"raw_id": "raw-winner", "message_count": 1}


def test_write_session_refuses_when_a_parallel_head_accepts_a_different_raw(tmp_path: Path) -> None:
    """P2 bot finding on PR #3527: with more than one ``raw_revision_heads``
    row for the same ``session_id`` (historical drift or an interrupted
    repair -- ``storage/raw_convergence.py``'s ``parallel_session_heads`` shape), the
    refusal decision must not depend on which row a bare ``LIMIT 1``
    happened to select. Two heads for this session: one already accepts
    the incoming raw, the other accepts a different raw -- the write must
    still be refused, regardless of which row is inserted (and therefore
    selected) first.

    Mutation that fails this: reverting to ``SELECT accepted_raw_id ...
    LIMIT 1`` and comparing only that single arbitrary row.
    """
    with open_connection(tmp_path / "index.db") as conn:
        conn.execute(
            "INSERT INTO raw_revision_heads (logical_source_key, session_id, accepted_raw_id, "
            "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, decided_at_ms) VALUES "
            "('codex:parallel-a','codex-session:parallel','raw-incoming','sr',?,'byte',1,0,1)",
            (b"\x0b" * 32,),
        )
        conn.execute(
            "INSERT INTO raw_revision_heads (logical_source_key, session_id, accepted_raw_id, "
            "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, decided_at_ms) VALUES "
            "('codex:parallel-b','codex-session:parallel','raw-other','sr',?,'byte',1,0,1)",
            (b"\x0c" * 32,),
        )
        conn.commit()

        incoming = _session_data(
            "codex-session:parallel",
            content_hash="hash-incoming",
            raw_id="raw-incoming",
            message_tuples=[
                _message_tuple(
                    "msg-incoming",
                    "codex-session:parallel",
                    role="user",
                    text="incoming content",
                    content_hash="hash-incoming-msg",
                    sort_key=1.0,
                )
            ],
        )
        changed, counts = write_fixture_ingest_payload(conn, incoming)
        conn.commit()

        stored = conn.execute(
            "SELECT message_count FROM sessions WHERE session_id = ?",
            ("codex-session:parallel",),
        ).fetchone()

    assert changed is False
    assert counts["skipped_sessions"] == 1
    # Refused before ever inserting: this was the session's first write attempt.
    assert stored is None


def test_write_session_refuses_a_raw_recorded_ambiguous_membership(tmp_path: Path) -> None:
    """``_write_session`` -- the daemon's default batch-ingest write path,
    used for most non-drive origins -- must refuse a session whose OWN
    ``raw_session_memberships.decision`` is recorded ``'ambiguous'``.

    This mirrors ``ArchiveStore._write_parsed_precedence_result``'s guard
    (#3397/#3398, polylogue-c737). Before this fix, this path had zero
    ``raw_session_memberships`` awareness: its only revision-authority check
    was against ``raw_revision_heads``, populated ONLY when a cohort has an
    ACCEPTED winner. A cohort ``classify_membership_revisions`` genuinely
    refused to arbitrate never gets an accepted head, so that check stayed
    silent and the ordinary freshness/precedence fallback below it wrote the
    session unconditionally on the raw's next reparse -- arbitrary
    last-writer-wins over the recorded verdict.

    Scoped per-membership (``raw_id`` AND ``provider_session_id``), not
    per-raw: one retained raw routinely lowers to many independently-
    arbitrated sessions (a Claude Code transcript plus its subagent
    sidechains, a bundle member set). #3398 measured 295 raws carrying a mix
    of decisions on the live archive, together holding 489 sessions whose own
    membership is NOT ambiguous. This test builds that exact shape -- one
    raw, two memberships, one ``ambiguous`` and one ``applied`` -- and
    asserts BOTH halves, so a raw-scoped predicate (which would refuse both)
    fails it just as it would have fixed nothing for the settled sibling.
    """
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    raw_id = "abcd1234abcd1234"
    source_db_path = archive_root / "source.db"

    with sqlite3.connect(str(source_db_path)) as source_setup_conn:
        source_setup_conn.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id,
                source_revision, normalized_content_hash, message_count,
                decision, decided_at_ms
            ) VALUES (?, 'codex:s-ambiguous', 's-ambiguous', 'rev-1', ?, 1, 'ambiguous', 1)
            """,
            (raw_id, b"0" * 32),
        )
        source_setup_conn.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id,
                source_revision, normalized_content_hash, message_count,
                decision, decided_at_ms
            ) VALUES (?, 'codex:s-settled', 's-settled', 'rev-1', ?, 1, 'applied', 1)
            """,
            (raw_id, b"1" * 32),
        )
        source_setup_conn.commit()

    with (
        open_connection(archive_root / "index.db") as conn,
        sqlite3.connect(str(source_db_path)) as source_conn,
    ):
        ambiguous_msg = _message_tuple(
            "a-0",
            "codex-session:s-ambiguous",
            role="user",
            text="left",
            content_hash="hash-a",
            sort_key=1.0,
        )
        settled_msg = _message_tuple(
            "b-0",
            "codex-session:s-settled",
            role="user",
            text="right",
            content_hash="hash-b",
            sort_key=1.0,
        )
        ambiguous_payload = _session_data(
            "codex-session:s-ambiguous",
            content_hash="hash-ambiguous",
            raw_id=raw_id,
            message_tuples=[ambiguous_msg],
        )
        settled_payload = _session_data(
            "codex-session:s-settled",
            content_hash="hash-settled",
            raw_id=raw_id,
            message_tuples=[settled_msg],
        )

        changed_ambiguous, counts_ambiguous = write_fixture_ingest_payload(
            conn, ambiguous_payload, source_conn=source_conn
        )
        changed_settled, counts_settled = write_fixture_ingest_payload(conn, settled_payload, source_conn=source_conn)
        conn.commit()

    assert changed_ambiguous is False
    assert counts_ambiguous["skipped_sessions"] == 1

    assert changed_settled is True
    assert counts_settled["skipped_sessions"] == 0

    with sqlite3.connect(str(archive_root / "index.db")) as verify_conn:
        # The ambiguous membership is still refused ...
        assert verify_conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", ("codex-session:s-ambiguous",)
        ).fetchone() == (0,)
        # ... and its settled sibling on the same raw is not collateral damage.
        assert verify_conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", ("codex-session:s-settled",)
        ).fetchone() == (1,)


def test_drain_ready_session_entries_writes_missing_parent_without_buffering(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        c_msg = _message_tuple(
            "msg-c",
            "codex-session:child",
            role="user",
            text="child",
            content_hash="hash-c",
            sort_key=1.0,
        )
        child = _session_data(
            "codex-session:child",
            content_hash="hash-child",
            parent_session_id="codex-session:parent",
            message_tuples=[c_msg],
        )

        summary = _IngestBatchSummary()
        materialized_ids: set[str] = set()

        _drain_ready_session_entries(
            conn,
            [("raw-child", child)],
            summary=summary,
            materialized_ids=materialized_ids,
        )
        conn.commit()

        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = ?",
            ("codex-session:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_session_id"] is None
        assert child.parsed_session.messages == []


def test_drain_ready_session_entries_preserves_same_result_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "index.db") as conn:
        p_msg = _message_tuple(
            "msg-p",
            "codex-session:parent",
            role="user",
            text="parent",
            content_hash="hash-p",
            sort_key=1.0,
        )
        c_msg = _message_tuple(
            "msg-c",
            "codex-session:child",
            role="user",
            text="child",
            content_hash="hash-c",
            sort_key=1.0,
        )
        parent = _session_data(
            "codex-session:parent",
            content_hash="hash-parent",
            message_tuples=[p_msg],
        )
        child = _session_data(
            "codex-session:child",
            content_hash="hash-child",
            parent_session_id="codex-session:parent",
            message_tuples=[c_msg],
        )

        _drain_ready_session_entries(
            conn,
            [("raw-child", child), ("raw-parent", parent)],
            summary=_IngestBatchSummary(),
            materialized_ids=set(),
        )
        conn.commit()

        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = ?",
            ("codex-session:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_session_id"] == "codex-session:parent"


def test_successful_raw_state_update_combines_parse_and_validation_fields() -> None:
    outcome = _RawIngestOutcome(
        raw_id="raw-1",
        payload_provider="chatgpt",
        validation_status="passed",
        validation_error=None,
        parse_error=None,
        error=None,
        had_sessions=True,
    )

    state = _successful_raw_state_update(
        outcome=outcome,
        parsed_at="2026-04-02T00:00:00Z",
        validation_mode="strict",
    )

    assert state == RawSessionStateUpdate(
        parsed_at="2026-04-02T00:00:00Z",
        parse_error=None,
        payload_provider="chatgpt",
        validation_status="passed",
        validation_error=None,
        validation_mode="strict",
    )


def test_failed_raw_state_update_combines_parse_and_validation_fields() -> None:
    outcome = _RawIngestOutcome(
        raw_id="raw-1",
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        parse_error="parse failed",
        error="parse failed",
        had_sessions=False,
    )

    state = _failed_raw_state_update(
        outcome=outcome,
        error="parse failed",
        validation_mode="strict",
    )

    assert state == RawSessionStateUpdate(
        parse_error="parse failed",
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        validation_mode="strict",
        detection_warnings="parse failed",
    )


def test_failed_raw_state_update_keeps_validation_only_failure_out_of_parse_error() -> None:
    outcome = _RawIngestOutcome(
        raw_id="raw-1",
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        parse_error=None,
        error="schema mismatch",
        had_sessions=False,
    )

    state = _failed_raw_state_update(
        outcome=outcome,
        error="schema mismatch",
        validation_mode="strict",
    )

    assert state == RawSessionStateUpdate(
        parse_error=None,
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        validation_mode="strict",
        detection_warnings=None,
    )


def test_failed_raw_state_update_persists_worker_diagnostic_at_boundary() -> None:
    outcome = _RawIngestOutcome(
        raw_id="raw-1",
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        parse_error="parse failed",
        error="parse failed",
        had_sessions=False,
        outcome_code="validation_rejected",
        retryable=False,
        evidence_ref="schema_validation_strict",
        remediation="repair source schema",
        diagnostic="missing required field: messages",
    )

    state = _failed_raw_state_update(
        outcome=outcome,
        error="parse failed",
        validation_mode="strict",
    )

    assert state.detection_warnings == "missing required field: messages"


def test_unattributed_batch_elapsed_subtracts_setup_and_teardown() -> None:
    summary = _IngestBatchSummary(
        setup_elapsed_s=0.12,
        result_wait_s=0.8,
        drain_elapsed_s=0.2,
        flush_elapsed_s=0.05,
        commit_elapsed_s=0.04,
        teardown_elapsed_s=0.31,
    )

    residual = _unattributed_batch_elapsed_s(
        elapsed_s=1.7,
        batch_summary=summary,
        raw_state_update_elapsed_s=0.08,
    )

    assert residual == pytest.approx(0.10)


def test_build_batch_memory_observation_separates_lifetime_peak_from_batch_growth() -> None:
    observation = _build_batch_memory_observation(
        rss_start_mb=512.0,
        rss_end_mb=544.5,
        peak_rss_self_start_mb=768.0,
        peak_rss_self_end_mb=1024.0,
        peak_rss_children_mb=64.0,
        max_current_rss_mb=812.2,
    )

    assert observation == {
        "rss_start_mb": 512.0,
        "rss_end_mb": 544.5,
        "rss_delta_mb": 32.5,
        "process_peak_rss_self_mb": 1024.0,
        "peak_rss_growth_mb": 256.0,
        "peak_rss_children_mb": 64.0,
        "max_current_rss_mb": 812.2,
    }


@pytest.mark.asyncio
async def test_persist_batch_raw_state_updates_uses_one_typed_update_per_raw() -> None:
    update_raw_state = AsyncMock()
    service = _FakeParsingService(update_raw_state)

    @asynccontextmanager
    async def _bulk_connection() -> AsyncIterator[None]:
        yield

    backend = _FakeBulkBackend(_bulk_connection)
    outcomes = {
        "raw-success": _RawIngestOutcome(
            raw_id="raw-success",
            payload_provider="chatgpt",
            validation_status="passed",
            validation_error=None,
            parse_error=None,
            error=None,
            had_sessions=True,
        ),
        "raw-failed": _RawIngestOutcome(
            raw_id="raw-failed",
            payload_provider="codex",
            validation_status="failed",
            validation_error="bad schema",
            parse_error="parse failed",
            error="parse failed",
            had_sessions=False,
        ),
    }

    elapsed_s = await _persist_batch_raw_state_updates(
        service,
        backend,
        outcomes=outcomes,
        succeeded_raw_ids={"raw-success"},
        skipped_raw_ids=set(),
        failed_raw_ids={"raw-failed": "parse failed"},
        validation_mode="strict",
    )

    assert elapsed_s >= 0.0
    assert update_raw_state.await_count == 2
    success_call, failed_call = update_raw_state.await_args_list
    assert success_call.args == ("raw-success",)
    assert success_call.kwargs["state"].validation_status == "passed"
    assert success_call.kwargs["state"].parsed_at is not None
    assert failed_call.args == ("raw-failed",)
    assert failed_call.kwargs["state"].parse_error == "parse failed"
    assert failed_call.kwargs["state"].validation_error == "bad schema"


@pytest.mark.asyncio
async def test_persist_batch_raw_state_updates_marks_skipped_raw_before_success() -> None:
    update_raw_state = AsyncMock()
    service = _FakeParsingService(update_raw_state)

    @asynccontextmanager
    async def _bulk_connection() -> AsyncIterator[None]:
        yield

    backend = _FakeBulkBackend(_bulk_connection)
    outcomes = {
        "raw-duplicate": _RawIngestOutcome(
            raw_id="raw-duplicate",
            payload_provider="chatgpt",
            validation_status="passed",
            validation_error=None,
            parse_error=None,
            error=None,
            had_sessions=True,
        ),
    }

    elapsed_s = await _persist_batch_raw_state_updates(
        service,
        backend,
        outcomes=outcomes,
        succeeded_raw_ids={"raw-duplicate"},
        skipped_raw_ids={"raw-duplicate"},
        failed_raw_ids={},
        validation_mode="advisory",
    )

    assert elapsed_s >= 0.0
    update_raw_state.assert_awaited_once()
    call = update_raw_state.await_args
    assert call is not None
    assert call.args == ("raw-duplicate",)
    state = call.kwargs["state"]
    assert state.validation_status == "skipped"
    assert state.validation_error == "parsed raw payload produced no new materialized sessions"
    assert state.parse_error is None
    assert state.parsed_at is not None


@pytest.mark.asyncio
async def test_persist_batch_raw_state_updates_preserves_validation_only_failure_without_quarantine() -> None:
    update_raw_state = AsyncMock()
    service = _FakeParsingService(update_raw_state)

    @asynccontextmanager
    async def _bulk_connection() -> AsyncIterator[None]:
        yield

    backend = _FakeBulkBackend(_bulk_connection)
    outcomes = {
        "raw-schema-invalid": _RawIngestOutcome(
            raw_id="raw-schema-invalid",
            payload_provider="chatgpt",
            validation_status="failed",
            validation_error="bad schema",
            parse_error=None,
            error="bad schema",
            had_sessions=False,
        ),
    }

    elapsed_s = await _persist_batch_raw_state_updates(
        service,
        backend,
        outcomes=outcomes,
        succeeded_raw_ids=set(),
        skipped_raw_ids=set(),
        failed_raw_ids={"raw-schema-invalid": "bad schema"},
        validation_mode="strict",
    )

    assert elapsed_s >= 0.0
    update_raw_state.assert_awaited_once()
    await_args = update_raw_state.await_args
    assert await_args is not None
    state = await_args.kwargs["state"]
    assert state.parse_error is None
    assert state.validation_error == "bad schema"
    assert state.validation_status == "failed"


@pytest.mark.asyncio
async def test_persist_batch_raw_state_updates_persists_terminal_worker_disposition_to_source(
    tmp_path: Path,
) -> None:
    """The ordinary batch boundary retains typed terminal evidence at the raw coordinate."""
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="batch-unsupported.jsonl",
            source_index=7,
            payload=b"unsupported-shape",
            acquired_at_ms=1,
        )

    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = SimpleNamespace(repository=repository)
    outcome = _RawIngestOutcome(
        raw_id=raw_id,
        payload_provider="codex",
        validation_status="passed",
        validation_error=None,
        parse_error="parse: session artifact produced no materializable sessions",
        error="parse: session artifact produced no materializable sessions",
        had_sessions=False,
        outcome_code="unsupported_shape",
        retryable=False,
        evidence_ref="empty_parsed_sessions",
        remediation="open a source-support issue",
        diagnostic="worker rejected unsupported shape",
    )

    try:
        await _persist_batch_raw_state_updates(
            service,
            repository.backend,
            outcomes={raw_id: outcome},
            succeeded_raw_ids=set(),
            skipped_raw_ids=set(),
            failed_raw_ids={raw_id: outcome.error or "worker failure"},
            validation_mode="strict",
        )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        state = conn.execute(
            "SELECT parse_error, source_path, source_index FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
        artifact = conn.execute(
            """
            SELECT raw_id, origin, source_path, source_index, artifact_kind, support_status,
                   classification_reason, decode_error
            FROM raw_artifacts
            WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchone()

    assert state == (outcome.parse_error, "batch-unsupported.jsonl", 7)
    assert artifact is not None
    assert tuple(artifact[:6]) == (
        raw_id,
        Origin.CODEX_SESSION.value,
        "batch-unsupported.jsonl",
        7,
        RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
        "unsupported_parseable",
    )
    carrier = json.loads(artifact[6])
    assert carrier == {
        "diagnostic": outcome.diagnostic,
        "evidence_ref": outcome.evidence_ref,
        "outcome_code": outcome.outcome_code,
        "remediation": outcome.remediation,
        "retryable": False,
    }
    assert artifact[7] == outcome.diagnostic

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0
    assert lifecycle.blocking is False
    status = raw_failure_info_for_root(tmp_path)
    assert status["terminal_rejections"] == 1
    assert status["unexplained_failures"] == 0
    samples = cast(list[RawFailureSample], status["samples"])
    assert samples[0].failure_kind == RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value
    assert lifecycle.state == "degraded"


@pytest.mark.asyncio
async def test_persist_batch_success_supersedes_deferred_cas_evidence_in_source_transaction(
    tmp_path: Path,
) -> None:
    """The async batch success route revokes stale CAS replay authority."""
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="batch-success.jsonl",
            source_index=2,
            payload=b"batch-success",
            acquired_at_ms=1,
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="deferred-cas",
                origin=Origin.CODEX_SESSION,
                source_path="batch-success.jsonl",
                source_index=2,
                artifact_kind=RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
                classification_reason="deferred CAS",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                parse_as_session=True,
                schema_eligible=True,
                first_observed_at_ms=1,
                last_observed_at_ms=1,
            ),
        )

    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = SimpleNamespace(repository=repository)
    outcome = _RawIngestOutcome(
        raw_id=raw_id,
        payload_provider="codex",
        validation_status="passed",
        validation_error=None,
        parse_error=None,
        error=None,
        had_sessions=True,
    )
    try:
        await _persist_batch_raw_state_updates(
            service,
            repository.backend,
            outcomes={raw_id: outcome},
            succeeded_raw_ids={raw_id},
            skipped_raw_ids=set(),
            failed_raw_ids={},
            validation_mode="strict",
        )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT artifact_id, artifact_kind, support_status FROM raw_artifacts WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == (
            "deferred-cas",
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
        )
        assert (
            conn.execute("SELECT parse_error, parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
            is None
        )

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.deferred == 0
    assert lifecycle.terminal == 0
    assert lifecycle.unexplained == 0


@pytest.mark.asyncio
async def test_persist_batch_untyped_failure_retires_stale_terminal_evidence(tmp_path: Path) -> None:
    """A later untyped parser failure cannot inherit an older terminal cause."""
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="untyped-successor.jsonl",
            source_index=0,
            payload=b"untyped-successor",
            acquired_at_ms=1,
        )

    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = SimpleNamespace(repository=repository)
    terminal_outcome = _RawIngestOutcome(
        raw_id=raw_id,
        payload_provider="codex",
        validation_status="passed",
        validation_error=None,
        parse_error="unsupported shape",
        error="unsupported shape",
        had_sessions=False,
        outcome_code="unsupported_shape",
        evidence_ref="shape",
        remediation="support it",
        diagnostic="unsupported shape",
    )
    untyped_outcome = _RawIngestOutcome(
        raw_id=raw_id,
        payload_provider="codex",
        validation_status="passed",
        validation_error=None,
        parse_error="parser defect",
        error="parser defect",
        had_sessions=False,
        outcome_code="parser_defect",
        diagnostic="parser defect",
    )
    try:
        await _persist_batch_raw_state_updates(
            service,
            repository.backend,
            outcomes={raw_id: terminal_outcome},
            succeeded_raw_ids=set(),
            skipped_raw_ids=set(),
            failed_raw_ids={raw_id: terminal_outcome.error or "failure"},
            validation_mode="strict",
        )
        await _persist_batch_raw_state_updates(
            service,
            repository.backend,
            outcomes={raw_id: untyped_outcome},
            succeeded_raw_ids=set(),
            skipped_raw_ids=set(),
            failed_raw_ids={raw_id: untyped_outcome.error or "failure"},
            validation_mode="strict",
        )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
        )
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 0
    assert lifecycle.unexplained == 1
    assert lifecycle.blocking is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "diagnostic"),
    [
        (b"", "decode: Input is a zero-length, empty document"),
        (b"{", "decode: Input data was truncated"),
    ],
    ids=["zero-length", "decode-failure"],
)
async def test_persist_batch_corrupt_input_remains_terminal_in_lifecycle(
    tmp_path: Path,
    payload: bytes,
    diagnostic: str,
) -> None:
    """Worker validation failure plus typed corrupt evidence is explainable."""
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="batch-corrupt.jsonl",
            source_index=0,
            payload=payload,
            acquired_at_ms=1,
        )

    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = SimpleNamespace(repository=repository)
    outcome = _RawIngestOutcome(
        raw_id=raw_id,
        payload_provider="codex",
        validation_status="failed",
        validation_error="payload failed validation",
        parse_error=diagnostic,
        error=diagnostic,
        had_sessions=False,
        outcome_code="corrupt_input",
        retryable=False,
        evidence_ref="decode",
        remediation="retain and inspect raw bytes",
        diagnostic=diagnostic,
    )
    try:
        await _persist_batch_raw_state_updates(
            service,
            repository.backend,
            outcomes={raw_id: outcome},
            succeeded_raw_ids=set(),
            skipped_raw_ids=set(),
            failed_raw_ids={raw_id: diagnostic},
            validation_mode="strict",
        )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT validation_status, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("failed", diagnostic)
        assert conn.execute(
            "SELECT artifact_kind, support_status FROM raw_artifacts WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value, "decode_failed")

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.validation_failures == 1
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0
    assert lifecycle.blocking is False


@pytest.mark.asyncio
async def test_persist_batch_raw_state_updates_rolls_back_typed_evidence_with_raw_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-tier carrier failure rolls back the paired raw-state mutation."""
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="batch-atomic.jsonl",
            source_index=3,
            payload=b"unsupported-shape",
            acquired_at_ms=1,
        )

    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = SimpleNamespace(repository=repository)
    outcome = _RawIngestOutcome(
        raw_id=raw_id,
        payload_provider="codex",
        validation_status="passed",
        validation_error=None,
        parse_error="unsupported shape",
        error="unsupported shape",
        had_sessions=False,
        outcome_code="unsupported_shape",
        retryable=False,
        evidence_ref="shape",
        remediation="support it",
        diagnostic="unsupported",
    )

    async def fail_evidence(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("carrier write failed")

    monkeypatch.setattr(repository.source_backend, "save_raw_failure_evidence", fail_evidence)
    with pytest.raises(RuntimeError, match="carrier write failed"):
        await _persist_batch_raw_state_updates(
            service,
            repository.backend,
            outcomes={raw_id: outcome},
            succeeded_raw_ids=set(),
            skipped_raw_ids=set(),
            failed_raw_ids={raw_id: "unsupported shape"},
            validation_mode="strict",
        )
    await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parse_error, validation_status FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (None, None)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)


def test_a_skipped_excised_raw_keeps_its_refusal_reason() -> None:
    """The durable raw state names the excision, not a generic empty parse.

    Anti-vacuity (Codex P2, #5696): write the generic skip reason for every
    skipped raw and nothing durable records the typed refusal after restart.
    """
    from polylogue.pipeline.services.ingest_batch._core import _skipped_raw_state_update

    outcome = SimpleNamespace(payload_provider=None, diagnostic="content_excised: sidecar hash excised")
    update = _skipped_raw_state_update(outcome=outcome, parsed_at="2026-01-01T00:00:00Z", validation_mode="advisory")  # type: ignore[arg-type]

    assert str(update.validation_error).startswith("content_excised")


def test_drive_cohort_snapshot_copies_every_row_of_a_large_cohort() -> None:
    """A cohort beyond the old 1,000-row cap is snapshotted whole, not marked stale.

    Anti-vacuity: the ``LIMIT 1001`` snapshot plus the ``> 1000 rows`` stale
    predicate marked every preparation of such a cohort stale, so its raw was
    retried forever without converging.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE raw_session_memberships (raw_id TEXT, logical_source_key TEXT)")
        conn.executemany(
            "INSERT INTO raw_session_memberships VALUES (?, 'drive:key')",
            [(f"raw-{index:05d}",) for index in range(1_500)],
        )
        snapshot = ingest_batch_core._source_snapshot(
            conn, "raw_session_memberships", "logical_source_key = ?", ("drive:key",)
        )
    finally:
        conn.close()

    assert len(snapshot.rows) == 1_500


_CANONICAL_CODEX_PAYLOAD = (
    Path(__file__).parents[2] / "fixtures" / "origin-capability" / "codex-session.jsonl"
).read_bytes()


@asynccontextmanager
async def _canonical_parsing_service(
    root: Path,
    *,
    repository: SessionRepository | None = None,
) -> AsyncIterator[ParsingService]:
    """A real parsing service whose publication is the canonical retained owner."""
    owned_repository = repository is None
    repo = repository or SessionRepository(backend=SQLiteBackend(db_path=root / "index.db"), archive_root=root)
    try:
        async with prepared_live_convergence_owner(root) as owner:
            yield ParsingService(
                repository=repo,
                archive_root=root,
                config=Config(archive_root=root, render_root=root / "render", sources=[]),
                ingest_workers=1,
                retained_runner=owner.ingest_retained_raw_ids,
            )
    finally:
        if owned_repository:
            await repo.close()


def _seed_retained_raw(root: Path, *, origin: Origin, source_path: str, payload: bytes) -> str:
    BlobStore(root / "blob").write_from_bytes(payload)
    with sqlite3.connect(root / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=origin,
            source_path=source_path,
            source_index=0,
            payload=payload,
            acquired_at_ms=1,
        )
        conn.commit()
    return raw_id


def _publication_mode(monkeypatch: pytest.MonkeyPatch, mode: str = "off") -> None:
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: SimpleNamespace(schema_validation="advisory", sinex_mode=mode),
    )


@pytest.mark.asyncio
async def test_process_ingest_batch_requires_its_retained_owner(tmp_path: Path) -> None:
    """The public batch route publishes only through a supplied retained owner.

    Anti-vacuity: falling back to a local parse/write when no owner is
    supplied would publish outside the canonical owner and this refusal
    would disappear.
    """
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="no-owner.jsonl", payload=_CANONICAL_CODEX_PAYLOAD
    )
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(
        repository=repository,
        archive_root=tmp_path,
        config=Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[]),
        ingest_workers=1,
    )
    try:
        with pytest.raises(PermissionError, match="retained Raw owner"):
            await ingest_batch_core.process_ingest_batch(service, repository.backend, [raw_id], ParseResult(), None)
    finally:
        await repository.close()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [member.value for member in PublicationMode if member is not PublicationMode.OFF])
async def test_process_ingest_batch_refuses_sinex_publication_before_the_retained_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A non-OFF publication mode is refused before any local projection.

    Replaces the predecessor primary-mode laws: the retained route has no
    accepted-marker/outbox producer, so it must refuse before the owner
    publishes Index or FTS rows.
    """
    from polylogue.sinex.material_adapter import PublicationEncodingError

    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="sinex.jsonl", payload=_CANONICAL_CODEX_PAYLOAD
    )
    _publication_mode(monkeypatch, mode)
    calls: list[tuple[str, ...]] = []

    async def owner_must_not_run(raw_ids: Sequence[str]) -> tuple[PreparedRevisionReplayResult, ...]:
        calls.append(tuple(raw_ids))
        return ()

    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(
        repository=repository,
        archive_root=tmp_path,
        config=Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[]),
        ingest_workers=1,
        retained_runner=owner_must_not_run,
    )
    try:
        with pytest.raises(PublicationEncodingError, match="accepted-marker and outbox producer"):
            await ingest_batch_core.process_ingest_batch(service, repository.backend, [raw_id], ParseResult(), None)
    finally:
        await repository.close()
    assert calls == []
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone() == (0,)


@pytest.mark.asyncio
async def test_process_ingest_batch_publishes_and_invalidates_search_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed session from the canonical route is searchable, never served stale.

    Ported from the predecessor sync-route law: the search cache epoch covers
    ordinary ingest writes within one generation, so publication that changes
    a session must advance it. Anti-vacuity: drop the invalidation and the
    cached empty result is served after publication.
    """
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="search.jsonl", payload=_CANONICAL_CODEX_PAYLOAD
    )
    _publication_mode(monkeypatch)
    needle = "witness"
    before = search_messages(needle, archive_root=tmp_path, db_path=tmp_path / "index.db", limit=10)
    assert before.hits == ()
    version_before = get_cache_stats()["cache_version"]
    parse_result = ParseResult()
    async with _canonical_parsing_service(tmp_path) as service:
        await ingest_batch_core.process_ingest_batch(service, service.repository.backend, [raw_id], parse_result, None)
    assert len(parse_result.processed_ids) == 1
    assert get_cache_stats()["cache_version"] > version_before
    after = search_messages(needle, archive_root=tmp_path, db_path=tmp_path / "index.db", limit=10)
    assert {hit.session_id for hit in after.hits} == set(parse_result.processed_ids)


@pytest.mark.asyncio
async def test_process_ingest_batch_public_route_retires_deferred_cas_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public batch route supersedes deferred CAS evidence after publication."""
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    payload = (Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-conversation-v1.json").read_bytes()
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CHATGPT_EXPORT, source_path="public-batch.json", payload=payload
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="public-deferred-cas",
                origin=Origin.CHATGPT_EXPORT,
                source_path="public-batch.json",
                source_index=0,
                artifact_kind=RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
                classification_reason="deferred CAS",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                parse_as_session=True,
                schema_eligible=True,
                first_observed_at_ms=1,
                last_observed_at_ms=1,
            ),
        )
        conn.commit()
    _publication_mode(monkeypatch)
    parse_result = ParseResult()
    async with _canonical_parsing_service(tmp_path) as service:
        await ingest_batch_core.process_ingest_batch(service, service.repository.backend, [raw_id], parse_result, None)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, None)
        assert conn.execute(
            "SELECT artifact_kind, support_status FROM raw_artifacts WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
        )
    assert parse_result.processed_ids


@pytest.mark.asyncio
async def test_process_ingest_batch_off_mode_supports_repository_without_source_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFF publication keeps an index-only repository usable and writes no marker inputs."""
    await asyncio.to_thread(initialize_active_archive_root, tmp_path)
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="index-only.jsonl", payload=_CANONICAL_CODEX_PAYLOAD
    )
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"))
    assert repository.source_backend is None
    _publication_mode(monkeypatch)
    parse_result = ParseResult()
    try:
        async with _canonical_parsing_service(tmp_path, repository=repository) as service:
            await ingest_batch_core.process_ingest_batch(service, repository.backend, [raw_id], parse_result, None)
    finally:
        await repository.close()

    assert len(parse_result.processed_ids) == 1
    (session_id,) = parse_result.processed_ids
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone() == (2,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, None)
        assert conn.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)
    assert parse_result.parse_failures == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "diagnostic"),
    [
        (b"", "decode: Input is a zero-length, empty document"),
        (b"{", "decode: Input data was truncated"),
    ],
    ids=["zero-length-public-route", "decode-failure-public-route"],
)
async def test_process_ingest_batch_public_route_persists_corrupt_input_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    diagnostic: str,
) -> None:
    """The canonical route makes corrupt input terminal and status-readable."""
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="public-corrupt.jsonl", payload=payload
    )
    _publication_mode(monkeypatch)
    parse_result = ParseResult()
    async with _canonical_parsing_service(tmp_path) as service:
        await ingest_batch_core.process_ingest_batch(service, service.repository.backend, [raw_id], parse_result, None)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute(
            "SELECT artifact_kind, support_status FROM raw_artifacts WHERE raw_id = ?", (raw_id,)
        ).fetchone()
    assert artifact == (RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value, "decode_failed"), diagnostic
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0
    assert lifecycle.blocking is False
    status = raw_failure_info_for_root(tmp_path)
    assert status["terminal_rejections"] == 1
    assert status["unexplained_failures"] == 0

"""Focused tests for sync ingest-batch DB writes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, TypeAlias, cast
from unittest.mock import AsyncMock

import aiosqlite
import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_INGEST_FLAG
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.services import ingest_worker as ingest_worker_mod
from polylogue.pipeline.services.ingest_batch import (
    _build_batch_memory_observation,
    _drain_ready_session_entries,
    _failed_raw_state_update,
    _IngestBatchSummary,
    _IngestWorkerRequest,
    _iter_ingest_results_sync,
    _persist_batch_raw_state_updates,
    _process_ingest_batch_sync,
    _RawIngestOutcome,
    _select_ingest_worker_count,
    _successful_raw_state_update,
    _topo_sort_session_entries,
    _unattributed_batch_elapsed_s,
    refresh_session_insights_bulk,
)
from polylogue.pipeline.services.ingest_batch._observations import _build_parse_batch_observation
from polylogue.pipeline.services.ingest_worker import (
    IngestRecordResult,
    SessionWritePayload,
)
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.insights.session.refresh import SessionInsightRefreshChunkObservation
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.search.cache import get_cache_stats
from polylogue.storage.search.runtime import search_messages
from polylogue.storage.sqlite.archive_tiers.write import _attachment_id
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection
from polylogue.types import SessionId

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
        provider_session_id="315bcba7-700a-4c0e-b318-ab86d8636376",
        title="Session",
        messages=[],
    )

    normalized = ingest_worker_mod._normalized_session(
        parsed,
        fallback_timestamp=None,
    )

    assert str(make_session_id("claude-code-session", normalized.provider_session_id)) == (
        "claude-code-session:315bcba7-700a-4c0e-b318-ab86d8636376"
    )
    assert normalized.source_name == Provider.CLAUDE_CODE
    assert parsed.source_name == Provider.CLAUDE_CODE


def test_parse_batch_observation_reports_unsupported_write_mode() -> None:
    summary = _IngestBatchSummary()

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


def test_sync_index_connection_ensures_runtime_indexes(tmp_path: Path) -> None:
    conn = ingest_batch_core._open_sync_connection(tmp_path / "archive" / "index.db")
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_messages_active_leaf'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None


class _FakeConnectionBackend:
    def __init__(self, connection: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]) -> None:
        self._connection = connection

    def connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]:
        return self._connection()


class _FakeBulkBackend:
    def __init__(self, connection: Callable[[], AbstractAsyncContextManager[object]]) -> None:
        self._connection = connection

    def bulk_connection(self) -> AbstractAsyncContextManager[object]:
        return self._connection()


class _FakeRawStateRepository:
    def __init__(self, update_raw_state: AsyncMock) -> None:
        self._update_raw_state = update_raw_state

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


def _attachment_tuple(attachment_id: str, *, mime_type: str = "image/png") -> ParsedAttachment:
    return ParsedAttachment(
        provider_attachment_id=attachment_id,
        mime_type=mime_type,
        size_bytes=1024,
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

        _write_session(conn, child)
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

        _write_session(conn, parent)
        _write_session(conn, child)
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
        changed, counts = _write_session(conn, archive)
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
        changed, counts = _write_session(conn, v2)
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

        changed_initial, _initial_counts = _write_session(conn, initial)
        changed_tail, tail_counts = _write_session(conn, tail)
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

        changed_initial, _initial_counts = _write_session(conn, initial)
        changed_recapture, recapture_counts = _write_session(conn, recapture)
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
        changed, _ = _write_session(conn, archive)
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
        unchanged, counts = _write_session(conn, v2)
        assert unchanged is False
        assert counts["skipped_sessions"] == 1

        forced, counts = _write_session(conn, v2, force_write=True)
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
        changed, _ = _write_session(conn, newer)
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
        skipped, skipped_counts = _write_session(conn, older)
        assert skipped is False
        assert skipped_counts["messages"] == 0

        forced, forced_counts = _write_session(conn, older, force_write=True)
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
        changed, _ = _write_session(conn, first)
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
        unchanged, counts = _write_session(conn, recapture)
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

        changed, _ = _write_session(conn, first)
        unchanged, counts = _write_session(conn, recapture)
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

        changed_full, _counts_full = _write_session(conn, fuller)
        changed_stale, counts_stale = _write_session(conn, stale)
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

        changed_existing, _counts_existing = _write_session(conn, existing)
        changed_duplicate, counts_duplicate = _write_session(conn, duplicate)
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

        changed_native, _counts_native = _write_session(conn, native)
        changed_fallback, counts_fallback = _write_session(conn, dom_fallback)
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
            SELECT event_type, summary
            FROM session_events
            WHERE session_id = ?
            """,
            ("codex-session:dom-precedence",),
        ).fetchone()
        assert event["event_type"] == "capture_gap"
        assert "DOM browser-capture fallback" in event["summary"]
        assert "raw-dom-fallback" in event["summary"]


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

        changed_native, _counts_native = _write_session(conn, native)
        changed_fallback, counts_fallback = _write_session(conn, dom_fallback)
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

        changed_fallback, _counts_fallback = _write_session(conn, dom_fallback)
        changed_native, counts_native = _write_session(conn, native)
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
        ("native", 2, "export", 2, "Native browser", 2, "raw-initial", False),
        ("export", 2, "native", 2, "Native browser", 2, "raw-incoming", False),
        ("native", 2, "export", 1, "Native browser", 2, "raw-initial", False),
        ("export", 1, "native", 2, "Native browser", 2, "raw-incoming", False),
        ("native", 2, "export", 3, "Fuller export", 3, "raw-incoming", False),
        ("export", 3, "native", 2, "Fuller export", 3, "raw-initial", False),
        ("export", 2, "native", 2, "Native browser", 2, "raw-incoming", True),
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
        changed_initial, _counts_initial = _write_session(
            conn,
            payload(initial_kind, initial_count, "raw-initial", updated_at="2026-04-03T00:00:00Z"),
        )
        changed_incoming, counts_incoming = _write_session(
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


def test_write_session_skips_new_with_zero_messages(tmp_path: Path) -> None:
    """A new session with zero messages is skipped, not left as a manifest-only row."""
    with open_connection(tmp_path / "index.db") as conn:
        empty = _session_data(
            "codex-session:empty-manifest",
            content_hash="hash-empty",
            message_tuples=[],
        )
        changed, counts = _write_session(conn, empty)
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
        _write_session(conn, first)
        conn.commit()

        # Same session, different hash, zero messages — should be allowed
        update = _session_data(
            "codex-session:keep",
            content_hash="hash-2",
            message_tuples=[],
        )
        changed, counts = _write_session(conn, update)
        conn.commit()

        assert changed is True
        assert counts["skipped_sessions"] == 0


def test_iter_ingest_results_sync_runs_inline_for_single_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_artifacts = [
        RawSessionRecord(
            raw_id="raw-1",
            source_name="codex",
            source_path="/tmp/raw-1.jsonl",
            blob_size=12,
            acquired_at="2026-04-02T00:00:00Z",
        ),
        RawSessionRecord(
            raw_id="raw-2",
            source_name="codex",
            source_path="/tmp/raw-2.jsonl",
            blob_size=12,
            acquired_at="2026-04-02T00:00:00Z",
        ),
    ]
    seen: list[str] = []

    def fake_ingest_record(
        raw_record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str = "strict",
        measure_ingest_result_size: bool = False,
        *,
        blob_root_str: str | None = None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        seen.append(raw_record.raw_id)
        return IngestRecordResult(raw_id=raw_record.raw_id)

    def fail_process_pool_executor(*, max_workers: int) -> NoReturn:
        raise AssertionError(f"process pool should not be used for single-worker batches: {max_workers}")

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)
    monkeypatch.setattr(ingest_batch_core, "process_pool_executor", fail_process_pool_executor)

    results = list(
        _iter_ingest_results_sync(
            raw_artifacts,
            request=_IngestWorkerRequest(
                archive_root_str="/tmp/archive",
                blob_root_str="/tmp/blob-store",
                validation_mode="strict",
                measure_ingest_result_size=False,
            ),
            worker_count=1,
        )
    )

    assert seen == ["raw-1", "raw-2"]
    assert [result.raw_id for result in results] == ["raw-1", "raw-2"]


async def test_process_ingest_batch_uses_archive_root_blob_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "explicit-archive"
    ambient_blob_root = tmp_path / "ambient-xdg" / "blob"
    expected_blob_root = archive_root / "blob"
    raw_record = RawSessionRecord(
        raw_id="raw-1",
        source_name="chatgpt",
        source_path="/sources/session.json",
        blob_size=17,
        acquired_at="2026-04-02T00:00:00Z",
    )

    repository = SimpleNamespace(get_raw_sessions_batch=AsyncMock(return_value=[raw_record]))
    service = SimpleNamespace(
        repository=repository,
        archive_root=archive_root,
        ingest_workers=1,
        measure_ingest_result_size=False,
    )
    backend = SimpleNamespace(db_path=archive_root / "index.db")
    seen: dict[str, object] = {}

    def fake_process_sync(
        raw_artifacts: list[RawSessionRecord],
        *,
        db_path: Path,
        archive_root_str: str,
        blob_root_str: str,
        validation_mode: str,
        ingest_workers: int | None,
        measure_ingest_result_size: bool,
        force_write: bool,
        repair_message_fts: bool,
        ingest_result_chunk_size: int,
        suspend_fts_triggers: bool,
    ) -> _IngestBatchSummary:
        seen.update(
            {
                "raw_artifacts": raw_artifacts,
                "db_path": db_path,
                "archive_root_str": archive_root_str,
                "blob_root_str": blob_root_str,
                "validation_mode": validation_mode,
                "ingest_workers": ingest_workers,
                "measure_ingest_result_size": measure_ingest_result_size,
                "force_write": force_write,
                "repair_message_fts": repair_message_fts,
                "ingest_result_chunk_size": ingest_result_chunk_size,
                "suspend_fts_triggers": suspend_fts_triggers,
            }
        )
        return _IngestBatchSummary(raw_record_count=1)

    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: ambient_blob_root)
    monkeypatch.setattr(ingest_batch_core, "blob_store_root", lambda: ambient_blob_root, raising=False)
    monkeypatch.setattr(ingest_batch_core, "_process_ingest_batch_sync", fake_process_sync)
    monkeypatch.setattr(ingest_batch_core, "_persist_batch_raw_state_updates", AsyncMock(return_value=0.0))

    await ingest_batch_core.process_ingest_batch(
        cast(ParsingService, service),
        cast(SQLiteBackend, backend),
        ["raw-1"],
        ParseResult(),
        progress_callback=None,
    )

    assert seen["archive_root_str"] == str(archive_root)
    assert seen["blob_root_str"] == str(expected_blob_root)
    assert seen["blob_root_str"] != str(ambient_blob_root)


def test_iter_ingest_results_sync_bounds_in_flight_process_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_artifacts = [
        RawSessionRecord(
            raw_id=f"raw-{index}",
            source_name="codex",
            source_path=f"/tmp/raw-{index}.jsonl",
            blob_size=12,
            acquired_at="2026-04-02T00:00:00Z",
        )
        for index in range(10)
    ]
    pending_sizes: list[int] = []

    class FakeExecutor:
        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(
            self,
            fn: object,
            raw_record: RawSessionRecord,
            request: _IngestWorkerRequest,
        ) -> Future[IngestRecordResult]:
            del fn, request
            future: Future[IngestRecordResult] = Future()
            future.set_result(IngestRecordResult(raw_id=raw_record.raw_id))
            return future

    def fake_process_pool_executor(*, max_workers: int) -> FakeExecutor:
        assert max_workers == 2
        return FakeExecutor()

    def fake_wait(
        futures: object,
        *,
        timeout: float | None = None,
        return_when: object | None = None,
    ) -> tuple[set[Future[IngestRecordResult]], set[Future[IngestRecordResult]]]:
        del timeout, return_when
        pending = list(futures) if isinstance(futures, tuple) else []
        pending_sizes.append(len(pending))
        return set(pending[:1]), set(pending[1:])

    monkeypatch.setattr(ingest_batch_core, "process_pool_executor", fake_process_pool_executor)
    monkeypatch.setattr(ingest_batch_core, "wait", fake_wait)

    results = list(
        _iter_ingest_results_sync(
            raw_artifacts,
            request=_IngestWorkerRequest(
                archive_root_str="/tmp/archive",
                blob_root_str="/tmp/blob-store",
                validation_mode="strict",
                measure_ingest_result_size=False,
            ),
            worker_count=2,
        )
    )

    assert [result.raw_id for result in results] == [record.raw_id for record in raw_artifacts]
    assert max(pending_sizes) == 2


def test_iter_ingest_results_sync_emits_heartbeat_while_workers_are_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_artifacts = [
        RawSessionRecord(
            raw_id="raw-1",
            source_name="codex",
            source_path="/tmp/raw-1.jsonl",
            blob_size=12,
            acquired_at="2026-04-02T00:00:00Z",
        )
    ]
    future: Future[IngestRecordResult] = Future()
    wait_calls = 0
    heartbeat_count = 0

    class FakeExecutor:
        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(
            self,
            fn: object,
            raw_record: RawSessionRecord,
            request: _IngestWorkerRequest,
        ) -> Future[IngestRecordResult]:
            del fn, raw_record, request
            return future

    def fake_process_pool_executor(*, max_workers: int) -> FakeExecutor:
        assert max_workers == 2
        return FakeExecutor()

    def fake_wait(
        futures: object,
        *,
        timeout: float | None = None,
        return_when: object | None = None,
    ) -> tuple[set[Future[IngestRecordResult]], set[Future[IngestRecordResult]]]:
        nonlocal wait_calls
        del timeout, return_when
        pending = set(futures) if isinstance(futures, tuple) else set()
        wait_calls += 1
        if wait_calls == 1:
            return set(), pending
        future.set_result(IngestRecordResult(raw_id="raw-1"))
        return {future}, set()

    def heartbeat() -> None:
        nonlocal heartbeat_count
        heartbeat_count += 1

    monkeypatch.setattr(ingest_batch_core, "process_pool_executor", fake_process_pool_executor)
    monkeypatch.setattr(ingest_batch_core, "wait", fake_wait)

    results = list(
        _iter_ingest_results_sync(
            raw_artifacts,
            request=_IngestWorkerRequest(
                archive_root_str="/tmp/archive",
                blob_root_str="/tmp/blob-store",
                validation_mode="strict",
                measure_ingest_result_size=False,
            ),
            worker_count=2,
            heartbeat=heartbeat,
        )
    )

    assert [result.raw_id for result in results] == ["raw-1"]
    assert heartbeat_count == 1


def test_process_ingest_batch_sync_commits_fts_repair_and_invalidates_search_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    archive_root = tmp_path / "archive"
    blob_root = tmp_path / "blob"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_id = "raw-sync-side-effects"
    session_id = "codex-session:sync-side-effects"
    message_id = "msg-sync-side-effects"
    needle = "syncsideeffectneedle"

    raw_record = RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )
    session = _session_data(
        session_id,
        content_hash="hash-sync-side-effects",
        message_tuples=[
            _message_tuple(
                message_id,
                session_id,
                role="user",
                text=f"cached search should see {needle}",
                content_hash="hash-sync-message",
                sort_key=0.0,
            )
        ],
    )

    with open_connection(db_path) as conn:
        conn.commit()

    first_result = search_messages(needle, archive_root=archive_root, db_path=db_path, limit=10)
    cache_version_before = get_cache_stats()["cache_version"]
    assert first_result.hits == []

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        assert record.raw_id == raw_id
        return IngestRecordResult(raw_id=record.raw_id, sessions=[session])

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.suspend_fts_triggers_sync",
        lambda _conn: (_ for _ in ()).throw(AssertionError("live/default ingest must keep FTS triggers active")),
    )

    summary = _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(blob_root),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert summary.changed_session_ids == [session_id]
    assert get_cache_stats()["cache_version"] == cache_version_before + 1

    with open_connection(db_path) as conn:
        trigger_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE '%_fts_%'"
            ).fetchall()
        }
        assert {"messages_fts_ai", "messages_fts_ad", "messages_fts_au"}.issubset(trigger_names)
        # messages_fts is a contentless FTS5 table (content=''); column values are
        # not retrievable, so verify indexing via MATCH on the indexed block text.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
                (needle,),
            ).fetchone()[0]
            == 1
        )

    second_result = search_messages(needle, archive_root=archive_root, db_path=db_path, limit=10)
    assert [hit.session_id for hit in second_result.hits] == [session_id]


def test_process_ingest_batch_sync_replaces_stale_sessions_for_same_raw_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    archive_root = tmp_path / "archive"
    blob_root = tmp_path / "blob"
    source_path = tmp_path / "raw.jsonl.txt.json"
    source_path.write_text("{}", encoding="utf-8")
    raw_id = "raw-provider-redetected"
    stale_session_id = "aistudio-drive:provider-redetected"
    replacement_session_id = "codex-session:provider-redetected"

    raw_record = RawSessionRecord(
        raw_id=raw_id,
        source_name="gemini",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )
    stale = _session_data(
        stale_session_id,
        content_hash="stale-provider-session",
        raw_id=raw_id,
        message_tuples=[
            _message_tuple(
                "msg-stale-provider-redetected",
                stale_session_id,
                role="user",
                text="stale drive payload",
                content_hash="stale-message",
                sort_key=0.0,
            )
        ],
    )
    replacement = _session_data(
        replacement_session_id,
        content_hash="replacement-provider-session",
        raw_id=raw_id,
        message_tuples=[
            _message_tuple(
                "msg-provider-redetected",
                replacement_session_id,
                role="user",
                text="redetected stream payload",
                content_hash="replacement-message",
                sort_key=0.0,
            )
        ],
    )

    with open_connection(db_path) as conn:
        _write_session(conn, stale)
        conn.commit()

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        assert record.raw_id == raw_id
        return IngestRecordResult(raw_id=record.raw_id, sessions=[replacement])

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)

    _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(blob_root),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
        force_write=True,
    )

    with open_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id FROM sessions WHERE raw_id = ? ORDER BY session_id", (raw_id,)
        ).fetchall()
        assert [row[0] for row in rows] == [replacement_session_id]
        assert (
            conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (stale_session_id,)).fetchone()[0] == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (stale_session_id,)).fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (replacement_session_id,)).fetchone()[0]
            == 1
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_select_ingest_worker_count_uses_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch._core.os.cpu_count", lambda: 16)
    raw_artifacts = [SimpleNamespace(blob_size=16 * 1024 * 1024) for _ in range(6)]
    worker_count = _select_ingest_worker_count(raw_artifacts, None)
    # min(max(6,1), 16, 8) = 6 — uses all available artifacts
    assert worker_count == 6


def test_select_ingest_worker_count_avoids_process_pool_for_tiny_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch._core.os.cpu_count", lambda: 16)
    raw_artifacts = [SimpleNamespace(blob_size=512 * 1024) for _ in range(10)]
    worker_count = _select_ingest_worker_count(raw_artifacts, None)
    assert worker_count == 1


def test_select_ingest_worker_count_caps_small_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch._core.os.cpu_count", lambda: 16)
    raw_artifacts = [SimpleNamespace(blob_size=4 * 1024 * 1024) for _ in range(10)]
    worker_count = _select_ingest_worker_count(raw_artifacts, None)
    assert worker_count == 4


def test_select_ingest_worker_count_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch._core.os.cpu_count", lambda: 16)
    raw_artifacts = [SimpleNamespace(blob_size=16 * 1024 * 1024) for _ in range(60)]
    worker_count = _select_ingest_worker_count(raw_artifacts, ingest_workers=4)
    # min(max(60,1), 16, 4) = 4 — respects explicit limit
    assert worker_count == 4


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


@pytest.mark.asyncio
async def test_refresh_session_insights_bulk_dedupes_related_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = _FakeRefreshConnection()

    @asynccontextmanager
    async def _connection() -> AsyncIterator[aiosqlite.Connection]:
        yield fake_conn

    fake_backend = _FakeConnectionBackend(_connection)

    async def _fake_apply(conn: object, session_ids: list[str], *, transaction_depth: int) -> object:
        del conn, transaction_depth
        assert session_ids == ["conv-1", "conv-2", "conv-3"]
        return SimpleNamespace(
            counts={
                "profiles": 3,
                "work_events": 0,
                "phases": 0,
                "threads": 0,
                "tag_rollups": 0,
                "day_summaries": 0,
            },
            affected_groups={
                ("chatgpt", "2026-04-02"),
                ("chatgpt", "2026-04-03"),
            },
            thread_root_ids={"root-a", "root-b"},
            chunk_observations=[
                SessionInsightRefreshChunkObservation(
                    session_count=3,
                    estimated_message_count=3,
                    max_estimated_session_messages=1,
                    hydrated_count=3,
                    profiles_written=3,
                    work_events_written=0,
                    phases_written=0,
                    load_ms=12.5,
                    hydrate_ms=3.1,
                    build_ms=9.9,
                    write_ms=7.7,
                    total_ms=33.2,
                ),
            ],
        )

    refresh_thread_roots = AsyncMock(return_value=2)
    refresh_aggregates = AsyncMock()

    monkeypatch.setattr(
        "polylogue.storage.insights.session.refresh._apply_session_insight_session_updates_async",
        _fake_apply,
    )
    monkeypatch.setattr(
        "polylogue.storage.insights.session.refresh._refresh_thread_roots_async",
        refresh_thread_roots,
    )
    monkeypatch.setattr(
        "polylogue.storage.insights.session.refresh.refresh_async_provider_day_aggregates",
        refresh_aggregates,
    )

    observation = await refresh_session_insights_bulk(
        fake_backend,
        ["conv-1", "conv-2", "conv-3"],
    )

    refresh_thread_roots.assert_awaited_once()
    thread_args = refresh_thread_roots.await_args
    assert thread_args is not None
    assert thread_args.args[0] is fake_conn
    assert thread_args.args[1] == ["root-a", "root-b"]
    assert thread_args.kwargs["transaction_depth"] == 1
    refresh_aggregates.assert_awaited_once()
    aggregate_args = refresh_aggregates.await_args
    assert aggregate_args is not None
    assert aggregate_args.args[0] is fake_conn
    assert aggregate_args.args[1] == {
        ("chatgpt", "2026-04-02"),
        ("chatgpt", "2026-04-03"),
    }
    assert aggregate_args.kwargs["transaction_depth"] == 1
    fake_conn.commit_mock.assert_awaited_once()
    assert observation is not None
    assert observation["sessions"] == 3
    assert observation["unique_thread_roots"] == 2
    assert observation["unique_provider_days"] == 2
    assert _float_value(observation["elapsed_ms"]) >= 0.0
    assert _float_value(observation["update_ms"]) >= 0.0
    assert _float_value(observation["thread_refresh_ms"]) >= 0.0
    assert _float_value(observation["aggregate_refresh_ms"]) >= 0.0
    assert observation["update_chunk_count"] == 1
    assert observation["update_slow_chunk_count"] == 0
    assert observation["update_max_chunk_ms"] == 33.2
    assert observation["update_max_chunk_load_ms"] == 12.5
    assert observation["update_max_chunk_hydrate_ms"] == 3.1
    assert observation["update_max_chunk_build_ms"] == 9.9
    assert observation["update_max_chunk_write_ms"] == 7.7


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

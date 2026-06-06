"""Batch ingest orchestration: ProcessPool workers + sync sqlite3 writes.

Architecture:
- CPU-bound work (decode/validate/parse/transform) in ProcessPoolExecutor
- DB writes in main thread via sync sqlite3 (no aiosqlite async overhead)
- as_completed yields results as workers finish; writes drain completed worker
  results without retaining the whole parsed batch in memory

"""

from __future__ import annotations

import contextlib
import os
import pickle
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.archive.write_gateway import ArchiveWriteGateway, WriteOperation
from polylogue.core.common import (
    SQL_ACTION_EVENT_INSERT as _ACTION_EVENT_INSERT_SQL,
)
from polylogue.core.common import (
    SQL_ATTACHMENT_REF_INSERT as _ATTACHMENT_REF_INSERT_SQL,
)
from polylogue.core.common import (
    SQL_ATTACHMENT_UPSERT as _ATTACHMENT_UPSERT_SQL,
)
from polylogue.core.common import (
    SQL_CONTENT_BLOCK_UPSERT as _CONTENT_BLOCK_UPSERT_SQL,
)
from polylogue.core.common import (
    SQL_IDENTITY_LEDGER_UPSERT as _IDENTITY_LEDGER_UPSERT_SQL,
)
from polylogue.core.common import (
    SQL_MESSAGE_UPSERT as _MESSAGE_UPSERT_SQL,
)
from polylogue.core.common import (
    SQL_SESSION_UPSERT as _SESSION_UPSERT_SQL,
)
from polylogue.core.common import (
    SQL_STATS_UPSERT as _STATS_UPSERT_SQL,
)
from polylogue.core.memory import release_process_memory
from polylogue.core.metrics import (
    read_current_rss_mb,
    read_peak_rss_children_mb,
    read_peak_rss_self_mb,
)
from polylogue.logging import get_logger
from polylogue.paths import blob_store_root
from polylogue.pipeline.payload_types import MaterializeStageObservation, ParseBatchObservation
from polylogue.pipeline.services.ingest_batch._append_helpers import (
    append_content_hash,
    provider_event_tuple_without_raw_id,
    session_tuple_without_raw_id,
    tail_content_hash,
)
from polylogue.pipeline.services.ingest_batch._append_stats import existing_message_signatures, upsert_stats_for_append
from polylogue.pipeline.services.ingest_worker import (
    IngestRecordResult,
    MessageTuple,
    SessionData,
    SessionTuple,
    ingest_record,
)
from polylogue.pipeline.services.process_pool import process_pool_executor
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.session_replacement import (
    recount_and_prune_attachments_sync,
    replace_session_runtime_state_sync,
)
from polylogue.storage.sqlite.connection import _load_sqlite_vec
from polylogue.storage.sqlite.connection_profile import (
    DB_TIMEOUT,
    WRITE_CONNECTION_PRAGMA_STATEMENTS,
)
from polylogue.storage.sqlite.provider_event_writes import insert_provider_events_sync
from polylogue.types import ContentHash

if TYPE_CHECKING:
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.protocols import ProgressCallback
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

from polylogue.pipeline.services.ingest_batch._memory import (
    INGEST_RELEASE_BLOB_MB_THRESHOLD,
    INGEST_RELEASE_MESSAGE_THRESHOLD,
    discard_ingest_result_payload,
    discard_session_data_payload,
    ingest_result_needs_memory_release,
)
from polylogue.pipeline.services.ingest_batch._models import (
    _DEFAULT_INGEST_WORKER_LIMIT,
    _BulkConnectionBackendLike,
    _ConnectionBackendLike,
    _IngestBatchSummary,
    _IngestWorkerRequest,
    _ParsingServiceRawStateLike,
    _RawIngestOutcome,
    _SessionEntry,
)
from polylogue.pipeline.services.ingest_batch._observations import _build_parse_batch_observation
from polylogue.pipeline.services.ingest_batch._summary import (
    apply_ingest_batch_summary,
    progressed_raw_count,
    successful_raw_ids,
)

logger = get_logger(__name__)


@dataclass
class _WorkerProgress:
    in_flight_raw_ids: list[str] = field(default_factory=list)
    completed_raw_count: int = 0
    total_raw_count: int = 0


class _BlobSized(Protocol):
    blob_size: int


IngestHeartbeat = Callable[[], None]
_INGEST_RESULT_WAIT_HEARTBEAT_S = 15.0
_INGEST_RESULT_CHUNK_SIZE = 100


# Sync DB writer
# ---------------------------------------------------------------------------


def _open_sync_connection(db_path: Path) -> sqlite3.Connection:
    """Open a sync sqlite3 connection with the same pragmas as the async backend."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    for statement in WRITE_CONNECTION_PRAGMA_STATEMENTS:
        conn.execute(statement)
    _load_sqlite_vec(conn)
    return conn


def _check_content_unchanged(conn: sqlite3.Connection, cid: str, content_hash: str) -> bool:
    """Check if session content is unchanged (skip message writes)."""
    row = conn.execute(
        "SELECT content_hash FROM sessions WHERE session_id = ?",
        (cid,),
    ).fetchone()
    return row is not None and row[0] == content_hash


def _stored_message_count(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _topo_sort_message_tuples(tuples: list[MessageTuple]) -> list[MessageTuple]:
    """Sort message tuples so parents come before children (FK constraint).

    message_id is at index 0, parent_message_id is at index 8.
    """
    if not any(t[8] for t in tuples):
        return tuples
    ids_in_batch = {t[0] for t in tuples}
    no_parent: list[MessageTuple] = []
    has_parent: list[MessageTuple] = []
    for t in tuples:
        if t[8] and t[8] in ids_in_batch:
            has_parent.append(t)
        else:
            no_parent.append(t)
    if not has_parent:
        return tuples
    ordered = list(no_parent)
    inserted_ids = {t[0] for t in ordered}
    remaining = list(has_parent)
    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        next_remaining: list[MessageTuple] = []
        for t in remaining:
            if t[8] in inserted_ids:
                ordered.append(t)
                inserted_ids.add(t[0])
            else:
                next_remaining.append(t)
        remaining = next_remaining
    ordered.extend(remaining)
    return ordered


def _session_parent_id(cdata: SessionData) -> str | None:
    parent_id = cdata.session_tuple[11]
    return parent_id if isinstance(parent_id, str) else None


def _topo_sort_session_entries(
    entries: list[_SessionEntry],
) -> list[_SessionEntry]:
    """Sort session entries so parents in the same batch precede children."""
    ids_in_batch = {entry[1].session_id for entry in entries}
    no_parent: list[_SessionEntry] = []
    has_parent: list[_SessionEntry] = []

    for entry in entries:
        parent_id = _session_parent_id(entry[1])
        if parent_id and parent_id in ids_in_batch and parent_id != entry[1].session_id:
            has_parent.append(entry)
        else:
            no_parent.append(entry)

    if not has_parent:
        return entries

    ordered = list(no_parent)
    inserted_ids = {entry[1].session_id for entry in ordered}
    remaining = list(has_parent)
    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        next_remaining: list[_SessionEntry] = []
        for entry in remaining:
            parent_id = _session_parent_id(entry[1])
            if parent_id in inserted_ids:
                ordered.append(entry)
                inserted_ids.add(entry[1].session_id)
            else:
                next_remaining.append(entry)
        remaining = next_remaining
    ordered.extend(remaining)
    return ordered


def _session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row is not None


def _resolved_session_tuple(
    conn: sqlite3.Connection,
    cdata: SessionData,
) -> SessionTuple:
    """Resolve parent session links against the currently materialized archive.

    Parent session links are only durable when the parent already exists.
    This mirrors the prepare/enrichment path and avoids rejecting the whole
    session when a child arrives before its parent.
    """
    parent_id = _session_parent_id(cdata)
    if parent_id is None or parent_id == cdata.session_id:
        return cdata.session_tuple
    if _session_exists(conn, parent_id):
        return cdata.session_tuple
    return (
        cdata.session_tuple[0],
        cdata.session_tuple[1],
        cdata.session_tuple[2],
        cdata.session_tuple[3],
        cdata.session_tuple[4],
        cdata.session_tuple[5],
        cdata.session_tuple[6],
        cdata.session_tuple[7],
        cdata.session_tuple[8],
        cdata.session_tuple[9],
        cdata.session_tuple[10],
        None,
        cdata.session_tuple[12],
        cdata.session_tuple[13],
        cdata.session_tuple[14] if len(cdata.session_tuple) > 14 else "",
        cdata.session_tuple[15] if len(cdata.session_tuple) > 15 else None,
        cdata.session_tuple[16] if len(cdata.session_tuple) > 16 else None,
        cdata.session_tuple[17] if len(cdata.session_tuple) > 17 else None,
    )


def _session_tuple_with_hash(session: SessionTuple, content_hash: str) -> SessionTuple:
    return (
        session[0],
        session[1],
        session[2],
        session[3],
        session[4],
        session[5],
        session[6],
        ContentHash(content_hash),
        session[8],
        session[9],
        session[10],
        session[11],
        session[12],
        session[13],
        session[14] if len(session) > 14 else "",
        session[15] if len(session) > 15 else None,
        session[16] if len(session) > 16 else None,
        session[17] if len(session) > 17 else None,
    )


def _existing_provider_event_ids(conn: sqlite3.Connection, event_ids: Sequence[str]) -> set[str]:
    if not event_ids:
        return set()
    existing: set[str] = set()
    for offset in range(0, len(event_ids), _WRITE_SELECT_CHUNK_SIZE):
        chunk = event_ids[offset : offset + _WRITE_SELECT_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT event_id FROM provider_events WHERE event_id IN ({placeholders})",
            tuple(chunk),
        ).fetchall()
        existing.update(str(row["event_id"]) for row in rows)
    return existing


def _upsert_stats_from_messages(conn: sqlite3.Connection, session_id: str, source_name: str) -> None:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS message_count,
            COALESCE(SUM(word_count), 0) AS word_count,
            COALESCE(SUM(has_tool_use), 0) AS tool_use_count,
            COALESCE(SUM(has_thinking), 0) AS thinking_count,
            COALESCE(SUM(has_paste), 0) AS paste_count,
            COALESCE(SUM(CASE WHEN role = 'user'      THEN 1 ELSE 0 END), 0) AS user_msg_count,
            COALESCE(SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END), 0) AS assistant_msg_count,
            COALESCE(SUM(CASE WHEN role = 'system'    THEN 1 ELSE 0 END), 0) AS system_msg_count,
            COALESCE(SUM(CASE WHEN role = 'tool'      THEN 1 ELSE 0 END), 0) AS tool_msg_count,
            COALESCE(SUM(CASE WHEN role = 'user'      THEN word_count ELSE 0 END), 0) AS user_word_count,
            COALESCE(SUM(CASE WHEN role = 'assistant' THEN word_count ELSE 0 END), 0) AS assistant_word_count
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    conn.execute(
        _STATS_UPSERT_SQL,
        (
            session_id,
            source_name,
            int(row["message_count"] or 0),
            int(row["word_count"] or 0),
            int(row["tool_use_count"] or 0),
            int(row["thinking_count"] or 0),
            int(row["paste_count"] or 0),
            int(row["user_msg_count"] or 0),
            int(row["assistant_msg_count"] or 0),
            int(row["system_msg_count"] or 0),
            int(row["tool_msg_count"] or 0),
            int(row["user_word_count"] or 0),
            int(row["assistant_word_count"] or 0),
        ),
    )


_ACTION_EVENT_INSERT_OR_IGNORE_SQL = _ACTION_EVENT_INSERT_SQL.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1)
_WRITE_SELECT_CHUNK_SIZE = 900
_WRITE_EXECUTEMANY_CHUNK_SIZE = 1_000
_FTS_REPAIR_COUNT_KEY = "_fts_repair"


def _needs_session_fts_repair(conn: sqlite3.Connection, session_id: str) -> bool:
    from polylogue.storage.fts.session_repair import session_fts_needs_repair_sync

    return session_fts_needs_repair_sync(conn, session_id)


def _executemany_chunked(conn: sqlite3.Connection, sql: str, rows: Sequence[Sequence[object]]) -> None:
    """Run executemany in bounded chunks while preserving caller transaction scope."""
    if not rows:
        return
    for offset in range(0, len(rows), _WRITE_EXECUTEMANY_CHUNK_SIZE):
        conn.executemany(sql, rows[offset : offset + _WRITE_EXECUTEMANY_CHUNK_SIZE])


def _append_session(
    conn: sqlite3.Connection,
    cdata: SessionData,
    *,
    existing_hash: str | None,
) -> tuple[bool, dict[str, int]]:
    counts: dict[str, int] = {
        "sessions": 0,
        "messages": 0,
        "attachments": 0,
        "provider_events": 0,
        "skipped_sessions": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
        "skipped_provider_events": 0,
    }
    message_ids = [str(message[0]) for message in cdata.message_tuples]
    existing_messages = existing_message_signatures(conn, message_ids)
    changed_messages = [
        message
        for message in cdata.message_tuples
        if existing_messages.get(str(message[0]), ("", 0, 0, 0, 0))[0] != str(message[6])
    ]
    changed_message_ids = {str(message[0]) for message in changed_messages}
    event_ids = [str(event[0]) for event in cdata.provider_event_tuples]
    existing_provider_events = _existing_provider_event_ids(conn, event_ids)
    changed_provider_events = [
        event for event in cdata.provider_event_tuples if str(event[0]) not in existing_provider_events
    ]
    if not changed_messages and not cdata.attachment_tuples and not changed_provider_events:
        upsert_stats_for_append(
            conn,
            cdata.session_id,
            cdata.source_name,
            changed_messages,
            existing_messages,
            full_recount=_upsert_stats_from_messages,
        )
        counts["skipped_sessions"] = 1
        counts["skipped_messages"] = len(cdata.message_tuples)
        counts["skipped_attachments"] = len(cdata.attachment_tuples)
        counts["skipped_provider_events"] = len(cdata.provider_event_tuples)
        if _needs_session_fts_repair(conn, cdata.session_id):
            counts[_FTS_REPAIR_COUNT_KEY] = 1
        return False, counts

    counts["skipped_messages"] = len(cdata.message_tuples) - len(changed_messages)
    counts["skipped_provider_events"] = len(cdata.provider_event_tuples) - len(changed_provider_events)

    merged_hash = append_content_hash(existing_hash, tail_content_hash(changed_messages, cdata.content_hash))
    resolved_tuple = session_tuple_without_raw_id(_resolved_session_tuple(conn, cdata))
    conn.execute(_SESSION_UPSERT_SQL, _session_tuple_with_hash(resolved_tuple, merged_hash))
    # Record the identity mapping so re-ingest after reset preserves session_id.
    # Tuple indices: 1=source_name, 2=provider_session_id, 13=raw_id, 14=source_name
    conn.execute(
        _IDENTITY_LEDGER_UPSERT_SQL,
        (
            resolved_tuple[1],  # provider
            resolved_tuple[14],  # source (from provider_meta)
            "",  # source_path (not available at this level)
            resolved_tuple[2],  # provider_session_id
            resolved_tuple[13] or "",  # raw_hash (raw_id)
            resolved_tuple[0],  # current_session_id
        ),
    )

    if changed_messages:
        sorted_msgs = _topo_sort_message_tuples(changed_messages)
        _executemany_chunked(conn, _MESSAGE_UPSERT_SQL, sorted_msgs)
        counts["messages"] = len(changed_messages)

    changed_blocks = [block for block in cdata.block_tuples if str(block[1]) in changed_message_ids]
    if changed_blocks:
        changed_block_message_ids = sorted({str(block[1]) for block in changed_blocks})
        placeholders = ", ".join("?" for _ in changed_block_message_ids)
        conn.execute(
            f"DELETE FROM content_blocks WHERE message_id IN ({placeholders})", tuple(changed_block_message_ids)
        )
        _executemany_chunked(conn, _CONTENT_BLOCK_UPSERT_SQL, changed_blocks)

    changed_action_events = [event for event in cdata.action_event_tuples if str(event[2]) in changed_message_ids]
    if changed_action_events:
        changed_action_message_ids = sorted({str(event[2]) for event in changed_action_events})
        placeholders = ", ".join("?" for _ in changed_action_message_ids)
        conn.execute(
            f"DELETE FROM action_events WHERE message_id IN ({placeholders})", tuple(changed_action_message_ids)
        )
        _executemany_chunked(conn, _ACTION_EVENT_INSERT_OR_IGNORE_SQL, changed_action_events)

    if changed_provider_events:
        rawless_provider_events = [provider_event_tuple_without_raw_id(event) for event in changed_provider_events]
        insert_provider_events_sync(conn, rawless_provider_events, ignore_existing=True)
        counts["provider_events"] = len(changed_provider_events)

    affected_attachment_ids = {str(attachment_id) for attachment_id, *_rest in cdata.attachment_tuples}
    if cdata.attachment_tuples:
        _executemany_chunked(conn, _ATTACHMENT_UPSERT_SQL, cdata.attachment_tuples)
        _executemany_chunked(conn, _ATTACHMENT_REF_INSERT_SQL, cdata.attachment_ref_tuples)
        counts["attachments"] = len(cdata.attachment_tuples)
        recount_and_prune_attachments_sync(conn, affected_attachment_ids)

    upsert_stats_for_append(
        conn,
        cdata.session_id,
        cdata.source_name,
        changed_messages,
        existing_messages,
        full_recount=_upsert_stats_from_messages,
    )
    counts["sessions"] = 1
    return True, counts


def _write_session(
    conn: sqlite3.Connection, cdata: SessionData, *, force_write: bool = False
) -> tuple[bool, dict[str, int]]:
    """Write one session's data to DB via sync sqlite3.

    Returns (content_changed, counts).
    """
    counts: dict[str, int] = {
        "sessions": 0,
        "messages": 0,
        "attachments": 0,
        "provider_events": 0,
        "skipped_sessions": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
        "skipped_provider_events": 0,
    }

    existing_row = conn.execute(
        "SELECT content_hash, raw_id FROM sessions WHERE session_id = ?",
        (cdata.session_id,),
    ).fetchone()
    content_unchanged = existing_row is not None and str(existing_row["content_hash"]) == cdata.content_hash
    if cdata.append_only and existing_row is not None:
        return _append_session(conn, cdata, existing_hash=str(existing_row["content_hash"]))

    if not force_write and content_unchanged:
        counts["skipped_sessions"] = 1
        counts["skipped_messages"] = len(cdata.message_tuples)
        counts["skipped_attachments"] = len(cdata.attachment_tuples)
        counts["skipped_provider_events"] = len(cdata.provider_event_tuples)
        if _needs_session_fts_repair(conn, cdata.session_id):
            counts[_FTS_REPAIR_COUNT_KEY] = 1
        return False, counts

    existing_raw_id = str(existing_row["raw_id"] or "") if existing_row is not None else ""
    if (
        not force_write
        and existing_raw_id
        and cdata.raw_id
        and existing_raw_id != cdata.raw_id
        and len(cdata.message_tuples) < _stored_message_count(conn, cdata.session_id)
    ):
        counts["skipped_sessions"] = 1
        counts["skipped_messages"] = len(cdata.message_tuples)
        counts["skipped_attachments"] = len(cdata.attachment_tuples)
        counts["skipped_provider_events"] = len(cdata.provider_event_tuples)
        return False, counts

    # Guard against manifest-only session rows: a new session with
    # zero messages likely came from an interrupted bulk import that registered
    # the ID before message ingestion completed. Skip it rather than leaving a
    # row that has message_count=0, work_event_count=0, substantive_count=0.
    if existing_row is None and not cdata.message_tuples and not force_write:
        counts["skipped_sessions"] = 1
        return False, counts

    conn.execute(_SESSION_UPSERT_SQL, _resolved_session_tuple(conn, cdata))

    affected_attachment_ids: set[str] = set()
    if not content_unchanged:
        counts["sessions"] = 1
        if existing_row is not None:
            affected_attachment_ids = replace_session_runtime_state_sync(conn, cdata.session_id)
    else:
        counts["sessions"] = 0

    if cdata.message_tuples:
        sorted_msgs = _topo_sort_message_tuples(cdata.message_tuples)
        _executemany_chunked(conn, _MESSAGE_UPSERT_SQL, sorted_msgs)
        counts["messages"] = len(sorted_msgs)

    # Session stats
    if cdata.stats_tuple and not content_unchanged:
        conn.execute(_STATS_UPSERT_SQL, cdata.stats_tuple)

    if not content_unchanged:
        conn.execute("DELETE FROM content_blocks WHERE session_id = ?", (cdata.session_id,))
        if cdata.block_tuples:
            _executemany_chunked(conn, _CONTENT_BLOCK_UPSERT_SQL, cdata.block_tuples)

    if not content_unchanged:
        conn.execute("DELETE FROM action_events WHERE session_id = ?", (cdata.session_id,))
        if cdata.action_event_tuples:
            _executemany_chunked(conn, _ACTION_EVENT_INSERT_SQL, cdata.action_event_tuples)

    if not content_unchanged:
        conn.execute("DELETE FROM provider_events WHERE session_id = ?", (cdata.session_id,))
        if cdata.provider_event_tuples:
            insert_provider_events_sync(conn, cdata.provider_event_tuples)
            counts["provider_events"] = len(cdata.provider_event_tuples)

    # Attachments
    if not content_unchanged:
        new_attachment_ids = {attachment_id for attachment_id, *_rest in cdata.attachment_tuples}
        affected_attachment_ids |= new_attachment_ids
        if cdata.attachment_tuples:
            _executemany_chunked(conn, _ATTACHMENT_UPSERT_SQL, cdata.attachment_tuples)
            _executemany_chunked(conn, _ATTACHMENT_REF_INSERT_SQL, cdata.attachment_ref_tuples)
            counts["attachments"] = len(cdata.attachment_tuples)
        recount_and_prune_attachments_sync(conn, affected_attachment_ids)
    else:
        counts["attachments"] = 0

    return True, counts


def _record_outcome(summary: _IngestBatchSummary, ir: IngestRecordResult) -> None:
    summary.outcomes[ir.raw_id] = _RawIngestOutcome(
        raw_id=ir.raw_id,
        payload_provider=ir.payload_provider,
        validation_status=ir.validation_status,
        validation_error=ir.validation_error,
        parse_error=ir.parse_error,
        error=ir.error,
        had_sessions=bool(ir.sessions),
    )
    if ir.serialized_size_bytes is not None:
        summary.total_result_bytes += ir.serialized_size_bytes
        if ir.serialized_size_bytes > summary.max_result_bytes:
            summary.max_result_bytes = ir.serialized_size_bytes
            summary.max_result_raw_id = ir.raw_id


def _observe_current_rss(summary: _IngestBatchSummary) -> None:
    current_rss_mb = read_current_rss_mb()
    if current_rss_mb is None:
        return
    if summary.max_current_rss_mb is None or current_rss_mb > summary.max_current_rss_mb:
        summary.max_current_rss_mb = current_rss_mb


def _record_write_result(
    summary: _IngestBatchSummary,
    cdata: SessionData,
    *,
    content_changed: bool,
    counts: dict[str, int],
) -> None:
    summary.total_convos += 1
    summary.total_msgs += len(cdata.message_tuples)

    ingest_changed = (counts["sessions"] + counts["messages"] + counts["attachments"] + counts["provider_events"]) > 0

    if ingest_changed or content_changed:
        summary.processed_ids.add(cdata.session_id)
    if content_changed:
        summary.changed_counts["sessions"] += 1
        summary.changed_session_ids.append(cdata.session_id)
        summary.fts_repair_session_ids.append(cdata.session_id)
    elif counts.get(_FTS_REPAIR_COUNT_KEY, 0):
        summary.fts_repair_session_ids.append(cdata.session_id)
    if counts["messages"]:
        summary.changed_counts["messages"] += counts["messages"]
    if counts["attachments"]:
        summary.changed_counts["attachments"] += counts["attachments"]
    if counts["provider_events"]:
        summary.changed_counts["provider_events"] += counts["provider_events"]
    for key, value in counts.items():
        if key in summary.counts:
            summary.counts[key] += value


def _write_session_entry(
    conn: sqlite3.Connection,
    raw_id: str,
    cdata: SessionData,
    *,
    summary: _IngestBatchSummary,
    force_write: bool = False,
) -> bool:
    try:
        t_write = time.perf_counter()
        content_changed, counts = _write_session(conn, cdata, force_write=force_write)
        write_elapsed = time.perf_counter() - t_write
        summary.write_elapsed_s += write_elapsed
        if write_elapsed > summary.max_write_elapsed_s:
            summary.max_write_elapsed_s = write_elapsed
        _record_write_result(
            summary,
            cdata,
            content_changed=content_changed,
            counts=counts,
        )
        if write_elapsed >= 1.0:
            logger.info(
                "slow_write",
                cid=cdata.session_id[:20],
                elapsed_s=round(write_elapsed, 2),
                msgs=len(cdata.message_tuples),
                changed_messages=counts["messages"],
                skipped_messages=counts["skipped_messages"],
                blocks=len(cdata.block_tuples),
                actions=len(cdata.action_event_tuples),
                provider_events=len(cdata.provider_event_tuples),
                changed_provider_events=counts["provider_events"],
                attachments=len(cdata.attachment_tuples),
            )
        return True
    except Exception as exc:
        logger.error("Error writing session: %s", exc)
        summary.parse_failures += 1
        summary.failed_raw_ids[raw_id] = str(exc)[:500]
        return False


def _drain_ready_session_entries(
    conn: sqlite3.Connection,
    ready_entries: list[_SessionEntry],
    *,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    force_write: bool = False,
) -> None:
    for raw_id, cdata in _topo_sort_session_entries(ready_entries):
        wrote = _write_session_entry(conn, raw_id, cdata, summary=summary, force_write=force_write)
        discard_session_data_payload(cdata)
        if not wrote:
            continue
        materialized_ids.add(cdata.session_id)


def _run_ingest_record(
    raw_record: RawSessionRecord,
    request: _IngestWorkerRequest,
) -> IngestRecordResult:
    return ingest_record(
        raw_record,
        request.archive_root_str,
        request.validation_mode,
        request.measure_ingest_result_size,
        blob_root_str=request.blob_root_str,
    )


def _iter_ingest_results_sync(
    raw_artifacts: list[RawSessionRecord],
    *,
    request: _IngestWorkerRequest,
    worker_count: int,
    heartbeat: IngestHeartbeat | None = None,
    progress: _WorkerProgress | None = None,
    chunk_size: int = 0,
    force_process_pool: bool = False,
) -> Iterable[IngestRecordResult]:
    """Yield ingest results, optionally chunked to bound parsed-result memory."""
    total = len(raw_artifacts)
    if progress is not None:
        progress.total_raw_count = total
        progress.completed_raw_count = 0
        progress.in_flight_raw_ids.clear()

    if chunk_size <= 0 or total <= chunk_size:
        yield from _iter_ingest_results_chunk(
            raw_artifacts,
            request=request,
            worker_count=worker_count,
            heartbeat=heartbeat,
            progress=progress,
            force_process_pool=force_process_pool,
        )
        return

    for chunk_start in range(0, total, chunk_size):
        chunk = raw_artifacts[chunk_start : chunk_start + chunk_size]
        yield from _iter_ingest_results_chunk(
            chunk,
            request=request,
            worker_count=worker_count,
            heartbeat=heartbeat,
            progress=progress,
            force_process_pool=force_process_pool,
        )


def _iter_ingest_results_chunk(
    raw_artifacts: list[RawSessionRecord],
    *,
    request: _IngestWorkerRequest,
    worker_count: int,
    heartbeat: IngestHeartbeat | None = None,
    progress: _WorkerProgress | None = None,
    force_process_pool: bool = False,
) -> Iterable[IngestRecordResult]:
    """Process one chunk of raw_artifacts through the process pool."""
    if worker_count <= 1 and not force_process_pool:
        for raw_record in raw_artifacts:
            if progress is not None:
                progress.in_flight_raw_ids[:] = [raw_record.raw_id]
            if heartbeat is not None:
                heartbeat()
            yield _run_ingest_record(raw_record, request)
            if progress is not None:
                progress.completed_raw_count += 1
                progress.in_flight_raw_ids.clear()
        return
    try:
        with process_pool_executor(max_workers=max(1, worker_count)) as executor:
            raw_iter = iter(raw_artifacts)
            futures: dict[Future[IngestRecordResult], str] = {}
            max_in_flight = max(1, worker_count)

            def submit_next() -> bool:
                try:
                    raw_record = next(raw_iter)
                except StopIteration:
                    return False
                future = executor.submit(_run_ingest_record, raw_record, request)
                futures[future] = raw_record.raw_id
                if progress is not None:
                    progress.in_flight_raw_ids[:] = list(futures.values())
                return True

            for _ in range(max_in_flight):
                if not submit_next():
                    break
            while futures:
                done, _pending = wait(
                    tuple(futures),
                    timeout=_INGEST_RESULT_WAIT_HEARTBEAT_S,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    if heartbeat is not None:
                        heartbeat()
                    continue
                for future in done:
                    raw_id = futures.pop(future)
                    if progress is not None:
                        progress.in_flight_raw_ids[:] = list(futures.values())
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = IngestRecordResult(raw_id=raw_id, error=f"worker: {exc}")
                    submit_next()
                    if progress is not None:
                        progress.in_flight_raw_ids[:] = list(futures.values())
                        progress.completed_raw_count += 1
                    yield result
    except (TypeError, pickle.PicklingError):
        for raw_record in raw_artifacts:
            if progress is not None:
                progress.in_flight_raw_ids[:] = [raw_record.raw_id]
            if heartbeat is not None:
                heartbeat()
            yield _run_ingest_record(raw_record, request)
            if progress is not None:
                progress.completed_raw_count += 1
                progress.in_flight_raw_ids.clear()


def _resolved_ingest_worker_limit(value: int | None) -> int:
    return value if value is not None else _DEFAULT_INGEST_WORKER_LIMIT


def _select_ingest_worker_count(raw_artifacts: Sequence[_BlobSized], ingest_workers: int | None) -> int:
    total_blob_size = sum(record.blob_size for record in raw_artifacts)
    if total_blob_size <= 8 * 1024 * 1024:
        return 1
    if total_blob_size <= 64 * 1024 * 1024:
        return min(
            max(len(raw_artifacts), 1),
            os.cpu_count() or 4,
            _resolved_ingest_worker_limit(ingest_workers),
            4,
        )
    return min(
        max(len(raw_artifacts), 1),
        os.cpu_count() or 4,
        _resolved_ingest_worker_limit(ingest_workers),
    )


def _new_ingest_batch_summary(
    raw_artifacts: list[RawSessionRecord],
    *,
    ingest_workers: int | None,
) -> _IngestBatchSummary:
    summary = _IngestBatchSummary()
    summary.raw_record_count = len(raw_artifacts)
    summary.worker_count = _select_ingest_worker_count(raw_artifacts, ingest_workers)
    summary.total_blob_mb = sum(record.blob_size for record in raw_artifacts) / (1024 * 1024)
    return summary


def _make_ingest_worker_request(
    *,
    archive_root_str: str,
    blob_root_str: str,
    validation_mode: str,
    measure_ingest_result_size: bool,
) -> _IngestWorkerRequest:
    return _IngestWorkerRequest(
        archive_root_str=archive_root_str,
        blob_root_str=blob_root_str,
        validation_mode=validation_mode,
        measure_ingest_result_size=measure_ingest_result_size,
    )


def _record_failed_ingest_result(summary: _IngestBatchSummary, ir: IngestRecordResult) -> None:
    logger.error("Failed to ingest raw record", raw_id=ir.raw_id, error=ir.error)
    summary.parse_failures += 1
    summary.failed_raw_ids[ir.raw_id] = (ir.error or "unknown worker failure")[:500]


def _drain_ingest_result(
    conn: sqlite3.Connection,
    ir: IngestRecordResult,
    *,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    force_write: bool = False,
) -> None:
    _record_outcome(summary, ir)
    _observe_current_rss(summary)

    if ir.error:
        _record_failed_ingest_result(summary, ir)
        return

    if not ir.sessions:
        summary.skipped_raw_ids.add(ir.raw_id)
        return

    drain_started = time.perf_counter()
    _drain_ready_session_entries(
        conn,
        [(ir.raw_id, cdata) for cdata in ir.sessions],
        summary=summary,
        materialized_ids=materialized_ids,
        force_write=force_write,
    )
    summary.drain_elapsed_s += time.perf_counter() - drain_started
    _observe_current_rss(summary)


def _consume_ingest_results(
    conn: sqlite3.Connection,
    raw_artifacts: list[RawSessionRecord],
    *,
    worker_request: _IngestWorkerRequest,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    force_write: bool = False,
    heartbeat: IngestHeartbeat | None = None,
    progress: _WorkerProgress | None = None,
    ingest_result_chunk_size: int = 0,
    suspend_fts_triggers: bool = False,
    force_process_pool: bool = False,
) -> bool:
    result_iterator = iter(
        _iter_ingest_results_sync(
            raw_artifacts,
            request=worker_request,
            worker_count=summary.worker_count,
            heartbeat=heartbeat,
            progress=progress,
            chunk_size=ingest_result_chunk_size,
            force_process_pool=force_process_pool,
        )
    )
    transaction_started = False
    while True:
        wait_started = time.perf_counter()
        try:
            ir = next(result_iterator)
        except StopIteration:
            summary.teardown_elapsed_s = time.perf_counter() - wait_started
            break
        summary.result_wait_s += time.perf_counter() - wait_started
        release_after_drain = ingest_result_needs_memory_release(ir)
        try:
            if not transaction_started:
                conn.execute("BEGIN IMMEDIATE")
                if suspend_fts_triggers:
                    from polylogue.storage.fts.fts_lifecycle import suspend_fts_triggers_sync

                    suspend_fts_triggers_sync(conn, mark_stale=False)
                transaction_started = True
            _drain_ingest_result(
                conn,
                ir,
                summary=summary,
                materialized_ids=materialized_ids,
                force_write=force_write,
            )
        finally:
            discard_ingest_result_payload(ir)
            if release_after_drain:
                release_process_memory()
                _observe_current_rss(summary)
    return transaction_started


def _flush_ingest_results(
    conn: sqlite3.Connection,
    *,
    summary: _IngestBatchSummary,
) -> None:
    """Record final drain timing before committing.

    The commit + FTS trigger restore + FTS repair are deliberately
    deferred to ``_commit_sync_ingest_side_effects`` so they land as a
    single atomic write through the archive write gateway (#1242). This
    closes the gap where row commit and post-commit FTS repair were two
    separate transactions: if the repair failed, the data was already
    committed and the FTS index drifted silently.
    """
    flush_started = time.perf_counter()
    summary.flush_elapsed_s = time.perf_counter() - flush_started
    _observe_current_rss(summary)


def _commit_sync_ingest_side_effects(
    conn: sqlite3.Connection,
    *,
    db_path: Path,
    changed_session_ids: Sequence[str],
    repair_message_fts: bool = True,
    repair_action_fts: bool = True,
) -> None:
    """Run post-ingest side effects through the canonical write-effects path."""
    ArchiveWriteGateway(db_path).commit_write_sync(
        WriteOperation.INGEST,
        {
            "_connection": conn,
            "changed_session_ids": tuple(changed_session_ids),
            "repair_message_fts": repair_message_fts,
            "repair_action_fts": repair_action_fts,
        },
    )


def _process_ingest_batch_sync(
    raw_artifacts: list[RawSessionRecord],
    *,
    db_path: Path,
    archive_root_str: str,
    blob_root_str: str,
    validation_mode: str,
    ingest_workers: int | None,
    measure_ingest_result_size: bool,
    force_write: bool = False,
    repair_message_fts: bool = True,
    repair_action_fts: bool = True,
    heartbeat: IngestHeartbeat | None = None,
    progress: _WorkerProgress | None = None,
    ingest_result_chunk_size: int = 0,
    suspend_fts_triggers: bool = False,
    force_process_pool: bool = False,
) -> _IngestBatchSummary:
    if progress is None:
        progress = _WorkerProgress()
    summary = _new_ingest_batch_summary(raw_artifacts, ingest_workers=ingest_workers)
    worker_request = _make_ingest_worker_request(
        archive_root_str=archive_root_str,
        blob_root_str=blob_root_str,
        validation_mode=validation_mode,
        measure_ingest_result_size=measure_ingest_result_size,
    )
    t_start = time.perf_counter()
    setup_started = time.perf_counter()
    conn = _open_sync_connection(db_path)
    summary.setup_elapsed_s = time.perf_counter() - setup_started
    heavy_batch = summary.total_blob_mb >= INGEST_RELEASE_BLOB_MB_THRESHOLD
    materialized_ids: set[str] = set()
    _observe_current_rss(summary)
    transaction_started = False
    try:
        transaction_started = _consume_ingest_results(
            conn,
            raw_artifacts,
            worker_request=worker_request,
            summary=summary,
            materialized_ids=materialized_ids,
            force_write=force_write,
            heartbeat=heartbeat,
            progress=progress,
            ingest_result_chunk_size=ingest_result_chunk_size,
            suspend_fts_triggers=suspend_fts_triggers,
            force_process_pool=force_process_pool,
        )
        _flush_ingest_results(
            conn,
            summary=summary,
        )
        if transaction_started:
            fts_repair_ids = set(summary.fts_repair_session_ids)
            # Side effects run before releasing the connection so data and post-
            # write effects share one transaction. The previous arrangement
            # ran side effects in a `finally` block — they fired even after
            # a rollback (silently restoring triggers on top of nothing) and
            # ran AFTER the row commit, so a failure between commit and FTS
            # repair would leave the index drifted. See #1242.
            commit_started = time.perf_counter()
            _commit_sync_ingest_side_effects(
                conn,
                db_path=db_path,
                changed_session_ids=tuple(fts_repair_ids),
                repair_message_fts=repair_message_fts or suspend_fts_triggers,
                repair_action_fts=repair_action_fts or suspend_fts_triggers,
            )
            summary.commit_elapsed_s = time.perf_counter() - commit_started
            from polylogue.storage.sqlite.wal_checkpoint import maybe_checkpoint_wal

            wal_observation = maybe_checkpoint_wal(db_path, reason="ingest_batch_commit")
            summary.wal_checkpoint_mode = wal_observation.mode
            summary.wal_bytes_before_checkpoint = wal_observation.wal_bytes_before
            summary.wal_bytes_after_checkpoint = wal_observation.wal_bytes_after
            summary.wal_checkpointed_pages = wal_observation.checkpointed_pages
            summary.wal_busy_pages = wal_observation.busy_pages
            summary.wal_checkpoint_elapsed_s = wal_observation.elapsed_s
            summary.wal_checkpoint_error = wal_observation.error
            if wal_observation.blocking_processes:
                logger.warning(
                    "wal_checkpoint_blocked",
                    reason=wal_observation.reason,
                    mode=wal_observation.mode,
                    wal_bytes_before=wal_observation.wal_bytes_before,
                    wal_bytes_after=wal_observation.wal_bytes_after,
                    busy_pages=wal_observation.busy_pages,
                    blocking_processes=wal_observation.blocking_processes[:5],
                )
    except Exception:
        # Roll back the row writes.  If a caller explicitly opted into
        # dropped-trigger bulk mode, restore triggers before propagating
        # so ordinary exceptions do not leave the database in a drift
        # state.  Daemon live ingest leaves triggers active and therefore
        # has no dropped-trigger window to recover from here.
        with contextlib.suppress(Exception):
            conn.rollback()
        if suspend_fts_triggers:
            from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync

            with contextlib.suppress(Exception):
                restore_fts_triggers_sync(conn)
                conn.commit()
        raise
    finally:
        conn.close()
    summary.worker_progress_in_flight = len(progress.in_flight_raw_ids)
    summary.worker_progress_completed = progress.completed_raw_count
    summary.worker_progress_total = progress.total_raw_count
    summary.elapsed_s = time.perf_counter() - t_start
    if heavy_batch or summary.total_msgs >= INGEST_RELEASE_MESSAGE_THRESHOLD:
        release_process_memory()
        _observe_current_rss(summary)
    return summary


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


async def process_ingest_batch(
    service: ParsingService,
    backend: SQLiteBackend,
    batch_ids: list[str],
    result: ParseResult,
    progress_callback: ProgressCallback | None,
    *,
    force_write: bool = False,
    ingest_result_chunk_size: int = 0,
) -> ParseBatchObservation | None:
    """Process a batch of raw records through the unified ingest pipeline.

    1. Submit all records to ProcessPool (decode + validate + parse + transform)
    2. Consume results via as_completed — write to DB as each worker finishes
    3. Defer session insight refresh to caller (done once after all batches)

    When *ingest_result_chunk_size* > 0 and *batch_ids* exceeds it, raw
    records are split into sub-batches to bound the memory held by parsed
    results in the process pool and drain loop.
    """
    import asyncio

    raw_artifacts = await service.repository.get_raw_sessions_batch(batch_ids)
    if not raw_artifacts:
        return None

    archive_root_str = str(service.archive_root)
    blob_root_str = str(blob_store_root())
    batch_started = time.perf_counter()
    rss_start_mb = read_current_rss_mb()
    peak_rss_self_start_mb = read_peak_rss_self_mb()

    # Get validation mode from environment
    validation_mode = os.environ.get("POLYLOGUE_SCHEMA_VALIDATION", "advisory")

    batch_summary = await asyncio.to_thread(
        _process_ingest_batch_sync,
        raw_artifacts,
        db_path=backend.db_path,
        archive_root_str=archive_root_str,
        blob_root_str=blob_root_str,
        validation_mode=validation_mode,
        ingest_workers=service.ingest_workers,
        measure_ingest_result_size=service.measure_ingest_result_size,
        force_write=force_write,
        ingest_result_chunk_size=ingest_result_chunk_size,
    )
    heavy_batch = (
        batch_summary.total_blob_mb >= INGEST_RELEASE_BLOB_MB_THRESHOLD
        or batch_summary.total_msgs >= INGEST_RELEASE_MESSAGE_THRESHOLD
    )
    raw_artifacts.clear()

    apply_ingest_batch_summary(result, batch_summary)
    progressed = progressed_raw_count(batch_summary)
    if progress_callback and progressed:
        progress_callback(progressed)

    if batch_summary.elapsed_s > 0.0:
        logger.info(
            "ingest_batch",
            elapsed_s=round(batch_summary.elapsed_s, 2),
            records=batch_summary.raw_record_count,
            blob_mb=round(batch_summary.total_blob_mb, 1),
            sessions=batch_summary.total_convos,
            messages=batch_summary.total_msgs,
            workers=batch_summary.worker_count,
            changed=len(batch_summary.changed_session_ids),
            result_mb=round(batch_summary.total_result_bytes / (1024 * 1024), 1),
            max_result_mb=round(batch_summary.max_result_bytes / (1024 * 1024), 1),
            max_result_raw_id=batch_summary.max_result_raw_id,
            max_current_rss_mb=batch_summary.max_current_rss_mb,
            write_s=round(batch_summary.write_elapsed_s, 2),
            max_write_s=round(batch_summary.max_write_elapsed_s, 2),
            commit_s=round(batch_summary.commit_elapsed_s, 2),
            wal_mode=batch_summary.wal_checkpoint_mode,
            wal_before=batch_summary.wal_bytes_before_checkpoint,
            wal_after=batch_summary.wal_bytes_after_checkpoint,
            wal_busy=batch_summary.wal_busy_pages,
            drain_s=round(batch_summary.drain_elapsed_s, 2),
            flush_s=round(batch_summary.flush_elapsed_s, 2),
            wait_s=round(batch_summary.result_wait_s, 2),
            setup_s=round(batch_summary.setup_elapsed_s, 2),
            tear_s=round(batch_summary.teardown_elapsed_s, 2),
        )

    succeeded_raw_ids = successful_raw_ids(batch_summary)
    raw_state_update_elapsed_s = await _persist_batch_raw_state_updates(
        service,
        backend,
        outcomes=batch_summary.outcomes,
        succeeded_raw_ids=succeeded_raw_ids,
        skipped_raw_ids=batch_summary.skipped_raw_ids,
        failed_raw_ids=batch_summary.failed_raw_ids,
        validation_mode=validation_mode,
    )

    elapsed_s = time.perf_counter() - batch_started
    rss_end_mb = read_current_rss_mb()
    peak_rss_self_end_mb = read_peak_rss_self_mb()
    peak_rss_children_mb = read_peak_rss_children_mb()
    observation = _build_parse_batch_observation(
        batch_summary=batch_summary,
        elapsed_s=elapsed_s,
        raw_state_update_elapsed_s=raw_state_update_elapsed_s,
        rss_start_mb=rss_start_mb,
        rss_end_mb=rss_end_mb,
        peak_rss_self_start_mb=peak_rss_self_start_mb,
        peak_rss_self_end_mb=peak_rss_self_end_mb,
        peak_rss_children_mb=peak_rss_children_mb,
    )
    if heavy_batch:
        del batch_summary
        release_process_memory()
    return observation


def _successful_raw_state_update(
    *,
    outcome: _RawIngestOutcome | None,
    parsed_at: str,
    validation_mode: str,
) -> RawSessionStateUpdate:
    if outcome is None:
        return RawSessionStateUpdate(
            parsed_at=parsed_at,
            parse_error=None,
        )
    return RawSessionStateUpdate(
        parsed_at=parsed_at,
        parse_error=None,
        payload_provider=outcome.payload_provider,
        validation_status=outcome.validation_status,
        validation_error=outcome.validation_error,
        validation_mode=validation_mode,
    )


def _failed_raw_state_update(
    *,
    outcome: _RawIngestOutcome | None,
    error: str,
    validation_mode: str,
) -> RawSessionStateUpdate:
    if outcome is None:
        return RawSessionStateUpdate(
            parse_error=error,
            detection_warnings=error[:500] if error else None,
        )
    return RawSessionStateUpdate(
        parse_error=outcome.parse_error,
        detection_warnings=outcome.parse_error[:500] if outcome.parse_error else None,
        payload_provider=outcome.payload_provider,
        validation_status=outcome.validation_status,
        validation_error=outcome.validation_error or error,
        validation_mode=validation_mode,
    )


async def _persist_batch_raw_state_updates(
    service: _ParsingServiceRawStateLike,
    backend: _BulkConnectionBackendLike,
    *,
    outcomes: dict[str, _RawIngestOutcome],
    succeeded_raw_ids: set[str],
    skipped_raw_ids: set[str],
    failed_raw_ids: dict[str, str],
    validation_mode: str,
) -> float:
    now_iso = datetime.now(timezone.utc).isoformat()
    raw_state_update_started = time.perf_counter()
    async with backend.bulk_connection():
        for rid in succeeded_raw_ids:
            await service.repository.update_raw_state(
                rid,
                state=_successful_raw_state_update(
                    outcome=outcomes.get(rid),
                    parsed_at=now_iso,
                    validation_mode=validation_mode,
                ),
            )
        for rid in skipped_raw_ids:
            if rid in failed_raw_ids or rid in succeeded_raw_ids:
                continue
            await service.repository.update_raw_state(
                rid,
                state=_successful_raw_state_update(
                    outcome=outcomes.get(rid),
                    parsed_at=now_iso,
                    validation_mode=validation_mode,
                ),
            )
        for rid, error in failed_raw_ids.items():
            await service.repository.update_raw_state(
                rid,
                state=_failed_raw_state_update(
                    outcome=outcomes.get(rid),
                    error=error,
                    validation_mode=validation_mode,
                ),
            )
    return time.perf_counter() - raw_state_update_started


async def refresh_session_insights_bulk(
    backend: _ConnectionBackendLike,
    changed_session_ids: list[str],
) -> MaterializeStageObservation | None:
    """Bulk session insight refresh — once after all batches, not per-batch."""
    if not changed_session_ids:
        return None

    t_start = time.perf_counter()
    update_elapsed = 0.0
    thread_elapsed = 0.0
    aggregate_elapsed = 0.0
    try:
        from polylogue.storage.insights.session.refresh import (
            _apply_session_insight_session_updates_async,
            _refresh_thread_roots_async,
            refresh_async_provider_day_aggregates,
        )

        async with backend.connection() as conn:
            t_updates = time.perf_counter()
            update = await _apply_session_insight_session_updates_async(
                conn,
                changed_session_ids,
                transaction_depth=1,
            )
            update_elapsed = time.perf_counter() - t_updates
            t_threads = time.perf_counter()
            thread_root_ids = update.thread_root_ids
            await _refresh_thread_roots_async(
                conn,
                sorted(thread_root_ids),
                transaction_depth=1,
            )
            thread_elapsed = time.perf_counter() - t_threads
            t_aggregates = time.perf_counter()
            affected_groups = update.affected_groups
            if affected_groups:
                await refresh_async_provider_day_aggregates(
                    conn,
                    affected_groups,
                    transaction_depth=1,
                )
            aggregate_elapsed = time.perf_counter() - t_aggregates
            await conn.commit()

        elapsed = time.perf_counter() - t_start
        chunk_observations = update.chunk_observations
        observation: MaterializeStageObservation = {
            "sessions": len(changed_session_ids),
            "unique_thread_roots": len(thread_root_ids),
            "unique_provider_days": len(affected_groups),
            "elapsed_ms": round(elapsed * 1000.0, 1),
            "update_ms": round(update_elapsed * 1000.0, 1),
            "thread_refresh_ms": round(thread_elapsed * 1000.0, 1),
            "aggregate_refresh_ms": round(aggregate_elapsed * 1000.0, 1),
            "update_chunk_count": len(chunk_observations),
            "update_slow_chunk_count": sum(1 for chunk in chunk_observations if chunk.slow),
        }
        if chunk_observations:
            observation.update(
                {
                    "update_max_chunk_ms": round(max(chunk.total_ms for chunk in chunk_observations), 1),
                    "update_max_chunk_load_ms": round(max(chunk.load_ms for chunk in chunk_observations), 1),
                    "update_max_chunk_hydrate_ms": round(max(chunk.hydrate_ms for chunk in chunk_observations), 1),
                    "update_max_chunk_build_ms": round(max(chunk.build_ms for chunk in chunk_observations), 1),
                    "update_max_chunk_write_ms": round(max(chunk.write_ms for chunk in chunk_observations), 1),
                }
            )
            observation["update_chunks"] = [chunk.to_observation() for chunk in chunk_observations]
        if elapsed > 2.0:
            logger.info(
                "session_insight_refresh",
                sessions=len(changed_session_ids),
                unique_thread_roots=len(thread_root_ids),
                unique_provider_days=len(affected_groups),
                elapsed_s=round(elapsed, 2),
                update_s=round(update_elapsed, 2),
                thread_refresh_s=round(thread_elapsed, 2),
                aggregate_refresh_s=round(aggregate_elapsed, 2),
                rate=round(len(changed_session_ids) / elapsed, 1) if elapsed > 0 else 0,
            )
        return observation
    except Exception as exc:
        logger.warning("Session insight refresh failed (non-fatal): %s", exc, exc_info=True)
        return {
            "sessions": len(changed_session_ids),
            "failed": True,
            "error": str(exc),
        }


__all__ = ["_INGEST_RESULT_CHUNK_SIZE", "process_ingest_batch", "refresh_session_insights_bulk"]

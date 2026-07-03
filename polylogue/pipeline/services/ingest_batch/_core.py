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
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.archive.write_gateway import ArchiveWriteGateway, WriteOperation
from polylogue.core.memory import release_process_memory
from polylogue.core.metrics import (
    read_current_rss_mb,
    read_peak_rss_children_mb,
    read_peak_rss_self_mb,
)
from polylogue.logging import get_logger
from polylogue.paths import blob_store_root
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.payload_types import MaterializeStageObservation, ParseBatchObservation
from polylogue.pipeline.services.ingest_worker import (
    IngestRecordResult,
    SessionWritePayload,
    ingest_record,
)
from polylogue.pipeline.services.process_pool import process_pool_executor
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.parsers.browser_capture import DOM_FALLBACK_INGEST_FLAG
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.write import (
    upsert_parser_ingest_flag_tags,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.connection import _load_sqlite_vec
from polylogue.storage.sqlite.connection_profile import (
    DB_TIMEOUT,
    WRITE_CONNECTION_PRAGMA_STATEMENTS,
)
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

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
    if db_path.name == "index.db":
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        initialize_active_archive_root(db_path.parent)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    for statement in WRITE_CONNECTION_PRAGMA_STATEMENTS:
        conn.execute(statement)
    if db_path.name == "index.db":
        ensure_runtime_indexes_sync(conn)
    _load_sqlite_vec(conn)
    return conn


def _format_foreign_key_violations(
    rows: Sequence[sqlite3.Row | tuple[object, ...] | Mapping[str, object | None]],
) -> str:
    """Render PRAGMA foreign_key_check rows without sqlite3.Row object reprs."""
    formatted: list[dict[str, object | None]] = []
    columns = ("table", "rowid", "parent", "fkid")
    for row in rows:
        if isinstance(row, Mapping):
            formatted.append(dict(row))
            continue
        if isinstance(row, sqlite3.Row):
            formatted.append({key: row[key] for key in row.keys()})  # noqa: SIM118 - sqlite3.Row iterates values
            continue
        formatted.append({key: row[idx] if idx < len(row) else None for idx, key in enumerate(columns)})
    return repr(formatted)


def _foreign_key_violations_for_sessions(
    conn: sqlite3.Connection,
    session_ids: Iterable[str],
    *,
    limit: int = 10,
) -> list[dict[str, object | None]]:
    """Return transcript FK violations scoped to sessions changed in this batch."""
    scoped_session_ids = tuple(sorted({session_id for session_id in session_ids if session_id}))
    if not scoped_session_ids:
        return []
    placeholders = ",".join("?" for _ in scoped_session_ids)
    checks: tuple[tuple[str, str, int], ...] = (
        (
            "messages",
            f"""
            SELECT rowid, session_id, message_id, parent_message_id AS child_key
            FROM messages
            WHERE session_id IN ({placeholders})
              AND parent_message_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM messages parent
                WHERE parent.message_id = messages.parent_message_id
              )
            """,
            0,
        ),
        (
            "messages",
            f"""
            SELECT rowid, session_id, message_id, session_id AS child_key
            FROM messages
            WHERE session_id IN ({placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM sessions parent
                WHERE parent.session_id = messages.session_id
              )
            """,
            1,
        ),
        (
            "blocks",
            f"""
            SELECT rowid, session_id, message_id, session_id AS child_key
            FROM blocks
            WHERE session_id IN ({placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM sessions parent
                WHERE parent.session_id = blocks.session_id
              )
            """,
            0,
        ),
        (
            "blocks",
            f"""
            SELECT rowid, session_id, message_id, message_id AS child_key
            FROM blocks
            WHERE session_id IN ({placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM messages parent
                WHERE parent.message_id = blocks.message_id
              )
            """,
            1,
        ),
        (
            "web_content_constructs",
            f"""
            SELECT rowid, session_id, message_id, block_id AS child_key
            FROM web_content_constructs
            WHERE session_id IN ({placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM blocks parent
                WHERE parent.block_id = web_content_constructs.block_id
              )
            """,
            0,
        ),
        (
            "web_content_constructs",
            f"""
            SELECT rowid, session_id, message_id, message_id AS child_key
            FROM web_content_constructs
            WHERE session_id IN ({placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM messages parent
                WHERE parent.message_id = web_content_constructs.message_id
              )
            """,
            1,
        ),
        (
            "web_content_constructs",
            f"""
            SELECT rowid, session_id, message_id, session_id AS child_key
            FROM web_content_constructs
            WHERE session_id IN ({placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM sessions parent
                WHERE parent.session_id = web_content_constructs.session_id
              )
            """,
            2,
        ),
    )
    violations: list[dict[str, object | None]] = []
    for table, sql, fkid in checks:
        for row in conn.execute(sql, scoped_session_ids).fetchall():
            violations.append(
                {
                    "table": table,
                    "rowid": row["rowid"],
                    "parent": _SCOPED_FK_PARENTS[(table, fkid)],
                    "fkid": fkid,
                    "session_id": row["session_id"],
                    "message_id": row["message_id"],
                    "child_key": row["child_key"],
                }
            )
            if len(violations) >= limit:
                return violations
    return violations


_SCOPED_FK_PARENTS = {
    ("messages", 0): "messages",
    ("messages", 1): "sessions",
    ("blocks", 0): "sessions",
    ("blocks", 1): "messages",
    ("web_content_constructs", 0): "blocks",
    ("web_content_constructs", 1): "messages",
    ("web_content_constructs", 2): "sessions",
}


def _stored_message_count(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _session_has_parser_ingest_flag(conn: sqlite3.Connection, session_id: str, flag: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM session_tags
        WHERE session_id = ?
          AND tag = ?
          AND tag_source = 'auto'
          AND method = 'parser'
        LIMIT 1
        """,
        (session_id, flag),
    ).fetchone()
    return row is not None


def _incoming_has_ingest_flag(payload: SessionWritePayload, flag: str) -> bool:
    return flag in payload.parsed_session.ingest_flags


def _record_capture_gap_event(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    existing_raw_id: str,
    incoming_raw_id: str,
    stored_message_count: int,
    incoming_message_count: int,
) -> None:
    row = conn.execute(
        """
        SELECT MAX(position) + 1
        FROM (
            SELECT position FROM session_events WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_agent_policies WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_provider_usage_events WHERE session_id = ?
        )
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    position = int(row[0] or 0) if row is not None else 0
    summary = (
        "Skipped lower-precedence DOM browser-capture fallback "
        f"{incoming_raw_id!r}; existing raw {existing_raw_id!r} has "
        f"{stored_message_count} message(s), incoming fallback has {incoming_message_count}."
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO session_events (
            session_id, source_message_id, position, event_type, summary, occurred_at_ms
        ) VALUES (?, NULL, ?, 'capture_gap', ?, NULL)
        """,
        (session_id, position, summary),
    )


def _session_parent_id(payload: SessionWritePayload) -> str | None:
    parent_native_id = payload.parsed_session.parent_session_provider_id
    if not parent_native_id:
        return None
    return str(make_session_id(payload.parsed_session.source_name, parent_native_id))


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


_WRITE_SELECT_CHUNK_SIZE = 900
_FTS_REPAIR_COUNT_KEY = "_fts_repair"


def _needs_session_fts_repair(conn: sqlite3.Connection, session_id: str) -> bool:
    from polylogue.storage.fts.session_repair import session_fts_needs_repair_sync

    return session_fts_needs_repair_sync(conn, session_id)


def _existing_native_message_ids(conn: sqlite3.Connection, session_id: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT native_id FROM messages WHERE session_id = ? AND native_id IS NOT NULL",
            (session_id,),
        ).fetchall()
    }


def _append_delta_payload(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
) -> tuple[ParsedSession | None, int]:
    existing_native_ids = _existing_native_message_ids(conn, payload.session_id)
    delta_messages: list[ParsedMessage] = []
    for message in payload.parsed_session.messages:
        if message.provider_message_id in existing_native_ids:
            continue
        delta_messages.append(message.model_copy(update={"position": None}))
    if not delta_messages and not payload.attachment_count:
        return None, len(payload.parsed_session.messages)
    return payload.parsed_session.model_copy(update={"messages": delta_messages}), len(
        payload.parsed_session.messages
    ) - len(delta_messages)


def _write_session(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
    *,
    force_write: bool = False,
    signature_cache: dict[str, list[tuple[str, str]]] | None = None,
    stage_timings_s: dict[str, float] | None = None,
) -> tuple[bool, dict[str, int]]:
    """Write one parsed session payload into the current archive index.

    Returns (content_changed, counts).
    """
    counts: dict[str, int] = {
        "sessions": 0,
        "messages": 0,
        "attachments": 0,
        "session_events": 0,
        "skipped_sessions": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
        "skipped_session_events": 0,
    }

    existing_row = conn.execute(
        "SELECT content_hash, raw_id FROM sessions WHERE session_id = ?",
        (payload.session_id,),
    ).fetchone()
    existing_hash = existing_row["content_hash"] if existing_row is not None else None
    existing_hash_hex = existing_hash.hex() if isinstance(existing_hash, bytes) else str(existing_hash or "")
    content_unchanged = existing_row is not None and existing_hash_hex == payload.content_hash
    session_to_write = payload.parsed_session
    merge_append = False
    if payload.append_only and existing_row is not None:
        delta, skipped_messages = _append_delta_payload(conn, payload)
        counts["skipped_messages"] = skipped_messages
        if delta is None:
            if payload.parsed_session.ingest_flags:
                upsert_parser_ingest_flag_tags(conn, payload.session_id, payload.parsed_session.ingest_flags)
            counts["skipped_sessions"] = 1
            counts["skipped_attachments"] = payload.attachment_count
            counts["skipped_session_events"] = len(payload.parsed_session.session_events)
            if _needs_session_fts_repair(conn, payload.session_id):
                counts[_FTS_REPAIR_COUNT_KEY] = 1
            return False, counts
        session_to_write = delta
        merge_append = True

    if not force_write and content_unchanged:
        if payload.parsed_session.ingest_flags:
            upsert_parser_ingest_flag_tags(conn, payload.session_id, payload.parsed_session.ingest_flags)
        counts["skipped_sessions"] = 1
        counts["skipped_messages"] = payload.message_count
        counts["skipped_attachments"] = payload.attachment_count
        counts["skipped_session_events"] = len(payload.parsed_session.session_events)
        if _needs_session_fts_repair(conn, payload.session_id):
            counts[_FTS_REPAIR_COUNT_KEY] = 1
        return False, counts

    existing_raw_id = str(existing_row["raw_id"] or "") if existing_row is not None else ""
    if not force_write and existing_raw_id and payload.raw_id and existing_raw_id != payload.raw_id:
        existing_is_dom_fallback = _session_has_parser_ingest_flag(conn, payload.session_id, DOM_FALLBACK_INGEST_FLAG)
        incoming_is_dom_fallback = _incoming_has_ingest_flag(payload, DOM_FALLBACK_INGEST_FLAG)
        stored_message_count = _stored_message_count(conn, payload.session_id)
        lower_precedence_fallback = incoming_is_dom_fallback and not existing_is_dom_fallback
        strictly_less_complete = payload.message_count < stored_message_count and not (
            existing_is_dom_fallback and not incoming_is_dom_fallback
        )
        if lower_precedence_fallback or strictly_less_complete:
            if lower_precedence_fallback:
                _record_capture_gap_event(
                    conn,
                    session_id=payload.session_id,
                    existing_raw_id=existing_raw_id,
                    incoming_raw_id=payload.raw_id,
                    stored_message_count=stored_message_count,
                    incoming_message_count=payload.message_count,
                )
                counts["session_events"] = 1
            counts["skipped_sessions"] = 1
            counts["skipped_messages"] = payload.message_count
            counts["skipped_attachments"] = payload.attachment_count
            counts["skipped_session_events"] = len(payload.parsed_session.session_events)
            return False, counts

    if existing_row is None and not payload.parsed_session.messages and not force_write:
        counts["skipped_sessions"] = 1
        return False, counts

    write_parsed_session_to_archive(
        conn,
        session_to_write,
        content_hash=payload.content_hash,
        raw_id=payload.raw_id,
        merge_append=merge_append,
        signature_cache=signature_cache,
        stage_timings_s=stage_timings_s,
    )
    counts["sessions"] = 1
    counts["messages"] = len(session_to_write.messages)
    counts["attachments"] = len(session_to_write.attachments)
    counts["session_events"] = len(session_to_write.session_events)

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
    cdata: SessionWritePayload,
    *,
    content_changed: bool,
    counts: dict[str, int],
) -> None:
    summary.total_convos += 1
    summary.total_msgs += cdata.message_count

    ingest_changed = (counts["sessions"] + counts["messages"] + counts["attachments"] + counts["session_events"]) > 0

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
    if counts["session_events"]:
        summary.changed_counts["session_events"] += counts["session_events"]
    for key, value in counts.items():
        if key in summary.counts:
            summary.counts[key] += value


def _write_session_entry(
    conn: sqlite3.Connection,
    raw_id: str,
    cdata: SessionWritePayload,
    *,
    summary: _IngestBatchSummary,
    force_write: bool = False,
    signature_cache: dict[str, list[tuple[str, str]]] | None = None,
) -> bool:
    try:
        t_write = time.perf_counter()
        write_stage_timings: dict[str, float] = {}
        content_changed, counts = _write_session(
            conn,
            cdata,
            force_write=force_write,
            signature_cache=signature_cache,
            stage_timings_s=write_stage_timings,
        )
        for stage, elapsed_s in write_stage_timings.items():
            summary.stage_timings_s[stage] = summary.stage_timings_s.get(stage, 0.0) + elapsed_s
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
                msgs=cdata.message_count,
                changed_messages=counts["messages"],
                skipped_messages=counts["skipped_messages"],
                changed_session_events=counts["session_events"],
                attachments=cdata.attachment_count,
                stage_top=_top_stage_timings(write_stage_timings),
            )
        return True
    except Exception as exc:
        logger.error("Error writing session: %s", exc)
        summary.parse_failures += 1
        summary.failed_raw_ids[raw_id] = str(exc)[:500]
        return False


def _top_stage_timings(stage_timings_s: dict[str, float], *, limit: int = 5) -> dict[str, float]:
    if not stage_timings_s:
        return {}
    return {
        stage: round(elapsed_s, 3)
        for stage, elapsed_s in sorted(stage_timings_s.items(), key=lambda item: item[1], reverse=True)[:limit]
    }


def _delete_stale_sessions_for_raw_entries(conn: sqlite3.Connection, ready_entries: list[_SessionEntry]) -> None:
    if not hasattr(conn, "execute"):
        return

    expected_by_raw_id: dict[str, set[str]] = {}
    for raw_id, cdata in ready_entries:
        if raw_id:
            expected_by_raw_id.setdefault(raw_id, set()).add(cdata.session_id)

    for raw_id, expected_session_ids in expected_by_raw_id.items():
        if not expected_session_ids:
            continue
        placeholders = ",".join("?" for _ in expected_session_ids)
        stale_session_ids = [
            str(row[0])
            for row in conn.execute(
                f"SELECT session_id FROM sessions WHERE raw_id = ? AND session_id NOT IN ({placeholders})",
                (raw_id, *sorted(expected_session_ids)),
            ).fetchall()
        ]
        if not stale_session_ids:
            continue
        _delete_sessions_without_fk_cascade(conn, stale_session_ids)
        conn.execute(
            f"DELETE FROM sessions WHERE raw_id = ? AND session_id NOT IN ({placeholders})",
            (raw_id, *sorted(expected_session_ids)),
        )


def _delete_sessions_without_fk_cascade(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Apply sessions FK actions manually while bulk ingest has FKs disabled."""
    if not session_ids:
        return
    placeholders = ",".join("?" for _ in session_ids)
    params = tuple(session_ids)
    for table_name, from_column, on_delete in _session_foreign_key_actions(conn):
        table = _quote_identifier(table_name)
        column = _quote_identifier(from_column)
        if table_name == "sessions" or on_delete.upper() == "SET NULL":
            conn.execute(f"UPDATE {table} SET {column} = NULL WHERE {column} IN ({placeholders})", params)
        elif on_delete.upper() == "CASCADE":
            conn.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", params)


def _session_foreign_key_actions(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    actions: list[tuple[str, str, str]] = []
    table_rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    for table_row in table_rows:
        table_name = str(table_row[0])
        for fk in conn.execute(f"PRAGMA foreign_key_list({_quote_identifier(table_name)})").fetchall():
            if str(fk[2]) == "sessions":
                actions.append((table_name, str(fk[3]), str(fk[6] or "")))
    actions.sort(key=lambda item: item[0] == "sessions")
    return actions


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _drain_ready_session_entries(
    conn: sqlite3.Connection,
    ready_entries: list[_SessionEntry],
    *,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    force_write: bool = False,
) -> int:
    _delete_stale_sessions_for_raw_entries(conn, ready_entries)
    written_count = 0
    # One signature cache per drained batch memoizes each session's own composed
    # signatures so a parent with K fork-children is computed once, not K times
    # (#2475, hotspot 1). Entries are invalidated when their own rows are
    # rewritten or re-extracted in this same batch.
    signature_cache: dict[str, list[tuple[str, str]]] = {}
    for raw_id, cdata in _topo_sort_session_entries(ready_entries):
        wrote = _write_session_entry(
            conn, raw_id, cdata, summary=summary, force_write=force_write, signature_cache=signature_cache
        )
        discard_session_data_payload(cdata)
        if not wrote:
            continue
        written_count += 1
        materialized_ids.add(cdata.session_id)
    return written_count


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
    written_count = _drain_ready_session_entries(
        conn,
        [(ir.raw_id, cdata) for cdata in ir.sessions],
        summary=summary,
        materialized_ids=materialized_ids,
        force_write=force_write,
    )
    if written_count == 0:
        summary.skipped_raw_ids.add(ir.raw_id)
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
    mark_fts_stale_on_suspend: bool = False,
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
                if suspend_fts_triggers:
                    conn.execute("PRAGMA foreign_keys = OFF")
                conn.execute("BEGIN IMMEDIATE")
                if suspend_fts_triggers:
                    from polylogue.storage.fts.fts_lifecycle import suspend_fts_triggers_sync

                    suspend_fts_triggers_sync(conn, mark_stale=mark_fts_stale_on_suspend)
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
) -> None:
    """Run post-ingest side effects through the canonical write-effects path."""
    ArchiveWriteGateway(db_path).commit_write_sync(
        WriteOperation.INGEST,
        {
            "_connection": conn,
            "changed_session_ids": tuple(changed_session_ids),
            "repair_message_fts": repair_message_fts,
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
            mark_fts_stale_on_suspend=suspend_fts_triggers and not repair_message_fts,
            force_process_pool=force_process_pool,
        )
        _flush_ingest_results(
            conn,
            summary=summary,
        )
        if transaction_started:
            if suspend_fts_triggers:
                fk_violations = _foreign_key_violations_for_sessions(conn, materialized_ids)
                if fk_violations:
                    detail = _format_foreign_key_violations(fk_violations)
                    raise sqlite3.IntegrityError(f"foreign key check failed during bulk ingest: {detail}")
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
                repair_message_fts=repair_message_fts,
            )
            summary.commit_elapsed_s = time.perf_counter() - commit_started
            from polylogue.storage.sqlite.wal_checkpoint import maybe_checkpoint_wal

            wal_observation = maybe_checkpoint_wal(db_path, reason="ingest_batch_commit", allow_truncate=False)
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
            from polylogue.storage.sqlite.maintenance import maybe_optimize_sqlite

            optimize_observation = maybe_optimize_sqlite(conn, reason="ingest_batch_commit")
            if optimize_observation.error is not None:
                logger.warning(
                    "sqlite_optimize_failed",
                    reason=optimize_observation.reason,
                    analysis_limit=optimize_observation.analysis_limit,
                    error=optimize_observation.error,
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
        if suspend_fts_triggers:
            with contextlib.suppress(Exception):
                conn.execute("PRAGMA foreign_keys = ON")
        conn.close()
    summary.worker_progress_in_flight = len(progress.in_flight_raw_ids)
    summary.worker_progress_completed = progress.completed_raw_count
    summary.worker_progress_total = progress.total_raw_count
    summary.elapsed_s = time.perf_counter() - t_start
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
    repair_message_fts: bool = True,
    ingest_result_chunk_size: int = 0,
    suspend_fts_triggers: bool = False,
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
        repair_message_fts=repair_message_fts,
        ingest_result_chunk_size=ingest_result_chunk_size,
        suspend_fts_triggers=suspend_fts_triggers,
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


def _skipped_raw_state_update(
    *,
    outcome: _RawIngestOutcome | None,
    parsed_at: str,
    validation_mode: str,
) -> RawSessionStateUpdate:
    return RawSessionStateUpdate(
        parsed_at=parsed_at,
        parse_error=None,
        payload_provider=outcome.payload_provider if outcome is not None else None,
        validation_status="skipped",
        validation_error="parsed raw payload produced no new materialized sessions",
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
    source_backend = getattr(service.repository, "_source_backend", None)
    async with AsyncExitStack() as stack:
        raw_state_backend = source_backend if source_backend is not None else backend
        await stack.enter_async_context(raw_state_backend.bulk_connection())
        for rid in succeeded_raw_ids:
            if rid in skipped_raw_ids:
                continue
            await service.repository.update_raw_state(
                rid,
                state=_successful_raw_state_update(
                    outcome=outcomes.get(rid),
                    parsed_at=now_iso,
                    validation_mode=validation_mode,
                ),
            )
        for rid in skipped_raw_ids:
            if rid in failed_raw_ids:
                continue
            await service.repository.update_raw_state(
                rid,
                state=_skipped_raw_state_update(
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

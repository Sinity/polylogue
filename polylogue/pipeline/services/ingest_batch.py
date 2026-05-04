"""Batch ingest orchestration: ProcessPool workers + sync sqlite3 writes.

Architecture:
- CPU-bound work (decode/validate/parse/transform) in ProcessPoolExecutor
- DB writes in main thread via sync sqlite3 (no aiosqlite async overhead)
- as_completed yields results as workers finish; writes run after all results collected

Replaces: parsing_batch.py, parsing_workflow.py, validation_flow.py.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, as_completed
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.archive.write_effects import commit_archive_write_effects
from polylogue.archive.write_gateway import WriteOperation
from polylogue.core.metrics import (
    read_current_rss_mb,
    read_peak_rss_children_mb,
    read_peak_rss_self_mb,
)
from polylogue.logging import get_logger
from polylogue.paths import blob_store_root
from polylogue.pipeline.payload_types import MaterializeStageObservation, ParseBatchObservation
from polylogue.pipeline.services.ingest_worker import (
    ConversationData,
    ConversationTuple,
    IngestRecordResult,
    MessageTuple,
    ingest_record,
)
from polylogue.pipeline.services.process_pool import process_pool_executor
from polylogue.storage.conversation_replacement import (
    recount_and_prune_attachments_sync,
    replace_conversation_runtime_state_sync,
)
from polylogue.storage.raw.models import RawConversationStateUpdate
from polylogue.storage.runtime import RawConversationRecord
from polylogue.storage.sqlite.connection import _load_sqlite_vec

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.protocols import ProgressCallback
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

_DEFAULT_INGEST_WORKER_LIMIT = 8
_INGEST_SOFT_BLOB_LIMIT_BYTES = 48 * 1024 * 1024
_INGEST_HIGH_BLOB_LIMIT_BYTES = 96 * 1024 * 1024
_INGEST_EXTREME_BLOB_LIMIT_BYTES = 256 * 1024 * 1024

_IDENTITY_LEDGER_UPSERT_SQL = """
INSERT OR IGNORE INTO identity_ledger (
    provider, source, source_path, provider_conversation_id, raw_hash, current_conversation_id
) VALUES (?, ?, ?, ?, ?, ?)
"""


class _RawStateRepositoryLike(Protocol):
    async def update_raw_state(self, raw_id: str, *, state: RawConversationStateUpdate) -> object: ...


class _ParsingServiceRawStateLike(Protocol):
    @property
    def repository(self) -> _RawStateRepositoryLike: ...


class _BulkConnectionBackendLike(Protocol):
    def bulk_connection(self) -> AbstractAsyncContextManager[object]: ...


class _ConnectionBackendLike(Protocol):
    def connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...


@dataclass(slots=True)
class _RawIngestOutcome:
    raw_id: str
    payload_provider: str | None
    validation_status: str
    validation_error: str | None
    parse_error: str | None
    error: str | None
    had_conversations: bool


@dataclass(slots=True)
class _IngestBatchSummary:
    outcomes: dict[str, _RawIngestOutcome] = field(default_factory=dict)
    failed_raw_ids: dict[str, str] = field(default_factory=dict)
    skipped_raw_ids: set[str] = field(default_factory=set)
    processed_ids: set[str] = field(default_factory=set)
    changed_conversation_ids: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }
    )
    changed_counts: dict[str, int] = field(
        default_factory=lambda: {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
        }
    )
    parse_failures: int = 0
    total_msgs: int = 0
    total_convos: int = 0
    raw_record_count: int = 0
    worker_count: int = 0
    total_blob_mb: float = 0.0
    total_result_bytes: int = 0
    max_result_bytes: int = 0
    max_result_raw_id: str | None = None
    elapsed_s: float = 0.0
    setup_elapsed_s: float = 0.0
    max_current_rss_mb: float | None = None
    result_wait_s: float = 0.0
    drain_elapsed_s: float = 0.0
    write_elapsed_s: float = 0.0
    max_write_elapsed_s: float = 0.0
    flush_elapsed_s: float = 0.0
    commit_elapsed_s: float = 0.0
    teardown_elapsed_s: float = 0.0


@dataclass(frozen=True, slots=True)
class _IngestWorkerRequest:
    archive_root_str: str
    blob_root_str: str
    validation_mode: str
    measure_ingest_result_size: bool


_ConversationEntry = tuple[str, ConversationData]


# ---------------------------------------------------------------------------
# SQL statements (copied from async query modules — sync versions)
# ---------------------------------------------------------------------------

_CONVERSATION_UPSERT_SQL = """
INSERT INTO conversations (
    conversation_id, provider_name, provider_conversation_id, title,
    created_at, updated_at, sort_key, content_hash,
    provider_meta, metadata, version,
    parent_conversation_id, branch_type, raw_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
    title = excluded.title,
    created_at = excluded.created_at,
    updated_at = excluded.updated_at,
    sort_key = excluded.sort_key,
    content_hash = excluded.content_hash,
    provider_meta = excluded.provider_meta,
    metadata = COALESCE(excluded.metadata, conversations.metadata),
    parent_conversation_id = excluded.parent_conversation_id,
    branch_type = excluded.branch_type,
    raw_id = COALESCE(excluded.raw_id, conversations.raw_id)
WHERE
    content_hash != excluded.content_hash
    OR IFNULL(title, '') != IFNULL(excluded.title, '')
    OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
    OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
    OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
    OR IFNULL(parent_conversation_id, '') != IFNULL(excluded.parent_conversation_id, '')
    OR IFNULL(branch_type, '') != IFNULL(excluded.branch_type, '')
    OR IFNULL(raw_id, '') != IFNULL(excluded.raw_id, '')
    OR IFNULL(sort_key, 0) != IFNULL(excluded.sort_key, 0)
"""

_MESSAGE_UPSERT_SQL = """
INSERT INTO messages (
    message_id, conversation_id, provider_message_id, role, text,
    sort_key, content_hash, version, parent_message_id, branch_index,
    provider_name, word_count, has_tool_use, has_thinking, has_paste, message_type
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(message_id) DO UPDATE SET
    role = excluded.role,
    text = excluded.text,
    sort_key = excluded.sort_key,
    content_hash = excluded.content_hash,
    parent_message_id = excluded.parent_message_id,
    branch_index = excluded.branch_index,
    provider_name = excluded.provider_name,
    word_count = excluded.word_count,
    has_tool_use = excluded.has_tool_use,
    has_thinking = excluded.has_thinking,
    has_paste = excluded.has_paste,
    message_type = excluded.message_type
WHERE
    content_hash != excluded.content_hash
    OR IFNULL(role, '') != IFNULL(excluded.role, '')
    OR IFNULL(text, '') != IFNULL(excluded.text, '')
    OR IFNULL(sort_key, 0) != IFNULL(excluded.sort_key, 0)
    OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
    OR branch_index != excluded.branch_index
    OR word_count != excluded.word_count
    OR has_tool_use != excluded.has_tool_use
    OR has_thinking != excluded.has_thinking
    OR has_paste != excluded.has_paste
    OR message_type != excluded.message_type
"""

_CONTENT_BLOCK_UPSERT_SQL = """
INSERT INTO content_blocks (
    block_id, message_id, conversation_id, block_index,
    type, text, tool_name, tool_id, tool_input,
    media_type, metadata, semantic_type
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(message_id, block_index) DO UPDATE SET
    type = excluded.type,
    text = excluded.text,
    tool_name = excluded.tool_name,
    tool_id = excluded.tool_id,
    tool_input = excluded.tool_input,
    media_type = excluded.media_type,
    metadata = excluded.metadata,
    semantic_type = excluded.semantic_type
"""

_STATS_UPSERT_SQL = """
INSERT INTO conversation_stats
    (conversation_id, provider_name, message_count, word_count, tool_use_count, thinking_count, paste_count)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
    provider_name  = excluded.provider_name,
    message_count  = excluded.message_count,
    word_count     = excluded.word_count,
    tool_use_count = excluded.tool_use_count,
    thinking_count = excluded.thinking_count,
    paste_count    = excluded.paste_count
"""

_ACTION_EVENT_INSERT_SQL = """
INSERT INTO action_events (
    event_id, conversation_id, message_id, materializer_version,
    source_block_id, timestamp, sort_key, sequence_index,
    provider_name, action_kind, tool_name, normalized_tool_name, tool_id,
    affected_paths_json, cwd_path, branch_names_json,
    command, query_text, url, output_text, search_text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_ATTACHMENT_UPSERT_SQL = """
INSERT INTO attachments (
    attachment_id, mime_type, size_bytes, path, ref_count, provider_meta
) VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(attachment_id) DO UPDATE SET
    mime_type = COALESCE(excluded.mime_type, attachments.mime_type),
    size_bytes = COALESCE(excluded.size_bytes, attachments.size_bytes),
    path = COALESCE(excluded.path, attachments.path),
    provider_meta = COALESCE(excluded.provider_meta, attachments.provider_meta)
"""

_ATTACHMENT_REF_INSERT_SQL = """
INSERT OR IGNORE INTO attachment_refs (
    ref_id, attachment_id, conversation_id, message_id, provider_meta
) VALUES (?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# Sync DB writer
# ---------------------------------------------------------------------------


def _write_identity_ledger(conn: sqlite3.Connection, cdata: ConversationData) -> None:
    """Write identity ledger row so re-imported conversations are recognized."""
    source_path = cdata.source_name
    provider_conversation_id = cdata.conversation_tuple[2]
    conn.execute(
        _IDENTITY_LEDGER_UPSERT_SQL,
        (
            cdata.provider_name,
            cdata.source_name,
            source_path,
            provider_conversation_id,
            cdata.content_hash,
            cdata.conversation_id,
        ),
    )


def _open_sync_connection(db_path: Path) -> sqlite3.Connection:
    """Open a sync sqlite3 connection with the same pragmas as the async backend."""
    from polylogue.storage.sqlite.connection_profile import open_connection

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_connection(db_path)
    conn.row_factory = sqlite3.Row
    _load_sqlite_vec(conn)
    return conn


def _check_content_unchanged(conn: sqlite3.Connection, cid: str, content_hash: str) -> bool:
    """Check if conversation content is unchanged (skip message writes)."""
    row = conn.execute(
        "SELECT content_hash FROM conversations WHERE conversation_id = ?",
        (cid,),
    ).fetchone()
    return row is not None and row[0] == content_hash


def _topo_sort_message_tuples(tuples: list[MessageTuple]) -> list[MessageTuple]:
    """Sort message tuples so parents come before children (FK constraint).

    message_id is at index 0, parent_message_id is at index 8.
    """
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


def _conversation_parent_id(cdata: ConversationData) -> str | None:
    parent_id = cdata.conversation_tuple[11]
    return parent_id if isinstance(parent_id, str) else None


def _topo_sort_conversation_entries(
    entries: list[_ConversationEntry],
) -> list[_ConversationEntry]:
    """Sort conversation entries so parents in the same batch precede children."""
    ids_in_batch = {entry[1].conversation_id for entry in entries}
    no_parent: list[_ConversationEntry] = []
    has_parent: list[_ConversationEntry] = []

    for entry in entries:
        parent_id = _conversation_parent_id(entry[1])
        if parent_id and parent_id in ids_in_batch and parent_id != entry[1].conversation_id:
            has_parent.append(entry)
        else:
            no_parent.append(entry)

    if not has_parent:
        return entries

    ordered = list(no_parent)
    inserted_ids = {entry[1].conversation_id for entry in ordered}
    remaining = list(has_parent)
    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        next_remaining: list[_ConversationEntry] = []
        for entry in remaining:
            parent_id = _conversation_parent_id(entry[1])
            if parent_id in inserted_ids:
                ordered.append(entry)
                inserted_ids.add(entry[1].conversation_id)
            else:
                next_remaining.append(entry)
        remaining = next_remaining
    ordered.extend(remaining)
    return ordered


def _conversation_exists(conn: sqlite3.Connection, conversation_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    return row is not None


def _resolved_conversation_tuple(
    conn: sqlite3.Connection,
    cdata: ConversationData,
) -> ConversationTuple:
    """Resolve parent conversation links against the currently materialized archive.

    Parent conversation links are only durable when the parent already exists.
    This mirrors the prepare/enrichment path and avoids rejecting the whole
    conversation when a child arrives before its parent.
    """
    parent_id = _conversation_parent_id(cdata)
    if parent_id is None or parent_id == cdata.conversation_id:
        return cdata.conversation_tuple
    if _conversation_exists(conn, parent_id):
        return cdata.conversation_tuple
    return (
        cdata.conversation_tuple[0],
        cdata.conversation_tuple[1],
        cdata.conversation_tuple[2],
        cdata.conversation_tuple[3],
        cdata.conversation_tuple[4],
        cdata.conversation_tuple[5],
        cdata.conversation_tuple[6],
        cdata.conversation_tuple[7],
        cdata.conversation_tuple[8],
        cdata.conversation_tuple[9],
        cdata.conversation_tuple[10],
        None,
        cdata.conversation_tuple[12],
        cdata.conversation_tuple[13],
    )


def _parent_ready(
    conn: sqlite3.Connection,
    cdata: ConversationData,
    materialized_ids: set[str],
) -> bool:
    parent_id = _conversation_parent_id(cdata)
    if parent_id is None or parent_id == cdata.conversation_id:
        return True
    return parent_id in materialized_ids or _conversation_exists(conn, parent_id)


def _write_conversation(
    conn: sqlite3.Connection, cdata: ConversationData, *, force_write: bool = False
) -> tuple[bool, dict[str, int]]:
    """Write one conversation's data to DB via sync sqlite3.

    Returns (content_changed, counts).
    """
    counts: dict[str, int] = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }

    content_unchanged = _check_content_unchanged(conn, cdata.conversation_id, cdata.content_hash)

    if not force_write and content_unchanged:
        counts["skipped_conversations"] = 1
        counts["skipped_messages"] = len(cdata.message_tuples)
        counts["skipped_attachments"] = len(cdata.attachment_tuples)
        return False, counts

    conn.execute(_CONVERSATION_UPSERT_SQL, _resolved_conversation_tuple(conn, cdata))

    affected_attachment_ids: set[str] = set()
    if not content_unchanged:
        counts["conversations"] = 1
        affected_attachment_ids = replace_conversation_runtime_state_sync(conn, cdata.conversation_id)
    else:
        counts["conversations"] = 0

    if cdata.message_tuples:
        sorted_msgs = _topo_sort_message_tuples(cdata.message_tuples)
        conn.executemany(_MESSAGE_UPSERT_SQL, sorted_msgs)
        counts["messages"] = len(sorted_msgs)

    # Conversation stats
    if cdata.stats_tuple and not content_unchanged:
        conn.execute(_STATS_UPSERT_SQL, cdata.stats_tuple)

    if not content_unchanged:
        conn.execute("DELETE FROM content_blocks WHERE conversation_id = ?", (cdata.conversation_id,))
        if cdata.block_tuples:
            conn.executemany(_CONTENT_BLOCK_UPSERT_SQL, cdata.block_tuples)

    if not content_unchanged:
        conn.execute("DELETE FROM action_events WHERE conversation_id = ?", (cdata.conversation_id,))
        if cdata.action_event_tuples:
            conn.executemany(_ACTION_EVENT_INSERT_SQL, cdata.action_event_tuples)

    # Attachments
    if not content_unchanged:
        new_attachment_ids = {
            attachment_id for attachment_id, *_rest in cdata.attachment_tuples if isinstance(attachment_id, str)
        }
        affected_attachment_ids |= new_attachment_ids
        if cdata.attachment_tuples:
            conn.executemany(_ATTACHMENT_UPSERT_SQL, cdata.attachment_tuples)
            conn.executemany(_ATTACHMENT_REF_INSERT_SQL, cdata.attachment_ref_tuples)
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
        had_conversations=bool(ir.conversations),
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
    cdata: ConversationData,
    *,
    content_changed: bool,
    counts: dict[str, int],
) -> None:
    summary.total_convos += 1
    summary.total_msgs += len(cdata.message_tuples)

    ingest_changed = (counts["conversations"] + counts["messages"] + counts["attachments"]) > 0

    if ingest_changed or content_changed:
        summary.processed_ids.add(cdata.conversation_id)
    if content_changed:
        summary.changed_counts["conversations"] += 1
        summary.changed_conversation_ids.append(cdata.conversation_id)
    if counts["messages"]:
        summary.changed_counts["messages"] += counts["messages"]
    if counts["attachments"]:
        summary.changed_counts["attachments"] += counts["attachments"]
    for key, value in counts.items():
        if key in summary.counts:
            summary.counts[key] += value


def _write_conversation_entry(
    conn: sqlite3.Connection,
    raw_id: str,
    cdata: ConversationData,
    *,
    summary: _IngestBatchSummary,
    force_write: bool = False,
) -> bool:
    try:
        t_write = time.perf_counter()
        content_changed, counts = _write_conversation(conn, cdata, force_write=force_write)
        write_elapsed = time.perf_counter() - t_write
        summary.write_elapsed_s += write_elapsed
        if write_elapsed > summary.max_write_elapsed_s:
            summary.max_write_elapsed_s = write_elapsed
        if content_changed:
            _write_identity_ledger(conn, cdata)
        _record_write_result(
            summary,
            cdata,
            content_changed=content_changed,
            counts=counts,
        )
        if write_elapsed >= 1.0:
            logger.info(
                "slow_write",
                cid=cdata.conversation_id[:20],
                elapsed_s=round(write_elapsed, 2),
                msgs=len(cdata.message_tuples),
            )
        return True
    except Exception as exc:
        logger.error("Error writing conversation: %s", exc)
        summary.parse_failures += 1
        summary.failed_raw_ids[raw_id] = str(exc)[:500]
        return False


def _drain_ready_conversation_entries(
    conn: sqlite3.Connection,
    ready_entries: list[_ConversationEntry],
    *,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    pending_by_parent: dict[str, list[_ConversationEntry]],
    force_write: bool = False,
) -> None:
    stack = list(reversed(ready_entries))
    while stack:
        raw_id, cdata = stack.pop()
        if not _parent_ready(conn, cdata, materialized_ids):
            parent_id = _conversation_parent_id(cdata)
            if parent_id is not None:
                pending_by_parent.setdefault(parent_id, []).append((raw_id, cdata))
            continue
        if not _write_conversation_entry(conn, raw_id, cdata, summary=summary, force_write=force_write):
            continue
        materialized_ids.add(cdata.conversation_id)
        children = pending_by_parent.pop(cdata.conversation_id, [])
        if children:
            stack.extend(reversed(children))


def _flush_pending_conversation_entries(
    conn: sqlite3.Connection,
    pending_by_parent: dict[str, list[_ConversationEntry]],
    *,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
) -> None:
    remaining = [entry for entries in pending_by_parent.values() for entry in entries]
    if not remaining:
        return
    pending_by_parent.clear()
    for raw_id, cdata in _topo_sort_conversation_entries(remaining):
        if _write_conversation_entry(conn, raw_id, cdata, summary=summary):
            materialized_ids.add(cdata.conversation_id)


def _run_ingest_record(
    raw_record: RawConversationRecord,
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
    raw_artifacts: list[RawConversationRecord],
    *,
    request: _IngestWorkerRequest,
    worker_count: int,
) -> Iterable[IngestRecordResult]:
    if worker_count <= 1:
        for raw_record in raw_artifacts:
            yield _run_ingest_record(raw_record, request)
        return

    try:
        with process_pool_executor(max_workers=worker_count) as executor:
            futures: dict[Future[IngestRecordResult], str] = {}
            for raw_record in raw_artifacts:
                future = executor.submit(_run_ingest_record, raw_record, request)
                futures[future] = raw_record.raw_id

            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as exc:
                    raw_id = futures[future]
                    yield IngestRecordResult(
                        raw_id=raw_id,
                        error=f"worker: {exc}",
                    )
    except (TypeError, pickle.PicklingError):
        for raw_record in raw_artifacts:
            yield _run_ingest_record(raw_record, request)


def _resolved_ingest_worker_limit(value: int | None) -> int:
    return value if value is not None else _DEFAULT_INGEST_WORKER_LIMIT


def _record_blob_size(record: object) -> int:
    return max(int(getattr(record, "blob_size", 0) or 0), 0)


def _select_ingest_worker_count(raw_artifacts: Sequence[object], ingest_workers: int | None) -> int:
    base_worker_count = min(
        len(raw_artifacts),
        os.cpu_count() or 4,
        _resolved_ingest_worker_limit(ingest_workers),
    )
    if base_worker_count <= 1:
        return base_worker_count

    blob_sizes = [_record_blob_size(record) for record in raw_artifacts]
    total_blob_bytes = sum(blob_sizes)
    max_blob_bytes = max(blob_sizes, default=0)

    if max_blob_bytes >= _INGEST_EXTREME_BLOB_LIMIT_BYTES or total_blob_bytes >= _INGEST_EXTREME_BLOB_LIMIT_BYTES:
        return 1
    if max_blob_bytes >= _INGEST_HIGH_BLOB_LIMIT_BYTES or total_blob_bytes >= _INGEST_HIGH_BLOB_LIMIT_BYTES:
        return min(base_worker_count, 2)
    if total_blob_bytes >= _INGEST_SOFT_BLOB_LIMIT_BYTES:
        return min(base_worker_count, 4)
    return base_worker_count


def _new_ingest_batch_summary(
    raw_artifacts: list[RawConversationRecord],
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
    pending_by_parent: dict[str, list[_ConversationEntry]],
    force_write: bool = False,
) -> None:
    _record_outcome(summary, ir)
    _observe_current_rss(summary)

    if ir.error:
        _record_failed_ingest_result(summary, ir)
        return

    if not ir.conversations:
        summary.skipped_raw_ids.add(ir.raw_id)
        return

    for cdata in ir.conversations:
        drain_started = time.perf_counter()
        _drain_ready_conversation_entries(
            conn,
            [(ir.raw_id, cdata)],
            summary=summary,
            materialized_ids=materialized_ids,
            pending_by_parent=pending_by_parent,
            force_write=force_write,
        )
        summary.drain_elapsed_s += time.perf_counter() - drain_started
        _observe_current_rss(summary)


def _consume_ingest_results(
    conn: sqlite3.Connection,
    raw_artifacts: list[RawConversationRecord],
    *,
    worker_request: _IngestWorkerRequest,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    pending_by_parent: dict[str, list[_ConversationEntry]],
    force_write: bool = False,
) -> None:
    result_iterator = iter(
        _iter_ingest_results_sync(
            raw_artifacts,
            request=worker_request,
            worker_count=summary.worker_count,
        )
    )
    while True:
        wait_started = time.perf_counter()
        try:
            ir = next(result_iterator)
        except StopIteration:
            summary.teardown_elapsed_s = time.perf_counter() - wait_started
            break
        summary.result_wait_s += time.perf_counter() - wait_started
        _drain_ingest_result(
            conn,
            ir,
            summary=summary,
            materialized_ids=materialized_ids,
            pending_by_parent=pending_by_parent,
            force_write=force_write,
        )


def _commit_ingest_results(
    conn: sqlite3.Connection,
    *,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    pending_by_parent: dict[str, list[_ConversationEntry]],
) -> None:
    flush_started = time.perf_counter()
    _flush_pending_conversation_entries(
        conn,
        pending_by_parent,
        summary=summary,
        materialized_ids=materialized_ids,
    )
    summary.flush_elapsed_s = time.perf_counter() - flush_started
    _observe_current_rss(summary)
    commit_started = time.perf_counter()
    conn.commit()
    summary.commit_elapsed_s = time.perf_counter() - commit_started


def _commit_sync_ingest_side_effects(conn: sqlite3.Connection, changed_conversation_ids: Sequence[str]) -> None:
    """Run post-ingest side effects through the canonical write-effects path."""
    commit_archive_write_effects(
        conn,
        WriteOperation.INGEST,
        {"changed_conversation_ids": tuple(changed_conversation_ids)},
    )


def _process_ingest_batch_sync(
    raw_artifacts: list[RawConversationRecord],
    *,
    db_path: Path,
    archive_root_str: str,
    blob_root_str: str,
    validation_mode: str,
    ingest_workers: int | None,
    measure_ingest_result_size: bool,
    force_write: bool = False,
) -> _IngestBatchSummary:
    from polylogue.storage.fts.fts_lifecycle import (
        suspend_fts_triggers_sync,
    )

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
    pending_by_parent: dict[str, list[_ConversationEntry]] = {}
    _observe_current_rss(summary)

    changed_ids: set[str] = set()
    try:
        suspend_fts_triggers_sync(conn)
        conn.execute("BEGIN IMMEDIATE")
        _consume_ingest_results(
            conn,
            raw_artifacts,
            worker_request=worker_request,
            summary=summary,
            materialized_ids=materialized_ids,
            pending_by_parent=pending_by_parent,
            force_write=force_write,
        )
        _commit_ingest_results(
            conn,
            summary=summary,
            materialized_ids=materialized_ids,
            pending_by_parent=pending_by_parent,
        )
        changed_ids = set(summary.changed_conversation_ids)
    except Exception:
        conn.rollback()
        raise
    finally:
        # Post-commit side effects (FTS repair + cache invalidation) run after
        # the main transaction commits because commit_archive_write_effects()
        # does its own conn.commit() internally (restores FTS triggers + rebuilds
        # FTS indexes in a separate transaction). Nesting that inside the BEGIN
        # IMMEDIATE transaction would cause nested-commit errors.
        #
        # This ordering also means a transient FTS repair failure won't roll
        # back the already-durable data writes — and a data-write rollback in
        # the except branch above leaves changed_ids empty, so side effects are
        # a no-op.
        try:
            _commit_sync_ingest_side_effects(conn, tuple(changed_ids))
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    summary.elapsed_s = time.perf_counter() - t_start
    return summary


def _unattributed_batch_elapsed_s(
    *,
    elapsed_s: float,
    batch_summary: _IngestBatchSummary,
    raw_state_update_elapsed_s: float,
) -> float:
    accounted_elapsed_s = (
        batch_summary.setup_elapsed_s
        + batch_summary.result_wait_s
        + batch_summary.drain_elapsed_s
        + batch_summary.flush_elapsed_s
        + batch_summary.commit_elapsed_s
        + batch_summary.teardown_elapsed_s
        + raw_state_update_elapsed_s
    )
    return max(elapsed_s - accounted_elapsed_s, 0.0)


def _build_batch_memory_observation(
    *,
    rss_start_mb: float | None,
    rss_end_mb: float | None,
    peak_rss_self_start_mb: float | None,
    peak_rss_self_end_mb: float | None,
    peak_rss_children_mb: float | None,
    max_current_rss_mb: float | None,
) -> ParseBatchObservation:
    observation: ParseBatchObservation = {}
    if rss_start_mb is not None:
        observation["rss_start_mb"] = rss_start_mb
    if rss_end_mb is not None:
        observation["rss_end_mb"] = rss_end_mb
    if rss_start_mb is not None and rss_end_mb is not None:
        observation["rss_delta_mb"] = round(rss_end_mb - rss_start_mb, 1)
    if peak_rss_self_end_mb is not None:
        observation["process_peak_rss_self_mb"] = peak_rss_self_end_mb
    if peak_rss_self_start_mb is not None and peak_rss_self_end_mb is not None:
        observation["peak_rss_growth_mb"] = round(max(peak_rss_self_end_mb - peak_rss_self_start_mb, 0.0), 1)
    if peak_rss_children_mb is not None:
        observation["peak_rss_children_mb"] = peak_rss_children_mb
    if max_current_rss_mb is not None:
        observation["max_current_rss_mb"] = max_current_rss_mb
    return observation


def _apply_ingest_batch_summary(result: ParseResult, batch_summary: _IngestBatchSummary) -> None:
    result.parse_failures += batch_summary.parse_failures
    result.processed_ids.update(batch_summary.processed_ids)
    result._changed_conversation_ids.extend(batch_summary.changed_conversation_ids)
    for key, value in batch_summary.counts.items():
        if key in result.counts:
            result.counts[key] += value
    for key, value in batch_summary.changed_counts.items():
        if key in result.changed_counts:
            result.changed_counts[key] += value


def _progressed_raw_count(batch_summary: _IngestBatchSummary) -> int:
    return sum(1 for outcome in batch_summary.outcomes.values() if outcome.had_conversations and outcome.error is None)


def _successful_raw_ids(batch_summary: _IngestBatchSummary) -> set[str]:
    return {
        raw_id
        for raw_id, outcome in batch_summary.outcomes.items()
        if outcome.had_conversations and raw_id not in batch_summary.failed_raw_ids
    }


def _build_parse_batch_observation(
    *,
    batch_summary: _IngestBatchSummary,
    elapsed_s: float,
    raw_state_update_elapsed_s: float,
    rss_start_mb: float | None,
    rss_end_mb: float | None,
    peak_rss_self_start_mb: float | None,
    peak_rss_self_end_mb: float | None,
    peak_rss_children_mb: float | None,
) -> ParseBatchObservation:
    observation: ParseBatchObservation = {
        "records": batch_summary.raw_record_count,
        "blob_mb": round(batch_summary.total_blob_mb, 1),
        "result_mb": round(batch_summary.total_result_bytes / (1024 * 1024), 3),
        "max_result_mb": round(batch_summary.max_result_bytes / (1024 * 1024), 3),
        "conversations": batch_summary.total_convos,
        "messages": batch_summary.total_msgs,
        "changed_conversations": len(batch_summary.changed_conversation_ids),
        "workers": batch_summary.worker_count,
        "failed_raw_count": len(batch_summary.failed_raw_ids),
        "skipped_raw_count": len(batch_summary.skipped_raw_ids),
        "elapsed_ms": round(elapsed_s * 1000, 1),
        "sync_ingest_elapsed_ms": round(batch_summary.elapsed_s * 1000, 1),
        "sync_setup_elapsed_ms": round(batch_summary.setup_elapsed_s * 1000, 1),
        "result_wait_elapsed_ms": round(batch_summary.result_wait_s * 1000, 1),
        "drain_elapsed_ms": round(batch_summary.drain_elapsed_s * 1000, 1),
        "write_elapsed_ms": round(batch_summary.write_elapsed_s * 1000, 1),
        "max_write_elapsed_ms": round(batch_summary.max_write_elapsed_s * 1000, 1),
        "flush_elapsed_ms": round(batch_summary.flush_elapsed_s * 1000, 1),
        "commit_elapsed_ms": round(batch_summary.commit_elapsed_s * 1000, 1),
        "executor_teardown_elapsed_ms": round(batch_summary.teardown_elapsed_s * 1000, 1),
        "raw_state_update_elapsed_ms": round(raw_state_update_elapsed_s * 1000, 1),
    }
    residual_elapsed_s = _unattributed_batch_elapsed_s(
        elapsed_s=elapsed_s,
        batch_summary=batch_summary,
        raw_state_update_elapsed_s=raw_state_update_elapsed_s,
    )
    observation["unattributed_elapsed_ms"] = round(residual_elapsed_s * 1000, 1)
    observation.update(
        _build_batch_memory_observation(
            rss_start_mb=rss_start_mb,
            rss_end_mb=rss_end_mb,
            peak_rss_self_start_mb=peak_rss_self_start_mb,
            peak_rss_self_end_mb=peak_rss_self_end_mb,
            peak_rss_children_mb=peak_rss_children_mb,
            max_current_rss_mb=batch_summary.max_current_rss_mb,
        )
    )
    if batch_summary.max_result_raw_id is not None:
        observation["max_result_raw_id"] = batch_summary.max_result_raw_id
    return observation


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
) -> ParseBatchObservation | None:
    """Process a batch of raw records through the unified ingest pipeline.

    1. Submit all records to ProcessPool (decode + validate + parse + transform)
    2. Consume results via as_completed — write to DB as each worker finishes
    3. Defer session insight refresh to caller (done once after all batches)
    """
    import asyncio

    raw_artifacts = await service.repository.get_raw_conversations_batch(batch_ids)
    if not raw_artifacts:
        return None

    archive_root_str = str(service.archive_root)
    blob_root_str = str(blob_store_root())
    batch_started = time.perf_counter()
    rss_start_mb = read_current_rss_mb()
    peak_rss_self_start_mb = read_peak_rss_self_mb()

    # Get validation mode from environment
    validation_mode = os.environ.get("POLYLOGUE_SCHEMA_VALIDATION", "strict")

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
    )

    _apply_ingest_batch_summary(result, batch_summary)
    progressed_raw_count = _progressed_raw_count(batch_summary)
    if progress_callback and progressed_raw_count:
        progress_callback(progressed_raw_count)

    if batch_summary.elapsed_s > 2.0:
        logger.info(
            "ingest_batch",
            elapsed_s=round(batch_summary.elapsed_s, 2),
            records=batch_summary.raw_record_count,
            blob_mb=round(batch_summary.total_blob_mb, 1),
            conversations=batch_summary.total_convos,
            messages=batch_summary.total_msgs,
            workers=batch_summary.worker_count,
            changed=len(batch_summary.changed_conversation_ids),
        )

    raw_state_update_elapsed_s = await _persist_batch_raw_state_updates(
        service,
        backend,
        outcomes=batch_summary.outcomes,
        succeeded_raw_ids=_successful_raw_ids(batch_summary),
        skipped_raw_ids=batch_summary.skipped_raw_ids,
        failed_raw_ids=batch_summary.failed_raw_ids,
        validation_mode=validation_mode,
    )

    elapsed_s = time.perf_counter() - batch_started
    rss_end_mb = read_current_rss_mb()
    peak_rss_self_end_mb = read_peak_rss_self_mb()
    peak_rss_children_mb = read_peak_rss_children_mb()
    return _build_parse_batch_observation(
        batch_summary=batch_summary,
        elapsed_s=elapsed_s,
        raw_state_update_elapsed_s=raw_state_update_elapsed_s,
        rss_start_mb=rss_start_mb,
        rss_end_mb=rss_end_mb,
        peak_rss_self_start_mb=peak_rss_self_start_mb,
        peak_rss_self_end_mb=peak_rss_self_end_mb,
        peak_rss_children_mb=peak_rss_children_mb,
    )


def _successful_raw_state_update(
    *,
    outcome: _RawIngestOutcome | None,
    parsed_at: str,
    validation_mode: str,
) -> RawConversationStateUpdate:
    if outcome is None:
        return RawConversationStateUpdate(
            parsed_at=parsed_at,
            parse_error=None,
        )
    return RawConversationStateUpdate(
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
) -> RawConversationStateUpdate:
    if outcome is None:
        return RawConversationStateUpdate(
            parse_error=error,
        )
    return RawConversationStateUpdate(
        parse_error=outcome.parse_error,
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
    changed_conversation_ids: list[str],
) -> MaterializeStageObservation | None:
    """Bulk session insight refresh — once after all batches, not per-batch."""
    if not changed_conversation_ids:
        return None

    t_start = time.perf_counter()
    update_elapsed = 0.0
    thread_elapsed = 0.0
    aggregate_elapsed = 0.0
    try:
        from polylogue.storage.insights.session.refresh import (
            _apply_session_insight_conversation_updates_async,
            _refresh_thread_roots_async,
            refresh_async_provider_day_aggregates,
        )

        async with backend.connection() as conn:
            t_updates = time.perf_counter()
            update = await _apply_session_insight_conversation_updates_async(
                conn,
                changed_conversation_ids,
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
            "conversations": len(changed_conversation_ids),
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
                conversations=len(changed_conversation_ids),
                unique_thread_roots=len(thread_root_ids),
                unique_provider_days=len(affected_groups),
                elapsed_s=round(elapsed, 2),
                update_s=round(update_elapsed, 2),
                thread_refresh_s=round(thread_elapsed, 2),
                aggregate_refresh_s=round(aggregate_elapsed, 2),
                rate=round(len(changed_conversation_ids) / elapsed, 1) if elapsed > 0 else 0,
            )
        return observation
    except Exception as exc:
        logger.warning("Session insight refresh failed (non-fatal): %s", exc, exc_info=True)
        return {
            "conversations": len(changed_conversation_ids),
            "failed": True,
            "error": str(exc),
        }


__all__ = ["process_ingest_batch", "refresh_session_insights_bulk"]

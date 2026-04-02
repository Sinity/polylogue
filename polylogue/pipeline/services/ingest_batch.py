"""Batch ingest orchestration: ProcessPool workers + sync sqlite3 writes.

Producer-consumer architecture:
- CPU-bound work (decode/validate/parse/transform) in ProcessPoolExecutor
- DB writes in main thread via sync sqlite3 (no aiosqlite async overhead)
- as_completed yields results as workers finish — parse and write overlap

Replaces: parsing_batch.py, parsing_workflow.py, validation_flow.py.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.pipeline.services.ingest_worker import (
    ConversationData,
    IngestRecordResult,
    ingest_record,
)
from polylogue.storage.backends.connection import DB_TIMEOUT
from polylogue.storage.state_views import RawConversationStateUpdate

if TYPE_CHECKING:
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)


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
"""

_MESSAGE_UPSERT_SQL = """
INSERT INTO messages (
    message_id, conversation_id, provider_message_id, role, text,
    sort_key, content_hash, version, parent_message_id, branch_index,
    provider_name, word_count, has_tool_use, has_thinking
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    has_thinking = excluded.has_thinking
WHERE
    content_hash != excluded.content_hash
    OR IFNULL(role, '') != IFNULL(excluded.role, '')
    OR IFNULL(text, '') != IFNULL(excluded.text, '')
    OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
    OR branch_index != excluded.branch_index
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
    (conversation_id, provider_name, message_count, word_count, tool_use_count, thinking_count)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
    provider_name  = excluded.provider_name,
    message_count  = excluded.message_count,
    word_count     = excluded.word_count,
    tool_use_count = excluded.tool_use_count,
    thinking_count = excluded.thinking_count
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


def _open_sync_connection(db_path: Path) -> sqlite3.Connection:
    """Open a sync sqlite3 connection with the same pragmas as the async backend."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")
    conn.execute("PRAGMA cache_size = -524288")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA mmap_size = 1073741824")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA wal_autocheckpoint = 10000")
    return conn


def _check_content_unchanged(conn: sqlite3.Connection, cid: str, content_hash: str) -> bool:
    """Check if conversation content is unchanged (skip message writes)."""
    row = conn.execute(
        "SELECT content_hash FROM conversations WHERE conversation_id = ?",
        (cid,),
    ).fetchone()
    return row is not None and row[0] == content_hash


def _topo_sort_message_tuples(tuples: list[tuple]) -> list[tuple]:
    """Sort message tuples so parents come before children (FK constraint).

    message_id is at index 0, parent_message_id is at index 8.
    """
    ids_in_batch = {t[0] for t in tuples}
    no_parent: list[tuple] = []
    has_parent: list[tuple] = []
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
        next_remaining: list[tuple] = []
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
    return cdata.conversation_tuple[11]


def _topo_sort_conversation_entries(
    entries: list[tuple[str, ConversationData]],
) -> list[tuple[str, ConversationData]]:
    """Sort conversation entries so parents in the same batch precede children."""
    ids_in_batch = {entry[1].conversation_id for entry in entries}
    no_parent: list[tuple[str, ConversationData]] = []
    has_parent: list[tuple[str, ConversationData]] = []

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
        next_remaining: list[tuple[str, ConversationData]] = []
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
) -> tuple:
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
    updated = list(cdata.conversation_tuple)
    updated[11] = None
    return tuple(updated)


def _write_conversation(conn: sqlite3.Connection, cdata: ConversationData) -> tuple[bool, dict[str, int]]:
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

    # Always upsert conversation record (updates metadata even if content unchanged)
    conn.execute(_CONVERSATION_UPSERT_SQL, _resolved_conversation_tuple(conn, cdata))

    if content_unchanged:
        counts["skipped_conversations"] = 1
        counts["skipped_messages"] = len(cdata.message_tuples)
        counts["skipped_attachments"] = len(cdata.attachment_tuples)
        return False, counts

    counts["conversations"] = 1

    # Messages (topo-sorted for parent FK)
    if cdata.message_tuples:
        sorted_msgs = _topo_sort_message_tuples(cdata.message_tuples)
        conn.executemany(_MESSAGE_UPSERT_SQL, sorted_msgs)
        counts["messages"] = len(sorted_msgs)

    # Conversation stats
    if cdata.stats_tuple:
        conn.execute(_STATS_UPSERT_SQL, cdata.stats_tuple)

    # Content blocks
    if cdata.block_tuples:
        conn.executemany(_CONTENT_BLOCK_UPSERT_SQL, cdata.block_tuples)

    # Action events (replace all for this conversation)
    conn.execute("DELETE FROM action_events WHERE conversation_id = ?", (cdata.conversation_id,))
    if cdata.action_event_tuples:
        conn.executemany(_ACTION_EVENT_INSERT_SQL, cdata.action_event_tuples)

    # Attachments
    if cdata.attachment_tuples:
        # Prune old refs first
        new_aids = {t[0] for t in cdata.attachment_tuples}
        if new_aids:
            placeholders = ",".join("?" * len(new_aids))
            conn.execute(
                f"DELETE FROM attachment_refs WHERE conversation_id = ? AND attachment_id NOT IN ({placeholders})",
                (cdata.conversation_id, *new_aids),
            )
        conn.executemany(_ATTACHMENT_UPSERT_SQL, cdata.attachment_tuples)
        conn.executemany(_ATTACHMENT_REF_INSERT_SQL, cdata.attachment_ref_tuples)
        # Update ref counts
        aid_list = list(new_aids)
        placeholders = ", ".join("?" for _ in aid_list)
        conn.execute(
            f"""UPDATE attachments SET ref_count = (
                SELECT COUNT(*) FROM attachment_refs
                WHERE attachment_refs.attachment_id = attachments.attachment_id
            ) WHERE attachment_id IN ({placeholders})""",
            tuple(aid_list),
        )
        counts["attachments"] = len(cdata.attachment_tuples)

    return True, counts


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


async def process_ingest_batch(
    service: ParsingService,
    backend: SQLiteBackend,
    batch_ids: list[str],
    result: ParseResult,
    progress_callback: ProgressCallback | None,
) -> None:
    """Process a batch of raw records through the unified ingest pipeline.

    1. Submit all records to ProcessPool (decode + validate + parse + transform)
    2. Consume results via as_completed — write to DB as each worker finishes
    3. Defer session product refresh to caller (done once after all batches)
    """
    import asyncio

    raw_records = await service.repository.get_raw_conversations_batch(batch_ids)
    if not raw_records:
        return

    archive_root_str = str(service.archive_root)

    # Get validation mode from environment
    validation_mode = os.environ.get("POLYLOGUE_SCHEMA_VALIDATION", "strict")

    worker_count = min(len(raw_records), os.cpu_count() or 4, 8)
    t_start = time.perf_counter()

    # Phase 1: Submit all records to ProcessPool
    def _run_pool():
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures: dict[Future, str] = {}
                for raw_record in raw_records:
                    f = executor.submit(ingest_record, raw_record, archive_root_str, validation_mode)
                    futures[f] = raw_record.raw_id

                results_list: list[IngestRecordResult] = []
                for future in as_completed(futures):
                    try:
                        results_list.append(future.result())
                    except Exception as exc:
                        raw_id = futures[future]
                        results_list.append(IngestRecordResult(
                            raw_id=raw_id,
                            error=f"worker: {exc}",
                        ))
                return results_list
        except (TypeError, pickle.PicklingError):
            # Fallback for unpicklable records (e.g. MagicMock in tests)
            return [ingest_record(r, archive_root_str, validation_mode) for r in raw_records]

    ingest_results: list[IngestRecordResult] = await asyncio.to_thread(_run_pool)
    parse_elapsed = time.perf_counter() - t_start

    # Phase 2: Write results to DB via sync sqlite3
    # Open a direct sync connection — bypasses aiosqlite async overhead.
    db_path = backend.db_path
    conn = _open_sync_connection(db_path)

    from polylogue.storage.fts_lifecycle import suspend_fts_triggers_sync

    suspend_fts_triggers_sync(conn)
    conn.execute("BEGIN IMMEDIATE")

    changed_conversation_ids: list[str] = []
    succeeded_raw_ids: set[str] = set()
    failed_raw_ids: dict[str, str] = {}
    skipped_raw_ids: set[str] = set()
    payload_providers: dict[str, str | None] = {}
    total_msgs = 0
    total_convos = 0

    try:
        conversation_entries: list[tuple[str, ConversationData]] = []
        for ir in ingest_results:
            payload_providers[ir.raw_id] = ir.payload_provider

            if ir.error:
                logger.error("Failed to ingest raw record", raw_id=ir.raw_id, error=ir.error)
                result.parse_failures += 1
                failed_raw_ids[ir.raw_id] = ir.error[:500]
                continue

            if not ir.conversations:
                skipped_raw_ids.add(ir.raw_id)
                continue

            conversation_entries.extend((ir.raw_id, cdata) for cdata in ir.conversations)

        for raw_id, cdata in _topo_sort_conversation_entries(conversation_entries):
            try:
                t_write = time.perf_counter()
                content_changed, counts = _write_conversation(conn, cdata)
                write_elapsed = time.perf_counter() - t_write

                total_convos += 1
                total_msgs += len(cdata.message_tuples)

                ingest_changed = (
                    counts["conversations"]
                    + counts["messages"]
                    + counts["attachments"]
                ) > 0

                if ingest_changed or content_changed:
                    result.processed_ids.add(cdata.conversation_id)
                if content_changed:
                    result.changed_counts["conversations"] += 1
                    changed_conversation_ids.append(cdata.conversation_id)
                if counts["messages"]:
                    result.changed_counts["messages"] += counts["messages"]
                if counts["attachments"]:
                    result.changed_counts["attachments"] += counts["attachments"]
                for key, value in counts.items():
                    if key in result.counts:
                        result.counts[key] += value

                if write_elapsed >= 1.0:
                    logger.info(
                        "slow_write",
                        cid=cdata.conversation_id[:20],
                        elapsed_s=round(write_elapsed, 2),
                        msgs=len(cdata.message_tuples),
                    )

            except Exception as exc:
                logger.error("Error writing conversation: %s", exc)
                result.parse_failures += 1
                failed_raw_ids[raw_id] = str(exc)[:500]

        for ir in ingest_results:
            if ir.error or not ir.conversations:
                continue
            if ir.raw_id not in failed_raw_ids:
                succeeded_raw_ids.add(ir.raw_id)

            if progress_callback:
                progress_callback(1)

        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if parse_elapsed > 2.0:
        total_blob_mb = sum(r.blob_size for r in raw_records) / (1024 * 1024)
        logger.info(
            "ingest_batch",
            elapsed_s=round(parse_elapsed, 2),
            records=len(raw_records),
            blob_mb=round(total_blob_mb, 1),
            conversations=total_convos,
            messages=total_msgs,
            workers=worker_count,
            changed=len(changed_conversation_ids),
        )

    # Phase 3: Update raw record states + persist validation status
    # Build a map of validation results from the ingest worker output
    validation_statuses: dict[str, IngestRecordResult] = {ir.raw_id: ir for ir in ingest_results}

    now_iso = datetime.now(timezone.utc).isoformat()
    validation_mode_str = validation_mode

    for rid in succeeded_raw_ids:
        if rid not in failed_raw_ids:
            ir = validation_statuses.get(rid)
            await service.repository.update_raw_state(
                rid,
                state=RawConversationStateUpdate(
                    parsed_at=now_iso,
                    parse_error=None,
                    payload_provider=payload_providers.get(rid),
                ),
            )
            if ir:
                await service.repository.mark_raw_validated(
                    rid,
                    status=ir.validation_status,
                    error=ir.validation_error,
                    payload_provider=ir.payload_provider,
                    mode=validation_mode_str,
                )
    for rid in skipped_raw_ids:
        if rid not in failed_raw_ids and rid not in succeeded_raw_ids:
            ir = validation_statuses.get(rid)
            await service.repository.update_raw_state(
                rid,
                state=RawConversationStateUpdate(
                    parsed_at=now_iso,
                    parse_error=None,
                    payload_provider=payload_providers.get(rid),
                ),
            )
            if ir:
                await service.repository.mark_raw_validated(
                    rid,
                    status=ir.validation_status,
                    error=ir.validation_error,
                    payload_provider=ir.payload_provider,
                    mode=validation_mode_str,
                )
    for rid, error in failed_raw_ids.items():
        ir = validation_statuses.get(rid)
        await service.repository.update_raw_state(
            rid,
            state=RawConversationStateUpdate(
                parse_error=error,
                payload_provider=payload_providers.get(rid),
            ),
        )
        if ir:
            await service.repository.mark_raw_validated(
                rid,
                status=ir.validation_status,
                error=ir.validation_error or error,
                payload_provider=ir.payload_provider,
                mode=validation_mode_str,
            )

    # Store changed_conversation_ids for deferred session refresh
    result._changed_conversation_ids.extend(changed_conversation_ids)


async def refresh_session_products_bulk(
    backend: SQLiteBackend,
    changed_conversation_ids: list[str],
) -> None:
    """Bulk session product refresh — once after all batches, not per-batch."""
    if not changed_conversation_ids:
        return

    t_start = time.perf_counter()
    try:
        from polylogue.storage.session_product_refresh import (
            refresh_session_products_for_conversation_async,
        )

        async with backend.connection() as conn:
            for cid in changed_conversation_ids:
                await refresh_session_products_for_conversation_async(
                    conn,
                    cid,
                    transaction_depth=1,
                )
            await conn.commit()

        elapsed = time.perf_counter() - t_start
        if elapsed > 2.0:
            logger.info(
                "session_product_refresh",
                elapsed_s=round(elapsed, 2),
                conversations=len(changed_conversation_ids),
                rate=round(len(changed_conversation_ids) / elapsed, 1) if elapsed > 0 else 0,
            )
    except Exception as exc:
        logger.warning("Session product refresh failed (non-fatal): %s", exc)


__all__ = ["process_ingest_batch", "refresh_session_products_bulk"]

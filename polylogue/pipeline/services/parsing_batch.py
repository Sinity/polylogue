"""Batch processing of raw conversation records through the parse pipeline.

Performance-critical: uses bulk_connection() for all writes to eliminate
per-operation connection overhead, processes conversations sequentially
(SQLite is single-writer), and defers session product refresh to a
single post-batch pass.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import orjson

from polylogue.logging import get_logger
from polylogue.pipeline.ids import conversation_id as make_conversation_id
from polylogue.pipeline.prepare import PrepareCache
from polylogue.storage.state_views import RawConversationStateUpdate

if TYPE_CHECKING:
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.protocols import ProgressCallback
    from polylogue.sources.parsers.base import ParsedConversation
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)


async def process_raw_batch(
    service: ParsingService,
    backend: SQLiteBackend,
    batch_ids: list[str],
    result: ParseResult,
    progress_callback: ProgressCallback | None,
) -> None:
    """Process a batch of raw conversation IDs.

    Architecture:
    1. Load raw records from DB (batched read)
    2. Parse each record (CPU-bound, fast — ~36K msgs/s)
    3. Write all conversations under bulk_connection() (sequential,
       single transaction — eliminates per-connection overhead)
    4. Refresh session products for changed conversations (batched)
    """
    import asyncio as _asyncio
    import os as _os
    from concurrent.futures import ProcessPoolExecutor

    from polylogue.pipeline.services.parsing_parallel import ingest_record_sync

    raw_records = await service.repository.get_raw_conversations_batch(batch_ids)

    # Phase 1: Decode + validate + parse in parallel subprocesses.
    # Combines what were separate validation and parse stages — the blob
    # is decoded ONCE, then validated and parsed in the same process.
    work_items: list[tuple[ParsedConversation, str, str]] = []
    failed_raw_ids: dict[str, str] = {}
    payload_providers: dict[str, str | None] = {}
    skipped_raw_ids: set[str] = set()

    t_parse_batch = time.perf_counter()
    worker_count = min(len(raw_records), _os.cpu_count() or 4, 8)
    archive_root_str = str(service.archive_root)

    def _run_ingest_batch():
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(
                ingest_record_sync,
                raw_records,
                [archive_root_str] * len(raw_records),
                chunksize=max(1, len(raw_records) // worker_count),
            ))

    parse_results = await _asyncio.to_thread(_run_ingest_batch)
    parse_elapsed = time.perf_counter() - t_parse_batch

    total_msgs = 0
    for pr in parse_results:
        payload_providers[pr.raw_id] = pr.payload_provider
        # Update validation state in DB (inline validation)
        if pr.validation_status:
            await service.repository.mark_raw_validated(
                pr.raw_id,
                status=pr.validation_status,
                error=pr.validation_error,
                provider=pr.payload_provider,
                mode="advisory",
                payload_provider=pr.payload_provider,
            )
        if pr.error:
            logger.error("Failed to parse raw conversation", raw_id=pr.raw_id, error=pr.error)
            result.parse_failures += 1
            failed_raw_ids[pr.raw_id] = pr.error[:500]
        elif not pr.conversations:
            skipped_raw_ids.add(pr.raw_id)
        else:
            for convo in pr.conversations:
                total_msgs += len(convo.messages)
                work_items.append((convo, pr.source_name or "", pr.raw_id))

    if parse_elapsed > 2.0:
        total_blob_mb = sum(r.blob_size for r in raw_records) / (1024 * 1024)
        logger.info(
            "parse_batch",
            elapsed_s=round(parse_elapsed, 2),
            records=len(raw_records),
            blob_mb=round(total_blob_mb, 1),
            conversations=len(work_items),
            messages=total_msgs,
            workers=worker_count,
        )

    del raw_records, parse_results

    # Drive attachment download is deferred — it requires network access
    # and can block for minutes on slow/throttled connections. Attachments
    # are best-effort metadata; conversations are saved without them.
    # A separate `polylogue run --stage=attachments` can fill them in later.

    if not work_items:
        for rid, error in failed_raw_ids.items():
            await service.repository.update_raw_state(
                rid,
                state=RawConversationStateUpdate(
                    parse_error=error,
                    payload_provider=payload_providers.get(rid),
                ),
            )
        return

    # Phase 2: Transform + write using SYNC SQLite (no aiosqlite overhead).
    # Each async await via aiosqlite costs ~1.2ms thread-crossing. With ~10
    # calls per conversation × thousands of conversations, this adds up to
    # minutes. Direct sqlite3 calls cost ~0.01ms each.
    import sqlite3 as _sqlite3

    from polylogue.pipeline.prepare_transform import transform_to_records
    from polylogue.storage.fts_lifecycle import suspend_fts_triggers_sync
    from polylogue.storage.repository_write_batch_sync import save_conversation_sync

    # Open a sync connection for direct writes
    db_path = backend._db_path
    sync_conn = _sqlite3.connect(db_path, timeout=30)
    sync_conn.row_factory = _sqlite3.Row
    sync_conn.execute("PRAGMA journal_mode=WAL")
    sync_conn.execute("PRAGMA synchronous=NORMAL")
    sync_conn.execute("PRAGMA cache_size=-524288")
    sync_conn.execute("PRAGMA mmap_size=1073741824")
    sync_conn.execute("PRAGMA temp_store=MEMORY")
    sync_conn.execute("PRAGMA wal_autocheckpoint=10000")

    # Suspend FTS triggers on the sync connection
    suspend_fts_triggers_sync(sync_conn)
    sync_conn.execute("BEGIN IMMEDIATE")

    succeeded_raw_ids: set[str] = set()
    changed_conversation_ids: list[str] = []
    flush_count = 0

    for convo_item, source_name_item, raw_id in work_items:
        t_item = time.perf_counter()
        msg_count = len(convo_item.messages)
        try:
            transform = transform_to_records(
                convo_item, source_name_item, archive_root=service.archive_root,
            )
            transform.bundle.conversation.raw_id = raw_id

            counts = save_conversation_sync(
                sync_conn,
                transform.bundle.conversation,
                transform.bundle.messages,
                transform.bundle.attachments,
                transform.bundle.content_blocks,
            )
            content_changed = counts["conversations"] > 0
            await result.merge_result(
                str(transform.bundle.conversation.conversation_id),
                counts,
                content_changed,
            )
            succeeded_raw_ids.add(raw_id)
            if content_changed:
                changed_conversation_ids.append(str(transform.bundle.conversation.conversation_id))
        except Exception as exc:
            logger.error("Error processing conversation: %s", exc)
            result.parse_failures += 1
            failed_raw_ids[raw_id] = str(exc)[:500]
        finally:
            flush_count += 1
            if flush_count >= 100:
                sync_conn.commit()
                sync_conn.execute("BEGIN IMMEDIATE")
                flush_count = 0
            item_elapsed = time.perf_counter() - t_item
            if item_elapsed >= 1.0:
                logger.info(
                    "slow_item",
                    raw_id=raw_id[:16],
                    elapsed_s=round(item_elapsed, 2),
                    msgs=msg_count,
                    provider=str(convo_item.provider_name),
                )
            if progress_callback:
                progress_callback(1)

    sync_conn.commit()
    sync_conn.close()
    del work_items

    # Phase 3: Refresh session products for changed conversations (post-batch)
    # Uses the incremental per-conversation refresh, not the full rebuild.
    # The full rebuild does DELETE + re-INSERT for ALL threads/tags/day-summaries
    # which is O(total_conversations). Incremental refresh only touches the
    # affected conversations.
    if changed_conversation_ids:
        t_refresh = time.perf_counter()
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
            refresh_elapsed = time.perf_counter() - t_refresh
            if refresh_elapsed > 2.0:
                logger.info(
                    "slow_refresh",
                    elapsed_s=round(refresh_elapsed, 2),
                    conversations=len(changed_conversation_ids),
                    rate=round(len(changed_conversation_ids) / refresh_elapsed, 1) if refresh_elapsed > 0 else 0,
                )
        except Exception as exc:
            logger.warning(
                "Session product refresh failed (non-fatal): %s", exc,
            )

    # Phase 4: Update raw record states
    for rid in succeeded_raw_ids:
        if rid not in failed_raw_ids:
            await service.repository.update_raw_state(
                rid,
                state=RawConversationStateUpdate(
                    parsed_at=datetime.now(timezone.utc).isoformat(),
                    parse_error=None,
                    payload_provider=payload_providers.get(rid),
                ),
            )
    for rid in skipped_raw_ids:
        if rid not in failed_raw_ids and rid not in succeeded_raw_ids:
            await service.repository.update_raw_state(
                rid,
                state=RawConversationStateUpdate(
                    parsed_at=datetime.now(timezone.utc).isoformat(),
                    parse_error=None,
                    payload_provider=payload_providers.get(rid),
                ),
            )
    for rid, error in failed_raw_ids.items():
        await service.repository.update_raw_state(
            rid,
            state=RawConversationStateUpdate(
                parse_error=error,
                payload_provider=payload_providers.get(rid),
            ),
        )


__all__ = ["process_raw_batch"]

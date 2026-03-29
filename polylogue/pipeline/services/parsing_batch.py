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
from polylogue.pipeline.prepare import PrepareCache, prepare_records
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
    raw_records = await service.repository.get_raw_conversations_batch(batch_ids)

    # Phase 1: Parse all records (fast, CPU-bound)
    work_items: list[tuple[ParsedConversation, str, str]] = []
    failed_raw_ids: dict[str, str] = {}
    payload_providers: dict[str, str | None] = {}
    skipped_raw_ids: set[str] = set()

    for raw_record in raw_records:
        try:
            parsed_convos = await service._parse_raw_record(raw_record)
            payload_providers[raw_record.raw_id] = raw_record.payload_provider
            if not parsed_convos:
                skipped_raw_ids.add(raw_record.raw_id)
                continue
            source_name = raw_record.source_name or raw_record.source_path
            for convo in parsed_convos:
                work_items.append((convo, source_name, raw_record.raw_id))
        except (json.JSONDecodeError, orjson.JSONDecodeError, ValueError, TypeError) as exc:
            logger.error(
                "Failed to parse raw conversation",
                raw_id=raw_record.raw_id,
                provider=raw_record.provider_name,
                error=str(exc),
            )
            result.parse_failures += 1
            failed_raw_ids[raw_record.raw_id] = str(exc)[:500]
            payload_providers[raw_record.raw_id] = raw_record.payload_provider

    del raw_records

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

    candidate_cids: set[str] = set()
    for convo, _, _ in work_items:
        cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)
        candidate_cids.add(cid)
        if convo.parent_conversation_provider_id:
            parent_cid = make_conversation_id(
                convo.provider_name,
                convo.parent_conversation_provider_id,
            )
            candidate_cids.add(parent_cid)

    cache = await PrepareCache.load(backend, candidate_cids)

    # Phase 2: Write all conversations under bulk_connection()
    # Sequential writes: SQLite is single-writer, concurrent workers
    # just add lock contention and async thread-crossing overhead.
    succeeded_raw_ids: set[str] = set()
    changed_conversation_ids: list[str] = []

    _SLOW_THRESHOLD_S = 2.0

    async with backend.bulk_connection():
        for convo_item, source_name_item, raw_id in work_items:
            t_item = time.perf_counter()
            try:
                persisted = await prepare_records(
                    convo_item,
                    source_name_item,
                    archive_root=service.archive_root,
                    backend=backend,
                    repository=service.repository,
                    raw_id=raw_id,
                    cache=cache,
                )
                await result.merge_result(
                    persisted.conversation_id,
                    persisted.counts,
                    persisted.content_changed,
                )
                succeeded_raw_ids.add(raw_id)
                if persisted.content_changed:
                    changed_conversation_ids.append(persisted.conversation_id)
            except Exception as exc:
                logger.error("Error processing conversation: %s", exc)
                result.parse_failures += 1
                failed_raw_ids[raw_id] = str(exc)[:500]
            finally:
                item_elapsed = time.perf_counter() - t_item
                if item_elapsed >= _SLOW_THRESHOLD_S:
                    logger.info(
                        "slow_parse_item",
                        raw_id=raw_id,
                        elapsed_s=round(item_elapsed, 2),
                    )
                if progress_callback:
                    progress_callback(1)

    del work_items

    # Phase 3: Refresh session products for changed conversations (post-batch)
    # Uses the incremental per-conversation refresh, not the full rebuild.
    # The full rebuild does DELETE + re-INSERT for ALL threads/tags/day-summaries
    # which is O(total_conversations). Incremental refresh only touches the
    # affected conversations.
    if changed_conversation_ids:
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

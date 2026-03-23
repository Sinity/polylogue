from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

import orjson

from polylogue.logging import get_logger
from polylogue.pipeline.ids import conversation_id as make_conversation_id
from polylogue.pipeline.prepare import PrepareCache, prepare_records

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
    """Process a batch of raw conversation IDs."""
    raw_records = await backend.get_raw_conversations_batch(batch_ids)

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
            await backend.mark_raw_parsed(
                rid,
                error=error,
                payload_provider=payload_providers.get(rid),
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

    worker_count = min(os.cpu_count() or 4, 16)
    queue: asyncio.Queue[tuple[ParsedConversation, str, str] | None] = asyncio.Queue(
        maxsize=worker_count * 2
    )
    succeeded_raw_ids: set[str] = set()
    tracking_lock = asyncio.Lock()

    async def _worker() -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            convo_item, source_name_item, raw_id = item
            try:
                convo_id, result_counts, content_changed = await prepare_records(
                    convo_item,
                    source_name_item,
                    archive_root=service.archive_root,
                    backend=backend,
                    repository=service.repository,
                    raw_id=raw_id,
                    cache=cache,
                )
                await result.merge_result(convo_id, result_counts, content_changed)
                async with tracking_lock:
                    succeeded_raw_ids.add(raw_id)
            except Exception as exc:
                logger.error("Error processing conversation: %s", exc)
                result.parse_failures += 1
                async with tracking_lock:
                    failed_raw_ids[raw_id] = str(exc)[:500]
            finally:
                if progress_callback:
                    progress_callback(1)
                queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]

    for item in work_items:
        await queue.put(item)
    del work_items

    await queue.join()
    for _ in range(worker_count):
        await queue.put(None)
    await asyncio.gather(*workers)

    for rid in succeeded_raw_ids:
        if rid not in failed_raw_ids:
            await backend.mark_raw_parsed(
                rid,
                payload_provider=payload_providers.get(rid),
            )
    for rid in skipped_raw_ids:
        if rid not in failed_raw_ids and rid not in succeeded_raw_ids:
            await backend.mark_raw_parsed(
                rid,
                payload_provider=payload_providers.get(rid),
            )
    for rid, error in failed_raw_ids.items():
        await backend.mark_raw_parsed(
            rid,
            error=error,
            payload_provider=payload_providers.get(rid),
        )
__all__ = ["process_raw_batch"]

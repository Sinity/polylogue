"""Streaming helpers for acquisition service source traversal."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.pipeline.services.acquisition_records import make_lightweight_record, make_raw_record
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import DriveConfig, Source

logger = get_logger(__name__)


async def iter_source_raw_stream(
    source: Source,
    *,
    known_mtimes: dict[str, str] | None = None,
) -> AsyncIterator[RawConversationData]:
    """Stream raw source payloads without materializing the full iterator."""
    from polylogue.pipeline.services import acquisition as acquisition_root

    iterator = acquisition_root.iter_source_raw_data(source, known_mtimes=known_mtimes)
    sentinel = object()
    batch_size = 128

    def _next_batch() -> list[RawConversationData]:
        batch: list[RawConversationData] = []
        for _ in range(batch_size):
            item = next(iterator, sentinel)
            if item is sentinel:
                break
            batch.append(item)
        return batch

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            batch = await loop.run_in_executor(executor, _next_batch)
            if not batch:
                break
            for item in batch:
                yield item


async def iter_drive_raw_stream(
    source: Source,
    *,
    known_mtimes: dict[str, str] | None = None,
    ui: object | None = None,
    cursor_state: dict[str, object] | None = None,
    drive_config: DriveConfig | None = None,
) -> AsyncIterator[RawConversationData]:
    """Stream Drive payloads as raw records without touching the local cache."""
    from polylogue.sources.drive import iter_drive_raw_data

    sentinel = object()
    batch_size = 32
    iterator = iter_drive_raw_data(
        source=source,
        ui=ui,
        cursor_state=cursor_state,
        drive_config=drive_config,
        known_mtimes=known_mtimes,
    )

    def _next_batch() -> list[RawConversationData]:
        batch: list[RawConversationData] = []
        for _ in range(batch_size):
            item = next(iterator, sentinel)
            if item is sentinel:
                break
            batch.append(item)
        return batch

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            batch = await loop.run_in_executor(executor, _next_batch)
            if not batch:
                break
            for item in batch:
                yield item


async def iter_raw_record_stream(
    source: Source,
    *,
    known_mtimes: dict[str, str] | None = None,
    ui: object | None = None,
    cursor_state: dict[str, object] | None = None,
    drive_config: DriveConfig | None = None,
    lightweight: bool = False,
    lightweight_full_limit: int = 50,
) -> AsyncIterator[RawConversationRecord]:
    """Yield prepared RawConversationRecord values for a source.

    Args:
        lightweight: If True, strip raw_content bytes from records after
            hashing, except for the first ``lightweight_full_limit`` records
            (which keep bytes for preview validation). Prevents OOM on
            large archives by not holding multi-MB payloads in memory.
        lightweight_full_limit: How many records keep full bytes in
            lightweight mode (for preview validation sample).
    """
    raw_stream: AsyncIterator[RawConversationData]
    if source.is_drive:
        raw_stream = iter_drive_raw_stream(
            source,
            known_mtimes=known_mtimes,
            ui=ui,
            cursor_state=cursor_state,
            drive_config=drive_config,
        )
    else:
        raw_stream = iter_source_raw_stream(
            source,
            known_mtimes=known_mtimes,
        )

    full_count = 0
    async for raw_data in raw_stream:
        try:
            keep = not lightweight or full_count < lightweight_full_limit
            record = make_raw_record(raw_data, source.name, keep_content=keep)
            if keep:
                full_count += 1
            # Explicitly break reference to raw bytes so GC can collect them
            # before the next iteration reads the next file.
            del raw_data
            yield record
        except ValueError as exc:
            logger.warning(
                "Skipping raw payload",
                source=source.name,
                path=raw_data.source_path,
                error=str(exc),
            )


__all__ = ["iter_drive_raw_stream", "iter_raw_record_stream", "iter_source_raw_stream"]

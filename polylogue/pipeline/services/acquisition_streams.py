"""Streaming helpers for acquisition service source traversal."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.pipeline.services.acquisition_records import make_raw_record
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.state_views import CursorStatePayload
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import Source
    from polylogue.sources.drive_types import DriveConfigLike, DriveUILike

logger = get_logger(__name__)
ObservationCallback = Callable[[dict[str, object]], None]


def _drain_batch(
    iterator: Iterator[RawConversationData],
    *,
    batch_size: int,
) -> list[RawConversationData]:
    """Read up to ``batch_size`` items without sentinel gymnastics."""
    return list(islice(iterator, batch_size))


async def iter_source_raw_stream(
    source: Source,
    *,
    known_mtimes: dict[str, str] | None = None,
    observation_callback: ObservationCallback | None = None,
    progress_callback: Callable[[int, str | None], None] | None = None,
) -> AsyncIterator[RawConversationData]:
    """Stream raw source payloads without materializing the full iterator."""
    from polylogue.pipeline.services import acquisition as acquisition_root

    loop = asyncio.get_running_loop()

    def _status_callback(desc: str) -> None:
        if progress_callback is None:
            return
        loop.call_soon_threadsafe(progress_callback, 0, desc)

    iterator = iter(
        acquisition_root.iter_source_raw_data(
            source,
            known_mtimes=known_mtimes,
            observation_callback=observation_callback,
            status_callback=_status_callback if progress_callback is not None else None,
        )
    )
    batch_size = 128

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            batch = await loop.run_in_executor(
                executor,
                partial(_drain_batch, iterator, batch_size=batch_size),
            )
            if not batch:
                break
            for item in batch:
                yield item


async def iter_drive_raw_stream(
    source: Source,
    *,
    known_mtimes: dict[str, str] | None = None,
    ui: DriveUILike | None = None,
    cursor_state: CursorStatePayload | None = None,
    drive_config: DriveConfigLike | None = None,
) -> AsyncIterator[RawConversationData]:
    """Stream Drive payloads as raw records without touching the local cache."""
    from polylogue.sources.drive import iter_drive_raw_data

    batch_size = 32
    iterator = iter(
        iter_drive_raw_data(
            source=source,
            ui=ui,
            cursor_state=cursor_state,
            drive_config=drive_config,
            known_mtimes=known_mtimes,
        )
    )

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            batch = await loop.run_in_executor(
                executor,
                partial(_drain_batch, iterator, batch_size=batch_size),
            )
            if not batch:
                break
            for item in batch:
                yield item


async def iter_raw_record_stream(
    source: Source,
    *,
    known_mtimes: dict[str, str] | None = None,
    ui: DriveUILike | None = None,
    cursor_state: CursorStatePayload | None = None,
    drive_config: DriveConfigLike | None = None,
    observation_callback: ObservationCallback | None = None,
    progress_callback: Callable[[int, str | None], None] | None = None,
) -> AsyncIterator[RawConversationRecord]:
    """Yield prepared RawConversationRecord values for a source."""
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
            observation_callback=observation_callback,
            progress_callback=progress_callback,
        )

    async for raw_data in raw_stream:
        if not raw_data.raw_bytes and not raw_data.blob_hash:
            continue
        raw_source_path = raw_data.source_path
        try:
            record = make_raw_record(raw_data, source.name)
            # Explicitly break reference to raw bytes so GC can collect them
            # before the next iteration reads the next file.
            del raw_data
            yield record
        except ValueError as exc:
            logger.warning(
                "Skipping raw payload",
                source=source.name,
                path=raw_source_path,
                error=str(exc),
            )


__all__ = ["iter_drive_raw_stream", "iter_raw_record_stream", "iter_source_raw_stream"]

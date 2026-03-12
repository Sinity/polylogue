"""Async acquisition service for pipeline operations.

The acquire stage reads source files and stores raw bytes to raw_conversations.
It does NOT parse - that's the parse stage's job.

Data flow:
    [Source Files] → ACQUIRE → [raw_conversations]
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from polylogue.lib.log import get_logger
from polylogue.lib.provider_identity import canonical_runtime_provider
from polylogue.protocols import ProgressCallback
from polylogue.sources.parsers.base import RawConversationData
from polylogue.sources.source import iter_source_raw_data
from polylogue.storage.store import MAX_RAW_CONTENT_SIZE, RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import DriveConfig, Source
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["AcquisitionService", "AcquireResult"]


class AcquireResult:
    """Result of an acquisition operation."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "acquired": 0,
            "skipped": 0,  # Already in database (by raw_id hash)
            "errors": 0,
        }
        self.raw_ids: list[str] = []  # List of acquired raw_ids


class ScanResult:
    """Result of scanning raw payloads from sources without persisting them."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "scanned": 0,
            "errors": 0,
        }
        self.cursors: dict[str, dict[str, object]] = {}


class AcquisitionService:
    """Service for acquiring raw conversation data from sources.

    This service implements the ACQUIRE stage of the pipeline:
    - Reads source files (JSON, JSONL, ZIP)
    - Computes content hash (raw_id)
    - Stores raw bytes in raw_conversations table
    - Does NOT parse or transform the data

    The stored raw data can then be processed by the parse stage.
    """

    def __init__(self, backend: SQLiteBackend):
        """Initialize the async acquisition service.

        Args:
            backend: Async SQLite backend for database operations
        """
        self.backend = backend

    async def _persist_record(
        self,
        record: RawConversationRecord,
        *,
        result: AcquireResult,
    ) -> None:
        """Persist one raw record and update acquisition counters."""
        try:
            inserted = await self.backend.save_raw_conversation(record)
            if inserted:
                result.counts["acquired"] += 1
                result.raw_ids.append(record.raw_id)
            else:
                result.counts["skipped"] += 1
        except Exception as exc:
            logger.error(
                "Failed to store raw conversation",
                source=record.source_name,
                path=record.source_path,
                error=str(exc),
            )
            result.counts["errors"] += 1

    async def visit_sources(
        self,
        sources: list[Source],
        *,
        progress_callback: ProgressCallback | None = None,
        ui: object | None = None,
        drive_config: DriveConfig | None = None,
        progress_label: str = "Scanning",
        on_record: Callable[[RawConversationRecord], Awaitable[None]] | None = None,
    ) -> ScanResult:
        """Visit source raw payloads incrementally without forcing list materialization."""
        result = ScanResult()
        known_mtimes = await self.backend.get_known_source_mtimes()

        async def _consume(record: RawConversationRecord) -> None:
            if on_record is not None:
                await on_record(record)
            result.counts["scanned"] += 1

        for source in sources:
            logger.debug("Scanning source", source=source.name)
            cursor_state: dict[str, object] = {}
            try:
                async for record in self._iter_raw_record_stream(
                    source,
                    known_mtimes=known_mtimes,
                    ui=ui,
                    cursor_state=cursor_state,
                    drive_config=drive_config,
                ):
                    await _consume(record)
                    if progress_callback:
                        progress_callback(1, desc=f"{progress_label} [{source.name}]")
            except Exception as exc:
                logger.error(
                    "Failed to scan source",
                    source=source.name,
                    error=str(exc),
                )
                result.counts["errors"] += 1
                cursor_state["error_count"] = cursor_state.get("error_count", 0) + 1
                cursor_state["latest_error"] = str(exc)

            if cursor_state:
                result.cursors[source.name] = cursor_state

        return result

    async def acquire_sources(
        self,
        sources: list[Source],
        *,
        ui: object | None = None,
        progress_callback: ProgressCallback | None = None,
        drive_config: DriveConfig | None = None,
    ) -> AcquireResult:
        """Acquire raw data from multiple sources.

        Reads source files and stores raw bytes in ``raw_conversations`` without
        materializing the full corpus in memory first.

        Args:
            sources: List of sources to acquire from
            progress_callback: Optional callback(count, desc=...) for progress

        Returns:
            AcquireResult with counts and list of acquired raw_ids
        """
        result = AcquireResult()
        flush_interval = 500
        items_since_flush = 0
        seen_raw_ids: set[str] = set()

        async def _store(record: RawConversationRecord) -> None:
            nonlocal items_since_flush
            if record.raw_id in seen_raw_ids:
                result.counts["skipped"] += 1
                return
            seen_raw_ids.add(record.raw_id)
            await self._persist_record(record, result=result)
            items_since_flush += 1
            if items_since_flush >= flush_interval:
                await self.backend.bulk_flush()
                items_since_flush = 0

        async with self.backend.bulk_connection():
            visit_result = await self.visit_sources(
                sources,
                progress_callback=progress_callback,
                ui=ui,
                drive_config=drive_config,
                progress_label="Acquiring",
                on_record=_store,
            )
            result.counts["errors"] += visit_result.counts["errors"]

        return result

    async def _iter_source_raw_stream(
        self,
        source: Source,
        *,
        known_mtimes: dict[str, str] | None = None,
    ) -> AsyncIterator[RawConversationData]:
        """Stream raw source payloads without materializing the full iterator.

        Args:
            source: Source to iterate
            known_mtimes: Optional {source_path: file_mtime} for skipping unchanged files

        Yields:
            RawConversationData entries
        """
        iterator = iter_source_raw_data(source, known_mtimes=known_mtimes)
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

    async def _iter_drive_raw_stream(
        self,
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

    async def _iter_raw_record_stream(
        self,
        source: Source,
        *,
        known_mtimes: dict[str, str] | None = None,
        ui: object | None = None,
        cursor_state: dict[str, object] | None = None,
        drive_config: DriveConfig | None = None,
    ) -> AsyncIterator[RawConversationRecord]:
        """Yield prepared RawConversationRecord values for a source."""
        raw_stream: AsyncIterator[RawConversationData]
        if source.is_drive:
            raw_stream = self._iter_drive_raw_stream(
                source,
                known_mtimes=known_mtimes,
                ui=ui,
                cursor_state=cursor_state,
                drive_config=drive_config,
            )
        else:
            raw_stream = self._iter_source_raw_stream(
                source,
                known_mtimes=known_mtimes,
            )

        async for raw_data in raw_stream:
            try:
                yield self._make_raw_record(raw_data, source.name)
            except ValueError as exc:
                logger.warning(
                    "Skipping raw payload",
                    source=source.name,
                    path=raw_data.source_path,
                    error=str(exc),
                )

    def _make_raw_record(self, raw_data: RawConversationData, source_name: str) -> RawConversationRecord:
        """Prepare a raw conversation record from scanned payload bytes.

        Args:
            raw_data: Raw conversation data to store
            source_name: Config source name (e.g., "inbox"), stored separately

        Returns:
            RawConversationRecord ready for persistence
        """
        size = len(raw_data.raw_bytes)
        if size > MAX_RAW_CONTENT_SIZE:
            raise ValueError(
                f"Oversized source file at {raw_data.source_path} "
                f"({size} bytes > {MAX_RAW_CONTENT_SIZE} max)"
            )

        raw_id = hashlib.sha256(raw_data.raw_bytes).hexdigest()
        acquired_at = datetime.now(timezone.utc).isoformat()

        return RawConversationRecord(
            raw_id=raw_id,
            provider_name=canonical_runtime_provider(
                raw_data.provider_hint or source_name,
                preserve_unknown=True,
                default=source_name,
            ),
            source_name=source_name,  # Config source name, distinct from provider
            source_path=raw_data.source_path,
            source_index=raw_data.source_index,
            raw_content=raw_data.raw_bytes,
            acquired_at=acquired_at,
            file_mtime=raw_data.file_mtime,
        )

"""Async acquisition service for pipeline operations.

The acquire stage reads source files and stores raw bytes to raw_conversations.
It does NOT parse - that's the parse stage's job.

Data flow:
    [Source Files] → ACQUIRE → [raw_conversations]
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from polylogue.lib.log import get_logger
from polylogue.sources.parsers.base import RawConversationData
from polylogue.sources.source import iter_source_conversations_with_raw
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import Source
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

    async def acquire_sources(
        self,
        sources: list[Source],
        *,
        progress_callback: Any | None = None,
    ) -> AcquireResult:
        """Acquire raw data from multiple sources.

        Reads source files and stores raw bytes in raw_conversations.
        Source iteration is performed in a thread pool to avoid blocking.

        Args:
            sources: List of sources to acquire from
            progress_callback: Optional callback(count, desc=...) for progress

        Returns:
            AcquireResult with counts and list of acquired raw_ids
        """
        result = AcquireResult()

        for source in sources:
            logger.debug("Acquiring from source", source=source.name)

            try:
                # Stream source items from a dedicated worker thread to avoid
                # event-loop blocking and full-list materialization.
                async for raw_data, _parsed in self._iter_source_conversations_stream(source):
                    if raw_data is None:
                        # This shouldn't happen with capture_raw=True, but handle it
                        logger.warning("No raw data captured for conversation")
                        result.counts["errors"] += 1
                        continue

                    try:
                        raw_id = await self._store_raw(raw_data, source.name)
                        if raw_id:
                            result.counts["acquired"] += 1
                            result.raw_ids.append(raw_id)
                        else:
                            result.counts["skipped"] += 1
                    except sqlite3.DatabaseError as exc:
                        logger.error(
                            "Failed to store raw conversation",
                            source=source.name,
                            path=raw_data.source_path,
                            error=str(exc),
                        )
                        result.counts["errors"] += 1

                    if progress_callback:
                        progress_callback(1, desc="Acquiring")

            except Exception as exc:
                logger.error(
                    "Failed to iterate source",
                    source=source.name,
                    error=str(exc),
                )
                result.counts["errors"] += 1

        logger.info(
            "Acquisition complete",
            acquired=result.counts["acquired"],
            skipped=result.counts["skipped"],
            errors=result.counts["errors"],
        )

        return result

    async def _iter_source_conversations_stream(
        self,
        source: Source,
    ) -> AsyncIterator[tuple[RawConversationData | None, Any]]:
        """Stream source conversations without materializing the full iterator.

        Args:
            source: Source to iterate

        Yields:
            Tuples of (raw_data, parsed)
        """
        iterator = iter_source_conversations_with_raw(source, capture_raw=True)
        sentinel = object()
        batch_size = 32

        def _next_batch() -> list[tuple[RawConversationData | None, Any]]:
            batch: list[tuple[RawConversationData | None, Any]] = []
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

    async def _store_raw(self, raw_data: RawConversationData, source_name: str) -> str | None:
        """Store raw conversation data.

        Args:
            raw_data: Raw conversation data to store
            source_name: Config source name (e.g., "inbox"), stored separately

        Returns:
            raw_id if newly stored, None if already exists
        """
        raw_id = hashlib.sha256(raw_data.raw_bytes).hexdigest()
        acquired_at = datetime.now(timezone.utc).isoformat()

        record = RawConversationRecord(
            raw_id=raw_id,
            provider_name=raw_data.provider_hint or source_name,
            source_name=source_name,  # Config source name, distinct from provider
            source_path=raw_data.source_path,
            source_index=raw_data.source_index,
            raw_content=raw_data.raw_bytes,
            acquired_at=acquired_at,
            file_mtime=raw_data.file_mtime,
        )

        if await self.backend.save_raw_conversation(record):
            return raw_id
        return None

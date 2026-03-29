"""Async acquisition service for pipeline operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.pipeline.services.acquisition_persistence import persist_raw_record
from polylogue.pipeline.services.acquisition_records import ScanResult
from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.protocols import ProgressCallback
from polylogue.sources.source_acquisition import iter_source_raw_data
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import DriveConfig, Source
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)

__all__ = ["AcquisitionService", "AcquireResult", "iter_source_raw_data"]


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
        from polylogue.storage.repository import ConversationRepository

        self.repository: ConversationRepository = ConversationRepository(backend=backend)

    async def _persist_record(
        self,
        record: RawConversationRecord,
        *,
        result: AcquireResult,
    ) -> None:
        await persist_raw_record(self.repository, record, result=result)

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
        known_mtimes = await self.repository.get_known_source_mtimes()

        async def _consume(record: RawConversationRecord) -> None:
            if on_record is not None:
                await on_record(record)
            result.counts["scanned"] += 1

        for source in sources:
            logger.debug("Scanning source", source=source.name)
            cursor_state: dict[str, object] = {}
            try:
                async for record in iter_raw_record_stream(
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
        import gc as _gc

        result = AcquireResult()
        flush_interval = 50
        items_since_flush = 0

        async def _store(record: RawConversationRecord) -> None:
            nonlocal items_since_flush
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
            result.errors += visit_result.counts["errors"]

        return result

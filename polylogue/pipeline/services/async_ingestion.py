"""Async ingestion service wrapping sync operations.

Provides async/await interface for ingestion while using the existing
sync ingestion logic in thread pools. This gives async benefits without
requiring complete rewrite of ID management and deduplication logic.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from polylogue.pipeline.services.ingestion import IngestionService, IngestResult

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.config import Config, Source
    from polylogue.storage.repository import StorageRepository

logger = structlog.get_logger(__name__)


class AsyncIngestionService:
    """Async wrapper around sync ingestion service.

    Runs sync ingestion operations in thread pools for concurrent execution.
    Simpler than full async rewrite while still providing performance benefits.

    Example:
        service = AsyncIngestionService(repository, archive_root, config)
        result = await service.ingest_sources(sources)
        print(f"Imported {result.counts['conversations']} conversations")
    """

    def __init__(
        self,
        repository: StorageRepository,
        archive_root: Path,
        config: Config,
    ) -> None:
        """Initialize async ingestion service.

        Args:
            repository: Storage repository for database operations
            archive_root: Root directory for archived conversations
            config: Application configuration
        """
        self._sync_service = IngestionService(repository, archive_root, config)

    async def ingest_sources(
        self,
        sources: list[Source],
        *,
        ui: object | None = None,
        download_assets: bool = True,
        progress_callback: object | None = None,
    ) -> IngestResult:
        """Ingest conversations from multiple sources asynchronously.

        Runs the sync ingestion in a thread pool to avoid blocking.

        Args:
            sources: List of sources to ingest
            ui: Optional UI object for user interaction
            download_assets: Whether to download attachments
            progress_callback: Optional progress callback function

        Returns:
            IngestResult with statistics about ingestion
        """
        log = logger.bind(source_count=len(sources))
        log.info("async_ingesting_sources")

        # Run sync ingestion in thread pool
        result = await asyncio.to_thread(
            self._sync_service.ingest_sources,
            sources,
            ui=ui,
            download_assets=download_assets,
            progress_callback=progress_callback,
        )

        log.info(
            "async_sources_ingested",
            conversations=result.counts["conversations"],
            messages=result.counts["messages"],
            skipped=result.counts["skipped_conversations"],
        )

        return result

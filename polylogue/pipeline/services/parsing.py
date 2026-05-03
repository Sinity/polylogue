"""Async parsing service for pipeline operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.errors import DatabaseError
from polylogue.pipeline.services.parsing_models import (
    IngestPhase,
    IngestResult,
    IngestState,
    ParseResult,
)
from polylogue.pipeline.services.parsing_workflow import ingest_sources, parse_from_raw

if TYPE_CHECKING:
    from polylogue.config import Config, Source
    from polylogue.protocols import ProgressCallback
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class ParsingService:
    """Service for parsing conversations from sources asynchronously."""

    DEFAULT_RAW_BATCH_SIZE = 50
    DEFAULT_RAW_BATCH_BLOB_LIMIT_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        repository: ConversationRepository,
        archive_root: Path,
        config: Config,
        *,
        raw_batch_size: int = DEFAULT_RAW_BATCH_SIZE,
        ingest_workers: int | None = None,
        measure_ingest_result_size: bool = False,
    ) -> None:
        if raw_batch_size <= 0:
            raise ValueError("raw_batch_size must be a positive integer")
        if ingest_workers is not None and ingest_workers <= 0:
            raise ValueError("ingest_workers must be a positive integer")
        self.repository = repository
        self.archive_root = archive_root
        self.config = config
        self._raw_batch_size = raw_batch_size
        self._ingest_workers = ingest_workers
        self._measure_ingest_result_size = measure_ingest_result_size

    def _require_backend(self) -> SQLiteBackend:
        """Return the repository backend or fail explicitly."""
        backend = self.repository.backend
        if backend is None:
            raise DatabaseError("repository backend is not initialized")
        return backend

    async def parse_sources(
        self,
        sources: list[Source],
        *,
        ui: object | None = None,
        download_assets: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> ParseResult:
        ingest_result = await self.ingest_sources(
            sources=sources,
            progress_callback=progress_callback,
            parse_records=True,
        )
        return ingest_result.parse_result

    async def ingest_sources(
        self,
        *,
        sources: list[Source],
        stage: str = "all",
        ui: object | None = None,
        progress_callback: ProgressCallback | None = None,
        parse_records: bool = True,
        skip_acquire: bool = False,
        force_write: bool = False,
    ) -> IngestResult:
        return await ingest_sources(
            self,
            sources=sources,
            stage=stage,
            ui=ui,
            progress_callback=progress_callback,
            parse_records=parse_records,
            skip_acquire=skip_acquire,
            force_write=force_write,
        )

    @property
    def raw_batch_size(self) -> int:
        return self._raw_batch_size

    @property
    def raw_batch_blob_limit_bytes(self) -> int:
        return self.DEFAULT_RAW_BATCH_BLOB_LIMIT_BYTES

    @property
    def ingest_workers(self) -> int | None:
        return self._ingest_workers

    @property
    def measure_ingest_result_size(self) -> bool:
        return self._measure_ingest_result_size

    async def parse_from_raw(
        self,
        *,
        raw_ids: list[str] | None = None,
        provider: str | None = None,
        progress_callback: ProgressCallback | None = None,
        force_write: bool = False,
    ) -> ParseResult:
        return await parse_from_raw(
            self,
            raw_ids=raw_ids,
            provider=provider,
            progress_callback=progress_callback,
            force_write=force_write,
        )


__all__ = [
    "IngestPhase",
    "IngestResult",
    "IngestState",
    "ParseResult",
    "ParsingService",
]

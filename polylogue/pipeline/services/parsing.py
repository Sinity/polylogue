"""Async parsing service for pipeline operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.core.errors import DatabaseError
from polylogue.pipeline.services.parsing_models import (
    IngestPhase,
    IngestResult,
    IngestState,
    ParseResult,
)
from polylogue.pipeline.services.parsing_workflow import ingest_sources, parse_from_raw

if TYPE_CHECKING:
    from polylogue.config import Config, Source
    from polylogue.core.protocols import ProgressCallback
    from polylogue.core.raw_failure_evidence import CohortMembershipRefusalError, RetainedRawDecodeRefusalError
    from polylogue.pipeline.services.ingest_execution import IngestExecution
    from polylogue.sources.revision_backfill import RetainedReplayOutcome
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class IngestRetainedRunner(Protocol):
    """The retained owner's publication, settling each typed refusal through a callback."""

    def __call__(
        self,
        raw_ids: Sequence[str],
        *,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None] | None = None,
    ) -> Awaitable[RetainedReplayOutcome]: ...


class ParsingService:
    """Service for parsing sessions from sources asynchronously."""

    #: Raw records per parse page; a paging granularity, not an admission limit.
    RAW_BATCH_SIZE = 50
    DEFAULT_RAW_BATCH_BLOB_LIMIT_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        repository: SessionRepository,
        archive_root: Path,
        config: Config,
        *,
        execution: IngestExecution | None = None,
        retained_runner: IngestRetainedRunner | None = None,
    ) -> None:
        self.repository = repository
        self.archive_root = archive_root
        self.config = config
        self.execution = execution
        self.retained_runner = retained_runner

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
        max_pass_seconds: float | None = None,
    ) -> IngestResult:
        return await ingest_sources(
            self,
            sources=sources,
            stage=stage,
            ui=ui,
            progress_callback=progress_callback,
            parse_records=parse_records,
            skip_acquire=skip_acquire,
            max_pass_seconds=max_pass_seconds,
        )

    @property
    def raw_batch_blob_limit_bytes(self) -> int:
        return self.DEFAULT_RAW_BATCH_BLOB_LIMIT_BYTES

    async def parse_from_raw(
        self,
        *,
        raw_ids: list[str] | None = None,
        provider: str | None = None,
        progress_callback: ProgressCallback | None = None,
        max_pass_seconds: float | None = None,
    ) -> ParseResult:
        return await parse_from_raw(
            self,
            raw_ids=raw_ids,
            provider=provider,
            progress_callback=progress_callback,
            max_pass_seconds=max_pass_seconds,
        )


__all__ = [
    "IngestPhase",
    "IngestResult",
    "IngestState",
    "ParseResult",
    "ParsingService",
]

"""Async parsing service for pipeline operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.pipeline.services.parsing_batch import process_raw_batch
from polylogue.pipeline.services.parsing_models import (
    IngestPhase,
    IngestResult,
    IngestState,
    ParseResult,
)
from polylogue.pipeline.services.parsing_workflow import ingest_sources, parse_from_raw
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.sources.dispatch import parse_payload

if TYPE_CHECKING:
    from polylogue.config import Config, Source
    from polylogue.protocols import ProgressCallback
    from polylogue.sources.parsers.base import ParsedConversation
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.store import RawConversationRecord


class ParsingService:
    """Service for parsing conversations from sources asynchronously."""

    def __init__(
        self,
        repository: ConversationRepository,
        archive_root: Path,
        config: Config,
    ):
        self.repository = repository
        self.archive_root = archive_root
        self.config = config

    def _require_backend(self) -> SQLiteBackend:
        """Return the repository backend or fail explicitly."""
        backend = self.repository.backend
        if backend is None:
            raise RuntimeError("repository backend is not initialized")
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
        skip_validate: bool = False,
    ) -> IngestResult:
        return await ingest_sources(
            self,
            sources=sources,
            stage=stage,
            ui=ui,
            progress_callback=progress_callback,
            parse_records=parse_records,
            skip_acquire=skip_acquire,
            skip_validate=skip_validate,
        )

    RAW_BATCH_SIZE = 50

    async def parse_from_raw(
        self,
        *,
        raw_ids: list[str] | None = None,
        provider: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ParseResult:
        return await parse_from_raw(
            self,
            raw_ids=raw_ids,
            provider=provider,
            progress_callback=progress_callback,
        )

    async def _process_raw_batch(
        self,
        backend: SQLiteBackend,
        batch_ids: list[str],
        result: ParseResult,
        progress_callback: ProgressCallback | None,
    ) -> None:
        await process_raw_batch(self, backend, batch_ids, result, progress_callback)

    async def _parse_raw_record(
        self,
        raw_record: RawConversationRecord,
    ) -> list[ParsedConversation]:
        stored_payload_provider = raw_record.payload_provider
        if not isinstance(stored_payload_provider, str) or not stored_payload_provider.strip():
            stored_payload_provider = None
        envelope = build_raw_payload_envelope(
            raw_record.raw_content,
            source_path=raw_record.source_path,
            fallback_provider=raw_record.provider_name,
            payload_provider=stored_payload_provider,
        )
        raw_record.payload_provider = envelope.provider
        if not envelope.artifact.parse_as_conversation:
            return []

        schema_resolution = SchemaRegistry().resolve_payload(
            envelope.provider,
            envelope.payload,
            source_path=raw_record.source_path,
        )
        return parse_payload(
            envelope.provider,
            envelope.payload,
            raw_record.raw_id,
            schema_resolution=schema_resolution,
        )


__all__ = [
    "IngestPhase",
    "IngestResult",
    "IngestState",
    "ParseResult",
    "ParsingService",
]

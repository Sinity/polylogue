"""Async parsing service for pipeline operations.

Entry point for the unified ingest pipeline. Delegates to:
- parsing_workflow.py for orchestration (acquire → ingest)
- ingest_batch.py for batch processing (ProcessPool + sync writes)
- ingest_worker.py for per-record work (decode + validate + parse + transform)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository


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

    async def _parse_raw_record(
        self,
        raw_record,
    ):
        """Parse a single raw record into conversations.

        Thin wrapper over the subprocess worker for test compatibility.
        Returns list[ParsedConversation] matching the old API.
        """
        import re

        from polylogue.lib.raw_payload import build_raw_payload_envelope
        from polylogue.schemas.runtime_registry import SchemaRegistry
        from polylogue.sources.dispatch import parse_payload
        from polylogue.storage.blob_store import get_blob_store

        _SOURCE_HASH_SUFFIX = re.compile(r"-(?:[0-9a-f]{16,64})$", re.IGNORECASE)

        def _fallback_id(source_path, raw_id):
            if not source_path:
                return raw_id
            normalized = source_path.replace("\\", "/")
            entry_path = normalized.rsplit(":", 1)[-1]
            stem = Path(entry_path).stem
            if not stem:
                return raw_id
            cleaned = _SOURCE_HASH_SUFFIX.sub("", stem).strip("._- ")
            return cleaned or stem

        stored_payload_provider = raw_record.payload_provider
        if not isinstance(stored_payload_provider, str) or not stored_payload_provider.strip():
            stored_payload_provider = None
        blob_store = get_blob_store()
        raw_source = blob_store.blob_path(raw_record.raw_id)
        envelope = build_raw_payload_envelope(
            raw_source,
            source_path=raw_record.source_path,
            fallback_provider=raw_record.provider_name,
            payload_provider=stored_payload_provider,
        )
        raw_record.payload_provider = envelope.provider
        if not envelope.artifact.parse_as_conversation:
            return []

        registry = SchemaRegistry()
        schema_resolution = registry.resolve_payload(
            envelope.provider,
            envelope.payload,
            source_path=raw_record.source_path,
        )
        conversations = parse_payload(
            envelope.provider,
            envelope.payload,
            _fallback_id(raw_record.source_path, raw_record.raw_id),
            schema_resolution=schema_resolution,
        )

        # Apply timestamp defaults
        fallback_timestamp = raw_record.file_mtime
        enriched = []
        for convo in conversations:
            updates: dict[str, object] = {}
            if convo.created_at is None and fallback_timestamp:
                updates["created_at"] = fallback_timestamp
            effective_created = updates.get("created_at", convo.created_at)
            if convo.updated_at is None and isinstance(effective_created, str) and effective_created:
                updates["updated_at"] = effective_created
            enriched.append(convo.model_copy(update=updates) if updates else convo)
        return enriched

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


__all__ = [
    "IngestPhase",
    "IngestResult",
    "IngestState",
    "ParseResult",
    "ParsingService",
]

"""Async parsing service for pipeline operations.

This service implements the async PARSE stage of the pipeline:
- Reads from raw_conversations (DB)
- Parses into typed conversation/message records
- Stores in the conversations/messages tables

It does NOT handle raw storage - that's the acquisition stage's job.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from polylogue.lib.log import get_logger
from polylogue.pipeline.async_prepare import async_prepare_records
from polylogue.sources.source import _parse_json_payload
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import Config, Source
    from polylogue.sources.parsers.base import ParsedConversation
    from polylogue.storage.async_repository import AsyncConversationRepository

logger = get_logger(__name__)

_ProcessResult: TypeAlias = tuple[str, dict[str, int], bool]


class AsyncParseResult:
    """Result of an async parsing operation."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }
        self.changed_counts: dict[str, int] = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
        }
        self.processed_ids: set[str] = set()
        self.parse_failures: int = 0
        self.drift_counts: dict[str, int] = {}  # provider -> drift warning count
        self._lock = asyncio.Lock()

    async def merge_result(
        self,
        conversation_id: str,
        result_counts: dict[str, int],
        content_changed: bool,
    ) -> None:
        """Merge a single conversation's result into the aggregate.

        Args:
            conversation_id: ID of the processed conversation
            result_counts: Count dictionary from async_prepare_records
            content_changed: Whether content hash changed
        """
        ingest_changed = (result_counts["conversations"] + result_counts["messages"] + result_counts["attachments"]) > 0

        async with self._lock:
            if ingest_changed or content_changed:
                self.processed_ids.add(conversation_id)
            if content_changed:
                self.changed_counts["conversations"] += 1
            if result_counts["messages"]:
                self.changed_counts["messages"] += result_counts["messages"]
            if result_counts["attachments"]:
                self.changed_counts["attachments"] += result_counts["attachments"]
            for key, value in result_counts.items():
                if key in self.counts:
                    self.counts[key] += value


class AsyncParsingService:
    """Service for parsing conversations from sources asynchronously.

    This service implements the async PARSE stage of the pipeline.
    It does NOT handle raw storage - use AcquisitionService for that.
    """

    def __init__(
        self,
        repository: AsyncConversationRepository,
        archive_root: Path,
        config: Config,
        drive_client_factory: Any | None = None,
    ):
        """Initialize the async parsing service.

        Args:
            repository: Async storage repository for database operations
            archive_root: Root directory for archived conversations
            config: Application configuration
            drive_client_factory: Optional factory callable returning a DriveClient
        """
        self.repository = repository
        self.archive_root = archive_root
        self.config = config
        self.drive_client_factory = drive_client_factory
        # Per-run drift tracking (reset each parse_from_raw call)
        self._drift_counts: dict[str, int] = {}
        self._drift_lock = asyncio.Lock()

    async def parse_sources(
        self,
        sources: list[Source],
        *,
        ui: object | None = None,
        download_assets: bool = True,
        progress_callback: Any | None = None,
    ) -> AsyncParseResult:
        """Parse conversations from sources via acquire → parse flow.

        This is a convenience method that runs both stages:
        1. ACQUIRE: Store raw bytes to raw_conversations
        2. PARSE: Parse raw_conversations into conversations

        Args:
            sources: List of sources to ingest
            ui: Optional UI object for user interaction
            download_assets: Whether to download attachments from Drive
            progress_callback: Optional callback for progress updates

        Returns:
            AsyncParseResult with counts and processed conversation IDs
        """
        from polylogue.pipeline.services.async_acquisition import AsyncAcquisitionService

        backend = self.repository._backend
        if backend is None:
            raise RuntimeError("Repository backend is not initialized")

        # Stage 1: ACQUIRE - store raw bytes (async)
        acquire_service = AsyncAcquisitionService(backend=backend)
        acquire_result = await acquire_service.acquire_sources(
            sources,
            progress_callback=progress_callback,
        )

        # Find orphaned raw records (raw data exists but conversation was deleted/missing)
        async with backend._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT r.raw_id
                FROM raw_conversations r
                LEFT JOIN conversations c ON r.raw_id = c.raw_id
                WHERE c.conversation_id IS NULL
            """)
            orphaned_rows = await cursor.fetchall()
        orphaned_ids = [row["raw_id"] for row in orphaned_rows]

        # Combine newly acquired + orphaned raw IDs
        all_raw_ids = list(acquire_result.raw_ids)
        if orphaned_ids:
            seen = set(all_raw_ids)
            for oid in orphaned_ids:
                if oid not in seen:
                    all_raw_ids.append(oid)
                    seen.add(oid)
            logger.info(
                "Found orphaned raw records to re-parse",
                orphaned=len(orphaned_ids),
                newly_acquired=len(acquire_result.raw_ids),
            )

        # Stage 2: PARSE - parse raw_conversations into conversations
        if all_raw_ids:
            return await self.parse_from_raw(
                raw_ids=all_raw_ids,
                progress_callback=progress_callback,
            )
        else:
            # Nothing to process
            return AsyncParseResult()

    # Batch size for processing raw records to limit memory usage.
    # Each raw record may contain multi-MB JSONL content; loading thousands
    # at once caused OOM kills on archives with >3000 conversations.
    RAW_BATCH_SIZE = 200

    async def parse_from_raw(
        self,
        *,
        raw_ids: list[str] | None = None,
        provider: str | None = None,
        progress_callback: Any | None = None,
    ) -> AsyncParseResult:
        """Parse raw_conversations from DB into conversations.

        This is the proper PARSE stage: reads from raw_conversations table
        (populated by AcquisitionService), parses, and stores to conversations.

        Processes records in batches to limit memory usage. Each raw record
        may be a multi-MB JSONL file; loading all of them at once OOMs.

        Args:
            raw_ids: Optional list of specific raw_ids to process.
                     If None, processes all raw_conversations (optionally filtered).
            provider: Optional provider filter (only process this provider)
            progress_callback: Optional callback for progress updates

        Returns:
            AsyncParseResult with counts and processed conversation IDs
        """
        result = AsyncParseResult()

        # Reset per-run drift tracking
        async with self._drift_lock:
            self._drift_counts.clear()

        # Use the repository's backend - same connection management
        backend = self.repository._backend
        if backend is None:
            raise RuntimeError("Repository backend is not initialized")

        # Collect raw_ids to process (just IDs, not full records — memory-safe)
        if raw_ids is not None:
            ids_to_process = list(raw_ids)
        else:
            ids_to_process = [
                r.raw_id async for r in backend.iter_raw_conversations(provider=provider)
            ]

        # Process in batches to limit memory
        for batch_start in range(0, len(ids_to_process), self.RAW_BATCH_SIZE):
            batch_ids = ids_to_process[batch_start : batch_start + self.RAW_BATCH_SIZE]
            await self._process_raw_batch(backend, batch_ids, result, progress_callback)

        if result.processed_ids:
            invalidate_search_cache()
            logger.debug("Search cache invalidated after parsing %d conversations", len(result.processed_ids))

        # Copy per-run drift counts to result and check regeneration threshold
        result.drift_counts = dict(self._drift_counts)
        await self._maybe_regenerate_schemas(result.drift_counts)

        return result

    async def _process_raw_batch(
        self,
        backend: Any,
        batch_ids: list[str],
        result: AsyncParseResult,
        progress_callback: Any | None,
    ) -> None:
        """Process a batch of raw conversation IDs."""
        # Load only this batch into memory
        raw_records = [await backend.get_raw_conversation(raw_id) for raw_id in batch_ids]
        raw_records = [r for r in raw_records if r is not None]

        # Parse raw records into conversation objects
        items_to_process: list[tuple[ParsedConversation, str, str]] = []

        for raw_record in raw_records:
            try:
                parsed_convos = await self._parse_raw_record(raw_record)
                source_name = raw_record.source_name or raw_record.source_path
                for convo in parsed_convos:
                    items_to_process.append((convo, source_name, raw_record.raw_id))
            except Exception as exc:
                logger.error(
                    "Failed to parse raw conversation",
                    raw_id=raw_record.raw_id,
                    provider=raw_record.provider_name,
                    error=str(exc),
                )
                result.parse_failures += 1

        # Free raw records from memory before processing
        del raw_records

        # Process parsed conversations with asyncio concurrency control
        semaphore = asyncio.Semaphore(16)  # Limit concurrent operations

        async def _process_one(
            convo_item: ParsedConversation,
            source_name_item: str,
            raw_id: str,
        ) -> _ProcessResult:
            async with semaphore:
                convo_id, counts, changed = await async_prepare_records(
                    convo_item,
                    source_name_item,
                    archive_root=self.archive_root,
                    backend=backend,
                    repository=self.repository,
                    raw_id=raw_id,
                )
                return (convo_id, counts, changed)

        # Process all items concurrently with semaphore limiting concurrency
        tasks = [_process_one(convo, source_name, raw_id) for convo, source_name, raw_id in items_to_process]

        for coro in asyncio.as_completed(tasks):
            try:
                convo_id, result_counts, content_changed = await coro
                await result.merge_result(convo_id, result_counts, content_changed)
                if progress_callback:
                    progress_callback(1, desc="Parsing")
            except Exception as exc:
                logger.error("Error processing conversation: %s", exc)
                result.parse_failures += 1
                if progress_callback:
                    progress_callback(1, desc="Parsing")

    async def _parse_raw_record(self, raw_record: RawConversationRecord) -> list[ParsedConversation]:
        """Parse a raw conversation record into ParsedConversation(s).

        Handles both single JSON documents and JSONL (newline-delimited JSON).
        JSONL is the format used by claude-code, codex, and gemini sources.

        Args:
            raw_record: Raw conversation record from database

        Returns:
            List of parsed conversations (usually 1, but could be more for bundles)
        """
        content = raw_record.raw_content
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)

        # Try single JSON first (fast path for chatgpt, claude-ai)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Fall back to JSONL parsing (claude-code, codex, gemini)
            lines = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if not lines:
                raise
            payload = lines

        # Schema validation: non-blocking, logs drift and errors
        await self._validate_payload(raw_record.provider_name, payload, raw_record.raw_id)

        # Use the existing parser dispatcher
        return _parse_json_payload(
            raw_record.provider_name,
            payload,
            raw_record.raw_id,  # Use raw_id as fallback conversation ID
        )

    async def _validate_payload(
        self,
        provider_name: str,
        payload: Any,
        raw_id: str,
    ) -> None:
        """Run schema validation on a parsed payload, logging drift and errors.

        Never raises — validation is advisory, not blocking.
        For list payloads (JSONL), validates the first item as a sample.
        """
        from polylogue.schemas.validator import SchemaValidator

        try:
            validator = SchemaValidator.for_provider(provider_name)
        except (FileNotFoundError, ImportError):
            return

        # For JSONL payloads, sample-validate the first dict item
        sample: Any
        if isinstance(payload, list):
            dicts = [item for item in payload if isinstance(item, dict)]
            if not dicts:
                return
            sample = dicts[0]
        elif isinstance(payload, dict):
            sample = payload
        else:
            return

        try:
            result = validator.validate(sample)
            if not result.is_valid:
                logger.warning(
                    "Schema validation errors for %s",
                    provider_name,
                    raw_id=raw_id,
                    errors=result.errors[:5],
                )
            if result.has_drift:
                logger.info(
                    "Schema drift detected for %s",
                    provider_name,
                    raw_id=raw_id,
                    drift=result.drift_warnings[:10],
                )
                async with self._drift_lock:
                    self._drift_counts[provider_name] = self._drift_counts.get(provider_name, 0) + 1
        except Exception as exc:
            logger.debug(
                "Schema validation skipped for %s: %s",
                provider_name,
                exc,
            )

    # Drift threshold: auto-regenerate schema if a provider exceeds this
    # many drift warnings in a single parsing run.
    DRIFT_REGEN_THRESHOLD = 5

    async def _maybe_regenerate_schemas(self, drift_counts: dict[str, int]) -> None:
        """Auto-regenerate schemas for providers with excessive drift.

        If a provider accumulated more than DRIFT_REGEN_THRESHOLD drift
        warnings during this parsing run, regenerate its schema from the
        current database samples and register it as a new version.
        """
        from polylogue.schemas.registry import SchemaRegistry
        from polylogue.schemas.schema_inference import generate_provider_schema

        providers_to_regen = [p for p, count in drift_counts.items() if count > self.DRIFT_REGEN_THRESHOLD]

        if not providers_to_regen:
            return

        backend = self.repository._backend
        db_path = backend._db_path if backend else None

        registry = SchemaRegistry()

        for provider in providers_to_regen:
            try:
                gen_result = generate_provider_schema(provider, db_path=db_path)
                if gen_result.success and gen_result.schema:
                    version = registry.register_schema(provider, gen_result.schema)
                    logger.info(
                        "Auto-regenerated schema for %s as %s (drift=%d)",
                        provider,
                        version,
                        drift_counts[provider],
                    )
                else:
                    logger.warning(
                        "Schema regeneration failed for %s: %s",
                        provider,
                        gen_result.error,
                    )
            except Exception as exc:
                logger.warning(
                    "Schema regeneration error for %s: %s",
                    provider,
                    exc,
                )


__all__ = ["AsyncParsingService", "AsyncParseResult"]

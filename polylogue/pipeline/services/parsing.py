"""Async parsing service for pipeline operations.

This service implements the async PARSE stage of the pipeline:
- Reads from raw_conversations (DB)
- Parses into typed conversation/message records
- Stores in the conversations/messages tables

It does NOT handle raw storage - that's the acquisition stage's job.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os

import orjson
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.lib.log import get_logger
from polylogue.lib.raw_payload import decode_raw_payload, infer_payload_provider
from polylogue.pipeline.services.ingest_state import IngestState
from polylogue.pipeline.ids import conversation_id as make_conversation_id
from polylogue.pipeline.prepare import PrepareCache, prepare_records
from polylogue.sources.source import parse_payload
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import Config, Source
    from polylogue.sources.parsers.base import ParsedConversation
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


class ParseResult:
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
            result_counts: Count dictionary from prepare_records
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


@dataclass
class IngestResult:
    """Result of acquire -> validate -> parse orchestration."""

    acquire_result: Any
    validation_result: Any | None
    parse_result: ParseResult
    parse_raw_ids: list[str]


class ParsingService:
    """Service for parsing conversations from sources asynchronously.

    This service implements the async PARSE stage of the pipeline.
    It does NOT handle raw storage - use AcquisitionService for that.
    """

    def __init__(
        self,
        repository: ConversationRepository,
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

    def _require_backend(self) -> Any:
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
        progress_callback: Any | None = None,
    ) -> ParseResult:
        """Parse conversations from sources via acquire → validate → parse flow.

        This is a convenience method that runs the three pipeline stages:
        1. ACQUIRE: Store raw bytes to raw_conversations
        2. VALIDATE: Persist validation status for raw payloads
        3. PARSE: Parse validated raw_conversations into conversations

        Args:
            sources: List of sources to ingest
            ui: Optional UI object for user interaction
            download_assets: Whether to download attachments from Drive
            progress_callback: Optional callback for progress updates

        Returns:
            ParseResult with counts and processed conversation IDs
        """
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
        progress_callback: Any | None = None,
        parse_records: bool = True,
    ) -> IngestResult:
        """Canonical ingestion orchestration for runtime callers.

        Flow:
        1. Build a canonical plan from source scans + persisted raw state
        2. Acquire any newly scanned raw payloads
        3. Validate pending raw payloads (new + backlog)
        4. Optionally parse validated raw payloads
        """
        from polylogue.pipeline.services.acquisition import AcquisitionService
        from polylogue.pipeline.services.planning import PlanningService
        from polylogue.pipeline.services.validation import ValidationService

        backend = self._require_backend()

        plan_stage = stage if stage in {"acquire", "validate", "parse", "all"} else ("all" if parse_records else "validate")
        planning_service = PlanningService(backend=backend, config=self.config)
        plan = await planning_service.build_plan(
            sources=sources,
            stage=plan_stage,
            ui=ui,
            progress_callback=progress_callback,
        )

        acquire_service = AcquisitionService(backend=backend)
        acquire_result = await acquire_service.store_records(
            plan.store_records,
        )
        source_names = [source.name for source in sources]
        ingest_state = IngestState(
            source_names=tuple(source_names),
            parse_requested=parse_records,
        )
        ingest_state.record_acquired(acquire_result.raw_ids)

        validation_ids = [record.raw_id for record in plan.validate_records]
        ingest_state.record_validation_candidates(validation_ids)

        validation_result = None
        if validation_ids:
            validation_service = ValidationService(backend=backend)
            validation_result = await validation_service.validate_raw_ids(
                raw_ids=validation_ids,
                progress_callback=progress_callback,
            )
        ingest_state.record_validation_result(
            validation_result.parseable_raw_ids if validation_result else [],
        )

        parse_raw_ids: list[str] = []
        parse_result = ParseResult()
        if parse_records:
            parse_raw_ids = list(plan.parse_ready_raw_ids)
            if validation_result is not None:
                parse_raw_ids.extend(validation_result.parseable_raw_ids)
                parse_raw_ids = list(dict.fromkeys(parse_raw_ids))
            current_validation_ids = set(ingest_state.validation_raw_ids)
            persisted_validated_ids = [
                raw_id
                for raw_id in parse_raw_ids
                if raw_id not in current_validation_ids
            ]
            ingest_state.record_parse_candidates(
                parse_raw_ids,
                persisted_validated_raw_ids=persisted_validated_ids,
            )
            if parse_raw_ids:
                parse_result = await self.parse_from_raw(
                    raw_ids=parse_raw_ids,
                    progress_callback=progress_callback,
                )
            ingest_state.record_parse_completed()
            parse_raw_ids = ingest_state.parse_raw_ids

        return IngestResult(
            acquire_result=acquire_result,
            validation_result=validation_result,
            parse_result=parse_result,
            parse_raw_ids=parse_raw_ids,
        )

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
    ) -> ParseResult:
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
            ParseResult with counts and processed conversation IDs
        """
        result = ParseResult()

        # Use the repository's backend - same connection management
        backend = self._require_backend()

        # Collect raw_ids to process (just IDs, not full records — memory-safe)
        if raw_ids is not None:
            total = len(raw_ids)
            if progress_callback is not None:
                progress_callback(0, desc=f"Parsing ({total:,} raw)")
            for batch_start in range(0, total, self.RAW_BATCH_SIZE):
                batch_ids = raw_ids[batch_start : batch_start + self.RAW_BATCH_SIZE]
                await self._process_raw_batch(
                    backend,
                    batch_ids,
                    result,
                    progress_callback,
                )
        else:
            if progress_callback is not None:
                progress_callback(0, desc="Parsing")
            batch_ids: list[str] = []
            async for raw_record in backend.iter_raw_conversations(provider=provider):
                batch_ids.append(raw_record.raw_id)
                if len(batch_ids) >= self.RAW_BATCH_SIZE:
                    await self._process_raw_batch(
                        backend,
                        batch_ids,
                        result,
                        progress_callback,
                    )
                    batch_ids = []
            if batch_ids:
                await self._process_raw_batch(
                    backend,
                    batch_ids,
                    result,
                    progress_callback,
                )

        if result.processed_ids:
            invalidate_search_cache()
            logger.debug("Search cache invalidated after parsing %d conversations", len(result.processed_ids))

        return result

    async def _process_raw_batch(
        self,
        backend: Any,
        batch_ids: list[str],
        result: ParseResult,
        progress_callback: Any | None,
    ) -> None:
        """Process a batch of raw conversation IDs.

        Uses PrepareCache to replace N per-conversation DB queries with 2 bulk
        queries for the entire batch. Flow:
        1. Batch-load raw records from DB
        2. Parse all into conversations (collect, don't queue yet)
        3. Pre-compute candidate CIDs → bulk-load PrepareCache
        4. Queue work items to async workers with shared cache
        """
        # Batch-load raw records in one query instead of N sequential round-trips
        raw_records = await backend.get_raw_conversations_batch(batch_ids)

        # Phase 1: Parse all raw records, collecting work items
        work_items: list[tuple[ParsedConversation, str, str]] = []  # (convo, source_name, raw_id)
        failed_raw_ids: dict[str, str] = {}  # raw_id -> error message

        for raw_record in raw_records:
            try:
                parsed_convos = await self._parse_raw_record(raw_record)
                source_name = raw_record.source_name or raw_record.source_path
                for convo in parsed_convos:
                    work_items.append((convo, source_name, raw_record.raw_id))
            except (json.JSONDecodeError, orjson.JSONDecodeError, ValueError, TypeError) as exc:
                logger.error(
                    "Failed to parse raw conversation",
                    raw_id=raw_record.raw_id,
                    provider=raw_record.provider_name,
                    error=str(exc),
                )
                result.parse_failures += 1
                failed_raw_ids[raw_record.raw_id] = str(exc)[:500]

        # Free raw records — parsed conversations are much smaller
        del raw_records

        if not work_items:
            # All records failed to parse — mark failures and return
            for rid, error in failed_raw_ids.items():
                await backend.mark_raw_parsed(rid, error=error)
            return

        # Phase 2: Pre-compute candidate CIDs and bulk-load cache
        candidate_cids: set[str] = set()
        for convo, _, _ in work_items:
            cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)
            candidate_cids.add(cid)
            # Also include parent CIDs for FK resolution
            if convo.parent_conversation_provider_id:
                parent_cid = make_conversation_id(convo.provider_name, convo.parent_conversation_provider_id)
                candidate_cids.add(parent_cid)

        cache = await PrepareCache.load(backend, candidate_cids)

        # Phase 3: Queue work to async workers with shared cache
        worker_count = min(os.cpu_count() or 4, 16)
        queue: asyncio.Queue[tuple[ParsedConversation, str, str] | None] = asyncio.Queue(
            maxsize=worker_count * 2
        )
        succeeded_raw_ids: set[str] = set()
        tracking_lock = asyncio.Lock()

        async def _worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                convo_item, source_name_item, raw_id = item
                try:
                    convo_id, result_counts, content_changed = await prepare_records(
                        convo_item,
                        source_name_item,
                        archive_root=self.archive_root,
                        backend=backend,
                        repository=self.repository,
                        raw_id=raw_id,
                        cache=cache,
                    )
                    await result.merge_result(convo_id, result_counts, content_changed)
                    async with tracking_lock:
                        succeeded_raw_ids.add(raw_id)
                except Exception as exc:
                    logger.error("Error processing conversation: %s", exc)
                    result.parse_failures += 1
                    async with tracking_lock:
                        failed_raw_ids[raw_id] = str(exc)[:500]
                finally:
                    if progress_callback:
                        progress_callback(1)
                    queue.task_done()

        workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]

        for item in work_items:
            await queue.put(item)
        del work_items  # Free parsed conversations as they're consumed

        await queue.join()
        for _ in range(worker_count):
            await queue.put(None)
        await asyncio.gather(*workers)

        # Mark parse status for all processed raw records.
        # Three cases:
        # - Only succeeded: mark as parsed (clears any previous error)
        # - Only failed: mark with error (will be retried next run)
        # - Both (bundle partially failed): mark with error so operators
        #   can see what's wrong; record stays unparsed for retry
        for rid in succeeded_raw_ids:
            if rid not in failed_raw_ids:
                await backend.mark_raw_parsed(rid)
        for rid, error in failed_raw_ids.items():
            await backend.mark_raw_parsed(rid, error=error)

    async def _parse_raw_record(
        self,
        raw_record: RawConversationRecord,
    ) -> list[ParsedConversation]:
        """Parse a raw conversation record into ParsedConversation(s).

        Handles both single JSON documents and JSONL (newline-delimited JSON).
        JSONL is the format used by claude-code, codex, and gemini sources.

        Args:
            raw_record: Raw conversation record from database

        Returns:
            List of parsed conversations (usually 1, but could be more for bundles)
        """
        payload = decode_raw_payload(raw_record.raw_content).payload
        provider = infer_payload_provider(
            payload,
            source_path=raw_record.source_path,
            fallback_provider=raw_record.provider_name,
        )

        # Use the existing parser dispatcher
        return parse_payload(
            provider,
            payload,
            raw_record.raw_id,  # Use raw_id as fallback conversation ID
        )

__all__ = ["ParsingService", "ParseResult", "IngestResult"]

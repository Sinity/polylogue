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
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from polylogue.logging import get_logger
from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.pipeline.ids import conversation_id as make_conversation_id
from polylogue.pipeline.prepare import PrepareCache, prepare_records
from polylogue.protocols import ProgressCallback
from polylogue.sources.source import parse_payload
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import Config, Source
    from polylogue.pipeline.services.acquisition import AcquireResult
    from polylogue.pipeline.services.validation import ValidateResult
    from polylogue.sources.parsers.base import ParsedConversation
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


class IngestPhase(str, Enum):
    """Phases for acquire → validate → parse orchestration."""

    INIT = "init"
    ACQUIRED = "acquired"
    VALIDATED = "validated"
    PARSED = "parsed"


def _dedupe_ids(raw_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(raw_ids))


@dataclass(slots=True)
class IngestState:
    """Tracks ingest-state transitions and validates phase ordering."""

    source_names: tuple[str, ...]
    parse_requested: bool
    phase: IngestPhase = IngestPhase.INIT
    acquired_raw_ids: list[str] = field(default_factory=list)
    validation_raw_ids: list[str] = field(default_factory=list)
    parseable_raw_ids: list[str] = field(default_factory=list)
    parse_raw_ids: list[str] = field(default_factory=list)

    def record_acquired(self, raw_ids: Iterable[str]) -> None:
        self._expect_phase(IngestPhase.INIT, "record acquired raw IDs")
        self.acquired_raw_ids = _dedupe_ids(raw_ids)
        self.phase = IngestPhase.ACQUIRED

    def record_validation_candidates(self, raw_ids: Iterable[str]) -> None:
        self._expect_phase(IngestPhase.ACQUIRED, "record validation candidates")
        self.validation_raw_ids = _dedupe_ids(raw_ids)

    def record_validation_result(self, parseable_raw_ids: Iterable[str] | None) -> None:
        self._expect_phase(IngestPhase.ACQUIRED, "record validation result")
        parseable = _dedupe_ids(parseable_raw_ids or [])
        allowed = set(self.validation_raw_ids)
        unexpected = [raw_id for raw_id in parseable if raw_id not in allowed]
        if unexpected:
            raise ValueError(
                "Validation result contains raw IDs outside validation candidates: "
                + ", ".join(unexpected[:5])
            )
        self.parseable_raw_ids = parseable
        self.phase = IngestPhase.VALIDATED

    def record_parse_candidates(
        self,
        raw_ids: Iterable[str],
        *,
        persisted_validated_raw_ids: Iterable[str] = (),
    ) -> None:
        self._expect_phase(IngestPhase.VALIDATED, "record parse candidates")
        parse_ids = _dedupe_ids(raw_ids)
        allowed = set(self.validation_raw_ids) | set(persisted_validated_raw_ids)
        unexpected = [raw_id for raw_id in parse_ids if raw_id not in allowed]
        if unexpected:
            raise ValueError(
                "Parse candidates contain raw IDs outside validation candidates: "
                + ", ".join(unexpected[:5])
            )
        self.parse_raw_ids = parse_ids

    def record_parse_completed(self) -> None:
        self._expect_phase(IngestPhase.VALIDATED, "record parse completion")
        self.phase = IngestPhase.PARSED

    def _expect_phase(self, expected: IngestPhase, action: str) -> None:
        if self.phase != expected:
            raise RuntimeError(
                f"Cannot {action}: expected phase {expected.value}, got {self.phase.value}"
            )


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

    acquire_result: AcquireResult
    validation_result: ValidateResult | None
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
    ):
        """Initialize the async parsing service.

        Args:
            repository: Async storage repository for database operations
            archive_root: Root directory for archived conversations
            config: Application configuration
        """
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
        progress_callback: ProgressCallback | None = None,
        parse_records: bool = True,
        skip_acquire: bool = False,
        skip_validate: bool = False,
    ) -> IngestResult:
        """Canonical ingestion orchestration for runtime callers.

        Flow:
        1. Acquire raw payloads directly into ``raw_conversations`` (streaming)
           — skipped when ``skip_acquire=True`` (stage=="validate"|"parse")
        2. Collect pending validation backlog scoped to the selected sources
        3. Validate pending raw payloads (new + backlog)
           — skipped when ``skip_validate=True`` (stage=="parse")
        4. Optionally parse validated raw payloads

        Stage independence:
        - ``stage=="validate"``: ``skip_acquire=True`` — validates backlog without re-acquiring
        - ``stage=="parse"``: ``skip_acquire=True, skip_validate=True`` — parses backlog without re-running predecessors
        - ``stage=="all"``: full pipeline (default, no skips)
        """
        from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
        from polylogue.pipeline.services.planning import PlanningService
        from polylogue.pipeline.services.validation import ValidationService

        backend = self._require_backend()
        source_names = [source.name for source in sources]

        # --- Acquire ---
        if skip_acquire:
            acquire_result = AcquireResult()
        else:
            acquire_service = AcquisitionService(backend=backend)
            acquire_result = await acquire_service.acquire_sources(
                sources,
                ui=ui,
                progress_callback=progress_callback,
                drive_config=self.config.drive_config,
            )
        ingest_state = IngestState(
            source_names=tuple(source_names),
            parse_requested=parse_records,
        )
        ingest_state.record_acquired(acquire_result.raw_ids)

        planning_service = PlanningService(backend=backend, config=self.config)

        # --- Validate ---
        validation_result = None
        validation_ids: list[str] = []
        if skip_validate:
            # Advance IngestState through validation phases with empty data
            ingest_state.record_validation_candidates([])
            ingest_state.record_validation_result([])
        else:
            validation_ids = list(acquire_result.raw_ids)
            if stage in {"validate", "parse", "all"}:
                validation_ids.extend(
                    await planning_service.collect_validation_backlog(
                        source_names=source_names or None,
                        exclude_raw_ids=validation_ids,
                    )
                )
            ingest_state.record_validation_candidates(validation_ids)

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
            parse_raw_ids = await planning_service.collect_parse_backlog(
                source_names=source_names or None,
                exclude_raw_ids=validation_ids,
            )
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
    # 50 records × ~2-5MB avg = 100-250MB peak per batch — sustainable on
    # memory-constrained systems (200 caused OOM with a 14GB raw DB).
    RAW_BATCH_SIZE = 50

    async def parse_from_raw(
        self,
        *,
        raw_ids: list[str] | None = None,
        provider: str | None = None,
        progress_callback: ProgressCallback | None = None,
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
            async for raw_id in backend.iter_raw_ids(provider_name=provider):
                batch_ids.append(raw_id)
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
        backend: SQLiteBackend,
        batch_ids: list[str],
        result: ParseResult,
        progress_callback: ProgressCallback | None,
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
        payload_providers: dict[str, str | None] = {}

        for raw_record in raw_records:
            try:
                parsed_convos = await self._parse_raw_record(raw_record)
                payload_providers[raw_record.raw_id] = raw_record.payload_provider
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
                payload_providers[raw_record.raw_id] = raw_record.payload_provider

        # Free raw records — parsed conversations are much smaller
        del raw_records

        if not work_items:
            # All records failed to parse — mark failures and return
            for rid, error in failed_raw_ids.items():
                await backend.mark_raw_parsed(
                    rid,
                    error=error,
                    payload_provider=payload_providers.get(rid),
                )
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
                await backend.mark_raw_parsed(rid, payload_provider=payload_providers.get(rid))
        for rid, error in failed_raw_ids.items():
            await backend.mark_raw_parsed(
                rid,
                error=error,
                payload_provider=payload_providers.get(rid),
            )

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

        # Use the existing parser dispatcher
        return parse_payload(
            envelope.provider,
            envelope.payload,
            raw_record.raw_id,  # Use raw_id as fallback conversation ID
        )

__all__ = ["ParsingService", "ParseResult", "IngestResult"]

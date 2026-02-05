"""Ingestion service for pipeline operations.

This service implements the PARSE stage of the pipeline:
- Reads from raw_conversations (DB)
- Parses into typed conversation/message records
- Stores in the conversations/messages tables

It does NOT handle raw storage - that's the acquisition stage's job.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from polylogue.lib.log import get_logger
from polylogue.pipeline.ingest import prepare_ingest
from polylogue.sources.source import _parse_json_payload
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import Config, Source
    from polylogue.sources.parsers.base import ParsedConversation
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)

_ProcessResult: TypeAlias = tuple[str, dict[str, int], bool]


class IngestResult:
    """Result of an ingestion operation."""

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
        self._lock = threading.Lock()

    def merge_result(
        self,
        conversation_id: str,
        result_counts: dict[str, int],
        content_changed: bool,
    ) -> None:
        """Merge a single conversation's result into the aggregate.

        Args:
            conversation_id: ID of the processed conversation
            result_counts: Count dictionary from prepare_ingest
            content_changed: Whether content hash changed
        """
        ingest_changed = (result_counts["conversations"] + result_counts["messages"] + result_counts["attachments"]) > 0

        with self._lock:
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


class IngestionService:
    """Service for ingesting conversations from sources.

    This service implements the PARSE stage of the pipeline.
    It does NOT handle raw storage - use AcquisitionService for that.
    """

    def __init__(
        self,
        repository: ConversationRepository,
        archive_root: Path,
        config: Config,
        drive_client_factory: Any | None = None,
    ):
        """Initialize the ingestion service.

        Args:
            repository: Storage repository for database operations
            archive_root: Root directory for archived conversations
            config: Application configuration
            drive_client_factory: Optional factory callable returning a DriveClient
        """
        self.repository = repository
        self.archive_root = archive_root
        self.config = config
        self.drive_client_factory = drive_client_factory

    def ingest_sources(
        self,
        sources: list[Source],
        *,
        ui: object | None = None,
        download_assets: bool = True,
        progress_callback: Any | None = None,
    ) -> IngestResult:
        """Ingest conversations from sources via acquire → parse flow.

        This is a convenience method that runs both stages:
        1. ACQUIRE: Store raw bytes to raw_conversations
        2. PARSE: Parse raw_conversations into conversations

        Args:
            sources: List of sources to ingest
            ui: Optional UI object for user interaction
            download_assets: Whether to download attachments from Drive
            progress_callback: Optional callback for progress updates

        Returns:
            IngestResult with counts and processed conversation IDs
        """
        from polylogue.pipeline.services.acquisition import AcquisitionService

        backend = self.repository._backend
        if backend is None:
            raise RuntimeError("Repository backend is not initialized")

        # Stage 1: ACQUIRE - store raw bytes
        acquire_service = AcquisitionService(backend=backend)
        acquire_result = acquire_service.acquire_sources(
            sources,
            progress_callback=progress_callback,
        )

        # Stage 2: PARSE - parse raw_conversations into conversations
        if acquire_result.raw_ids:
            return self.ingest_from_raw(
                raw_ids=acquire_result.raw_ids,
                progress_callback=progress_callback,
            )
        else:
            # Nothing acquired, return empty result
            return IngestResult()

    def ingest_from_raw(
        self,
        *,
        raw_ids: list[str] | None = None,
        provider: str | None = None,
        progress_callback: Any | None = None,
    ) -> IngestResult:
        """Parse raw_conversations from DB into conversations.

        This is the proper PARSE stage: reads from raw_conversations table
        (populated by AcquisitionService), parses, and stores to conversations.

        Uses the same backend as the repository to avoid connection conflicts.

        Args:
            raw_ids: Optional list of specific raw_ids to process.
                     If None, processes all raw_conversations (optionally filtered).
            provider: Optional provider filter (only process this provider)
            progress_callback: Optional callback for progress updates

        Returns:
            IngestResult with counts and processed conversation IDs
        """
        result = IngestResult()

        # Use the repository's backend - same connection management
        backend = self.repository._backend
        if backend is None:
            raise RuntimeError("Repository backend is not initialized")

        # Collect raw records to process (materialize to list before writes)
        if raw_ids is not None:
            raw_records = [backend.get_raw_conversation(raw_id) for raw_id in raw_ids]
            raw_records = [r for r in raw_records if r is not None]
        else:
            # Get all raw conversations, optionally filtered by provider
            raw_records = list(backend.iter_raw_conversations(provider=provider))

        # Ensure main thread's read connection is in a clean state before
        # worker threads start writing. This prevents SQLite lock contention.
        main_conn = backend._get_connection()
        main_conn.commit()  # Release any implicit read transaction

        # Parse each raw record
        items_to_process: list[tuple[ParsedConversation, str, str]] = []  # (convo, source_name, raw_id)

        for raw_record in raw_records:
            if raw_record is None:
                continue
            try:
                parsed_convos = self._parse_raw_record(raw_record)
                # Use source_name (config source), fallback to source_path for older records
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

        # Process parsed conversations
        # Use backend's thread-local connection management (not connection_context)
        # to avoid connection conflicts
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures: dict[concurrent.futures.Future[_ProcessResult], str] = {}

            def _process_one(
                convo_item: ParsedConversation,
                source_name_item: str,
                raw_id: str,
            ) -> _ProcessResult:
                """Process a single conversation with raw_id link."""
                # Use backend's thread-local connection (each thread gets its own)
                thread_conn = backend._get_connection()

                convo_id, counts, changed = prepare_ingest(
                    convo_item,
                    source_name_item,
                    archive_root=self.archive_root,
                    conn=thread_conn,
                    repository=self.repository,
                    raw_id=raw_id,  # Link to raw source!
                )

                return (convo_id, counts, changed)

            def _handle_future(fut: concurrent.futures.Future[_ProcessResult]) -> None:
                convo_id, result_counts, content_changed = fut.result()
                result.merge_result(convo_id, result_counts, content_changed)
                if progress_callback:
                    progress_callback(1, desc="Parsing")

            for convo, source_name, raw_id in items_to_process:
                while len(futures) > 16:
                    done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        try:
                            _handle_future(fut)
                        finally:
                            del futures[fut]

                future = executor.submit(_process_one, convo, source_name, raw_id)
                futures[future] = convo.provider_conversation_id

            for fut in concurrent.futures.as_completed(futures):
                try:
                    _handle_future(fut)
                except Exception as exc:
                    logger.error("Error processing conversation", error=str(exc))
                    raise

        if result.processed_ids:
            invalidate_search_cache()
            logger.debug("Search cache invalidated after parsing %d conversations", len(result.processed_ids))

        return result

    def _parse_raw_record(self, raw_record: RawConversationRecord) -> list[ParsedConversation]:
        """Parse a raw conversation record into ParsedConversation(s).

        Args:
            raw_record: Raw conversation record from database

        Returns:
            List of parsed conversations (usually 1, but could be more for bundles)
        """
        # Deserialize the raw JSON bytes
        payload = json.loads(raw_record.raw_content)

        # Use the existing parser dispatcher
        return _parse_json_payload(
            raw_record.provider_name,
            payload,
            raw_record.raw_id,  # Use raw_id as fallback conversation ID
        )

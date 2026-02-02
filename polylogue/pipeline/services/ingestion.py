"""Ingestion service for pipeline operations."""

from __future__ import annotations

import concurrent.futures
import hashlib
import threading
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.core.log import get_logger
from polylogue.importers.base import RawConversationData
from polylogue.ingestion import DriveAuthError, iter_drive_conversations, iter_source_conversations
from polylogue.ingestion.source import iter_source_conversations_with_raw
from polylogue.pipeline.ingest import prepare_ingest
from polylogue.storage.backends.sqlite import connection_context
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import Config, Source
    from polylogue.importers.base import ParsedConversation
    from polylogue.storage.repository import StorageRepository

logger = get_logger(__name__)


class IngestResult:
    """Result of an ingestion operation."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "raw_conversations": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
            "skipped_raw": 0,
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
        raw_stored: bool = False,
    ) -> None:
        """Merge a single conversation's result into the aggregate.

        Args:
            conversation_id: ID of the processed conversation
            result_counts: Count dictionary from prepare_ingest
            content_changed: Whether content hash changed
            raw_stored: Whether raw conversation was stored
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
                self.counts[key] += value
            if raw_stored:
                self.counts["raw_conversations"] += 1
            else:
                self.counts["skipped_raw"] += 1


class IngestionService:
    """Service for ingesting conversations from sources."""

    def __init__(
        self,
        repository: StorageRepository,
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
        capture_raw: bool = False,  # Disabled by default to avoid database locking issues
    ) -> IngestResult:
        """Ingest conversations from multiple sources.

        Args:
            sources: List of sources to ingest
            ui: Optional UI object for user interaction
            download_assets: Whether to download attachments from Drive
            progress_callback: Optional callback for progress updates
            capture_raw: Whether to capture and store raw JSON bytes (default True)

        Returns:
            IngestResult with counts and processed conversation IDs
        """
        result = IngestResult()

        # Type alias for the future result tuple
        ProcessResult = tuple[str, dict[str, int], bool, bool]  # convo_id, counts, changed, raw_stored

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures: dict[concurrent.futures.Future[ProcessResult], str] = {}

            def _process_one(
                convo_item: ParsedConversation,
                source_name_item: str,
                raw_data: RawConversationData | None,
            ) -> ProcessResult:
                """Process a single conversation with optional raw capture."""
                db_path = self.repository._db_path
                raw_stored = False
                raw_id: str | None = None

                # Parse and ingest the conversation first
                with connection_context(db_path=db_path) as thread_conn:
                    convo_id, counts, changed = prepare_ingest(
                        convo_item,
                        source_name_item,
                        archive_root=self.archive_root,
                        conn=thread_conn,
                        repository=self.repository,
                    )

                # Store raw conversation AFTER ingestion completes (separate transaction)
                # This avoids nested transaction issues with the repository
                if raw_data is not None:
                    raw_id = hashlib.sha256(raw_data.raw_bytes).hexdigest()
                    acquired_at = datetime.now(timezone.utc).isoformat()

                    raw_record = RawConversationRecord(
                        raw_id=raw_id,
                        provider_name=raw_data.provider_hint or convo_item.provider_name,
                        source_path=raw_data.source_path,
                        source_index=raw_data.source_index,
                        raw_content=raw_data.raw_bytes,
                        acquired_at=acquired_at,
                        file_mtime=raw_data.file_mtime,
                        parsed_conversation_id=convo_id,  # Link directly since we have the ID
                    )

                    # Use repository's backend for raw storage
                    backend = self.repository._backend
                    if backend is not None:
                        raw_stored = backend.save_raw_conversation(raw_record)

                return (convo_id, counts, changed, raw_stored)

            def _handle_future(fut: concurrent.futures.Future[ProcessResult]) -> None:
                convo_id, result_counts, content_changed, raw_stored = fut.result()
                result.merge_result(convo_id, result_counts, content_changed, raw_stored)
                if progress_callback:
                    progress_callback(1, desc="Ingesting")

            for source in sources:
                conversations = self._iter_source_conversations_with_raw_safe(
                    source=source,
                    ui=ui,
                    download_assets=download_assets,
                    capture_raw=capture_raw,
                )

                for raw_data, convo in conversations:
                    # Bounded submission to prevent memory explosion
                    while len(futures) > 16:
                        done, _ = concurrent.futures.wait(
                            futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            try:
                                _handle_future(fut)
                            finally:
                                del futures[fut]

                    future = executor.submit(_process_one, convo, source.name, raw_data)
                    futures[future] = convo.provider_conversation_id

            # Drain remaining futures
            for fut in concurrent.futures.as_completed(futures):
                try:
                    _handle_future(fut)
                except Exception as exc:
                    logger.error("Error processing conversation", error=str(exc))
                    raise

        # Invalidate search cache after ingestion
        if result.processed_ids:
            invalidate_search_cache()
            logger.debug("Search cache invalidated after ingesting %d conversations", len(result.processed_ids))

        return result

    def _iter_source_conversations_safe(
        self,
        *,
        source: Source,
        ui: object | None,
        download_assets: bool,
        cursor_state: dict[str, Any] | None = None,
    ) -> Generator[ParsedConversation, None, None]:
        """Iterate over conversations from a source with error handling.

        Args:
            source: Source to ingest from
            ui: Optional UI object
            download_assets: Whether to download Drive assets
            cursor_state: Optional cursor state tracking

        Yields:
            ParsedConversation objects
        """
        if source.folder:
            try:
                # Instantiate DriveClient only if needed (using DI factory if available)
                client = self.drive_client_factory() if self.drive_client_factory else None

                yield from iter_drive_conversations(
                    source=source,
                    archive_root=self.archive_root,
                    ui=ui,
                    download_assets=download_assets,
                    cursor_state=cursor_state,
                    drive_config=self.config.drive_config,
                    client=client,
                )
            except DriveAuthError as exc:
                logger.warning("Skipping Drive source %s: %s", source.name, exc)
                if cursor_state is not None:
                    cursor_state["error_count"] = cursor_state.get("error_count", 0) + 1
                    cursor_state["latest_error"] = str(exc)
                    cursor_state["latest_error_source"] = source.name
                return
        else:
            yield from iter_source_conversations(source, cursor_state=cursor_state)

    def _iter_source_conversations_with_raw_safe(
        self,
        *,
        source: Source,
        ui: object | None,
        download_assets: bool,
        capture_raw: bool = True,
        cursor_state: dict[str, Any] | None = None,
    ) -> Generator[tuple[RawConversationData | None, ParsedConversation], None, None]:
        """Iterate over conversations with optional raw byte capture.

        Args:
            source: Source to ingest from
            ui: Optional UI object
            download_assets: Whether to download Drive assets
            capture_raw: Whether to capture raw bytes
            cursor_state: Optional cursor state tracking

        Yields:
            Tuples of (RawConversationData | None, ParsedConversation)
        """
        if source.folder:
            # Drive sources - no raw capture support yet
            try:
                client = self.drive_client_factory() if self.drive_client_factory else None

                for convo in iter_drive_conversations(
                    source=source,
                    archive_root=self.archive_root,
                    ui=ui,
                    download_assets=download_assets,
                    cursor_state=cursor_state,
                    drive_config=self.config.drive_config,
                    client=client,
                ):
                    yield (None, convo)  # No raw capture for Drive
            except DriveAuthError as exc:
                logger.warning("Skipping Drive source %s: %s", source.name, exc)
                if cursor_state is not None:
                    cursor_state["error_count"] = cursor_state.get("error_count", 0) + 1
                    cursor_state["latest_error"] = str(exc)
                    cursor_state["latest_error_source"] = source.name
                return
        else:
            yield from iter_source_conversations_with_raw(
                source, cursor_state=cursor_state, capture_raw=capture_raw
            )

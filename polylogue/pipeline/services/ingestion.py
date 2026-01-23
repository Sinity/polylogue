"""Ingestion service for pipeline operations."""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from polylogue.core.log import get_logger
from polylogue.ingestion import DriveAuthError, iter_drive_conversations, iter_source_conversations
from polylogue.pipeline.ingest import prepare_ingest
from polylogue.storage.db import connection_context
from polylogue.storage.search_cache import invalidate_search_cache

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

    def merge_result(self, conversation_id: str, result_counts: dict[str, int], content_changed: bool) -> None:
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
                self.counts[key] += value


class IngestionService:
    """Service for ingesting conversations from sources."""

    def __init__(
        self,
        repository: StorageRepository,
        archive_root: Path,
        config: Config,
    ):
        """Initialize the ingestion service.

        Args:
            repository: Storage repository for database operations
            archive_root: Root directory for archived conversations
            config: Application configuration
        """
        self.repository = repository
        self.archive_root = archive_root
        self.config = config

    def ingest_sources(
        self,
        sources: list[Source],
        *,
        ui: object | None = None,
        download_assets: bool = True,
        progress_callback: Any | None = None,
    ) -> IngestResult:
        """Ingest conversations from multiple sources.

        Args:
            sources: List of sources to ingest
            ui: Optional UI object for user interaction
            download_assets: Whether to download attachments from Drive
            progress_callback: Optional callback for progress updates

        Returns:
            IngestResult with counts and processed conversation IDs
        """
        result = IngestResult()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures: dict[concurrent.futures.Future[tuple[str, dict[str, int], bool]], str] = {}

            def _process_one(convo_item: ParsedConversation, source_name_item: str) -> tuple[str, dict[str, int], bool]:
                # Run preparation in a separate thread with its own connection for reads
                with connection_context(None) as thread_conn:
                    return prepare_ingest(
                        convo_item,
                        source_name_item,
                        archive_root=self.archive_root,
                        conn=thread_conn,
                        repository=self.repository,
                    )

            def _handle_future(fut: concurrent.futures.Future[tuple[str, dict[str, int], bool]]) -> None:
                convo_id, result_counts, content_changed = fut.result()
                result.merge_result(convo_id, result_counts, content_changed)
                if progress_callback:
                    progress_callback(1, desc="Ingesting")

            for source in sources:
                conversations = self._iter_source_conversations_safe(
                    source=source,
                    ui=ui,
                    download_assets=download_assets,
                )

                for convo in conversations:
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

                    future = executor.submit(_process_one, convo, source.name)
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
                yield from iter_drive_conversations(
                    source=source,
                    archive_root=self.archive_root,
                    ui=ui,
                    download_assets=download_assets,
                    cursor_state=cursor_state,
                    drive_config=self.config.drive_config,
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

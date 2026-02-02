"""Acquisition service for pipeline operations.

The acquire stage reads source files and stores raw bytes to raw_conversations.
It does NOT parse - that's the parse stage's job.

Data flow:
    [Source Files] → ACQUIRE → [raw_conversations]
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from polylogue.lib.log import get_logger
from polylogue.importers.base import RawConversationData
from polylogue.ingestion.source import iter_source_conversations_with_raw
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import Source
    from polylogue.storage.backends.sqlite import SQLiteBackend

logger = get_logger(__name__)


class AcquireResult:
    """Result of an acquisition operation."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "acquired": 0,
            "skipped": 0,  # Already in database (by raw_id hash)
            "errors": 0,
        }
        self.raw_ids: list[str] = []  # List of acquired raw_ids


class AcquisitionService:
    """Service for acquiring raw conversation data from sources.

    This service implements the ACQUIRE stage of the pipeline:
    - Reads source files (JSON, JSONL, ZIP)
    - Computes content hash (raw_id)
    - Stores raw bytes in raw_conversations table
    - Does NOT parse or transform the data

    The stored raw data can then be processed by the parse stage.
    """

    def __init__(self, backend: SQLiteBackend):
        """Initialize the acquisition service.

        Args:
            backend: SQLite backend for database operations
        """
        self.backend = backend

    def acquire_sources(
        self,
        sources: list[Source],
        *,
        progress_callback: Any | None = None,
    ) -> AcquireResult:
        """Acquire raw data from multiple sources.

        Reads source files and stores raw bytes in raw_conversations.
        This is a single-threaded operation - no concurrent database access.

        Args:
            sources: List of sources to acquire from
            progress_callback: Optional callback(count, desc=...) for progress

        Returns:
            AcquireResult with counts and list of acquired raw_ids
        """
        result = AcquireResult()

        for source in sources:
            logger.debug("Acquiring from source", source=source.name)

            try:
                for raw_data, _parsed in iter_source_conversations_with_raw(
                    source,
                    capture_raw=True,
                ):
                    if raw_data is None:
                        # This shouldn't happen with capture_raw=True, but handle it
                        logger.warning("No raw data captured for conversation")
                        result.counts["errors"] += 1
                        continue

                    try:
                        raw_id = self._store_raw(raw_data, source.name)
                        if raw_id:
                            result.counts["acquired"] += 1
                            result.raw_ids.append(raw_id)
                        else:
                            result.counts["skipped"] += 1

                        if progress_callback:
                            progress_callback(1, desc="Acquiring")

                    except Exception as exc:
                        logger.error(
                            "Failed to store raw conversation",
                            source=source.name,
                            path=raw_data.source_path,
                            error=str(exc),
                        )
                        result.counts["errors"] += 1

            except Exception as exc:
                logger.error(
                    "Failed to iterate source",
                    source=source.name,
                    error=str(exc),
                )
                result.counts["errors"] += 1

        logger.info(
            "Acquisition complete",
            acquired=result.counts["acquired"],
            skipped=result.counts["skipped"],
            errors=result.counts["errors"],
        )

        return result

    def _store_raw(self, raw_data: RawConversationData, source_name: str) -> str | None:
        """Store raw conversation data.

        Args:
            raw_data: Raw conversation data to store
            source_name: Config source name (e.g., "inbox"), stored separately

        Returns:
            raw_id if newly stored, None if already exists
        """
        raw_id = hashlib.sha256(raw_data.raw_bytes).hexdigest()
        acquired_at = datetime.now(timezone.utc).isoformat()

        record = RawConversationRecord(
            raw_id=raw_id,
            provider_name=raw_data.provider_hint or source_name,
            source_name=source_name,  # Config source name, distinct from provider
            source_path=raw_data.source_path,
            source_index=raw_data.source_index,
            raw_content=raw_data.raw_bytes,
            acquired_at=acquired_at,
            file_mtime=raw_data.file_mtime,
        )

        if self.backend.save_raw_conversation(record):
            return raw_id
        return None

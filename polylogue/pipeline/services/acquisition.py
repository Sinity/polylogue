"""Async acquisition service for pipeline operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

from polylogue.lib.metrics import read_peak_rss_self_mb
from polylogue.logging import get_logger
from polylogue.pipeline.services.acquisition_persistence import persist_raw_record
from polylogue.pipeline.services.acquisition_records import ScanResult
from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.protocols import ProgressCallback
from polylogue.sources.source_acquisition import iter_source_raw_data
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.config import DriveConfig, Source
    from polylogue.sources.drive_types import DriveUILike
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)

__all__ = ["AcquisitionService", "AcquireResult", "iter_source_raw_data"]


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
        """Initialize the async acquisition service.

        Args:
            backend: Async SQLite backend for database operations
        """
        self.backend = backend
        from polylogue.storage.repository import ConversationRepository

        self.repository: ConversationRepository = ConversationRepository(backend=backend)

    async def _persist_record(
        self,
        record: RawConversationRecord,
        *,
        result: AcquireResult,
    ) -> None:
        await persist_raw_record(self.repository, record, result=result)

    async def visit_sources(
        self,
        sources: list[Source],
        *,
        progress_callback: ProgressCallback | None = None,
        ui: object | None = None,
        drive_config: DriveConfig | None = None,
        progress_label: str = "Scanning",
        on_record: Callable[[RawConversationRecord], Awaitable[None]] | None = None,
        observation_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> ScanResult:
        """Visit source raw payloads incrementally without forcing list materialization."""
        result = ScanResult()
        known_mtimes = await self.repository.get_known_source_mtimes()
        drive_ui = cast("DriveUILike | None", ui)

        async def _consume(record: RawConversationRecord) -> None:
            if on_record is not None:
                await on_record(record)
            result.counts["scanned"] += 1

        for source in sources:
            logger.debug("Scanning source", source=source.name)
            cursor_state: CursorStatePayload = {}
            try:
                async for record in iter_raw_record_stream(
                    source,
                    known_mtimes=known_mtimes,
                    ui=drive_ui,
                    cursor_state=cursor_state,
                    drive_config=drive_config,
                    observation_callback=observation_callback,
                    progress_callback=progress_callback,
                ):
                    await _consume(record)
                    if progress_callback:
                        progress_callback(1, desc=f"{progress_label} [{source.name}]")
            except Exception as exc:
                logger.error(
                    "Failed to scan source",
                    source=source.name,
                    error=str(exc),
                )
                result.counts["errors"] += 1
                prior_errors = cursor_state.get("error_count", 0)
                cursor_state["error_count"] = (prior_errors if isinstance(prior_errors, int) else 0) + 1
                cursor_state["latest_error"] = str(exc)

            if cursor_state:
                result.cursors[source.name] = cursor_state

        return result

    async def acquire_sources(
        self,
        sources: list[Source],
        *,
        ui: object | None = None,
        progress_callback: ProgressCallback | None = None,
        drive_config: DriveConfig | None = None,
    ) -> AcquireResult:
        """Acquire raw data from multiple sources.

        Reads source files and stores raw bytes in ``raw_conversations`` without
        materializing the full corpus in memory first.

        Args:
            sources: List of sources to acquire from
            progress_callback: Optional callback(count, desc=...) for progress

        Returns:
            AcquireResult with counts and list of acquired raw_ids
        """
        result = AcquireResult()
        # Records are metadata-only (~1 KB each, no BLOBs). Larger batches
        # reduce commit frequency and async thread-crossing overhead.
        flush_interval = 500
        items_since_flush = 0
        peak_observation: dict[str, object] | None = None
        observation_count = 0
        peak_baseline = read_peak_rss_self_mb() or 0.0
        split_payload_summary = {
            "count": 0,
            "total_blob_mb": 0.0,
            "max_blob_mb": 0.0,
            "total_detect_provider_ms": 0.0,
            "total_classify_ms": 0.0,
            "total_serialize_ms": 0.0,
            "max_detect_provider_ms": 0.0,
            "max_classify_ms": 0.0,
            "max_serialize_ms": 0.0,
        }

        def _observe(observation: dict[str, object]) -> None:
            nonlocal peak_observation, observation_count, peak_baseline
            observation_count += 1
            if observation.get("phase") == "zip-entry-split-payload-serialized":
                split_payload_summary["count"] += 1
                blob_mb = observation.get("blob_mb")
                if isinstance(blob_mb, int | float):
                    split_payload_summary["total_blob_mb"] += float(blob_mb)
                    split_payload_summary["max_blob_mb"] = max(split_payload_summary["max_blob_mb"], float(blob_mb))
                for field, total_key, max_key in (
                    ("detect_provider_ms", "total_detect_provider_ms", "max_detect_provider_ms"),
                    ("classify_ms", "total_classify_ms", "max_classify_ms"),
                    ("serialize_ms", "total_serialize_ms", "max_serialize_ms"),
                ):
                    value = observation.get(field)
                    if isinstance(value, int | float):
                        split_payload_summary[total_key] += float(value)
                        split_payload_summary[max_key] = max(split_payload_summary[max_key], float(value))
            peak_rss_self_mb = observation.get("peak_rss_self_mb")
            if not isinstance(peak_rss_self_mb, int | float):
                return
            if float(peak_rss_self_mb) <= peak_baseline:
                return
            peak_baseline = float(peak_rss_self_mb)
            if peak_observation is None:
                peak_observation = dict(observation)
                return
            peak_value = peak_observation.get("peak_rss_self_mb")
            if not isinstance(peak_value, int | float) or peak_rss_self_mb > peak_value:
                peak_observation = dict(observation)

        async def _store(record: RawConversationRecord) -> None:
            nonlocal items_since_flush
            await self._persist_record(record, result=result)
            items_since_flush += 1
            if items_since_flush >= flush_interval:
                await self.backend.bulk_flush()
                items_since_flush = 0

        async with self.backend.bulk_connection():
            visit_result = await self.visit_sources(
                sources,
                progress_callback=progress_callback,
                ui=ui,
                drive_config=drive_config,
                progress_label="Scanning",
                on_record=_store,
                observation_callback=_observe,
            )
            result.errors += visit_result.counts["errors"]

        if peak_observation is not None:
            result.diagnostics["peak_observation"] = peak_observation
            result.diagnostics["observation_count"] = observation_count
        if split_payload_summary["count"]:
            result.diagnostics["split_payload_summary"] = {
                key: round(value, 3) if isinstance(value, float) else value
                for key, value in split_payload_summary.items()
            }

        return result

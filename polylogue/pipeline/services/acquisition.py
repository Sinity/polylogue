"""Async acquisition service for pipeline operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from polylogue.core.json import JSONDocument
from polylogue.core.metrics import read_peak_rss_self_mb
from polylogue.core.protocols import ProgressCallback
from polylogue.logging import get_logger
from polylogue.pipeline.payload_types import AcquireSplitPayloadSummary
from polylogue.pipeline.services.acquisition_persistence import persist_raw_record
from polylogue.pipeline.services.acquisition_records import ScanResult
from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.sources.drive.types import DriveUILike
from polylogue.sources.source_acquisition import iter_source_raw_data
from polylogue.sources.source_walk import _resolve_source_paths
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.runtime import RawSessionRecord

if TYPE_CHECKING:
    from polylogue.config import DriveConfig, Source
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["AcquisitionService", "AcquireResult", "iter_source_raw_data"]


class AcquisitionService:
    """Service for acquiring raw session data from sources.

    This service implements the ACQUIRE stage of the pipeline:
    - Reads source files (JSON, JSONL, ZIP)
    - Computes content hash (raw_id)
    - Stores raw bytes in raw_sessions table
    - Does NOT parse or transform the data

    The stored raw data can then be processed by the parse stage.
    """

    def __init__(self, backend: SQLiteBackend):
        """Initialize the async acquisition service.

        Args:
            backend: Async SQLite backend for database operations
        """
        self.backend = backend
        from polylogue.storage.repository import SessionRepository

        self.repository: SessionRepository = SessionRepository(backend=backend)

    async def _persist_record(
        self,
        record: RawSessionRecord,
        *,
        result: AcquireResult,
    ) -> None:
        await persist_raw_record(self.repository, record, result=result)

    async def _persist_source_cursors(self, source: Source) -> None:
        """Slize B: Populate source_file_cursor stats for all source paths."""
        if source.path is None:
            return
        for file_path in _resolve_source_paths(source):
            try:
                st = file_path.stat()
                await self.repository.upsert_source_file_cursor(
                    str(file_path),
                    st_dev=st.st_dev,
                    st_ino=st.st_ino,
                    st_size=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                )
            except OSError:
                continue

    async def visit_sources(
        self,
        sources: list[Source],
        *,
        progress_callback: ProgressCallback | None = None,
        ui: object | None = None,
        drive_config: DriveConfig | None = None,
        progress_label: str = "Scanning",
        on_record: Callable[[RawSessionRecord], Awaitable[None]] | None = None,
        on_source_complete: Callable[[], Awaitable[None]] | None = None,
        observation_callback: Callable[[JSONDocument], None] | None = None,
        persist_cursors: bool = True,
        blob_store: BlobStore | None = None,
    ) -> ScanResult:
        """Visit source raw payloads incrementally without forcing list materialization.

        Args:
            persist_cursors: When True (default), save cursor stat fields after
                each source so subsequent runs can skip unchanged files. Set to
                False when the caller only needs a preview scan (e.g. planning)
                and does not want to influence later acquire passes.
        """
        result = ScanResult()
        known_mtimes = await self.repository.get_known_source_mtimes()
        # Slice B: load known cursors for the stat-based fast path.
        known_cursors = await self.repository.get_known_source_cursors()
        if ui is not None and not isinstance(ui, DriveUILike):
            raise TypeError(f"Drive acquisition UI must satisfy DriveUILike, got {type(ui).__name__}")
        drive_ui = ui

        async def _consume(record: RawSessionRecord) -> None:
            if on_record is not None:
                await on_record(record)
            result.counts["scanned"] += 1

        for source in sources:
            logger.debug("Scanning source", source=source.name)
            cursor_state: CursorStatePayload = {}
            try:
                async for record in iter_raw_record_stream(
                    source,
                    blob_root=self.backend.db_path.parent / "blob",
                    blob_store=blob_store,
                    known_mtimes=known_mtimes,
                    known_cursors=known_cursors,
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
                    exc_info=True,
                )
                result.counts["errors"] += 1
                prior_errors = cursor_state.get("error_count", 0)
                cursor_state["error_count"] = int(prior_errors) + 1
                cursor_state["latest_error"] = str(exc)

            if on_source_complete is not None:
                await on_source_complete()

            # Slice B: persist cursor stat fields for all source files after
            # processing so the next run can skip unchanged files.
            if persist_cursors and not source.is_drive:
                await self._persist_source_cursors(source)

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

        Reads source files and stores raw bytes in ``raw_sessions`` without
        materializing the full corpus in memory first.

        Args:
            sources: List of sources to acquire from
            progress_callback: Optional callback(count, desc=...) for progress

        Returns:
            AcquireResult with counts and list of acquired raw_ids
        """
        result = AcquireResult()
        from polylogue.storage.blob_publication import ArchiveBlobPublisher

        blob_publisher = ArchiveBlobPublisher(
            self.backend.db_path.parent / "source.db",
            self.backend.db_path.parent / "blob",
        )
        # Records are metadata-only (~1 KB each, no BLOBs). Larger batches
        # reduce commit frequency and async thread-crossing overhead.
        flush_interval = 500
        pending_records: list[RawSessionRecord] = []
        peak_observation: JSONDocument | None = None
        observation_count = 0
        peak_baseline = read_peak_rss_self_mb() or 0.0
        split_payload_totals = {
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

        def _observe(observation: JSONDocument) -> None:
            nonlocal peak_observation, observation_count, peak_baseline
            observation_count += 1
            if observation.get("phase") == "zip-entry-split-payload-serialized":
                split_payload_totals["count"] += 1
                blob_mb = observation.get("blob_mb")
                if isinstance(blob_mb, int | float):
                    split_payload_totals["total_blob_mb"] += float(blob_mb)
                    split_payload_totals["max_blob_mb"] = max(split_payload_totals["max_blob_mb"], float(blob_mb))
                for field, total_key, max_key in (
                    ("detect_provider_ms", "total_detect_provider_ms", "max_detect_provider_ms"),
                    ("classify_ms", "total_classify_ms", "max_classify_ms"),
                    ("serialize_ms", "total_serialize_ms", "max_serialize_ms"),
                ):
                    value = observation.get(field)
                    if isinstance(value, int | float):
                        split_payload_totals[total_key] += float(value)
                        split_payload_totals[max_key] = max(split_payload_totals[max_key], float(value))
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

        async def _flush_pending() -> None:
            if not pending_records:
                return
            records = tuple(pending_records)
            pending_records.clear()
            async with self.backend.bulk_connection():
                for pending_record in records:
                    await self._persist_record(pending_record, result=result)

        async def _store(record: RawSessionRecord) -> None:
            pending_records.append(record)
            if len(pending_records) >= flush_interval:
                await _flush_pending()

        try:
            visit_result = await self.visit_sources(
                sources,
                progress_callback=progress_callback,
                ui=ui,
                drive_config=drive_config,
                progress_label="Scanning",
                on_record=_store,
                on_source_complete=_flush_pending,
                observation_callback=_observe,
                blob_store=blob_publisher,
            )
            await _flush_pending()
        finally:
            blob_publisher.discard_pending()
        result.errors += visit_result.counts["errors"]

        if peak_observation is not None:
            result.diagnostics["peak_observation"] = peak_observation
            result.diagnostics["observation_count"] = observation_count
        if split_payload_totals["count"]:
            result.diagnostics["split_payload_summary"] = AcquireSplitPayloadSummary(
                count=int(split_payload_totals["count"]),
                total_blob_mb=round(split_payload_totals["total_blob_mb"], 3),
                max_blob_mb=round(split_payload_totals["max_blob_mb"], 3),
                total_detect_provider_ms=round(split_payload_totals["total_detect_provider_ms"], 3),
                total_classify_ms=round(split_payload_totals["total_classify_ms"], 3),
                total_serialize_ms=round(split_payload_totals["total_serialize_ms"], 3),
                max_detect_provider_ms=round(split_payload_totals["max_detect_provider_ms"], 3),
                max_classify_ms=round(split_payload_totals["max_classify_ms"], 3),
                max_serialize_ms=round(split_payload_totals["max_serialize_ms"], 3),
            )

        return result

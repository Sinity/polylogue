"""Async acquisition service for pipeline operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.json import JSONDocument
from polylogue.core.metrics import read_peak_rss_self_mb
from polylogue.core.protocols import ProgressCallback
from polylogue.logging import get_logger
from polylogue.pipeline.payload_types import AcquireSplitPayloadSummary
from polylogue.pipeline.services.acquisition_persistence import persist_raw_record
from polylogue.pipeline.services.acquisition_records import ScanResult
from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.security.excision_policy import ExcisionPolicySnapshot, build_excision_policy_snapshot
from polylogue.sources.cursor import _record_cursor_failure
from polylogue.sources.drive.types import DriveUILike
from polylogue.sources.drive.witness import DriveListingWitness
from polylogue.sources.source_acquisition import iter_source_acquisition_records
from polylogue.sources.source_snapshot import (
    SourceCutPolicy,
    SourceCutResult,
    execute_source_cut,
    preflight_source_cut,
)
from polylogue.storage.cursor_state import CursorFailurePayload, CursorStatePayload
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord

if TYPE_CHECKING:
    from polylogue.config import DriveConfig, Source
    from polylogue.maintenance.source_manifest_continuity import SourceDeclaration
    from polylogue.pipeline.services.ingest_execution import IngestExecution
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["AcquisitionService", "AcquireResult", "iter_source_acquisition_records"]


class AcquisitionService:
    """Service for acquiring raw session data from sources.

    This service implements the ACQUIRE stage of the pipeline:
    - Reads source files (JSON, JSONL, ZIP)
    - Computes content hash (raw_id)
    - Stores raw bytes in raw_sessions table
    - Does NOT parse or transform the data

    The stored raw data can then be processed by the parse stage.
    """

    def __init__(self, backend: SQLiteBackend, *, execution: IngestExecution | None = None):
        """Initialize the async acquisition service.

        Args:
            backend: Async SQLite backend for database operations
        """
        self.backend = backend
        self.execution = execution
        from polylogue.storage.repository import SessionRepository

        self.repository: SessionRepository = SessionRepository(backend=backend)

    @staticmethod
    def cut_source_inputs(
        declarations: list[SourceDeclaration],
        destination: Path,
        *,
        request_id: str = "source-cut",
        policies: Mapping[str, SourceCutPolicy] | None = None,
    ) -> SourceCutResult:
        """Create an immutable candidate boundary for the acquisition owner."""
        preflight = preflight_source_cut(declarations, request_id=request_id, policies=policies)
        return execute_source_cut(preflight, destination)

    async def _persist_record(
        self,
        record: RawSessionRecord,
        *,
        result: AcquireResult,
        policy_snapshot: ExcisionPolicySnapshot,
        prepared_observation: ArtifactObservationRecord | None = None,
        preparation_error: Exception | None = None,
        failures: list[CursorFailurePayload] | None = None,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> str | None:
        return await persist_raw_record(
            self.repository,
            record,
            result=result,
            policy_snapshot=policy_snapshot,
            prepared_observation=prepared_observation,
            preparation_error=preparation_error,
            failures=failures,
            on_failure=on_failure,
        )

    async def _persist_source_cursors(
        self,
        source: Source,
        *,
        cursor_state: CursorStatePayload | None = None,
        observations: dict[str, tuple[str, tuple[int, int, int, int, int], str | None]],
    ) -> None:
        """Persist stat cursors only for source paths acquired successfully."""
        if source.path is None:
            return
        # A failure names a physical file or a ZIP member as ``<file>:<member>``,
        # and a POSIX path may itself contain ``:``, so the first colon is not
        # the boundary. Resolve each failure against the real source files once;
        # the per-file check is then one set lookup however many files failed.
        source_keys = set(observations)
        failed_paths: set[str] = set()
        failed_everything = False
        if cursor_state:
            for failure in cursor_state.get("failed_files", []):
                raw_path = str(failure["path"])
                if raw_path in source_keys:
                    # A real file whose own name contains ``:``; its prefixes
                    # are other files that did not fail.
                    failed_paths.add(raw_path)
                    continue
                # A member coordinate. Its container is a real source file
                # named by a prefix ending before a ``:``; when several real
                # files qualify (``a.zip`` and a file named ``a.zip:m.json``)
                # the coordinate cannot say which, so all of them are
                # withheld -- a needless re-read is safe, a skipped member
                # is not.
                containers = {raw_path[:index] for index, char in enumerate(raw_path) if char == ":"} & source_keys
                if containers:
                    failed_paths.update(containers)
                else:
                    # A failure naming no file this walk resolved (a provenance
                    # path, such as a staged SQLite snapshot's original) cannot
                    # be scoped, so no cursor in this source is proven safe.
                    failed_everything = True
            # An unscoped failure means the pass did not prove any path safe
            # to skip.  Do not turn a failed persistence/read pass into a
            # successful stat cursor for every file in the source.
            failed_everything = failed_everything or bool(cursor_state.get("error_count"))
        for source_path, (canonical_path, observed, profile_key) in observations.items():
            if failed_everything or source_path in failed_paths:
                continue
            await self.repository.upsert_source_file_cursor(
                source_path,
                canonical_source_path=canonical_path,
                captured_profile_key=profile_key,
                st_dev=observed[0],
                st_ino=observed[1],
                st_size=observed[2],
                mtime_ns=observed[3],
            )

    async def visit_sources(
        self,
        sources: list[Source],
        *,
        progress_callback: ProgressCallback | None = None,
        ui: object | None = None,
        drive_config: DriveConfig | None = None,
        progress_label: str = "Scanning",
        on_record: Callable[[RawSessionRecord], Awaitable[None]] | None = None,
        on_source_complete: Callable[[CursorStatePayload], Awaitable[None]] | None = None,
        before_input_complete: Callable[[], Awaitable[None]] | None = None,
        observation_callback: Callable[[JSONDocument], None] | None = None,
        persist_cursors: bool = True,
        blob_store: BlobStore | None = None,
        drive_witnesses: dict[str, DriveListingWitness] | None = None,
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
            witness = None
            if source.is_drive and drive_witnesses is not None:
                witness = DriveListingWitness(source.name, source.folder or "")
                drive_witnesses[source.name] = witness
            observations: dict[str, tuple[str, tuple[int, int, int, int, int], str | None]] = {}

            def observe_input(
                semantic: str,
                physical: str,
                observed: tuple[int, int, int, int, int],
                profile: str | None,
                captured: dict[str, tuple[str, tuple[int, int, int, int, int], str | None]] = observations,
            ) -> None:
                captured[semantic] = (physical, observed, profile)

            try:
                async for record in iter_raw_record_stream(
                    source,
                    drive_witness=witness,
                    blob_root=self.backend.db_path.parent / "blob",
                    blob_store=blob_store,
                    known_mtimes=known_mtimes,
                    known_cursors=known_cursors,
                    ui=drive_ui,
                    cursor_state=cursor_state,
                    drive_config=drive_config,
                    observation_callback=observation_callback,
                    progress_callback=progress_callback,
                    execution=self.execution,
                    input_repository=self.repository if before_input_complete is not None else None,
                    before_input_complete=before_input_complete,
                    input_observation_callback=observe_input,
                ):
                    if record.canonical_source_path is not None and record.captured_file_observation is not None:
                        observations[record.source_path] = (
                            record.canonical_source_path,
                            record.captured_file_observation,
                            record.captured_profile_key,
                        )
                    await _consume(record)
                    if progress_callback:
                        progress_callback(1, desc=f"{progress_label} [{source.name}]")
            except DaemonOperationCancelled:
                raise
            except Exception as exc:
                logger.error(
                    "Failed to scan source",
                    source=source.name,
                    error=str(exc),
                    exc_info=True,
                )
                result.counts["errors"] += 1
                if witness is not None:
                    witness.enumeration_error = type(exc).__name__
                prior_errors = cursor_state.get("error_count", 0)
                cursor_state["error_count"] = int(prior_errors) + 1
                cursor_state["latest_error"] = str(exc)

            if witness is not None:
                # Download/blob failures are not raised by the per-file stream.
                result.counts["errors"] += witness.acquisition_failure_count
            if not source.is_drive:
                # Local reads record per-path failures instead of raising them.
                # Count before completion adds persistence failures, which the
                # persistence owner already counts in AcquireResult.
                result.counts["errors"] += cursor_state.get("failed_count", 0)
            if on_source_complete is not None:
                # The callback may record per-path persistence failures into
                # this source's cursor state before its cursors are saved.
                await on_source_complete(cursor_state)

            # Slice B: persist cursor stat fields for all source files after
            # processing so the next run can skip unchanged files.
            if persist_cursors and not source.is_drive:
                await self._persist_source_cursors(source, cursor_state=cursor_state, observations=observations)

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
        policy_snapshot = (
            build_excision_policy_snapshot(self.backend.db_path.parent)
            if self.execution is None
            else await self.execution.publish_sync(
                "policy", lambda: build_excision_policy_snapshot(self.backend.db_path.parent)
            )
        )
        from polylogue.storage.blob_publication import ArchiveBlobPublisher

        blob_publisher = ArchiveBlobPublisher(
            self.backend.db_path.parent / "source.db",
            self.backend.db_path.parent / "blob",
        )
        # Records are metadata-only (~1 KB each, no BLOBs). Larger batches
        # reduce commit frequency and async thread-crossing overhead.
        flush_interval = 500
        pending_records: list[tuple[RawSessionRecord, ArtifactObservationRecord | None, Exception | None]] = []
        persist_failures: list[CursorFailurePayload] = []
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

            async def persist() -> None:
                current_policy = (
                    policy_snapshot
                    if self.execution is None
                    else build_excision_policy_snapshot(self.backend.db_path.parent)
                )
                async with self.backend.bulk_connection():
                    for pending_record, observation, preparation_error in records:
                        from polylogue.sources.drive.witness import drive_source_prefix

                        witness = next(
                            (
                                value
                                for name, value in result.drive_witnesses.items()
                                if pending_record.source_path.startswith(drive_source_prefix(name))
                            ),
                            None,
                        )

                        def failed(
                            exc: Exception,
                            witness: DriveListingWitness | None = witness,
                            coordinate: str = pending_record.source_path,
                        ) -> None:
                            if witness is not None:
                                witness.record_failure(coordinate, "persist", exc)

                        raw_id = await self._persist_record(
                            pending_record,
                            result=result,
                            policy_snapshot=current_policy,
                            prepared_observation=observation,
                            preparation_error=preparation_error,
                            failures=persist_failures,
                            on_failure=failed,
                        )
                        if witness is not None and raw_id is not None:
                            witness.bind_raw(pending_record.source_path, raw_id)

            if self.execution is None:
                await persist()
            else:
                await self.execution.publish("raw", persist)

        async def _store(record: RawSessionRecord) -> None:
            observation = None
            preparation_error = None
            if self.execution is not None:
                from polylogue.storage.artifacts.inspection import inspect_raw_artifact

                try:
                    observation = await self.execution.prepare(
                        lambda: inspect_raw_artifact(record, blob_store=blob_publisher)
                    )
                except DaemonOperationCancelled:
                    raise
                except Exception as exc:
                    preparation_error = exc
            pending_records.append((record, observation, preparation_error))
            if len(pending_records) >= flush_interval:
                await _flush_pending()

        async def _complete_source(cursor_state: CursorStatePayload) -> None:
            # Pending records belong to the source that just finished (they
            # are flushed at every source boundary), so their persistence
            # failures withhold that source's cursors for exactly those paths.
            await _flush_pending()
            for failure in persist_failures:
                _record_cursor_failure(cursor_state, failure["path"], failure["error"])
            persist_failures.clear()

        try:
            visit_result = await self.visit_sources(
                sources,
                progress_callback=progress_callback,
                ui=ui,
                drive_config=drive_config,
                progress_label="Scanning",
                on_record=_store,
                on_source_complete=_complete_source,
                before_input_complete=_flush_pending,
                observation_callback=_observe,
                blob_store=blob_publisher,
                drive_witnesses=result.drive_witnesses,
            )
            await _flush_pending()
        except BaseException:
            # Preparation has physically drained before it escapes execution.
            # No caller receives this result, so no caller can own its witnesses.
            for witness in result.drive_witnesses.values():
                witness.close()
            result.drive_witnesses.clear()
            raise
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

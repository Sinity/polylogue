"""In-process live batch convergence for daemon source ingestion."""

from __future__ import annotations

import asyncio
import sqlite3
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from json import dumps as json_dumps
from json import loads as json_loads
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.config import Source
from polylogue.core.degraded import is_degraded
from polylogue.core.enums import Provider
from polylogue.core.memory import release_process_memory
from polylogue.core.metrics import (
    read_cgroup_memory_current_mb,
    read_cgroup_memory_peak_mb,
    read_cgroup_memory_swap_current_mb,
    read_cgroup_path,
    read_current_rss_mb,
    read_peak_rss_children_mb,
    read_peak_rss_self_mb,
)
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.errors import DatabaseError, SchemaVersionMismatchError
from polylogue.logging import get_logger
from polylogue.pipeline.services.ingest_batch._models import _IngestBatchSummary
from polylogue.sources.decoders import _iter_json_stream, _ZipEntryValidator
from polylogue.sources.dispatch import (
    _detect_provider_from_raw_bytes,
    is_stream_record_provider,
    parse_payload,
    parse_stream_payload,
)
from polylogue.sources.live.append_ingest import ingest_append_plans
from polylogue.sources.live.batch_observability import (
    record_attempt_progress,
)
from polylogue.sources.live.batch_support import (
    _DEFER_APPEND,
    _MAX_APPEND_PLAN_PAYLOAD_BYTES,
    _STREAMING_FULL_INGEST_BYTES,
    _accumulate_stage_timings,
    _append_plan_group_ready,
    _AppendPlan,
    _AppendResult,
    _archive_blob_exists,
    _blob_copy_heartbeat,
    _DeferredAppend,
    _detect_provider_from_path_sample,
    _full_ingest_result_from_summary,
    _full_ingest_worker_count,
    _full_parse_progress_groups,
    _FullIngestHeartbeat,
    _FullIngestResult,
    _jsonl_provider_and_session_artifact,
    _parse_path_as_session_artifact,
    _parse_payload_as_session_artifact,
    _path_size,
    _throttled_phase_heartbeat,
    cursor_state_after_full_ingest,
    fingerprint_file,
    last_complete_newline_from_tail,
    tail_hash_from_path,
)
from polylogue.sources.live.batch_support import (
    _LARGE_FULL_PARSE_PROGRESS_BYTES as _LARGE_FULL_PARSE_PROGRESS_BYTES,
)
from polylogue.sources.live.batch_support import (
    _SMALL_FULL_PARSE_PROGRESS_MAX_BYTES as _SMALL_FULL_PARSE_PROGRESS_MAX_BYTES,
)
from polylogue.sources.live.batch_support import (
    _SMALL_FULL_PARSE_PROGRESS_MAX_FILES as _SMALL_FULL_PARSE_PROGRESS_MAX_FILES,
)
from polylogue.sources.live.convergence_debt import (
    ConvergenceDebt,
    convergence_debt_from_state,
    convergence_debt_from_states,
    debt_by_path,
)
from polylogue.sources.live.convergence_outcome import record_convergence_outcome
from polylogue.sources.live.cursor import CursorRecord, CursorStore
from polylogue.sources.live.dedup import handle_schema_version_mismatch, handle_structural_database_error
from polylogue.sources.live.deferred_cursor import record_deferred_append_cursor
from polylogue.sources.live.metrics import LiveBatchMetrics, LiveFullIngestAggregate
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.sources.parsers import hermes_state
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    iter_zip_entry_raw_data,
)
from polylogue.sources.sqlite_snapshot import (
    hermes_profile_raw_id,
    original_sqlite_source_path,
    snapshot_sqlite_to_blob,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_active_archive_root as initialize_archive_root,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)

LiveBatchEventEmitter = Callable[[str, dict[str, object]], None]
_ARCHIVE_RUNTIME_TIERS = ",".join(spec.tier.value for spec in ARCHIVE_TIER_SPECS.values())
_ARCHIVE_NATIVE_WRITE_TIERS = "source,index"


def _single_route_stage_payload(*, append_file_count: int, full_file_count: int) -> dict[str, object] | None:
    if append_file_count > 0 and full_file_count == 0:
        return {"storage_route": "archive_append"}
    if full_file_count > 0 and append_file_count == 0:
        return {
            "storage_route": "archive_full",
            "storage_tiers": _ARCHIVE_RUNTIME_TIERS,
            "storage_write_tiers": _ARCHIVE_NATIVE_WRITE_TIERS,
        }
    return None


def _iso_to_epoch_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


@dataclass(slots=True)
class _ArchiveFullWriteResult:
    raw_ids: dict[str, str] = field(default_factory=dict)
    session_ids: list[str] = field(default_factory=list)
    session_count: int = 0
    message_count: int = 0
    stage_timings_s: dict[str, float] = field(default_factory=dict)


class LiveBatchProcessor:
    """Run the daemon live ingest batch path without filesystem watching."""

    def __init__(
        self,
        polylogue: Polylogue,
        sources: Iterable[Any],
        *,
        cursor: CursorStore,
        parser_fingerprint: str | Callable[[], str],
        converger: object | None = None,
        stop_requested: Callable[[], bool] | None = None,
        event_emitter: LiveBatchEventEmitter | None = None,
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._cursor = cursor
        self._parser_fingerprint = parser_fingerprint
        self._converger = converger
        self._stop_requested = stop_requested or (lambda: False)
        self._event_emitter = event_emitter
        self._last_cursor_write_stale = False
        self._raw_compaction_min_acquired_at = datetime.now(UTC).isoformat()

    async def ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
        emit_event: bool = True,
    ) -> LiveBatchMetrics:
        """Ingest files in batch, run post-ingest convergence, and return metrics."""
        if is_degraded():
            # The daemon has been marked structurally unable to ingest (e.g.
            # schema mismatch detected at preflight or on the first batch).
            # Do not enter the full-parse path — that is what produced the
            # IOPS storm in #1003.
            return self._degraded_skip_metrics(paths, queued_file_count, skipped_file_count)
        batch_started = time.perf_counter()
        db_bytes_before = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        input_bytes = sum(_path_size(path) for path in paths)
        attempt_id = self._cursor.begin_ingest_attempt(
            paths=paths,
            input_bytes=input_bytes,
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
        )
        self._record_attempt_progress(
            attempt_id,
            phase="planning",
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count,
            input_bytes=input_bytes,
            succeeded_file_count=0,
            failed_file_count=0,
            source_payload_read_bytes=0,
            cursor_fingerprint_read_bytes=0,
            parse_time_s=0.0,
            convergence_time_s=0.0,
            total_time_s=0.0,
        )
        source_payload_read_bytes = 0
        cursor_fingerprint_read_bytes = 0
        stale_cursor_write_count = 0
        parse_time_s = 0.0
        convergence_time_s = 0.0
        stage_timings: dict[str, float] = {}
        failed_paths: list[str] = []
        succeeded_paths: set[Path] = set()
        ingest_worker_count_max = 0
        full_ingest_aggregate = LiveFullIngestAggregate()
        cursor_records = self._cursor.get_records(paths)
        append_file_count = 0
        pending_append_plans: list[_AppendPlan] = []
        full_paths: list[Path] = []
        deferred_paths: list[Path] = []

        async def flush_append_plans() -> None:
            nonlocal convergence_time_s
            nonlocal cursor_fingerprint_read_bytes
            nonlocal ingest_worker_count_max
            nonlocal parse_time_s
            nonlocal pending_append_plans
            nonlocal stale_cursor_write_count
            if not pending_append_plans:
                return
            plans = pending_append_plans
            pending_append_plans = []
            self._record_attempt_progress(
                attempt_id,
                phase="append_parse",
                succeeded_file_count=len(succeeded_paths),
                failed_file_count=len(failed_paths),
                source_payload_read_bytes=source_payload_read_bytes,
                cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                parse_time_s=parse_time_s,
                current_source=plans[0].source_name,
                current_path=plans[0].path,
                stage_payload={"storage_route": "archive_append"},
            )
            t0 = time.perf_counter()
            try:
                append_result = await asyncio.to_thread(self._ingest_append_plans, plans)
            except SchemaVersionMismatchError as exc:
                handle_schema_version_mismatch(plans[0].source_name, exc)
                for plan in plans:
                    failed_paths.append(str(plan.path))
                # Use an empty result so the per-plan cleanup loop below
                # (``for plan in append_result.failed``) does NOT re-push the
                # same paths into ``failed_paths`` and does NOT call
                # ``_record_failed_cursor`` against the DB we already know is
                # structurally unusable. The by_source loop further down also
                # checks ``is_degraded()`` and skips the full-parse phase.
                append_result = _AppendResult(succeeded=[], failed=[], worker_count=0)
            ingest_worker_count_max = max(ingest_worker_count_max, append_result.worker_count)
            parse_time_s += time.perf_counter() - t0
            _accumulate_stage_timings(stage_timings, append_result.stage_timings_s)
            append_stage_payload: dict[str, object] = {
                "storage_route": "archive_append",
                "append_stage_timings_s": {
                    name: round(elapsed, 6) for name, elapsed in append_result.stage_timings_s.items()
                },
            }
            release_process_memory()
            self._record_attempt_progress(
                attempt_id,
                phase="convergence",
                succeeded_file_count=len(succeeded_paths),
                failed_file_count=len(failed_paths),
                source_payload_read_bytes=source_payload_read_bytes,
                cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                parse_time_s=parse_time_s,
                convergence_time_s=convergence_time_s,
                current_source=plans[0].source_name,
                current_path=plans[0].path,
                stage_payload=append_stage_payload,
            )
            _converged_paths, elapsed, timings, convergence_debt = await asyncio.to_thread(
                self._converge_paths,
                [plan.path for plan in append_result.succeeded],
            )
            convergence_time_s += elapsed
            release_process_memory()
            _accumulate_stage_timings(stage_timings, timings)
            debt_by_source_path = debt_by_path(convergence_debt)
            for plan in append_result.succeeded:
                succeeded_paths.add(plan.path)
                if not self._record_append_cursor(plan):
                    stale_cursor_write_count += 1
                self._record_convergence_outcome(plan.path, debt_by_source_path.get(plan.path, ()))
            for plan in append_result.failed:
                failed_paths.append(str(plan.path))
                cursor_fingerprint_read_bytes += self._record_failed_cursor(plan.path)

        for path in paths:
            if is_degraded():
                full_paths.append(path)
                continue
            cursor = cursor_records.get(path)
            append_plan = self._append_plan(path, cursor=cursor) if self._can_ingest_appends_directly() else None
            if isinstance(append_plan, _DeferredAppend):
                cursor_fingerprint_read_bytes += record_deferred_append_cursor(
                    self._cursor,
                    path,
                    cursor=cursor,
                    parser_fingerprint=self._current_parser_fingerprint(),
                    source_name=self._source_name_for(path),
                )
                deferred_paths.append(path)
            elif append_plan is None:
                full_paths.append(path)
            else:
                pending_append_plans.append(append_plan)
                append_file_count += 1
                source_payload_read_bytes += append_plan.bytes_read
                if _append_plan_group_ready(pending_append_plans):
                    await flush_append_plans()
        await flush_append_plans()

        by_source: dict[str, list[Path]] = {}
        for path in full_paths:
            by_source.setdefault(self._source_name_for(path), []).append(path)

        for source_name, grouped_paths in by_source.items():
            if is_degraded():
                # A prior source group hit a structural error this batch.
                # Don't burn IOPS on remaining groups.
                for path in grouped_paths:
                    failed_paths.append(str(path))
                continue
            for source_paths in _full_parse_progress_groups(grouped_paths):
                if self._stop_requested():
                    break
                if is_degraded():
                    break
                t0 = time.perf_counter()
                try:
                    self._record_attempt_progress(
                        attempt_id,
                        phase="full_parse",
                        succeeded_file_count=len(succeeded_paths),
                        failed_file_count=len(failed_paths),
                        source_payload_read_bytes=source_payload_read_bytes,
                        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                        parse_time_s=parse_time_s,
                        current_source=source_name,
                        current_path=source_paths[0] if source_paths else None,
                    )
                    current_path = source_paths[0] if source_paths else None
                    full_result = await self._ingest_full_paths(
                        source_paths,
                        source_name=source_name,
                        attempt_id=attempt_id,
                        heartbeat=self._full_ingest_heartbeat(
                            attempt_id,
                            source_name=source_name,
                            current_path=current_path,
                            succeeded_file_count=len(succeeded_paths),
                            failed_file_count=len(failed_paths),
                            source_payload_read_bytes=source_payload_read_bytes,
                            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                            parse_time_s=parse_time_s,
                            convergence_time_s=convergence_time_s,
                        ),
                    )
                    ingest_worker_count_max = max(ingest_worker_count_max, full_result.worker_count)
                    full_ingest_aggregate.add(full_result)
                except SchemaVersionMismatchError as exc:
                    handle_schema_version_mismatch(source_name, exc)
                    # Account for every queued path in this source group, not
                    # only the current progress chunk — later chunks would
                    # hit the same structural error with no information gain.
                    for path in grouped_paths:
                        failed_paths.append(str(path))
                    self._record_attempt_progress(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths),
                        failed_file_count=len(failed_paths),
                        source_payload_read_bytes=source_payload_read_bytes,
                        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                        parse_time_s=parse_time_s,
                        current_source=source_name,
                        current_path=source_paths[0] if source_paths else None,
                        error=str(exc),
                    )
                    # Stop processing this batch entirely — every remaining
                    # source group would hit the same structural error.
                    break
                except DatabaseError as exc:
                    handle_structural_database_error(source_name, exc)
                    for path in grouped_paths:
                        failed_paths.append(str(path))
                    self._record_attempt_progress(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths),
                        failed_file_count=len(failed_paths),
                        source_payload_read_bytes=source_payload_read_bytes,
                        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                        parse_time_s=parse_time_s,
                        current_source=source_name,
                        current_path=source_paths[0] if source_paths else None,
                        error=str(exc),
                    )
                    break
                except Exception as exc:
                    logger.warning("live.watcher: batch failed for %s: %s", source_name, exc)
                    for path in source_paths:
                        failed_paths.append(str(path))
                        cursor_fingerprint_read_bytes += self._record_failed_cursor(path)
                    self._record_attempt_progress(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths),
                        failed_file_count=len(failed_paths),
                        source_payload_read_bytes=source_payload_read_bytes,
                        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                        parse_time_s=parse_time_s,
                        current_source=source_name,
                        current_path=source_paths[0] if source_paths else None,
                        error=str(exc),
                    )
                    continue
                parse_elapsed = time.perf_counter() - t0
                parse_time_s += parse_elapsed
                source_payload_read_bytes += full_result.source_payload_read_bytes
                _accumulate_stage_timings(stage_timings, full_result.stage_timings_s)
                release_process_memory()
                self._record_attempt_progress(
                    attempt_id,
                    phase="convergence",
                    succeeded_file_count=len(succeeded_paths),
                    failed_file_count=len(failed_paths),
                    source_payload_read_bytes=source_payload_read_bytes,
                    cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                    parse_time_s=parse_time_s,
                    convergence_time_s=convergence_time_s,
                    current_source=source_name,
                    current_path=source_paths[0] if source_paths else None,
                )
                _converged_paths, elapsed, timings, convergence_debt = await asyncio.to_thread(
                    self._converge_paths,
                    full_result.succeeded,
                )
                convergence_time_s += elapsed
                release_process_memory()
                _accumulate_stage_timings(stage_timings, timings)
                debt_by_source_path = debt_by_path(convergence_debt)
                for path in full_result.succeeded:
                    succeeded_paths.add(path)
                    cursor_fingerprint_read_bytes += self._record_full_cursor(
                        path,
                        raw_fingerprint=full_result.raw_fingerprints.get(path),
                        raw_byte_size=full_result.raw_byte_sizes.get(path),
                        source_name=full_result.raw_source_names.get(path),
                        source_revision=full_result.raw_source_revisions.get(path),
                    )
                    if self._last_cursor_write_stale:
                        stale_cursor_write_count += 1
                    self._record_convergence_outcome(path, debt_by_source_path.get(path, ()))
                for path in full_result.failed:
                    failed_paths.append(str(path))
                    cursor_fingerprint_read_bytes += self._record_failed_cursor(path)
                logger.info(
                    "live.watcher: batch ingested %s — %d in %.1fs (%.1f/s)",
                    source_name,
                    len(full_result.succeeded),
                    parse_elapsed,
                    len(full_result.succeeded) / max(parse_elapsed, 0.01),
                )

        summary_stage_payload = _single_route_stage_payload(
            append_file_count=append_file_count,
            full_file_count=len(full_paths),
        )
        self._record_attempt_progress(
            attempt_id,
            phase="cursor_update",
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths) + len(deferred_paths),
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            parse_time_s=parse_time_s,
            convergence_time_s=convergence_time_s,
            stale_cursor_write_count=stale_cursor_write_count,
            stage_payload=summary_stage_payload,
        )

        if succeeded_paths:
            await asyncio.to_thread(self._compact_superseded_raw_snapshots, sorted(succeeded_paths))

        retry_paths = failed_paths + [str(path) for path in deferred_paths]
        db_bytes_after = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        metrics = LiveBatchMetrics(
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count,
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths),
            source_group_count=len({self._source_name_for(path) for path in paths}),
            input_bytes=input_bytes,
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            ingest_worker_count_max=ingest_worker_count_max,
            append_file_count=append_file_count,
            full_file_count=len(full_paths),
            archive_bytes_before=db_bytes_before,
            archive_bytes_after=db_bytes_after,
            archive_write_bytes_delta=max(0, db_bytes_after - db_bytes_before),
            parse_time_s=round(parse_time_s, 6),
            convergence_time_s=round(convergence_time_s, 6),
            total_time_s=round(time.perf_counter() - batch_started, 6),
            **full_ingest_aggregate.to_metric_kwargs(),
            rss_current_mb=read_current_rss_mb(),
            rss_peak_self_mb=read_peak_rss_self_mb(),
            rss_peak_children_mb=read_peak_rss_children_mb(),
            cgroup_path=read_cgroup_path(),
            cgroup_memory_current_mb=read_cgroup_memory_current_mb(),
            cgroup_memory_peak_mb=read_cgroup_memory_peak_mb(),
            cgroup_memory_swap_current_mb=read_cgroup_memory_swap_current_mb(),
            stale_cursor_write_count=stale_cursor_write_count,
            stage_timings_s={name: round(elapsed, 6) for name, elapsed in stage_timings.items()},
            failed_paths=retry_paths,
        )
        if emit_event and self._event_emitter is not None:
            self._event_emitter("ingestion_batch", metrics.to_payload())
        self._record_attempt_progress(
            attempt_id,
            phase="completed",
            status="completed",
            queued_file_count=metrics.queued_file_count,
            needed_file_count=metrics.needed_file_count,
            skipped_file_count=metrics.skipped_file_count,
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths) + len(deferred_paths),
            input_bytes=input_bytes,
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            archive_write_bytes_delta=metrics.archive_write_bytes_delta,
            parse_time_s=parse_time_s,
            convergence_time_s=convergence_time_s,
            total_time_s=metrics.total_time_s,
            stage_timings_s=metrics.stage_timings_s,
            stale_cursor_write_count=stale_cursor_write_count,
            stage_payload=summary_stage_payload,
        )
        self._cursor.finish_ingest_attempt(
            attempt_id,
            status="completed" if not retry_paths else "completed_with_failures",
            phase="completed",
            error="; ".join(retry_paths[:3]) if retry_paths else None,
        )
        return metrics

    def _degraded_skip_metrics(
        self,
        paths: list[Path],
        queued_file_count: int | None,
        skipped_file_count: int,
    ) -> LiveBatchMetrics:
        """Empty-ingest metrics for the degraded short-circuit path."""
        return LiveBatchMetrics(
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count + len(paths),
            succeeded_file_count=0,
            failed_file_count=0,
            source_group_count=len({self._source_name_for(path) for path in paths}),
            input_bytes=0,
            source_payload_read_bytes=0,
            cursor_fingerprint_read_bytes=0,
            ingest_worker_count_max=0,
            append_file_count=0,
            full_file_count=0,
            archive_bytes_before=0,
            archive_bytes_after=0,
            archive_write_bytes_delta=0,
            parse_time_s=0.0,
            convergence_time_s=0.0,
            total_time_s=0.0,
            rss_current_mb=read_current_rss_mb(),
            rss_peak_self_mb=read_peak_rss_self_mb(),
            rss_peak_children_mb=read_peak_rss_children_mb(),
            cgroup_path=read_cgroup_path(),
            cgroup_memory_current_mb=read_cgroup_memory_current_mb(),
            cgroup_memory_peak_mb=read_cgroup_memory_peak_mb(),
            cgroup_memory_swap_current_mb=read_cgroup_memory_swap_current_mb(),
            stale_cursor_write_count=0,
            stage_timings_s={},
            failed_paths=[],
        )

    def _record_attempt_progress(self, attempt_id: str, **kwargs: Any) -> None:
        record_attempt_progress(self._cursor, attempt_id, **kwargs)

    def _full_ingest_heartbeat(
        self,
        attempt_id: str,
        *,
        source_name: str,
        current_path: Path | None,
        succeeded_file_count: int,
        failed_file_count: int,
        source_payload_read_bytes: int,
        cursor_fingerprint_read_bytes: int,
        parse_time_s: float,
        convergence_time_s: float,
    ) -> _FullIngestHeartbeat:
        def emit(
            phase: str,
            *,
            current_path_override: Path | None = None,
            payload_read_bytes: int | None = None,
            stage_payload: dict[str, object] | None = None,
        ) -> None:
            self._record_attempt_progress(
                attempt_id,
                phase=phase,
                succeeded_file_count=succeeded_file_count,
                failed_file_count=failed_file_count,
                source_payload_read_bytes=(
                    source_payload_read_bytes if payload_read_bytes is None else payload_read_bytes
                ),
                cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                parse_time_s=parse_time_s,
                convergence_time_s=convergence_time_s,
                current_source=source_name,
                current_path=current_path if current_path_override is None else current_path_override,
                stage_payload=stage_payload,
            )

        return _throttled_phase_heartbeat(emit)

    def _record_failed_cursor(self, path: Path) -> int:
        try:
            stat = path.stat()
            fp, last_nl = fingerprint_file(path)
            tail_hash, _tail_bytes = tail_hash_from_path(path, stat.st_size)
        except FileNotFoundError:
            try:
                self._cursor.mark_failed(path)
            except sqlite3.OperationalError as exc:
                if not is_transient_sqlite_lock(exc):
                    raise
                logger.warning("live.watcher: skipped failed-cursor mark for missing file %s: %s", path, exc)
            return 0
        try:
            existing = self._cursor.get_record(path)
            self._cursor.set(
                path,
                stat.st_size,
                byte_offset=last_nl,
                last_complete_newline=last_nl,
                parser_fingerprint=self._current_parser_fingerprint(),
                content_fingerprint=fp,
                tail_hash=tail_hash,
                source_name=self._source_name_for(path),
                st_dev=stat.st_dev,
                st_ino=stat.st_ino,
                mtime_ns=stat.st_mtime_ns,
                failure_count=existing.failure_count if existing else 0,
                next_retry_at=existing.next_retry_at if existing else None,
                excluded=bool(existing.excluded) if existing else False,
            )
            self._cursor.mark_failed(path)
        except sqlite3.OperationalError as exc:
            if not is_transient_sqlite_lock(exc):
                raise
            logger.warning("live.watcher: skipped failed-cursor bookkeeping for %s: %s", path, exc)
        return stat.st_size

    def _record_full_cursor(
        self,
        path: Path,
        *,
        raw_fingerprint: str | None = None,
        raw_byte_size: int | None = None,
        source_name: str | None = None,
        source_revision: str | None = None,
    ) -> int:
        self._last_cursor_write_stale = False
        try:
            stat = path.stat()
        except FileNotFoundError:
            return 0
        raw_fingerprint = raw_fingerprint or self._latest_raw_fingerprint(path)
        if source_revision is not None:
            byte_size = stat.st_size
            fp = raw_fingerprint or ""
            last_nl = byte_size
            tail_hash = source_revision
            bytes_read = 0
        else:
            byte_size = stat.st_size if raw_byte_size is None else raw_byte_size
            fp, last_nl, tail_hash, bytes_read = cursor_state_after_full_ingest(
                path,
                byte_size,
                raw_fingerprint=raw_fingerprint,
            )
        updated = self._cursor.set(
            path,
            byte_size,
            byte_offset=last_nl,
            last_complete_newline=last_nl,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=fp,
            tail_hash=tail_hash,
            source_name=source_name or self._source_name_for(path),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            allow_backward=stat.st_size <= byte_size,
        )
        self._last_cursor_write_stale = not updated
        self._cursor.reset_failures(path)
        return bytes_read

    def _record_convergence_outcome(self, path: Path, debts: Iterable[ConvergenceDebt]) -> None:
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        record_convergence_outcome(self._cursor, path, debts, archive_root=archive_root)

    def _converge_paths(
        self, paths: Iterable[Path]
    ) -> tuple[set[Path], float, dict[str, float], list[ConvergenceDebt]]:
        unique_paths = tuple(sorted(dict.fromkeys(paths)))
        if not unique_paths:
            return set(), 0.0, {}, []
        if self._converger is None:
            return set(unique_paths), 0.0, {}, []

        started = time.perf_counter()
        try:
            converge_batch = getattr(self._converger, "converge_batch", None)
            if callable(converge_batch):
                states, timings = converge_batch(unique_paths)
                batch_completed = {
                    path for path in unique_paths if path in states and bool(getattr(states[path], "converged", False))
                }
                debt_items = convergence_debt_from_states(unique_paths, states)
                # #1654: after convergence, check for new hook events
                # that carry paste evidence and update matching messages.
                try:
                    from polylogue.sources.live.hook_paste_enrichment import enrich_paste_from_hooks

                    enrich_paste_from_hooks(self._cursor._db_path)
                except Exception:
                    logger.debug("hook_paste: enrichment failed (non-fatal)", exc_info=True)
                return (
                    batch_completed,
                    time.perf_counter() - started,
                    {stage_name: float(elapsed) for stage_name, elapsed in timings.items()},
                    debt_items,
                )

            per_file_completed: set[Path] = set()
            stage_timings: dict[str, float] = {}
            per_file_debt_items: list[ConvergenceDebt] = []
            for path in unique_paths:
                invalidate = getattr(self._converger, "invalidate_file", None)
                if callable(invalidate):
                    invalidate(path)
                state = self._converger.converge_file(path)  # type: ignore[attr-defined]
                for stage_name, elapsed in getattr(state, "last_stage_times", {}).items():
                    stage_timings[stage_name] = stage_timings.get(stage_name, 0.0) + float(elapsed)
                if bool(getattr(state, "converged", False)):
                    per_file_completed.add(path)
                else:
                    per_file_debt_items.extend(convergence_debt_from_state(path, state))
            return per_file_completed, time.perf_counter() - started, stage_timings, per_file_debt_items
        except Exception as exc:
            logger.warning("live.watcher: post-ingest converge failed: %s", exc)
            return (
                set(),
                time.perf_counter() - started,
                {},
                [ConvergenceDebt(path=path, stage="convergence", error=str(exc)) for path in unique_paths],
            )

    def _latest_raw_fingerprint(self, path: Path) -> str | None:
        return self._latest_archive_tiers_raw_fingerprint(path)

    def _latest_archive_tiers_raw_fingerprint(self, path: Path) -> str | None:
        source_db = self._cursor._db_path.with_name("source.db")
        if not source_db.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    """
                    SELECT raw_id, blob_hash
                    FROM raw_sessions
                    WHERE source_path = ?
                      AND COALESCE(source_index, 0) >= 0
                    ORDER BY acquired_at_ms DESC, raw_id DESC
                    LIMIT 1
                    """,
                    (str(path),),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        raw_id, blob_hash = row
        if isinstance(blob_hash, bytes):
            blob_hash_hex = blob_hash.hex()
        elif isinstance(blob_hash, str):
            blob_hash_hex = blob_hash.lower()
        else:
            return None
        if not _archive_blob_exists(source_db.parent, blob_hash_hex):
            return None
        return raw_id if isinstance(raw_id, str) and raw_id else None

    def _current_parser_fingerprint(self) -> str:
        if callable(self._parser_fingerprint):
            return self._parser_fingerprint()
        return self._parser_fingerprint

    def _source_name_for(self, path: Path) -> str:
        resolved = path.resolve()
        for source in self._sources:
            try:
                if resolved.is_relative_to(source.root.resolve()):
                    return str(source.name)
            except OSError:
                continue
        return path.parent.name

    def _can_ingest_appends_directly(self) -> bool:
        backend = getattr(self._polylogue, "backend", None)
        return isinstance(getattr(backend, "db_path", None), Path)

    async def _ingest_full_paths(
        self,
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: _FullIngestHeartbeat | None = None,
        attempt_id: str | None = None,
    ) -> _FullIngestResult:
        return await asyncio.to_thread(
            self._ingest_full_paths_sync, paths, source_name=source_name, heartbeat=heartbeat, attempt_id=attempt_id
        )

    def _ingest_full_paths_sync(
        self,
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: _FullIngestHeartbeat | None = None,
        attempt_id: str | None = None,
    ) -> _FullIngestResult:
        if not paths:
            return _FullIngestResult(succeeded=[], failed=[], source_payload_read_bytes=0)
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        blob_root = archive_root / "blob"
        from polylogue.storage.blob_publication import ArchiveBlobPublisher

        blob_store = ArchiveBlobPublisher(archive_root / "source.db", blob_root)
        raw_records: list[RawSessionRecord] = []
        raw_by_id: dict[str, Path] = {}
        raw_byte_sizes: dict[Path, int] = {}
        raw_payloads: dict[str, bytes] = {}
        raw_source_names: dict[Path, str] = {}
        raw_source_revisions: dict[Path, str] = {}
        failed: list[Path] = []
        ingested: list[Path] = []
        source_payload_read_bytes = 0
        fallback_provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))

        archive_bootstrapped = not self._archive_active(archive_root)
        if archive_bootstrapped:
            initialize_archive_root(archive_root)
        archive_active = self._archive_active(archive_root)
        if heartbeat is not None:
            heartbeat(
                "full_archive_storage_probe",
                current_path=paths[0],
                source_payload_read_bytes=0,
                stage_payload=self._archive_storage_probe_payload(
                    archive_root,
                    archive_active=archive_active,
                    archive_bootstrapped=archive_bootstrapped,
                ),
                force=True,
            )

        for path in paths:
            blob_hash: str | None = None
            blob_publication_receipt_id: str | None = None
            try:
                stat = path.stat()
            except OSError:
                failed.append(path)
                continue
            if heartbeat is not None:
                heartbeat(
                    "full_file_scan",
                    current_path=path,
                    source_payload_read_bytes=source_payload_read_bytes,
                )
            if path.suffix.lower() == ".zip":
                file_mtime = datetime.fromtimestamp(stat.st_mtime_ns / 1_000_000_000, UTC).isoformat()
                zip_records, zip_bytes = self._extract_zip_member_records(
                    path,
                    blob_store=blob_store,
                    fallback_provider=fallback_provider,
                    file_mtime=file_mtime,
                )
                if not zip_records:
                    self._mark_excluded_cursor(path, stat, source_name=fallback_provider.value)
                    continue
                for member_raw_id, member_record in zip_records:
                    raw_records.append(member_record)
                    raw_by_id[member_raw_id] = path
                source_payload_read_bytes += zip_bytes
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
                ingested.append(path)
                raw_byte_sizes[path] = stat.st_size
                continue
            if hermes_state.looks_like_state_db_path(path):
                provider = Provider.HERMES
                source_name = provider.value
                try:
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
                    snapshot = snapshot_sqlite_to_blob(
                        path,
                        blob_store,
                        heartbeat=_blob_copy_heartbeat(
                            heartbeat,
                            path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        ),
                    )
                    blob_hash, blob_size = snapshot.blob_hash, snapshot.blob_size
                    blob_publication_receipt_id = snapshot.blob_publication_receipt_id
                    source_path = original_sqlite_source_path(path) or path
                    raw_id = hermes_profile_raw_id(source_path, 0, blob_hash)
                    raw_source_revisions[path] = snapshot.source_revision
                except OSError:
                    failed.append(path)
                    continue
                source_payload_read_bytes += blob_size
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            elif path.suffix.lower() == ".jsonl":
                provider, parse_as_session = _jsonl_provider_and_session_artifact(path, fallback_provider)
                source_name = provider.value
                if not parse_as_session:
                    self._mark_excluded_cursor(path, stat, source_name=source_name)
                    continue
                if stat.st_size >= _STREAMING_FULL_INGEST_BYTES:
                    try:
                        if heartbeat is not None:
                            heartbeat(
                                "full_blob_copy",
                                current_path=path,
                                source_payload_read_bytes=source_payload_read_bytes,
                            )
                        raw_id, blob_size = blob_store.write_from_path(
                            path,
                            heartbeat=_blob_copy_heartbeat(
                                heartbeat,
                                path=path,
                                source_payload_read_bytes=source_payload_read_bytes,
                            ),
                        )
                        blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                    except OSError:
                        failed.append(path)
                        continue
                    source_payload_read_bytes += blob_size
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
                else:
                    try:
                        payload = path.read_bytes()
                    except OSError:
                        failed.append(path)
                        continue
                    raw_id, blob_size = blob_store.write_from_bytes(payload)
                    blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                    raw_payloads[raw_id] = payload
                    source_payload_read_bytes += len(payload)
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
            elif source_name == "browser-capture" and path.suffix.lower() == ".json":
                try:
                    payload = path.read_bytes()
                except OSError:
                    failed.append(path)
                    continue
                provider = _detect_provider_from_raw_bytes(payload, path.name, fallback_provider)
                source_name = provider.value
                if not _parse_payload_as_session_artifact(path, provider=provider, payload=payload):
                    self._mark_excluded_cursor(path, stat, source_name=source_name)
                    continue
                raw_id, blob_size = blob_store.write_from_bytes(payload)
                blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                raw_payloads[raw_id] = payload
                source_payload_read_bytes += len(payload)
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            elif stat.st_size >= _STREAMING_FULL_INGEST_BYTES:
                provider = _detect_provider_from_path_sample(path, fallback_provider)
                source_name = provider.value
                if not _parse_path_as_session_artifact(path, provider=provider):
                    self._mark_excluded_cursor(path, stat, source_name=source_name)
                    continue
                try:
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
                    raw_id, blob_size = blob_store.write_from_path(
                        path,
                        heartbeat=_blob_copy_heartbeat(
                            heartbeat,
                            path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        ),
                    )
                    blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                except OSError:
                    failed.append(path)
                    continue
                source_payload_read_bytes += blob_size
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            else:
                try:
                    payload = path.read_bytes()
                except OSError:
                    failed.append(path)
                    continue
                provider = _detect_provider_from_raw_bytes(payload, path.name, fallback_provider)
                source_name = provider.value
                if not _parse_payload_as_session_artifact(path, provider=provider, payload=payload):
                    self._mark_excluded_cursor(path, stat, source_name=source_name)
                    continue
                raw_id, blob_size = blob_store.write_from_bytes(payload)
                blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                raw_payloads[raw_id] = payload
                source_payload_read_bytes += len(payload)
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            ingested.append(path)
            raw_byte_sizes[path] = stat.st_size
            raw_source_names[path] = source_name
            raw_records.append(
                RawSessionRecord(
                    raw_id=raw_id,
                    blob_hash=(blob_hash if provider is Provider.HERMES and blob_hash is not None else None),
                    payload_provider=provider,
                    source_name=source_name,
                    source_path=(
                        str(original_sqlite_source_path(path) or path) if path in raw_source_revisions else str(path)
                    ),
                    source_index=0,
                    blob_size=blob_size,
                    blob_publication_receipt_id=blob_publication_receipt_id,
                    acquired_at=datetime.now(UTC).isoformat(),
                    file_mtime=datetime.fromtimestamp(stat.st_mtime_ns / 1_000_000_000, UTC).isoformat(),
                )
            )
            raw_by_id[raw_id] = path

        summary: _IngestBatchSummary | None = None
        if raw_records:
            blob_store.flush()
            available_records = [record for record in raw_records if record.raw_id in raw_payloads]
            missing_payload_records = [record for record in raw_records if record.raw_id not in raw_payloads]
            if heartbeat is not None:
                heartbeat(
                    "full_archive_write",
                    current_path=ingested[-1] if ingested else None,
                    source_payload_read_bytes=source_payload_read_bytes,
                    stage_payload={
                        "storage_route": "archive_full",
                        "storage_tiers": _ARCHIVE_RUNTIME_TIERS,
                        "storage_write_tiers": _ARCHIVE_NATIVE_WRITE_TIERS,
                        "input_file_count": len(raw_records),
                        "payload_available_file_count": len(available_records),
                        "payload_unavailable_file_count": len(missing_payload_records),
                        "payload_replayed_from_blob_file_count": len(missing_payload_records),
                    },
                    force=True,
                )
            archive_write = self._ingest_full_records_archive(raw_records, raw_payloads, blob_store)
            failed.extend(raw_by_id[raw_id] for raw_id in raw_by_id if raw_id not in archive_write.raw_ids)
            raw_by_id = {archive_write.raw_ids.get(raw_id, raw_id): path for raw_id, path in raw_by_id.items()}
            if heartbeat is not None:
                heartbeat(
                    "full_archive_write_completed",
                    current_path=ingested[-1] if ingested else None,
                    source_payload_read_bytes=source_payload_read_bytes,
                    stage_payload={
                        "storage_route": "archive_full",
                        "storage_tiers": _ARCHIVE_RUNTIME_TIERS,
                        "storage_write_tiers": _ARCHIVE_NATIVE_WRITE_TIERS,
                        "written_raw_count": len(archive_write.raw_ids),
                        "ingested_session_count": archive_write.session_count,
                        "ingested_message_count": archive_write.message_count,
                        "payload_unavailable_file_count": len(missing_payload_records),
                        "payload_replayed_from_blob_file_count": len(missing_payload_records),
                    },
                    force=True,
                )
            summary = _IngestBatchSummary(
                worker_count=1,
                total_convos=archive_write.session_count,
                total_msgs=archive_write.message_count,
                changed_session_ids=archive_write.session_ids,
                stage_timings_s=archive_write.stage_timings_s,
            )

        failed_set = set(failed)
        raw_fingerprints = {path: raw_id for raw_id, path in raw_by_id.items()}
        result = _full_ingest_result_from_summary(
            succeeded=[path for path in ingested if path not in failed_set],
            failed=failed,
            source_payload_read_bytes=source_payload_read_bytes,
            raw_fingerprints=raw_fingerprints,
            raw_byte_sizes=raw_byte_sizes,
            raw_source_names=raw_source_names,
            raw_source_revisions=raw_source_revisions,
            summary=summary,
        )
        raw_records.clear()
        raw_by_id.clear()
        blob_store.discard_pending()
        return result

    def _archive_active(self, archive_root: Path) -> bool:
        return (archive_root / "index.db").exists() and (archive_root / "source.db").exists()

    def _archive_storage_probe_payload(
        self,
        archive_root: Path,
        *,
        archive_active: bool,
        archive_bootstrapped: bool,
    ) -> dict[str, object]:
        tier_paths = {spec.tier.value: archive_root / spec.filename for spec in ARCHIVE_TIER_SPECS.values()}
        present = [tier for tier, path in tier_paths.items() if path.exists()]
        missing = [tier for tier, path in tier_paths.items() if not path.exists()]
        user_versions: dict[str, int | None] = {}
        for tier, path in tier_paths.items():
            if not path.exists():
                user_versions[tier] = None
                continue
            try:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    user_versions[tier] = int(conn.execute("PRAGMA user_version").fetchone()[0])
                finally:
                    conn.close()
            except sqlite3.Error:
                user_versions[tier] = -1
        return {
            "storage_route": "archive_full",
            "storage_tiers": _ARCHIVE_RUNTIME_TIERS,
            "storage_write_tiers": _ARCHIVE_NATIVE_WRITE_TIERS,
            "archive_active": archive_active,
            "archive_bootstrapped": archive_bootstrapped,
            "archive_present_tiers": ",".join(present),
            "archive_missing_tiers": ",".join(missing),
            "archive_tier_user_versions_json": json_dumps(user_versions, sort_keys=True),
        }

    def _ingest_full_records_archive(
        self,
        records: list[RawSessionRecord],
        raw_payloads: dict[str, bytes],
        blob_store: BlobStore,
    ) -> _ArchiveFullWriteResult:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        result = _ArchiveFullWriteResult()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            for record in records:
                provider: Provider | None = None
                source_raw_id: str | None = None
                try:
                    record_timings: dict[str, float] = {}
                    t0 = time.perf_counter()
                    provider = record.payload_provider or Provider.from_string(record.source_name)
                    payload = raw_payloads.get(record.raw_id)
                    source_name = Path(record.source_path).name
                    fallback_id = Path(record.source_path).stem
                    blob_hash = record.blob_hash or record.raw_id
                    acquired_at_ms = _iso_to_epoch_ms(record.acquired_at)
                    source_write_started = time.perf_counter()
                    if payload is None:
                        source_raw_id = archive.write_raw_blob_ref(
                            provider=provider,
                            blob_hash_hex=blob_hash,
                            blob_size=record.blob_size,
                            source_path=record.source_path,
                            source_index=record.source_index or 0,
                            raw_id=(record.raw_id if provider is Provider.HERMES else None),
                            acquired_at_ms=acquired_at_ms,
                            blob_publication_receipt_id=record.blob_publication_receipt_id,
                        )
                        source_write_name = "full.source_raw_blob_ref_write"
                    else:
                        source_raw_id = archive.write_raw_payload(
                            provider=provider,
                            payload=payload,
                            source_path=record.source_path,
                            source_index=record.source_index or 0,
                            acquired_at_ms=acquired_at_ms,
                            blob_publication_receipt_id=record.blob_publication_receipt_id,
                        )
                        source_write_name = "full.source_raw_write"
                    record_timings[source_write_name] = time.perf_counter() - source_write_started
                    t0 = time.perf_counter()
                    if provider is Provider.HERMES and hermes_state.looks_like_state_db_path(
                        blob_store.blob_path(blob_hash)
                    ):
                        sessions = hermes_state.parse_state_db(
                            blob_store.blob_path(blob_hash),
                            fallback_id=fallback_id,
                            profile_root=Path(record.source_path).parent,
                        )
                    elif is_stream_record_provider(record.source_path, str(provider)):
                        if payload is None:
                            with blob_store.open(blob_hash) as payload_handle:
                                sessions = parse_stream_payload(
                                    provider,
                                    _iter_json_stream(payload_handle, source_name),
                                    fallback_id,
                                )
                        else:
                            sessions = parse_stream_payload(
                                provider,
                                _iter_json_stream(BytesIO(payload), source_name),
                                fallback_id,
                            )
                    else:
                        if payload is None:
                            with blob_store.open(blob_hash) as payload_handle:
                                payloads = list(_iter_json_stream(payload_handle, source_name))
                        else:
                            payloads = list(_iter_json_stream(BytesIO(payload), source_name))
                        sessions = parse_payload(
                            provider,
                            payloads,
                            fallback_id,
                            source_path=record.source_path,
                        )
                    record_timings["full.provider_parse"] = record_timings.get("full.provider_parse", 0.0) + (
                        time.perf_counter() - t0
                    )
                    if not sessions:
                        archive.mark_raw_parse_failed(
                            source_raw_id,
                            provider=provider,
                            error=ValueError("parsed raw payload produced no sessions"),
                        )
                        continue
                    for session in sessions:
                        raw_id, session_id = archive.write_parsed_for_retained_raw(
                            session,
                            raw_id=source_raw_id,
                            source_path=record.source_path,
                            source_index=record.source_index or 0,
                            acquired_at_ms=acquired_at_ms,
                            stage_timings_s=record_timings,
                            stage_timing_prefix="full",
                            finalize_raw_parse=False,
                        )
                        result.raw_ids[record.raw_id] = raw_id
                        result.session_ids.append(session_id)
                        result.session_count += 1
                        result.message_count += len(session.messages)
                    archive.mark_raw_parse_succeeded(source_raw_id, provider=provider)
                    _accumulate_stage_timings(result.stage_timings_s, record_timings)
                except Exception as exc:
                    if provider is not None and source_raw_id is not None:
                        archive.mark_raw_parse_failed(source_raw_id, provider=provider, error=exc)
                    if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_lock(exc):
                        logger.debug(
                            "live.watcher: archive full ingest deferred for %s (retryable lock); will retry: %s",
                            record.source_path,
                            exc,
                        )
                    else:
                        logger.warning(
                            "live.watcher: archive full ingest failed for %s: %s: %s",
                            record.source_path,
                            type(exc).__name__,
                            exc,
                            exc_info=True,
                        )
        return result

    def _extract_zip_member_records(
        self,
        path: Path,
        *,
        blob_store: BlobStore,
        fallback_provider: Provider,
        file_mtime: str,
    ) -> tuple[list[tuple[str, RawSessionRecord]], int]:
        """Expand an inbox ZIP into one raw record per session member.

        Inbox ZIPs (account exports dropped into the watched inbox) were
        previously routed into the byte-level provider-detection branch, where
        ``orjson.loads`` over the ZIP container raised and the entire archive
        was silently marked excluded (#1683). This reuses the maintenance
        acquisition path's member extraction so inbox ZIPs ingest the same way
        as ``polylogue run`` source ZIPs: each member is provider-detected,
        multi-session members are split, and grouped/metadata entries preserve
        their original bytes.

        Returns ``([], 0)`` (caller marks the path excluded) only when the ZIP
        is unreadable or contains no session-bearing members.
        """
        source = Source(name=fallback_provider.value, path=path.parent)
        acquired_at = datetime.now(UTC).isoformat()
        records: list[tuple[str, RawSessionRecord]] = []
        total_bytes = 0
        validator = _ZipEntryValidator(
            fallback_provider,
            cursor_state=None,
            zip_path=path,
            session_only=False,
        )
        try:
            with zipfile.ZipFile(path) as zf:
                for info in validator.filter_entries(zf.infolist()):
                    if info.file_size == 0:
                        continue
                    for raw_data in iter_zip_entry_raw_data(
                        zf,
                        ZipEntryReadContext(
                            source=source,
                            zip_path=path,
                            entry=info,
                            file_mtime=file_mtime,
                            provider_hint=fallback_provider,
                            blob_store=blob_store,
                        ),
                    ):
                        if raw_data.blob_hash is None:
                            continue
                        member_provider = raw_data.provider_hint or fallback_provider
                        member_size = raw_data.blob_size or 0
                        total_bytes += member_size
                        records.append(
                            (
                                raw_data.blob_hash,
                                RawSessionRecord(
                                    raw_id=raw_data.blob_hash,
                                    payload_provider=member_provider,
                                    source_name=member_provider.value,
                                    source_path=raw_data.source_path,
                                    source_index=raw_data.source_index or 0,
                                    blob_size=member_size,
                                    blob_publication_receipt_id=raw_data.blob_publication_receipt_id,
                                    acquired_at=acquired_at,
                                    file_mtime=raw_data.file_mtime,
                                ),
                            )
                        )
        except (zipfile.BadZipFile, OSError) as exc:
            logger.warning("Failed to expand inbox ZIP %s: %s", path, exc)
            return [], 0
        return records, total_bytes

    def _mark_excluded_cursor(self, path: Path, stat: object, *, source_name: str) -> None:
        st_size = int(getattr(stat, "st_size", 0))
        self._cursor.set(
            path,
            st_size,
            byte_offset=st_size,
            last_complete_newline=st_size,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=None,
            source_name=source_name,
            st_dev=getattr(stat, "st_dev", None),
            st_ino=getattr(stat, "st_ino", None),
            mtime_ns=getattr(stat, "st_mtime_ns", None),
            excluded=True,
        )

    def _append_plan(self, path: Path, *, cursor: CursorRecord | None = None) -> _AppendPlan | _DeferredAppend | None:
        if self._source_name_for(path) == "browser-capture" and path.suffix.lower() == ".json":
            return None
        cursor = cursor or self._cursor.get_record(path)
        if cursor is None or cursor.parser_fingerprint != self._current_parser_fingerprint():
            return None
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        if stat.st_size <= cursor.byte_offset:
            return None
        if cursor.st_dev is not None and cursor.st_dev != stat.st_dev:
            return None
        if cursor.st_ino is not None and cursor.st_ino != stat.st_ino:
            return None

        start_offset = max(cursor.byte_offset, 0)
        append_window = min(stat.st_size - start_offset, _MAX_APPEND_PLAN_PAYLOAD_BYTES)
        with path.open("rb") as handle:
            handle.seek(start_offset)
            payload = handle.read(append_window)
        newline_at = payload.rfind(b"\n")
        if newline_at < 0:
            return _DEFER_APPEND
        complete_payload = payload[: newline_at + 1]
        if not complete_payload:
            return _DEFER_APPEND
        append_payload = self._append_payload_for_provider(path, self._source_name_for(path), complete_payload)
        if append_payload is None:
            return None
        tail_hash = sha256(complete_payload).hexdigest()
        return _AppendPlan(
            path=path,
            source_name=self._source_name_for(path),
            start_offset=start_offset,
            last_complete_newline=start_offset + newline_at + 1,
            stat_size=stat.st_size,
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            payload=append_payload,
            payload_hash=tail_hash,
            cursor_fingerprint=cursor.content_fingerprint,
            bytes_read=len(payload),
        )

    def _append_payload_for_provider(self, path: Path, source_name: str, payload: bytes) -> bytes | None:
        provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        if provider is Provider.CODEX:
            identity = self._existing_provider_session_id(path)
            if identity is None:
                return None
            session_meta = json_dumps(
                {"type": "session_meta", "payload": {"id": identity}},
                separators=(",", ":"),
            ).encode()
            return session_meta + b"\n" + payload
        if provider is Provider.CLAUDE_CODE and not self._claude_code_tail_matches_existing_identity(path, payload):
            return None
        return payload

    def _existing_provider_session_id(self, path: Path) -> str | None:
        identity = self._existing_archive_session_native_id(path)
        if identity is not None:
            return identity
        codex_identity = self._codex_session_meta_native_id(path)
        if codex_identity is None:
            return None
        if self._archive_has_native_session("codex-session", codex_identity):
            return codex_identity
        return None

    def _codex_session_meta_native_id(self, path: Path) -> str | None:
        try:
            with path.open("rb") as handle:
                line = handle.readline(1024 * 1024)
        except OSError:
            return None
        if not line:
            return None
        try:
            record = json_loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, TypeError):
            return None
        if not isinstance(record, dict) or record.get("type") != "session_meta":
            return None
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return None
        value = payload.get("id")
        return value if isinstance(value, str) and value.strip() else None

    def _archive_has_native_session(self, origin: str, native_id: str) -> bool:
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        index_db = archive_root / "index.db"
        if not index_db.exists():
            return False
        try:
            conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    """
                    SELECT 1
                    FROM sessions
                    WHERE origin = ? AND native_id = ?
                    LIMIT 1
                    """,
                    (origin, native_id),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return False
        return row is not None

    def _existing_archive_session_native_id(self, path: Path) -> str | None:
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        index_db = archive_root / "index.db"
        source_db = archive_root / "source.db"
        if not index_db.exists() or not source_db.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
            try:
                conn.execute("ATTACH DATABASE ? AS source_tier", (f"file:{source_db}?mode=ro",))
                row = conn.execute(
                    """
                    SELECT s.native_id
                    FROM sessions AS s
                    JOIN source_tier.raw_sessions AS r ON r.raw_id = s.raw_id
                    WHERE r.source_path = ?
                    ORDER BY s.sort_key_ms DESC, s.created_at_ms DESC, s.session_id DESC
                    LIMIT 1
                    """,
                    (str(path),),
                ).fetchone()
                conn.execute("DETACH DATABASE source_tier")
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        value = row[0]
        return value if isinstance(value, str) and value.strip() else None

    def _claude_code_tail_matches_existing_identity(self, path: Path, payload: bytes) -> bool:
        existing_id = self._existing_provider_session_id(path)
        if existing_id is None:
            return False
        session_ids: set[str] = set()
        for line in payload.splitlines():
            if not line.strip():
                continue
            try:
                record = json_loads(line)
            except ValueError:
                return False
            if not isinstance(record, dict):
                return False
            session_id = record.get("sessionId")
            if isinstance(session_id, str) and session_id.strip():
                session_ids.add(session_id)
        if not session_ids:
            return existing_id == path.stem
        return any(existing_id == session_id or existing_id.startswith(f"{session_id}:") for session_id in session_ids)

    def _ingest_append_plans(self, plans: list[_AppendPlan]) -> _AppendResult:
        return ingest_append_plans(self, plans)

    def _compact_superseded_raw_snapshots(self, paths: list[Path]) -> None:
        if not paths:
            return
        from polylogue.storage.raw_retention import compact_paths_superseded_raw_snapshots

        source_db = self._cursor._db_path.with_name("source.db")
        if not source_db.exists():
            return
        with sqlite3.connect(source_db) as conn:
            conn.row_factory = sqlite3.Row
            result = compact_paths_superseded_raw_snapshots(
                conn, paths, limit_per_path=25, min_acquired_at=self._raw_compaction_min_acquired_at
            )
        if result.errors:
            logger.warning("live.watcher: raw snapshot compaction errors: %s", "; ".join(result.errors[:3]))

    def _record_append_cursor(self, plan: _AppendPlan) -> bool:
        content_fingerprint = sha256(f"{plan.cursor_fingerprint or ''}\0{plan.payload_hash}".encode()).hexdigest()
        tail_hash, _tail_bytes = tail_hash_from_path(plan.path, plan.stat_size)
        updated = self._cursor.set(
            plan.path,
            plan.stat_size,
            byte_offset=plan.last_complete_newline,
            last_complete_newline=plan.last_complete_newline,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=content_fingerprint,
            tail_hash=tail_hash,
            source_name=plan.source_name,
            st_dev=plan.st_dev,
            st_ino=plan.st_ino,
            mtime_ns=plan.mtime_ns,
        )
        self._cursor.reset_failures(plan.path)
        return updated


# fmt: off
__all__ = ["LiveBatchMetrics", "LiveBatchProcessor", "_FullIngestResult", "_LARGE_FULL_PARSE_PROGRESS_BYTES", "_MAX_APPEND_PLAN_PAYLOAD_BYTES", "_SMALL_FULL_PARSE_PROGRESS_MAX_BYTES", "_SMALL_FULL_PARSE_PROGRESS_MAX_FILES", "_STREAMING_FULL_INGEST_BYTES", "_full_ingest_worker_count", "_full_parse_progress_groups", "fingerprint_file", "last_complete_newline_from_tail"]
# fmt: on

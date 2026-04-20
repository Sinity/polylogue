"""Ingest orchestration: acquire → unified ingest (validate + parse + write).

Validation is unconditionally part of ingest — done inline in subprocess
workers. No separate validation stage. Records already validated are still
re-validated (cheap — schema check is <1ms per record, and the blob is
already decoded for parsing anyway).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.pipeline.payload_types import ParseBatchObservation, ParseBatchObservationSummary
from polylogue.pipeline.run_support import PARSE_STAGES
from polylogue.pipeline.services.parsing_models import IngestResult, IngestState, ParseResult

if TYPE_CHECKING:
    from polylogue.config import Source
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.protocols import ProgressCallback

logger = get_logger(__name__)


def _iter_raw_id_batches(
    headers: Iterable[tuple[str, int]],
    *,
    max_records: int,
    max_blob_bytes: int,
) -> Iterator[list[str]]:
    batch_ids: list[str] = []
    batch_blob_bytes = 0

    for raw_id, blob_size in headers:
        record_blob_bytes = max(int(blob_size), 0)
        if batch_ids and (len(batch_ids) >= max_records or batch_blob_bytes + record_blob_bytes > max_blob_bytes):
            yield batch_ids
            batch_ids = []
            batch_blob_bytes = 0

        batch_ids.append(raw_id)
        batch_blob_bytes += record_blob_bytes

        if len(batch_ids) >= max_records or batch_blob_bytes >= max_blob_bytes:
            yield batch_ids
            batch_ids = []
            batch_blob_bytes = 0

    if batch_ids:
        yield batch_ids


async def _iter_raw_id_batches_async(
    headers: AsyncIterator[tuple[str, int]],
    *,
    max_records: int,
    max_blob_bytes: int,
) -> AsyncIterator[list[str]]:
    batch_ids: list[str] = []
    batch_blob_bytes = 0

    async for raw_id, blob_size in headers:
        record_blob_bytes = max(int(blob_size), 0)
        if batch_ids and (len(batch_ids) >= max_records or batch_blob_bytes + record_blob_bytes > max_blob_bytes):
            yield batch_ids
            batch_ids = []
            batch_blob_bytes = 0

        batch_ids.append(raw_id)
        batch_blob_bytes += record_blob_bytes

        if len(batch_ids) >= max_records or batch_blob_bytes >= max_blob_bytes:
            yield batch_ids
            batch_ids = []
            batch_blob_bytes = 0

    if batch_ids:
        yield batch_ids


def _append_unique_raw_ids(
    target: list[str],
    *,
    seen: set[str],
    raw_ids: Iterable[str],
) -> None:
    for raw_id in raw_ids:
        if raw_id in seen:
            continue
        seen.add(raw_id)
        target.append(raw_id)


def _summarize_batch_observations(
    batch_observations: list[ParseBatchObservation],
) -> ParseBatchObservationSummary:
    if not batch_observations:
        return {}

    def _observation_float(observation: ParseBatchObservation, field: str) -> float | None:
        value = observation.get(field)
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _max_float(field: str) -> float | None:
        values = [
            value for observation in batch_observations if (value := _observation_float(observation, field)) is not None
        ]
        return round(max(values), 1) if values else None

    slow_batch_count = 0
    for observation in batch_observations:
        elapsed_ms = _observation_float(observation, "elapsed_ms")
        if elapsed_ms is not None and elapsed_ms >= 2000.0:
            slow_batch_count += 1

    summary: ParseBatchObservationSummary = {
        "batch_count": len(batch_observations),
        "slow_batch_count": slow_batch_count,
        "batches": batch_observations,
    }
    if (value := _max_float("elapsed_ms")) is not None:
        summary["max_elapsed_ms"] = value
    if (value := _max_float("blob_mb")) is not None:
        summary["max_blob_mb"] = value
    if (value := _max_float("max_result_mb")) is not None:
        summary["max_result_mb"] = value
    if (value := _max_float("max_current_rss_mb")) is not None:
        summary["max_current_rss_mb"] = value
    if (value := _max_float("rss_end_mb")) is not None:
        summary["max_rss_end_mb"] = value
    if (value := _max_float("rss_delta_mb")) is not None:
        summary["max_rss_delta_mb"] = value
    if (value := _max_float("peak_rss_growth_mb")) is not None:
        summary["max_peak_rss_growth_mb"] = value
    return summary


async def ingest_sources(
    service: ParsingService,
    *,
    sources: list[Source],
    stage: str = "all",
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
    parse_records: bool = True,
    skip_acquire: bool = False,
) -> IngestResult:
    """Canonical ingestion orchestration.

    Two-stage flow:
    1. Acquire: walk sources, hash files to blob store
    2. Ingest: unified decode + validate + parse + transform + write
       (validation is inline in subprocess workers, not a separate stage)
    """
    from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
    from polylogue.pipeline.services.planning import PlanningService

    t_total = time.perf_counter()
    timings: dict[str, float] = {}
    backend = service._require_backend()
    source_names = [source.name for source in sources]

    # Stage 1: Acquire
    t0 = time.perf_counter()
    if skip_acquire:
        acquire_result = AcquireResult()
    else:
        acquire_result = await AcquisitionService(backend=backend).acquire_sources(
            sources,
            ui=ui,
            progress_callback=progress_callback,
            drive_config=service.config.drive_config,
        )
    timings["acquire"] = time.perf_counter() - t0
    logger.info(
        "acquire",
        elapsed_s=round(timings["acquire"], 2),
        raw_ids=len(acquire_result.raw_ids),
        **acquire_result.counts,
    )

    # Track state for the IngestResult interface.
    # Validation is inline — no separate validation phase.
    ingest_state = IngestState(
        source_names=tuple(source_names),
        parse_requested=parse_records,
    )
    ingest_state.record_acquired(acquire_result.raw_ids)
    ingest_state.record_validation_candidates([])
    ingest_state.record_validation_result([])

    # Stage 2: Unified ingest (validate + parse + transform + write)
    parse_raw_ids: list[str] = []
    parse_result = ParseResult()
    if parse_records:
        t0 = time.perf_counter()
        planning_service = PlanningService(backend=backend, config=service.config)

        # Collect all raw IDs that need ingesting: newly acquired + backlog
        seen_parse_raw_ids: set[str] = set()
        _append_unique_raw_ids(
            parse_raw_ids,
            seen=seen_parse_raw_ids,
            raw_ids=acquire_result.raw_ids,
        )
        if stage in PARSE_STAGES:
            backlog = await planning_service.collect_parse_backlog(
                source_names=source_names or None,
                exclude_raw_ids=parse_raw_ids,
            )
            _append_unique_raw_ids(
                parse_raw_ids,
                seen=seen_parse_raw_ids,
                raw_ids=backlog,
            )
            # Also collect validation backlog (records not yet validated/parsed)
            validation_backlog = await planning_service.collect_validation_backlog(
                source_names=source_names or None,
                exclude_raw_ids=parse_raw_ids,
            )
            _append_unique_raw_ids(
                parse_raw_ids,
                seen=seen_parse_raw_ids,
                raw_ids=validation_backlog,
            )

        # Satisfy IngestState invariants
        ingest_state.record_parse_candidates(
            parse_raw_ids,
            persisted_validated_raw_ids=parse_raw_ids,
        )

        if parse_raw_ids:
            parse_result = await service.parse_from_raw(
                raw_ids=parse_raw_ids,
                progress_callback=progress_callback,
            )
        ingest_state.record_parse_completed()
        parse_raw_ids = ingest_state.parse_raw_ids
        timings["ingest"] = time.perf_counter() - t0
        logger.info(
            "ingest",
            elapsed_s=round(timings["ingest"], 2),
            raw_ids=len(parse_raw_ids),
            processed=len(parse_result.processed_ids),
            failures=parse_result.parse_failures,
        )

    total_s = time.perf_counter() - t_total
    logger.info(
        "ingest_complete",
        total_s=round(total_s, 2),
        **{f"{k}_s": round(v, 2) for k, v in timings.items()},
    )

    return IngestResult(
        acquire_result=acquire_result,
        validation_result=None,
        parse_result=parse_result,
        parse_raw_ids=parse_raw_ids,
        timings=timings,
        diagnostics={
            "acquisition": acquire_result.diagnostics,
            "batch_observations": _summarize_batch_observations(parse_result.batch_observations),
        },
    )


async def parse_from_raw(
    service: ParsingService,
    *,
    raw_ids: list[str] | None = None,
    provider: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ParseResult:
    """Parse raw_conversations from DB into conversations.

    Uses the unified ingest batch processor (decode + validate + parse +
    transform + write in one pass). Derived session-product materialization
    happens in an explicit downstream pipeline stage.
    """
    from polylogue.pipeline.services.ingest_batch import process_ingest_batch

    result = ParseResult()
    backend = service._require_backend()
    t_start = time.perf_counter()
    batches_processed = 0

    if raw_ids is not None:
        raw_headers = await service.repository.get_raw_blob_sizes(raw_ids)
        total = len(raw_headers)
        if progress_callback is not None:
            progress_callback(0, desc=f"Ingesting ({total:,} raw)")
        processed_so_far = 0
        for batch_ids in _iter_raw_id_batches(
            raw_headers,
            max_records=service.raw_batch_size,
            max_blob_bytes=service.raw_batch_blob_limit_bytes,
        ):
            t_batch = time.perf_counter()
            batch_observation = await process_ingest_batch(
                service,
                backend,
                batch_ids,
                result,
                progress_callback,
            )
            batches_processed += 1
            batch_elapsed = time.perf_counter() - t_batch
            processed_so_far += len(batch_ids)
            if batch_observation is not None:
                batch_observation["batch"] = batches_processed
                batch_observation["processed_raw"] = processed_so_far
                result.batch_observations.append(batch_observation)
            if progress_callback is not None:
                progress_callback(
                    0,
                    desc=f"Ingesting ({processed_so_far:,}/{total:,} raw, batch {batches_processed})",
                )
            if batch_elapsed > 2.0:
                logger.info(
                    "slow_batch",
                    batch=batches_processed,
                    size=len(batch_ids),
                    elapsed_s=round(batch_elapsed, 2),
                    rate=round(len(batch_ids) / batch_elapsed, 1) if batch_elapsed > 0 else 0,
                )
    else:
        if progress_callback is not None:
            progress_callback(0, desc="Ingesting")
        total_raw = 0
        async for batch_ids in _iter_raw_id_batches_async(
            service.repository.iter_raw_headers(provider_name=provider),
            max_records=service.raw_batch_size,
            max_blob_bytes=service.raw_batch_blob_limit_bytes,
        ):
            batch_observation = await process_ingest_batch(
                service,
                backend,
                batch_ids,
                result,
                progress_callback,
            )
            batches_processed += 1
            total_raw += len(batch_ids)
            if batch_observation is not None:
                batch_observation["batch"] = batches_processed
                batch_observation["processed_raw"] = total_raw
                result.batch_observations.append(batch_observation)
            if progress_callback is not None:
                progress_callback(
                    0,
                    desc=f"Ingesting ({total_raw:,} raw, batch {batches_processed})",
                )
        total = total_raw

    elapsed = time.perf_counter() - t_start
    logger.info(
        "parse_from_raw_complete",
        total_raw=total,
        batches=batches_processed,
        elapsed_s=round(elapsed, 2),
        processed=len(result.processed_ids),
        rate_raw_per_s=round(total / elapsed, 1) if elapsed > 0 else 0,
    )
    return result


__all__ = ["ingest_sources", "parse_from_raw"]

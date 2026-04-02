"""Ingest orchestration: acquire → unified ingest (validate + parse + write).

Validation is unconditionally part of ingest — done inline in subprocess
workers. No separate validation stage. Records already validated are still
re-validated (cheap — schema check is <1ms per record, and the blob is
already decoded for parsing anyway).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.pipeline.services.parsing_models import IngestResult, IngestState, ParseResult

if TYPE_CHECKING:
    from polylogue.config import Source
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.protocols import ProgressCallback

logger = get_logger(__name__)


def _summarize_batch_observations(
    batch_observations: list[dict[str, object]],
) -> dict[str, object]:
    if not batch_observations:
        return {}

    def _max_float(field: str) -> float | None:
        values = [
            float(value)
            for observation in batch_observations
            if (value := observation.get(field)) is not None
        ]
        return round(max(values), 1) if values else None

    return {
        "batch_count": len(batch_observations),
        "slow_batch_count": sum(
            1
            for observation in batch_observations
            if float(observation["elapsed_ms"]) >= 2000.0
        ),
        "max_elapsed_ms": _max_float("elapsed_ms"),
        "max_blob_mb": _max_float("blob_mb"),
        "max_current_rss_mb": _max_float("max_current_rss_mb"),
        "max_rss_end_mb": _max_float("rss_end_mb"),
        "max_rss_delta_mb": _max_float("rss_delta_mb"),
        "batches": batch_observations,
    }


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
        parse_raw_ids = list(acquire_result.raw_ids)
        if stage in {"parse", "all"}:
            backlog = await planning_service.collect_parse_backlog(
                source_names=source_names or None,
                exclude_raw_ids=parse_raw_ids,
            )
            parse_raw_ids.extend(backlog)
            # Also collect validation backlog (records not yet validated/parsed)
            validation_backlog = await planning_service.collect_validation_backlog(
                source_names=source_names or None,
                exclude_raw_ids=parse_raw_ids,
            )
            parse_raw_ids.extend(validation_backlog)
        parse_raw_ids = list(dict.fromkeys(parse_raw_ids))

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
    transform + write in one pass). Session product refresh is deferred to
    a single bulk pass after all batches complete.
    """
    from polylogue.pipeline.services.ingest_batch import (
        process_ingest_batch,
        refresh_session_products_bulk,
    )

    result = ParseResult()
    backend = service._require_backend()
    t_start = time.perf_counter()
    batches_processed = 0

    if raw_ids is not None:
        total = len(raw_ids)
        if progress_callback is not None:
            progress_callback(0, desc=f"Ingesting ({total:,} raw)")
        for batch_start in range(0, total, service.RAW_BATCH_SIZE):
            batch_ids = raw_ids[batch_start : batch_start + service.RAW_BATCH_SIZE]
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
            processed_so_far = batch_start + len(batch_ids)
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
        batch_ids_acc: list[str] = []
        total_raw = 0
        async for raw_id in backend.queries.iter_raw_ids(provider_name=provider):
            batch_ids_acc.append(raw_id)
            total_raw += 1
            if len(batch_ids_acc) >= service.RAW_BATCH_SIZE:
                batch_observation = await process_ingest_batch(
                    service,
                    backend,
                    batch_ids_acc,
                    result,
                    progress_callback,
                )
                batches_processed += 1
                if batch_observation is not None:
                    batch_observation["batch"] = batches_processed
                    batch_observation["processed_raw"] = total_raw
                    result.batch_observations.append(batch_observation)
                if progress_callback is not None:
                    progress_callback(
                        0,
                        desc=f"Ingesting ({total_raw:,} raw, batch {batches_processed})",
                    )
                batch_ids_acc = []
        if batch_ids_acc:
            batch_observation = await process_ingest_batch(
                service,
                backend,
                batch_ids_acc,
                result,
                progress_callback,
            )
            batches_processed += 1
            if batch_observation is not None:
                batch_observation["batch"] = batches_processed
                batch_observation["processed_raw"] = total_raw
                result.batch_observations.append(batch_observation)
        total = total_raw

    # Deferred session product refresh — once after ALL batches
    changed_cids = result._changed_conversation_ids
    if changed_cids:
        t_refresh = time.perf_counter()
        if progress_callback is not None:
            progress_callback(0, desc=f"Refreshing session products ({len(changed_cids):,} conversations)")
        await refresh_session_products_bulk(backend, changed_cids)
        refresh_elapsed = time.perf_counter() - t_refresh
        if refresh_elapsed > 2.0:
            logger.info(
                "deferred_session_refresh",
                elapsed_s=round(refresh_elapsed, 2),
                conversations=len(changed_cids),
            )

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

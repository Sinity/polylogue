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


async def ingest_sources(
    service: ParsingService,
    *,
    sources: list[Source],
    stage: str = "all",
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
    parse_records: bool = True,
    skip_acquire: bool = False,
    skip_validate: bool = False,
) -> IngestResult:
    """Canonical ingestion orchestration for runtime callers."""
    from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
    from polylogue.pipeline.services.planning import PlanningService
    from polylogue.pipeline.services.validation import ValidationService

    t_total = time.perf_counter()
    timings: dict[str, float] = {}
    backend = service._require_backend()
    source_names = [source.name for source in sources]

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

    ingest_state = IngestState(
        source_names=tuple(source_names),
        parse_requested=parse_records,
    )
    ingest_state.record_acquired(acquire_result.raw_ids)
    planning_service = PlanningService(backend=backend, config=service.config)

    validation_result = None
    validation_ids: list[str] = []
    if skip_validate:
        ingest_state.record_validation_candidates([])
        ingest_state.record_validation_result([])
    else:
        t0 = time.perf_counter()
        validation_ids = list(acquire_result.raw_ids)
        if stage in {"validate", "parse", "all"}:
            validation_ids.extend(
                await planning_service.collect_validation_backlog(
                    source_names=source_names or None,
                    exclude_raw_ids=validation_ids,
                )
            )
        ingest_state.record_validation_candidates(validation_ids)

        if validation_ids:
            validation_result = await ValidationService(backend=backend).validate_raw_ids(
                raw_ids=validation_ids,
                progress_callback=progress_callback,
            )
        ingest_state.record_validation_result(
            validation_result.parseable_raw_ids if validation_result else [],
        )
        timings["validate"] = time.perf_counter() - t0
        logger.info(
            "validate",
            elapsed_s=round(timings["validate"], 2),
            candidates=len(validation_ids),
            parseable=len(validation_result.parseable_raw_ids) if validation_result else 0,
        )

    parse_raw_ids: list[str] = []
    parse_result = ParseResult()
    if parse_records:
        t0 = time.perf_counter()
        parse_raw_ids = await planning_service.collect_parse_backlog(
            source_names=source_names or None,
            exclude_raw_ids=validation_ids,
        )
        if validation_result is not None:
            parse_raw_ids.extend(validation_result.parseable_raw_ids)
            parse_raw_ids = list(dict.fromkeys(parse_raw_ids))
        current_validation_ids = set(ingest_state.validation_raw_ids)
        persisted_validated_ids = [
            raw_id for raw_id in parse_raw_ids if raw_id not in current_validation_ids
        ]
        ingest_state.record_parse_candidates(
            parse_raw_ids,
            persisted_validated_raw_ids=persisted_validated_ids,
        )
        if parse_raw_ids:
            parse_result = await service.parse_from_raw(
                raw_ids=parse_raw_ids,
                progress_callback=progress_callback,
            )
        ingest_state.record_parse_completed()
        parse_raw_ids = ingest_state.parse_raw_ids
        timings["parse"] = time.perf_counter() - t0
        logger.info(
            "parse",
            elapsed_s=round(timings["parse"], 2),
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
        validation_result=validation_result,
        parse_result=parse_result,
        parse_raw_ids=parse_raw_ids,
        timings=timings,
    )


async def parse_from_raw(
    service: ParsingService,
    *,
    raw_ids: list[str] | None = None,
    provider: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ParseResult:
    """Parse raw_conversations from DB into conversations."""
    result = ParseResult()
    backend = service._require_backend()
    t_start = time.perf_counter()
    batches_processed = 0

    if raw_ids is not None:
        total = len(raw_ids)
        if progress_callback is not None:
            progress_callback(0, desc=f"Parsing ({total:,} raw)")
        for batch_start in range(0, total, service.RAW_BATCH_SIZE):
            batch_ids = raw_ids[batch_start : batch_start + service.RAW_BATCH_SIZE]
            t_batch = time.perf_counter()
            await service._process_raw_batch(
                backend,
                batch_ids,
                result,
                progress_callback,
            )
            batches_processed += 1
            batch_elapsed = time.perf_counter() - t_batch
            if batch_elapsed > 2.0:
                logger.info(
                    "slow_batch",
                    batch=batches_processed,
                    size=len(batch_ids),
                    elapsed_s=round(batch_elapsed, 2),
                    rate=round(len(batch_ids) / batch_elapsed, 1) if batch_elapsed > 0 else 0,
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

    if progress_callback is not None:
        progress_callback(0, desc="Parsing")
    batch_ids: list[str] = []
    total_raw = 0
    async for raw_id in backend.queries.iter_raw_ids(provider_name=provider):
        batch_ids.append(raw_id)
        total_raw += 1
        if len(batch_ids) >= service.RAW_BATCH_SIZE:
            await service._process_raw_batch(
                backend,
                batch_ids,
                result,
                progress_callback,
            )
            batches_processed += 1
            batch_ids = []
    if batch_ids:
        await service._process_raw_batch(
            backend,
            batch_ids,
            result,
            progress_callback,
        )
        batches_processed += 1

    elapsed = time.perf_counter() - t_start
    logger.info(
        "parse_from_raw_complete",
        total_raw=total_raw,
        batches=batches_processed,
        elapsed_s=round(elapsed, 2),
        processed=len(result.processed_ids),
        rate_raw_per_s=round(total_raw / elapsed, 1) if elapsed > 0 else 0,
    )
    return result


__all__ = ["ingest_sources", "parse_from_raw"]

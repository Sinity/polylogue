from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.pipeline.services.parsing_models import IngestResult, IngestState, ParseResult

if TYPE_CHECKING:

    from polylogue.config import Source
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.protocols import ProgressCallback


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

    backend = service._require_backend()
    source_names = [source.name for source in sources]

    if skip_acquire:
        acquire_result = AcquireResult()
    else:
        acquire_result = await AcquisitionService(backend=backend).acquire_sources(
            sources,
            ui=ui,
            progress_callback=progress_callback,
            drive_config=service.config.drive_config,
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

    parse_raw_ids: list[str] = []
    parse_result = ParseResult()
    if parse_records:
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

    return IngestResult(
        acquire_result=acquire_result,
        validation_result=validation_result,
        parse_result=parse_result,
        parse_raw_ids=parse_raw_ids,
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

    if raw_ids is not None:
        total = len(raw_ids)
        if progress_callback is not None:
            progress_callback(0, desc=f"Parsing ({total:,} raw)")
        for batch_start in range(0, total, service.RAW_BATCH_SIZE):
            batch_ids = raw_ids[batch_start : batch_start + service.RAW_BATCH_SIZE]
            await service._process_raw_batch(
                backend,
                batch_ids,
                result,
                progress_callback,
            )
        return result

    if progress_callback is not None:
        progress_callback(0, desc="Parsing")
    batch_ids: list[str] = []
    async for raw_id in backend.queries.iter_raw_ids(provider_name=provider):
        batch_ids.append(raw_id)
        if len(batch_ids) >= service.RAW_BATCH_SIZE:
            await service._process_raw_batch(
                backend,
                batch_ids,
                result,
                progress_callback,
            )
            batch_ids = []
    if batch_ids:
        await service._process_raw_batch(
            backend,
            batch_ids,
            result,
            progress_callback,
        )

    return result


__all__ = ["ingest_sources", "parse_from_raw"]

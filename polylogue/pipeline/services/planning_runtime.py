"""Runtime plan construction for ingest orchestration."""

from __future__ import annotations

import time
from collections.abc import Sequence

from polylogue.config import Source
from polylogue.pipeline.run_support import expand_requested_stage, normalize_stage_sequence
from polylogue.pipeline.stage_models import ValidateResult
from polylogue.protocols import ProgressCallback
from polylogue.storage.state_views import PlanResult
from polylogue.storage.store import RawConversationRecord
from polylogue.types import PlanStage

from .acquisition import AcquisitionService
from .planning_backlog import collect_parse_backlog, collect_validation_backlog, dedupe_ids
from .planning_models import IngestPlan
from .validation import ValidationService

_SCAN_STATE_BATCH_SIZE = 200  # Metadata-only rows (no BLOBs) — can batch larger


def _summarize_plan_stage(
    *,
    stage: str,
    normalized_stage_sequence: tuple[str, ...],
    stage_sequence: Sequence[str] | None,
) -> str:
    if stage_sequence is None:
        return stage
    if normalized_stage_sequence == expand_requested_stage(stage):
        return stage
    return PlanStage.CUSTOM.value


async def build_ingest_plan(
    service,
    *,
    sources: list[Source],
    stage: str = "all",
    stage_sequence: Sequence[str] | None = None,
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
    preview: bool = False,
) -> IngestPlan:
    source_names = [source.name for source in sources]
    db_scope_names = source_names or None

    normalized_stage_sequence = normalize_stage_sequence(stage=stage, stage_sequence=stage_sequence)
    summary_stage = _summarize_plan_stage(
        stage=stage,
        normalized_stage_sequence=normalized_stage_sequence,
        stage_sequence=stage_sequence,
    )
    has_acquire = "acquire" in normalized_stage_sequence
    has_parse = "parse" in normalized_stage_sequence
    has_materialize = "materialize" in normalized_stage_sequence
    has_render = "render" in normalized_stage_sequence
    has_index = "index" in normalized_stage_sequence

    if not has_acquire and not has_parse:
        conversation_count = 0
        if has_materialize or has_render or has_index:
            conversation_count = await service.backend.queries.count_conversation_ids(source_names=db_scope_names)
        counts: dict[str, int] = {}
        if has_materialize and conversation_count:
            counts["materialize"] = conversation_count
        if has_render and conversation_count:
            counts["render"] = conversation_count
        if has_index and conversation_count:
            counts["index"] = conversation_count
        return IngestPlan(
            summary=PlanResult(
                timestamp=int(time.time()),
                stage=summary_stage,
                stage_sequence=list(normalized_stage_sequence),
                counts=counts,
                sources=source_names,
                cursors={},
            ),
            validate_raw_ids=[],
            parse_ready_raw_ids=[],
        )

    if has_parse and not has_acquire:
        validate_raw_ids = await collect_validation_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=[],
        )
        parse_ready_raw_ids = await collect_parse_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=validate_raw_ids,
        )
        reprocess_raw_ids = dedupe_ids([*validate_raw_ids, *parse_ready_raw_ids])
        counts: dict[str, int] = {}
        details: dict[str, int] = {}
        if validate_raw_ids:
            counts["validate"] = len(validate_raw_ids)
            details["backlog_validate"] = len(validate_raw_ids)
        if reprocess_raw_ids:
            counts["parse"] = len(reprocess_raw_ids)
            details["backlog_parse"] = len(parse_ready_raw_ids)
            if has_materialize:
                counts["materialize"] = len(reprocess_raw_ids)
            if has_render:
                counts["render"] = len(reprocess_raw_ids)
            if has_index:
                counts["index"] = len(reprocess_raw_ids)
        return IngestPlan(
            summary=PlanResult(
                timestamp=int(time.time()),
                stage=summary_stage,
                stage_sequence=list(normalized_stage_sequence),
                counts=counts,
                details=details,
                sources=source_names,
                cursors={},
            ),
            validate_raw_ids=validate_raw_ids,
            parse_ready_raw_ids=reprocess_raw_ids,
        )

    acquisition = AcquisitionService(service.backend)
    validation = ValidationService(service.backend)
    scanned_count = 0
    validate_raw_ids: list[str] = []
    parse_ready_raw_ids: list[str] = []
    details = {
        "new_raw": 0,
        "existing_raw": 0,
        "duplicate_raw": 0,
        "backlog_validate": 0,
        "backlog_parse": 0,
    }
    pending_records: list[RawConversationRecord] = []
    preview_validation = ValidateResult() if preview and has_parse else None
    seen_scanned_raw_ids: set[str] = set()

    max_preview_validation_records = 50  # Cap preview validation to bound memory

    async def flush_pending_records() -> None:
        nonlocal pending_records
        if not pending_records:
            return
        # Extract only the raw_ids for DB state comparison — avoid keeping
        # full raw bytes in working set longer than necessary.
        batch_raw_ids = dedupe_ids([record.raw_id for record in pending_records])
        scanned_states = await service.repository.get_raw_conversation_states(batch_raw_ids)

        preview_records: list[RawConversationRecord] = []
        for record in pending_records:
            if record.raw_id in seen_scanned_raw_ids:
                details["duplicate_raw"] += 1
                continue
            seen_scanned_raw_ids.add(record.raw_id)
            state = scanned_states.get(record.raw_id)
            if state is None:
                details["new_raw"] += 1
            else:
                details["existing_raw"] += 1

            if has_parse:
                current_status = state.validation_status if state is not None else None
                parsed_at = state.parsed_at if state is not None else None
                if parsed_at is None:
                    if current_status is None:
                        validate_raw_ids.append(record.raw_id)
                        # Only accumulate a limited sample for preview validation
                        # to bound memory. Full validation happens during actual runs.
                        if preview and preview_validation is not None and len(preview_records) < max_preview_validation_records:
                            preview_records.append(record)
                    elif current_status in {"passed", "skipped"}:
                        parse_ready_raw_ids.append(record.raw_id)

        if preview_records and preview_validation is not None:
            scanned_preview_validation = await validation.evaluate_raw_records(
                raw_records=preview_records,
                persist=False,
            )
            preview_validation.merge(scanned_preview_validation)
        pending_records.clear()

    async def process_record(record: RawConversationRecord) -> None:
        nonlocal scanned_count
        scanned_count += 1
        pending_records.append(record)
        if len(pending_records) >= _SCAN_STATE_BATCH_SIZE:
            await flush_pending_records()

    scan_result = await acquisition.visit_sources(
        sources,
        progress_callback=progress_callback,
        ui=ui,
        drive_config=service.config.drive_config,
        on_record=process_record,
    )
    await flush_pending_records()

    if has_parse:
        backlog_validate_ids = await collect_validation_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=validate_raw_ids,
        )
        if backlog_validate_ids:
            details["backlog_validate"] = len(backlog_validate_ids)
            validate_raw_ids.extend(backlog_validate_ids)

        if preview:
            if backlog_validate_ids and preview_validation is not None:
                backlog_preview_validation = await validation.validate_raw_ids(
                    raw_ids=backlog_validate_ids,
                    persist=False,
                )
                preview_validation.merge(backlog_preview_validation)
            if preview_validation is not None and preview_validation.counts["invalid"]:
                details["preview_invalid"] = preview_validation.counts["invalid"]
            if preview_validation is not None and preview_validation.counts["skipped_no_schema"]:
                details["preview_skipped_no_schema"] = preview_validation.counts["skipped_no_schema"]

    if has_parse:
        backlog_parse_ids = await collect_parse_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=validate_raw_ids,
        )
        details["backlog_parse"] = len(backlog_parse_ids)
        parse_ready_raw_ids.extend(backlog_parse_ids)
        if preview_validation is not None:
            parse_ready_raw_ids.extend(preview_validation.parseable_raw_ids)
        parse_ready_raw_ids = dedupe_ids(parse_ready_raw_ids)

    counts: dict[str, int] = {}
    if scanned_count:
        counts["scan"] = scanned_count
    if details["new_raw"] and has_acquire:
        counts["store_raw"] = details["new_raw"]
    if has_parse and validate_raw_ids:
        counts["validate"] = len(validate_raw_ids)
    if has_parse and parse_ready_raw_ids:
        counts["parse"] = len(parse_ready_raw_ids)
        if has_materialize:
            counts["materialize"] = len(parse_ready_raw_ids)
        if has_render:
            counts["render"] = len(parse_ready_raw_ids)
        if has_index:
            counts["index"] = len(parse_ready_raw_ids)
    elif not has_parse and (has_materialize or has_render or has_index):
        conversation_count = await service.backend.queries.count_conversation_ids(source_names=db_scope_names)
        if has_materialize and conversation_count:
            counts["materialize"] = conversation_count
        if has_render and conversation_count:
            counts["render"] = conversation_count
        if has_index and conversation_count:
            counts["index"] = conversation_count

    summary = PlanResult(
        timestamp=int(time.time()),
        stage=summary_stage,
        stage_sequence=list(normalized_stage_sequence),
        counts=counts,
        details={key: value for key, value in details.items() if value},
        sources=source_names,
        cursors=scan_result.cursors,
    )
    return IngestPlan(
        summary=summary,
        validate_raw_ids=validate_raw_ids,
        parse_ready_raw_ids=parse_ready_raw_ids,
    )


__all__ = ["build_ingest_plan"]

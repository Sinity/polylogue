"""Runtime plan construction for ingest orchestration."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from polylogue.config import Source
from polylogue.pipeline.run_support import expand_requested_stage, normalize_stage_sequence
from polylogue.pipeline.stage_models import ValidateResult
from polylogue.protocols import ProgressCallback
from polylogue.storage.raw_ingest_artifacts import RawIngestArtifactState
from polylogue.storage.state_views import PlanResult
from polylogue.storage.store import RawConversationRecord
from polylogue.types import PlanStage

from .acquisition import AcquisitionService
from .planning_backlog import collect_parse_backlog, collect_validation_backlog, dedupe_ids
from .planning_models import IngestPlan
from .validation import ValidationService

if TYPE_CHECKING:
    from .planning import PlanningService

_SCAN_STATE_BATCH_SIZE = 200  # Metadata-only rows (no BLOBs) — can batch larger


def _normalize_plan_stage_sequence(normalized_stage_sequence: tuple[str, ...]) -> list[PlanStage]:
    return [PlanStage.from_string(stage_name) for stage_name in normalized_stage_sequence]


def _summarize_plan_stage(
    *,
    stage: str,
    normalized_stage_sequence: tuple[str, ...],
    stage_sequence: Sequence[str] | None,
) -> PlanStage:
    requested_stage = PlanStage.from_string(stage)
    if stage_sequence is None:
        return requested_stage
    if normalized_stage_sequence == expand_requested_stage(stage):
        return requested_stage
    return PlanStage.CUSTOM


async def build_ingest_plan(
    service: PlanningService,
    *,
    sources: list[Source],
    stage: str = "all",
    stage_sequence: Sequence[str] | None = None,
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
    preview: bool = False,
    force_reparse: bool = False,
) -> IngestPlan:
    source_names = [source.name for source in sources]
    db_scope_names = source_names or None

    normalized_stage_names = normalize_stage_sequence(stage=stage, stage_sequence=stage_sequence)
    normalized_plan_stage_sequence = _normalize_plan_stage_sequence(normalized_stage_names)
    summary_stage = _summarize_plan_stage(
        stage=stage,
        normalized_stage_sequence=normalized_stage_names,
        stage_sequence=stage_sequence,
    )
    has_acquire = "acquire" in normalized_stage_names
    has_parse = "parse" in normalized_stage_names
    has_materialize = "materialize" in normalized_stage_names
    has_render = "render" in normalized_stage_names
    has_index = "index" in normalized_stage_names

    if not has_acquire and not has_parse:
        conversation_count = 0
        if has_materialize or has_render or has_index:
            conversation_count = await service.backend.count_conversation_ids(source_names=db_scope_names)
        stage_counts: dict[str, int] = {}
        if has_materialize and conversation_count:
            stage_counts["materialize"] = conversation_count
        if has_render and conversation_count:
            stage_counts["render"] = conversation_count
        if has_index and conversation_count:
            stage_counts["index"] = conversation_count
        return IngestPlan(
            summary=PlanResult(
                timestamp=int(time.time()),
                stage=summary_stage,
                stage_sequence=normalized_plan_stage_sequence,
                counts=stage_counts,
                sources=source_names,
                cursors={},
            ),
            validate_raw_ids=[],
            parse_ready_raw_ids=[],
        )

    if has_parse and not has_acquire:
        validate_backlog_ids = await collect_validation_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=[],
            force_reparse=force_reparse,
        )
        parse_backlog_ids = await collect_parse_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=validate_backlog_ids,
            force_reparse=force_reparse,
        )
        reprocess_raw_ids = dedupe_ids([*validate_backlog_ids, *parse_backlog_ids])
        backlog_counts: dict[str, int] = {}
        backlog_details: dict[str, int] = {}
        if validate_backlog_ids:
            backlog_counts["validate"] = len(validate_backlog_ids)
            backlog_details["backlog_validate"] = len(validate_backlog_ids)
        if reprocess_raw_ids:
            backlog_counts["parse"] = len(reprocess_raw_ids)
            backlog_details["backlog_parse"] = len(parse_backlog_ids)
            if has_materialize:
                backlog_counts["materialize"] = len(reprocess_raw_ids)
            if has_render:
                backlog_counts["render"] = len(reprocess_raw_ids)
            if has_index:
                backlog_counts["index"] = len(reprocess_raw_ids)
        return IngestPlan(
            summary=PlanResult(
                timestamp=int(time.time()),
                stage=summary_stage,
                stage_sequence=normalized_plan_stage_sequence,
                counts=backlog_counts,
                details=backlog_details,
                sources=source_names,
                cursors={},
            ),
            validate_raw_ids=validate_backlog_ids,
            parse_ready_raw_ids=reprocess_raw_ids,
        )

    acquisition = AcquisitionService(service.backend)
    validation = ValidationService(service.backend)
    scanned_count = 0
    validate_raw_ids: list[str] = []
    parse_ready_raw_ids: list[str] = []
    plan_details: dict[str, int] = {
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

        batch_raw_ids = dedupe_ids([record.raw_id for record in pending_records])
        scanned_states = await service.repository.get_raw_conversation_states(batch_raw_ids)
        preview_records: list[RawConversationRecord] = []

        for record in pending_records:
            if record.raw_id in seen_scanned_raw_ids:
                plan_details["duplicate_raw"] += 1
                continue
            seen_scanned_raw_ids.add(record.raw_id)
            state = scanned_states.get(record.raw_id)
            if state is None:
                plan_details["new_raw"] += 1
                if has_parse:
                    validate_raw_ids.append(record.raw_id)
                    if (
                        preview
                        and preview_validation is not None
                        and len(preview_records) < max_preview_validation_records
                    ):
                        preview_records.append(record)
                continue

            plan_details["existing_raw"] += 1
            if not has_parse:
                continue

            artifact_state = RawIngestArtifactState.from_state(state)
            if artifact_state.needs_validation_backlog(force_reparse=force_reparse):
                validate_raw_ids.append(record.raw_id)
                if preview and preview_validation is not None and len(preview_records) < max_preview_validation_records:
                    preview_records.append(record)
            if artifact_state.needs_parse_backlog(force_reparse=force_reparse):
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
            force_reparse=force_reparse,
        )
        if backlog_validate_ids:
            plan_details["backlog_validate"] = len(backlog_validate_ids)
            validate_raw_ids.extend(backlog_validate_ids)

        if preview:
            if backlog_validate_ids and preview_validation is not None:
                backlog_preview_validation = await validation.validate_raw_ids(
                    raw_ids=backlog_validate_ids,
                    persist=False,
                )
                preview_validation.merge(backlog_preview_validation)
            if preview_validation is not None and preview_validation.counts["invalid"]:
                plan_details["preview_invalid"] = preview_validation.counts["invalid"]
            if preview_validation is not None and preview_validation.counts["skipped_no_schema"]:
                plan_details["preview_skipped_no_schema"] = preview_validation.counts["skipped_no_schema"]

    if has_parse:
        backlog_parse_ids = await collect_parse_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=validate_raw_ids,
            force_reparse=force_reparse,
        )
        plan_details["backlog_parse"] = len(backlog_parse_ids)
        parse_ready_raw_ids.extend(backlog_parse_ids)
        if preview_validation is not None:
            parse_ready_raw_ids.extend(preview_validation.parseable_raw_ids)
        parse_ready_raw_ids = dedupe_ids(parse_ready_raw_ids)
        validate_raw_ids = dedupe_ids(validate_raw_ids)

    final_counts: dict[str, int] = {}
    if scanned_count:
        final_counts["scan"] = scanned_count
    if plan_details["new_raw"] and has_acquire:
        final_counts["store_raw"] = plan_details["new_raw"]
    if has_parse and validate_raw_ids:
        final_counts["validate"] = len(validate_raw_ids)
    if has_parse and parse_ready_raw_ids:
        final_counts["parse"] = len(parse_ready_raw_ids)
        if has_materialize:
            final_counts["materialize"] = len(parse_ready_raw_ids)
        if has_render:
            final_counts["render"] = len(parse_ready_raw_ids)
        if has_index:
            final_counts["index"] = len(parse_ready_raw_ids)
    elif not has_parse and (has_materialize or has_render or has_index):
        conversation_count = await service.backend.count_conversation_ids(source_names=db_scope_names)
        if has_materialize and conversation_count:
            final_counts["materialize"] = conversation_count
        if has_render and conversation_count:
            final_counts["render"] = conversation_count
        if has_index and conversation_count:
            final_counts["index"] = conversation_count

    summary = PlanResult(
        timestamp=int(time.time()),
        stage=summary_stage,
        stage_sequence=normalized_plan_stage_sequence,
        counts=final_counts,
        details={key: value for key, value in plan_details.items() if value},
        sources=source_names,
        cursors=scan_result.cursors,
    )
    return IngestPlan(
        summary=summary,
        validate_raw_ids=validate_raw_ids,
        parse_ready_raw_ids=parse_ready_raw_ids,
    )


__all__ = ["build_ingest_plan"]

"""Runtime plan construction for ingest orchestration."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.config import Source
from polylogue.pipeline.run_support import expand_requested_stage, normalize_stage_sequence
from polylogue.pipeline.stage_models import ValidateResult
from polylogue.protocols import ProgressCallback
from polylogue.storage.raw.artifacts import RawIngestArtifactState
from polylogue.storage.run_state import PlanCounts, PlanDetails, PlanResult
from polylogue.storage.runtime import RawConversationRecord
from polylogue.types import PlanStage

from .acquisition import AcquisitionService
from .planning_backlog import collect_parse_backlog, collect_validation_backlog, dedupe_ids
from .planning_models import IngestPlan
from .validation import ValidationService

if TYPE_CHECKING:
    from polylogue.storage.cursor_state import CursorStatePayload

    from .planning import PlanningService

_SCAN_STATE_BATCH_SIZE = 200  # Metadata-only rows (no BLOBs) — can batch larger


@dataclass(frozen=True)
class _PlanStageFlags:
    acquire: bool
    parse: bool
    materialize: bool
    render: bool
    index: bool

    @classmethod
    def from_stage_names(cls, stage_names: tuple[str, ...]) -> _PlanStageFlags:
        return cls(
            acquire="acquire" in stage_names,
            parse="parse" in stage_names,
            materialize="materialize" in stage_names,
            render="render" in stage_names,
            index="index" in stage_names,
        )

    @property
    def has_conversation_products(self) -> bool:
        return self.materialize or self.render or self.index


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


def _apply_conversation_stage_counts(counts: PlanCounts, *, conversation_count: int, flags: _PlanStageFlags) -> None:
    if flags.materialize and conversation_count:
        counts.materialize = conversation_count
    if flags.render and conversation_count:
        counts.render = conversation_count
    if flags.index and conversation_count:
        counts.index = conversation_count


def _plan_summary(
    *,
    summary_stage: PlanStage,
    stage_sequence: list[PlanStage],
    counts: PlanCounts,
    sources: list[str],
    details: PlanDetails | None = None,
    cursors: dict[str, CursorStatePayload] | None = None,
) -> PlanResult:
    return PlanResult(
        timestamp=int(time.time()),
        stage=summary_stage,
        stage_sequence=stage_sequence,
        counts=counts,
        details=details or PlanDetails(),
        sources=sources,
        cursors=cursors or {},
    )


async def _plan_existing_conversation_stages(
    service: PlanningService,
    *,
    source_names: list[str],
    db_scope_names: list[str] | None,
    summary_stage: PlanStage,
    stage_sequence: list[PlanStage],
    flags: _PlanStageFlags,
) -> IngestPlan:
    conversation_count = 0
    if flags.has_conversation_products:
        conversation_count = await service.backend.count_conversation_ids(source_names=db_scope_names)

    counts = PlanCounts()
    _apply_conversation_stage_counts(counts, conversation_count=conversation_count, flags=flags)
    return IngestPlan(
        summary=_plan_summary(
            summary_stage=summary_stage,
            stage_sequence=stage_sequence,
            counts=counts,
            sources=source_names,
        ),
        validate_raw_ids=[],
        parse_ready_raw_ids=[],
    )


async def _plan_parse_backlog(
    service: PlanningService,
    *,
    source_names: list[str],
    db_scope_names: list[str] | None,
    summary_stage: PlanStage,
    stage_sequence: list[PlanStage],
    flags: _PlanStageFlags,
    force_reparse: bool,
) -> IngestPlan:
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
    counts = PlanCounts()
    details = PlanDetails()
    if validate_backlog_ids:
        counts.validate_count = len(validate_backlog_ids)
        details.backlog_validate = len(validate_backlog_ids)
    if reprocess_raw_ids:
        counts.parse = len(reprocess_raw_ids)
        details.backlog_parse = len(parse_backlog_ids)
        _apply_conversation_stage_counts(counts, conversation_count=len(reprocess_raw_ids), flags=flags)
    return IngestPlan(
        summary=_plan_summary(
            summary_stage=summary_stage,
            stage_sequence=stage_sequence,
            counts=counts,
            details=details,
            sources=source_names,
        ),
        validate_raw_ids=validate_backlog_ids,
        parse_ready_raw_ids=reprocess_raw_ids,
    )


async def _final_scan_counts(
    service: PlanningService,
    *,
    scanned_count: int,
    plan_details: PlanDetails,
    validate_raw_ids: list[str],
    parse_ready_raw_ids: list[str],
    db_scope_names: list[str] | None,
    flags: _PlanStageFlags,
) -> PlanCounts:
    counts = PlanCounts()
    if scanned_count:
        counts.scan = scanned_count
    if plan_details.int_value("new_raw") and flags.acquire:
        counts.store_raw = plan_details.int_value("new_raw")
    if flags.parse and validate_raw_ids:
        counts.validate_count = len(validate_raw_ids)
    if flags.parse and parse_ready_raw_ids:
        counts.parse = len(parse_ready_raw_ids)
        _apply_conversation_stage_counts(counts, conversation_count=len(parse_ready_raw_ids), flags=flags)
    elif not flags.parse and flags.has_conversation_products:
        conversation_count = await service.backend.count_conversation_ids(source_names=db_scope_names)
        _apply_conversation_stage_counts(counts, conversation_count=conversation_count, flags=flags)
    return counts


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
    stage_flags = _PlanStageFlags.from_stage_names(normalized_stage_names)
    summary_stage = _summarize_plan_stage(
        stage=stage,
        normalized_stage_sequence=normalized_stage_names,
        stage_sequence=stage_sequence,
    )

    if not stage_flags.acquire and not stage_flags.parse:
        return await _plan_existing_conversation_stages(
            service,
            source_names=source_names,
            db_scope_names=db_scope_names,
            summary_stage=summary_stage,
            stage_sequence=normalized_plan_stage_sequence,
            flags=stage_flags,
        )

    if stage_flags.parse and not stage_flags.acquire:
        return await _plan_parse_backlog(
            service,
            source_names=source_names,
            db_scope_names=db_scope_names,
            summary_stage=summary_stage,
            stage_sequence=normalized_plan_stage_sequence,
            flags=stage_flags,
            force_reparse=force_reparse,
        )

    acquisition = AcquisitionService(service.backend)
    validation = ValidationService(service.backend)
    scanned_count = 0
    validate_raw_ids: list[str] = []
    parse_ready_raw_ids: list[str] = []
    plan_details = PlanDetails(
        new_raw=0,
        existing_raw=0,
        duplicate_raw=0,
        backlog_validate=0,
        backlog_parse=0,
    )
    pending_records: list[RawConversationRecord] = []
    preview_validation = ValidateResult() if preview and stage_flags.parse else None
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
                plan_details.duplicate_raw = plan_details.int_value("duplicate_raw") + 1
                continue
            seen_scanned_raw_ids.add(record.raw_id)
            state = scanned_states.get(record.raw_id)
            if state is None:
                plan_details.new_raw = plan_details.int_value("new_raw") + 1
                if stage_flags.parse:
                    validate_raw_ids.append(record.raw_id)
                    if (
                        preview
                        and preview_validation is not None
                        and len(preview_records) < max_preview_validation_records
                    ):
                        preview_records.append(record)
                continue

            plan_details.existing_raw = plan_details.int_value("existing_raw") + 1
            if not stage_flags.parse:
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

    if stage_flags.parse:
        backlog_validate_ids = await collect_validation_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=validate_raw_ids,
            force_reparse=force_reparse,
        )
        if backlog_validate_ids:
            plan_details.backlog_validate = len(backlog_validate_ids)
            validate_raw_ids.extend(backlog_validate_ids)

        if preview:
            if backlog_validate_ids and preview_validation is not None:
                backlog_preview_validation = await validation.validate_raw_ids(
                    raw_ids=backlog_validate_ids,
                    persist=False,
                )
                preview_validation.merge(backlog_preview_validation)
            if preview_validation is not None and preview_validation.counts["invalid"]:
                plan_details.preview_invalid = preview_validation.counts["invalid"]
            if preview_validation is not None and preview_validation.counts["skipped_no_schema"]:
                plan_details.preview_skipped_no_schema = preview_validation.counts["skipped_no_schema"]

    if stage_flags.parse:
        backlog_parse_ids = await collect_parse_backlog(
            service.backend,
            source_names=db_scope_names,
            exclude_raw_ids=validate_raw_ids,
            force_reparse=force_reparse,
        )
        plan_details.backlog_parse = len(backlog_parse_ids)
        parse_ready_raw_ids.extend(backlog_parse_ids)
        if preview_validation is not None:
            parse_ready_raw_ids.extend(preview_validation.parseable_raw_ids)
        parse_ready_raw_ids = dedupe_ids(parse_ready_raw_ids)
        validate_raw_ids = dedupe_ids(validate_raw_ids)

    final_counts = await _final_scan_counts(
        service,
        scanned_count=scanned_count,
        plan_details=plan_details,
        validate_raw_ids=validate_raw_ids,
        parse_ready_raw_ids=parse_ready_raw_ids,
        db_scope_names=db_scope_names,
        flags=stage_flags,
    )
    return IngestPlan(
        summary=_plan_summary(
            summary_stage=summary_stage,
            stage_sequence=normalized_plan_stage_sequence,
            counts=final_counts,
            details=plan_details,
            sources=source_names,
            cursors=scan_result.cursors,
        ),
        validate_raw_ids=validate_raw_ids,
        parse_ready_raw_ids=parse_ready_raw_ids,
    )


__all__ = ["build_ingest_plan"]

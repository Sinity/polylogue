"""Canonical ingest planning for preview and runtime orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

from polylogue.config import Config, Source
from polylogue.protocols import ProgressCallback
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import PlanResult, RawConversationRecord

from .acquisition import AcquisitionService
from .validation import ValidateResult, ValidationService

_VALIDATE_STAGES = frozenset({"validate", "parse", "all"})
_PARSE_STAGES = frozenset({"parse", "all"})
_SCAN_STATE_BATCH_SIZE = 128


def _dedupe_ids(raw_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_id in raw_ids:
        if raw_id in seen:
            continue
        seen.add(raw_id)
        ordered.append(raw_id)
    return ordered


@dataclass
class IngestPlan:
    """Internal plan consumed by runtime execution."""

    summary: PlanResult
    validate_raw_ids: list[str]
    parse_ready_raw_ids: list[str]


class PlanningService:
    """Build canonical preview/runtime plans from source scans and DB state."""

    def __init__(self, backend: SQLiteBackend, config: Config):
        self.backend = backend
        self.config = config

    async def collect_validation_backlog(
        self,
        *,
        source_names: list[str] | None,
        exclude_raw_ids: list[str] | None = None,
    ) -> list[str]:
        """Collect unvalidated/unparsed raw IDs for the scoped sources."""
        exclude = set(exclude_raw_ids or [])
        backlog_validate_ids = await self.backend.list_raw_ids(
            source_names=source_names,
            require_unparsed=True,
            require_unvalidated=True,
        )
        return [
            raw_id
            for raw_id in backlog_validate_ids
            if raw_id not in exclude
        ]

    async def collect_parse_backlog(
        self,
        *,
        source_names: list[str] | None,
        exclude_raw_ids: list[str] | None = None,
    ) -> list[str]:
        """Collect parsed-ready raw IDs for the scoped sources."""
        exclude = set(exclude_raw_ids or [])
        backlog_parse_ids = await self.backend.list_raw_ids(
            source_names=source_names,
            require_unparsed=True,
            validation_statuses=["passed", "skipped"],
        )
        return _dedupe_ids([
            raw_id
            for raw_id in backlog_parse_ids
            if raw_id not in exclude
        ])

    async def build_plan(
        self,
        *,
        sources: list[Source],
        stage: str = "all",
        ui: object | None = None,
        progress_callback: ProgressCallback | None = None,
        preview: bool = False,
    ) -> IngestPlan:
        source_names = [source.name for source in sources]
        db_scope_names = source_names or None

        if stage == "render":
            render_ids = await self.backend.list_conversation_ids(source_names=db_scope_names)
            return IngestPlan(
                summary=PlanResult(
                    timestamp=int(time.time()),
                    stage=stage,
                    counts={"render": len(render_ids)} if render_ids else {},
                    sources=source_names,
                    cursors={},
                ),
                validate_raw_ids=[],
                parse_ready_raw_ids=[],
            )

        if stage == "index":
            index_ids = await self.backend.list_conversation_ids(source_names=db_scope_names)
            return IngestPlan(
                summary=PlanResult(
                    timestamp=int(time.time()),
                    stage=stage,
                    counts={"index": len(index_ids)} if index_ids else {},
                    sources=source_names,
                    cursors={},
                ),
                validate_raw_ids=[],
                parse_ready_raw_ids=[],
            )

        if stage == "generate-schemas":
            return IngestPlan(
                summary=PlanResult(
                    timestamp=int(time.time()),
                    stage=stage,
                    counts={},
                    sources=source_names,
                    cursors={},
                ),
                validate_raw_ids=[],
                parse_ready_raw_ids=[],
            )

        acquisition = AcquisitionService(self.backend)
        validation = ValidationService(self.backend)
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
        preview_validation = ValidateResult() if preview and stage in _VALIDATE_STAGES else None
        seen_scanned_raw_ids: set[str] = set()

        async def _flush_pending_records() -> None:
            nonlocal pending_records
            if not pending_records:
                return
            scanned_states = await self.backend.get_raw_conversation_states(
                _dedupe_ids([record.raw_id for record in pending_records])
            )
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

                if stage in _VALIDATE_STAGES:
                    current_status = state.validation_status if state is not None else None
                    parsed_at = state.parsed_at if state is not None else None
                    if parsed_at is None:
                        if current_status is None:
                            validate_raw_ids.append(record.raw_id)
                            if preview:
                                preview_records.append(record)
                        elif stage in _PARSE_STAGES and current_status in {"passed", "skipped"}:
                            parse_ready_raw_ids.append(record.raw_id)

            if preview_records and preview_validation is not None:
                scanned_preview_validation = await validation.evaluate_raw_records(
                    raw_records=preview_records,
                    persist=False,
                )
                preview_validation.merge(scanned_preview_validation)
            pending_records = []

        async def _process_record(record: RawConversationRecord) -> None:
            nonlocal scanned_count
            scanned_count += 1
            pending_records.append(record)
            if len(pending_records) >= _SCAN_STATE_BATCH_SIZE:
                await _flush_pending_records()

        scan_result = await acquisition.visit_sources(
            sources,
            progress_callback=progress_callback,
            ui=ui,
            drive_config=self.config.drive_config,
            on_record=_process_record,
        )
        await _flush_pending_records()

        if stage in _VALIDATE_STAGES:
            backlog_validate_ids = await self.collect_validation_backlog(
                source_names=db_scope_names,
                exclude_raw_ids=validate_raw_ids,
            )
            if backlog_validate_ids:
                details["backlog_validate"] = len(backlog_validate_ids)
                validate_raw_ids.extend(backlog_validate_ids)

            if preview:
                if backlog_validate_ids:
                    backlog_preview_validation = await validation.validate_raw_ids(
                        raw_ids=backlog_validate_ids,
                        persist=False,
                    )
                    preview_validation.merge(backlog_preview_validation)
                if preview_validation.counts["invalid"]:
                    details["preview_invalid"] = preview_validation.counts["invalid"]
                if preview_validation.counts["skipped_no_schema"]:
                    details["preview_skipped_no_schema"] = preview_validation.counts["skipped_no_schema"]

        if stage in _PARSE_STAGES:
            backlog_parse_ids = await self.collect_parse_backlog(
                source_names=db_scope_names,
                exclude_raw_ids=validate_raw_ids,
            )
            details["backlog_parse"] = len(backlog_parse_ids)
            parse_ready_raw_ids.extend(backlog_parse_ids)
            if preview_validation is not None:
                parse_ready_raw_ids.extend(preview_validation.parseable_raw_ids)
            parse_ready_raw_ids = _dedupe_ids(parse_ready_raw_ids)

        counts: dict[str, int] = {}
        if scanned_count:
            counts["scan"] = scanned_count
        if details["new_raw"] and stage in {"acquire", "validate", "parse", "all"}:
            counts["store_raw"] = details["new_raw"]
        if validate_raw_ids:
            counts["validate"] = len(validate_raw_ids)
        if parse_ready_raw_ids:
            counts["parse"] = len(parse_ready_raw_ids)

        summary = PlanResult(
            timestamp=int(time.time()),
            stage=stage,
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


__all__ = ["IngestPlan", "PlanningService"]

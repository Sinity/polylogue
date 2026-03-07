"""Canonical ingest planning for preview and runtime orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

from polylogue.config import Config, Source
from polylogue.protocols import ProgressCallback
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import PlanResult, RawConversationRecord

from .acquisition import AcquisitionService
from .validation import ValidationService

_VALIDATE_STAGES = frozenset({"validate", "parse", "all"})
_PARSE_STAGES = frozenset({"parse", "all"})


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
    store_records: list[RawConversationRecord]
    validate_records: list[RawConversationRecord]
    parse_ready_raw_ids: list[str]


class PlanningService:
    """Build canonical preview/runtime plans from source scans and DB state."""

    def __init__(self, backend: SQLiteBackend, config: Config):
        self.backend = backend
        self.config = config

    async def build_plan(
        self,
        *,
        sources: list[Source],
        stage: str = "all",
        ui: object | None = None,
        progress_callback: ProgressCallback | None = None,
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
                store_records=[],
                validate_records=[],
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
                store_records=[],
                validate_records=[],
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
                store_records=[],
                validate_records=[],
                parse_ready_raw_ids=[],
            )

        acquisition = AcquisitionService(self.backend)
        validation = ValidationService(self.backend)
        scan_result = await acquisition.scan_sources(
            sources,
            progress_callback=progress_callback,
            ui=ui,
            drive_config=self.config.drive_config,
        )

        scanned_records = scan_result.records
        scanned_raw_ids = [record.raw_id for record in scanned_records]
        scanned_states = await self.backend.get_raw_conversation_states(scanned_raw_ids)

        store_records: list[RawConversationRecord] = []
        validate_records: dict[str, RawConversationRecord] = {}
        parse_ready_raw_ids: list[str] = []
        details = {
            "new_raw": 0,
            "existing_raw": 0,
            "backlog_validate": 0,
            "backlog_parse": 0,
        }

        for record in scanned_records:
            state = scanned_states.get(record.raw_id)
            if state is None:
                details["new_raw"] += 1
                if stage in {"acquire", "validate", "parse", "all"}:
                    store_records.append(record)
            else:
                details["existing_raw"] += 1

            if stage in _VALIDATE_STAGES:
                current_status = state.validation_status if state is not None else None
                parsed_at = state.parsed_at if state is not None else None
                if parsed_at is None:
                    if current_status is None:
                        validate_records[record.raw_id] = record
                    elif stage in _PARSE_STAGES and current_status in {"passed", "skipped"}:
                        parse_ready_raw_ids.append(record.raw_id)

        if stage in _VALIDATE_STAGES:
            backlog_validate_ids = await self.backend.list_raw_ids(
                source_names=db_scope_names,
                require_unparsed=True,
                require_unvalidated=True,
            )
            backlog_validate_ids = [
                raw_id
                for raw_id in backlog_validate_ids
                if raw_id not in validate_records
            ]
            if backlog_validate_ids:
                backlog_records = await self.backend.get_raw_conversations_batch(backlog_validate_ids)
                details["backlog_validate"] = len(backlog_records)
                for record in backlog_records:
                    validate_records[record.raw_id] = record

            preview_validation = await validation.evaluate_raw_records(
                raw_records=list(validate_records.values()),
                persist=False,
            )
            if preview_validation.counts["invalid"]:
                details["preview_invalid"] = preview_validation.counts["invalid"]
            if preview_validation.counts["skipped_no_schema"]:
                details["preview_skipped_no_schema"] = preview_validation.counts["skipped_no_schema"]
        else:
            preview_validation = None

        if stage in _PARSE_STAGES:
            backlog_parse_ids = await self.backend.list_raw_ids(
                source_names=db_scope_names,
                require_unparsed=True,
                validation_statuses=["passed", "skipped"],
            )
            backlog_parse_ids = [
                raw_id
                for raw_id in backlog_parse_ids
                if raw_id not in validate_records
            ]
            details["backlog_parse"] = len(backlog_parse_ids)
            parse_ready_raw_ids.extend(backlog_parse_ids)
            if preview_validation is not None:
                parse_ready_raw_ids.extend(preview_validation.parseable_raw_ids)
            parse_ready_raw_ids = _dedupe_ids(parse_ready_raw_ids)

        counts: dict[str, int] = {}
        if scanned_records:
            counts["scan"] = len(scanned_records)
        if store_records:
            counts["store_raw"] = len(store_records)
        if validate_records:
            counts["validate"] = len(validate_records)
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
            store_records=store_records,
            validate_records=list(validate_records.values()),
            parse_ready_raw_ids=parse_ready_raw_ids,
        )


__all__ = ["IngestPlan", "PlanningService"]

"""Async pipeline execution flow."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.lib.json import JSONDocument, JSONValue, json_document
from polylogue.lib.metrics import PipelineMetrics
from polylogue.logging import get_logger
from polylogue.pipeline.payload_types import SiteBuildOptions
from polylogue.pipeline.run_finalization import persist_run_result
from polylogue.pipeline.run_stages import (
    IndexStageOutcome,
    execute_acquire_stage,
    execute_index_stage,
    execute_ingest_stage,
    execute_materialize_stage,
    execute_render_stage,
    execute_schema_generation_stage,
    execute_site_stage,
)
from polylogue.pipeline.run_state import RunExecutionState
from polylogue.pipeline.run_support import (
    normalize_stage_sequence,
    select_sources,
)
from polylogue.pipeline.stage_specs import (
    PipelineStageSpec,
    stage_sequence_suspends_fts,
    stage_specs_for_sequence,
    validate_stage_contract,
)
from polylogue.storage.backends import create_backend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.run_state import RunResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.run_state import PlanResult

logger = get_logger(__name__)


def _json_detail_payload(payload: object) -> JSONDocument:
    if isinstance(payload, dict):
        return json_document(dict(payload))
    return {}


def _compact_log_details(value: JSONValue) -> JSONValue:
    if isinstance(value, dict):
        compact: JSONDocument = {}
        for key, nested in value.items():
            if isinstance(nested, list):
                count_key = f"{key}_count"
                if count_key not in value:
                    compact[count_key] = len(nested)
                continue
            compact[key] = _compact_log_details(nested)
        return compact
    return value


def _compact_stage_log_payload(payload: JSONDocument) -> JSONDocument:
    """Trim oversized telemetry from normal stage-complete log lines."""
    compact = dict(payload)
    details = compact.get("details")
    if isinstance(details, dict):
        compact["details"] = _compact_log_details(details)
    return compact


async def run_sources(
    *,
    config: Config,
    stage: str = "all",
    stage_sequence: Sequence[str] | None = None,
    plan: PlanResult | None = None,
    ui: object | None = None,
    source_names: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    render_format: str = "html",
    site_options: SiteBuildOptions | None = None,
    raw_batch_size: int = 50,
    ingest_workers: int | None = None,
    measure_ingest_result_size: bool = False,
    backend: SQLiteBackend | None = None,
    repository: ConversationRepository | None = None,
) -> RunResult:
    """Run the async pipeline with stage control."""
    start = time.perf_counter()
    metrics = PipelineMetrics()
    state = RunExecutionState()

    owns_backend = backend is None
    active_backend = backend or create_backend()
    owns_repository = repository is None
    active_repository = repository or ConversationRepository(backend=active_backend)

    try:
        selected_sources = select_sources(config, source_names)
        normalized_stage_sequence = normalize_stage_sequence(stage=stage, stage_sequence=stage_sequence)
        stage_specs = stage_specs_for_sequence(normalized_stage_sequence)
        explicit_sequence = stage_sequence is not None
        executed_stages: set[str] = set()
        executed_specs: list[PipelineStageSpec] = []
        index_outcome = IndexStageOutcome(indexed=False, item_count=0)

        async def _run_acquire_stage(spec: PipelineStageSpec) -> None:
            sm = metrics.start_stage(spec.log_stage)
            acquire_result = await execute_acquire_stage(
                config=config,
                backend=active_backend,
                sources=selected_sources,
                ui=ui,
                progress_callback=progress_callback,
            )
            sm.details.update(_json_detail_payload(acquire_result.diagnostics))
            sm.stop(items=acquire_result.counts["acquired"])
            state.record_acquire(acquire_result)
            logger.info("Acquire stage complete", **sm.to_dict(), **acquire_result.counts)

        async def _run_schema_stage(spec: PipelineStageSpec) -> None:
            sm = metrics.start_stage(spec.log_stage)
            schema_outcome = await execute_schema_generation_stage()
            state.record_schema_generation(
                generated=schema_outcome.generated,
                failed=schema_outcome.failed,
            )
            sm.stop(items=schema_outcome.generated)
            logger.info(
                "Schema generation complete",
                **sm.to_dict(),
                generated=schema_outcome.generated,
                failed=schema_outcome.failed,
            )

        async def _run_parse_stage(spec: PipelineStageSpec, ingest_stage: str) -> None:
            sm = metrics.start_stage(spec.log_stage)
            ingest_result = await execute_ingest_stage(
                config=config,
                repository=active_repository,
                archive_root=config.archive_root,
                sources=selected_sources,
                stage=ingest_stage,
                skip_acquire=True,
                ui=ui,
                progress_callback=progress_callback,
                raw_batch_size=raw_batch_size,
                ingest_workers=ingest_workers,
                measure_ingest_result_size=measure_ingest_result_size,
            )
            sm.sub_timings.update({f"{k}_s": v for k, v in ingest_result.timings.items()})
            sm.details.update(_json_detail_payload(ingest_result.diagnostics))
            sm.stop(items=len(ingest_result.parse_raw_ids))
            if "acquire" not in executed_stages:
                state.record_acquire(ingest_result.acquire_result)
            ingest_log_payload = _compact_stage_log_payload(sm.to_dict())
            logger.info(
                "Ingest complete",
                **ingest_log_payload,
                **ingest_result.acquire_result.counts,
            )

            validation_result = ingest_result.validation_result
            if validation_result is not None:
                state.record_validation(validation_result)

            state.record_parse(ingest_result.parse_result)
            logger.info(
                "Parse stage complete",
                processed_ids=len(state.processed_ids),
                parse_failures=ingest_result.parse_result.parse_failures,
            )

        async def _run_materialize_stage(spec: PipelineStageSpec, materialize_stage: str) -> None:
            sm = metrics.start_stage(spec.log_stage)
            materialize_outcome = await execute_materialize_stage(
                stage=materialize_stage,
                source_names=source_names,
                processed_ids=state.processed_ids,
                backend=active_backend,
                progress_callback=progress_callback,
            )
            if materialize_outcome.observation:
                sm.details.update(_json_detail_payload(materialize_outcome.observation))
            state.record_materialize(materialized=materialize_outcome.item_count)
            sm.stop(items=materialize_outcome.item_count)
            materialize_log_payload = _compact_stage_log_payload(sm.to_dict())
            logger.info(
                "Materialize stage complete",
                **materialize_log_payload,
                rebuilt=materialize_outcome.rebuilt,
            )

        async def _run_render_stage(spec: PipelineStageSpec, render_stage: str) -> None:
            sm = metrics.start_stage(spec.log_stage)
            render_outcome = await execute_render_stage(
                config=config,
                backend=active_backend,
                stage=render_stage,
                source_names=source_names,
                processed_ids=state.processed_ids,
                progress_callback=progress_callback,
                render_format=render_format,
            )
            state.record_render(
                rendered=render_outcome.rendered_count,
                failures=render_outcome.failures,
            )
            if render_outcome.observation:
                sm.details.update(_json_detail_payload(render_outcome.observation))
            sm.stop(items=state.counts.get("rendered", 0))
            logger.info(
                "Render stage complete",
                **sm.to_dict(),
                failures=len(render_outcome.failures),
                total=render_outcome.total,
            )

        async def _run_index_stage(spec: PipelineStageSpec, index_stage: str) -> IndexStageOutcome:
            sm = metrics.start_stage(spec.log_stage)
            next_index_outcome = await execute_index_stage(
                config=config,
                stage=index_stage,
                source_names=source_names,
                processed_ids=state.processed_ids,
                backend=active_backend,
                progress_callback=progress_callback,
            )
            if next_index_outcome.error is not None:
                logger.error("Indexing failed", error=next_index_outcome.error)
            sm.stop(items=next_index_outcome.item_count)
            logger.info(
                "Index stage complete",
                **sm.to_dict(),
                indexed=next_index_outcome.indexed,
            )
            return next_index_outcome

        async def _run_site_stage(spec: PipelineStageSpec) -> None:
            sm = metrics.start_stage(spec.log_stage)
            site_outcome = await execute_site_stage(
                backend=active_backend,
                repository=active_repository,
                site_options=site_options,
                progress_callback=progress_callback,
            )
            if site_outcome.error is not None:
                logger.error("Site build failed", error=site_outcome.error)
            sm.stop(items=site_outcome.rendered_pages)
            logger.info(
                "Site stage complete",
                **sm.to_dict(),
                conversations=site_outcome.conversations,
                index_pages=site_outcome.index_pages,
                rendered_pages=site_outcome.rendered_pages,
            )

        async def _run_stage_spec(spec: PipelineStageSpec) -> None:
            nonlocal index_outcome
            if not spec.pipeline_managed:
                executed_stages.add(spec.name)
                executed_specs.append(spec)
                return
            validate_stage_contract(spec, executed_specs=executed_specs)
            execution_stage = spec.execution_stage(
                requested_stage=stage,
                explicit_sequence=explicit_sequence,
                executed_stages=executed_stages,
            )
            if spec.name == "acquire":
                await _run_acquire_stage(spec)
            elif spec.name == "schema":
                await _run_schema_stage(spec)
            elif spec.name == "parse":
                await _run_parse_stage(spec, execution_stage)
            elif spec.name == "materialize":
                await _run_materialize_stage(spec, execution_stage)
            elif spec.name == "render":
                await _run_render_stage(spec, execution_stage)
            elif spec.name == "site":
                await _run_site_stage(spec)
            elif spec.name == "index":
                index_outcome = await _run_index_stage(spec, execution_stage)
            else:
                raise ValueError(f"Unknown pipeline stage spec: {spec.name}")
            executed_stages.add(spec.name)
            executed_specs.append(spec)

        # Suspend FTS triggers during bulk pipeline operations.
        # Triggers fire per-row during message INSERTs, causing massive
        # overhead (~8s per 50 updates with realistic text). The index
        # stage rebuilds FTS at the end anyway, making trigger updates
        # pure waste during ingest.
        if stage_sequence_suspends_fts(stage_specs):
            from polylogue.storage.fts.fts_lifecycle import suspend_fts_triggers_async

            async with active_backend.connection() as conn:
                await suspend_fts_triggers_async(conn)
                await conn.commit()

        for stage_spec in stage_specs:
            await _run_stage_spec(stage_spec)

        return await persist_run_result(
            config=config,
            repository=active_repository,
            plan=plan,
            state=state,
            metrics=metrics,
            index_outcome=index_outcome,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
    finally:
        # Restore FTS triggers that were suspended for bulk operations.
        # Failure here leaves the search index in a degraded state until the
        # next full rebuild; surface it loudly so operators see degraded state
        # in logs rather than discovering stale results later.
        try:
            from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_async

            async with active_backend.connection() as conn:
                await restore_fts_triggers_async(conn)
                await conn.commit()
        except Exception:
            logger.exception(
                "FTS trigger restoration failed; search index may return stale "
                "results until the next 'devtools verify' or 'polylogue run index' rebuild",
            )
        if owns_repository:
            await active_repository.close()
        elif owns_backend:
            await active_backend.close()


__all__ = ["run_sources"]

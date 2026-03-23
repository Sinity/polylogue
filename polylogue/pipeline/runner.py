"""Async pipeline runner logic."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

from polylogue.config import Config
from polylogue.lib.metrics import PipelineMetrics
from polylogue.logging import get_logger
from polylogue.pipeline.run_stages import (
    execute_acquire_stage,
    execute_index_stage,
    execute_ingest_stage,
    execute_render_stage,
    execute_schema_generation_stage,
)
from polylogue.pipeline.run_state import RunExecutionState
from polylogue.pipeline.run_support import (
    INGEST_STAGES,
    PARSE_STAGES,
    RENDER_STAGES,
    RUN_STAGE_CHOICES,
    run_coroutine_sync,
    select_sources,
    write_run_json,
)
from polylogue.storage.backends import create_backend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import PlanResult, RunRecord, RunResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

_select_sources = select_sources
_run_coroutine_sync = run_coroutine_sync
_write_run_json = write_run_json


def _needs_parse_source_fallback(stage: str, sources, ingest_result) -> bool:
    return bool(
        stage == "parse"
        and sources
        and not ingest_result.acquire_result.raw_ids
        and ingest_result.validation_result is None
        and not ingest_result.parse_raw_ids
        and not ingest_result.parse_result.processed_ids
    )


def plan_sources(
    config: Config,
    *,
    stage: str = "all",
    ui: object | None = None,
    source_names: Sequence[str] | None = None,
    backend: SQLiteBackend | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PlanResult:
    """Build a canonical preview plan without writing pipeline state."""
    from polylogue.pipeline.services.planning import PlanningService

    async def _build() -> PlanResult:
        planner = PlanningService(backend=active_backend, config=config)
        plan = await planner.build_plan(
            sources=_select_sources(config, source_names),
            stage=stage,
            ui=ui,
            progress_callback=progress_callback,
            preview=True,
        )
        return plan.summary

    owns_backend = backend is None
    active_backend = backend or create_backend()
    try:
        return _run_coroutine_sync(_build())
    finally:
        if owns_backend:
            _run_coroutine_sync(active_backend.close())


async def run_sources(
    *,
    config: Config,
    stage: str = "all",
    plan: PlanResult | None = None,
    ui: object | None = None,
    source_names: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    render_format: str = "html",
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
        selected_sources = _select_sources(config, source_names)

        if stage == "acquire":
            sm = metrics.start_stage("acquire")
            acquire_result = await execute_acquire_stage(
                config=config,
                backend=active_backend,
                sources=selected_sources,
                ui=ui,
                progress_callback=progress_callback,
            )
            sm.stop(items=acquire_result.counts["acquired"])
            state.record_acquire(acquire_result)
            logger.info("Acquire stage complete", **sm.to_dict(), **acquire_result.counts)

        elif stage in INGEST_STAGES:
            ingest_result = await execute_ingest_stage(
                config=config,
                repository=active_repository,
                archive_root=config.archive_root,
                sources=selected_sources,
                stage=stage,
                ui=ui,
                progress_callback=progress_callback,
            )
            if _needs_parse_source_fallback(stage, selected_sources, ingest_result):
                ingest_result = await execute_ingest_stage(
                    config=config,
                    repository=active_repository,
                    archive_root=config.archive_root,
                    sources=selected_sources,
                    stage="all",
                    ui=ui,
                    progress_callback=progress_callback,
                )
            state.record_acquire(ingest_result.acquire_result)
            logger.info("Acquire stage complete", **ingest_result.acquire_result.counts)

            validation_result = ingest_result.validation_result
            if validation_result is not None:
                state.record_validation(validation_result)
                logger.info(
                    "Validate stage complete",
                    parseable=len(validation_result.parseable_raw_ids),
                    invalid=validation_result.counts["invalid"],
                    drift=validation_result.counts["drift"],
                    skipped_no_schema=validation_result.counts["skipped_no_schema"],
                    errors=validation_result.counts["errors"],
                )

            if stage in PARSE_STAGES:
                state.record_parse(ingest_result.parse_result)
                logger.info(
                    "Parse stage complete",
                    processed_ids=len(state.processed_ids),
                    parse_failures=ingest_result.parse_result.parse_failures,
                )

        if stage == "generate-schemas":
            stage_t0 = time.perf_counter()
            schema_outcome = await execute_schema_generation_stage()
            state.record_schema_generation(
                generated=schema_outcome.generated,
                failed=schema_outcome.failed,
            )
            logger.info(
                "Schema generation complete",
                elapsed_s=round(time.perf_counter() - stage_t0, 1),
                generated=schema_outcome.generated,
                failed=schema_outcome.failed,
            )

        if stage in RENDER_STAGES:
            sm = metrics.start_stage("render")
            render_outcome = await execute_render_stage(
                config=config,
                backend=active_backend,
                stage=stage,
                source_names=source_names,
                processed_ids=state.processed_ids,
                progress_callback=progress_callback,
                render_format=render_format,
            )
            state.record_render(
                rendered=render_outcome.rendered_count,
                failures=render_outcome.failures,
            )
            sm.stop(items=state.counts.get("rendered", 0))
            logger.info(
                "Render stage complete",
                **sm.to_dict(),
                failures=len(render_outcome.failures),
                total=render_outcome.total,
            )

        sm = metrics.start_stage("index")
        index_outcome = await execute_index_stage(
            config=config,
            stage=stage,
            source_names=source_names,
            processed_ids=state.processed_ids,
            backend=active_backend,
            progress_callback=progress_callback,
        )
        if index_outcome.error is not None:
            logger.error("Indexing failed", error=index_outcome.error)
        sm.stop(items=index_outcome.item_count)
        logger.info(
            "Index stage complete",
            **sm.to_dict(),
            indexed=index_outcome.indexed,
        )

        duration_ms = int((time.perf_counter() - start) * 1000)
        drift = state.finalize()

        run_id = uuid4().hex
        run_payload = {
            "run_id": run_id,
            "timestamp": int(time.time()),
            "counts": state.counts,
            "drift": drift,
            "indexed": index_outcome.indexed,
            "index_error": index_outcome.error,
            "duration_ms": duration_ms,
            "metrics": metrics.to_summary(),
        }
        _write_run_json(config.archive_root, run_payload)

        await active_repository.record_run(
            RunRecord(
                run_id=run_id,
                timestamp=str(run_payload["timestamp"]),
                plan_snapshot=plan.model_dump() if plan else None,
                counts=state.counts,
                drift=drift,
                indexed=index_outcome.indexed,
                duration_ms=duration_ms,
            ),
        )

        return RunResult(
            run_id=run_id,
            counts=state.counts,
            drift=drift,
            indexed=index_outcome.indexed,
            index_error=index_outcome.error,
            duration_ms=duration_ms,
            render_failures=state.render_failures,
        )
    finally:
        if owns_repository:
            await active_repository.close()
        elif owns_backend:
            await active_backend.close()


async def latest_run(backend: SQLiteBackend | None = None) -> RunRecord | None:
    """Fetch the most recent run record from the database asynchronously."""
    owns_backend = backend is None
    active_backend = backend or create_backend()
    try:
        return await active_backend.queries.get_latest_run()
    finally:
        if owns_backend:
            await active_backend.close()


__all__ = [
    "RUN_STAGE_CHOICES",
    "latest_run",
    "plan_sources",
    "run_sources",
]

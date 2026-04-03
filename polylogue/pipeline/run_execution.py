"""Async pipeline execution flow."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.lib.metrics import PipelineMetrics
from polylogue.logging import get_logger
from polylogue.pipeline.run_finalization import persist_run_result
from polylogue.pipeline.run_stages import (
    IndexStageOutcome,
    execute_acquire_stage,
    execute_index_stage,
    execute_ingest_stage,
    execute_materialize_stage,
    execute_render_stage,
    execute_schema_generation_stage,
)
from polylogue.pipeline.run_state import RunExecutionState
from polylogue.pipeline.run_support import (
    expand_requested_stage,
    select_sources,
)
from polylogue.storage.backends import create_backend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.state_views import RunResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.state_views import PlanResult

logger = get_logger(__name__)


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
        selected_sources = select_sources(config, source_names)
        stage_sequence = expand_requested_stage(stage)
        executed_stages: set[str] = set()

        # Suspend FTS triggers during bulk pipeline operations.
        # Triggers fire per-row during message INSERTs, causing massive
        # overhead (~8s per 50 updates with realistic text). The index
        # stage rebuilds FTS at the end anyway, making trigger updates
        # pure waste during ingest.
        if any(stage_name in {"parse", "render", "index"} for stage_name in stage_sequence):
            from polylogue.storage.fts_lifecycle import suspend_fts_triggers_async

            async with active_backend.connection() as conn:
                await suspend_fts_triggers_async(conn)
                await conn.commit()

        if "acquire" in stage_sequence:
            sm = metrics.start_stage("acquire")
            acquire_result = await execute_acquire_stage(
                config=config,
                backend=active_backend,
                sources=selected_sources,
                ui=ui,
                progress_callback=progress_callback,
            )
            sm.details.update(acquire_result.diagnostics)
            sm.stop(items=acquire_result.counts["acquired"])
            state.record_acquire(acquire_result)
            logger.info("Acquire stage complete", **sm.to_dict(), **acquire_result.counts)
            executed_stages.add("acquire")

        if "generate-schemas" in stage_sequence:
            sm = metrics.start_stage("generate-schemas")
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
            executed_stages.add("generate-schemas")

        if "parse" in stage_sequence:
            sm = metrics.start_stage("ingest")
            skip_acquire = True
            ingest_result = await execute_ingest_stage(
                config=config,
                repository=active_repository,
                archive_root=config.archive_root,
                sources=selected_sources,
                stage=stage,
                skip_acquire=skip_acquire,
                ui=ui,
                progress_callback=progress_callback,
            )
            sm.sub_timings.update({
                f"{k}_s": v for k, v in ingest_result.timings.items()
            })
            sm.details.update(ingest_result.diagnostics)
            sm.stop(items=len(ingest_result.parse_raw_ids))
            if "acquire" not in executed_stages:
                state.record_acquire(ingest_result.acquire_result)
            logger.info(
                "Ingest complete",
                **sm.to_dict(),
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
            executed_stages.add("parse")

        if "materialize" in stage_sequence:
            sm = metrics.start_stage("materialize")
            materialize_outcome = await execute_materialize_stage(
                stage=stage,
                source_names=source_names,
                processed_ids=state.processed_ids,
                backend=active_backend,
                progress_callback=progress_callback,
            )
            if materialize_outcome.observation:
                sm.details.update(materialize_outcome.observation)
            state.record_materialize(materialized=materialize_outcome.item_count)
            sm.stop(items=materialize_outcome.item_count)
            logger.info(
                "Materialize stage complete",
                **sm.to_dict(),
                rebuilt=materialize_outcome.rebuilt,
            )
            executed_stages.add("materialize")

        if "render" in stage_sequence:
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
            executed_stages.add("render")

        if "index" in stage_sequence:
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
            executed_stages.add("index")
        else:
            index_outcome = IndexStageOutcome(indexed=False, item_count=0)

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
        # Restore FTS triggers that were suspended for bulk operations
        try:
            from polylogue.storage.fts_lifecycle import restore_fts_triggers_async

            async with active_backend.connection() as conn:
                await restore_fts_triggers_async(conn)
                await conn.commit()
        except Exception:
            pass  # Don't fail on trigger restore — index rebuild handles FTS
        if owns_repository:
            await active_repository.close()
        elif owns_backend:
            await active_backend.close()


__all__ = ["run_sources"]

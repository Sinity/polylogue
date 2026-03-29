"""Async pipeline execution flow."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.lib.metrics import PipelineMetrics
from polylogue.logging import get_logger
from polylogue.pipeline.run_finalization import persist_run_result
from polylogue.pipeline.run_stages import (
    execute_acquire_stage,
    execute_index_stage,
    execute_ingest_stage,
    execute_render_stage,
    execute_schema_generation_stage,
)
from polylogue.pipeline.run_state import RunExecutionState
from polylogue.pipeline.run_support import INGEST_STAGES, PARSE_STAGES, RENDER_STAGES, select_sources
from polylogue.storage.backends import create_backend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.state_views import RunResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.state_views import PlanResult

logger = get_logger(__name__)


def needs_parse_source_fallback(stage: str, sources, ingest_result) -> bool:
    """Return whether parse-stage replay should widen back to the full ingest flow."""
    return bool(
        stage == "parse"
        and sources
        and not ingest_result.acquire_result.raw_ids
        and ingest_result.validation_result is None
        and not ingest_result.parse_raw_ids
        and not ingest_result.parse_result.processed_ids
    )


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

        # Suspend FTS triggers during bulk pipeline operations.
        # Triggers fire per-row during message INSERTs, causing massive
        # overhead (~8s per 50 updates with realistic text). The index
        # stage rebuilds FTS at the end anyway, making trigger updates
        # pure waste during ingest.
        if stage in ("all", "parse", "render", "index") or stage in INGEST_STAGES:
            from polylogue.storage.fts_lifecycle import suspend_fts_triggers_async

            async with active_backend.connection() as conn:
                await suspend_fts_triggers_async(conn)
                await conn.commit()

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
            sm = metrics.start_stage("ingest")
            ingest_result = await execute_ingest_stage(
                config=config,
                repository=active_repository,
                archive_root=config.archive_root,
                sources=selected_sources,
                stage=stage,
                ui=ui,
                progress_callback=progress_callback,
            )
            if needs_parse_source_fallback(stage, selected_sources, ingest_result):
                ingest_result = await execute_ingest_stage(
                    config=config,
                    repository=active_repository,
                    archive_root=config.archive_root,
                    sources=selected_sources,
                    stage="all",
                    ui=ui,
                    progress_callback=progress_callback,
                )
            sm.sub_timings.update({
                f"{k}_s": v for k, v in ingest_result.timings.items()
            })
            sm.stop(items=len(ingest_result.parse_raw_ids))
            state.record_acquire(ingest_result.acquire_result)
            logger.info(
                "Ingest complete",
                **sm.to_dict(),
                **ingest_result.acquire_result.counts,
            )

            validation_result = ingest_result.validation_result
            if validation_result is not None:
                state.record_validation(validation_result)

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


__all__ = ["needs_parse_source_fallback", "run_sources"]

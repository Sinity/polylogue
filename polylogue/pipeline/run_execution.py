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
    execute_embed_stage,
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
    stage_sequence: Sequence[str] | None = None,
    plan: PlanResult | None = None,
    ui: object | None = None,
    source_names: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    render_format: str = "html",
    site_options: dict[str, object] | None = None,
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
        executed_stages: set[str] = set()
        index_outcome = IndexStageOutcome(indexed=False, item_count=0)

        async def _run_acquire_stage() -> None:
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

        async def _run_schema_stage() -> None:
            sm = metrics.start_stage("schema")
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
            executed_stages.add("schema")

        async def _run_parse_stage(ingest_stage: str) -> None:
            sm = metrics.start_stage("ingest")
            ingest_result = await execute_ingest_stage(
                config=config,
                repository=active_repository,
                archive_root=config.archive_root,
                sources=selected_sources,
                stage=ingest_stage,
                skip_acquire=True,
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

        async def _run_materialize_stage(materialize_stage: str) -> None:
            sm = metrics.start_stage("materialize")
            materialize_outcome = await execute_materialize_stage(
                stage=materialize_stage,
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

        async def _run_render_stage(render_stage: str) -> None:
            sm = metrics.start_stage("render")
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
            sm.stop(items=state.counts.get("rendered", 0))
            logger.info(
                "Render stage complete",
                **sm.to_dict(),
                failures=len(render_outcome.failures),
                total=render_outcome.total,
            )
            executed_stages.add("render")

        async def _run_index_stage(index_stage: str) -> IndexStageOutcome:
            sm = metrics.start_stage("index")
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
            executed_stages.add("index")
            return next_index_outcome

        async def _run_site_stage() -> None:
            sm = metrics.start_stage("site")
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
            executed_stages.add("site")

        def _explicit_leaf_stage_context(leaf_stage: str) -> str:
            if leaf_stage == "parse":
                return "parse"
            if leaf_stage in {"materialize", "render", "index"} and "parse" in executed_stages:
                return "all"
            return leaf_stage

        # Suspend FTS triggers during bulk pipeline operations.
        # Triggers fire per-row during message INSERTs, causing massive
        # overhead (~8s per 50 updates with realistic text). The index
        # stage rebuilds FTS at the end anyway, making trigger updates
        # pure waste during ingest.
        if any(stage_name in {"parse", "render", "index"} for stage_name in normalized_stage_sequence):
            from polylogue.storage.fts_lifecycle import suspend_fts_triggers_async

            async with active_backend.connection() as conn:
                await suspend_fts_triggers_async(conn)
                await conn.commit()

        if stage_sequence is None:
            if "acquire" in normalized_stage_sequence:
                await _run_acquire_stage()

            if "schema" in normalized_stage_sequence:
                await _run_schema_stage()

            if "parse" in normalized_stage_sequence:
                await _run_parse_stage(stage)

            if "materialize" in normalized_stage_sequence:
                await _run_materialize_stage(stage)

            if "render" in normalized_stage_sequence:
                await _run_render_stage(stage)

            if "site" in normalized_stage_sequence:
                await _run_site_stage()

            if "index" in normalized_stage_sequence:
                index_outcome = await _run_index_stage(stage)
        else:
            for leaf_stage in normalized_stage_sequence:
                if leaf_stage == "acquire":
                    await _run_acquire_stage()
                    continue
                if leaf_stage == "schema":
                    await _run_schema_stage()
                    continue
                if leaf_stage == "parse":
                    await _run_parse_stage(_explicit_leaf_stage_context(leaf_stage))
                    continue
                if leaf_stage == "materialize":
                    await _run_materialize_stage(_explicit_leaf_stage_context(leaf_stage))
                    continue
                if leaf_stage == "render":
                    await _run_render_stage(_explicit_leaf_stage_context(leaf_stage))
                    continue
                if leaf_stage == "site":
                    await _run_site_stage()
                    continue
                if leaf_stage == "index":
                    index_outcome = await _run_index_stage(_explicit_leaf_stage_context(leaf_stage))

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

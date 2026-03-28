"""Async pipeline runner logic.

Async/await version of the pipeline runner with support for:
- Async acquisition, parsing, rendering, and indexing stages
- Progress callbacks for long-running operations
- Full pipeline orchestration with stage control
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

from polylogue.config import Config, Source
from polylogue.lib.json import dumps
from polylogue.lib.log import get_logger
from polylogue.lib.metrics import PipelineMetrics
from polylogue.protocols import ProgressCallback
from polylogue.storage.backends import SQLiteBackend, create_backend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import PlanResult, RunRecord, RunResult

if TYPE_CHECKING:
    from polylogue.pipeline.services.indexing import IndexService

T = TypeVar("T")

logger = get_logger(__name__)

RUN_STAGE_CHOICES: tuple[str, ...] = (
    "acquire",
    "validate",
    "parse",
    "render",
    "index",
    "generate-schemas",
    "all",
)
_INGEST_STAGES = frozenset({"validate", "parse", "all"})
_PARSE_STAGES = frozenset({"parse", "all"})
_RENDER_STAGES = frozenset({"render", "all"})


def _select_sources(config: Config, source_names: Sequence[str] | None) -> list[Source]:
    """Select sources from config, filtering by names if provided.

    Args:
        config: Application configuration
        source_names: Optional list of source names to include

    Returns:
        List of selected sources
    """
    if not source_names:
        return list(config.sources)
    name_set = set(source_names)
    return [source for source in config.sources if source.name in name_set]


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


def _run_coroutine_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code, even when already inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[T] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - re-raised on caller thread
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    if not result:
        raise RuntimeError("Coroutine thread completed without returning a result")
    return result[0]


async def _all_conversation_ids(
    backend: SQLiteBackend,
    source_names: Sequence[str] | None = None,
) -> list[str]:
    """Fetch all conversation IDs from database, optionally filtered by source names.

    Args:
        backend: SQLiteBackend instance
        source_names: Optional list of source names to filter by

    Returns:
        List of conversation IDs
    """
    return await backend.list_conversation_ids(
        source_names=list(source_names) if source_names is not None else None
    )


async def _run_index_stage(
    *,
    stage: str,
    source_names: Sequence[str] | None,
    processed_ids: set[str],
    backend: SQLiteBackend,
    index_service: IndexService,
    progress_callback: Callable[..., None] | None = None,
) -> tuple[bool, int]:
    """Execute index behavior for `index`/`all` stages.

    Returns:
        `(indexed, item_count)` where item_count is used for stage metrics.
    """
    if stage == "index":
        if progress_callback is not None:
            progress_callback(0, desc="Indexing")
        if source_names:
            ids = await _all_conversation_ids(backend, source_names)
            if ids:
                return await index_service.update_index(ids), len(ids)
            return await index_service.ensure_index_exists(), 0
        return await index_service.rebuild_index(), 0

    if stage == "all":
        idx = await index_service.get_index_status()
        if not idx["exists"]:
            if progress_callback is not None:
                progress_callback(0, desc="Indexing (rebuild)")
            return await index_service.rebuild_index(), len(processed_ids)
        if processed_ids:
            if progress_callback is not None:
                progress_callback(0, desc=f"Indexing: {len(processed_ids)} conversations")
            return await index_service.update_index(list(processed_ids)), len(processed_ids)

    return False, 0


def _write_run_json(archive_root: Path, payload: dict[str, object]) -> Path:
    """Write run result JSON to the runs directory.

    Args:
        archive_root: Root directory for archived data
        payload: Run result payload to write

    Returns:
        Path to the written JSON file
    """
    runs_dir = archive_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload.get("run_id", "unknown")
    run_path = runs_dir / f"run-{payload['timestamp']}-{run_id}.json"
    run_path.write_text(dumps(payload, option=None), encoding="utf-8")
    return run_path


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
    """Run the async pipeline with stage control.

    Async version of run_sources supporting:
    - Parallel acquisition, parsing, rendering, and indexing
    - Progress callbacks for long-running operations
    - Full stage control ("acquire", "validate", "parse", "render", "index", "generate-schemas", "all")

    Args:
        config: Application configuration
        stage: Pipeline stage ("acquire", "validate", "parse", "render", "index", "generate-schemas", "all")
        plan: Optional plan result for drift detection
        ui: Optional UI object for user interaction
        source_names: Optional list of source names to process
        progress_callback: Optional callback(count, desc=...) for progress updates
        render_format: Output format for rendering ("markdown" or "html", default: "html")

    Returns:
        RunResult with counts, drift, indexing status, and metadata
    """

    start = time.perf_counter()
    metrics = PipelineMetrics()

    owns_backend = backend is None
    active_backend = backend or create_backend()
    owns_repository = repository is None
    active_repository = repository or ConversationRepository(backend=active_backend)

    # Track counts for reporting
    counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
        "rendered": 0,
    }
    changed_counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
    }
    processed_ids: set[str] = set()
    render_failures: list[dict[str, str]] = []

    try:
        # Acquire stage (raw storage only)
        if stage == "acquire":
            from polylogue.pipeline.services.acquisition import AcquisitionService

            sm = metrics.start_stage("acquire")
            acquire_service = AcquisitionService(backend=active_backend)
            sources = _select_sources(config, source_names)
            acquire_result = await acquire_service.acquire_sources(
                sources,
                ui=ui,
                progress_callback=progress_callback,
                drive_config=config.drive_config,
            )
            sm.stop(items=acquire_result.counts["acquired"])
            counts["acquired"] = acquire_result.counts["acquired"]
            counts["skipped"] = acquire_result.counts["skipped"]
            logger.info("Acquire stage complete", **sm.to_dict(), **acquire_result.counts)

        # Validate/Parse stages (canonical ingest orchestration)
        elif stage in _INGEST_STAGES:
            from polylogue.pipeline.services.parsing import ParsingService

            sources = _select_sources(config, source_names)
            parsing_service = ParsingService(
                repository=active_repository,
                archive_root=config.archive_root,
                config=config,
            )

            ingest_result = await parsing_service.ingest_sources(
                sources=sources,
                stage=stage,
                ui=ui,
                progress_callback=progress_callback,
                parse_records=stage in _PARSE_STAGES,
            )
            acquire_result = ingest_result.acquire_result
            validation_result = ingest_result.validation_result

            counts["acquired"] = acquire_result.counts["acquired"]
            counts["skipped"] = acquire_result.counts["skipped"]
            counts["acquire_errors"] = acquire_result.counts["errors"]
            logger.info("Acquire stage complete", **acquire_result.counts)

            if validation_result is not None:
                counts["validated"] = validation_result.counts["validated"]
                counts["validation_invalid"] = validation_result.counts["invalid"]
                counts["validation_drift"] = validation_result.counts["drift"]
                counts["validation_skipped_no_schema"] = validation_result.counts["skipped_no_schema"]
                counts["validation_errors"] = validation_result.counts["errors"]
                logger.info(
                    "Validate stage complete",
                    parseable=len(validation_result.parseable_raw_ids),
                    invalid=validation_result.counts["invalid"],
                    drift=validation_result.counts["drift"],
                    skipped_no_schema=validation_result.counts["skipped_no_schema"],
                    errors=validation_result.counts["errors"],
                )

            if stage in _PARSE_STAGES:
                parse_result = ingest_result.parse_result
                for key, value in parse_result.counts.items():
                    counts[key] = value
                if parse_result.parse_failures:
                    counts["parse_failures"] = parse_result.parse_failures
                changed_counts.update(parse_result.changed_counts)
                processed_ids = parse_result.processed_ids
                counts["conversations"] = len(processed_ids)
                logger.info(
                    "Parse stage complete",
                    processed_ids=len(processed_ids),
                    parse_failures=parse_result.parse_failures,
                )

        if stage == "generate-schemas":
            from polylogue.paths import db_path as _db_path
            from polylogue.schemas.schema_inference import generate_all_schemas

            stage_t0 = time.perf_counter()
            output_dir = config.archive_root.parent / "schemas"
            results = await asyncio.to_thread(
                generate_all_schemas,
                output_dir=output_dir,
                db_path=_db_path(),
            )
            counts["schemas_generated"] = sum(1 for r in results if r.success)
            counts["schemas_failed"] = sum(1 for r in results if not r.success)
            logger.info(
                "Schema generation complete",
                elapsed_s=round(time.perf_counter() - stage_t0, 1),
                generated=counts["schemas_generated"],
                failed=counts["schemas_failed"],
            )

        if stage in _RENDER_STAGES:
            from polylogue.pipeline.services.rendering import RenderService
            from polylogue.rendering.renderers import create_renderer

            sm = metrics.start_stage("render")
            ids = (
                await _all_conversation_ids(active_backend, source_names)
                if stage == "render"
                else list(processed_ids)
            )
            if ids:
                if progress_callback is not None:
                    progress_callback(0, desc=f"Rendering: 0/{len(ids)}")
                renderer = create_renderer(
                    format=render_format,
                    config=config,
                    backend=active_backend,
                )
                render_service = RenderService(
                    renderer=renderer,
                    render_root=config.archive_root / "render",
                    backend=active_backend,
                )
                render_result = await render_service.render_conversations(
                    ids,
                    progress_callback=progress_callback,
                )
                counts["rendered"] = render_result.rendered_count
                render_failures = render_result.failures
                if render_failures:
                    counts["render_failures"] = len(render_failures)
            sm.stop(items=counts.get("rendered", 0))
            logger.info(
                "Render stage complete",
                **sm.to_dict(),
                failures=len(render_failures),
                total=len(ids) if ids else 0,
            )

        indexed = False
        index_error: str | None = None

        from polylogue.pipeline.services.indexing import IndexService

        index_service = IndexService(config=config, backend=active_backend)

        sm = metrics.start_stage("index")
        index_items = 0
        try:
            indexed, index_items = await _run_index_stage(
                stage=stage,
                source_names=source_names,
                processed_ids=processed_ids,
                backend=active_backend,
                index_service=index_service,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            logger.error("Indexing failed", error=str(exc), exc_info=True)
            index_error = f"{type(exc).__name__}: {exc}"
            indexed = False
        sm.stop(items=index_items)
        logger.info("Index stage complete", **sm.to_dict(), indexed=indexed)

        duration_ms = int((time.perf_counter() - start) * 1000)
        drift = {
            "new": {"conversations": 0, "messages": 0, "attachments": 0},
            "removed": {"conversations": 0, "messages": 0, "attachments": 0},
            "changed": dict(changed_counts),
        }

        drift["new"]["conversations"] = counts["conversations"]
        drift["new"]["messages"] = counts["messages"]
        drift["new"]["attachments"] = counts["attachments"]

        run_id = uuid4().hex
        run_payload = {
            "run_id": run_id,
            "timestamp": int(time.time()),
            "counts": counts,
            "drift": drift,
            "indexed": indexed,
            "index_error": index_error,
            "duration_ms": duration_ms,
            "metrics": metrics.to_summary(),
        }
        _write_run_json(config.archive_root, run_payload)

        await active_repository.record_run(
            RunRecord(
                run_id=run_id,
                timestamp=str(run_payload["timestamp"]),
                plan_snapshot=plan.model_dump() if plan else None,
                counts=counts,
                drift=drift,
                indexed=indexed,
                duration_ms=duration_ms,
            ),
        )

        return RunResult(
            run_id=run_id,
            counts=counts,
            drift=drift,
            indexed=indexed,
            index_error=index_error,
            duration_ms=duration_ms,
            render_failures=render_failures,
        )
    finally:
        if owns_repository:
            await active_repository.close()
        elif owns_backend:
            await active_backend.close()


async def latest_run(backend: SQLiteBackend | None = None) -> RunRecord | None:
    """Fetch the most recent run record from the database asynchronously.

    Args:
        backend: Optional SQLiteBackend. If None, creates a temporary backend.

    Returns:
        RunRecord if a run exists, None otherwise
    """
    owns_backend = backend is None
    active_backend = backend or create_backend()
    try:
        return await active_backend.get_latest_run()
    finally:
        if owns_backend:
            await active_backend.close()


__all__ = [
    "RUN_STAGE_CHOICES",
    "plan_sources",
    "run_sources",
    "latest_run",
]

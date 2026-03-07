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
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from polylogue.config import Config, Source
from polylogue.lib.json import dumps, loads
from polylogue.lib.log import get_logger
from polylogue.lib.metrics import PipelineMetrics
from polylogue.storage.store import PlanResult, RunRecord, RunResult

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
) -> PlanResult:
    """Build a canonical preview plan without writing pipeline state."""
    from polylogue.services import get_backend
    from polylogue.pipeline.services.planning import PlanningService

    async def _build() -> PlanResult:
        backend = get_backend()
        planner = PlanningService(backend=backend, config=config)
        plan = await planner.build_plan(
            sources=_select_sources(config, source_names),
            stage=stage,
            ui=ui,
        )
        return plan.summary

    return _run_coroutine_sync(_build())


def _run_coroutine_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even when already inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - re-raised on caller thread
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result.get("value")


async def _all_conversation_ids(backend: Any, source_names: Sequence[str] | None = None) -> list[str]:
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
    backend: Any,
    index_service: Any,
    progress_callback: Any | None = None,
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
    progress_callback: Any | None = None,
    render_format: str = "html",
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
    from polylogue.services import get_backend, get_repository

    start = time.perf_counter()
    metrics = PipelineMetrics()

    backend = get_backend()
    repository = get_repository()

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

    # Acquire stage (raw storage only)
    if stage == "acquire":
        from polylogue.pipeline.services.acquisition import AcquisitionService

        sm = metrics.start_stage("acquire")
        acquire_service = AcquisitionService(backend=backend)
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
            repository=repository,
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
            # Use deduplicated conversation count for user-facing summary
            # (counts["conversations"] accumulates per-raw-record, double-counting
            # conversations that appear in multiple source files)
            counts["conversations"] = len(processed_ids)
            logger.info(
                "Parse stage complete",
                processed_ids=len(processed_ids),
                parse_failures=parse_result.parse_failures,
            )

    # Schema generation stage (sync, run in thread pool)
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

    # Rendering stage
    if stage in _RENDER_STAGES:
        from polylogue.pipeline.services.rendering import RenderService
        from polylogue.rendering.renderers import create_renderer

        sm = metrics.start_stage("render")
        ids = (
            await _all_conversation_ids(backend, source_names)
            if stage == "render"
            else list(processed_ids)
        )
        if ids:
            if progress_callback is not None:
                progress_callback(0, desc=f"Rendering: 0/{len(ids)}")
            renderer = create_renderer(
                format=render_format, config=config, backend=backend,
            )
            render_service = RenderService(
                renderer=renderer,
                render_root=config.archive_root / "render",
                backend=backend,
            )
            render_result = await render_service.render_conversations(
                ids, progress_callback=progress_callback,
            )
            counts["rendered"] = render_result.rendered_count
            render_failures = render_result.failures
            if render_failures:
                counts["render_failures"] = len(render_failures)
        sm.stop(items=counts.get("rendered", 0))
        logger.info(
            "Render stage complete", **sm.to_dict(),
            failures=len(render_failures),
            total=len(ids) if ids else 0,
        )

    # Indexing stage
    indexed = False
    index_error: str | None = None

    from polylogue.pipeline.services.indexing import IndexService

    index_service = IndexService(config=config, backend=backend)

    sm = metrics.start_stage("index")
    index_items = 0
    try:
        indexed, index_items = await _run_index_stage(
            stage=stage,
            source_names=source_names,
            processed_ids=processed_ids,
            backend=backend,
            index_service=index_service,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.error("Indexing failed", error=str(exc))
        index_error = str(exc)
        indexed = False
    sm.stop(items=index_items)
    logger.info("Index stage complete", **sm.to_dict(), indexed=indexed)

    # Calculate drift and finalize
    duration_ms = int((time.perf_counter() - start) * 1000)
    drift = {
        "new": {"conversations": 0, "messages": 0, "attachments": 0},
        "removed": {"conversations": 0, "messages": 0, "attachments": 0},
        "changed": dict(changed_counts),
    }

    processed_conversations = counts["conversations"] + counts["skipped_conversations"]
    processed_messages = counts["messages"] + counts["skipped_messages"]
    processed_attachments = counts["attachments"] + counts["skipped_attachments"]

    drift["new"]["conversations"] = counts["conversations"]
    drift["new"]["messages"] = counts["messages"]
    drift["new"]["attachments"] = counts["attachments"]

    # Record run
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

    await repository.record_run(
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


async def latest_run(backend: Any | None = None) -> RunRecord | None:
    """Fetch the most recent run record from the database asynchronously.

    Args:
        backend: Optional SQLiteBackend. If None, uses get_backend()

    Returns:
        RunRecord if a run exists, None otherwise
    """
    if backend is None:
        from polylogue.services import get_backend

        backend = get_backend()

    async with backend._get_connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1"
        )
        row = await cursor.fetchone()

    if not row:
        return None

    # Parse JSON columns
    plan_snapshot = None
    counts = None
    drift = None

    raw_plan = row["plan_snapshot"]
    if isinstance(raw_plan, str) and raw_plan:
        try:
            plan_snapshot = loads(raw_plan)
        except (ValueError, TypeError) as exc:
            logger.debug("Corrupt plan_snapshot JSON in run %s: %s", row["run_id"], exc)

    raw_counts = row["counts_json"]
    if isinstance(raw_counts, str) and raw_counts:
        try:
            counts = loads(raw_counts)
        except (ValueError, TypeError) as exc:
            logger.debug("Corrupt counts_json in run %s: %s", row["run_id"], exc)

    raw_drift = row["drift_json"]
    if isinstance(raw_drift, str) and raw_drift:
        try:
            drift = loads(raw_drift)
        except (ValueError, TypeError) as exc:
            logger.debug("Corrupt drift_json in run %s: %s", row["run_id"], exc)

    return RunRecord(
        run_id=row["run_id"],
        timestamp=row["timestamp"],
        plan_snapshot=plan_snapshot,
        counts=counts,
        drift=drift,
        indexed=bool(row["indexed"]) if row["indexed"] is not None else None,
        duration_ms=row["duration_ms"],
    )


__all__ = [
    "RUN_STAGE_CHOICES",
    "plan_sources",
    "run_sources",
    "latest_run",
]

"""Async pipeline runner logic.

Async/await version of the pipeline runner with support for:
- Async acquisition, parsing, rendering, and indexing stages
- Progress callbacks for long-running operations
- Full pipeline orchestration with stage control
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from polylogue.config import Config, Source
from polylogue.lib.json import dumps, loads
from polylogue.lib.log import get_logger
from polylogue.storage.store import PlanResult, RunRecord, RunResult

logger = get_logger(__name__)


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
    ui: object | None = None,
    source_names: Sequence[str] | None = None,
) -> PlanResult:
    """Plan the pipeline by counting conversations and messages from sources.

    This is a sync function because it only counts conversations from source
    files without parsing. Source iteration is synchronous file I/O.

    Args:
        config: Application configuration
        ui: Optional UI object for user interaction
        source_names: Optional list of source names to process

    Returns:
        PlanResult with conversation/message/attachment counts
    """
    from polylogue.sources import DriveAuthError, iter_drive_conversations, iter_source_conversations

    counts: dict[str, int] = {"conversations": 0, "messages": 0, "attachments": 0}
    sources: list[str] = []
    cursors: dict[str, dict[str, Any]] = {}

    for source in _select_sources(config, source_names):
        sources.append(source.name)
        cursor_state: dict[str, Any] = {}

        # Iterate source conversations (sync file I/O)
        if source.folder:
            try:
                conversations = iter_drive_conversations(
                    source=source,
                    archive_root=config.archive_root,
                    ui=ui,
                    download_assets=False,
                    cursor_state=cursor_state,
                    drive_config=config.drive_config,
                )
            except DriveAuthError as exc:
                logger.warning("Skipping Drive source %s: %s", source.name, exc)
                if cursor_state is not None:
                    cursor_state["error_count"] = cursor_state.get("error_count", 0) + 1
                    cursor_state["latest_error"] = str(exc)
                    cursor_state["latest_error_source"] = source.name
                continue
        else:
            conversations = iter_source_conversations(source, cursor_state=cursor_state)

        for convo in conversations:
            counts["conversations"] += 1
            counts["messages"] += len(convo.messages)
            counts["attachments"] += len(convo.attachments)

        if cursor_state:
            cursors[source.name] = cursor_state

    return PlanResult(timestamp=int(time.time()), counts=counts, sources=sources, cursors=cursors)


async def _all_conversation_ids(backend: Any, source_names: Sequence[str] | None = None) -> list[str]:
    """Fetch all conversation IDs from database, optionally filtered by source names.

    Args:
        backend: SQLiteBackend instance
        source_names: Optional list of source names to filter by

    Returns:
        List of conversation IDs
    """
    async with backend._get_connection() as conn:
        # Skip provider_meta if no filtering needed
        if not source_names:
            cursor = await conn.execute("SELECT conversation_id FROM conversations")
            rows = await cursor.fetchall()
            return [row["conversation_id"] for row in rows]

        cursor = await conn.execute(
            "SELECT conversation_id, provider_name, provider_meta FROM conversations"
        )
        rows = await cursor.fetchall()

    selected: list[str] = []
    name_set = set(source_names)

    for row in rows:
        if row["provider_name"] in name_set:
            selected.append(row["conversation_id"])
            continue

        meta = row["provider_meta"]
        if not meta:
            continue

        try:
            payload = loads(meta)
        except (ValueError, TypeError):
            logger.warning(
                "Skipping conversation with invalid provider_meta JSON",
                conversation_id=row["conversation_id"],
                provider=row["provider_name"],
            )
            continue

        if isinstance(payload, dict) and payload.get("source") in name_set:
            selected.append(row["conversation_id"])

    return selected


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
    - Full stage control ("acquire", "parse", "render", "index", "generate-schemas", "all")

    Args:
        config: Application configuration
        stage: Pipeline stage ("acquire", "parse", "render", "index", "generate-schemas", "all")
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

        acquire_service = AcquisitionService(backend=backend)
        sources = _select_sources(config, source_names)
        acquire_result = await acquire_service.acquire_sources(
            sources,
            progress_callback=progress_callback,
        )
        counts["acquired"] = acquire_result.counts["acquired"]
        counts["skipped"] = acquire_result.counts["skipped"]

    # Parse stage (acquire + parse, replaces old "ingest")
    elif stage in {"parse", "all"}:
        from polylogue.pipeline.services.parsing import ParsingService

        parsing_service = ParsingService(
            repository=repository,
            archive_root=config.archive_root,
            config=config,
        )
        sources = _select_sources(config, source_names)
        parse_result = await parsing_service.parse_sources(
            sources,
            ui=ui,
            download_assets=True,
            progress_callback=progress_callback,
        )

        # Merge results
        for key, value in parse_result.counts.items():
            counts[key] = value
        if parse_result.parse_failures:
            counts["parse_failures"] = parse_result.parse_failures
        changed_counts.update(parse_result.changed_counts)
        processed_ids = parse_result.processed_ids

    # Schema generation stage (sync, run in thread pool)
    if stage == "generate-schemas":
        from polylogue.paths import db_path as _db_path
        from polylogue.schemas.schema_inference import generate_all_schemas

        output_dir = config.archive_root.parent / "schemas"
        results = await asyncio.to_thread(
            generate_all_schemas,
            output_dir=output_dir,
            db_path=_db_path(),
        )
        counts["schemas_generated"] = sum(1 for r in results if r.success)
        counts["schemas_failed"] = sum(1 for r in results if not r.success)

    # Rendering stage
    if stage in {"render", "all"}:
        from polylogue.pipeline.services.rendering import RenderService
        from polylogue.rendering.renderers import create_renderer

        ids = (
            await _all_conversation_ids(backend, source_names)
            if stage == "render"
            else list(processed_ids)
        )
        if ids:
            if progress_callback is not None:
                progress_callback(0, desc=f"Rendering: 0/{len(ids)}")
            renderer = create_renderer(format=render_format, config=config)
            render_service = RenderService(
                renderer=renderer,
                render_root=config.archive_root / "render",
            )
            render_result = await render_service.render_conversations(
                ids, progress_callback=progress_callback,
            )
            counts["rendered"] = render_result.rendered_count
            render_failures = render_result.failures
            if render_failures:
                counts["render_failures"] = len(render_failures)

    # Indexing stage
    indexed = False
    index_error: str | None = None

    from polylogue.pipeline.services.indexing import IndexService

    index_service = IndexService(config=config, backend=backend)

    try:
        if stage == "index":
            if progress_callback is not None:
                progress_callback(0, desc="Indexing")
            if source_names:
                ids = await _all_conversation_ids(backend, source_names)
                if ids:
                    indexed = await index_service.update_index(ids)
                else:
                    indexed = await index_service.ensure_index_exists()
            else:
                indexed = await index_service.rebuild_index()
        elif stage == "all":
            idx = await index_service.get_index_status()
            if not idx["exists"]:
                if progress_callback is not None:
                    progress_callback(0, desc="Indexing (rebuild)")
                indexed = await index_service.rebuild_index()
            elif processed_ids:
                if progress_callback is not None:
                    progress_callback(0, desc=f"Indexing: {len(processed_ids)} conversations")
                indexed = await index_service.update_index(list(processed_ids))
    except Exception as exc:
        logger.error("Indexing failed", error=str(exc))
        index_error = str(exc)
        indexed = False

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

    if plan:
        expected_conversations = plan.counts.get("conversations", 0)
        expected_messages = plan.counts.get("messages", 0)
        expected_attachments = plan.counts.get("attachments", 0)
        drift["new"]["conversations"] = max(processed_conversations - expected_conversations, 0)
        drift["new"]["messages"] = max(processed_messages - expected_messages, 0)
        drift["new"]["attachments"] = max(processed_attachments - expected_attachments, 0)
        drift["removed"]["conversations"] = max(expected_conversations - processed_conversations, 0)
        drift["removed"]["messages"] = max(expected_messages - processed_messages, 0)
        drift["removed"]["attachments"] = max(expected_attachments - processed_attachments, 0)
    else:
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
    }
    _write_run_json(config.archive_root, run_payload)

    await repository.record_run(
        RunRecord(
            run_id=run_id,
            timestamp=str(run_payload["timestamp"]),
            plan_snapshot=plan.counts if plan else None,
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
    "plan_sources",
    "run_sources",
    "latest_run",
]

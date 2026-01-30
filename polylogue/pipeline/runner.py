"""Pipeline runner logic."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from dependency_injector import providers

from polylogue.config import Config, Source
from polylogue.core.json import dumps, loads
from polylogue.core.log import get_logger
from polylogue.ingestion import DriveAuthError, iter_drive_conversations, iter_source_conversations
from polylogue.ingestion.source import ParsedConversation
from polylogue.storage.store import PlanResult, RunResult
from polylogue.storage.backends.sqlite import connection_context
from polylogue.storage.store import RunRecord

logger = get_logger(__name__)


def _select_sources(config: Config, source_names: Sequence[str] | None) -> list[Source]:
    if not source_names:
        return list(config.sources)
    name_set = set(source_names)
    return [source for source in config.sources if source.name in name_set]


def _iter_source_conversations_safe(
    *,
    source: Source,
    archive_root: Path,
    ui: object | None,
    download_assets: bool,
    cursor_state: dict[str, Any] | None = None,
    drive_config: object | None = None,
) -> Iterable[ParsedConversation]:
    if source.folder:
        try:
            yield from iter_drive_conversations(
                source=source,
                archive_root=archive_root,
                ui=ui,
                download_assets=download_assets,
                cursor_state=cursor_state,
                drive_config=drive_config,
            )
        except DriveAuthError as exc:
            logger.warning("Skipping Drive source %s: %s", source.name, exc)
            if cursor_state is not None:
                cursor_state["error_count"] = cursor_state.get("error_count", 0) + 1
                cursor_state["latest_error"] = str(exc)
                cursor_state["latest_error_source"] = source.name
            return
    else:
        yield from iter_source_conversations(source, cursor_state=cursor_state)


def plan_sources(
    config: Config,
    *,
    ui: object | None = None,
    source_names: Sequence[str] | None = None,
) -> PlanResult:
    counts: dict[str, int] = {"conversations": 0, "messages": 0, "attachments": 0}
    sources: list[str] = []
    cursors: dict[str, dict[str, Any]] = {}
    for source in _select_sources(config, source_names):
        sources.append(source.name)
        cursor_state: dict[str, Any] = {}
        conversations = _iter_source_conversations_safe(
            source=source,
            archive_root=config.archive_root,
            ui=ui,
            download_assets=False,
            cursor_state=cursor_state,
            drive_config=config.drive_config,
        )
        for convo in conversations:
            counts["conversations"] += 1
            counts["messages"] += len(convo.messages)
            counts["attachments"] += len(convo.attachments)
        if cursor_state:
            cursors[source.name] = cursor_state
    return PlanResult(timestamp=int(time.time()), counts=counts, sources=sources, cursors=cursors)


def _all_conversation_ids(source_names: Sequence[str] | None = None) -> list[str]:
    with connection_context(None) as conn:
        # Skip provider_meta if no filtering needed
        if not source_names:
            rows = conn.execute("SELECT conversation_id FROM conversations").fetchall()
            return [row["conversation_id"] for row in rows]

        rows = conn.execute("SELECT conversation_id, provider_name, provider_meta FROM conversations").fetchall()

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
            # Invalid JSON in provider_meta - skip this conversation
            continue
        if isinstance(payload, dict) and payload.get("source") in name_set:
            selected.append(row["conversation_id"])
    return selected


def _write_run_json(archive_root: Path, payload: dict[str, object]) -> Path:
    runs_dir = archive_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload.get("run_id", "unknown")
    run_path = runs_dir / f"run-{payload['timestamp']}-{run_id}.json"
    run_path.write_text(dumps(payload, option=None), encoding="utf-8")
    return run_path


def run_sources(
    *,
    config: Config,
    stage: str = "all",
    plan: PlanResult | None = None,
    ui: object | None = None,
    source_names: Sequence[str] | None = None,
    progress_callback: Any | None = None,
    render_format: str = "html",
) -> RunResult:
    """Run the pipeline with stage control.

    Args:
        config: Application configuration
        stage: Pipeline stage ("ingest", "render", "index", or "all")
        plan: Optional plan result for drift detection
        ui: Optional UI object for user interaction
        source_names: Optional list of source names to process
        progress_callback: Optional callback for progress updates
        render_format: Output format for rendering ("markdown" or "html", default: "html")

    Returns:
        RunResult with counts and metadata
    """
    start = time.perf_counter()

    from polylogue.container import create_container

    container = create_container()
    # Override config singleton if a different config was passed
    container.config.override(config)

    repository = container.storage()
    ingestion_service = container.ingestion_service()

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

    with connection_context(None) as conn:
        # Ingestion stage
        if stage in {"ingest", "all"}:
            sources = _select_sources(config, source_names)
            ingest_result = ingestion_service.ingest_sources(
                sources,
                ui=ui,
                download_assets=True,
                progress_callback=progress_callback,
            )

            # Merge results
            for key, value in ingest_result.counts.items():
                counts[key] = value
            changed_counts.update(ingest_result.changed_counts)
            processed_ids = ingest_result.processed_ids

        # Rendering stage
        render_failures: list[dict[str, str]] = []
        if stage in {"render", "all"}:
            ids = _all_conversation_ids(source_names) if stage == "render" else list(processed_ids)
            # Use rendering service from container (which uses correct config)
            # We can optionally override the renderer format if needed
            if render_format == "markdown":
                from polylogue.rendering.renderers import create_renderer

                container.renderer.override(providers.Factory(create_renderer, format="markdown", config=config))

            render_service = container.rendering_service()
            render_result = render_service.render_conversations(ids)
            counts["rendered"] = render_result.rendered_count
            render_failures = render_result.failures
            if render_failures:
                counts["render_failures"] = len(render_failures)

        # Indexing stage
        indexed = False
        index_error: str | None = None
        index_service = container.indexing_service()
        # Ensure we use the active connection if provided
        index_service.conn = conn

        try:
            if stage == "index":
                if source_names:
                    ids = _all_conversation_ids(source_names)
                    if ids:
                        indexed = index_service.update_index(ids)
                    else:
                        indexed = index_service.ensure_index_exists()
                else:
                    indexed = index_service.rebuild_index()
            elif stage == "all":
                idx = index_service.get_index_status()
                if not idx["exists"]:
                    indexed = index_service.rebuild_index()
                elif processed_ids:
                    indexed = index_service.update_index(list(processed_ids))
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
        repository.record_run(
            RunRecord(
                run_id=run_id,
                timestamp=str(run_payload["timestamp"]),
                plan_snapshot=plan.counts if plan else None,  # type: ignore[arg-type]
                counts=counts,  # type: ignore[arg-type]
                drift=drift,  # type: ignore[arg-type]
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


def latest_run() -> RunRecord | None:
    """Fetch the most recent run record from the database.

    Returns:
        RunRecord if a run exists, None otherwise.
    """
    with connection_context(None) as conn:
        row = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1").fetchone()
    if not row:
        return None

    # Parse JSON columns
    plan_snapshot = None
    counts = None
    drift = None

    raw_plan = row["plan_snapshot"]
    if isinstance(raw_plan, str) and raw_plan:
        with contextlib.suppress(ValueError, TypeError):
            plan_snapshot = loads(raw_plan)

    raw_counts = row["counts_json"]
    if isinstance(raw_counts, str) and raw_counts:
        with contextlib.suppress(ValueError, TypeError):
            counts = loads(raw_counts)

    raw_drift = row["drift_json"]
    if isinstance(raw_drift, str) and raw_drift:
        with contextlib.suppress(ValueError, TypeError):
            drift = loads(raw_drift)

    return RunRecord(
        run_id=row["run_id"],
        timestamp=row["timestamp"],
        plan_snapshot=plan_snapshot,
        counts=counts,
        drift=drift,
        indexed=bool(row["indexed"]) if row["indexed"] is not None else None,
        duration_ms=row["duration_ms"],
    )

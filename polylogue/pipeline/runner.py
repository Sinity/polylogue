"""Pipeline runner logic."""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from polylogue.config import Config, Source
from polylogue.core.json import dumps, loads
from polylogue.core.log import get_logger
from polylogue.db import connection_context
from polylogue.drive_ingest import iter_drive_conversations
from polylogue.index import index_status, rebuild_index, update_index_for_conversations
from polylogue.pipeline.ingest import prepare_ingest
from polylogue.pipeline.models import PlanResult, RunResult
from polylogue.render import render_conversation
from polylogue.source_ingest import iter_source_conversations
from polylogue.store import RunRecord, record_run

logger = get_logger(__name__)


def _select_sources(config: Config, source_names: Sequence[str] | None) -> list[Source]:
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
    counts = {"conversations": 0, "messages": 0, "attachments": 0}
    sources = []
    cursors: dict[str, dict] = {}
    for source in _select_sources(config, source_names):
        sources.append(source.name)
        cursor_state: dict[str, object] = {}
        if source.folder:
            conversations = iter_drive_conversations(
                source=source,
                archive_root=config.archive_root,
                ui=ui,
                download_assets=False,
                cursor_state=cursor_state,
            )
        else:
            conversations = iter_source_conversations(source, cursor_state=cursor_state)
        for convo in conversations:
            counts["conversations"] += 1
            counts["messages"] += len(convo.messages)
            counts["attachments"] += len(convo.attachments)
        if cursor_state:
            cursors[source.name] = cursor_state
    return PlanResult(timestamp=int(time.time()), counts=counts, sources=sources, cursors=cursors)


def _all_conversation_ids(source_names: Sequence[str] | None = None) -> list[str]:
    with connection_context(None) as conn:
        rows = conn.execute("SELECT conversation_id, provider_name, provider_meta FROM conversations").fetchall()
    if not source_names:
        return [row["conversation_id"] for row in rows]
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
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("source") in name_set:
            selected.append(row["conversation_id"])
    return selected


def _write_run_json(archive_root: Path, payload: dict) -> Path:
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
) -> RunResult:
    start = time.perf_counter()
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            def _process_one(convo_item, source_name_item):
                # Run preparation in a separate thread with its own connection for reads
                with connection_context(None) as thread_conn:
                    return prepare_ingest(
                        convo_item,
                        source_name_item,
                        archive_root=config.archive_root,
                        conn=thread_conn,
                    )

            if stage in {"ingest", "all"}:
                for source in _select_sources(config, source_names):
                    if source.folder:
                        conversations = iter_drive_conversations(
                            source=source,
                            archive_root=config.archive_root,
                            ui=ui,
                            download_assets=True,
                        )
                    else:
                        conversations = iter_source_conversations(source)

                    for convo in conversations:
                        # Bounded submission to prevent memory explosion
                        while len(futures) > 16:
                            done, _ = concurrent.futures.wait(
                                futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for fut in done:
                                try:
                                    convo_id, result_counts, content_changed = fut.result()
                                    ingest_changed = (
                                        result_counts["conversations"]
                                        + result_counts["messages"]
                                        + result_counts["attachments"]
                                    ) > 0
                                    if ingest_changed or content_changed:
                                        processed_ids.add(convo_id)
                                    if content_changed:
                                        changed_counts["conversations"] += 1
                                    for key, value in result_counts.items():
                                        counts[key] += value
                                finally:
                                    del futures[fut]

                        future = executor.submit(_process_one, convo, source.name)
                        futures[future] = convo.provider_conversation_id
                        if progress_callback:
                            progress_callback(1, desc="Ingesting")

                # Drain remaining
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        convo_id, result_counts, content_changed = fut.result()
                        ingest_changed = (
                            result_counts["conversations"] + result_counts["messages"] + result_counts["attachments"]
                        ) > 0
                        if ingest_changed or content_changed:
                            processed_ids.add(convo_id)
                        if content_changed:
                            changed_counts["conversations"] += 1
                        for key, value in result_counts.items():
                            counts[key] += value
                        if progress_callback:
                            progress_callback(1, desc="Ingesting")
                    except Exception as exc:
                        logger.error("Error processing conversation", error=str(exc))
                        raise

        if stage in {"render", "all"}:
            ids = _all_conversation_ids(source_names) if stage == "render" else list(processed_ids)
            for convo_id in ids:
                render_conversation(
                    conversation_id=convo_id,
                    archive_root=config.archive_root,
                    render_root_path=config.render_root,
                    template_path=config.template_path,
                )
                counts["rendered"] += 1

        indexed = False
        index_error: str | None = None
        try:
            if stage == "index":
                rebuild_index(conn)
                indexed = True
            elif stage == "all":
                idx = index_status()
                if not idx["exists"]:
                    rebuild_index(conn)
                    indexed = True
                elif processed_ids:
                    update_index_for_conversations(list(processed_ids), conn)
                    indexed = True
        except Exception as exc:
            index_error = str(exc)
            indexed = False

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
        record_run(
            conn,
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
        # Context manager handles commit if needed, but explicit is fine
        # connection_context doesn't have commit() on it if explicitly yielded conn,
        # but conn.commit() is valid.

    return RunResult(
        run_id=run_id,
        counts=counts,
        drift=drift,
        indexed=indexed,
        index_error=index_error,
        duration_ms=duration_ms,
    )


def latest_run() -> dict | None:
    with connection_context(None) as conn:
        row = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1").fetchone()
    return dict(row) if row else None

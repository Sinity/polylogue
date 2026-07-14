"""``maintenance rebuild-index``: authority-safe source-to-index rebuild."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import click

from polylogue.config import Config
from polylogue.logging import configure_logging
from polylogue.paths import archive_root, render_root


def _count_source_raw_sessions(root: Path) -> int:
    source_db = root / "source.db"
    if not source_db.exists():
        return 0
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0) as conn:
        row = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()
    return int(row[0]) if row is not None else 0


def _missing_index_raw_ids(root: Path) -> list[str]:
    source_db = root / "source.db"
    index_db = root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return []
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0) as conn:
        conn.execute("ATTACH DATABASE ? AS idx", (str(index_db),))
        rows = conn.execute(
            """
            SELECT r.raw_id
            FROM raw_sessions r
            WHERE NOT EXISTS (
                SELECT 1
                FROM idx.sessions s
                WHERE s.raw_id = r.raw_id
            )
            ORDER BY r.acquired_at_ms, r.raw_id
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def _all_index_rebuild_raw_ids(root: Path) -> list[str]:
    source_db = root / "source.db"
    if not source_db.exists():
        return []
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0) as conn:
        rows = conn.execute(
            """
            SELECT raw_id
            FROM raw_sessions
            ORDER BY acquired_at_ms, raw_id
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def _filter_raw_ids_by_max_blob_size(root: Path, raw_ids: list[str], max_blob_mb: float | None) -> list[str]:
    if max_blob_mb is None or not raw_ids:
        return raw_ids
    max_bytes = int(max_blob_mb * 1024 * 1024)
    source_db = root / "source.db"
    placeholders = ",".join("?" for _ in raw_ids)
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0) as conn:
        rows = conn.execute(
            f"""
            SELECT raw_id
            FROM raw_sessions
            WHERE raw_id IN ({placeholders})
              AND blob_size <= ?
            ORDER BY acquired_at_ms, raw_id
            """,
            (*raw_ids, max_bytes),
        ).fetchall()
    return [str(row[0]) for row in rows]


def _rebuild_index_selection_plan(
    root: Path,
    *,
    selected_raw_ids: list[str] | None,
    raw_session_count: int,
    selected_raw_count: int,
    skipped_by_blob_limit_count: int,
    only_missing: bool,
    explicit_raw_id_count: int,
    max_blob_mb: float | None,
    limit: int,
) -> dict[str, object]:
    source_db = root / "source.db"
    index_db = root / "index.db"
    if not source_db.exists():
        return {
            "archive_root": str(root),
            "status": "empty-source",
            "raw_session_count": raw_session_count,
            "selected_raw_count": 0,
            "skipped_by_blob_limit_count": skipped_by_blob_limit_count,
            "only_missing": only_missing,
            "raw_id_count": explicit_raw_id_count,
            "max_blob_mb": max_blob_mb,
            "totals": {},
            "top_rows": [],
        }

    selected_clause = ""
    params: list[object] = []
    if selected_raw_ids is not None:
        if not selected_raw_ids:
            return {
                "archive_root": str(root),
                "status": "ok",
                "raw_session_count": raw_session_count,
                "selected_raw_count": 0,
                "skipped_by_blob_limit_count": skipped_by_blob_limit_count,
                "only_missing": only_missing,
                "raw_id_count": explicit_raw_id_count,
                "max_blob_mb": max_blob_mb,
                "replay_order": "acquired_at_ms_asc_raw_id_asc",
                "risk_order": "blob_size_desc",
                "cost_basis": {
                    "primary": "source.db raw_sessions.blob_size",
                    "secondary": [
                        "index.db sessions.message_count when already materialized",
                        "index.db session_events count when already materialized",
                    ],
                },
                "totals": {
                    "blob_bytes": 0,
                    "materialized_sessions": 0,
                    "materialized_messages": 0,
                    "materialized_session_events": 0,
                },
                "top_rows": [],
                "top_groups": [],
            }
        placeholders = ",".join("?" for _ in selected_raw_ids)
        selected_clause = f"WHERE r.raw_id IN ({placeholders})"
        params.extend(selected_raw_ids)

    index_metrics = (
        """
        COALESCE((SELECT COUNT(*) FROM idx.sessions s WHERE s.raw_id = r.raw_id), 0) AS materialized_sessions,
        COALESCE((SELECT SUM(s.message_count) FROM idx.sessions s WHERE s.raw_id = r.raw_id), 0) AS materialized_messages,
        COALESCE((
            SELECT COUNT(*)
            FROM idx.sessions s
            JOIN idx.session_events e ON e.session_id = s.session_id
            WHERE s.raw_id = r.raw_id
        ), 0) AS materialized_session_events
        """
        if index_db.exists()
        else """
        0 AS materialized_sessions,
        0 AS materialized_messages,
        0 AS materialized_session_events
        """
    )
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0) as conn:
        if index_db.exists():
            conn.execute("ATTACH DATABASE ? AS idx", (str(index_db),))
        rows = conn.execute(
            f"""
            SELECT
                r.raw_id,
                r.origin,
                r.native_id,
                r.source_path,
                r.source_index,
                r.blob_size,
                r.acquired_at_ms,
                {index_metrics}
            FROM raw_sessions r
            {selected_clause}
            ORDER BY r.acquired_at_ms, r.raw_id
            """,
            params,
        ).fetchall()

    top_rows = sorted(
        rows,
        key=lambda row: (-(int(row[5] or 0)), -(int(row[6] or 0)), str(row[0])),
    )
    top_groups_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        origin = str(row[1] or "")
        source_or_native = str(row[3] or row[2] or row[0])
        key = (origin, source_or_native)
        group = top_groups_by_key.setdefault(
            key,
            {
                "origin": origin,
                "native_id": row[2],
                "source_path": row[3],
                "row_count": 0,
                "blob_bytes": 0,
                "first_acquired_at_ms": row[6],
                "last_acquired_at_ms": row[6],
                "materialized_sessions": 0,
                "materialized_messages": 0,
                "materialized_session_events": 0,
            },
        )
        group["row_count"] = int(group["row_count"]) + 1
        group["blob_bytes"] = int(group["blob_bytes"]) + int(row[5] or 0)
        acquired_at_ms = row[6]
        if group["first_acquired_at_ms"] is None or (
            acquired_at_ms is not None and int(acquired_at_ms) < int(group["first_acquired_at_ms"])
        ):
            group["first_acquired_at_ms"] = acquired_at_ms
        if group["last_acquired_at_ms"] is None or (
            acquired_at_ms is not None and int(acquired_at_ms) > int(group["last_acquired_at_ms"])
        ):
            group["last_acquired_at_ms"] = acquired_at_ms
        group["materialized_sessions"] = int(group["materialized_sessions"]) + int(row[7] or 0)
        group["materialized_messages"] = int(group["materialized_messages"]) + int(row[8] or 0)
        group["materialized_session_events"] = int(group["materialized_session_events"]) + int(row[9] or 0)
    top_groups = sorted(
        top_groups_by_key.values(),
        key=lambda group: (-int(group["blob_bytes"]), -int(group["row_count"]), str(group["origin"])),
    )
    totals = {
        "blob_bytes": sum(int(row[5] or 0) for row in rows),
        "materialized_sessions": sum(int(row[7] or 0) for row in rows),
        "materialized_messages": sum(int(row[8] or 0) for row in rows),
        "materialized_session_events": sum(int(row[9] or 0) for row in rows),
    }
    return {
        "archive_root": str(root),
        "status": "ok",
        "raw_session_count": raw_session_count,
        "selected_raw_count": selected_raw_count,
        "skipped_by_blob_limit_count": skipped_by_blob_limit_count,
        "only_missing": only_missing,
        "raw_id_count": explicit_raw_id_count,
        "max_blob_mb": max_blob_mb,
        "replay_order": "acquired_at_ms_asc_raw_id_asc",
        "risk_order": "blob_size_desc",
        "cost_basis": {
            "primary": "source.db raw_sessions.blob_size",
            "secondary": [
                "index.db sessions.message_count when already materialized",
                "index.db session_events count when already materialized",
            ],
        },
        "totals": totals,
        "top_rows": [
            {
                "raw_id": str(row[0]),
                "origin": row[1],
                "native_id": row[2],
                "source_path": row[3],
                "source_index": row[4],
                "blob_bytes": int(row[5] or 0),
                "acquired_at_ms": row[6],
                "materialized_sessions": int(row[7] or 0),
                "materialized_messages": int(row[8] or 0),
                "materialized_session_events": int(row[9] or 0),
            }
            for row in top_rows[:limit]
        ],
        "top_groups": top_groups[:limit],
    }


@click.command("rebuild-index")
@click.option(
    "--only-missing",
    is_flag=True,
    help="Replay only source raw rows that do not yet have index.sessions rows.",
)
@click.option(
    "--raw-id",
    "raw_ids",
    multiple=True,
    help="Replay a specific source raw_id. May be supplied multiple times.",
)
@click.option(
    "--max-blob-mb",
    type=float,
    default=None,
    help="Bound an explicit replay selection to raw rows at or below this blob size.",
)
@click.option(
    "--plan",
    "plan_only",
    is_flag=True,
    help="Print selected raw-row weight totals and top rows without replaying.",
)
@click.option("--plan-limit", type=int, default=10, show_default=True, help="Rows to include in --plan top_rows.")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.option("--no-promote", is_flag=True, help="Leave an exact-ready generation inactive after rebuilding it.")
def rebuild_index_command(
    only_missing: bool,
    raw_ids: tuple[str, ...],
    max_blob_mb: float | None,
    plan_only: bool,
    plan_limit: int,
    output_format: str,
    no_promote: bool,
) -> None:
    """Inspect or execute an authority-safe source-to-index rebuild.

    Execution expands the requested rows to complete logical revision cohorts;
    selection order and batch boundaries never participate in authority.
    """
    from polylogue.maintenance.replay import rebuild_index_from_source

    configure_logging()
    if raw_ids and only_missing:
        raise click.UsageError("--raw-id cannot be combined with --only-missing")
    if (raw_ids or only_missing) and not no_promote and not plan_only:
        raise click.UsageError("partial rebuild selections require --no-promote and can never replace the active index")
    if max_blob_mb is not None and max_blob_mb <= 0:
        raise click.BadParameter("max blob size must be positive", param_hint="--max-blob-mb")
    if max_blob_mb is not None and not raw_ids and not only_missing:
        raise click.UsageError("--max-blob-mb requires --only-missing or --raw-id")
    if plan_limit <= 0:
        raise click.BadParameter("plan limit must be positive", param_hint="--plan-limit")

    root = archive_root()
    raw_count = _count_source_raw_sessions(root)
    if raw_count == 0:
        payload = {
            "archive_root": str(root),
            "raw_session_count": 0,
            "selected_raw_count": 0,
            "skipped_by_blob_limit_count": 0,
            "status": "empty-source",
            "materialized": False,
        }
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            click.echo(f"Archive root: {root}")
            click.echo("No source.db raw_sessions rows found.")
        return
    selected_raw_ids = (
        list(dict.fromkeys(raw_ids))
        if raw_ids
        else _missing_index_raw_ids(root)
        if only_missing
        else _all_index_rebuild_raw_ids(root)
    )
    unfiltered_selected_raw_count = len(selected_raw_ids)
    selected_raw_ids = _filter_raw_ids_by_max_blob_size(root, selected_raw_ids, max_blob_mb)
    selected_raw_count = len(selected_raw_ids)
    skipped_by_blob_limit_count = unfiltered_selected_raw_count - selected_raw_count
    if plan_only:
        payload = _rebuild_index_selection_plan(
            root,
            selected_raw_ids=selected_raw_ids,
            raw_session_count=raw_count,
            selected_raw_count=selected_raw_count,
            skipped_by_blob_limit_count=skipped_by_blob_limit_count,
            only_missing=only_missing,
            explicit_raw_id_count=len(raw_ids),
            max_blob_mb=max_blob_mb,
            limit=plan_limit,
        )
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        click.echo(f"Archive root: {root}")
        click.echo(f"Raw rows:     {raw_count:,}")
        click.echo(f"Selected:     {selected_raw_count:,} raw row(s)")
        if skipped_by_blob_limit_count:
            click.echo(f"Blob limit:   skipped {skipped_by_blob_limit_count:,} raw row(s)")
        totals = payload["totals"] if isinstance(payload["totals"], dict) else {}
        click.echo(f"Blob bytes:   {int(totals.get('blob_bytes', 0)):,}")
        click.echo(f"Messages:     {int(totals.get('materialized_messages', 0)):,} already materialized")
        click.echo(f"Events:       {int(totals.get('materialized_session_events', 0)):,} already materialized")
        top_rows = payload["top_rows"] if isinstance(payload["top_rows"], list) else []
        if top_rows:
            click.echo("Top rows by blob size:")
            for row in top_rows:
                if isinstance(row, dict):
                    click.echo(
                        f"  {row['raw_id']} {row['origin']} blob={int(row['blob_bytes']):,} "
                        f"messages={int(row['materialized_messages']):,} events={int(row['materialized_session_events']):,}"
                    )
        raw_top_groups: object = payload.get("top_groups")
        top_groups = raw_top_groups if isinstance(raw_top_groups, list) else []
        if top_groups:
            click.echo("Top groups by blob size:")
            for group in top_groups:
                if isinstance(group, dict):
                    click.echo(
                        f"  {group['origin']} rows={int(group['row_count']):,} "
                        f"blob={int(group['blob_bytes']):,} source={group['source_path']}"
                    )
        return
    from polylogue.cli.commands.status import _archive_readiness_status
    from polylogue.maintenance.offline_guard import running_daemon_pid
    from polylogue.storage.index_generation import (
        IndexGenerationStore,
        RebuildLease,
        source_revision_snapshot,
    )

    generation_store = IndexGenerationStore(root)
    with RebuildLease(root):
        active_config = Config(archive_root=root, render_root=render_root(), sources=[], db_path=root / "index.db")
        daemon_pid = running_daemon_pid(active_config)
        if daemon_pid is not None:
            raise click.ClickException(f"offline rebuild refused while polylogued PID {daemon_pid} is running")
        raw_count = _count_source_raw_sessions(root)
        selected_raw_ids = (
            list(dict.fromkeys(raw_ids))
            if raw_ids
            else _missing_index_raw_ids(root)
            if only_missing
            else _all_index_rebuild_raw_ids(root)
        )
        unfiltered_selected_raw_count = len(selected_raw_ids)
        selected_raw_ids = _filter_raw_ids_by_max_blob_size(root, selected_raw_ids, max_blob_mb)
        selected_raw_count = len(selected_raw_ids)
        skipped_by_blob_limit_count = unfiltered_selected_raw_count - selected_raw_count
        source_snapshot = source_revision_snapshot(root)
        generation = generation_store.create(source_snapshot=source_snapshot)
        try:
            generation_root = Path(generation.index_path).parent
            config = Config(
                archive_root=generation_root,
                render_root=render_root(),
                sources=[],
                db_path=Path(generation.index_path),
            )
            result = asyncio.run(
                rebuild_index_from_source(
                    config,
                    raw_ids=selected_raw_ids,
                    raw_batch_size=500,
                    ingest_workers=None,
                    materialize=True,
                    progress_callback=None,
                    owned_inactive_generation=(generation.generation_id, generation.owner_id),
                )
            )
            from polylogue.storage.repair import repair_session_insights

            insight_result = repair_session_insights(
                config,
                dry_run=False,
                archive_root_override=generation_root,
                owned_inactive_generation=(generation.generation_id, generation.owner_id),
            )
            if not insight_result.success:
                raise click.ClickException(f"session insight materialization failed: {insight_result.detail}")
            if source_revision_snapshot(root) != generation.source_snapshot:
                raise click.ClickException(f"source evidence changed while rebuilding {generation.generation_id}")
            readiness = _archive_readiness_status(generation_root)
            if not readiness.get("checked") or int(readiness.get("blocked_surface_count", 1)) != 0:
                blocked = [
                    name
                    for name, info in cast(dict[str, dict[str, object]], readiness.get("surfaces", {})).items()
                    if info.get("ready") is not True
                ]
                detail = (
                    f"reason: {readiness.get('reason')}"
                    if not readiness.get("checked")
                    else "blocked surfaces: " + ", ".join(blocked)
                )
                raise click.ClickException(
                    f"inactive generation {generation.generation_id} is not exact-ready; {detail}"
                )
            if not no_promote:
                generation = generation_store.promote(generation)
        except Exception:
            with contextlib.suppress(Exception):
                generation_store.discard_if_inactive(generation)
            raise
    payload = {
        "archive_root": str(root),
        "raw_session_count": raw_count,
        "selected_raw_count": selected_raw_count,
        "skipped_by_blob_limit_count": skipped_by_blob_limit_count,
        "status": "replayed",
        "materialized": True,
        "materialization": insight_result.to_dict(),
        "generation": asdict(generation),
        "readiness": readiness,
        **result,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(f"Archive root: {root}")
    click.echo(f"Classified:   {int(cast(Any, result['classified_full_count'])):,} full revision(s)")
    click.echo(f"Replayed:     {int(cast(Any, result['replayed_logical_source_count'])):,} logical source(s)")
    click.echo(f"Quarantined:  {int(cast(Any, result['quarantined_raw_count'])):,} raw row(s)")

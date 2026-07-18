"""``maintenance rebuild-index``: authority-safe source-to-index rebuild."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click

from polylogue.logging import configure_logging
from polylogue.paths import archive_root
from polylogue.storage.archive_identity import ArchiveLocation


def _run_daemon_rebuild(
    daemon_url: str,
    *,
    only_missing: bool,
    raw_ids: tuple[str, ...],
    max_blob_mb: float | None,
    no_promote: bool,
    operation_id: str | None,
    raw_batch_size: int,
    pass_byte_budget_mb: float | None,
    pass_deadline_seconds: float | None,
) -> dict[str, object]:
    """Execute one rebuild pass through the daemon-owned writer."""
    from polylogue.config import load_polylogue_config

    body = json.dumps(
        {
            "only_missing": only_missing,
            "raw_ids": list(raw_ids),
            "max_blob_mb": max_blob_mb,
            "promote": not no_promote,
            "operation_id": operation_id,
            "raw_batch_size": raw_batch_size,
            "pass_byte_budget_mb": pass_byte_budget_mb,
            "pass_deadline_seconds": pass_deadline_seconds,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if auth_token := load_polylogue_config().api_auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    request = Request(
        f"{daemon_url.rstrip('/')}/api/maintenance/rebuild-index",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise click.ClickException(f"daemon rebuild rejected by {daemon_url}: HTTP {exc.code}: {detail}") from exc
    except (URLError, OSError, ValueError) as exc:
        raise click.ClickException(f"could not reach daemon at {daemon_url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"daemon at {daemon_url} returned an invalid rebuild receipt")
    return cast(dict[str, object], payload)


def _count_source_raw_sessions(root: Path) -> int:
    from polylogue.maintenance.rebuild_index import count_source_raw_sessions

    return count_source_raw_sessions(root)


def _missing_index_raw_ids(root: Path) -> list[str]:
    from polylogue.maintenance.rebuild_index import missing_index_raw_ids

    return missing_index_raw_ids(root)


def _all_index_rebuild_raw_ids(root: Path) -> list[str]:
    from polylogue.maintenance.rebuild_index import all_index_rebuild_raw_ids

    return all_index_rebuild_raw_ids(root)


def _filter_raw_ids_by_max_blob_size(root: Path, raw_ids: list[str], max_blob_mb: float | None) -> list[str]:
    from polylogue.maintenance.rebuild_index import filter_raw_ids_by_max_blob_size

    return filter_raw_ids_by_max_blob_size(root, raw_ids, max_blob_mb)


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
    index_db = ArchiveLocation.resolve(root).active_index_path
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
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
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
    "--operation-id",
    type=str,
    default=None,
    help="Resume the retained candidate generation for this rebuild operation.",
)
@click.option(
    "--raw-batch-size",
    type=int,
    default=500,
    show_default=True,
    help="Maximum source rows scheduled by this invocation; rerun with --operation-id to continue.",
)
@click.option(
    "--pass-byte-budget-mb",
    type=float,
    default=None,
    help="Aggregate raw bytes scheduled per resumable pass; never excludes later source rows.",
)
@click.option(
    "--pass-deadline-seconds",
    type=float,
    default=None,
    help="Wall-clock deadline for one resumable pass; expiry defers remaining source rows.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.option("--no-promote", is_flag=True, help="Leave an exact-ready generation inactive after rebuilding it.")
@click.option(
    "--daemon", "use_daemon", is_flag=True, help="Run the bounded rebuild through the live polylogued daemon."
)
@click.option(
    "--daemon-url",
    default=lambda: __import__("os").environ.get("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:8766"),
    show_default="POLYLOGUE_DAEMON_URL or http://127.0.0.1:8766",
    help="Daemon HTTP base URL used with --daemon.",
)
def rebuild_index_command(
    only_missing: bool,
    raw_ids: tuple[str, ...],
    max_blob_mb: float | None,
    plan_only: bool,
    plan_limit: int,
    operation_id: str | None,
    raw_batch_size: int,
    pass_byte_budget_mb: float | None,
    pass_deadline_seconds: float | None,
    output_format: str,
    no_promote: bool,
    use_daemon: bool,
    daemon_url: str,
) -> None:
    """Inspect or execute an authority-safe source-to-index rebuild.

    Execution expands the requested rows to complete logical revision cohorts;
    selection order and batch boundaries never participate in authority.
    """
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
    if use_daemon and plan_only:
        raise click.UsageError("--daemon executes a rebuild; --plan is always a local read-only preview")
    if raw_batch_size <= 0:
        raise click.BadParameter("raw batch size must be positive", param_hint="--raw-batch-size")
    if pass_byte_budget_mb is not None and pass_byte_budget_mb <= 0:
        raise click.BadParameter("pass byte budget must be positive", param_hint="--pass-byte-budget-mb")
    if pass_deadline_seconds is not None and pass_deadline_seconds <= 0:
        raise click.BadParameter("pass deadline must be positive", param_hint="--pass-deadline-seconds")
    if operation_id is not None and (raw_ids or only_missing or max_blob_mb is not None or plan_only):
        raise click.UsageError("--operation-id only resumes an unfiltered full-source rebuild")
    if operation_id is not None and (pass_byte_budget_mb is not None or pass_deadline_seconds is not None):
        raise click.UsageError("resumed rebuild budgets are durable; omit pass budget options with --operation-id")

    root = archive_root()
    if use_daemon:
        payload = _run_daemon_rebuild(
            daemon_url,
            only_missing=only_missing,
            raw_ids=raw_ids,
            max_blob_mb=max_blob_mb,
            no_promote=no_promote,
            operation_id=operation_id,
            raw_batch_size=raw_batch_size,
            pass_byte_budget_mb=pass_byte_budget_mb,
            pass_deadline_seconds=pass_deadline_seconds,
        )
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        click.echo(f"Archive root: {payload.get('archive_root', root)}")
        click.echo(f"Classified:   {int(cast(Any, payload['classified_full_count'])):,} full revision(s)")
        click.echo(f"Replayed:     {int(cast(Any, payload['replayed_logical_source_count'])):,} logical source(s)")
        click.echo(f"Quarantined:  {int(cast(Any, payload['quarantined_raw_count'])):,} raw row(s)")
        return
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
    if plan_only:
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
    from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync

    try:
        receipt = rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                only_missing=only_missing,
                raw_ids=raw_ids,
                max_blob_mb=max_blob_mb,
                promote=not no_promote,
                operation_id=operation_id,
                raw_batch_size=raw_batch_size,
                pass_byte_budget_mb=pass_byte_budget_mb,
                pass_deadline_seconds=pass_deadline_seconds,
            )
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = receipt.to_dict()
    result = payload
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(f"Archive root: {root}")
    click.echo(f"Classified:   {int(cast(Any, result['classified_full_count'])):,} full revision(s)")
    click.echo(f"Replayed:     {int(cast(Any, result['replayed_logical_source_count'])):,} logical source(s)")
    click.echo(f"Quarantined:  {int(cast(Any, result['quarantined_raw_count'])):,} raw row(s)")

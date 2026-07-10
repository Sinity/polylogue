"""Maintenance command group: preview and run backfills."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from polylogue.archive.raw_materialization import source_path_native_id_candidates
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from polylogue.logging import configure_logging
from polylogue.maintenance.envelope import envelope_from_operation
from polylogue.maintenance.planner import preview_backfill
from polylogue.maintenance.preview import ALL_SCOPES, staleness_inventory
from polylogue.maintenance.registry import MaintenanceOperationRegistry, OperationRecord
from polylogue.maintenance.replay import ReplayProgress, execute_replay, rebuild_index_from_source
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.maintenance.targets import MAINTENANCE_TARGET_NAMES, build_maintenance_target_catalog
from polylogue.paths import archive_file_set_root_for_paths, archive_root, blob_store_root, db_path, render_root
from polylogue.storage.blob_gc import read_gc_history, run_blob_gc_report
from polylogue.storage.blob_integrity import (
    BlobReferenceDebtClassificationReport,
    BlobReferenceOrphanPruneReport,
    BlobReferenceRecoveryPlanReport,
    BlobReferenceSourceReplaceReport,
    classify_blob_reference_debt,
    plan_raw_backed_blob_reference_recovery,
    prune_orphan_blob_reference_debt,
    replace_raw_backed_blob_reference_debt_from_source,
    scan_attachment_acquisition_debt,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.archive_init import (
    ArchiveInitBlockedError,
    ArchiveInitResult,
    initialize_archive_tier_files_from_plan,
)
from polylogue.storage.sqlite.archive_tiers.archive_plan import ArchiveInitPlan, build_archive_init_plan
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.ops_write import record_ingest_attempt
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    AssertionKind,
    assertion_envelope_to_payload,
    list_assertions_for_export,
)
from polylogue.storage.sqlite.migration_runner import (
    DURABLE_MIGRATION_TIERS,
    MigrationError,
    migrate_archive_tier,
)

_BACKUP_POLICIES: dict[ArchiveTier, dict[str, object]] = {
    ArchiveTier.SOURCE: {
        "backup_class": "critical",
        "policy": "back_up",
        "restore_role": "raw acquisition evidence and rebuild root",
    },
    ArchiveTier.INDEX: {
        "backup_class": "warm_cache",
        "policy": "optional_full_evidence",
        "restore_role": "parsed sessions, search indexes, graph rows, and derived read models",
    },
    ArchiveTier.EMBEDDINGS: {
        "backup_class": "expensive_rebuild",
        "policy": "back_up_when_present",
        "restore_role": "vector rows and embedding catch-up state",
    },
    ArchiveTier.USER: {
        "backup_class": "critical",
        "policy": "always_back_up",
        "restore_role": "human and agent overlays",
    },
    ArchiveTier.OPS: {
        "backup_class": "diagnostic",
        "policy": "diagnostics_only",
        "restore_role": "daemon cursors, convergence debt, and runtime telemetry",
    },
}

_BACKUP_PROFILES: tuple[dict[str, object], ...] = (
    {
        "name": "full_evidence",
        "include": ["source.db", "index.db", "embeddings.db", "user.db", "blob/", "ops.db optional"],
        "exclude": ["*-wal after checkpoint", "*-shm after checkpoint"],
        "use_case": "fastest complete restore with raw evidence, read models, vectors, and overlays",
    },
    {
        "name": "user_overlays",
        "include": ["user.db", "user-owned referenced blobs"],
        "exclude": ["index.db", "ops.db", "rebuildable derived models"],
        "use_case": "protect irreplaceable human and agent state before resets or schema rebuilds",
    },
    {
        "name": "rebuildable_cache_exclude",
        "include": ["source.db", "user.db", "blob/", "embeddings.db optional"],
        "exclude": ["index.db", "ops.db", "derived/cache artifacts"],
        "use_case": "smaller backup that can rebuild parsed and indexed data locally",
    },
    {
        "name": "diagnostics_bundle",
        "include": ["ops.db", "backup-plan json", "workload diagnostics json", "logs", "readonly status outputs"],
        "exclude": ["private raw blobs unless explicitly needed"],
        "use_case": "incident triage without over-sharing archive contents",
    },
)


def _build_scope_filter(
    *,
    session_ids: tuple[str, ...],
    origin: str | None,
    source_family: str | None,
    source_root: str | None,
    since: str | None,
    until: str | None,
    failure_kind: str | None,
    parser_version: str | None,
) -> MaintenanceScopeFilter:
    """Translate CLI options into a :class:`MaintenanceScopeFilter`.

    Helper exists so the CLI ``plan`` and ``run`` commands share one
    parsing path and one error surface.
    """

    time_range: tuple[datetime, datetime] | None
    if since is not None or until is not None:
        if since is None or until is None:
            raise click.UsageError("--since and --until must be supplied together")
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except ValueError as exc:
            raise click.UsageError(f"--since/--until must be ISO-8601 timestamps: {exc}") from exc
        time_range = (since_dt, until_dt)
    else:
        time_range = None

    return MaintenanceScopeFilter(
        session_ids=session_ids if session_ids else None,
        provider=provider_from_origin(Origin(origin)).value if origin is not None else None,
        source_family=source_family,
        source_root=Path(source_root) if source_root else None,
        time_range=time_range,
        failure_kind=failure_kind,
        parser_version=parser_version,
    )


_SCOPE_FILTER_OPTIONS = [
    click.option(
        "--session-id",
        "session_ids",
        multiple=True,
        help="Restrict scope to one or more session ids (repeatable).",
    ),
    click.option("--origin", "-o", type=str, default=None, help="Restrict scope to one origin token."),
    click.option(
        "--source-family",
        type=str,
        default=None,
        help="Restrict scope to one source family (e.g. claude-code-session).",
    ),
    click.option(
        "--source-root",
        type=str,
        default=None,
        help="Restrict scope to one source runtime root (e.g. ~/.claude/projects).",
    ),
    click.option("--since", type=str, default=None, help="Inclusive ISO-8601 lower bound of the time range."),
    click.option("--until", type=str, default=None, help="Inclusive ISO-8601 upper bound of the time range."),
    click.option(
        "--failure-kind",
        type=str,
        default=None,
        help="Restrict scope to attempts that failed with one kind.",
    ),
    click.option(
        "--parser-version",
        type=str,
        default=None,
        help="Restrict scope to one parser/materializer version.",
    ),
]


def _apply_scope_filter_options(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator stacking the shared scope-filter options onto a command."""
    for option in reversed(_SCOPE_FILTER_OPTIONS):
        fn = option(fn)
    return fn


_MAINTENANCE_TARGET_HELP = build_maintenance_target_catalog().help_text()


def _raw_blob_path_for_hash(root: Path, blob_hash: bytes | str) -> Path | None:
    hex_hash = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash).lower()
    if len(hex_hash) != 64 or any(char not in "0123456789abcdef" for char in hex_hash):
        return None
    return root / "blob" / hex_hash[:2] / hex_hash[2:]


def _missing_raw_blob_cursor_candidates(root: Path, *, limit: int | None = None) -> list[dict[str, object]]:
    source_db = root / "source.db"
    index_db = root / "index.db"
    ops_db = root / "ops.db"
    if not source_db.exists() or not index_db.exists() or not ops_db.exists():
        return []
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    ops_conn = sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True)
    ops_conn.row_factory = sqlite3.Row
    try:
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
        rows = conn.execute(
            """
            SELECT
                r.raw_id,
                r.origin,
                r.native_id,
                r.source_path,
                r.blob_hash,
                r.blob_size,
                r.validation_status,
                r.parse_error
            FROM raw_sessions AS r
            LEFT JOIN index_tier.sessions AS s_by_raw ON s_by_raw.raw_id = r.raw_id
            LEFT JOIN index_tier.sessions AS s_by_native
              ON r.native_id IS NOT NULL
             AND s_by_native.origin = r.origin
             AND s_by_native.native_id = r.native_id
            WHERE r.blob_hash IS NOT NULL
              AND r.source_path IS NOT NULL
              AND s_by_raw.raw_id IS NULL
              AND s_by_native.native_id IS NULL
              AND NOT (
                r.validation_status = 'skipped'
                AND r.parsed_at_ms IS NOT NULL
                AND r.parse_error IS NULL
              )
            ORDER BY r.origin, r.blob_size DESC, r.raw_id
            """
        ).fetchall()
        candidates: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        for row in rows:
            source_path = str(row["source_path"] or "")
            if not source_path or source_path in seen_paths:
                continue
            blob_path = _raw_blob_path_for_hash(root, row["blob_hash"])
            if blob_path is None or blob_path.exists() or not Path(source_path).exists():
                continue
            if _raw_materialized_by_source_path_candidate(conn, row):
                continue
            cursor = ops_conn.execute(
                "SELECT stat_size, byte_offset, updated_at_ms FROM ingest_cursor WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            if cursor is None:
                continue
            candidates.append(
                {
                    "source_path": source_path,
                    "raw_id": str(row["raw_id"]),
                    "origin": str(row["origin"] or ""),
                    "native_id": str(row["native_id"] or ""),
                    "blob_path": str(blob_path),
                    "blob_size": int(row["blob_size"] or 0),
                    "cursor_stat_size": int(cursor["stat_size"] or 0),
                    "cursor_byte_offset": int(cursor["byte_offset"] or 0),
                    "cursor_updated_at_ms": int(cursor["updated_at_ms"] or 0),
                }
            )
            seen_paths.add(source_path)
            if limit is not None and len(candidates) >= limit:
                break
        return candidates
    finally:
        ops_conn.close()
        conn.close()


def _raw_materialized_by_source_path_candidate(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    origin = str(row["origin"] or "")
    if not origin:
        return False
    for native_id in source_path_native_id_candidates(str(row["source_path"] or "")):
        existing = conn.execute(
            """
            SELECT 1
            FROM index_tier.sessions
            WHERE origin = ?
              AND native_id = ?
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


@click.group("maintenance")
def maintenance_group() -> None:
    """Preview and run maintenance backfill operations."""


@maintenance_group.command("plan")
@click.option(
    "--target",
    "targets",
    multiple=True,
    type=click.Choice(MAINTENANCE_TARGET_NAMES),
    help=_MAINTENANCE_TARGET_HELP,
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format. ``json`` emits the shared MaintenanceOperationEnvelope.",
)
@_apply_scope_filter_options
@click.pass_obj
def plan_command(
    env: AppEnv,
    targets: tuple[str, ...],
    output_format: str,
    session_ids: tuple[str, ...],
    origin: str | None,
    source_family: str | None,
    source_root: str | None,
    since: str | None,
    until: str | None,
    failure_kind: str | None,
    parser_version: str | None,
) -> None:
    """Dry-run summary: show what would be rebuilt without executing.

    Displays affected rows and estimated time for each target.
    Read-only — no mutations are performed.
    """
    configure_logging()
    config = Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],  # maintenance doesn't need source acquisition
    )
    scope_filter = _build_scope_filter(
        session_ids=session_ids,
        origin=origin,
        source_family=source_family,
        source_root=source_root,
        since=since,
        until=until,
        failure_kind=failure_kind,
        parser_version=parser_version,
    )
    result = preview_backfill(config, targets=targets, scope_filter=scope_filter)

    if output_format == "json":
        envelope = envelope_from_operation(result, origin="cli", mode="preview")
        click.echo(json.dumps(envelope.to_dict(), indent=2, sort_keys=True))
        return

    click.echo(f"Operation: {result.operation_id}")
    click.echo(f"Targets:  {', '.join(result.targets) if result.targets else 'all'}")
    click.echo(f"Affected: {result.affected_rows:,} rows")
    if result.estimated_time_s > 0:
        click.echo(f"Estimate: ~{result.estimated_time_s:.1f}s")

    if result.results:
        click.echo("\nPer-target preview:")
        for r in result.results:
            name = r.get("name", "unknown")
            issue_count = r.get("issue_count", 0)
            healthy = r.get("healthy", True)
            detail = r.get("detail", "")
            status_str = "OK" if healthy else f"{issue_count:,} issues"
            click.echo(f"  {name}: {status_str}")
            if detail and not healthy:
                click.echo(f"    {detail}")

    if result.error:
        click.echo(f"\nError: {result.error}", err=True)


@maintenance_group.command("archive-plan")
@click.option(
    "--replace-existing",
    is_flag=True,
    help="Classify existing archive tier files as replaceable instead of blocking.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def archive_plan_command(replace_existing: bool, output_format: str) -> None:
    """Inspect readiness for the archive file set.

    Read-only. The command reports the planned source/index/embeddings/user/ops
    target files, required backups, and blockers before archive initialization.
    """
    plan = build_archive_init_plan(
        archive_root=archive_root(),
        replace_existing=replace_existing,
    )

    if output_format == "json":
        click.echo(json.dumps(_archive_plan_payload(plan), indent=2, sort_keys=True))
        return

    _render_archive_plain_plan(plan)


def _archive_plan_payload(plan: ArchiveInitPlan) -> dict[str, object]:
    return {
        "ready": plan.ready,
        "archive_root": str(plan.archive_root),
        "blockers": list(plan.blockers),
        "tiers": [
            {
                "tier": tier_plan.tier.value,
                "path": str(tier_plan.path),
                "durability": tier_plan.durability,
                "exists": tier_plan.exists,
                "user_version": tier_plan.user_version,
                "expected_user_version": tier_plan.expected_user_version,
                "backup_required": tier_plan.backup_required,
                "action": tier_plan.action.value,
                "backup_path": str(tier_plan.backup_path) if tier_plan.backup_path is not None else None,
                "blockers": list(tier_plan.blockers),
            }
            for tier_plan in plan.tiers
        ],
    }


@maintenance_group.command("backup-plan")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def backup_plan_command(output_format: str) -> None:
    """Inspect archive backup boundaries without copying data.

    Reports the backup class and presence of each archive tier, blob-store
    boundary, and the named backup profiles documented for operators.
    """
    root = archive_root()
    payload = _backup_plan_payload(root)

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_backup_plain_plan(payload)


@maintenance_group.command("assertion-export")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "jsonl"]),
    default="jsonl",
    show_default=True,
    help="Export format for assertion rows.",
)
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None, help="Write export to this path.")
@click.option(
    "--kind",
    "kinds",
    multiple=True,
    type=click.Choice([kind.value for kind in AssertionKind]),
    help="Restrict export to one assertion kind; repeatable.",
)
@click.option("--status", "statuses", multiple=True, help="Restrict export to one assertion status; repeatable.")
@click.option("--limit", "-l", type=click.IntRange(min=0), default=None, help="Maximum assertion rows to export.")
def assertion_export_command(
    output_format: str,
    out_path: Path | None,
    kinds: tuple[str, ...],
    statuses: tuple[str, ...],
    limit: int | None,
) -> None:
    """Export the durable assertion substrate from user.db."""

    root = archive_root()
    user_db_path = root / ARCHIVE_TIER_SPECS[ArchiveTier.USER].filename
    rows = _read_assertion_export_rows(
        user_db_path,
        kinds=kinds or None,
        statuses=statuses or None,
        limit=limit,
    )
    payload_rows = [assertion_envelope_to_payload(row) for row in rows]

    if output_format == "json":
        content = (
            json.dumps(
                {
                    "ok": True,
                    "mode": "assertion_export",
                    "archive_root": str(root),
                    "user_db_path": str(user_db_path),
                    "count": len(payload_rows),
                    "assertions": payload_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload_rows)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        click.echo(f"Exported {len(payload_rows)} assertions to {out_path}")
        return

    click.echo(content, nl=False)


def _read_assertion_export_rows(
    user_db_path: Path,
    *,
    kinds: tuple[str, ...] | None,
    statuses: tuple[str, ...] | None,
    limit: int | None,
) -> list[ArchiveAssertionEnvelope]:
    if not user_db_path.exists():
        return []
    uri = f"file:{user_db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return list_assertions_for_export(
            conn,
            kinds=kinds,
            statuses=statuses,
            limit=limit,
        )


def _backup_plan_payload(root: Path) -> dict[str, Any]:
    tier_payloads = []
    wal_warnings: list[str] = []
    for tier in ArchiveTier:
        spec = ARCHIVE_TIER_SPECS[tier]
        path = root / spec.filename
        wal_path = path.with_name(f"{path.name}-wal")
        shm_path = path.with_name(f"{path.name}-shm")
        wal_present = wal_path.exists()
        if wal_present:
            wal_warnings.append(f"{spec.filename}-wal is present; checkpoint before copying {spec.filename}")
        policy = _BACKUP_POLICIES[tier]
        tier_payloads.append(
            {
                "tier": tier.value,
                "filename": spec.filename,
                "path": str(path),
                "present": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "durability": spec.durability,
                "expected_user_version": spec.version,
                "backup_required": spec.backup_required,
                "backup_class": policy["backup_class"],
                "backup_policy": policy["policy"],
                "restore_role": policy["restore_role"],
                "wal_present": wal_present,
                "shm_present": shm_path.exists(),
                "checkpoint_recommended": wal_present,
            }
        )

    blob_root = blob_store_root()
    return {
        "ok": True,
        "mode": "backup_plan",
        "archive_root": str(root),
        "tiers": tier_payloads,
        "blob_store": {
            "path": str(blob_root),
            "present": blob_root.exists(),
            "backup_policy": "back_up_referenced_blobs_with_source_and_user_tiers",
            "gc_safety_boundary": "source.db raw references plus the gc_generations age floor are authoritative",
        },
        "profiles": list(_BACKUP_PROFILES),
        "warnings": wal_warnings,
        "mutates": False,
    }


def _render_backup_plain_plan(payload: dict[str, Any]) -> None:
    click.echo("Archive backup plan")
    click.echo(f"Archive root: {payload['archive_root']}")
    click.echo("")
    click.echo("Tiers:")
    for tier in payload["tiers"]:
        assert isinstance(tier, dict)
        status = "present" if tier["present"] else "missing"
        checkpoint = " checkpoint-before-copy" if tier["checkpoint_recommended"] else ""
        click.echo(f"  {tier['filename']}: {tier['backup_class']} policy={tier['backup_policy']} {status}{checkpoint}")
    click.echo("")
    click.echo("Profiles:")
    for profile in payload["profiles"]:
        assert isinstance(profile, dict)
        click.echo(f"  {profile['name']}: {profile['use_case']}")
    warnings = payload["warnings"]
    if warnings:
        click.echo("")
        click.echo("Warnings:")
        for warning in warnings:
            click.echo(f"  {warning}")


@maintenance_group.command("archive-read")
@click.option("--query", "-q", type=str, default=None, help="Search block text instead of listing sessions.")
@click.option("--origin", type=str, default=None, help="Restrict reads to one origin token.")
@click.option("--limit", "-l", type=int, default=20, show_default=True, help="Maximum rows to return.")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def archive_read_command(query: str | None, origin: str | None, limit: int, output_format: str) -> None:
    """Read index sessions from the archive."""
    root = archive_file_set_root_for_paths(archive_root_path=archive_root(), db_anchor=db_path())
    index_db_path = root / "index.db"
    if not index_db_path.exists():
        message = f"archive index.db does not exist: {index_db_path}"
        if output_format == "json":
            click.echo(json.dumps({"ok": False, "error": message, "index_db_path": str(index_db_path)}, indent=2))
        else:
            click.echo(f"Blocked: {message}", err=True)
        raise SystemExit(1)

    with ArchiveStore.open_existing(root) as archive:
        if query is not None:
            hits = archive.search_summaries(query, limit=limit, origin=origin)
            if output_format == "json":
                click.echo(
                    json.dumps(
                        {
                            "ok": True,
                            "mode": "search",
                            "query": query,
                            "origin": origin,
                            "index_db_path": str(index_db_path),
                            "hits": [
                                {
                                    "rank": hit.rank,
                                    "session_id": hit.session_id,
                                    "block_id": hit.block_id,
                                    "message_id": hit.message_id,
                                    "origin": hit.origin,
                                    "title": hit.title,
                                    "snippet": hit.snippet,
                                }
                                for hit in hits
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return
            for hit in hits:
                title = f" {hit.title}" if hit.title else ""
                click.echo(f"{hit.rank}. {hit.session_id}{title}")
                click.echo(f"   {hit.block_id}: {hit.snippet}")
            return

        summaries = archive.list_summaries(limit=limit, origin=origin)
        if output_format == "json":
            click.echo(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "list",
                        "origin": origin,
                        "index_db_path": str(index_db_path),
                        "sessions": [
                            {
                                "session_id": summary.session_id,
                                "native_id": summary.native_id,
                                "origin": summary.origin,
                                "title": summary.title,
                                "created_at": summary.created_at,
                                "updated_at": summary.updated_at,
                                "message_count": summary.message_count,
                                "word_count": summary.word_count,
                                "tags": list(summary.tags),
                            }
                            for summary in summaries
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        for summary in summaries:
            title = f" {summary.title}" if summary.title else ""
            click.echo(f"{summary.session_id}{title}")
            click.echo(f"  {summary.origin} messages={summary.message_count:,} words={summary.word_count:,}")


@maintenance_group.command("archive-init")
@click.option(
    "--replace-existing",
    is_flag=True,
    help="Back up and replace existing durable archive tier files; recreate disposable ops state.",
)
@click.option("--yes", "-y", is_flag=True, help="Actually create the archive tier files.")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def archive_init_command(replace_existing: bool, yes: bool, output_format: str) -> None:
    """Initialize the archive file set after explicit confirmation.

    This command backs up any durable archive targets it is allowed to replace,
    then creates fresh source, index, embeddings, user, and ops databases.
    Ingest and read surfaces populate and consume the archive.
    """
    plan = build_archive_init_plan(
        archive_root=archive_root(),
        replace_existing=replace_existing,
    )
    if not yes:
        if output_format == "json":
            payload = _archive_plan_payload(plan)
            payload["executed"] = False
            payload["next_action"] = "rerun with --yes to initialize the archive tier file set"
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        click.echo("Dry run only. Use --yes to initialize the archive tier file set.")
        click.echo("")
        _render_archive_plain_plan(plan)
        return

    try:
        result = initialize_archive_tier_files_from_plan(plan)
    except ArchiveInitBlockedError as exc:
        if output_format == "json":
            payload = _archive_plan_payload(plan)
            payload["executed"] = False
            payload["error"] = str(exc)
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _render_archive_plain_plan(plan)
            click.echo(f"\nBlocked: {exc}", err=True)
        raise SystemExit(1) from exc

    if output_format == "json":
        payload = _archive_init_payload(result)
        payload["executed"] = True
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("Initialized:")
    for tier_result in result.tier_results:
        backup = f" backup={tier_result.backup_path}" if tier_result.backup_path is not None else ""
        click.echo(f"  {tier_result.tier}: {tier_result.action.value} {tier_result.path}{backup}")


def _render_archive_plain_plan(plan: ArchiveInitPlan) -> None:
    status = "ready" if plan.ready else "blocked"
    click.echo(f"Archive tier initialization: {status}")
    click.echo(f"Archive root: {plan.archive_root}")
    click.echo("")
    click.echo("Targets:")
    for tier_plan in plan.tiers:
        backup = f" backup={tier_plan.backup_path}" if tier_plan.backup_path is not None else ""
        version = f" version={tier_plan.user_version}" if tier_plan.user_version is not None else ""
        click.echo(
            f"  {tier_plan.tier.value}: {tier_plan.action.value} {tier_plan.path}"
            f" durability={tier_plan.durability} expected_version={tier_plan.expected_user_version}"
            f"{version}{backup}"
        )
        for blocker in tier_plan.blockers:
            click.echo(f"    blocker: {blocker}")
    if plan.blockers:
        click.echo("")
        click.echo("Blockers:")
        for blocker in plan.blockers:
            click.echo(f"  {blocker}")


def _archive_init_payload(result: ArchiveInitResult) -> dict[str, object]:
    return {
        "tiers": [
            {
                "tier": tier_result.tier,
                "path": str(tier_result.path),
                "action": tier_result.action.value,
                "backup_path": str(tier_result.backup_path) if tier_result.backup_path is not None else None,
                "initialized": tier_result.initialized,
            }
            for tier_result in result.tier_results
        ],
    }


@maintenance_group.command("migrate-tier")
@click.argument("tier", type=click.Choice(tuple(sorted(tier.value for tier in DURABLE_MIGRATION_TIERS))))
@click.option(
    "--backup-manifest",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Verified polylogue ops backup manifest or backup directory containing manifest.json.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def migrate_tier_command(tier: str, backup_manifest: Path, output_format: str) -> None:
    """Apply additive migrations for one durable archive tier.

    Derived tiers are intentionally excluded from this command; rebuild or
    blue-green replace those from source evidence instead.
    """
    archive_tier = ArchiveTier(tier)
    spec = ARCHIVE_TIER_SPECS[archive_tier]
    path = archive_root() / spec.filename
    try:
        with sqlite3.connect(path) as conn:
            result = migrate_archive_tier(conn, archive_tier, backup_manifest=backup_manifest)
    except (sqlite3.Error, MigrationError) as exc:
        if output_format == "json":
            click.echo(
                json.dumps(
                    {
                        "ok": False,
                        "tier": tier,
                        "path": str(path),
                        "backup_manifest": str(backup_manifest),
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            click.echo(f"Migration blocked for {tier}: {exc}", err=True)
        raise SystemExit(1) from exc

    payload = {
        "ok": True,
        "tier": tier,
        "path": str(path),
        "backup_manifest": str(backup_manifest),
        "from_version": result.from_version,
        "to_version": result.to_version,
        "applied_versions": list(result.applied_versions),
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    applied = ", ".join(str(version) for version in result.applied_versions) or "none"
    click.echo(f"Migrated {tier}: {result.from_version} -> {result.to_version} (applied: {applied})")


@maintenance_group.command("run")
@click.option(
    "--target",
    "targets",
    multiple=True,
    type=click.Choice(MAINTENANCE_TARGET_NAMES),
    help=_MAINTENANCE_TARGET_HELP,
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would happen without executing",
)
@click.option(
    "--operation-id",
    "operation_id",
    type=str,
    default=None,
    help=("Reuse a previous operation id to resume an interrupted run; omit to mint a fresh uuid for a new operation."),
)
@click.option(
    "--resume",
    "resume_cursor",
    type=str,
    default=None,
    help=(
        "Explicit resume cursor (e.g. 'target:2'). When omitted and "
        "--operation-id matches a persisted state file, the cursor is "
        "loaded automatically."
    ),
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format. ``json`` emits the shared MaintenanceOperationEnvelope.",
)
@_apply_scope_filter_options
@click.pass_obj
def run_command(
    env: AppEnv,
    targets: tuple[str, ...],
    dry_run: bool,
    operation_id: str | None,
    resume_cursor: str | None,
    output_format: str,
    session_ids: tuple[str, ...],
    origin: str | None,
    source_family: str | None,
    source_root: str | None,
    since: str | None,
    until: str | None,
    failure_kind: str | None,
    parser_version: str | None,
) -> None:
    """Run (or dry-run) maintenance backfill operations.

    Executes targeted rebuilds using existing repair infrastructure.
    Per-target failures are isolated: one failing target does not abort
    the remaining work. Use --operation-id together with --resume to
    pick up an interrupted operation from its last checkpoint.
    """
    configure_logging()
    config = Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],
    )

    def _emit_progress(snapshot: ReplayProgress) -> None:
        detail = f" {snapshot.progress_desc}" if snapshot.progress_desc else ""
        amount = f" amount={snapshot.progress_amount}" if snapshot.progress_amount is not None else ""
        click.echo(
            f"  [{snapshot.processed}/{snapshot.total}] {snapshot.target} "
            f"cursor={snapshot.cursor} failures={snapshot.in_flight_failures}{amount}{detail}",
            err=True,
        )

    scope_filter = _build_scope_filter(
        session_ids=session_ids,
        origin=origin,
        source_family=source_family,
        source_root=source_root,
        since=since,
        until=until,
        failure_kind=failure_kind,
        parser_version=parser_version,
    )
    result = execute_replay(
        config,
        targets=targets,
        operation_id=operation_id,
        resume_cursor=resume_cursor,
        dry_run=dry_run,
        progress_callback=_emit_progress,
        scope_filter=scope_filter,
    )

    if output_format == "json":
        envelope = envelope_from_operation(result, origin="cli", mode="execute")
        click.echo(json.dumps(envelope.to_dict(), indent=2, sort_keys=True))
        return

    action = "Would affect" if dry_run else "Processed"
    click.echo(f"Operation: {result.operation_id}")
    click.echo(f"Targets:  {', '.join(result.targets) if result.targets else 'all'}")
    click.echo(f"Status:   {result.status.value}")
    click.echo(f"Cursor:   {result.resume_cursor}")
    click.echo(f"{action}:  {result.affected_rows:,} rows")

    if result.results:
        click.echo(f"\n{'Would-be' if dry_run else ''} Results:")
        for r in result.results:
            name = r.get("name", "unknown")
            success = r.get("success", False)
            repaired = r.get("repaired_count", 0)
            detail = r.get("detail", "")
            status_icon = "OK" if success else "FAILED"
            click.echo(f"  {name}: {status_icon} ({repaired} items)")
            if detail:
                click.echo(f"    {detail}")

    if result.error:
        click.echo(f"\nError: {result.error}", err=True)

    if result.failure_samples.samples:
        click.echo("\nFailures:", err=True)
        for sample in result.failure_samples.samples:
            click.echo(f"  {sample.kind} @ {sample.locator}: {sample.message}", err=True)
        if result.failure_samples.truncated:
            click.echo("  (failure samples truncated)", err=True)

    if result.completed_at:
        from datetime import datetime

        if result.started_at:
            started = datetime.fromisoformat(result.started_at)
            completed = datetime.fromisoformat(result.completed_at)
            elapsed = (completed - started).total_seconds()
            click.echo(f"\nElapsed: {elapsed:.1f}s")


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


def _selected_raw_blob_bytes(root: Path, raw_ids: list[str]) -> int:
    if not raw_ids:
        return 0
    source_db = root / "source.db"
    if not source_db.exists():
        return 0
    total = 0
    chunk_size = 900
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0) as conn:
        for offset in range(0, len(raw_ids), chunk_size):
            chunk = raw_ids[offset : offset + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            row = conn.execute(
                f"SELECT COALESCE(SUM(blob_size), 0) FROM raw_sessions WHERE raw_id IN ({placeholders})",
                tuple(chunk),
            ).fetchone()
            total += int(row[0] or 0) if row is not None else 0
    return total


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


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _record_rebuild_index_attempt(
    root: Path,
    *,
    attempt_id: str | None = None,
    status: str,
    started_at_ms: int,
    finished_at_ms: int | None = None,
    parsed_raw_count: int = 0,
    materialized_count: int = 0,
    error_message: str | None = None,
) -> str | None:
    ops_db = root / "ops.db"
    if not ops_db.exists():
        return None
    try:
        with sqlite3.connect(ops_db, timeout=10.0) as conn:
            return record_ingest_attempt(
                conn,
                attempt_id=attempt_id,
                source_path=str(root / "source.db"),
                status=status,
                phase="rebuild-index",
                storage_route="maintenance",
                started_at_ms=started_at_ms,
                heartbeat_at_ms=_now_ms() if status == "running" else None,
                finished_at_ms=finished_at_ms,
                parsed_raw_count=parsed_raw_count,
                materialized_count=materialized_count,
                error_message=error_message,
                source_paths_json=json.dumps([str(root / "source.db")]),
            )
    except sqlite3.Error:
        return None


@maintenance_group.command("rebuild-index")
@click.option("--batch-size", type=int, default=50, show_default=True, help="Maximum raw records per ingest batch.")
@click.option("--workers", type=int, default=None, help="Optional ingest worker count.")
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
    "--force-write",
    is_flag=True,
    help="Rewrite sessions even when the parsed content hash matches existing index rows.",
)
@click.option(
    "--no-materialize",
    is_flag=True,
    help="Skip the final full session-insight materialization pass.",
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
def rebuild_index_command(
    batch_size: int,
    workers: int | None,
    only_missing: bool,
    raw_ids: tuple[str, ...],
    max_blob_mb: float | None,
    force_write: bool,
    no_materialize: bool,
    plan_only: bool,
    plan_limit: int,
    output_format: str,
) -> None:
    """Replay durable source rows into index.db, then rebuild read models.

    This is the canonical post-reset path for a rebuildable index tier:
    ``source.db`` remains the durable evidence root, ``index.db`` is recreated
    from those rows, and session insight tables are rebuilt from the resulting
    sessions unless ``--no-materialize`` is supplied.
    """
    configure_logging()
    if batch_size <= 0:
        raise click.BadParameter("batch size must be positive", param_hint="--batch-size")
    if workers is not None and workers <= 0:
        raise click.BadParameter("workers must be positive", param_hint="--workers")
    if raw_ids and only_missing:
        raise click.UsageError("--raw-id cannot be combined with --only-missing")
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
    selected_blob_bytes = _selected_raw_blob_bytes(root, selected_raw_ids)
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
    if only_missing and selected_raw_count == 0 and no_materialize:
        payload = {
            "archive_root": str(root),
            "raw_session_count": raw_count,
            "selected_raw_count": 0,
            "skipped_by_blob_limit_count": 0,
            "selected_blob_bytes": 0,
            "only_missing": True,
            "status": "ok",
            "materialized": False,
            "parse_counts": {},
            "changed_counts": {},
            "processed_session_count": 0,
            "parse_failure_count": 0,
            "batch_count": 0,
            "materialized_session_count": 0,
            "materialized_rebuilt": False,
            "materialize_observation": None,
        }
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            click.echo(f"Archive root: {root}")
            click.echo(f"Raw rows:     {raw_count:,}")
            click.echo("Selected:     0 missing raw row(s)")
            click.echo("Materialized: skipped")
            click.echo("Elapsed:      0.0s")
        return

    started = datetime.now(UTC)
    started_ms = int(started.timestamp() * 1000)
    config = Config(archive_root=root, render_root=render_root(), sources=[])
    rebuild_attempt_id = _record_rebuild_index_attempt(
        root,
        status="running",
        started_at_ms=started_ms,
        parsed_raw_count=0,
        materialized_count=0,
    )
    if output_format == "plain":
        click.echo(f"Selected:     {selected_raw_count:,} raw row(s)")
        click.echo(f"Blob bytes:   {selected_blob_bytes:,}")

    def _emit_progress(amount: int, desc: str | None = None) -> None:
        del amount
        if desc:
            click.echo(f"  {desc}", err=True)

    try:
        result = asyncio.run(
            rebuild_index_from_source(
                config,
                raw_ids=selected_raw_ids,
                raw_batch_size=batch_size,
                ingest_workers=workers,
                force_write=force_write,
                materialize=not no_materialize,
                progress_callback=_emit_progress if output_format == "plain" else None,
            )
        )
    except BaseException as exc:
        _record_rebuild_index_attempt(
            root,
            attempt_id=rebuild_attempt_id,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            started_at_ms=started_ms,
            finished_at_ms=_now_ms(),
            error_message=str(exc),
        )
        raise
    completed = datetime.now(UTC)
    _record_rebuild_index_attempt(
        root,
        attempt_id=rebuild_attempt_id,
        status="completed" if result["parse_failure_count"] == 0 else "failed",
        started_at_ms=started_ms,
        finished_at_ms=int(completed.timestamp() * 1000),
        parsed_raw_count=selected_raw_count,
        materialized_count=result["materialized_session_count"]
        if isinstance(result["materialized_session_count"], int)
        else 0,
        error_message=None
        if result["parse_failure_count"] == 0
        else f"{result['parse_failure_count']} parse failure(s)",
    )
    payload = {
        "archive_root": str(root),
        "raw_session_count": raw_count,
        "selected_raw_count": selected_raw_count,
        "selected_blob_bytes": selected_blob_bytes,
        "skipped_by_blob_limit_count": skipped_by_blob_limit_count,
        "only_missing": only_missing,
        "raw_id_count": len(raw_ids),
        "max_blob_mb": max_blob_mb,
        "batch_size": batch_size,
        "workers": workers,
        "force_write": force_write,
        "status": "ok" if result["parse_failure_count"] == 0 else "parse-failures",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_s": round((completed - started).total_seconds(), 3),
        **result,
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Archive root: {root}")
    click.echo(f"Raw rows:     {raw_count:,}")
    if only_missing:
        click.echo(f"Selected:     {selected_raw_count:,} missing raw row(s)")
    elif raw_ids:
        click.echo(f"Selected:     {selected_raw_count:,} explicit raw row(s)")
    if skipped_by_blob_limit_count:
        click.echo(f"Blob limit:   skipped {skipped_by_blob_limit_count:,} raw row(s)")
    click.echo(f"Parsed:       {result['processed_session_count']:,} changed session(s)")
    click.echo(f"Batches:      {result['batch_count']:,}")
    click.echo(f"Failures:     {result['parse_failure_count']:,}")
    if result["materialized"]:
        click.echo(f"Materialized: {result['materialized_session_count']:,} session insight row(s)")
    else:
        click.echo("Materialized: skipped")
    click.echo(f"Elapsed:      {payload['elapsed_s']:.1f}s")


@maintenance_group.command("missing-raw-blob-cursors")
@click.option("--apply", "apply_changes", is_flag=True, help="Delete matching rebuildable live cursor rows.")
@click.option("--limit", "-l", type=int, default=None, help="Limit the number of candidate source paths.")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.pass_obj
def missing_raw_blob_cursors_command(
    env: AppEnv,
    apply_changes: bool,
    limit: int | None,
    output_format: str,
) -> None:
    """Invalidate cursors hiding missing raw-blob re-acquisition debt.

    This command only touches ``ops.db.ingest_cursor`` rows. It leaves
    source-tier raw rows, source files, blobs, index rows, and user state
    intact so the next daemon catch-up can re-acquire through the normal
    ingestion path.
    """
    del env
    root = archive_root()
    candidates = _missing_raw_blob_cursor_candidates(root, limit=limit)
    deleted = 0
    if apply_changes and candidates:
        ops_db = root / "ops.db"
        with sqlite3.connect(ops_db) as conn:
            for candidate in candidates:
                deleted += conn.execute(
                    "DELETE FROM ingest_cursor WHERE source_path = ?",
                    (str(candidate["source_path"]),),
                ).rowcount
            conn.commit()

    payload = {
        "archive_root": str(root),
        "mode": "apply" if apply_changes else "dry-run",
        "candidate_count": len(candidates),
        "deleted_cursor_count": deleted,
        "candidates": candidates,
        "next_action": "restart or run polylogued catch-up" if apply_changes and deleted else None,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    action = "Deleted" if apply_changes else "Would delete"
    click.echo(f"{action} {deleted if apply_changes else len(candidates)} live cursor row(s)")
    for candidate in candidates[:10]:
        click.echo(
            f"  {candidate['origin']} {candidate['source_path']} "
            f"raw={candidate['raw_id']} blob_size={candidate['blob_size']}"
        )
    if len(candidates) > 10:
        click.echo(f"  ... {len(candidates) - 10} more")
    if apply_changes and deleted:
        click.echo("Next: restart or run polylogued catch-up.")


@maintenance_group.command("preview")
@click.option(
    "--scope",
    "scopes",
    multiple=True,
    type=click.Choice(ALL_SCOPES),
    help="Limit preview to named scopes (derived, retrieval, archive_cleanup, backfill).",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--shallow",
    is_flag=True,
    help="Skip the expensive full-verification path (faster, slightly less accurate).",
)
@click.pass_obj
def preview_command(
    env: AppEnv,
    scopes: tuple[str, ...],
    output_format: str,
    shallow: bool,
) -> None:
    """Staleness inventory by model and scope. Read-only.

    Shows per-model counts of stale/missing/orphan rows with typed
    :class:`InvalidationReason` tags. Use before triggering ``polylogue
    maintenance run`` so the operator knows what will be rebuilt and why.
    Models with nothing stale produce explicit zero rows rather than
    being absent from the output.
    """

    configure_logging()
    inventory = staleness_inventory(
        scopes=scopes or None,
        verify_full=not shallow,
    )

    if output_format == "json":
        click.echo(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
        return

    click.echo(f"Captured: {inventory.captured_at}")
    click.echo(f"Database: {inventory.db_path}")
    click.echo(f"Scopes:   {', '.join(inventory.scopes)}")
    click.echo(f"Total stale rows: {inventory.total_stale():,}")
    click.echo("")

    by_model = inventory.by_model()
    if not by_model:
        click.echo("No models inventoried.")
        return

    for model, items in sorted(by_model.items()):
        click.echo(f"{model}:")
        for item in items:
            fraction_pct = item.fraction * 100.0
            click.echo(
                f"  {item.reason.value:>20s}  count={item.count:>10,}  fraction={fraction_pct:>5.1f}%  {item.detail}"
            )
        click.echo("")


@maintenance_group.command("blob-gc")
@click.option(
    "--max-batch",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of eligible blobs to delete or preview.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Actually delete eligible blobs. Without this flag the command is a dry-run preview.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_gc_command(max_batch: int, yes: bool, output_format: str) -> None:
    """Preview or run lease-safe blob garbage collection.

    The default is a dry-run report. Pass ``--yes`` to reclaim eligible blobs.
    """
    configure_logging()
    config = Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],
    )
    result = run_blob_gc_report(
        config.db_path,
        blob_store_root(),
        max_batch=max_batch,
        dry_run=not yes,
    )
    payload = {
        "ok": True,
        "mode": "blob_gc",
        "mutates": bool(yes),
        **result.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    action = "would delete" if result.dry_run else "deleted"
    affected = result.would_delete_count if result.dry_run else result.deleted_count
    click.echo("Blob GC dry-run" if result.dry_run else "Blob GC")
    click.echo(f"Archive DB: {result.db_path}")
    click.echo(f"Blob root:  {result.blob_dir}")
    click.echo(f"Candidates: {result.candidate_count:,}")
    click.echo(f"Inspected:  {result.inspected_count:,}")
    click.echo(f"Result:     {action} {affected:,} blob(s)")
    click.echo(
        "Skipped:    "
        f"referenced={result.skipped_referenced:,} "
        f"reserved={result.skipped_reserved:,} "
        f"missing={result.skipped_missing:,} "
        f"unlink_error={result.skipped_unlink_error:,}"
    )
    if not result.dry_run:
        click.echo(f"Reclaimed:  {result.reclaimed_bytes:,} byte(s)")
        if result.generation_id is not None:
            click.echo(f"Generation: {result.generation_id}")


@maintenance_group.command("blob-reference-debt")
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative missing-blob samples to include.",
)
@click.option(
    "--group-limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of grouped classifications to include.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_reference_debt_command(sample_limit: int, group_limit: int, output_format: str) -> None:
    """Classify missing referenced blobs without mutating the archive."""
    report = classify_blob_reference_debt(
        archive_root() / "source.db",
        sample_size=sample_limit,
        group_limit=group_limit,
    )
    payload = {
        "mode": "blob_reference_debt",
        "mutates": False,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_blob_reference_debt_plain(report)


def _render_blob_reference_debt_plain(report: BlobReferenceDebtClassificationReport) -> None:
    click.echo("Blob reference debt")
    click.echo(f"Source DB:    {report.source_db}")
    click.echo(f"Blob root:    {report.blob_root}")
    click.echo(f"References:   {report.reference_rows:,} row(s), {report.distinct_referenced_blobs:,} distinct blob(s)")
    click.echo(f"Missing:      {report.missing_distinct_blobs:,} distinct blob(s)")
    click.echo(f"Status:       {'ok' if report.ok else 'debt-present'}")

    def _render_counts(label: str, counts: dict[str, int]) -> None:
        if not counts:
            return
        rendered = ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items()))
        click.echo(f"{label}: {rendered}")

    _render_counts("By table    ", report.missing_by_table)
    _render_counts("By ref type ", report.missing_by_ref_type)
    _render_counts("By origin   ", report.missing_by_origin)
    _render_counts("Ref-id join ", report.missing_ref_id_join)
    _render_counts("Source paths", report.missing_source_path_presence)
    _render_counts("Validation  ", report.missing_validation_status)
    _render_counts("Parse errors", report.missing_parse_error)

    if report.top_groups:
        click.echo("Top groups:")
        for group in report.top_groups:
            tables_value = group.get("tables", ())
            ref_types_value = group.get("ref_types", ())
            origins_value = group.get("origins", ())
            count_value = group.get("count", 0)
            tables = ",".join(str(item) for item in tables_value) if isinstance(tables_value, list | tuple) else ""
            ref_types = (
                ",".join(str(item) for item in ref_types_value) if isinstance(ref_types_value, list | tuple) else ""
            )
            origins = ",".join(str(item) for item in origins_value) if isinstance(origins_value, list | tuple) else ""
            count = count_value if isinstance(count_value, int) else 0
            click.echo(f"  {count:>8,}  tables={tables} ref_types={ref_types} origins={origins}")

    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            source = sample.sample_source_path or "(none)"
            origin = ",".join(sample.origins) if sample.origins else "(none)"
            click.echo(
                f"  {sample.blob_hash} origin={origin} source_available={sample.sample_source_available} {source}"
            )


@maintenance_group.command("attachment-acquisition-debt")
@click.option(
    "--sample-limit",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of representative acquired-but-missing attachment ids to include.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def attachment_acquisition_debt_command(sample_limit: int, output_format: str) -> None:
    """Classify index-tier attachment acquisition state without mutating the archive.

    Deliberately separate from ``blob-reference-debt``: unfetched attachments
    (``blob_hash IS NULL``) are an honest floor, never counted as missing
    referenced blobs. Only an acquired attachment whose blob file is absent
    from the store is genuine attachment acquisition debt.
    """
    report = scan_attachment_acquisition_debt(
        archive_root() / "index.db",
        sample_size=sample_limit,
    )
    payload = {
        "mode": "attachment_acquisition_debt",
        "mutates": False,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("Attachment acquisition debt")
    click.echo(f"Total attachments: {report.total_attachments:,}")
    click.echo(f"Acquired:          {report.acquired_count:,}")
    click.echo(f"Unavailable:       {report.unavailable_count:,}")
    click.echo(f"Unfetched:         {report.unfetched_count:,} (honest floor, not missing blobs)")
    click.echo(f"Acquired missing:  {report.acquired_missing_blob_count:,} (genuine debt)")
    click.echo(f"Status:            {'ok' if report.ok else 'debt-present'}")
    if report.acquired_missing_blob_sample:
        click.echo("Sample attachment ids with a missing blob file:")
        for attachment_id in report.acquired_missing_blob_sample:
            click.echo(f"  {attachment_id}")


@maintenance_group.command("blob-reference-recovery-plan")
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative raw-backed missing blob rows to include.",
)
@click.option(
    "--manifest-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSONL destination for the complete raw-backed missing blob recovery manifest.",
)
@click.option(
    "--include-rows",
    is_flag=True,
    help="Include every plan row in JSON output. By default JSON output includes samples plus aggregate counts.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_reference_recovery_plan_command(
    sample_limit: int,
    manifest_file: Path | None,
    include_rows: bool,
    output_format: str,
) -> None:
    """Plan recovery for raw-backed missing blobs without mutating archive state."""
    report = plan_raw_backed_blob_reference_recovery(
        archive_root() / "source.db",
        manifest_path=manifest_file,
        sample_size=sample_limit,
        include_rows=include_rows,
    )
    payload = {
        "mode": "blob_reference_recovery_plan",
        "mutates": False,
        "writes_manifest": manifest_file is not None,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_blob_reference_recovery_plan_plain(report)


def _render_blob_reference_recovery_plan_plain(report: BlobReferenceRecoveryPlanReport) -> None:
    click.echo("Blob reference raw-backed recovery plan")
    click.echo(f"Source DB:    {report.source_db}")
    click.echo(f"Blob root:    {report.blob_root}")
    click.echo(f"Missing:      {report.missing_raw_backed_blobs:,} raw-backed blob(s)")

    def _render_counts(label: str, counts: dict[str, int]) -> None:
        if not counts:
            return
        rendered = ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items()))
        click.echo(f"{label}: {rendered}")

    _render_counts("By action   ", report.by_action)
    _render_counts("By origin   ", report.by_origin)
    _render_counts("By shape    ", report.by_source_shape)
    if report.manifest_path:
        click.echo(f"Manifest:    {report.manifest_path}")
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            source = sample.source_path or "(none)"
            click.echo(f"  {sample.action} {sample.blob_hash} origin={sample.origin} {source}")


@maintenance_group.command("blob-reference-replace-from-source")
@click.option(
    "--yes",
    "apply",
    is_flag=True,
    help="Apply the replacement. Without this flag the command is a dry run.",
)
@click.option(
    "--manifest-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="JSONL destination for before/after replacement rows. Required with --yes.",
)
@click.option("--max-count", type=int, default=None, help="Maximum number of candidate rows to process.")
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative replacement rows to include.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_reference_replace_from_source_command(
    apply: bool,
    manifest_file: Path | None,
    max_count: int | None,
    sample_limit: int,
    output_format: str,
) -> None:
    """Replace raw-backed missing blob refs with current source-derived bytes."""
    if apply and manifest_file is None:
        raise click.UsageError("--manifest-file is required with --yes")
    report = replace_raw_backed_blob_reference_debt_from_source(
        archive_root() / "source.db",
        dry_run=not apply,
        manifest_path=manifest_file,
        max_count=max_count,
        sample_size=sample_limit,
    )
    payload = {
        "mode": "blob_reference_replace_from_source",
        "mutates": apply,
        "writes_manifest": manifest_file is not None,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_blob_reference_replace_from_source_plain(report)


def _render_blob_reference_replace_from_source_plain(report: BlobReferenceSourceReplaceReport) -> None:
    click.echo("Blob reference current-source replacement")
    click.echo(f"Source DB:    {report.source_db}")
    click.echo(f"Blob root:    {report.blob_root}")
    click.echo(f"Mode:         {'dry-run' if report.dry_run else 'apply'}")
    click.echo(f"Scanned:      {report.scanned_rows:,} raw-backed row(s)")
    click.echo(f"Candidates:   {report.candidate_rows:,}")
    click.echo(f"Replaced:     {report.replaced_rows:,}")
    click.echo(f"Written:      {report.written_blobs:,} blob(s), {report.written_bytes:,} byte(s)")
    click.echo(
        "Skipped:      "
        f"existing={report.skipped_existing_blob:,} "
        f"no_source={report.skipped_no_source_path:,} "
        f"source_missing={report.skipped_source_missing:,} "
        f"source_index={report.skipped_source_index:,} "
        f"unsupported={report.skipped_unsupported_source:,} "
        f"error={report.skipped_error:,}"
    )
    if report.manifest_path:
        click.echo(f"Manifest:    {report.manifest_path}")
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            detail = f" reason={sample.reason}" if sample.reason else ""
            click.echo(
                f"  {sample.action} raw_id={sample.raw_id} old={sample.old_blob_hash} "
                f"new={sample.new_blob_hash}{detail}"
            )


@maintenance_group.command("blob-reference-prune-orphans")
@click.option(
    "--max-count",
    type=int,
    default=None,
    help="Maximum number of orphan blob-reference rows to prune or preview.",
)
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative samples to include.",
)
@click.option(
    "--quarantine-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "JSONL destination for rows removed by --yes. Defaults to "
        "<archive-root>/.maintenance-state/blob-ref-quarantine/<timestamp>.jsonl."
    ),
)
@click.option(
    "--yes",
    is_flag=True,
    help="Delete orphan blob_refs after writing them to the quarantine JSONL.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_reference_prune_orphans_command(
    max_count: int | None,
    sample_limit: int,
    quarantine_file: Path | None,
    yes: bool,
    output_format: str,
) -> None:
    """Quarantine and prune missing blob_refs that no longer have raw rows."""
    report = prune_orphan_blob_reference_debt(
        archive_root() / "source.db",
        dry_run=not yes,
        quarantine_path=quarantine_file,
        max_count=max_count,
        sample_size=sample_limit,
    )
    payload = {
        "mode": "blob_reference_prune_orphans",
        "mutates": bool(yes),
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_blob_reference_prune_orphans_plain(report)


def _render_blob_reference_prune_orphans_plain(report: BlobReferenceOrphanPruneReport) -> None:
    click.echo("Blob reference orphan prune")
    click.echo(f"Source DB:    {report.source_db}")
    click.echo(f"Blob root:    {report.blob_root}")
    click.echo(f"Mode:         {'dry-run' if report.dry_run else 'apply'}")
    click.echo(f"Blob refs:    {report.scanned_blob_refs:,} scanned")
    click.echo(
        f"Orphans:      {report.missing_orphan_refs:,} row(s), "
        f"{report.missing_orphan_distinct_blobs:,} distinct blob(s)"
    )
    action = "would prune" if report.dry_run else "pruned"
    click.echo(
        f"Result:       {action} {report.missing_orphan_refs if report.dry_run else report.pruned_refs:,} row(s)"
    )
    click.echo(
        "Skipped:      "
        f"existing_blob={report.skipped_existing_blob:,} "
        f"raw_session_present={report.skipped_raw_session_present:,}"
    )
    if report.quarantine_path:
        click.echo(f"Quarantine:   {report.quarantine_path}")
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            source = sample.source_path or "(none)"
            click.echo(f"  {sample.action} {sample.blob_hash} ref_id={sample.ref_id} {source}")


@maintenance_group.command("gc-history")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of recent GC passes to display.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.pass_obj
def gc_history_command(env: AppEnv, limit: int, output_format: str) -> None:
    """Show recent blob-GC passes recorded in ``gc_generations``.

    Surfaces the typed reclaim counters (``reclaimed_count`` /
    ``reclaimed_bytes``) and start/completion timestamps written by
    ``run_blob_gc`` so operators can audit GC reclamation over time
    without bespoke SQLite tooling.

    A pass whose ``completed_at_ms`` is null crashed mid-run; the row is
    still surfaced so operators can see it happened.
    """
    configure_logging()
    config = Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],
    )
    history = read_gc_history(config.db_path, limit=limit)

    def _iso_ms(epoch_ms: int | None) -> str | None:
        if epoch_ms is None:
            return None
        return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat()

    if output_format == "json":
        payload = [
            {
                "generation_id": row.generation_id,
                "started_at_ms": row.started_at_ms,
                "started_at_iso": _iso_ms(row.started_at_ms),
                "completed_at_ms": row.completed_at_ms,
                "completed_at_iso": _iso_ms(row.completed_at_ms),
                "reclaimed_count": row.reclaimed_count,
                "reclaimed_bytes": row.reclaimed_bytes,
            }
            for row in history
        ]
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not history:
        click.echo("No GC generations recorded yet.")
        return

    click.echo(f"Recent blob-GC passes (newest first, limit={limit}):")
    click.echo("")
    for row in history:
        when = (
            datetime.fromtimestamp(row.completed_at_ms / 1000, tz=UTC).isoformat(timespec="seconds")
            if row.completed_at_ms is not None
            else "unknown (crashed mid-pass)"
        )
        click.echo(f"  generation={row.generation_id}  completed_at={when}")
        click.echo(f"    reclaimed_count={row.reclaimed_count}  reclaimed_bytes={row.reclaimed_bytes}")


@maintenance_group.command("status")
@click.option(
    "--operation-id",
    "operation_id",
    type=str,
    default=None,
    help="Show one operation by id. Omit to list all in-flight and recent operations.",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Include completed operations in the listing (default: only running / failed).",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format. ``json`` emits the shared MaintenanceOperationEnvelope per record.",
)
@click.pass_obj
def status_command(
    env: AppEnv,
    operation_id: str | None,
    show_all: bool,
    output_format: str,
) -> None:
    """Inspect persisted maintenance operations (#1197).

    Without ``--operation-id``, lists every persisted operation under
    ``<archive_root>/.maintenance-state/``. By default the listing hides
    completed operations to surface only in-flight or failed work; pass
    ``--all`` to include them.

    With ``--operation-id``, tails one operation: emits the same shared
    :class:`~polylogue.maintenance.envelope.MaintenanceOperationEnvelope`
    that the CLI ``plan``/``run`` commands, daemon HTTP, and MCP tools
    return.
    """
    configure_logging()
    config = Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],
    )
    registry = MaintenanceOperationRegistry(config=config)

    if operation_id is not None:
        record = registry.get_operation(operation_id)
        if record is None:
            if output_format == "json":
                click.echo(json.dumps({"error": "not_found", "operation_id": operation_id}))
            else:
                click.echo(f"No persisted operation with id {operation_id!r}.", err=True)
            raise click.exceptions.Exit(code=1)
        envelope = envelope_from_operation(record.operation, origin="cli", mode="execute")
        if output_format == "json":
            single_payload: dict[str, object] = {
                "envelope": envelope.to_dict(),
                "updated_at": record.updated_at,
                "state_path": str(record.state_path),
            }
            click.echo(json.dumps(single_payload, indent=2, sort_keys=True))
            return
        _render_record_plain(record)
        return

    records = registry.list_operations()
    if not show_all:
        records = tuple(r for r in records if r.status.value != "completed")

    if output_format == "json":
        list_payload: dict[str, object] = {
            "operations": [
                {
                    "envelope": envelope_from_operation(r.operation, origin="cli", mode="execute").to_dict(),
                    "updated_at": r.updated_at,
                    "state_path": str(r.state_path),
                }
                for r in records
            ],
            "total": len(records),
        }
        click.echo(json.dumps(list_payload, indent=2, sort_keys=True))
        return

    if not records:
        click.echo("No persisted maintenance operations.")
        return
    click.echo(f"Persisted maintenance operations ({len(records)} total, newest first):")
    click.echo("")
    for record in records:
        targets = ", ".join(record.operation.targets) if record.operation.targets else "all"
        click.echo(
            f"  {record.operation_id}  status={record.status.value:>9s}  "
            f"updated_at={record.updated_at}  targets={targets}"
        )
        if record.operation.resume_cursor:
            click.echo(f"      cursor={record.operation.resume_cursor}")
        if record.operation.failure_samples.samples:
            n = len(record.operation.failure_samples.samples)
            click.echo(f"      failures={n}")


def _render_record_plain(record: OperationRecord) -> None:
    """Render one operation record in human-readable form."""
    op = record.operation
    click.echo(f"Operation: {op.operation_id}")
    click.echo(f"Status:    {op.status.value}")
    click.echo(f"Updated:   {record.updated_at}")
    click.echo(f"Targets:   {', '.join(op.targets) if op.targets else 'all'}")
    click.echo(f"Progress:  {op.progress * 100.0:.1f}%")
    click.echo(f"Affected:  {op.affected_rows:,} rows")
    if op.resume_cursor:
        click.echo(f"Cursor:    {op.resume_cursor}")
    if op.started_at:
        click.echo(f"Started:   {op.started_at}")
    if op.completed_at:
        click.echo(f"Completed: {op.completed_at}")
    if op.error:
        click.echo(f"Error:     {op.error}", err=True)
    if op.failure_samples.samples:
        click.echo("Failures:", err=True)
        for sample in op.failure_samples.samples:
            click.echo(f"  {sample.kind} @ {sample.locator}: {sample.message}", err=True)
        if op.failure_samples.truncated:
            click.echo("  (failure samples truncated)", err=True)
    click.echo(f"State file: {record.state_path}")


__all__ = [
    "assertion_export_command",
    "blob_reference_debt_command",
    "blob_reference_prune_orphans_command",
    "blob_reference_recovery_plan_command",
    "blob_reference_replace_from_source_command",
    "gc_history_command",
    "maintenance_group",
    "plan_command",
    "preview_command",
    "run_command",
    "status_command",
]

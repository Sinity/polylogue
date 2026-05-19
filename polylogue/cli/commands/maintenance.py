"""Maintenance command group: preview and run backfills."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.logging import configure_logging
from polylogue.maintenance.envelope import envelope_from_operation
from polylogue.maintenance.planner import preview_backfill
from polylogue.maintenance.preview import ALL_SCOPES, staleness_inventory
from polylogue.maintenance.registry import MaintenanceOperationRegistry, OperationRecord
from polylogue.maintenance.replay import ReplayProgress, execute_replay
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.maintenance.targets import MAINTENANCE_TARGET_NAMES, build_maintenance_target_catalog
from polylogue.paths import archive_root, render_root
from polylogue.storage.blob_gc import read_gc_history


def _build_scope_filter(
    *,
    conversation_ids: tuple[str, ...],
    provider: str | None,
    source_family: str | None,
    source_root: str | None,
    raw_artifact_id: str | None,
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
        conversation_ids=conversation_ids if conversation_ids else None,
        provider=provider,
        source_family=source_family,
        source_root=Path(source_root) if source_root else None,
        raw_artifact_id=raw_artifact_id,
        time_range=time_range,
        failure_kind=failure_kind,
        parser_version=parser_version,
    )


_SCOPE_FILTER_OPTIONS = [
    click.option(
        "--conversation-id",
        "conversation_ids",
        multiple=True,
        help="Restrict scope to one or more conversation ids (repeatable).",
    ),
    click.option("--provider", "-p", type=str, default=None, help="Restrict scope to one provider name."),
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
    click.option(
        "--raw-artifact",
        "raw_artifact_id",
        type=str,
        default=None,
        help="Restrict scope to one raw artifact id.",
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
    conversation_ids: tuple[str, ...],
    provider: str | None,
    source_family: str | None,
    source_root: str | None,
    raw_artifact_id: str | None,
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
        conversation_ids=conversation_ids,
        provider=provider,
        source_family=source_family,
        source_root=source_root,
        raw_artifact_id=raw_artifact_id,
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
    conversation_ids: tuple[str, ...],
    provider: str | None,
    source_family: str | None,
    source_root: str | None,
    raw_artifact_id: str | None,
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
        click.echo(
            f"  [{snapshot.processed}/{snapshot.total}] {snapshot.target} "
            f"cursor={snapshot.cursor} failures={snapshot.in_flight_failures}",
            err=True,
        )

    scope_filter = _build_scope_filter(
        conversation_ids=conversation_ids,
        provider=provider,
        source_family=source_family,
        source_root=source_root,
        raw_artifact_id=raw_artifact_id,
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
    """Show recent blob-GC passes recorded in ``gc_generations.evidence``.

    Surfaces per-pass evidence (inspected/skipped/deleted, skip reasons,
    dry-run flag, batch bound) written by ``run_blob_gc`` so operators
    can audit GC behaviour over time without bespoke SQLite tooling.

    Pre-#1190 generations and crashed-mid-pass rows surface with
    ``evidence: null`` so operators can see they happened.
    """
    configure_logging()
    config = Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],
    )
    db_path = config.archive_root / "archive.db"
    history = read_gc_history(db_path, limit=limit)

    if output_format == "json":
        payload = [
            {
                "generation": row.generation,
                "completed_at": row.completed_at,
                "completed_at_iso": (
                    datetime.fromtimestamp(row.completed_at, tz=UTC).isoformat() if row.completed_at else None
                ),
                "evidence": asdict(row.evidence) if row.evidence else None,
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
            datetime.fromtimestamp(row.completed_at, tz=UTC).isoformat(timespec="seconds")
            if row.completed_at
            else "unknown"
        )
        click.echo(f"  generation={row.generation:>6}  completed_at={when}")
        ev = row.evidence
        if ev is None:
            click.echo("    (no evidence — pre-#1190 row or crashed pass)")
            continue
        dry = " [DRY-RUN]" if ev.dry_run else ""
        click.echo(
            f"    inspected={ev.inspected:>4}  deleted={ev.deleted:>4}{dry}  "
            f"skipped_ref={ev.skipped_referenced} skipped_leased={ev.skipped_leased} "
            f"skipped_missing={ev.skipped_missing} unlink_errors={ev.skipped_unlink_error}"
        )


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
    "gc_history_command",
    "maintenance_group",
    "plan_command",
    "preview_command",
    "run_command",
    "status_command",
]

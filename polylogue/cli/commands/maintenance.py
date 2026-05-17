"""Maintenance command group: preview and run backfills."""

from __future__ import annotations

import json

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.logging import configure_logging
from polylogue.maintenance.envelope import envelope_from_operation
from polylogue.maintenance.planner import preview_backfill
from polylogue.maintenance.preview import ALL_SCOPES, staleness_inventory
from polylogue.maintenance.replay import ReplayProgress, execute_replay
from polylogue.maintenance.targets import MAINTENANCE_TARGET_NAMES, build_maintenance_target_catalog
from polylogue.paths import archive_root, render_root

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
@click.pass_obj
def plan_command(env: AppEnv, targets: tuple[str, ...], output_format: str) -> None:
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
    result = preview_backfill(config, targets=targets)

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
@click.pass_obj
def run_command(
    env: AppEnv,
    targets: tuple[str, ...],
    dry_run: bool,
    operation_id: str | None,
    resume_cursor: str | None,
    output_format: str,
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

    result = execute_replay(
        config,
        targets=targets,
        operation_id=operation_id,
        resume_cursor=resume_cursor,
        dry_run=dry_run,
        progress_callback=_emit_progress,
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


__all__ = ["maintenance_group", "plan_command", "preview_command", "run_command"]

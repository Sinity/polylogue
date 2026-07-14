"""``maintenance run``: execute (or dry-run) maintenance backfill operations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.cli.commands.maintenance._shared import _apply_scope_filter_options, _build_scope_filter
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.logging import configure_logging
from polylogue.maintenance.targets import MAINTENANCE_TARGET_NAMES, build_maintenance_target_catalog
from polylogue.paths import archive_root, render_root

if TYPE_CHECKING:
    from polylogue.maintenance.replay import ReplayProgress

_MAINTENANCE_TARGET_HELP = build_maintenance_target_catalog().help_text()


@click.command("run")
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
    from polylogue.maintenance.envelope import envelope_from_operation
    from polylogue.maintenance.replay import execute_replay

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

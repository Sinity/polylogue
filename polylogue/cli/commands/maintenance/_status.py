"""``maintenance status``: inspect persisted maintenance operations (#1197)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.logging import configure_logging
from polylogue.paths import archive_root, render_root

if TYPE_CHECKING:
    from polylogue.maintenance.registry import OperationRecord


@click.command("status")
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
    from polylogue.maintenance.envelope import envelope_from_operation
    from polylogue.maintenance.registry import MaintenanceOperationRegistry

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


__all__ = ["status_command"]

"""Bounded inspection and authorized adjudication for interrupted operations."""

from __future__ import annotations

import json
from typing import Literal, cast

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.operations.audit import AuditRepository


def _outcomes(values: tuple[str, ...]) -> dict[str, Literal["applied", "not-applied", "unknown"]]:
    result: dict[str, Literal["applied", "not-applied", "unknown"]] = {}
    for value in values:
        # rsplit (not partition/split) from the LAST "=": target refs can
        # themselves contain "=" (e.g. "session:native=id"), so splitting at
        # the first "=" would mis-split the ref and corrupt it. The outcome
        # vocabulary (applied/not-applied/unknown) never contains "=", so the
        # last occurrence unambiguously separates ref from outcome
        # (polylogue-39pdi).
        target, separator, outcome = value.rpartition("=")
        if not separator or outcome not in {"applied", "not-applied", "unknown"} or not target or target in result:
            raise click.ClickException("--target-outcome must be unique target_ref=applied|not-applied|unknown")
        result[target] = cast(Literal["applied", "not-applied", "unknown"], outcome)
    return result


@click.command("operation-recovery")
@click.option("--operation-id", required=False, help="Interrupted operation id to inspect or adjudicate.")
@click.option("--list", "list_operations", is_flag=True, help="List all interrupted operations and target states.")
@click.option("--target-outcome", "target_outcomes", multiple=True, help="target_ref=applied|not-applied|unknown")
@click.option("--reason", default=None, help="Operator evidence supporting an adjudication.")
@click.option("--confirm", is_flag=True, help="Authorize the bounded per-target adjudication.")
@click.option(
    "--adjudicator",
    default="user:local",
    show_default=True,
    help="Actor recorded on the recovery_adjudicated audit event.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
@click.pass_obj
def operation_recovery_command(
    env: AppEnv,
    operation_id: str | None,
    list_operations: bool,
    target_outcomes: tuple[str, ...],
    reason: str | None,
    confirm: bool,
    adjudicator: str,
    output_format: str,
) -> None:
    """Inspect recovery evidence, or adjudicate at most 256 durable targets."""

    audit = AuditRepository.for_archive_root(env.config.archive_root)
    if list_operations:
        if operation_id or target_outcomes or reason or confirm:
            raise click.ClickException("--list cannot be combined with an operation id or adjudication options")
        operations = audit.list_recovery_operations()
        if output_format == "json":
            click.echo(json.dumps({"operations": list(operations)}, sort_keys=True, default=str))
        else:
            click.echo(f"Interrupted operations: {len(operations)}")
            for item in operations:
                listed_operation = cast(dict[str, object], item["operation"])
                click.echo(f"Operation recovery: {listed_operation['status']}")
                click.echo(f"Operation: {listed_operation['operation_id']}")
                targets = cast(tuple[dict[str, object], ...], item["targets"])
                click.echo(f"Targets: {len(targets)}")
                for target in targets:
                    click.echo(f"  {target['target_ref']}={target['state']}")
        return
    if operation_id is None:
        raise click.ClickException("either --operation-id or --list is required")
    outcomes = _outcomes(target_outcomes)
    if confirm or target_outcomes:
        # ``--confirm`` alone still adjudicates: a run whose plan resolved to
        # zero durable targets has no outcome to name, and must remain
        # closable.  The audit repository refuses any partial target set.
        if not confirm or not reason:
            raise click.ClickException("adjudication requires --confirm and --reason")
        # The CLI is an offline surface with no writer lease of its own.
        # Refuse rather than become a second writer beside a live daemon.
        if block_reason := offline_maintenance_block_reason(env.config, active=True, dry_run=False):
            raise click.ClickException(block_reason)
        try:
            audit.adjudicate_recovery(operation_id, target_outcomes=outcomes, reason=reason, adjudicator=adjudicator)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    operation = audit.get_operation(operation_id)
    if operation is None:
        raise click.ClickException(f"operation not found: {operation_id}")
    payload = {
        "operation": operation,
        "events": audit.list_events(operation_id),
        "targets": audit.list_targets(operation_id),
    }
    if output_format == "json":
        click.echo(json.dumps(payload, sort_keys=True, default=str))
    else:
        click.echo(f"Operation recovery: {operation['status']}")
        click.echo(f"Operation: {operation_id}")
        click.echo(f"Events: {len(payload['events'])}")
        targets = cast(tuple[dict[str, object], ...], payload["targets"])
        click.echo(f"Targets: {len(targets)}")
        for target in targets:
            click.echo(f"  {target['target_ref']}={target['state']}")


__all__ = ["operation_recovery_command"]

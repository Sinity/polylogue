"""Bounded inspection and authorized adjudication for interrupted operations."""

from __future__ import annotations

import json
from typing import Literal, cast

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.operations.audit import AuditRepository


def _outcomes(values: tuple[str, ...]) -> dict[str, Literal["applied", "not-applied", "unknown"]]:
    result: dict[str, Literal["applied", "not-applied", "unknown"]] = {}
    for value in values:
        target, separator, outcome = value.partition("=")
        if not separator or outcome not in {"applied", "not-applied", "unknown"} or not target or target in result:
            raise click.ClickException("--target-outcome must be unique target_ref=applied|not-applied|unknown")
        result[target] = cast(Literal["applied", "not-applied", "unknown"], outcome)
    return result


@click.command("operation-recovery")
@click.option("--operation-id", required=True, help="Interrupted operation id to inspect or adjudicate.")
@click.option("--target-outcome", "target_outcomes", multiple=True, help="target_ref=applied|not-applied|unknown")
@click.option("--reason", default=None, help="Operator evidence supporting an adjudication.")
@click.option("--confirm", is_flag=True, help="Authorize the bounded per-target adjudication.")
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
@click.pass_obj
def operation_recovery_command(
    env: AppEnv,
    operation_id: str,
    target_outcomes: tuple[str, ...],
    reason: str | None,
    confirm: bool,
    output_format: str,
) -> None:
    """Inspect recovery evidence, or adjudicate at most 256 durable targets."""

    audit = AuditRepository.for_archive_root(env.config.archive_root)
    outcomes = _outcomes(target_outcomes)
    if outcomes:
        if not confirm or not reason:
            raise click.ClickException("adjudication requires --confirm and --reason")
        audit.adjudicate_recovery(operation_id, target_outcomes=outcomes, reason=reason)
    operation = audit.get_operation(operation_id)
    if operation is None:
        raise click.ClickException(f"operation not found: {operation_id}")
    payload = {"operation": operation, "events": audit.list_events(operation_id)}
    if output_format == "json":
        click.echo(json.dumps(payload, sort_keys=True, default=str))
    else:
        click.echo(f"Operation recovery: {operation['status']}")
        click.echo(f"Operation: {operation_id}")
        click.echo(f"Events: {len(payload['events'])}")


__all__ = ["operation_recovery_command"]

"""CLI adapter for the typed raw-authority recovery family."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.maintenance.raw_authority_recovery import (
    RawAuthorityRecoveryError,
    RawAuthorityRecoveryPlan,
    RecoveryOperation,
    apply_raw_authority_recovery,
    inspect_raw_authority_recovery,
    write_recovery_plan,
)


@click.command("raw-authority-recovery")
@click.option(
    "--operation",
    type=click.Choice([operation.value for operation in RecoveryOperation]),
    required=True,
    help="One exact recovery actuator to inspect or apply.",
)
@click.option("--apply", "apply_changes", is_flag=True, help="Apply the exact previously inspected plan.")
@click.option(
    "--plan-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Plan artifact to write during inspect or consume during --apply.",
)
@click.option(
    "--backup-manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Verified source/index backup authority, required for apply.",
)
@click.option(
    "--receipt-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Immutable receipt destination under the archive-owned maintenance-state directory.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
@click.pass_obj
def raw_authority_recovery_command(
    env: AppEnv,
    operation: str,
    apply_changes: bool,
    plan_file: Path | None,
    backup_manifest: Path | None,
    receipt_file: Path | None,
    output_format: str,
) -> None:
    """Inspect raw-authority recovery, or apply one exact guarded plan."""
    plan_obj: RawAuthorityRecoveryPlan | None = None
    try:
        selected = RecoveryOperation(operation)
        if apply_changes:
            if plan_file is None:
                raise click.ClickException("--apply requires --plan-file from a prior inspect")
            report = apply_raw_authority_recovery(plan_file, backup_manifest=backup_manifest)
            payload = report.to_dict()
        else:
            plan_obj = inspect_raw_authority_recovery(
                env.config.archive_root,
                selected,
                backup_manifest=backup_manifest,
                receipt_path=receipt_file,
            )
            if plan_file is not None:
                write_recovery_plan(plan_obj, plan_file)
            payload = {"mode": "dry-run", **plan_obj.to_dict()}
    except (RawAuthorityRecoveryError, FileNotFoundError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if apply_changes:
        click.echo(f"Raw-authority recovery: {payload.get('status', 'unknown')}")
        click.echo(f"Operation: {payload.get('operation_id', 'unknown')}")
        if payload.get("receipt_path"):
            click.echo(f"Receipt: {payload['receipt_path']}")
        return
    assert plan_obj is not None
    click.echo("Raw-authority recovery dry-run")
    click.echo(f"Operation: {plan_obj.operation_id}")
    click.echo(f"Plan digest: {plan_obj.plan_digest}")
    click.echo(f"Before: {json.dumps(plan_obj.before_counts, sort_keys=True)}")
    click.echo(f"Plan file: {plan_file if plan_file is not None else 'stdout only'}")


__all__ = ["raw_authority_recovery_command"]

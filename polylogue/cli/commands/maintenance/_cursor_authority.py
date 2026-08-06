"""``maintenance cursor-authority-reconcile`` command."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("cursor-authority-reconcile")
@click.option("--source-path-file", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--output-plan", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--plan", "plan_path", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--backup-manifest", type=click.Path(path_type=Path, file_okay=True, dir_okay=True), default=None)
@click.option("--receipt", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--apply", "apply_changes", is_flag=True, help="Apply one previously written reconciliation plan.")
def cursor_authority_reconcile_command(
    source_path_file: Path | None,
    output_plan: Path | None,
    plan_path: Path | None,
    backup_manifest: Path | None,
    receipt: Path | None,
    apply_changes: bool,
) -> None:
    """Plan or apply one backup-gated cursor-authority reconciliation."""

    from polylogue.maintenance.cursor_authority_reconcile import (
        CursorAuthorityReconciliationError,
        apply_reconciliation,
        build_reconciliation_plan,
    )

    try:
        if apply_changes:
            if plan_path is None or backup_manifest is None or receipt is None:
                raise click.UsageError("--apply requires --plan, --backup-manifest, and --receipt")
            if source_path_file is not None or output_plan is not None:
                raise click.UsageError("--apply does not accept --source-path-file or --output-plan")
            result = apply_reconciliation(plan_path=plan_path, backup_manifest=backup_manifest, receipt=receipt)
        else:
            if source_path_file is None or output_plan is None:
                raise click.UsageError("dry-run requires --source-path-file and --output-plan")
            if plan_path is not None or backup_manifest is not None or receipt is not None:
                raise click.UsageError("dry-run accepts only --source-path-file and --output-plan")
            result = build_reconciliation_plan(source_path_file=source_path_file, output_plan=output_plan)
    except CursorAuthorityReconciliationError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2, sort_keys=True))


__all__ = ["cursor_authority_reconcile_command"]

"""CLI adapter for the offline message-owner scope compatibility pass."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.paths import archive_root


@click.command("message-owner-scope-backfill")
@click.option("--apply", "apply_changes", is_flag=True, help="Apply exact owners; default is a read-only census.")
@click.option(
    "--output-plan",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the immutable census plan here (census mode only).",
)
@click.option(
    "--plan-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Immutable census plan consumed by --apply.",
)
@click.option(
    "--backup-manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Verified user-tier backup manifest required with --apply.",
)
@click.option(
    "--receipt-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="New immutable receipt path required with --apply.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def message_owner_scope_backfill_command(
    apply_changes: bool,
    output_plan: Path | None,
    plan_file: Path | None,
    backup_manifest: Path | None,
    receipt_file: Path | None,
    output_format: str,
) -> None:
    """Backfill exact message owners before replacing the rebuildable index."""
    from polylogue.maintenance.message_owner_scope_backfill import (
        MessageOwnerScopeBackfillError,
        apply_message_owner_scope_backfill,
        census_message_owner_scope_backfill,
        write_message_owner_scope_backfill_plan,
    )

    if apply_changes:
        if output_plan is not None:
            raise click.UsageError("--output-plan is census-only")
        if plan_file is None or backup_manifest is None or receipt_file is None:
            raise click.UsageError("--apply requires --plan-file, --backup-manifest, and --receipt-file")
        try:
            report = apply_message_owner_scope_backfill(
                archive_root(),
                plan_path=plan_file,
                backup_manifest=backup_manifest,
                receipt_path=receipt_file,
                dry_run=False,
            )
        except MessageOwnerScopeBackfillError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        if plan_file is not None or backup_manifest is not None or receipt_file is not None:
            raise click.UsageError("--plan-file, --backup-manifest, and --receipt-file require --apply")
        try:
            plan = census_message_owner_scope_backfill(archive_root())
            if output_plan is not None:
                write_message_owner_scope_backfill_plan(plan, output_plan)
            payload: dict[str, object] = {"mode": "census", "plan": plan.to_dict(), "counts": plan.counts}
        except MessageOwnerScopeBackfillError as exc:
            raise click.ClickException(str(exc)) from exc
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        click.echo("Message-owner scope backfill census")
        click.echo(f"Exact-resolvable: {plan.counts['exact-resolvable']:,}")
        click.echo(f"Already scoped:   {plan.counts['already-scoped']:,}")
        click.echo(f"Missing owner:    {plan.counts['missing-index-owner']:,}")
        click.echo(f"Malformed scope:  {plan.counts['malformed-scope']:,}")
        click.echo(f"Conflicting scope:{plan.counts['conflicting-scope']:,}")
        click.echo(f"Plan digest:      {plan.plan_digest}")
        if output_plan is not None:
            click.echo(f"Plan:             {output_plan}")
        return

    payload = report.to_dict()
    blocked = report.after_plan is not None and report.after_plan.unresolved_denominator != 0
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo("Message-owner scope backfill apply")
        click.echo(f"Updated:          {report.updated_count:,}")
        click.echo(f"Terminal state:   {report.terminal_state}")
        click.echo(f"Unresolved:       {report.after_plan.unresolved_denominator if report.after_plan else 0:,}")
        click.echo(f"Receipt:          {receipt_file}")
    if blocked:
        if output_format == "json":
            raise click.exceptions.Exit(1)
        raise click.ClickException("message-owner scope backfill remains blocked by unresolved owners")


__all__ = ["message_owner_scope_backfill_command"]

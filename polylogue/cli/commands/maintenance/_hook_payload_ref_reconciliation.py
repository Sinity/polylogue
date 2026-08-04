"""CLI adapter for the guarded hook-payload reference reconciler."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.paths import archive_root


@click.command("hook-payload-ref-reconcile")
@click.option(
    "--apply", "apply_changes", is_flag=True, help="Apply only exact hook-payload matches; default is read-only."
)
@click.option(
    "--backup-manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Verified source-tier backup manifest required with --apply.",
)
@click.option(
    "--receipt-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="New immutable JSONL receipt path required with --apply.",
)
@click.option(
    "--output-format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
def hook_payload_ref_reconcile_command(
    apply_changes: bool,
    backup_manifest: Path | None,
    receipt_file: Path | None,
    output_format: str,
) -> None:
    """Classify historical hook refs, or reconcile exact matches with safeguards."""
    from polylogue.maintenance.hook_payload_ref_reconciliation_apply import apply_hook_payload_ref_reconciliation

    report = apply_hook_payload_ref_reconciliation(
        archive_root(),
        backup_manifest=backup_manifest,
        receipt_path=receipt_file,
        dry_run=not apply_changes,
    )
    payload = {"mode": "hook_payload_ref_reconciliation", "mutates": report.applied, **report.to_dict()}
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo("Hook-payload reference reconciliation")
    click.echo(f"Mode:       {'apply' if report.applied else 'dry-run'}")
    click.echo(f"Scanned:    {report.scanned_count:,} orphaned raw_payload ref(s)")
    click.echo(f"Exact:      {report.matched_count:,} ref(s), {report.matched_bytes:,} bytes")
    click.echo(f"Unmatched:  {report.unmatched_count:,} ref(s) left untouched")
    click.echo(f"Reconciled: {len(report.reconciled_hook_event_ids):,} hook event(s)")
    if report.receipt_path is not None:
        click.echo(f"Receipt:    {report.receipt_path}")


__all__ = ["hook_payload_ref_reconcile_command"]

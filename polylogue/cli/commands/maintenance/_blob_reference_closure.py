"""CLI adapter for acquired blob-reference closure repair."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.paths import archive_root


@click.command("blob-reference-closure")
@click.option("--apply", "apply_changes", is_flag=True, help="Apply only deterministic exact reference repairs.")
@click.option(
    "--backup-manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Verified manifest covering source.db and index.db; required with --apply.",
)
@click.option(
    "--receipt-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="New immutable receipt path; required with --apply.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def blob_reference_closure_command(
    apply_changes: bool,
    backup_manifest: Path | None,
    receipt_file: Path | None,
    output_format: str,
) -> None:
    """Audit closure, or repair exact refs from existing source evidence."""
    from polylogue.maintenance.blob_reference_closure import (
        BlobReferenceClosureError,
        reconcile_blob_reference_closure,
    )

    try:
        report = reconcile_blob_reference_closure(
            archive_root(),
            backup_manifest=backup_manifest,
            receipt_path=receipt_file,
            dry_run=not apply_changes,
        )
    except BlobReferenceClosureError as exc:
        raise click.ClickException(str(exc)) from exc

    payload = {"mode": "blob_reference_closure", **report.to_dict()}
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo("Blob-reference closure")
    click.echo(f"Mode:        {'apply' if report.applied else 'dry-run'}")
    click.echo(f"Raw repair:  {report.raw_repaired_count:,}")
    click.echo(f"Attachment:  {report.attachment_repaired_count:,}")
    click.echo(f"Blockers:    {len(report.plan.blockers):,}")
    for blocker in report.plan.blockers:
        click.echo(f"  blocker [{blocker.kind.value}] {blocker.object_id}: {blocker.detail}")


__all__ = ["blob_reference_closure_command"]

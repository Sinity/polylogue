"""CLI adapter for the offline blob namespace quarantine actuator."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.paths import archive_root


@click.command("blob-namespace-quarantine")
@click.option(
    "--apply", "apply_changes", is_flag=True, help="Move the proven invalid namespace entries into quarantine."
)
@click.option(
    "--backup-manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Verified, attested source-tier backup manifest required with --apply.",
)
@click.option(
    "--receipt-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="New directory for immutable before.json and after.json receipts; required with --apply or --recover.",
)
@click.option(
    "--recover", is_flag=True, help="Classify an interrupted operation from its receipts without moving anything."
)
@click.option(
    "--output-format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
def blob_namespace_quarantine_command(
    apply_changes: bool,
    backup_manifest: Path | None,
    receipt_dir: Path | None,
    recover: bool,
    output_format: str,
) -> None:
    """Census invalid blob namespace entries; apply only as offline quarantine.

    The default is a complete, read-only census. ``--apply`` requires the
    daemon stopped, the archive writer lease to be clear, a verified attested
    source backup, a clean WAL checkpoint, and a fresh explicit receipt
    directory. It only uses atomic same-filesystem moves and never deletes,
    garbage-collects, changes SQLite rows, or moves canonical blobs.
    """

    from polylogue.operations.blob_namespace_quarantine import (
        BlobNamespaceQuarantineError,
        classify_blob_namespace_quarantine_recovery,
        quarantine_blob_namespace,
    )

    if recover:
        if apply_changes or backup_manifest is not None:
            raise click.UsageError("--recover cannot be combined with --apply or --backup-manifest")
        if receipt_dir is None:
            raise click.UsageError("--recover requires --receipt-dir")
        recovery_report = classify_blob_namespace_quarantine_recovery(receipt_dir)
        payload = {"mode": "blob_namespace_quarantine_recovery", **recovery_report.to_dict()}
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            click.echo(f"Blob namespace quarantine recovery: {recovery_report.outcome}")
            click.echo(f"Receipt directory: {recovery_report.receipt_dir}")
            click.echo(
                f"Sources present: {recovery_report.source_present}; destinations present: "
                f"{recovery_report.destination_present}; matching destinations: "
                f"{recovery_report.matching_destinations}"
            )
            for conflict in recovery_report.conflicts:
                click.echo(f"  conflict: {conflict}")
        if recovery_report.outcome == "indeterminate":
            raise SystemExit(1)
        return

    try:
        quarantine_report = quarantine_blob_namespace(
            archive_root(),
            backup_manifest=backup_manifest,
            receipt_dir=receipt_dir,
            dry_run=not apply_changes,
        )
    except BlobNamespaceQuarantineError as exc:
        raise click.ClickException(str(exc)) from exc
    payload = {"mode": "blob_namespace_quarantine", **quarantine_report.to_dict()}
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo("Blob namespace quarantine")
    click.echo(f"Mode:       {'apply' if quarantine_report.applied else 'dry-run'}")
    click.echo(f"Blob root:  {quarantine_report.blob_root}")
    click.echo(f"Canonical:  {len(quarantine_report.census.canonical):,}")
    click.echo(f"Candidates: {len(quarantine_report.census.candidates):,}")
    click.echo(f"Blockers:   {len(quarantine_report.census.blockers):,}")
    if quarantine_report.applied:
        click.echo(f"Moved:      {quarantine_report.moved_count:,}")
        assert quarantine_report.receipt_dir is not None
        assert quarantine_report.quarantine_root is not None
        click.echo(f"Receipts:   {quarantine_report.receipt_dir}")
        click.echo(f"Quarantine: {quarantine_report.quarantine_root}")
    for blocker in quarantine_report.census.blockers:
        click.echo(f"  blocker: {blocker}")


__all__ = ["blob_namespace_quarantine_command"]

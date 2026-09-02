"""Read-only blob conservation command."""

from __future__ import annotations

import json

import click

from polylogue.paths import archive_root


@click.command("blob-conservation")
@click.option("--sample-size", type=int, default=20, show_default=True)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def blob_conservation_command(sample_size: int, output_format: str) -> None:
    """Verify both directions of blob/reference conservation without mutation."""
    from polylogue.maintenance.blob_conservation import check_blob_conservation

    report = check_blob_conservation(archive_root(), sample_size=sample_size)
    if output_format == "json":
        click.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        click.echo(f"Blob conservation: {'PASS' if report.ok else 'FAIL'}")
        click.echo(f"Referenced: {report.referenced_blobs:,}; present: {report.present_blobs:,}")
        click.echo(
            f"Orphans: {report.orphan_blobs:,}; dangling: {report.dangling_references:,}; corrupt: {report.corrupt_blobs:,}"
        )
        click.echo(f"Recoverable: {report.recoverable_references:,}; reserved: {report.reserved_blobs:,}")
        click.echo(
            f"Invalid namespace: {report.invalid_namespace_entries:,}; staged in-flight: {report.staged_in_flight:,}"
        )
    if not report.ok:
        raise click.exceptions.Exit(1)


__all__ = ["blob_conservation_command"]

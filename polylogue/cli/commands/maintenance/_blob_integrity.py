"""Blob-reference debt classification (read-only)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.paths import archive_root

if TYPE_CHECKING:
    from polylogue.storage.blob_integrity import BlobReferenceDebtClassificationReport


@click.command("blob-reference-debt")
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative missing-blob samples to include.",
)
@click.option(
    "--group-limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of grouped classifications to include.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_reference_debt_command(sample_limit: int, group_limit: int, output_format: str) -> None:
    """Classify missing referenced blobs without mutating the archive."""
    from polylogue.storage.blob_integrity import classify_blob_reference_debt

    report = classify_blob_reference_debt(
        archive_root() / "source.db",
        sample_size=sample_limit,
        group_limit=group_limit,
    )
    payload = {
        "mode": "blob_reference_debt",
        "mutates": False,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_blob_reference_debt_plain(report)


def _render_blob_reference_debt_plain(report: BlobReferenceDebtClassificationReport) -> None:
    click.echo("Blob reference debt")
    click.echo(f"Source DB:    {report.source_db}")
    click.echo(f"Blob root:    {report.blob_root}")
    click.echo(f"References:   {report.reference_rows:,} row(s), {report.distinct_referenced_blobs:,} distinct blob(s)")
    click.echo(f"Missing:      {report.missing_distinct_blobs:,} distinct blob(s)")
    click.echo(f"Status:       {'ok' if report.ok else 'debt-present'}")

    def _render_counts(label: str, counts: dict[str, int]) -> None:
        if not counts:
            return
        rendered = ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items()))
        click.echo(f"{label}: {rendered}")

    _render_counts("By table    ", report.missing_by_table)
    _render_counts("By ref type ", report.missing_by_ref_type)
    _render_counts("By origin   ", report.missing_by_origin)
    _render_counts("Ref-id join ", report.missing_ref_id_join)
    _render_counts("Source paths", report.missing_source_path_presence)
    _render_counts("Validation  ", report.missing_validation_status)
    _render_counts("Parse errors", report.missing_parse_error)

    if report.top_groups:
        click.echo("Top groups:")
        for group in report.top_groups:
            tables_value = group.get("tables", ())
            ref_types_value = group.get("ref_types", ())
            origins_value = group.get("origins", ())
            count_value = group.get("count", 0)
            tables = ",".join(str(item) for item in tables_value) if isinstance(tables_value, list | tuple) else ""
            ref_types = (
                ",".join(str(item) for item in ref_types_value) if isinstance(ref_types_value, list | tuple) else ""
            )
            origins = ",".join(str(item) for item in origins_value) if isinstance(origins_value, list | tuple) else ""
            count = count_value if isinstance(count_value, int) else 0
            click.echo(f"  {count:>8,}  tables={tables} ref_types={ref_types} origins={origins}")

    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            source = sample.sample_source_path or "(none)"
            origin = ",".join(sample.origins) if sample.origins else "(none)"
            click.echo(
                f"  {sample.blob_hash} origin={origin} source_available={sample.sample_source_available} {source}"
            )

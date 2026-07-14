"""Blob-reference debt classification, recovery planning, replace, and prune."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click

from polylogue.paths import archive_root

if TYPE_CHECKING:
    from polylogue.storage.blob_integrity import (
        BlobReferenceDebtClassificationReport,
        BlobReferenceOrphanPruneReport,
        BlobReferenceRecoveryPlanReport,
        BlobReferenceSourceReplaceReport,
    )


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


@click.command("attachment-acquisition-debt")
@click.option(
    "--sample-limit",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of representative acquired-but-missing attachment ids to include.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def attachment_acquisition_debt_command(sample_limit: int, output_format: str) -> None:
    """Classify index-tier attachment acquisition state without mutating the archive.

    Deliberately separate from ``blob-reference-debt``: unfetched attachments
    (``blob_hash IS NULL``) are an honest floor, never counted as missing
    referenced blobs. Only an acquired attachment whose blob file is absent
    from the store is genuine attachment acquisition debt.
    """
    from polylogue.storage.blob_integrity import scan_attachment_acquisition_debt

    report = scan_attachment_acquisition_debt(
        archive_root() / "index.db",
        sample_size=sample_limit,
    )
    payload = {
        "mode": "attachment_acquisition_debt",
        "mutates": False,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("Attachment acquisition debt")
    click.echo(f"Total attachments: {report.total_attachments:,}")
    click.echo(f"Acquired:          {report.acquired_count:,}")
    click.echo(f"Unavailable:       {report.unavailable_count:,}")
    click.echo(f"Unfetched:         {report.unfetched_count:,} (honest floor, not missing blobs)")
    click.echo(f"Acquired missing:  {report.acquired_missing_blob_count:,} (genuine debt)")
    click.echo(f"Status:            {'ok' if report.ok else 'debt-present'}")
    if report.acquired_missing_blob_sample:
        click.echo("Sample attachment ids with a missing blob file:")
        for attachment_id in report.acquired_missing_blob_sample:
            click.echo(f"  {attachment_id}")


@click.command("blob-reference-recovery-plan")
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative raw-backed missing blob rows to include.",
)
@click.option(
    "--manifest-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSONL destination for the complete raw-backed missing blob recovery manifest.",
)
@click.option(
    "--include-rows",
    is_flag=True,
    help="Include every plan row in JSON output. By default JSON output includes samples plus aggregate counts.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_reference_recovery_plan_command(
    sample_limit: int,
    manifest_file: Path | None,
    include_rows: bool,
    output_format: str,
) -> None:
    """Plan recovery for raw-backed missing blobs without mutating archive state."""
    from polylogue.storage.blob_integrity import plan_raw_backed_blob_reference_recovery

    report = plan_raw_backed_blob_reference_recovery(
        archive_root() / "source.db",
        manifest_path=manifest_file,
        sample_size=sample_limit,
        include_rows=include_rows,
    )
    payload = {
        "mode": "blob_reference_recovery_plan",
        "mutates": False,
        "writes_manifest": manifest_file is not None,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_blob_reference_recovery_plan_plain(report)


def _render_blob_reference_recovery_plan_plain(report: BlobReferenceRecoveryPlanReport) -> None:
    click.echo("Blob reference raw-backed recovery plan")
    click.echo(f"Source DB:    {report.source_db}")
    click.echo(f"Blob root:    {report.blob_root}")
    click.echo(f"Missing:      {report.missing_raw_backed_blobs:,} raw-backed blob(s)")

    def _render_counts(label: str, counts: dict[str, int]) -> None:
        if not counts:
            return
        rendered = ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items()))
        click.echo(f"{label}: {rendered}")

    _render_counts("By action   ", report.by_action)
    _render_counts("By origin   ", report.by_origin)
    _render_counts("By shape    ", report.by_source_shape)
    if report.manifest_path:
        click.echo(f"Manifest:    {report.manifest_path}")
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            source = sample.source_path or "(none)"
            click.echo(f"  {sample.action} {sample.blob_hash} origin={sample.origin} {source}")


@click.command("blob-reference-replace-from-source")
@click.option(
    "--yes",
    "apply",
    is_flag=True,
    help="Apply the replacement. Without this flag the command is a dry run.",
)
@click.option(
    "--manifest-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="JSONL destination for before/after replacement rows. Required with --yes.",
)
@click.option("--max-count", type=int, default=None, help="Maximum number of candidate rows to process.")
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative replacement rows to include.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_reference_replace_from_source_command(
    apply: bool,
    manifest_file: Path | None,
    max_count: int | None,
    sample_limit: int,
    output_format: str,
) -> None:
    """Replace raw-backed missing blob refs with current source-derived bytes."""
    from polylogue.storage.blob_integrity import replace_raw_backed_blob_reference_debt_from_source

    if apply and manifest_file is None:
        raise click.UsageError("--manifest-file is required with --yes")
    report = replace_raw_backed_blob_reference_debt_from_source(
        archive_root() / "source.db",
        dry_run=not apply,
        manifest_path=manifest_file,
        max_count=max_count,
        sample_size=sample_limit,
    )
    payload = {
        "mode": "blob_reference_replace_from_source",
        "mutates": apply,
        "writes_manifest": manifest_file is not None,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_blob_reference_replace_from_source_plain(report)


def _render_blob_reference_replace_from_source_plain(report: BlobReferenceSourceReplaceReport) -> None:
    click.echo("Blob reference current-source replacement")
    click.echo(f"Source DB:    {report.source_db}")
    click.echo(f"Blob root:    {report.blob_root}")
    click.echo(f"Mode:         {'dry-run' if report.dry_run else 'apply'}")
    click.echo(f"Scanned:      {report.scanned_rows:,} raw-backed row(s)")
    click.echo(f"Candidates:   {report.candidate_rows:,}")
    click.echo(f"Replaced:     {report.replaced_rows:,}")
    click.echo(f"Written:      {report.written_blobs:,} blob(s), {report.written_bytes:,} byte(s)")
    click.echo(
        "Skipped:      "
        f"existing={report.skipped_existing_blob:,} "
        f"no_source={report.skipped_no_source_path:,} "
        f"source_missing={report.skipped_source_missing:,} "
        f"source_index={report.skipped_source_index:,} "
        f"unsupported={report.skipped_unsupported_source:,} "
        f"error={report.skipped_error:,}"
    )
    if report.manifest_path:
        click.echo(f"Manifest:    {report.manifest_path}")
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            detail = f" reason={sample.reason}" if sample.reason else ""
            click.echo(
                f"  {sample.action} raw_id={sample.raw_id} old={sample.old_blob_hash} "
                f"new={sample.new_blob_hash}{detail}"
            )


@click.command("blob-reference-prune-orphans")
@click.option(
    "--max-count",
    type=int,
    default=None,
    help="Maximum number of orphan blob-reference rows to prune or preview.",
)
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative samples to include.",
)
@click.option(
    "--quarantine-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "JSONL destination for rows removed by --yes. Defaults to "
        "<archive-root>/.maintenance-state/blob-ref-quarantine/<timestamp>.jsonl."
    ),
)
@click.option(
    "--yes",
    is_flag=True,
    help="Delete orphan blob_refs after writing them to the quarantine JSONL.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_reference_prune_orphans_command(
    max_count: int | None,
    sample_limit: int,
    quarantine_file: Path | None,
    yes: bool,
    output_format: str,
) -> None:
    """Quarantine and prune missing blob_refs that no longer have raw rows."""
    from polylogue.storage.blob_integrity import prune_orphan_blob_reference_debt

    report = prune_orphan_blob_reference_debt(
        archive_root() / "source.db",
        dry_run=not yes,
        quarantine_path=quarantine_file,
        max_count=max_count,
        sample_size=sample_limit,
    )
    payload = {
        "mode": "blob_reference_prune_orphans",
        "mutates": bool(yes),
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_blob_reference_prune_orphans_plain(report)


def _render_blob_reference_prune_orphans_plain(report: BlobReferenceOrphanPruneReport) -> None:
    click.echo("Blob reference orphan prune")
    click.echo(f"Source DB:    {report.source_db}")
    click.echo(f"Blob root:    {report.blob_root}")
    click.echo(f"Mode:         {'dry-run' if report.dry_run else 'apply'}")
    click.echo(f"Blob refs:    {report.scanned_blob_refs:,} scanned")
    click.echo(
        f"Orphans:      {report.missing_orphan_refs:,} row(s), "
        f"{report.missing_orphan_distinct_blobs:,} distinct blob(s)"
    )
    action = "would prune" if report.dry_run else "pruned"
    click.echo(
        f"Result:       {action} {report.missing_orphan_refs if report.dry_run else report.pruned_refs:,} row(s)"
    )
    click.echo(
        "Skipped:      "
        f"existing_blob={report.skipped_existing_blob:,} "
        f"raw_session_present={report.skipped_raw_session_present:,}"
    )
    if report.quarantine_path:
        click.echo(f"Quarantine:   {report.quarantine_path}")
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            source = sample.source_path or "(none)"
            click.echo(f"  {sample.action} {sample.blob_hash} ref_id={sample.ref_id} {source}")

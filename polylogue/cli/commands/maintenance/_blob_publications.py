"""``maintenance blob-publications``: inspect/abandon publication receipts."""

from __future__ import annotations

import json

import click

from polylogue.paths import archive_root
from polylogue.storage.blob_publication import abandon_blob_publication_receipts, inspect_blob_publication_receipts


@click.command("blob-publications")
@click.option(
    "--abandon",
    "publication_ids",
    multiple=True,
    help="Publication receipt ID to abandon. Repeat for multiple receipts.",
)
@click.option("--yes", is_flag=True, help="Confirm abandonment of the selected unreferenced receipts.")
@click.option(
    "--output-format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
def blob_publications_command(publication_ids: tuple[str, ...], yes: bool, output_format: str) -> None:
    """Inspect publication receipts or explicitly abandon selected debt."""
    root = archive_root()
    source_db = root / "source.db"
    if publication_ids and not yes:
        raise click.UsageError("--yes is required with --abandon")
    abandonment = None
    if publication_ids:
        abandonment = abandon_blob_publication_receipts(
            source_db,
            root / "blob",
            publication_ids,
            confirmed=True,
            index_db_path=root / "index.db",
        )
    receipts = inspect_blob_publication_receipts(
        source_db,
        root / "blob",
        index_db_path=root / "index.db",
    )
    payload = {
        "mode": "blob_publications",
        "mutates": bool(publication_ids),
        "abandonment": (
            {
                "abandoned": abandonment.abandoned,
                "skipped_referenced": abandonment.skipped_referenced,
                "missing_receipts": abandonment.missing_receipts,
            }
            if abandonment is not None
            else None
        ),
        "receipts": [
            {
                "publication_id": item.publication_id,
                "blob_hash": item.blob_hash,
                "size_bytes": item.size_bytes,
                "publisher_id": item.publisher_id,
                "reserved_at_ms": item.reserved_at_ms,
                "blob_present": item.blob_present,
                "referenced": item.referenced,
            }
            for item in receipts
        ],
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if abandonment is not None:
        click.echo(
            "Abandoned: "
            f"{abandonment.abandoned} receipt(s); "
            f"referenced={abandonment.skipped_referenced}, missing={abandonment.missing_receipts}"
        )
    click.echo(f"Publication receipts: {len(receipts)}")
    for item in receipts:
        state = "referenced" if item.referenced else "present" if item.blob_present else "missing"
        click.echo(f"  {item.publication_id} {item.blob_hash} {item.size_bytes}B {state}")

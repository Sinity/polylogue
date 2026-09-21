"""``maintenance blob-publications``: inspect/abandon publication receipts.

Inspection is an ordinary read. Abandonment is a durable ``source.db``
mutation and is therefore the resident daemon's: this command lowers it to
``maintenance.blob-publications.abandon`` and renders the typed result. It
previously called ``abandon_blob_publication_receipts`` in the CLI's own
process, which ``DELETE``s from and commits against the durable source tier
with no daemon involved, no write lease, and no audited preview.
"""

from __future__ import annotations

import json

import click

from polylogue.config import Config
from polylogue.paths import archive_root, render_root
from polylogue.storage.blob_publication import inspect_blob_publication_receipts


def _submit_abandonment(config: Config, publication_ids: tuple[str, ...]) -> dict[str, object]:
    from polylogue.cli.operation_kernel import OperationKernelError, configured_mutation_operation
    from polylogue.cli.shared.helpers import mutation_refusal

    operation = "maintenance.blob-publications.abandon"
    try:
        return configured_mutation_operation(config, operation, {"publication_ids": list(publication_ids)})
    except OperationKernelError as exc:
        raise mutation_refusal(exc, operation) from exc


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
    abandonment: dict[str, object] | None = None
    receipt_ref: str | None = None
    if publication_ids:
        config = Config(archive_root=root, render_root=render_root(), sources=[])
        result = _submit_abandonment(config, publication_ids)
        value = result.get("result")
        abandonment = value if isinstance(value, dict) else {}
        receipt_ref_value = result.get("receipt_ref")
        receipt_ref = str(receipt_ref_value) if receipt_ref_value is not None else None
    receipts = inspect_blob_publication_receipts(
        source_db,
        root / "blob",
        index_db_path=root / "index.db",
    )
    payload = {
        "mode": "blob_publications",
        "mutates": bool(publication_ids),
        "abandonment": abandonment,
        "receipt_ref": receipt_ref,
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
            f"{abandonment['abandoned']} receipt(s); "
            f"referenced={abandonment['skipped_referenced']}, missing={abandonment['missing_receipts']}"
        )
    click.echo(f"Publication receipts: {len(receipts)}")
    for item in receipts:
        state = "referenced" if item.referenced else "present" if item.blob_present else "missing"
        click.echo(f"  {item.publication_id} {item.blob_hash} {item.size_bytes}B {state}")


__all__ = ["blob_publications_command"]

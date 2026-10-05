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
    """Abandon every requested receipt, one bounded daemon request per chunk.

    The daemon request carries at most ``BlobPublicationsAbandonRequest``'s
    per-request bound so its result stays within the operation result budget;
    a longer operator selection is chunked here rather than refused.
    """
    from polylogue.cli.operation_kernel import OperationKernelError, configured_mutation_operation
    from polylogue.cli.shared.helpers import mutation_refusal
    from polylogue.operations.daemon_protocol import BlobPublicationsAbandonRequest

    operation = "maintenance.blob-publications.abandon"
    chunk = _abandon_chunk_size(BlobPublicationsAbandonRequest)
    results: list[dict[str, object]] = []
    for start in range(0, len(publication_ids), chunk):
        try:
            results.append(
                configured_mutation_operation(
                    config,
                    operation,
                    {"publication_ids": list(publication_ids[start : start + chunk]), "confirm": True},
                )
            )
        except OperationKernelError as exc:
            raise mutation_refusal(exc, operation) from exc
    return _merge_abandonment_results(results)


def _abandon_chunk_size(request_type: type[object]) -> int:
    field = request_type.model_fields["publication_ids"]  # type: ignore[attr-defined]
    for item in field.metadata:
        bound = getattr(item, "max_length", None)
        if isinstance(bound, int):
            return bound
    raise RuntimeError("publication_ids declares no per-request bound")


def _merge_abandonment_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Combine per-chunk results: lists concatenate, counts add, receipts accumulate."""
    if len(results) == 1:
        return results[0]
    merged: dict[str, object] = {}
    receipt_refs: list[str] = []
    for result in results:
        receipt_ref = result.get("receipt_ref")
        if receipt_ref is not None:
            receipt_refs.append(str(receipt_ref))
        value = result.get("result")
        if not isinstance(value, dict):
            continue
        for key, item in value.items():
            current = merged.get(key)
            if isinstance(item, list) and isinstance(current, list):
                merged[key] = [*current, *item]
            elif isinstance(item, int) and not isinstance(item, bool) and isinstance(current, int):
                merged[key] = current + item
            else:
                merged[key] = item
    return {"result": merged, "receipt_ref": ",".join(receipt_refs) if receipt_refs else None}


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
                "liveness_state": item.liveness.state.value,
                "blockers": list(item.liveness.blockers),
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
        state = item.liveness.state.value
        click.echo(f"  {item.publication_id} {item.blob_hash} {item.size_bytes}B {state} present={item.blob_present}")
        for blocker in item.liveness.blockers:
            click.echo(f"    {blocker}")


__all__ = ["blob_publications_command"]

"""``maintenance embedding-preservation``: carry vectors across a fresh start.

The three phases are separate invocations because the archive is discarded
between them: ``preserve`` runs against the outgoing archive, ``restore`` and
``verify`` against the rebuilt one, and ``discard`` only once the verification
receipt proves the reuse the copy existed to provide.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from polylogue.paths import archive_root


def _tier_paths(root: Path | None) -> tuple[Path, Path]:
    """The archive's embeddings tier and its active index generation."""
    from polylogue.maintenance.embedding_preservation import archive_tier_paths

    return archive_tier_paths(root if root is not None else archive_root())


def _resolved_model(model: str | None) -> str:
    if model is not None:
        return model
    from polylogue.config import load_polylogue_config

    return str(load_polylogue_config().embedding_model)


@click.group("embedding-preservation")
def embedding_preservation_group() -> None:
    """Preserve, restore, prove, and discard embedding vectors across a rebuild."""


@click.command("preserve")
@click.argument("copy_path", type=click.Path(path_type=Path))
@click.option("--archive-root", "root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--immutable/--no-immutable",
    default=False,
    show_default=True,
    help="Read the source without touching its sidecars. Requires a quiesced archive.",
)
@click.option("--output-format", "output_format", type=click.Choice(["plain", "json"]), default="plain")
def preserve_command(copy_path: Path, root: Path | None, immutable: bool, output_format: str) -> None:
    """AC1: copy the vector tables aside and record counts and a table-set digest."""
    from dataclasses import asdict

    from polylogue.maintenance.embedding_preservation import preserve_embedding_vectors

    embeddings_db, _index_db = _tier_paths(root)
    receipt = preserve_embedding_vectors(embeddings_db, copy_path, immutable=immutable)
    if output_format == "json":
        click.echo(json.dumps(asdict(receipt), indent=2, sort_keys=True))
        return
    click.echo(f"Preserved:     {receipt.copy}")
    click.echo(f"Source:        {receipt.source}")
    click.echo(f"Metadata rows: {receipt.metadata_rows:,}")
    click.echo(f"Vector rows:   {receipt.vector_rows:,}")
    click.echo(f"Digest:        {receipt.table_set_digest}")


@click.command("restore")
@click.argument("copy_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--archive-root", "root", type=click.Path(path_type=Path), default=None)
@click.option("--model", default=None, help="Embedding model to recompute addresses for (default: configured).")
@click.option("--output-format", "output_format", type=click.Choice(["plain", "json"]), default="plain")
def restore_command(copy_path: Path, root: Path | None, model: str | None, output_format: str) -> None:
    """Import the preserved vectors the rebuilt archive is about to ask for."""
    from polylogue.maintenance.embedding_preservation import recomputed_vector_hashes, restore_embedding_vectors
    from polylogue.operations.durable_change_train import acquire_durable_archive_ownership

    embeddings_db, index_db = _tier_paths(root)
    resolved = _resolved_model(model)
    wanted = recomputed_vector_hashes(index_db, model=resolved)
    archive_root_path = (root if root is not None else archive_root()).absolute()
    # Restoration mutates the rebuilt embeddings tier and must be the explicit
    # offline owner of the archive for the whole destination-binding and
    # restore window. The preserved source is opened read-only by the library.
    with acquire_durable_archive_ownership(archive_root_path, owner_id=f"embedding-restore:{os.getpid()}"):
        receipt = restore_embedding_vectors(embeddings_db, copy_path, set(wanted.values()))
    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "model": resolved,
                    "wanted_messages": len(wanted),
                    "restored_hashes": receipt.restored_hashes,
                    "misses": [
                        {"input_hash": miss.input_hash, "reason": miss.reason.value, "detail": miss.detail}
                        for miss in receipt.misses
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    click.echo(f"Model:     {resolved}")
    click.echo(f"Wanted:    {len(wanted):,} message(s)")
    click.echo(f"Restored:  {receipt.restored_hashes:,} hash(es)")
    click.echo(f"Misses:    {len(receipt.misses):,}")


@click.command("verify")
@click.argument("copy_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--archive-root", "root", type=click.Path(path_type=Path), default=None)
@click.option("--model", default=None, help="Embedding model to recompute addresses for (default: configured).")
@click.option("--minimum-hit-rate", type=float, default=None, help="Acceptance threshold (default: 0.95).")
@click.option(
    "--receipt",
    "receipt_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the AC2 proof here (default: the copy's own `.ac2.json`).",
)
@click.option("--output-format", "output_format", type=click.Choice(["plain", "json"]), default="plain")
def verify_command(
    copy_path: Path,
    root: Path | None,
    model: str | None,
    minimum_hit_rate: float | None,
    receipt_path: Path | None,
    output_format: str,
) -> None:
    """AC2: prove the rebuilt archive reuses the preserved vectors. Read-only."""
    from polylogue.maintenance.embedding_preservation import (
        DEFAULT_MINIMUM_HIT_RATE,
        ac2_receipt_path,
        recomputed_vector_hashes,
        verify_embedding_reuse,
    )

    embeddings_db, index_db = _tier_paths(root)
    resolved = _resolved_model(model)
    recomputed = recomputed_vector_hashes(index_db, model=resolved)
    verification = verify_embedding_reuse(
        embeddings_db,
        copy_path,
        recomputed,
        model=resolved,
        minimum_hit_rate=DEFAULT_MINIMUM_HIT_RATE if minimum_hit_rate is None else minimum_hit_rate,
        receipt_path=ac2_receipt_path(copy_path) if receipt_path is None else receipt_path,
    )
    proof = verification.as_receipt()
    if output_format == "json":
        click.echo(json.dumps(proof, indent=2, sort_keys=True))
    else:
        counts = proof["misses_by_reason"]
        assert isinstance(counts, dict)
        click.echo(f"Model:      {verification.model}")
        click.echo(f"Recomputed: {verification.recomputed_hashes:,} distinct address(es)")
        click.echo(f"Hits:       {verification.hit_hashes:,} ({verification.hit_rate:.4f})")
        click.echo(f"Threshold:  {verification.minimum_hit_rate}")
        for reason, count in sorted(counts.items()):
            click.echo(f"  miss {reason}: {count:,}")
        click.echo(f"AC2 passed: {verification.ac2_passed}")
    if not verification.ac2_passed:
        raise SystemExit(1)


@click.command("discard")
@click.argument("copy_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--receipt",
    "receipt_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="AC2 proof authorizing deletion (default: the copy's own `.ac2.json`).",
)
def discard_command(copy_path: Path, receipt_path: Path | None) -> None:
    """AC3: delete the preserved copy once its AC2 proof matches it."""
    from polylogue.maintenance.embedding_preservation import delete_preserved_copy

    delete_preserved_copy(copy_path, receipt_path=receipt_path)
    click.echo(f"Deleted: {copy_path}")


for _command in (preserve_command, restore_command, verify_command, discard_command):
    embedding_preservation_group.add_command(_command)

del _command

__all__ = ["embedding_preservation_group"]

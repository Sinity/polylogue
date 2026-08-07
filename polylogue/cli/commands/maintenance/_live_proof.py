"""``maintenance live-proof``: collect one fixed, immutable evidence receipt."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.paths import archive_root


@click.command("live-proof")
@click.option("--proof-id", required=True, help="One registered live-proof id.")
@click.option("--candidate-generation", type=str, help="Inactive generation id for the candidate proof route only.")
@click.option(
    "--apply-receipt",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, readable=True),
    help="Existing immutable apply receipt for the existing-apply route only.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, writable=True),
    required=True,
    help="New receipt path outside the archive. Existing files are refused.",
)
def live_proof_command(
    proof_id: str,
    candidate_generation: str | None,
    apply_receipt: Path | None,
    output: Path,
) -> None:
    """Collect one registered proof without mutating archive or daemon state."""

    from polylogue.maintenance.live_proof import LiveProofError, collect_live_proof, write_live_proof_receipt

    root = archive_root().resolve()
    target = output.expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        raise click.BadParameter("receipt output must be outside the archive root", param_hint="--output")
    try:
        receipt = collect_live_proof(
            proof_id,
            root,
            candidate_generation_id=candidate_generation,
            apply_receipt_path=apply_receipt,
        )
        write_live_proof_receipt(target, receipt)
    except LiveProofError as exc:
        raise click.ClickException(str(exc)) from exc
    except OSError as exc:
        raise click.ClickException("live-proof receipt output could not be written") from exc
    click.echo(json.dumps(receipt.to_document(), indent=2, sort_keys=True))

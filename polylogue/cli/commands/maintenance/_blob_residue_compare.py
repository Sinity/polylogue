"""``maintenance blob-residue-compare``: extend a residue census read-only."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("blob-residue-compare")
@click.option(
    "--census",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    required=True,
    help="Existing blob-residue census JSON.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Extended receipt destination.",
)
@click.option(
    "--blob-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False, readable=True),
    required=True,
    help="Content-addressed blob store root.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def blob_residue_compare_command(census: Path, output: Path, blob_root: Path, output_format: str) -> None:
    """Compare present residue candidates through the production parse route."""
    from polylogue.maintenance.blob_residue_comparison import extend_census

    try:
        receipt = extend_census(json.loads(census.read_text(encoding="utf-8")), blob_root=blob_root)
        output.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc

    comparison = receipt["normalized_comparison"]
    if not isinstance(comparison, dict):
        raise click.ClickException("receipt carries no normalized comparison")
    counts = comparison["outcome_counts"]
    authority = comparison["authority_outcome_counts"]
    unresolved = comparison["unresolved_candidate_count"]
    accepted = bool(comparison["accepted"])
    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "output": str(output),
                    "outcome_counts": counts,
                    "authority_outcome_counts": authority,
                    "unresolved_candidate_count": unresolved,
                    "accepted": accepted,
                },
                sort_keys=True,
            )
        )
    else:
        click.echo(f"Normalized residue comparison: {output}")
        click.echo(f"Outcomes: {json.dumps(counts, sort_keys=True)}")
        click.echo(f"Authority outcomes: {json.dumps(authority, sort_keys=True)}")
        click.echo(f"Accepted (zero unresolved): {accepted}")
        click.echo("Read-only: true")
    if not accepted:
        # The receipt is written either way; a non-zero status is what stops a
        # census with unresolved candidates from being read as an accounting.
        raise SystemExit(1)


__all__ = ["blob_residue_compare_command"]

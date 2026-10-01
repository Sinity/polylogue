"""Explicit authenticated backup restoration into a fresh destination."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.cli.shared.types import AppEnv


@click.command("restore-verified-backup")
@click.option("--backup-dir", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--destination", required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option("--format", "output_format", type=click.Choice(("plain", "json")), default="plain")
@click.pass_obj
def restore_verified_backup_command(env: AppEnv, backup_dir: Path, destination: Path, output_format: str) -> None:
    """Restore verified Source, User and Audit custody under new train authority."""
    from polylogue.cli.archive_query import submit_cli_mutation

    completed = submit_cli_mutation(
        env,
        "maintenance.restore_verified_backup",
        {"backup_dir": str(backup_dir.absolute()), "destination": str(destination.absolute())},
    )
    result = completed["result"]
    if output_format == "json":
        click.echo(json.dumps(result, sort_keys=True))
    else:
        click.echo(f"Restored archive: {destination}")
        if isinstance(result, dict) and result.get("unrestored_purchased_tiers"):
            click.echo("Purchased Embeddings remain unrestored; operational admission is degraded.")
        if isinstance(result, dict) and result.get("unrestored_referenced_blobs"):
            click.echo(
                f"Referenced blobs remain unrestored: {result['unrestored_referenced_blobs']}; admission is degraded."
            )
        if isinstance(result, dict) and result.get("requires_convergence"):
            click.echo("Derived tiers require convergence: " + ", ".join(result["requires_convergence"]))

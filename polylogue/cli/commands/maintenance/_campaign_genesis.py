"""``maintenance verify-campaign-genesis``: verify pinned historical evidence."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("verify-campaign-genesis")
@click.option(
    "--genesis",
    "genesis_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Campaign-genesis JSON record to verify.",
)
@click.option(
    "--repository",
    "repository_path",
    default=Path.cwd,
    show_default="current directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Git repository containing the historical blobs.",
)
def verify_campaign_genesis_command(genesis_path: Path, repository_path: Path) -> None:
    """Prove a campaign genesis record still binds its historical Git blobs.

    The command reads only the supplied immutable record and the revisions it
    pins. It does not inspect or access any live task authority.
    """
    from polylogue.maintenance.campaign_genesis import verify_campaign_genesis

    try:
        result = verify_campaign_genesis(genesis_path, cwd=repository_path)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "campaign_id": result.campaign_id,
                "snapshots": {
                    key: {"revision": revision, "path": path, "sha256": digest}
                    for key, (revision, path, digest) in result.snapshots.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )

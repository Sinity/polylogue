"""CLI adapter for the read-only retired Beads-origin census."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.maintenance.beads_origin_census import BeadsOriginCensusError, write_census_receipt


@click.command("beads-origin-census")
@click.option("--receipt", type=click.Path(path_type=Path), required=True, help="New immutable receipt path.")
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
@click.pass_obj
def beads_origin_census_command(env: AppEnv, receipt: Path, output_format: str) -> None:
    """Census configured archive/raw roots for retired beads-issue evidence."""
    try:
        payload = write_census_receipt(env.runtime, receipt)
    except (BeadsOriginCensusError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    states = {str(item["name"]): str(item["state"]) for item in payload["surfaces"]}
    click.echo(f"Beads origin census: {payload['plan_digest']}")
    click.echo(f"Receipt: {receipt}")
    click.echo(f"Surfaces: {json.dumps(states, sort_keys=True)}")
    click.echo("Production mutation performed: false")

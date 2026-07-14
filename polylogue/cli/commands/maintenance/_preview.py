"""``maintenance preview``: read-only staleness inventory by model and scope."""

from __future__ import annotations

import json

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.logging import configure_logging

# Mirrors polylogue.maintenance.preview.ALL_SCOPES. Hardcoded (not imported)
# so this decorator's choices don't force polylogue.maintenance.preview --
# and its storage.derived/storage.repair import chain -- onto the `--help`
# path; test_preview_scopes_match_preview_module asserts these stay in sync.
# See polylogue-sod7.
_ALL_SCOPES = ("derived", "retrieval", "archive_cleanup", "backfill")


@click.command("preview")
@click.option(
    "--scope",
    "scopes",
    multiple=True,
    type=click.Choice(_ALL_SCOPES),
    help="Limit preview to named scopes (derived, retrieval, archive_cleanup, backfill).",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--shallow",
    is_flag=True,
    help="Skip the expensive full-verification path (faster, slightly less accurate).",
)
@click.pass_obj
def preview_command(
    env: AppEnv,
    scopes: tuple[str, ...],
    output_format: str,
    shallow: bool,
) -> None:
    """Staleness inventory by model and scope. Read-only.

    Shows per-model counts of stale/missing/orphan rows with typed
    :class:`InvalidationReason` tags. Use before triggering ``polylogue
    maintenance run`` so the operator knows what will be rebuilt and why.
    Models with nothing stale produce explicit zero rows rather than
    being absent from the output.
    """
    from polylogue.maintenance.preview import staleness_inventory

    configure_logging()
    inventory = staleness_inventory(
        scopes=scopes or None,
        verify_full=not shallow,
    )

    if output_format == "json":
        click.echo(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
        return

    click.echo(f"Captured: {inventory.captured_at}")
    click.echo(f"Database: {inventory.db_path}")
    click.echo(f"Scopes:   {', '.join(inventory.scopes)}")
    click.echo(f"Total stale rows: {inventory.total_stale():,}")
    click.echo("")

    by_model = inventory.by_model()
    if not by_model:
        click.echo("No models inventoried.")
        return

    for model, items in sorted(by_model.items()):
        click.echo(f"{model}:")
        for item in items:
            fraction_pct = item.fraction * 100.0
            click.echo(
                f"  {item.reason.value:>20s}  count={item.count:>10,}  fraction={fraction_pct:>5.1f}%  {item.detail}"
            )
        click.echo("")

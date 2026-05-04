"""Tags command for listing and discovering tags."""

from __future__ import annotations

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv


@click.command("tags")
@click.option("--provider", "-p", default=None, help="Filter tags by provider")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format")
@click.option("--count", "-n", type=int, default=None, help="Show top N tags")
@click.pass_obj
def tags_command(
    env: AppEnv,
    provider: str | None,
    output_format: str | None,
    count: int | None,
) -> None:
    """List all tags with conversation counts.

    \b
    Examples:
        polylogue tags                  # List all tags
        polylogue tags -p claude-ai     # Tags for Claude conversations only
        polylogue tags --format json    # Machine-readable output
        polylogue tags -n 10            # Top 10 tags
    """
    tags = run_coroutine_sync(env.polylogue.list_tags(provider=provider))

    if count is not None:
        # Truncate to top N (already sorted by count desc from SQL)
        tags = dict(list(tags.items())[:count])

    if output_format == "json":
        emit_success({"tags": tags})
        return

    if not tags:
        if provider:
            click.echo(f"No tags found for provider '{provider}'.")
        else:
            click.echo("No tags found.")
        click.echo("Hint: use --add-tag to tag conversations, e.g.: polylogue --latest --add-tag important")
        return

    from rich.table import Table

    table = Table(title=f"Tags ({provider or 'all providers'}, {len(tags)} total)")
    table.add_column("Tag", style="bold cyan")
    table.add_column("Count", justify="right", style="green")

    for tag, tag_count in tags.items():
        table.add_row(tag, str(tag_count))

    env.ui.print(table)


__all__ = ["tags_command"]

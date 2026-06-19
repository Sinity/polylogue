"""Tags command for listing and mutating session tags."""

from __future__ import annotations

import click

from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv


@click.command("tags")
@click.option("--origin", "-o", default=None, help="Filter tags by origin")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format")
@click.option("--count", "-n", type=int, default=None, help="Show top N tags")
@click.pass_obj
def tags_command(
    env: AppEnv,
    origin: str | None,
    output_format: str | None,
    count: int | None,
) -> None:
    """List user tags with session counts.

    \b
    Examples:
        polylogue tags                         # List all tags
        polylogue tags -o claude-ai-export     # Tags for Claude web sessions only
        polylogue tags --format json           # Machine-readable output
        polylogue tags -n 10                   # Top 10 tags
        polylogue find id:abc then mark --tag-add tps
        polylogue find id:abc then mark --tag-remove tps
    """
    from polylogue.api.sync.bridge import run_coroutine_sync

    tags = run_coroutine_sync(env.polylogue.list_tags(origin=origin))

    if count is not None:
        tags = dict(list(tags.items())[:count])

    if output_format == "json":
        emit_success({"tags": tags})
        return

    if not tags:
        if origin:
            click.echo(f"No tags found for origin '{origin}'.")
        else:
            click.echo("No tags found.")
        click.echo("Hint: use `polylogue find QUERY then mark --tag-add <name>` to add a tag.")
        return

    max_width = max(len(t) for t in tags) if tags else 0
    for tag, tag_count in sorted(tags.items(), key=lambda x: x[1], reverse=True):
        click.echo(f"{tag:<{max_width}}  {tag_count}")

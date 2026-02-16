"""Tags command for listing and discovering tags."""

from __future__ import annotations

import json

import click

from polylogue.cli.types import AppEnv


@click.command("tags")
@click.option("--provider", "-p", default=None, help="Filter tags by provider")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.option("--count", "-n", type=int, default=None, help="Show top N tags")
@click.pass_obj
def tags_command(
    env: AppEnv,
    provider: str | None,
    json_mode: bool,
    count: int | None,
) -> None:
    """List all tags with conversation counts.

    \b
    Examples:
        polylogue tags                  # List all tags
        polylogue tags -p claude        # Tags for Claude conversations only
        polylogue tags --json           # Machine-readable output
        polylogue tags -n 10            # Top 10 tags
    """
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

    backend = SQLiteBackend()
    repo = ConversationRepository(backend=backend)

    tags = repo.list_tags(provider=provider)

    if count is not None:
        # Truncate to top N (already sorted by count desc from SQL)
        tags = dict(list(tags.items())[:count])

    if json_mode:
        click.echo(json.dumps(tags, indent=2))
        return

    if not tags:
        if provider:
            click.echo(f"No tags found for provider '{provider}'.")
        else:
            click.echo("No tags found.")
        click.echo("Hint: use --add-tag to tag conversations, e.g.: polylogue --latest --add-tag important")
        return

    # Calculate column widths
    max_tag_len = max(len(t) for t in tags)
    max_count_len = max(len(str(c)) for c in tags.values())

    header = provider or "all providers"
    click.echo(f"Tags ({header}, {len(tags)} total):\n")
    for tag, tag_count in tags.items():
        click.echo(f"  {tag:<{max_tag_len}}  {tag_count:>{max_count_len}}")


__all__ = ["tags_command"]

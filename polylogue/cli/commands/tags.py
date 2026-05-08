"""Tags command for listing and mutating conversation tags."""

from __future__ import annotations

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv


def _resolve_id(env: AppEnv, raw: str) -> str | None:
    """Resolve a short or full conversation ID via get_conversation."""
    conv = run_coroutine_sync(env.polylogue.get_conversation(raw))
    return str(conv.id) if conv is not None else None


@click.command("tags")
@click.argument("conversation_id", required=False)
@click.option("--provider", "-p", default=None, help="Filter tags by provider")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format")
@click.option("--count", "-n", type=int, default=None, help="Show top N tags")
@click.option("--add-tag", "add_tag_name", type=str, default=None, help="Add a tag to the given conversation")
@click.option(
    "--remove-tag", "remove_tag_name", type=str, default=None, help="Remove a tag from the given conversation"
)
@click.pass_obj
def tags_command(
    env: AppEnv,
    conversation_id: str | None,
    provider: str | None,
    output_format: str | None,
    count: int | None,
    add_tag_name: str | None,
    remove_tag_name: str | None,
) -> None:
    """List all tags with conversation counts, or add/remove tags.

    \b
    Examples:
        polylogue tags                         # List all tags
        polylogue tags -p claude-ai            # Tags for Claude conversations only
        polylogue tags --format json           # Machine-readable output
        polylogue tags -n 10                   # Top 10 tags
        polylogue tags conv:abc --add-tag tps  # Add a tag
        polylogue tags conv:abc --rm-tag tps   # Remove a tag
    """
    if add_tag_name or remove_tag_name:
        if not conversation_id:
            raise click.UsageError("A conversation_id is required with --add-tag or --remove-tag")

        resolved = _resolve_id(env, conversation_id)
        if resolved is None:
            raise click.ClickException(f"Conversation not found: {conversation_id}")

        if add_tag_name:
            was_added = run_coroutine_sync(env.polylogue.add_tag(resolved, add_tag_name))
            status = "ok" if was_added else "unchanged"
            detail = None if was_added else "already_present"
            if output_format == "json":
                emit_success({"status": status, "conversation_id": resolved, "tag": add_tag_name, "detail": detail})
            else:
                click.echo(f"Tag '{add_tag_name}' on {resolved}: {status}" + (f" ({detail})" if detail else ""))

        if remove_tag_name:
            was_removed = run_coroutine_sync(env.polylogue.remove_tag(resolved, remove_tag_name))
            status = "ok" if was_removed else "not_found"
            detail = None if was_removed else "tag_not_present"
            if output_format == "json":
                emit_success({"status": status, "conversation_id": resolved, "tag": remove_tag_name, "detail": detail})
            else:
                click.echo(f"Tag '{remove_tag_name}' on {resolved}: {status}" + (f" ({detail})" if detail else ""))

        return

    tags = run_coroutine_sync(env.polylogue.list_tags(provider=provider))

    if count is not None:
        tags = dict(list(tags.items())[:count])

    if output_format == "json":
        emit_success({"tags": tags})
        return

    if not tags:
        click.echo("No tags found.")
        return

    max_width = max(len(t) for t in tags) if tags else 0
    for tag, tag_count in sorted(tags.items(), key=lambda x: x[1], reverse=True):
        click.echo(f"{tag:<{max_width}}  {tag_count}")

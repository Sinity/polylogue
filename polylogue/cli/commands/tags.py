"""Tags command for listing and mutating session tags."""

from __future__ import annotations

import click

from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv


def _resolve_id(env: AppEnv, raw: str) -> str | None:
    """Resolve a short or full session ID via get_session."""
    from polylogue.api.sync.bridge import run_coroutine_sync

    conv = run_coroutine_sync(env.polylogue.get_session(raw))
    return str(conv.id) if conv is not None else None


@click.command("tags")
@click.argument("session_id", required=False)
@click.option("--origin", "-o", default=None, help="Filter tags by origin")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format")
@click.option("--count", "-n", type=int, default=None, help="Show top N tags")
@click.option("--add-tag", "add_tag_name", type=str, default=None, help="Add a tag to the given session")
@click.option("--remove-tag", "remove_tag_name", type=str, default=None, help="Remove a tag from the given session")
@click.pass_obj
def tags_command(
    env: AppEnv,
    session_id: str | None,
    origin: str | None,
    output_format: str | None,
    count: int | None,
    add_tag_name: str | None,
    remove_tag_name: str | None,
) -> None:
    """List all tags with session counts, or add/remove tags.

    \b
    Examples:
        polylogue tags                         # List all tags
        polylogue tags -o claude-ai-export     # Tags for Claude web sessions only
        polylogue tags --format json           # Machine-readable output
        polylogue tags -n 10                   # Top 10 tags
        polylogue tags conv:abc --add-tag tps  # Add a tag
        polylogue tags conv:abc --rm-tag tps   # Remove a tag
    """
    from polylogue.api.sync.bridge import run_coroutine_sync

    if add_tag_name or remove_tag_name:
        if not session_id:
            raise click.UsageError("A session_id is required with --add-tag or --remove-tag")

        resolved = _resolve_id(env, session_id)
        if resolved is None:
            raise click.ClickException(f"Session not found: {session_id}")

        if add_tag_name:
            result = run_coroutine_sync(env.polylogue.add_tag(resolved, add_tag_name))
            status = "ok" if result.outcome == "added" else "unchanged"
            if output_format == "json":
                emit_success(
                    {
                        "status": status,
                        "session_id": resolved,
                        "tag": add_tag_name,
                        "detail": result.detail,
                        "outcome": result.outcome,
                    }
                )
            else:
                click.echo(
                    f"Tag '{add_tag_name}' on {resolved}: {status} ({result.outcome})"
                    + (f" ({result.detail})" if result.detail else "")
                )

        if remove_tag_name:
            result = run_coroutine_sync(env.polylogue.remove_tag(resolved, remove_tag_name))
            status = "ok" if result.outcome == "removed" else "not_found"
            if output_format == "json":
                emit_success(
                    {
                        "status": status,
                        "session_id": resolved,
                        "tag": remove_tag_name,
                        "detail": result.detail,
                        "outcome": result.outcome,
                    }
                )
            else:
                click.echo(
                    f"Tag '{remove_tag_name}' on {resolved}: {status} ({result.outcome})"
                    + (f" ({result.detail})" if result.detail else "")
                )

        return

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
        click.echo("Hint: use --add-tag <name> <session_id> to add a tag.")
        return

    max_width = max(len(t) for t in tags) if tags else 0
    for tag, tag_count in sorted(tags.items(), key=lambda x: x[1], reverse=True):
        click.echo(f"{tag:<{max_width}}  {tag_count}")

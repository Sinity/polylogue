"""Single-session export command."""

from __future__ import annotations

import click

from polylogue.cli.shared.helper_support import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.rendering.formatting import SESSION_OUTPUT_FORMATS, format_session


def _root_message_roles(ctx: click.Context) -> tuple[object, ...]:
    """Read the parent ``--message-role`` filter; honor ``--dialogue-only``."""
    from polylogue.archive.message.roles import normalize_message_roles

    parent = ctx.parent
    if parent is None:
        return ()
    raw_roles = parent.params.get("message_role")
    if raw_roles:
        return tuple(normalize_message_roles(raw_roles))
    if parent.params.get("dialogue_only"):
        from polylogue.archive.message.roles import Role

        return (Role.USER, Role.ASSISTANT)
    return ()


@click.command("export")
@click.argument("session_id", required=False)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(SESSION_OUTPUT_FORMATS),
    default="markdown",
    show_default=True,
    help="Output format",
)
@click.option("--fields", help="Fields for JSON/YAML outputs")
@click.pass_context
def export_command(ctx: click.Context, session_id: str | None, output_format: str, fields: str | None) -> None:
    """Export one known session by ID. IDs can use the origin:id-prefix form (e.g. claude-ai-export:abc123).

    The session id may be omitted when a root filter like ``--latest`` or
    ``--origin`` narrows the archive to a single match (#1642).
    """
    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.shared.insight_command_contracts import find_root_params
    from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params

    env: AppEnv = ctx.obj
    if session_id is None:
        root_params = dict(find_root_params(ctx))
        session_id = resolve_session_id_from_root_params(root_params)
        if not session_id:
            fail("export", "export requires a session ID (positional, --id, or --latest)")
    session = run_coroutine_sync(env.polylogue.get_session(session_id))
    if session is None:
        fail("export", f"Session not found: {session_id}")
    roles = _root_message_roles(ctx)
    if roles:
        session = session.with_roles(roles)
    click.echo(format_session(session, output_format, fields))


__all__ = ["export_command"]

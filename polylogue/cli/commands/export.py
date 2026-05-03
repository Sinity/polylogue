"""Single-conversation export command."""

from __future__ import annotations

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.message.roles import normalize_message_roles
from polylogue.cli.shared.helper_support import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.rendering.formatting import CONVERSATION_OUTPUT_FORMATS, format_conversation


def _root_message_roles(ctx: click.Context) -> tuple[object, ...]:
    """Read the parent ``--message-role`` filter; honor ``--dialogue-only``."""
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
@click.argument("conversation_id")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(CONVERSATION_OUTPUT_FORMATS),
    default="markdown",
    show_default=True,
    help="Output format",
)
@click.option("--fields", help="Fields for JSON/YAML outputs")
@click.pass_context
def export_command(ctx: click.Context, conversation_id: str, output_format: str, fields: str | None) -> None:
    """Export one known conversation by ID. IDs can use the provider:id-prefix form (e.g. claude-ai:abc123)."""
    env: AppEnv = ctx.obj
    conversation = run_coroutine_sync(env.polylogue.get_conversation(conversation_id))
    if conversation is None:
        fail("export", f"Conversation not found: {conversation_id}")
    roles = _root_message_roles(ctx)
    if roles:
        conversation = conversation.with_roles(roles)
    click.echo(format_conversation(conversation, output_format, fields))


__all__ = ["export_command"]

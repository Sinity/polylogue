"""Single-conversation export command."""

from __future__ import annotations

import click

from polylogue.cli.helper_support import fail
from polylogue.cli.types import AppEnv
from polylogue.rendering.formatting import CONVERSATION_OUTPUT_FORMATS, format_conversation
from polylogue.sync_bridge import run_coroutine_sync


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
@click.pass_obj
def export_command(env: AppEnv, conversation_id: str, output_format: str, fields: str | None) -> None:
    """Export one known conversation by ID."""
    conversation = run_coroutine_sync(env.operations.get_conversation(conversation_id))
    if conversation is None:
        fail("export", f"Conversation not found: {conversation_id}")
    click.echo(format_conversation(conversation, output_format, fields))


__all__ = ["export_command"]

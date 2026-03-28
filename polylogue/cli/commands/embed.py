"""Embedding generation command."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import click

from polylogue.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.storage.repository import ConversationRepository


@click.command("embed")
@click.option(
    "--conversation", "-c",
    type=str,
    default=None,
    help="Embed a specific conversation by ID",
)
@click.option(
    "--model",
    type=click.Choice(["voyage-4", "voyage-4-large", "voyage-4-lite"]),
    default="voyage-4",
    help="Voyage AI model: voyage-4 (default), voyage-4-large, voyage-4-lite",
)
@click.option(
    "--rebuild", "-r",
    is_flag=True,
    help="Re-embed all conversations (ignore existing embeddings)",
)
@click.option(
    "--stats", "-s",
    is_flag=True,
    help="Show embedding statistics only",
)
@click.option(
    "--limit", "-n",
    type=int,
    default=None,
    help="Maximum number of conversations to embed",
)
@click.pass_obj
def embed_command(
    env: AppEnv,
    conversation: str | None,
    model: str,
    rebuild: bool,
    stats: bool,
    limit: int | None,
) -> None:
    """Generate semantic embeddings for conversations.

    Uses Voyage AI to generate vector embeddings for message content,
    stored in sqlite-vec for semantic search.

    Requires VOYAGE_API_KEY environment variable to be set.

    \b
    Examples:
        polylogue embed                    # Embed all unembedded conversations
        polylogue embed -c <id>            # Embed specific conversation
        polylogue embed --model voyage-4-large  # Use larger model
        polylogue embed --rebuild          # Re-embed everything
        polylogue embed --stats            # Show embedding statistics
    """
    import os

    from polylogue.storage.search_providers import create_vector_provider

    # Check for API key
    voyage_key = os.environ.get("POLYLOGUE_VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY")
    if not voyage_key and not stats:
        click.echo("Error: VOYAGE_API_KEY environment variable not set", err=True)
        click.echo("Set it with: export VOYAGE_API_KEY=your-api-key  (or POLYLOGUE_VOYAGE_API_KEY)", err=True)
        raise click.Abort()

    # Stats only mode
    if stats:
        _show_embedding_stats(env)
        return

    # Create vector provider
    vec_provider = create_vector_provider(voyage_api_key=voyage_key)
    if vec_provider is None:
        click.echo("Error: sqlite-vec not available", err=True)
        click.echo("sqlite-vec is not available (ensure it is in your Nix flake or virtualenv)", err=True)
        raise click.Abort()

    # Set model if different from default
    if model != "voyage-4":
        vec_provider.model = model

    repo = env.repository

    # Embed specific conversation
    if conversation:
        _embed_single(env, repo, vec_provider, conversation)
        return

    # Embed all unembedded (or all if rebuild)
    _embed_batch(env, repo, vec_provider, rebuild=rebuild, limit=limit)


def _show_embedding_stats(env: AppEnv) -> None:
    """Display embedding statistics."""
    from polylogue.storage.backends.connection import open_connection

    with open_connection(env.config.db_path) as conn:
        # Total conversations
        total_convs = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        # Conversations with embeddings
        try:
            embedded_convs = conn.execute(
                "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
            ).fetchone()[0]
        except sqlite3.OperationalError as exc:
            logger.debug("embedding_status query failed (table may not exist): %s", exc)
            embedded_convs = 0

        # Total embedded messages
        try:
            embedded_msgs = conn.execute(
                "SELECT COUNT(*) FROM message_embeddings"
            ).fetchone()[0]
        except sqlite3.OperationalError as exc:
            logger.debug("message_embeddings query failed (table may not exist): %s", exc)
            embedded_msgs = 0

        # Pending conversations
        try:
            pending = conn.execute(
                "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1"
            ).fetchone()[0]
        except sqlite3.OperationalError as exc:
            logger.debug("pending embeddings query failed: %s", exc)
            pending = total_convs - embedded_convs

    coverage = (embedded_convs / total_convs * 100) if total_convs > 0 else 0

    click.echo("\nEmbedding Statistics")
    click.echo(f"  Total conversations:    {total_convs}")
    click.echo(f"  Embedded conversations: {embedded_convs}")
    click.echo(f"  Embedded messages:      {embedded_msgs}")
    click.echo(f"  Coverage:               {coverage:.1f}%")
    click.echo(f"  Pending:                {pending}")


def _embed_single(
    env: AppEnv,
    repo: ConversationRepository,
    vec_provider: object,
    conversation_id: str,
) -> None:
    """Embed a single conversation."""
    import asyncio

    async def _fetch() -> tuple[object, list] | None:
        conv = await repo.view(conversation_id)  # view() resolves partial IDs
        if conv is None:
            return None
        messages = await repo.queries.get_messages(str(conv.id))
        return conv, messages

    result = asyncio.run(_fetch())
    if result is None:
        click.echo(f"Error: Conversation {conversation_id} not found", err=True)
        raise click.Abort()

    conv, messages = result

    if not messages:
        click.echo(f"No messages to embed in {conversation_id}")
        return

    click.echo(f"Embedding {len(messages)} messages from {conv.title or conversation_id[:12]}...")

    try:
        vec_provider.upsert(str(conv.id), messages)  # type: ignore
        click.echo(f"✓ Embedded {str(conv.id)[:12]}")
    except Exception as exc:
        click.echo(f"Error embedding {conversation_id}: {exc}", err=True)
        raise click.Abort() from exc


def _embed_batch(
    env: AppEnv,
    repo: ConversationRepository,
    vec_provider: object,
    *,
    rebuild: bool = False,
    limit: int | None = None,
) -> None:
    """Embed multiple conversations."""
    import asyncio

    from polylogue.storage.backends.connection import open_connection

    backend = repo.backend

    # Get conversations to embed — stream via fetchmany to bound memory.
    conv_ids: list[tuple[str, str | None]] = []
    with open_connection(backend.db_path) as conn:
        if rebuild:
            cursor = conn.execute(
                "SELECT conversation_id, title FROM conversations ORDER BY updated_at DESC"
            )
        else:
            cursor = conn.execute(
                """
                SELECT c.conversation_id, c.title
                FROM conversations c
                LEFT JOIN embedding_status e ON c.conversation_id = e.conversation_id
                WHERE e.conversation_id IS NULL OR e.needs_reindex = 1
                ORDER BY c.updated_at DESC
                """
            )
        while True:
            rows = cursor.fetchmany(500)
            if not rows:
                break
            for row in rows:
                conv_ids.append((row["conversation_id"], row["title"]))
                if limit and len(conv_ids) >= limit:
                    break
            if limit and len(conv_ids) >= limit:
                break

    if not conv_ids:
        click.echo("All conversations are already embedded.")
        return

    click.echo(f"Embedding {len(conv_ids)} conversations...")

    embedded_count = 0
    error_count = 0

    # Single event loop for all async backend calls, avoids per-call overhead.
    loop = asyncio.new_event_loop()

    def _embed_one(conversation_id: str) -> bool:
        """Embed a single conversation. Returns True on success."""
        messages = loop.run_until_complete(backend.queries.get_messages(conversation_id))

        if messages:
            vec_provider.upsert(conversation_id, messages)  # type: ignore
            return True
        return False

    try:
        with env.ui.progress("Embedding conversations", total=len(conv_ids)) as progress:
            for i, (conv_id, title) in enumerate(conv_ids, 1):
                if not env.ui.plain:
                    progress.update(description=f"Embedding {title or conv_id[:12]}...")
                try:
                    if _embed_one(conv_id):
                        embedded_count += 1
                except Exception as exc:
                    error_count += 1
                    label = title or conv_id[:12]
                    env.ui.console.print(f"Warning: [{i}/{len(conv_ids)}] {label}: {exc}")
                progress.advance()
    finally:
        loop.close()

    click.echo(
        f"\n✓ Embedded {embedded_count} conversations"
        + (f" ({error_count} errors)" if error_count else "")
    )


__all__ = ["embed_command"]

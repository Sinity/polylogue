"""Embedding generation command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click  # noqa: F401

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
    help="Voyage AI model to use (default: voyage-4)",
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

    from polylogue.storage.backends.sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search_providers import create_vector_provider

    # Check for API key
    voyage_key = os.environ.get("POLYLOGUE_VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY")
    if not voyage_key and not stats:
        click.echo("Error: VOYAGE_API_KEY environment variable not set", err=True)
        click.echo("Set it with: export VOYAGE_API_KEY=your-api-key", err=True)
        raise click.Abort()

    # Stats only mode
    if stats:
        _show_embedding_stats(env)
        return

    # Create vector provider
    vec_provider = create_vector_provider(voyage_api_key=voyage_key)
    if vec_provider is None:
        click.echo("Error: sqlite-vec not available", err=True)
        click.echo("Install with: pip install sqlite-vec", err=True)
        raise click.Abort()

    # Set model if different from default
    if model != "voyage-4":
        vec_provider.model = model

    backend = SQLiteBackend()
    repo = ConversationRepository(backend=backend)

    # Embed specific conversation
    if conversation:
        _embed_single(env, repo, vec_provider, conversation)
        return

    # Embed all unembedded (or all if rebuild)
    _embed_batch(env, repo, vec_provider, rebuild=rebuild, limit=limit)


def _show_embedding_stats(env: AppEnv) -> None:
    """Display embedding statistics."""
    from polylogue.storage.backends.sqlite import open_connection

    with open_connection(None) as conn:
        # Total conversations
        total_convs = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        # Conversations with embeddings
        try:
            embedded_convs = conn.execute(
                "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
            ).fetchone()[0]
        except Exception:
            embedded_convs = 0

        # Total embedded messages
        try:
            embedded_msgs = conn.execute(
                "SELECT COUNT(*) FROM message_embeddings"
            ).fetchone()[0]
        except Exception:
            embedded_msgs = 0

        # Pending conversations
        try:
            pending = conn.execute(
                "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1"
            ).fetchone()[0]
        except Exception:
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
    from polylogue.storage.backends.sqlite import open_connection
    from polylogue.storage.store import MessageRecord

    # Get conversation
    conv = repo.get(conversation_id)
    if conv is None:
        click.echo(f"Error: Conversation {conversation_id} not found", err=True)
        raise click.Abort()

    # Get messages
    with open_connection(None) as conn:
        rows = conn.execute(
            """
            SELECT message_id, conversation_id, role, text, content_hash, provider_meta, version
            FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()

        messages = []
        for row in rows:
            import contextlib
            import json
            provider_meta = None
            if row["provider_meta"]:
                with contextlib.suppress(Exception):
                    provider_meta = json.loads(row["provider_meta"])

            messages.append(MessageRecord(
                message_id=row["message_id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                text=row["text"],
                content_hash=row["content_hash"],
                provider_meta=provider_meta,
                version=row["version"],
            ))

    if not messages:
        click.echo(f"No messages to embed in {conversation_id}")
        return

    click.echo(f"Embedding {len(messages)} messages from {conv.title or conversation_id[:12]}...")

    try:
        vec_provider.upsert(conversation_id, messages)  # type: ignore
        click.echo(f"✓ Embedded {conversation_id[:12]}")
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
    from rich.console import Console as RichConsole
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    from polylogue.storage.backends.sqlite import open_connection
    from polylogue.storage.store import MessageRecord

    # Get conversations to embed
    with open_connection(None) as conn:
        if rebuild:
            # All conversations
            rows = conn.execute(
                "SELECT conversation_id, title FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        else:
            # Only those needing embedding
            rows = conn.execute(
                """
                SELECT c.conversation_id, c.title
                FROM conversations c
                LEFT JOIN embedding_status e ON c.conversation_id = e.conversation_id
                WHERE e.conversation_id IS NULL OR e.needs_reindex = 1
                ORDER BY c.updated_at DESC
                """
            ).fetchall()

    conv_ids = [(row["conversation_id"], row["title"]) for row in rows]

    if limit:
        conv_ids = conv_ids[:limit]

    if not conv_ids:
        click.echo("All conversations are already embedded.")
        return

    click.echo(f"Embedding {len(conv_ids)} conversations...")

    embedded_count = 0
    error_count = 0

    def _embed_one(conv_id: str, title: str | None) -> bool:
        """Embed a single conversation. Returns True on success."""
        import contextlib
        import json

        with open_connection(None) as conn:
            rows = conn.execute(
                """
                SELECT message_id, conversation_id, role, text, content_hash, provider_meta, version
                FROM messages
                WHERE conversation_id = ?
                """,
                (conv_id,),
            ).fetchall()

            messages = []
            for row in rows:
                provider_meta = None
                if row["provider_meta"]:
                    with contextlib.suppress(Exception):
                        provider_meta = json.loads(row["provider_meta"])

                messages.append(MessageRecord(
                    message_id=row["message_id"],
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    text=row["text"],
                    content_hash=row["content_hash"],
                    provider_meta=provider_meta,
                    version=row["version"],
                ))

        if messages:
            vec_provider.upsert(conv_id, messages)  # type: ignore
            return True
        return False

    # Use Rich progress bar only when console supports it
    use_rich = isinstance(env.ui.console, RichConsole)

    if use_rich:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=env.ui.console,
        ) as progress:
            task = progress.add_task("Embedding", total=len(conv_ids))

            for conv_id, title in conv_ids:
                progress.update(task, description=f"Embedding {title or conv_id[:12]}...")
                try:
                    if _embed_one(conv_id, title):
                        embedded_count += 1
                except Exception as exc:
                    error_count += 1
                    progress.console.print(f"Warning: {conv_id[:12]}: {exc}")
                progress.update(task, advance=1)
    else:
        for i, (conv_id, title) in enumerate(conv_ids, 1):
            label = title or conv_id[:12]
            click.echo(f"  [{i}/{len(conv_ids)}] {label}...", nl=False)
            try:
                if _embed_one(conv_id, title):
                    embedded_count += 1
                    click.echo(" ok")
                else:
                    click.echo(" (no messages)")
            except Exception as exc:
                error_count += 1
                click.echo(f" error: {exc}")

    click.echo(
        f"\n✓ Embedded {embedded_count} conversations"
        + (f" ({error_count} errors)" if error_count else "")
    )


__all__ = ["embed_command"]

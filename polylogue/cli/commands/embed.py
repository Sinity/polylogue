"""Embedding generation command."""

from __future__ import annotations

<<<<<<< HEAD
import sqlite3
||||||| parent of ca929b4c (refactor: split cli operator and path roots)
import json
=======
>>>>>>> ca929b4c (refactor: split cli operator and path roots)
from typing import TYPE_CHECKING

import click

<<<<<<< HEAD
from polylogue.logging import get_logger

logger = get_logger(__name__)
||||||| parent of ca929b4c (refactor: split cli operator and path roots)
=======
from polylogue.cli.embed_runtime import embed_batch, embed_single
from polylogue.cli.embed_stats import embedding_status_payload, render_embedding_stats
>>>>>>> ca929b4c (refactor: split cli operator and path roots)

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv

_embed_single = embed_single
_embed_batch = embed_batch


def _embedding_status_payload(env: AppEnv) -> dict[str, object]:
    return embedding_status_payload(env)


def _show_embedding_stats(env: AppEnv, *, json_output: bool = False) -> None:
    render_embedding_stats(_embedding_status_payload(env), json_output=json_output)


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
    """Generate semantic embeddings for conversations."""
    import os

    from polylogue.storage.search_providers import create_vector_provider

<<<<<<< HEAD
    # Check for API key
||||||| parent of ca929b4c (refactor: split cli operator and path roots)
    if json_output and not stats:
        click.echo("Error: --json requires --stats", err=True)
        raise click.Abort()

    # Check for API key
=======
    if json_output and not stats:
        click.echo("Error: --json requires --stats", err=True)
        raise click.Abort()

>>>>>>> ca929b4c (refactor: split cli operator and path roots)
    voyage_key = os.environ.get("POLYLOGUE_VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY")
    if not voyage_key and not stats:
        click.echo("Error: VOYAGE_API_KEY environment variable not set", err=True)
        click.echo("Set it with: export VOYAGE_API_KEY=your-api-key  (or POLYLOGUE_VOYAGE_API_KEY)", err=True)
        raise click.Abort()

    if stats:
        _show_embedding_stats(env)
        return

    vec_provider = create_vector_provider(voyage_api_key=voyage_key)
    if vec_provider is None:
        click.echo("Error: sqlite-vec not available", err=True)
        click.echo("sqlite-vec is not available (ensure it is in your Nix flake or virtualenv)", err=True)
        raise click.Abort()

    if model != "voyage-4":
        vec_provider.model = model

    repo = env.repository

    if conversation:
        _embed_single(env, repo, vec_provider, conversation)
        return

    _embed_batch(env, repo, vec_provider, rebuild=rebuild, limit=limit)


<<<<<<< HEAD
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
||||||| parent of ca929b4c (refactor: split cli operator and path roots)
def _embedding_status_payload(env: AppEnv) -> dict[str, object]:
    """Read canonical embedding-status statistics for operator surfaces."""
    from polylogue.storage.backends.connection import open_connection
    from polylogue.storage.embedding_stats import read_embedding_stats_sync

    with open_connection(env.config.db_path) as conn:
        total_convs = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        embedding_stats = read_embedding_stats_sync(conn)

    embedded_convs = embedding_stats.embedded_conversations
    embedded_msgs = embedding_stats.embedded_messages
    pending = embedding_stats.pending_conversations or max(total_convs - embedded_convs, 0)
    coverage = (embedded_convs / total_convs * 100) if total_convs > 0 else 0
    if total_convs <= 0:
        status = "empty"
    elif embedded_convs <= 0:
        status = "none"
    elif pending > 0:
        status = "partial"
    else:
        status = "complete"
    freshness_status = status
    if embedding_stats.embedded_messages > 0 and (
        embedding_stats.stale_messages > 0 or embedding_stats.messages_missing_provenance > 0
    ):
        freshness_status = "stale"

    return {
        "status": status,
        "total_conversations": int(total_convs),
        "embedded_conversations": int(embedded_convs),
        "embedded_messages": int(embedded_msgs),
        "pending_conversations": int(pending),
        "embedding_coverage_percent": round(float(coverage), 1),
        "retrieval_ready": bool(embedded_msgs > embedding_stats.stale_messages),
        "freshness_status": freshness_status,
        "stale_messages": int(embedding_stats.stale_messages),
        "messages_missing_provenance": int(embedding_stats.messages_missing_provenance),
        "oldest_embedded_at": embedding_stats.oldest_embedded_at,
        "newest_embedded_at": embedding_stats.newest_embedded_at,
        "embedding_models": embedding_stats.model_counts,
        "embedding_dimensions": embedding_stats.dimension_counts,
        "retrieval_bands": embedding_stats.retrieval_bands,
    }


def _show_embedding_stats(env: AppEnv, *, json_output: bool = False) -> None:
    """Display embedding statistics."""
    payload = _embedding_status_payload(env)

    if json_output:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("\nEmbedding Statistics")
    click.echo(f"  Status:                {payload['status']}")
    click.echo(f"  Total conversations:   {payload['total_conversations']}")
    click.echo(f"  Embedded conversations:{payload['embedded_conversations']:>4}")
    click.echo(f"  Embedded messages:     {payload['embedded_messages']}")
    click.echo(f"  Coverage:              {payload['embedding_coverage_percent']:.1f}%")
    click.echo(f"  Pending:               {payload['pending_conversations']}")
    click.echo(f"  Retrieval ready:       {'yes' if payload['retrieval_ready'] else 'no'}")
    click.echo(f"  Freshness:             {payload['freshness_status']}")
    click.echo(f"  Stale messages:        {payload['stale_messages']}")
    click.echo(f"  Missing provenance:    {payload['messages_missing_provenance']}")
    if payload["oldest_embedded_at"] or payload["newest_embedded_at"]:
        click.echo(
            f"  Embedded at:           {payload['oldest_embedded_at'] or '-'} -> {payload['newest_embedded_at'] or '-'}"
        )
    if payload["embedding_models"]:
        click.echo(
            f"  Models:                {', '.join(f'{name} ({count})' for name, count in payload['embedding_models'].items())}"
        )
    if payload["embedding_dimensions"]:
        click.echo(
            f"  Dimensions:            {', '.join(f'{dimension} ({count})' for dimension, count in payload['embedding_dimensions'].items())}"
        )
    if payload["retrieval_bands"]:
        click.echo("  Retrieval bands:")
        for band_name, band in payload["retrieval_bands"].items():
            status_text = "ready" if band.get("ready") else str(band.get("status", "pending"))
            click.echo(
                f"    {band_name}: {status_text}; "
                f"rows={int(band.get('materialized_rows', 0)):,}/{int(band.get('source_rows', 0) or 0):,}; "
                f"docs={int(band.get('materialized_documents', 0)):,}/{int(band.get('source_documents', 0) or 0):,}"
            )


def _embed_single(
    env: AppEnv,
    repo: ConversationRepository,
    vec_provider: object,
    conversation_id: str,
) -> None:
    """Embed a single conversation."""
    from polylogue.sync_bridge import run_coroutine_sync

    async def _fetch() -> tuple[object, list] | None:
        conv = await repo.view(conversation_id)  # view() resolves partial IDs
        if conv is None:
            return None
        messages = await repo.queries.get_messages(str(conv.id))
        return conv, messages

    result = run_coroutine_sync(_fetch())
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
    from polylogue.storage.backends.connection import open_connection
    from polylogue.sync_bridge import run_coroutine_sync

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

    def _embed_one(conversation_id: str) -> bool:
        """Embed a single conversation. Returns True on success."""
        messages = run_coroutine_sync(backend.queries.get_messages(conversation_id))

        if messages:
            vec_provider.upsert(conversation_id, messages)  # type: ignore
            return True
        return False

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

    click.echo(
        f"\n✓ Embedded {embedded_count} conversations"
        + (f" ({error_count} errors)" if error_count else "")
    )


__all__ = ["embed_command"]
=======
__all__ = [
    "_embed_batch",
    "_embed_single",
    "_embedding_status_payload",
    "_show_embedding_stats",
    "embed_command",
]
>>>>>>> ca929b4c (refactor: split cli operator and path roots)

"""Embedding execution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import click

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.store import MessageRecord


class _ProgressHandle(Protocol):
    def update(self, **kwargs: object) -> None: ...

    def advance(self, advance: float = 1) -> None: ...

    def __enter__(self) -> _ProgressHandle: ...

    def __exit__(self, *args: object) -> None: ...


class _ConsoleLike(Protocol):
    def print(self, *args: object, **kwargs: object) -> None: ...


class _EmbedUI(Protocol):
    plain: bool
    console: _ConsoleLike

    def progress(self, description: str, total: int = 0) -> _ProgressHandle: ...


class _HasUI(Protocol):
    ui: _EmbedUI


def embed_single(
    env: object,
    repo: ConversationRepository,
    vec_provider: VectorProvider,
    conversation_id: str,
) -> None:
    """Embed a single conversation."""
    from polylogue.sync_bridge import run_coroutine_sync

    async def _fetch() -> tuple[Conversation, list[MessageRecord]] | None:
        conv = await repo.view(conversation_id)
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
        vec_provider.upsert(str(conv.id), messages)
        click.echo(f"✓ Embedded {str(conv.id)[:12]}")
    except Exception as exc:
        click.echo(f"Error embedding {conversation_id}: {exc}", err=True)
        raise click.Abort() from exc


def embed_batch(
    env: object,
    repo: ConversationRepository,
    vec_provider: VectorProvider,
    *,
    rebuild: bool = False,
    limit: int | None = None,
) -> None:
    """Embed multiple conversations."""
    from polylogue.storage.backends.connection import open_read_connection
    from polylogue.sync_bridge import run_coroutine_sync

    ui = cast(_HasUI, env).ui
    backend = repo.backend
    conv_ids: list[tuple[str, str | None]] = []
    with open_read_connection(backend.db_path) as conn:
        if rebuild:
            cursor = conn.execute("SELECT conversation_id, title FROM conversations ORDER BY updated_at DESC")
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
        messages = run_coroutine_sync(repo.queries.get_messages(conversation_id))
        if messages:
            vec_provider.upsert(conversation_id, messages)
            return True
        return False

    with ui.progress("Embedding conversations", total=len(conv_ids)) as progress:
        for i, (conv_id, title) in enumerate(conv_ids, 1):
            if not ui.plain:
                progress.update(description=f"Embedding {title or conv_id[:12]}...")
            try:
                if _embed_one(conv_id):
                    embedded_count += 1
            except Exception as exc:
                error_count += 1
                label = title or conv_id[:12]
                ui.console.print(f"Warning: [{i}/{len(conv_ids)}] {label}: {exc}")
            progress.advance()

    click.echo(f"\n✓ Embedded {embedded_count} conversations" + (f" ({error_count} errors)" if error_count else ""))

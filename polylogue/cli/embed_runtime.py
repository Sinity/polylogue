"""CLI wrappers for the embedding substrate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import click

from polylogue.lib.embeddings.runtime import (
    EmbedConversationOutcome,
    embed_conversation_sync,
    iter_pending_conversations,
)

if TYPE_CHECKING:
    from polylogue.lib.embeddings.runtime import _EmbedConversationStore
    from polylogue.protocols import VectorProvider


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


@runtime_checkable
class _HasUI(Protocol):
    ui: _EmbedUI


def embed_single(
    env: object,
    repo: _EmbedConversationStore,
    vec_provider: VectorProvider,
    conversation_id: str,
) -> None:
    """Embed a single conversation, with CLI-style status output."""
    del env  # CLI surface doesn't need env state for the single path

    outcome = embed_conversation_sync(repo, vec_provider, conversation_id, fetch_title=True)
    if outcome.status == "not_found":
        click.echo(f"Error: Conversation {conversation_id} not found", err=True)
        raise click.Abort()
    if outcome.status == "no_messages":
        click.echo(f"No messages to embed in {outcome.conversation_id}")
        return

    label = outcome.title or outcome.conversation_id[:12]
    click.echo(f"Embedding {outcome.embedded_message_count} messages from {label}...")

    if outcome.status == "error":
        click.echo(f"Error embedding {conversation_id}: {outcome.error}", err=True)
        raise click.Abort()

    click.echo(f"✓ Embedded {outcome.conversation_id[:12]}")


def embed_batch(
    env: object,
    repo: _EmbedConversationStore,
    vec_provider: VectorProvider,
    *,
    rebuild: bool = False,
    limit: int | None = None,
) -> None:
    """Embed pending conversations using the rich CLI progress UI."""
    if not isinstance(env, _HasUI):
        raise TypeError(f"Embedding environment must expose ui, got {type(env).__name__}")
    ui = env.ui

    pending = iter_pending_conversations(repo.backend, rebuild=rebuild, limit=limit)
    if not pending:
        click.echo("All conversations are already embedded.")
        return

    click.echo(f"Embedding {len(pending)} conversations...")

    embedded_count = 0
    error_count = 0

    with ui.progress("Embedding conversations", total=len(pending)) as progress:
        for index, item in enumerate(pending, 1):
            label = item.title or item.conversation_id[:12]
            if not ui.plain:
                progress.update(description=f"Embedding {label}...")
            outcome = embed_conversation_sync(repo, vec_provider, item.conversation_id)
            if outcome.status == "embedded":
                embedded_count += 1
            elif outcome.status == "error":
                error_count += 1
                ui.console.print(f"Warning: [{index}/{len(pending)}] {label}: {outcome.error}")
            progress.advance()

    summary = f"\n✓ Embedded {embedded_count} conversations"
    if error_count:
        summary += f" ({error_count} errors)"
    click.echo(summary)


__all__ = ["EmbedConversationOutcome", "embed_batch", "embed_single"]

"""Substrate-side embedding execution (no CLI / click coupling).

Provides three primitives that surfaces compose into their own UI:

* :func:`iter_pending_conversations` — list conversations that need embedding.
* :func:`embed_conversation_sync` — embed messages for one conversation.
* :class:`EmbedConversationOutcome` — typed outcome record.

CLI (:mod:`polylogue.cli.shared.embed_runtime`) and pipeline
(:mod:`polylogue.pipeline.run_stages`) layer their progress and message
formatting on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
    from polylogue.storage.runtime import MessageRecord


EmbedSingleStatus = Literal["embedded", "no_messages", "not_found", "error"]


@dataclass(frozen=True, slots=True)
class PendingConversation:
    """Identifier and (optional) display title for one pending conversation."""

    conversation_id: str
    title: str | None = None


@dataclass(frozen=True, slots=True)
class EmbedConversationOutcome:
    """Typed outcome for embedding one conversation."""

    status: EmbedSingleStatus
    conversation_id: str
    title: str | None = None
    embedded_message_count: int = 0
    error: str | None = None


class _EmbedConversationStore(Protocol):
    @property
    def backend(self) -> RepositoryBackendProtocol: ...

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]: ...

    async def view(self, conversation_id: str) -> Conversation | None: ...


def iter_pending_conversations(
    backend: RepositoryBackendProtocol,
    *,
    rebuild: bool = False,
    limit: int | None = None,
) -> list[PendingConversation]:
    """Return conversations needing embedding.

    With ``rebuild=True`` returns every conversation; otherwise returns
    rows missing from ``embedding_status`` or flagged ``needs_reindex``.
    """
    from polylogue.storage.backends.connection import open_read_connection

    pending: list[PendingConversation] = []
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
                pending.append(PendingConversation(conversation_id=row["conversation_id"], title=row["title"]))
                if limit and len(pending) >= limit:
                    return pending
    return pending


def embed_conversation_sync(
    repo: _EmbedConversationStore,
    vec_provider: VectorProvider,
    conversation_id: str,
    *,
    fetch_title: bool = False,
) -> EmbedConversationOutcome:
    """Embed one conversation. Returns an outcome — does not raise on no-op.

    ``fetch_title=True`` issues an extra ``view`` lookup so callers can
    display a friendly label; when False the title field is left ``None``.
    """
    from polylogue.api.sync.bridge import run_coroutine_sync

    title: str | None = None
    if fetch_title:

        async def _view_title() -> Conversation | None:
            return await repo.view(conversation_id)

        conv = run_coroutine_sync(_view_title())
        if conv is None:
            return EmbedConversationOutcome(status="not_found", conversation_id=conversation_id)
        title = conv.title
        full_id = str(conv.id)
    else:
        full_id = conversation_id

    try:
        messages = run_coroutine_sync(repo.get_messages(full_id))
        if not messages:
            return EmbedConversationOutcome(status="no_messages", conversation_id=full_id, title=title)
        vec_provider.upsert(full_id, messages)
    except Exception as exc:  # noqa: BLE001 — surfacing as outcome
        return EmbedConversationOutcome(status="error", conversation_id=full_id, title=title, error=str(exc))
    return EmbedConversationOutcome(
        status="embedded",
        conversation_id=full_id,
        title=title,
        embedded_message_count=len(messages),
    )


__all__ = [
    "EmbedConversationOutcome",
    "EmbedSingleStatus",
    "PendingConversation",
    "embed_conversation_sync",
    "iter_pending_conversations",
]

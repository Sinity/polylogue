"""Projection-backed conversation formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.rendering.block_models import coerce_renderable_blocks
from polylogue.rendering.core_markdown import (
    _group_projection_attachments,
    _normalize_markdown_message,
    render_markdown_document,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository

if TYPE_CHECKING:
    from polylogue.storage.archive_views import ConversationRenderProjection


@dataclass
class FormattedConversationMetadata:
    """Typed metadata carried alongside the rendered markdown."""

    message_count: int
    attachment_count: int
    created_at: str | None
    updated_at: str | None


@dataclass
class FormattedConversation:
    """Structured representation of a rendered conversation."""

    title: str
    provider: str
    conversation_id: str
    markdown_text: str
    metadata: FormattedConversationMetadata


class ConversationFormatter:
    """Formats repository render projections to structured output."""

    def __init__(self, archive_root: Path, db_path: Path | None = None, backend: SQLiteBackend | None = None):
        self.archive_root = archive_root
        self.db_path = db_path
        self.backend = backend

    async def load_projection(self, conversation_id: str) -> ConversationRenderProjection:
        """Load the canonical repository-owned render projection."""
        backend = self.backend or SQLiteBackend(db_path=self.db_path)
        repository = ConversationRepository(backend=backend)
        owns_backend = self.backend is None
        try:
            projection = await repository.get_render_projection(conversation_id)
        finally:
            if owns_backend:
                await repository.close()
        if projection is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        return projection

    def format_projection(self, projection: ConversationRenderProjection) -> FormattedConversation:
        """Format a repository projection to structured output."""
        conversation = projection.conversation
        conversation_id = str(conversation.conversation_id)
        title = conversation.title or conversation_id
        provider = conversation.provider_name or "unknown"
        normalized_messages = [
            _normalize_markdown_message(
                message_id=message.message_id,
                role=message.role,
                text=message.text,
                timestamp=message.sort_key,
                default_role="message",
                content_blocks=coerce_renderable_blocks(message.content_blocks),
            )
            for message in projection.messages
        ]
        normalized_attachments = _group_projection_attachments(projection)
        markdown_text = render_markdown_document(
            title=title,
            provider=provider,
            conversation_id=conversation_id,
            messages=normalized_messages,
            attachments_by_message=normalized_attachments,
            archive_root=self.archive_root,
        )
        return FormattedConversation(
            title=title,
            provider=provider,
            conversation_id=conversation_id,
            markdown_text=markdown_text,
            metadata=FormattedConversationMetadata(
                message_count=len(projection.messages),
                attachment_count=len(projection.attachments),
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            ),
        )

    async def format(self, conversation_id: str) -> FormattedConversation:
        return self.format_projection(await self.load_projection(conversation_id))


__all__ = [
    "ConversationFormatter",
    "FormattedConversation",
    "FormattedConversationMetadata",
]

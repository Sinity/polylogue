"""Async storage repository for conversation persistence.

Provides async/await interface for storing and retrieving conversations.
Wraps AsyncSQLiteBackend for parallel operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.storage.backends.async_sqlite import AsyncSQLiteBackend
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.lib.models import Conversation


class AsyncConversationRepository:
    """Async repository for conversation storage operations.

    Wraps AsyncSQLiteBackend to provide high-level async storage interface.

    Example:
        async with AsyncConversationRepository() as repo:
            counts = await repo.save_conversation(conversation, messages, attachments)
            conv_record = await repo.get_conversation("claude:abc123")
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize async storage repository.

        Args:
            db_path: Optional path to database file
        """
        self._backend = AsyncSQLiteBackend(db_path=db_path)

    async def __aenter__(self) -> AsyncConversationRepository:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit async context manager."""
        await self.close()

    async def close(self) -> None:
        """Close database connections and release resources."""
        await self._backend.close()

    async def save_conversation(
        self,
        conversation: Conversation,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save a conversation with its messages and attachments.

        Args:
            conversation: Conversation model to save
            messages: List of message records
            attachments: List of attachment records

        Returns:
            Dictionary with counts of created/updated records:
                - conversations_created: Number of new conversations
                - conversations_updated: Number of updated conversations
                - messages_created: Number of new messages
                - attachments_created: Number of new attachments
                - attachment_refs_created: Number of new attachment references
        """
        # Convert Conversation model to ConversationRecord
        from typing import cast

        from polylogue.types import ContentHash, ConversationId

        created_at_str = conversation.created_at.isoformat() if conversation.created_at else None
        updated_at_str = conversation.updated_at.isoformat() if conversation.updated_at else (created_at_str or None)

        # Try to extract provider_id from canonical id (format: provider:id)
        # Fallback to whole ID if pattern doesn't match
        provider_id = str(conversation.id)
        if ":" in provider_id and conversation.provider:
            prefix = f"{conversation.provider}:"
            if provider_id.startswith(prefix):
                provider_id = provider_id[len(prefix) :]

        conv_record = ConversationRecord(
            conversation_id=cast(ConversationId, str(conversation.id)),
            provider_name=conversation.provider,
            provider_conversation_id=provider_id,
            title=conversation.title or "",
            created_at=created_at_str,
            updated_at=updated_at_str,
            content_hash=cast(ContentHash, conversation.metadata.get("content_hash", "")),
            provider_meta=cast(dict[str, object], conversation.metadata.get("provider_meta", {})),
            metadata=conversation.metadata,
        )

        # Save via async backend
        return await self._backend.save_conversation(conv_record, messages, attachments)

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Get conversation record by ID.

        Args:
            conversation_id: Full conversation ID

        Returns:
            ConversationRecord or None if not found
        """
        return await self._backend.get_conversation(conversation_id)

    async def conversation_exists(self, content_hash: str) -> bool:
        """Check if conversation with given content hash exists.

        Args:
            content_hash: SHA-256 hash of conversation content

        Returns:
            True if conversation exists, False otherwise
        """
        return await self._backend.conversation_exists_by_hash(content_hash)

    async def get_source_conversations(self, source_name: str) -> list[str]:
        """Get all conversation IDs from a specific source.

        Note: This filters by provider name, as source is derived from
        provider in the database schema.

        Args:
            source_name: Source identifier (e.g., "chatgpt", "claude")

        Returns:
            List of conversation IDs
        """
        conversations = await self._backend.list_conversations(
            provider=source_name,
            limit=100000,
        )
        return [conv.conversation_id for conv in conversations]

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


class AsyncStorageRepository:
    """Async repository for conversation storage operations.

    Wraps AsyncSQLiteBackend to provide high-level async storage interface.

    Example:
        async with AsyncStorageRepository() as repo:
            counts = await repo.save_conversation(conversation, messages, attachments)
            conv_record = await repo.get_conversation("claude:abc123")
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize async storage repository.

        Args:
            db_path: Optional path to database file
        """
        self._backend = AsyncSQLiteBackend(db_path=db_path)

    async def __aenter__(self) -> AsyncStorageRepository:
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
        conv_record = ConversationRecord(
            conversation_id=str(conversation.id),
            provider_name=conversation.provider,
            original_title=conversation.title or "",
            created_at=conversation.created_at,
            updated_at=conversation.updated_at or conversation.created_at,
            content_hash=conversation.metadata.get("content_hash", ""),
            source_name=conversation.metadata.get("source_name", ""),
            provider_meta=conversation.metadata.get("provider_meta", {}),
            metadata=conversation.metadata,
            message_count=len(messages),
            total_tokens=sum(msg.token_count or 0 for msg in messages),
            total_chars=sum(len(msg.text or "") for msg in messages),
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

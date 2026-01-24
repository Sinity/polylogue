"""Repository layer for conversation access.

This module provides the `ConversationRepository` class, which is the primary
interface for querying and retrieving conversations from the database.

The repository returns `Conversation` objects that support semantic projections
like `substantive_only()`, `iter_pairs()`, and `without_noise()`.

All database operations go through the StorageBackend protocol for clean abstraction.
"""

from __future__ import annotations

import builtins
import logging
from typing import TYPE_CHECKING

from polylogue.storage.db import DatabaseError
from polylogue.lib.models import Conversation
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.types import ConversationId

if TYPE_CHECKING:
    from polylogue.protocols import StorageBackend

logger = logging.getLogger(__name__)


def _records_to_conversation(
    conv_record: ConversationRecord,
    msg_records: list[MessageRecord],
    att_records: list[AttachmentRecord],
) -> Conversation:
    """Convert storage records to a Conversation domain object.

    Args:
        conv_record: Conversation record from storage
        msg_records: Message records from storage
        att_records: Attachment records from storage

    Returns:
        Conversation object with semantic projection support
    """
    return Conversation.from_records(conv_record, msg_records, att_records)


class ConversationRepository:
    """Repository for querying and retrieving conversations.

    This is the primary interface for accessing conversation data. It returns
    `Conversation` objects that support semantic projections like:

    - `substantive_only()` - Filter to substantive dialogue
    - `iter_pairs()` - Iterate user/assistant turn pairs
    - `without_noise()` - Remove tool calls, context dumps
    - `to_clean_text()` - Render as clean dialogue text

    All database operations go through a StorageBackend for clean abstraction.

    Example:
        from polylogue.storage.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(db_path="/path/to/db.db")
        repo = ConversationRepository(backend=backend)
        conv = repo.get("claude:abc123")
        if conv:
            for pair in conv.iter_pairs():
                print(pair.exchange)
    """

    def __init__(self, backend: "StorageBackend") -> None:
        """Initialize the repository.

        Args:
            backend: Storage backend for all database operations.
                    Use SQLiteBackend for SQLite, or implement the protocol
                    for PostgreSQL, DuckDB, etc.
        """
        self._backend = backend

    def resolve_id(self, id_prefix: str) -> ConversationId | None:
        """Resolve a partial ID to a full conversation ID.

        Supports both exact matches and prefix matches. If multiple
        conversations match the prefix, returns None (ambiguous).

        Args:
            id_prefix: Full or partial conversation ID.

        Returns:
            The full conversation ID if exactly one match, None otherwise.
        """
        resolved = self._backend.resolve_id(id_prefix)
        return ConversationId(resolved) if resolved else None

    def view(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with full semantic projection support.

        This is the primary API for consumers. The returned Conversation
        has methods like `substantive_only()`, `iter_pairs()`, `without_noise()`.

        Supports partial ID resolution - if a unique prefix is provided,
        it will be resolved to the full ID.

        Args:
            conversation_id: Full or partial conversation ID.

        Returns:
            A Conversation with projection methods, or None if not found.
        """
        full_id = self.resolve_id(conversation_id)
        if not full_id:
            return None
        return self.get(full_id)

    def get(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            conversation_id: Conversation ID to retrieve

        Returns:
            Conversation object with semantic projections, or None if not found
        """
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None

        msg_records = self._backend.get_messages(conversation_id)
        att_records = self._backend.get_attachments(conversation_id)

        return _records_to_conversation(conv_record, msg_records, att_records)

    def _get_many(self, conversation_ids: list[str]) -> list[Conversation]:
        """Bulk fetch full conversation objects.

        Args:
            conversation_ids: List of conversation IDs to fetch

        Returns:
            List of Conversation objects (may be fewer than requested if some don't exist)
        """
        if not conversation_ids:
            return []

        # Fetch one by one (backend doesn't have bulk API yet)
        # This could be optimized later by adding bulk methods to StorageBackend protocol
        results = []
        for cid in conversation_ids:
            conv = self.get(cid)
            if conv is not None:
                results.append(conv)
        return results

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
    ) -> list[Conversation]:
        """List conversations with optional filtering and pagination.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            provider: Optional provider filter (e.g., "claude", "chatgpt")

        Returns:
            List of Conversation objects ordered by updated_at DESC
        """
        conv_records = self._backend.list_conversations(
            provider=provider,
            limit=limit,
            offset=offset,
        )
        ids = [str(rec.conversation_id) for rec in conv_records]
        return self._get_many(ids)

    def search(self, query: str) -> "builtins.list[Conversation]":
        """Search conversations using full-text search.

        Args:
            query: Search query string

        Returns:
            List of matching conversations ordered by relevance

        Raises:
            DatabaseError: If search index not available
        """
        ids = self._backend.search_conversations(query, limit=20)
        return self._get_many(ids)

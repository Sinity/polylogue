"""Protocol definitions for external dependencies in Polylogue.

This module defines runtime-checkable Protocol interfaces that enable
dependency injection and testing throughout the codebase. These protocols
decouple the application from concrete implementations of search, storage,
vector indexing, and rendering.

Each protocol represents a well-defined contract that implementations must
fulfill, allowing for:
- Easy testing via mock implementations
- Runtime verification of interface compliance
- Flexible architecture with swappable backends
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation


@runtime_checkable
class SearchProvider(Protocol):
    """Full-text search provider for message content.

    This protocol defines the interface for indexing and searching message text.
    Implementations might use FTS5, Elasticsearch, or other search backends.

    Example implementations:
    - SQLite FTS5 (polylogue.storage.index)
    - External search engines (future)
    """

    def index(self, messages: list[MessageRecord]) -> None:
        """Index a list of messages for full-text search.

        Args:
            messages: Messages to index. Must include message_id, conversation_id,
                     and text content.

        Raises:
            SearchError: If indexing fails
        """
        ...

    def search(self, query: str) -> list[str]:
        """Execute a full-text search query.

        Args:
            query: Search query string (provider-specific syntax)

        Returns:
            List of message IDs matching the query, ordered by relevance

        Raises:
            SearchError: If query is malformed or search fails
        """
        ...


@runtime_checkable
class VectorProvider(Protocol):
    """Vector search provider for semantic similarity.

    This protocol defines the interface for vector embedding and similarity search.
    Implementations handle embedding generation and vector database operations.

    Example implementations:
    - Qdrant + Voyage AI (polylogue.storage.index_qdrant)
    - pgvector (future)
    - Pinecone (future)
    """

    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None:
        """Upsert message embeddings into the vector store.

        This operation should be idempotent - repeated calls with the same
        messages should result in the same final state.

        Args:
            conversation_id: ID of the conversation containing these messages
            messages: Messages to embed and store. Must include message_id and text.

        Raises:
            VectorError: If embedding generation or storage fails
        """
        ...

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Find semantically similar messages.

        Args:
            text: Query text to search for
            limit: Maximum number of results to return

        Returns:
            List of (message_id, similarity_score) tuples, ordered by descending
            similarity score. Score interpretation is provider-specific.

        Raises:
            VectorError: If query fails or embedding generation fails
        """
        ...


@runtime_checkable
class StorageBackend(Protocol):
    """Storage backend for conversation and message records.

    This protocol defines the interface for persisting and retrieving
    conversation data. Implementations might use SQLite, PostgreSQL, or
    other databases.

    Example implementations:
    - SQLite backend (polylogue.storage.store)
    - PostgreSQL (future)
    """

    def save_conversation(self, record: ConversationRecord) -> None:
        """Persist a conversation record.

        This operation should be upsert semantics - if a conversation with
        the same conversation_id exists, it should be updated.

        Args:
            record: Conversation record to save

        Raises:
            DatabaseError: If save operation fails
        """
        ...

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            ConversationRecord if found, None otherwise

        Raises:
            DatabaseError: If retrieval fails
        """
        ...

    def save_messages(self, records: list[MessageRecord]) -> None:
        """Persist multiple message records.

        This operation should handle messages atomically - either all messages
        are saved or none are. Messages belonging to the same conversation_id
        should be treated as a transaction.

        Args:
            records: Message records to save

        Raises:
            DatabaseError: If save operation fails
        """
        ...

    def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Retrieve all messages for a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of MessageRecords, ordered by timestamp (if available)

        Raises:
            DatabaseError: If retrieval fails
        """
        ...

    def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records.

        Args:
            records: Attachment records to save

        Raises:
            DatabaseError: If save operation fails
        """
        ...

    def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        """Retrieve all attachments for a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of AttachmentRecords

        Raises:
            DatabaseError: If retrieval fails
        """
        ...


@runtime_checkable
class Renderer(Protocol):
    """Rendering engine for conversation output.

    This protocol defines the interface for converting Conversation objects
    into human-readable formats (Markdown, HTML).

    Example implementations:
    - Jinja2 renderer (polylogue.rendering.render)
    - Alternative templating engines (future)
    """

    def render_markdown(self, conversation: Conversation, output_path: Path) -> Path:
        """Render a conversation as Markdown.

        Args:
            conversation: Conversation object to render
            output_path: Directory where markdown file should be written

        Returns:
            Path to the generated markdown file

        Raises:
            RenderError: If rendering fails or output path is invalid
        """
        ...

    def render_html(self, conversation: Conversation, output_path: Path) -> Path:
        """Render a conversation as HTML.

        Args:
            conversation: Conversation object to render
            output_path: Directory where HTML file should be written

        Returns:
            Path to the generated HTML file

        Raises:
            RenderError: If rendering fails or output path is invalid
        """
        ...


__all__ = [
    "SearchProvider",
    "VectorProvider",
    "StorageBackend",
    "Renderer",
]

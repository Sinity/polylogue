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

from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord, RunRecord

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
    - SQLite backend (polylogue.storage.backends.sqlite)
    - PostgreSQL (future)
    - DuckDB (future)
    """

    def get_conversation(self, id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID.

        Args:
            id: Unique identifier for the conversation

        Returns:
            ConversationRecord if found, None otherwise

        Raises:
            DatabaseError: If retrieval fails
        """
        ...

    def list_conversations(
        self,
        source: str | None = None,
        provider: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ConversationRecord]:
        """List all conversations with optional filtering and pagination.

        Args:
            source: Optional source name filter (from provider_meta.source)
            provider: Optional provider name filter (e.g., 'claude', 'chatgpt')
            limit: Optional maximum number of results to return
            offset: Number of results to skip (for pagination)

        Returns:
            List of ConversationRecords, ordered by updated_at DESC

        Raises:
            DatabaseError: If retrieval fails
        """
        ...

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

    def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records.

        Args:
            records: Attachment records to save

        Raises:
            DatabaseError: If save operation fails
        """
        ...

    def prune_attachments(self, conversation_id: str, keep_attachment_ids: set[str]) -> None:
        """Remove attachment refs not in keep set and clean up orphaned attachments.

        This is used during re-ingestion to remove attachments that are no longer
        part of the conversation bundle.

        Args:
            conversation_id: The conversation to prune attachments for
            keep_attachment_ids: Set of attachment IDs to keep (prune all others)

        Raises:
            DatabaseError: If prune operation fails
        """
        ...

    def resolve_id(self, id_prefix: str) -> str | None:
        """Resolve a partial conversation ID to a full ID.

        Supports both exact matches and prefix matches. If multiple
        conversations match the prefix, returns None (ambiguous).

        Args:
            id_prefix: Full or partial conversation ID to resolve

        Returns:
            The full conversation ID if exactly one match found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        ...

    def search_conversations(self, query: str, limit: int = 100) -> list[str]:
        """Search conversations using full-text search.

        Args:
            query: Search query string (backend-specific syntax)
            limit: Maximum number of conversation IDs to return

        Returns:
            List of conversation IDs matching the query, ordered by relevance

        Raises:
            DatabaseError: If search index not available or query fails
        """
        ...

    def begin(self) -> None:
        """Begin a transaction or savepoint.

        Raises:
            DatabaseError: If transaction start fails
        """
        ...

    def commit(self) -> None:
        """Commit the current transaction.

        Raises:
            DatabaseError: If commit fails
        """
        ...

    def rollback(self) -> None:
        """Rollback the current transaction.

        Raises:
            DatabaseError: If rollback fails
        """
        ...

    def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run audit entry.

        Args:
            record: Run record containing execution metadata

        Raises:
            DatabaseError: If record save fails
        """
        ...

    # --- Metadata CRUD ---

    def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata dict for a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Metadata dictionary (empty dict if no metadata)
        """
        ...

    def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        """Set a single metadata key.

        Args:
            conversation_id: ID of the conversation
            key: Metadata key to set
            value: Value to set (must be JSON-serializable)
        """
        ...

    def delete_metadata(self, conversation_id: str, key: str) -> None:
        """Remove a metadata key.

        Args:
            conversation_id: ID of the conversation
            key: Metadata key to remove
        """
        ...

    def add_tag(self, conversation_id: str, tag: str) -> None:
        """Add a tag to the conversation's tags list.

        Args:
            conversation_id: ID of the conversation
            tag: Tag to add
        """
        ...

    def remove_tag(self, conversation_id: str, tag: str) -> None:
        """Remove a tag from the conversation's tags list.

        Args:
            conversation_id: ID of the conversation
            tag: Tag to remove
        """
        ...

    def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        """Replace entire metadata dict.

        Args:
            conversation_id: ID of the conversation
            metadata: New metadata dictionary
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


@runtime_checkable
class OutputRenderer(Protocol):
    """Protocol for pluggable output renderers.

    This protocol defines the interface for rendering conversations to various
    output formats. Implementations can provide format-specific rendering logic
    while maintaining a consistent API.

    Example implementations:
    - MarkdownRenderer (plain text markdown output)
    - HTMLRenderer (HTML with Jinja2 templates)
    - JSONRenderer (future)
    """

    def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to the output format.

        Args:
            conversation_id: ID of the conversation to render
            output_path: Directory where the rendered file should be written

        Returns:
            Path to the generated output file

        Raises:
            ValueError: If conversation not found
            IOError: If output path is invalid or write fails
        """
        ...

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            Format identifier (e.g., 'markdown', 'html', 'json')
        """
        ...


__all__ = [
    "SearchProvider",
    "VectorProvider",
    "StorageBackend",
    "Renderer",
    "OutputRenderer",
]

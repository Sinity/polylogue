"""Storage layer for Polylogue - database, indexing, and search."""

from __future__ import annotations

# Database module
from .db import (
    DatabaseError,
    connection_context,
    default_db_path,
    open_connection,
)

# Store module - records and operations
from .store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
    record_run,
    store_records,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)

# Repository module - encapsulated storage operations
from .repository import StorageRepository

# Backend abstraction - new storage backend interface
from .backends import (
    SQLiteBackend,
    create_backend,
)

# Index module - FTS5 indexing
from .index import (
    ensure_index,
    index_status,
    rebuild_index,
    update_index_for_conversations,
)

# Search module - FTS5 search
from .search import (
    SearchHit,
    SearchResult,
    escape_fts5_query,
    search_messages,
)

# Qdrant vector index module - lazy import to avoid loading heavy dependencies
def __getattr__(name: str):
    """Lazy import for Qdrant-related exports to avoid loading heavy dependencies."""
    if name in ("QdrantError", "VectorStore", "get_embeddings", "update_qdrant_for_conversations"):
        from . import index_qdrant
        return getattr(index_qdrant, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Database
    "DatabaseError",
    "connection_context",
    "default_db_path",
    "open_connection",
    # Store
    "AttachmentRecord",
    "ConversationRecord",
    "MessageRecord",
    "RunRecord",
    "record_run",
    "store_records",
    "upsert_attachment",
    "upsert_conversation",
    "upsert_message",
    # Repository
    "StorageRepository",
    # Backends
    "SQLiteBackend",
    "create_backend",
    # Index
    "ensure_index",
    "index_status",
    "rebuild_index",
    "update_index_for_conversations",
    # Search
    "SearchHit",
    "SearchResult",
    "escape_fts5_query",
    "search_messages",
    # Qdrant
    "QdrantError",
    "VectorStore",
    "get_embeddings",
    "update_qdrant_for_conversations",
]

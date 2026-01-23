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

# Qdrant vector index module
from .index_qdrant import (
    QdrantError,
    VectorStore,
    get_embeddings,
    update_qdrant_for_conversations,
)

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

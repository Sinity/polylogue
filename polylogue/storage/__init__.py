"""Storage layer for Polylogue - database, indexing, and search."""

from __future__ import annotations

from .backends import SQLiteBackend
from .repository import ConversationRepository
from .search import SearchResult
from .store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
)


# Qdrant vector index module - lazy import to avoid loading heavy dependencies
def __getattr__(name: str) -> object:
    """Lazy import for Qdrant-related exports to avoid loading heavy dependencies."""
    if name in ("QdrantError", "VectorStore", "get_embeddings", "update_qdrant_for_conversations"):
        from . import index_qdrant

        return getattr(index_qdrant, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttachmentRecord",
    "ConversationRecord",
    "ConversationRepository",
    "MessageRecord",
    "RunRecord",
    "SQLiteBackend",
    "SearchResult",
]

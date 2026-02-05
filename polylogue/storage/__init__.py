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


__all__ = [
    "AttachmentRecord",
    "ConversationRecord",
    "ConversationRepository",
    "MessageRecord",
    "RunRecord",
    "SQLiteBackend",
    "SearchResult",
]

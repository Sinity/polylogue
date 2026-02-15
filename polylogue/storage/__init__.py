"""Storage layer for Polylogue - database, indexing, and search.

Async exports (AsyncConversationRepository, AsyncSQLiteBackend) are available
via direct import from their submodules to avoid circular import issues::

    from polylogue.storage.async_repository import AsyncConversationRepository
    from polylogue.storage.backends import AsyncSQLiteBackend
"""

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

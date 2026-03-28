"""Storage layer for Polylogue - database, indexing, and search."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.backends import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search import SearchResult
    from polylogue.storage.store import (
        AttachmentRecord,
        ConversationRecord,
        MessageRecord,
        RunRecord,
    )


def __getattr__(name: str) -> object:
    lazy_exports = {
        "SQLiteBackend": ("polylogue.storage.backends", "SQLiteBackend"),
        "ConversationRepository": ("polylogue.storage.repository", "ConversationRepository"),
        "SearchResult": ("polylogue.storage.search", "SearchResult"),
        "AttachmentRecord": ("polylogue.storage.store", "AttachmentRecord"),
        "ConversationRecord": ("polylogue.storage.store", "ConversationRecord"),
        "MessageRecord": ("polylogue.storage.store", "MessageRecord"),
        "RunRecord": ("polylogue.storage.store", "RunRecord"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
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

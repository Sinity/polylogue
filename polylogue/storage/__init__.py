"""Storage layer for Polylogue - database, indexing, and search."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.runtime import (
        AttachmentRecord,
        MessageRecord,
        SessionRecord,
    )
    from polylogue.storage.search import SearchResult
    from polylogue.storage.sqlite import SQLiteBackend


def __getattr__(name: str) -> object:
    lazy_exports = {
        "SQLiteBackend": ("polylogue.storage.sqlite", "SQLiteBackend"),
        "SessionRepository": ("polylogue.storage.repository", "SessionRepository"),
        "SearchResult": ("polylogue.storage.search", "SearchResult"),
        "AttachmentRecord": ("polylogue.storage.runtime", "AttachmentRecord"),
        "SessionRecord": ("polylogue.storage.runtime", "SessionRecord"),
        "MessageRecord": ("polylogue.storage.runtime", "MessageRecord"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttachmentRecord",
    "SessionRecord",
    "SessionRepository",
    "MessageRecord",
    "SQLiteBackend",
    "SearchResult",
]

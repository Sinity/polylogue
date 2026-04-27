"""Archive-core public API for Polylogue.

The package root intentionally exposes the archive-facing contract:

- the async facade
- the sync facade
- core conversation/message model types
- query/search result types tied to archive operations

Higher-order semantic-analysis helpers remain public, but they now live behind
their precise modules under ``polylogue.lib`` rather than being implied as part
of the same root-level product surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.api import ArchiveStats, Polylogue
    from polylogue.api.sync import SyncPolylogue
    from polylogue.errors import PolylogueError
    from polylogue.lib.conversation_models import Conversation
    from polylogue.lib.message_models import Message
    from polylogue.storage.search import SearchResult


def __getattr__(name: str) -> object:
    lazy_exports = {
        "ArchiveStats": ("polylogue.api", "ArchiveStats"),
        "Conversation": ("polylogue.lib.conversation_models", "Conversation"),
        "Message": ("polylogue.lib.message_models", "Message"),
        "Polylogue": ("polylogue.api", "Polylogue"),
        "PolylogueError": ("polylogue.errors", "PolylogueError"),
        "SearchResult": ("polylogue.storage.search", "SearchResult"),
        "SyncPolylogue": ("polylogue.api.sync", "SyncPolylogue"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArchiveStats",
    "Conversation",
    "Message",
    "Polylogue",
    "PolylogueError",
    "SearchResult",
    "SyncPolylogue",
]

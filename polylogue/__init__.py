"""Archive-core public API for Polylogue.

The package root intentionally exposes the archive-facing contract:

- the async facade
- the sync facade
- core session/message model types
- query/search result types tied to archive operations

Higher-order semantic-analysis helpers remain public, but they live behind
their precise modules under ``polylogue.archive`` and related packages rather
than being implied as part of the same root-level insight surface.
"""

from __future__ import annotations

from polylogue import _sqlite_compat  # noqa: F401  # isort:skip -- must run before any sqlite3 import

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.api import ArchiveStats, Polylogue
    from polylogue.api.sync import SyncPolylogue
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import Session
    from polylogue.core.errors import PolylogueError
    from polylogue.storage.search import SearchResult


def __getattr__(name: str) -> object:
    lazy_submodules = {
        "api",
        "archive",
        "config",
        "daemon",
        "demo",
        "insights",
        "mcp",
        "operations",
        "pipeline",
        "rendering",
        "scenarios",
        "services",
        "sources",
        "storage",
        "ui",
    }
    if name in lazy_submodules:
        return importlib.import_module(f"polylogue.{name}")

    lazy_exports = {
        "ArchiveStats": ("polylogue.api", "ArchiveStats"),
        "Session": ("polylogue.archive.session.domain_models", "Session"),
        "Message": ("polylogue.archive.message.models", "Message"),
        "Polylogue": ("polylogue.api", "Polylogue"),
        "PolylogueError": ("polylogue.core.errors", "PolylogueError"),
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
    "Session",
    "Message",
    "Polylogue",
    "PolylogueError",
    "SearchResult",
    "SyncPolylogue",
]

"""Service factories — the honest replacement for DI container.

Module-level singletons for backend and repository, with a reset()
function for test isolation. Service construction (IngestionService,
IndexService, etc.) stays in the CLI commands where they're instantiated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.storage.backends.sqlite import SQLiteBackend, create_default_backend
from polylogue.storage.repository import ConversationRepository

if TYPE_CHECKING:
    from polylogue.config import Config

_backend: SQLiteBackend | None = None
_repository: ConversationRepository | None = None


def get_backend() -> SQLiteBackend:
    global _backend
    if _backend is None:
        _backend = create_default_backend()
    return _backend


def get_repository() -> ConversationRepository:
    global _repository
    if _repository is None:
        _repository = ConversationRepository(backend=get_backend())
    return _repository


def get_service_config() -> Config:
    """Return the application configuration."""
    from polylogue.config import get_config

    return get_config()


def reset() -> None:
    """Reset singletons. For tests."""
    global _backend, _repository
    _backend = None
    _repository = None


__all__ = [
    "get_backend",
    "get_repository",
    "get_service_config",
    "reset",
]

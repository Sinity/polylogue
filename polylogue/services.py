"""Service factories — the honest replacement for DI container.

Module-level singletons for backend and repository, with a reset()
function for test isolation. Service construction (ParsingService,
IndexService, etc.) stays in the CLI commands where they're instantiated.

Both sync and async singletons are provided:
- get_backend() / get_repository() for sync callers
- get_async_backend() / get_async_repository() for async callers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.storage.async_repository import AsyncConversationRepository
from polylogue.storage.backends.async_sqlite import AsyncSQLiteBackend
from polylogue.storage.backends.sqlite import SQLiteBackend, create_default_backend
from polylogue.storage.repository import ConversationRepository

if TYPE_CHECKING:
    from polylogue.config import Config

_backend: SQLiteBackend | None = None
_repository: ConversationRepository | None = None

_async_backend: AsyncSQLiteBackend | None = None
_async_repository: AsyncConversationRepository | None = None


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


def get_async_backend() -> AsyncSQLiteBackend:
    global _async_backend
    if _async_backend is None:
        _async_backend = AsyncSQLiteBackend()
    return _async_backend


def get_async_repository() -> AsyncConversationRepository:
    global _async_repository
    if _async_repository is None:
        _async_repository = AsyncConversationRepository(backend=get_async_backend())
    return _async_repository


def get_service_config() -> Config:
    """Return the application configuration."""
    from polylogue.config import get_config

    return get_config()


def reset() -> None:
    """Reset singletons. For tests.

    Closes any open backend connection before clearing references,
    preventing connection leaks across test boundaries.
    """
    global _backend, _repository
    if _backend is not None:
        _backend.close()
    _backend = None
    _repository = None


async def async_reset() -> None:
    """Reset async singletons. For async tests.

    Closes any open async backend connection before clearing references.
    """
    global _async_backend, _async_repository
    if _async_backend is not None:
        await _async_backend.close()
    _async_backend = None
    _async_repository = None


__all__ = [
    "get_backend",
    "get_repository",
    "get_async_backend",
    "get_async_repository",
    "get_service_config",
    "reset",
    "async_reset",
]

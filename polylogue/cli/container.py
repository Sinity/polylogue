"""CLI dependency injection container.

Factory functions for creating service instances using the centralized
dependency injection container. This provides a backward-compatible API
while delegating to the ApplicationContainer for actual instantiation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.config import Config

if TYPE_CHECKING:
    from polylogue.protocols import StorageBackend
from polylogue.container import ApplicationContainer, create_container
from polylogue.pipeline.services.indexing import IndexService
from polylogue.pipeline.services.ingestion import IngestionService
from polylogue.pipeline.services.rendering import RenderService
from polylogue.storage.repository import StorageRepository

# Module-level container instance (lazy-initialized)
_container = None


def get_container() -> ApplicationContainer:
    """Get or create the application container.

    Returns:
        ApplicationContainer instance (singleton).
    """
    global _container
    if _container is None:
        _container = create_container()
    return _container


def reset_container() -> None:
    """Reset the global container instance.

    Useful for testing to ensure a clean state between tests.
    """
    global _container
    _container = None


def create_config() -> Config:
    """Return the hardcoded configuration (zero-config).

    Returns:
        Loaded configuration instance with XDG defaults.
    """
    container = get_container()
    return container.config()


def create_storage_repository() -> StorageRepository:
    """Create storage repository for database operations.

    Returns:
        StorageRepository instance with its own write lock for thread-safe operations.
    """
    container = get_container()
    return container.storage()


def create_repository(config: Config | None = None) -> StorageBackend:
    """Get the storage backend for read operations.

    This is used by ConversationRepository for query operations.

    Args:
        config: Optional configuration (ignored, backend comes from container)

    Returns:
        StorageBackend instance for database queries
    """

    container = get_container()
    storage_repo = container.storage()
    # Access the underlying backend
    backend: StorageBackend = storage_repo._backend
    return backend


def create_ingestion_service(
    config: Config,
    repository: StorageRepository,
) -> IngestionService:
    """Create ingestion service with dependencies.

    Args:
        config: Application configuration (not used, provided by container)
        repository: Storage repository (not used, provided by container)

    Returns:
        IngestionService instance ready for use.
    """
    container = get_container()
    return container.ingestion_service()


def create_index_service(config: Config) -> IndexService:
    """Create index service for FTS5 and Qdrant operations.

    Args:
        config: Application configuration (not used, provided by container)

    Returns:
        IndexService instance ready for use.
    """
    container = get_container()
    return container.indexing_service()


def create_render_service(config: Config) -> RenderService:
    """Create render service for conversation rendering.

    Args:
        config: Application configuration (not used, provided by container)

    Returns:
        RenderService instance ready for use.
    """
    container = get_container()
    return container.rendering_service()

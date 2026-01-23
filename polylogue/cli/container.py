"""CLI dependency injection container.

Factory functions for creating service instances using the centralized
dependency injection container. This provides a backward-compatible API
while delegating to the ApplicationContainer for actual instantiation.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.config import Config
from polylogue.container import create_container
from polylogue.pipeline.services.indexing import IndexService
from polylogue.pipeline.services.ingestion import IngestionService
from polylogue.pipeline.services.rendering import RenderService
from polylogue.storage.repository import StorageRepository

# Module-level container instance (lazy-initialized)
_container = None


def get_container(config_path: Path | None = None):
    """Get or create the application container.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        ApplicationContainer instance (singleton).
    """
    global _container
    if _container is None:
        _container = create_container(config_path)
    return _container


def reset_container():
    """Reset the global container instance.

    Useful for testing to ensure a clean state between tests.
    """
    global _container
    _container = None


def create_config(config_path: Path | None = None) -> Config:
    """Create configuration from file.

    Args:
        config_path: Optional path to config file. If None, uses default location
                    (respects POLYLOGUE_CONFIG env var).

    Returns:
        Loaded configuration instance.

    Raises:
        ConfigError: If config file is missing or invalid.
    """
    container = get_container(config_path)
    return container.config()


def create_storage_repository() -> StorageRepository:
    """Create storage repository for database operations.

    Returns:
        StorageRepository instance with its own write lock for thread-safe operations.
    """
    container = get_container()
    return container.storage()


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

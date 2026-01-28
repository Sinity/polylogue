"""Dependency injection container for polylogue.

This module provides a centralized dependency injection container using the
dependency-injector framework. It replaces manual factory functions with
declarative service wiring.

Architecture:
- ConfigProvider: Singleton configuration (hardcoded XDG paths)
- StorageProvider: Singleton repository with thread-safe write lock
- ServiceProvider: Factory providers for services (ingestion, indexing, rendering)

Benefits:
- Explicit dependency graph
- Easier testing via container overrides
- Centralized service initialization
- Type-safe dependency resolution
"""

from __future__ import annotations

from dependency_injector import containers, providers

from polylogue.config import get_config
from polylogue.ingestion.drive_client import DriveClient
from polylogue.pipeline.services.indexing import IndexService
from polylogue.pipeline.services.ingestion import IngestionService
from polylogue.pipeline.services.rendering import RenderService
from polylogue.rendering.renderers import create_renderer
from polylogue.storage.backends.sqlite import create_default_backend
from polylogue.storage.repository import StorageRepository
from polylogue.storage.search_providers import create_vector_provider


class ApplicationContainer(containers.DeclarativeContainer):
    """Application-wide dependency injection container.

    Provides centralized configuration and service instantiation with
    proper dependency wiring.

    Usage:
        container = ApplicationContainer()

        # Access services
        config = container.config()
        repository = container.storage()
        ingestion = container.ingestion_service()
    """

    # Core providers (zero-config: hardcoded XDG paths)
    config = providers.Singleton(get_config)

    # Backend provider (thread-safe SQLite)
    backend = providers.Singleton(create_default_backend)

    storage = providers.Singleton(
        StorageRepository,
        backend=backend,
    )

    # Vector provider (Qdrant + Voyage AI, optional)
    vector_provider = providers.Singleton(
        create_vector_provider,
        config=config,
    )

    # Service factories
    drive_client = providers.Factory(
        DriveClient,
        config=config,
    )

    ingestion_service = providers.Factory(
        IngestionService,
        repository=storage,
        archive_root=config.provided.archive_root,
        config=config,
        drive_client_factory=drive_client.provider,
    )

    indexing_service = providers.Factory(
        IndexService,
        config=config,
        conn=None,  # Uses default connection
    )

    # Renderer provider (defaults to HTML)
    renderer = providers.Factory(
        create_renderer,
        format="html",
        config=config,
    )

    rendering_service = providers.Factory(
        RenderService,
        renderer=renderer,
        render_root=config.provided.render_root,
    )


def create_container() -> ApplicationContainer:
    """Create and configure the application container.

    Returns:
        Configured ApplicationContainer ready for use.

    Example:
        container = create_container()
        config = container.config()
        ingestion = container.ingestion_service()
    """
    return ApplicationContainer()


__all__ = [
    "ApplicationContainer",
    "create_container",
]

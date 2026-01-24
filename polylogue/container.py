"""Dependency injection container for polylogue.

This module provides a centralized dependency injection container using the
dependency-injector framework. It replaces manual factory functions with
declarative service wiring.

Architecture:
- ConfigProvider: Singleton configuration loaded once per application
- StorageProvider: Singleton repository with thread-safe write lock
- ServiceProvider: Factory providers for services (ingestion, indexing, rendering)

Benefits:
- Explicit dependency graph
- Easier testing via container overrides
- Centralized service initialization
- Type-safe dependency resolution
"""

from __future__ import annotations

from pathlib import Path

from dependency_injector import containers, providers

from polylogue.config import Config, load_config
from polylogue.pipeline.services.indexing import IndexService
from polylogue.pipeline.services.ingestion import IngestionService
from polylogue.pipeline.services.rendering import RenderService
from polylogue.rendering.renderers import create_renderer
from polylogue.storage.backends.sqlite import create_default_backend
from polylogue.storage.repository import StorageRepository


class ApplicationContainer(containers.DeclarativeContainer):
    """Application-wide dependency injection container.

    Provides centralized configuration and service instantiation with
    proper dependency wiring.

    Usage:
        container = ApplicationContainer()
        container.config.override(load_config())
        container.wire(modules=[...])

        # Access services
        config = container.config()
        repository = container.storage()
        ingestion = container.ingestion_service()
    """

    # Core providers
    config = providers.Singleton(
        load_config,
        path=None,
    )

    # Backend provider (thread-safe SQLite)
    backend = providers.Singleton(
        create_default_backend,
    )

    storage = providers.Singleton(
        StorageRepository,
        backend=backend,
    )

    # Service factories
    ingestion_service = providers.Factory(
        IngestionService,
        repository=storage,
        archive_root=config.provided.archive_root,
        config=config,
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


def create_container(config_path: Path | None = None) -> ApplicationContainer:
    """Create and configure the application container.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        Configured ApplicationContainer ready for use.

    Example:
        container = create_container()
        config = container.config()
        ingestion = container.ingestion_service()
    """
    container = ApplicationContainer()
    # Override config provider with custom path if provided
    if config_path is not None:
        container.config.override(providers.Singleton(load_config, path=config_path))
    return container


__all__ = [
    "ApplicationContainer",
    "create_container",
]

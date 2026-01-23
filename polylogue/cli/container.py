"""CLI dependency injection container.

Factory functions for creating service instances with proper dependency injection.
This enables better testability by decoupling CLI commands from direct instantiation.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.config import Config, load_config
from polylogue.storage.repository import StorageRepository


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
    return load_config(config_path)


def create_storage_repository() -> StorageRepository:
    """Create storage repository for database operations.

    Returns:
        StorageRepository instance with its own write lock for thread-safe operations.
    """
    return StorageRepository()


# Placeholder for future service layer - will be implemented when
# pipeline/services.py is created by another agent.
# For now, CLI commands continue to use pipeline.runner directly.

# def create_ingestion_service(
#     config: Config,
#     repository: StorageRepository,
# ) -> IngestionService:
#     """Create ingestion service with dependencies.
#
#     Args:
#         config: Application configuration
#         repository: Storage repository for persisting data
#
#     Returns:
#         IngestionService instance ready for use.
#     """
#     from polylogue.pipeline.services import IngestionService
#     return IngestionService(config=config, repository=repository)


# def create_index_service(config: Config) -> IndexService:
#     """Create index service for FTS5 and Qdrant operations.
#
#     Args:
#         config: Application configuration (for archive_root, etc.)
#
#     Returns:
#         IndexService instance ready for use.
#     """
#     from polylogue.pipeline.services import IndexService
#     return IndexService(config=config)


# def create_render_service(config: Config) -> RenderService:
#     """Create render service for conversation rendering.
#
#     Args:
#         config: Application configuration (for render_root, template_path, etc.)
#
#     Returns:
#         RenderService instance ready for use.
#     """
#     from polylogue.pipeline.services import RenderService
#     return RenderService(config=config)


# def create_pipeline_runner(config: Config) -> PipelineRunner:
#     """Create pipeline runner orchestrator with all services.
#
#     Args:
#         config: Application configuration
#
#     Returns:
#         PipelineRunner instance with all services wired up.
#     """
#     from polylogue.pipeline.services import PipelineRunner
#     repository = create_storage_repository()
#     ingestion = create_ingestion_service(config, repository)
#     indexing = create_index_service(config)
#     rendering = create_render_service(config)
#     return PipelineRunner(
#         config=config,
#         ingestion=ingestion,
#         indexing=indexing,
#         rendering=rendering,
#     )

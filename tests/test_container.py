"""Tests for dependency injection container."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.container import ApplicationContainer, create_container
from polylogue.pipeline.services.indexing import IndexService
from polylogue.pipeline.services.ingestion import IngestionService
from polylogue.pipeline.services.rendering import RenderService
from polylogue.storage.repository import StorageRepository


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Create a test configuration."""
    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    config_path = tmp_path / "config.json"

    archive_root.mkdir()
    render_root.mkdir()

    return Config(
        version=2,
        archive_root=archive_root,
        render_root=render_root,
        sources=[Source(name="test", path=tmp_path / "inbox")],
        path=config_path,
        template_path=None,
    )


class TestApplicationContainer:
    """Test suite for ApplicationContainer."""

    def test_container_creation(self):
        """Test basic container instantiation."""
        container = ApplicationContainer()
        assert container is not None
        assert hasattr(container, "config")
        assert hasattr(container, "storage")
        assert hasattr(container, "ingestion_service")
        assert hasattr(container, "indexing_service")
        assert hasattr(container, "rendering_service")

    def test_config_provider_singleton(self, test_config: Config):
        """Test that config provider is a singleton."""
        container = ApplicationContainer()
        container.config.override(test_config)

        config1 = container.config()
        config2 = container.config()

        # Same instance should be returned
        assert config1 is config2
        assert config1.version == 2

    def test_storage_provider_singleton(self):
        """Test that storage provider is a singleton."""
        container = ApplicationContainer()

        storage1 = container.storage()
        storage2 = container.storage()

        # Same instance should be returned
        assert storage1 is storage2
        assert isinstance(storage1, StorageRepository)
        assert hasattr(storage1, "_write_lock")

    def test_ingestion_service_factory(self, test_config: Config):
        """Test that ingestion service is created as a factory (new instance each time)."""
        container = ApplicationContainer()
        container.config.override(test_config)

        service1 = container.ingestion_service()
        service2 = container.ingestion_service()

        # Different instances should be returned
        assert service1 is not service2
        assert isinstance(service1, IngestionService)
        assert isinstance(service2, IngestionService)

        # But they should share the same repository (singleton)
        assert service1.repository is service2.repository

    def test_indexing_service_factory(self, test_config: Config):
        """Test that indexing service is created as a factory."""
        container = ApplicationContainer()
        container.config.override(test_config)

        service1 = container.indexing_service()
        service2 = container.indexing_service()

        # Different instances
        assert service1 is not service2
        assert isinstance(service1, IndexService)
        assert isinstance(service2, IndexService)

    def test_rendering_service_factory(self, test_config: Config):
        """Test that rendering service is created as a factory."""
        container = ApplicationContainer()
        container.config.override(test_config)

        service1 = container.rendering_service()
        service2 = container.rendering_service()

        # Different instances
        assert service1 is not service2
        assert isinstance(service1, RenderService)
        assert isinstance(service2, RenderService)

    def test_ingestion_service_dependencies(self, test_config: Config):
        """Test that ingestion service receives correct dependencies."""
        container = ApplicationContainer()
        container.config.override(test_config)

        service = container.ingestion_service()

        assert service.repository is not None
        assert isinstance(service.repository, StorageRepository)
        assert service.archive_root == test_config.archive_root
        assert service.config.version == test_config.version

    def test_indexing_service_dependencies(self, test_config: Config):
        """Test that indexing service receives correct dependencies."""
        container = ApplicationContainer()
        container.config.override(test_config)

        service = container.indexing_service()

        assert service.config.version == test_config.version
        assert service.conn is None  # Default connection

    def test_rendering_service_dependencies(self, test_config: Config):
        """Test that rendering service receives correct dependencies."""
        container = ApplicationContainer()
        container.config.override(test_config)

        service = container.rendering_service()

        assert service.template_path == test_config.template_path
        assert service.render_root == test_config.render_root
        assert service.archive_root == test_config.archive_root

    def test_container_override_config(self, test_config: Config):
        """Test that container can be overridden for testing."""
        container = ApplicationContainer()

        # Override config provider
        container.config.override(test_config)

        config = container.config()
        assert config is test_config

    def test_container_override_storage(self):
        """Test that storage provider can be overridden."""
        container = ApplicationContainer()

        mock_repository = MagicMock(spec=StorageRepository)
        container.storage.override(mock_repository)

        repository = container.storage()
        assert repository is mock_repository


class TestCreateContainer:
    """Test suite for create_container factory function."""

    def test_create_container_no_path(self, test_config: Config):
        """Test creating container without config path."""
        container = create_container()
        container.config.override(test_config)

        # Container should be created and have the expected providers
        assert hasattr(container, "config")
        config = container.config()
        assert config.version == test_config.version

    def test_create_container_with_path(self, tmp_path: Path, test_config: Config):
        """Test creating container with explicit config path."""
        config_path = tmp_path / "custom_config.json"

        with patch("polylogue.container.load_config", return_value=test_config):
            container = create_container(config_path)

            # Container should have config provider
            assert hasattr(container, "config")
            config = container.config()

            # Config should be loaded correctly
            assert config.version == test_config.version

    def test_create_container_services_work(self, test_config: Config):
        """Test that services created from container work correctly."""
        with patch("polylogue.container.load_config", return_value=test_config):
            container = create_container()

            # All services should be instantiable
            ingestion = container.ingestion_service()
            indexing = container.indexing_service()
            rendering = container.rendering_service()

            assert isinstance(ingestion, IngestionService)
            assert isinstance(indexing, IndexService)
            assert isinstance(rendering, RenderService)


class TestContainerIntegration:
    """Integration tests for the DI container."""

    def test_shared_dependencies(self, test_config: Config):
        """Test that services share singleton dependencies correctly."""
        with patch("polylogue.container.load_config", return_value=test_config):
            container = create_container()

            # Create multiple services
            ingestion1 = container.ingestion_service()
            ingestion2 = container.ingestion_service()

            # They should have the same repository instance
            assert ingestion1.repository is ingestion2.repository

            # And the same config instance
            assert ingestion1.config is ingestion2.config

    def test_config_propagation(self, test_config: Config):
        """Test that config changes propagate to all services."""
        with patch("polylogue.container.load_config", return_value=test_config):
            container = create_container()

            # Get config instance
            config = container.config()

            # Create services
            ingestion = container.ingestion_service()
            indexing = container.indexing_service()
            rendering = container.rendering_service()

            # All services should reference the same config
            assert ingestion.config is config
            assert indexing.config is config

            # Rendering service gets specific config attributes
            assert rendering.archive_root == config.archive_root
            assert rendering.render_root == config.render_root

    def test_repository_thread_safety(self, test_config: Config):
        """Test that repository has thread-safe write lock."""
        with patch("polylogue.container.load_config", return_value=test_config):
            container = create_container()

            repository = container.storage()

            # Repository should have a write lock for thread safety
            assert hasattr(repository, "_write_lock")
            assert repository._write_lock is not None

    def test_multiple_containers_isolated(self, test_config: Config, tmp_path: Path):
        """Test that multiple containers are properly isolated."""
        # Create different config instances
        config1 = test_config
        config2 = Config(
            version=2,
            archive_root=tmp_path / "archive2",
            render_root=tmp_path / "render2",
            sources=[Source(name="test2", path=tmp_path / "inbox2")],
            path=tmp_path / "config2.json",
        )

        container1 = create_container()
        container1.config.override(config1)

        container2 = create_container()
        container2.config.override(config2)

        # Different containers
        assert container1 is not container2

        # With different configs
        cfg1 = container1.config()
        cfg2 = container2.config()

        assert cfg1.sources[0].name == "test"
        assert cfg2.sources[0].name == "test2"

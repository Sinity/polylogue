"""Tests for async ingestion service."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.services.async_ingestion import AsyncIngestionService
from polylogue.storage.backends.sqlite import create_default_backend
from polylogue.storage.repository import StorageRepository


@pytest.mark.asyncio
async def test_async_ingestion_service_initialization():
    """Test async ingestion service initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        archive_root = Path(tmpdir) / "archive"
        archive_root.mkdir()

        # Create repository
        backend = create_default_backend(db_path=db_path)
        repository = StorageRepository(backend=backend)

        # Create config
        config = Config(
            archive_root=archive_root,
            render_root=archive_root / "render",
            sources=[],
        )

        # Initialize service
        service = AsyncIngestionService(repository, archive_root, config)
        assert service is not None


@pytest.mark.asyncio
async def test_async_ingestion_empty_sources():
    """Test async ingestion with empty source list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        archive_root = Path(tmpdir) / "archive"
        archive_root.mkdir()

        backend = create_default_backend(db_path=db_path)
        repository = StorageRepository(backend=backend)

        config = Config(
            archive_root=archive_root,
            render_root=archive_root / "render",
            sources=[],
        )

        service = AsyncIngestionService(repository, archive_root, config)

        # Ingest empty source list
        result = await service.ingest_sources([])

        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0


@pytest.mark.asyncio
async def test_async_ingestion_concurrent():
    """Test that async ingestion can be called concurrently."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        archive_root = Path(tmpdir) / "archive"
        archive_root.mkdir()

        backend = create_default_backend(db_path=db_path)
        repository = StorageRepository(backend=backend)

        config = Config(
            archive_root=archive_root,
            render_root=archive_root / "render",
            sources=[],
        )

        service = AsyncIngestionService(repository, archive_root, config)

        # Run multiple ingestions concurrently
        results = await asyncio.gather(
            service.ingest_sources([]),
            service.ingest_sources([]),
            service.ingest_sources([]),
        )

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert result.counts["conversations"] == 0

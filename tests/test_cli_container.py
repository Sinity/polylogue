"""Tests for CLI dependency injection container."""

from __future__ import annotations

from polylogue.storage.backends.sqlite import create_default_backend
from polylogue.storage.repository import StorageRepository


def test_create_storage_repository() -> None:
    """Test creating storage repository returns StorageRepository instance."""
    backend = create_default_backend()
    repository = StorageRepository(backend=backend)

    assert isinstance(repository, StorageRepository)
    assert hasattr(repository, "_write_lock")


def test_create_storage_repository_independent_instances() -> None:
    """Test that each call creates a new independent repository instance."""
    backend1 = create_default_backend()
    backend2 = create_default_backend()
    repo1 = StorageRepository(backend=backend1)
    repo2 = StorageRepository(backend=backend2)

    assert repo1 is not repo2
    assert repo1._write_lock is not repo2._write_lock

"""Tests for StorageRepository with backend abstraction."""

from __future__ import annotations

import pytest

from polylogue.storage.backends import SQLiteBackend
from polylogue.storage.repository import StorageRepository
from tests.helpers import make_attachment, make_conversation, make_message


# sqlite_backend fixture is in conftest.py


@pytest.fixture
def repository_with_backend(sqlite_backend: SQLiteBackend) -> StorageRepository:
    """Create a repository using the backend."""
    return StorageRepository(backend=sqlite_backend)


def test_repository_save_via_backend(repository_with_backend: StorageRepository) -> None:
    """Test saving conversation via backend abstraction."""
    conv = make_conversation("conv1", title="Test Conversation")
    msg = make_message("msg1", "conv1", text="Hello")

    counts = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[],
    )

    assert counts["conversations"] == 1
    assert counts["messages"] == 1
    assert counts["skipped_conversations"] == 0
    assert counts["skipped_messages"] == 0


def test_repository_deduplication_via_backend(repository_with_backend: StorageRepository) -> None:
    """Test that duplicate conversations are skipped when using backend."""
    conv = make_conversation("conv1", title="Test", created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z", content_hash="samehash")
    msg = make_message("msg1", "conv1", text="Hello")

    # First save
    counts1 = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[],
    )
    assert counts1["conversations"] == 1
    assert counts1["messages"] == 1

    # Second save with same hash (should skip)
    counts2 = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[],
    )
    assert counts2["conversations"] == 0
    assert counts2["skipped_conversations"] == 1
    assert counts2["messages"] == 0
    assert counts2["skipped_messages"] == 1


def test_repository_with_attachments_via_backend(repository_with_backend: StorageRepository) -> None:
    """Test saving conversation with attachments via backend."""
    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="Hello")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    counts = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[att],
    )

    assert counts["conversations"] == 1
    assert counts["messages"] == 1
    assert counts["attachments"] == 1

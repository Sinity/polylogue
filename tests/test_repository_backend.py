"""Tests for StorageRepository with backend abstraction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.storage.backends import SQLiteBackend
from polylogue.storage.repository import StorageRepository
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteBackend:
    """Create a SQLite backend for testing."""
    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(db_path=db_path)
    yield backend
    backend.close()


@pytest.fixture
def repository_with_backend(backend: SQLiteBackend) -> StorageRepository:
    """Create a repository using the backend."""
    return StorageRepository(backend=backend)


@pytest.fixture
def repository_legacy() -> StorageRepository:
    """Create a repository without backend (legacy mode)."""
    return StorageRepository()


def test_repository_save_via_backend(repository_with_backend: StorageRepository) -> None:
    """Test saving conversation via backend abstraction."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test Conversation",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash123",
        version=1,
    )

    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="Hello",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="msghash1",
        version=1,
    )

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
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="samehash",
        version=1,
    )

    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="Hello",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="msghash1",
        version=1,
    )

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
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash123",
        version=1,
    )

    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="Hello",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="msghash1",
        version=1,
    )

    att = AttachmentRecord(
        attachment_id="att1",
        conversation_id="conv1",
        message_id="msg1",
        mime_type="image/png",
        size_bytes=1024,
        path="/path/to/file.png",
    )

    counts = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[att],
    )

    assert counts["conversations"] == 1
    assert counts["messages"] == 1
    assert counts["attachments"] == 1


def test_repository_backend_vs_legacy_compatibility(
    tmp_path: Path,
    repository_with_backend: StorageRepository,
) -> None:
    """Verify that backend and legacy modes produce same behavior."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash="hash123",
        version=1,
    )

    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="Hello",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="msghash1",
        version=1,
    )

    # Save via backend
    counts_backend = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[],
    )

    # Save via legacy (using a different database)
    from polylogue.storage.db import open_connection

    legacy_db = tmp_path / "legacy.db"
    repository_legacy = StorageRepository()  # No backend

    with open_connection(legacy_db) as conn:
        counts_legacy = repository_legacy.save_conversation(
            conversation=conv,
            messages=[msg],
            attachments=[],
            conn=conn,
        )

    # Both should report same counts
    assert counts_backend == counts_legacy

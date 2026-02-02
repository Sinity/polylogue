"""Tests for raw_conversations storage functionality."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.store import RawConversationRecord


class TestRawConversationStorage:
    """Tests for RawConversationRecord storage in SQLiteBackend."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a SQLiteBackend with a temp database."""
        db_path = tmp_path / "test.db"
        return SQLiteBackend(db_path=db_path)

    def test_save_raw_conversation_new(self, backend: SQLiteBackend) -> None:
        """Saving a new raw conversation returns True."""
        record = RawConversationRecord(
            raw_id="abc123",
            provider_name="test-provider",
            source_path="/tmp/test.json",
            source_index=0,
            raw_content=b'{"test": "data"}',
            acquired_at=datetime.now(timezone.utc).isoformat(),
            file_mtime=None,
            parsed_conversation_id=None,
        )

        result = backend.save_raw_conversation(record)

        assert result is True

    def test_save_raw_conversation_duplicate(self, backend: SQLiteBackend) -> None:
        """Saving a duplicate raw_id returns False (INSERT OR IGNORE)."""
        record = RawConversationRecord(
            raw_id="abc123",
            provider_name="test-provider",
            source_path="/tmp/test.json",
            source_index=0,
            raw_content=b'{"test": "data"}',
            acquired_at=datetime.now(timezone.utc).isoformat(),
        )

        # First save succeeds
        assert backend.save_raw_conversation(record) is True

        # Second save is ignored (same raw_id)
        assert backend.save_raw_conversation(record) is False

    def test_get_raw_conversation(self, backend: SQLiteBackend) -> None:
        """Retrieve a saved raw conversation by ID."""
        # Create without parsed_conversation_id to avoid FK constraint
        original = RawConversationRecord(
            raw_id="xyz789",
            provider_name="chatgpt",
            source_path="/path/to/export.json",
            source_index=5,
            raw_content=b'{"id": "conv-123", "messages": []}',
            acquired_at="2026-02-02T12:00:00+00:00",
            file_mtime="2026-01-15T08:30:00+00:00",
            parsed_conversation_id=None,  # FK requires actual conversation to exist
        )

        backend.save_raw_conversation(original)
        retrieved = backend.get_raw_conversation("xyz789")

        assert retrieved is not None
        assert retrieved.raw_id == original.raw_id
        assert retrieved.provider_name == original.provider_name
        assert retrieved.source_path == original.source_path
        assert retrieved.source_index == original.source_index
        assert retrieved.raw_content == original.raw_content
        assert retrieved.acquired_at == original.acquired_at
        assert retrieved.file_mtime == original.file_mtime
        assert retrieved.parsed_conversation_id is None

    def test_get_raw_conversation_not_found(self, backend: SQLiteBackend) -> None:
        """Retrieving non-existent raw conversation returns None."""
        result = backend.get_raw_conversation("nonexistent")

        assert result is None

    def test_iter_raw_conversations(self, backend: SQLiteBackend) -> None:
        """Iterate over all raw conversations."""
        records = [
            RawConversationRecord(
                raw_id=f"raw-{i}",
                provider_name="test" if i < 2 else "other",
                source_path=f"/path/{i}.json",
                raw_content=b'{}',
                acquired_at=datetime.now(timezone.utc).isoformat(),
            )
            for i in range(5)
        ]

        for r in records:
            backend.save_raw_conversation(r)

        all_records = list(backend.iter_raw_conversations())
        assert len(all_records) == 5

    def test_iter_raw_conversations_by_provider(self, backend: SQLiteBackend) -> None:
        """Filter iteration by provider name."""
        records = [
            RawConversationRecord(
                raw_id=f"raw-{i}",
                provider_name="chatgpt" if i % 2 == 0 else "claude",
                source_path=f"/path/{i}.json",
                raw_content=b'{}',
                acquired_at=datetime.now(timezone.utc).isoformat(),
            )
            for i in range(6)
        ]

        for r in records:
            backend.save_raw_conversation(r)

        chatgpt_records = list(backend.iter_raw_conversations(provider="chatgpt"))
        assert len(chatgpt_records) == 3

        claude_records = list(backend.iter_raw_conversations(provider="claude"))
        assert len(claude_records) == 3

    def test_iter_raw_conversations_with_limit(self, backend: SQLiteBackend) -> None:
        """Limit the number of records returned."""
        for i in range(10):
            backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"raw-{i}",
                    provider_name="test",
                    source_path=f"/path/{i}.json",
                    raw_content=b'{}',
                    acquired_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        limited = list(backend.iter_raw_conversations(limit=3))
        assert len(limited) == 3

    def test_link_raw_to_parsed(self, backend: SQLiteBackend) -> None:
        """Link a raw conversation to its parsed counterpart."""
        from polylogue.storage.store import ConversationRecord

        # First create a real conversation to link to (FK constraint)
        conv = ConversationRecord(
            conversation_id="conv-link-test",
            provider_name="test",
            provider_conversation_id="test-123",
            content_hash="hash123",
        )
        backend.save_conversation(conv)

        record = RawConversationRecord(
            raw_id="link-test",
            provider_name="test",
            source_path="/test.json",
            raw_content=b'{}',
            acquired_at=datetime.now(timezone.utc).isoformat(),
            parsed_conversation_id=None,
        )

        backend.save_raw_conversation(record)

        # Initially no link
        retrieved = backend.get_raw_conversation("link-test")
        assert retrieved is not None
        assert retrieved.parsed_conversation_id is None

        # Link to parsed (using actual conversation ID)
        result = backend.link_raw_to_parsed("link-test", "conv-link-test")
        assert result is True

        # Verify link
        retrieved = backend.get_raw_conversation("link-test")
        assert retrieved is not None
        assert retrieved.parsed_conversation_id == "conv-link-test"

    def test_link_raw_to_parsed_not_found(self, backend: SQLiteBackend) -> None:
        """Linking non-existent raw conversation returns False."""
        from polylogue.storage.store import ConversationRecord

        # Create a conversation first (FK constraint)
        conv = ConversationRecord(
            conversation_id="conv-123",
            provider_name="test",
            provider_conversation_id="test-456",
            content_hash="hash789",
        )
        backend.save_conversation(conv)

        # Linking a non-existent raw conversation returns False
        result = backend.link_raw_to_parsed("nonexistent", "conv-123")
        assert result is False

    def test_get_raw_conversation_count(self, backend: SQLiteBackend) -> None:
        """Count raw conversations."""
        # Initially empty
        assert backend.get_raw_conversation_count() == 0

        # Add some records
        for i in range(5):
            backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"count-{i}",
                    provider_name="chatgpt" if i < 3 else "claude",
                    source_path=f"/path/{i}.json",
                    raw_content=b'{}',
                    acquired_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        # Total count
        assert backend.get_raw_conversation_count() == 5

        # Filtered count
        assert backend.get_raw_conversation_count(provider="chatgpt") == 3
        assert backend.get_raw_conversation_count(provider="claude") == 2
        assert backend.get_raw_conversation_count(provider="codex") == 0


class TestRawConversationRecordValidation:
    """Tests for RawConversationRecord Pydantic validation."""

    def test_valid_record(self) -> None:
        """Valid record passes validation."""
        record = RawConversationRecord(
            raw_id="valid-id",
            provider_name="chatgpt",
            source_path="/path/to/file.json",
            raw_content=b'{"test": true}',
            acquired_at="2026-02-02T12:00:00Z",
        )

        assert record.raw_id == "valid-id"
        assert record.provider_name == "chatgpt"

    def test_empty_raw_id_fails(self) -> None:
        """Empty raw_id fails validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            RawConversationRecord(
                raw_id="",
                provider_name="test",
                source_path="/test.json",
                raw_content=b'{}',
                acquired_at="2026-02-02T12:00:00Z",
            )

    def test_empty_provider_name_fails(self) -> None:
        """Empty provider_name fails validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            RawConversationRecord(
                raw_id="test-id",
                provider_name="",
                source_path="/test.json",
                raw_content=b'{}',
                acquired_at="2026-02-02T12:00:00Z",
            )

    def test_empty_raw_content_fails(self) -> None:
        """Empty raw_content fails validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            RawConversationRecord(
                raw_id="test-id",
                provider_name="test",
                source_path="/test.json",
                raw_content=b'',
                acquired_at="2026-02-02T12:00:00Z",
            )

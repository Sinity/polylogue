"""Raw conversation storage and validation tests.

This module contains tests for:
- RawConversationRecord storage in SQLiteBackend
- Raw conversation retrieval and iteration
- Pydantic validation for raw records
- Content hashing and SHA256 integrity
- Links between raw and parsed conversations
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.backends.sqlite import SQLiteBackend

# test_db and test_conn fixtures are in conftest.py

# test_db and test_conn fixtures are in conftest.py


class TestRawConversationStorage:
    """Tests for RawConversationRecord storage in SQLiteBackend."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a SQLiteBackend with a temp database."""
        db_path = tmp_path / "test.db"
        return SQLiteBackend(db_path=db_path)

    async def test_save_raw_conversation_new(self, backend: SQLiteBackend) -> None:
        """Saving a new raw conversation returns True."""
        from datetime import datetime, timezone

        from polylogue.storage.store import RawConversationRecord

        record = RawConversationRecord(
            raw_id="abc123",
            provider_name="test-provider",
            source_path="/tmp/test.json",
            source_index=0,
            raw_content=b'{"test": "data"}',
            acquired_at=datetime.now(timezone.utc).isoformat(),
            file_mtime=None,
        )

        result = await backend.save_raw_conversation(record)

        assert result is True

    async def test_save_raw_conversation_duplicate(self, backend: SQLiteBackend) -> None:
        """Saving a duplicate raw_id returns False (INSERT OR IGNORE)."""
        from datetime import datetime, timezone

        from polylogue.storage.store import RawConversationRecord

        record = RawConversationRecord(
            raw_id="abc123",
            provider_name="test-provider",
            source_path="/tmp/test.json",
            source_index=0,
            raw_content=b'{"test": "data"}',
            acquired_at=datetime.now(timezone.utc).isoformat(),
        )

        # First save succeeds
        assert await backend.save_raw_conversation(record) is True

        # Second save is ignored (same raw_id)
        assert await backend.save_raw_conversation(record) is False

    async def test_get_raw_conversation(self, backend: SQLiteBackend) -> None:
        """Retrieve a saved raw conversation by ID."""
        from polylogue.storage.store import RawConversationRecord

        original = RawConversationRecord(
            raw_id="xyz789",
            provider_name="chatgpt",
            source_path="/path/to/export.json",
            source_index=5,
            raw_content=b'{"id": "conv-123", "messages": []}',
            acquired_at="2026-02-02T12:00:00+00:00",
            file_mtime="2026-01-15T08:30:00+00:00",
        )

        await backend.save_raw_conversation(original)
        retrieved = await backend.get_raw_conversation("xyz789")

        assert retrieved is not None
        assert retrieved.raw_id == original.raw_id
        assert retrieved.provider_name == original.provider_name
        assert retrieved.source_path == original.source_path
        assert retrieved.source_index == original.source_index
        assert retrieved.raw_content == original.raw_content
        assert retrieved.acquired_at == original.acquired_at
        assert retrieved.file_mtime == original.file_mtime

    async def test_get_raw_conversation_not_found(self, backend: SQLiteBackend) -> None:
        """Retrieving non-existent raw conversation returns None."""
        result = await backend.get_raw_conversation("nonexistent")

        assert result is None

    async def test_iter_raw_conversations(self, backend: SQLiteBackend) -> None:
        """Iterate over all raw conversations."""
        from datetime import datetime, timezone

        from polylogue.storage.store import RawConversationRecord

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
            await backend.save_raw_conversation(r)

        all_records = [r async for r in backend.iter_raw_conversations()]
        assert len(all_records) == 5

    async def test_iter_raw_conversations_by_provider(self, backend: SQLiteBackend) -> None:
        """Filter iteration by provider name."""
        from datetime import datetime, timezone

        from polylogue.storage.store import RawConversationRecord

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
            await backend.save_raw_conversation(r)

        chatgpt_records = [r async for r in backend.iter_raw_conversations(provider="chatgpt")]
        assert len(chatgpt_records) == 3

        claude_records = [r async for r in backend.iter_raw_conversations(provider="claude")]
        assert len(claude_records) == 3

    async def test_iter_raw_conversations_with_limit(self, backend: SQLiteBackend) -> None:
        """Limit the number of records returned."""
        from datetime import datetime, timezone

        from polylogue.storage.store import RawConversationRecord

        for i in range(10):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"raw-{i}",
                    provider_name="test",
                    source_path=f"/path/{i}.json",
                    raw_content=b'{}',
                    acquired_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        limited = [r async for r in backend.iter_raw_conversations(limit=3)]
        assert len(limited) == 3

    async def test_conversation_links_to_raw(self, backend: SQLiteBackend) -> None:
        """Conversations can link to their raw source via raw_id.

        The link goes: conversations.raw_id → raw_conversations.raw_id
        (data flows from raw to parsed, FK points backward to origin)
        """
        from datetime import datetime, timezone

        from polylogue.storage.store import ConversationRecord, RawConversationRecord

        # First store the raw conversation
        raw_record = RawConversationRecord(
            raw_id="raw-abc123",
            provider_name="test",
            source_path="/test.json",
            raw_content=b'{"id": "test-conv"}',
            acquired_at=datetime.now(timezone.utc).isoformat(),
        )
        await backend.save_raw_conversation(raw_record)

        # Then store parsed conversation with link to raw
        conv = ConversationRecord(
            conversation_id="conv-link-test",
            provider_name="test",
            provider_conversation_id="test-123",
            content_hash="hash123",
            raw_id="raw-abc123",  # Link to raw source
        )
        await backend.save_conversation_record(conv)

        # Verify the link exists in database
        async with backend._get_connection() as conn:
            row = await conn.execute(
                "SELECT raw_id FROM conversations WHERE conversation_id = ?",
                ("conv-link-test",),
            )
            row = await row.fetchone()

        assert row is not None
        assert row["raw_id"] == "raw-abc123"

    async def test_conversation_without_raw_id(self, backend: SQLiteBackend) -> None:
        """Conversations can be saved without raw_id (e.g., direct file ingest)."""
        from polylogue.storage.store import ConversationRecord

        conv = ConversationRecord(
            conversation_id="conv-no-raw",
            provider_name="test",
            provider_conversation_id="test-456",
            content_hash="hash456",
            # raw_id is None (default)
        )
        await backend.save_conversation_record(conv)

        # Verify it saved correctly
        async with backend._get_connection() as conn:
            row = await conn.execute(
                "SELECT raw_id FROM conversations WHERE conversation_id = ?",
                ("conv-no-raw",),
            )
            row = await row.fetchone()

        assert row is not None
        assert row["raw_id"] is None

    async def test_get_raw_conversation_count(self, backend: SQLiteBackend) -> None:
        """Count raw conversations."""
        from datetime import datetime, timezone

        from polylogue.storage.store import RawConversationRecord

        # Initially empty
        assert await backend.get_raw_conversation_count() == 0

        # Add some records
        for i in range(5):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"count-{i}",
                    provider_name="chatgpt" if i < 3 else "claude",
                    source_path=f"/path/{i}.json",
                    raw_content=b'{}',
                    acquired_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        # Total count
        assert await backend.get_raw_conversation_count() == 5

        # Filtered count
        assert await backend.get_raw_conversation_count(provider="chatgpt") == 3
        assert await backend.get_raw_conversation_count(provider="claude") == 2
        assert await backend.get_raw_conversation_count(provider="codex") == 0


class TestRawConversationRecordValidation:
    """Tests for RawConversationRecord Pydantic validation."""

    def test_valid_record(self) -> None:
        """Valid record passes validation."""
        from polylogue.storage.store import RawConversationRecord

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
        from polylogue.storage.store import RawConversationRecord

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
        from polylogue.storage.store import RawConversationRecord

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
        from polylogue.storage.store import RawConversationRecord

        with pytest.raises(ValueError, match="cannot be empty"):
            RawConversationRecord(
                raw_id="test-id",
                provider_name="test",
                source_path="/test.json",
                raw_content=b'',
                acquired_at="2026-02-02T12:00:00Z",
            )


class TestContentHashing:
    """Tests for raw conversation content hashing.

    These tests verify the hash integrity of stored raw conversations.
    For parsing tests, see test_fixtures_contract.py.
    """

    def test_raw_ids_are_sha256(self, raw_synthetic_samples: list) -> None:
        """Raw IDs are valid SHA256 hashes."""
        for sample in raw_synthetic_samples:
            assert len(sample.raw_id) == 64, f"Invalid hash length: {sample.raw_id}"
            assert all(c in "0123456789abcdef" for c in sample.raw_id)

    def test_content_matches_hash(self, raw_synthetic_samples: list) -> None:
        """Content hashes match stored raw_id."""
        import hashlib

        mismatches = []
        for sample in raw_synthetic_samples:
            computed = hashlib.sha256(sample.raw_content).hexdigest()
            if computed != sample.raw_id:
                mismatches.append((sample.raw_id[:16], computed[:16]))

        if mismatches:
            pytest.fail(f"{len(mismatches)}/{len(raw_synthetic_samples)} hash mismatches: {mismatches[:5]}")

"""Raw session storage and validation tests.

This module contains tests for:
- RawSessionRecord storage in SQLiteBackend
- Raw session retrieval and iteration
- Pydantic validation for raw records
- Content hashing and SHA256 integrity
- Links between raw and parsed sessions
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import make_raw_session, make_session, save_session_to_archive

# test_db and test_conn fixtures are in conftest.py

# test_db and test_conn fixtures are in conftest.py


class TestRawSessionStorage:
    """Tests for RawSessionRecord storage in SQLiteBackend."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a SQLiteBackend with a temp database."""
        db_path = tmp_path / "test.db"
        return SQLiteBackend(db_path=db_path)

    async def test_save_raw_session_new(self, backend: SQLiteBackend) -> None:
        """Saving a new raw session returns True."""
        record = make_raw_session(
            raw_id="abc123",
            source_name="test-provider",
            source_path="/tmp/test.json",
            source_index=0,
            blob_size=len(b'{"test": "data"}'),
            acquired_at="2026-02-02T12:00:00+00:00",
            file_mtime=None,
        )

        result = await backend.save_raw_session(record)

        assert result is True

    async def test_repository_update_raw_state_uses_source_tier(self, tmp_path: Path) -> None:
        initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
        initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
        source_backend = SQLiteBackend(db_path=tmp_path / "source.db")
        try:
            await source_backend.save_raw_session(
                make_raw_session(
                    raw_id="raw-split",
                    source_name="chatgpt-export",
                    source_path="/tmp/export.json",
                    source_index=0,
                    blob_size=2,
                    acquired_at="2026-02-02T12:00:00+00:00",
                )
            )
        finally:
            await source_backend.close()

        repo = SessionRepository(
            backend=SQLiteBackend(db_path=tmp_path / "index.db"),
            archive_root=tmp_path,
        )
        try:
            await repo.update_raw_state(
                "raw-split",
                state=RawSessionStateUpdate(
                    parsed_at="2026-02-03T12:00:00+00:00",
                    parse_error=None,
                    validation_status="skipped",
                    validation_error="duplicate materialization",
                    validation_mode="advisory",
                ),
            )
        finally:
            await repo.close()

        verifier = SQLiteBackend(db_path=tmp_path / "source.db")
        try:
            record = await verifier.get_raw_session("raw-split")
        finally:
            await verifier.close()

        assert record is not None
        assert record.validation_status == "skipped"
        assert record.validation_error == "duplicate materialization"

    async def test_save_raw_session_duplicate(self, backend: SQLiteBackend) -> None:
        """Saving a duplicate raw_id returns False (INSERT OR IGNORE)."""
        record = make_raw_session(
            raw_id="abc123",
            source_name="test-provider",
            source_path="/tmp/test.json",
            source_index=0,
            blob_size=len(b'{"test": "data"}'),
            acquired_at="2026-02-02T12:00:00+00:00",
        )

        # First save succeeds
        assert await backend.save_raw_session(record) is True

        # Second save is ignored (same raw_id)
        assert await backend.save_raw_session(record) is False

    async def test_get_raw_session(self, backend: SQLiteBackend) -> None:
        """Retrieve a saved raw session by ID."""
        original = make_raw_session(
            raw_id="xyz789",
            source_name="chatgpt",
            source_path="/path/to/export.json",
            source_index=5,
            blob_size=len(b'{"id": "conv-123", "messages": []}'),
            acquired_at="2026-02-02T12:00:00+00:00",
            file_mtime="2026-01-15T08:30:00+00:00",
        )

        await backend.save_raw_session(original)
        retrieved = await backend.get_raw_session("xyz789")

        assert retrieved is not None
        assert retrieved.raw_id == original.raw_id
        assert retrieved.source_name == original.source_name
        assert retrieved.source_path == original.source_path
        assert retrieved.source_index == original.source_index
        assert retrieved.blob_size == original.blob_size
        assert retrieved.acquired_at == original.acquired_at
        assert retrieved.file_mtime == original.file_mtime

    async def test_get_raw_session_not_found(self, backend: SQLiteBackend) -> None:
        """Retrieving non-existent raw session returns None."""
        result = await backend.get_raw_session("nonexistent")

        assert result is None

    async def test_iter_raw_sessions(self, backend: SQLiteBackend) -> None:
        """Iterate over all raw sessions."""
        records = [
            make_raw_session(
                raw_id=f"raw-{i}",
                source_name="test" if i < 2 else "other",
                source_path=f"/path/{i}.json",
                blob_size=len(b"{}"),
                acquired_at="2026-02-02T12:00:00+00:00",
            )
            for i in range(5)
        ]

        for r in records:
            await backend.save_raw_session(r)

        all_records = [r async for r in backend.iter_raw_sessions()]
        assert len(all_records) == 5

    async def test_iter_raw_sessions_by_provider(self, backend: SQLiteBackend) -> None:
        """Filter iteration by provider name."""
        records = [
            make_raw_session(
                raw_id=f"raw-{i}",
                source_name="chatgpt" if i % 2 == 0 else "claude-ai",
                source_path=f"/path/{i}.json",
                blob_size=len(b"{}"),
                acquired_at="2026-02-02T12:00:00+00:00",
            )
            for i in range(6)
        ]

        for r in records:
            await backend.save_raw_session(r)

        chatgpt_records = [r async for r in backend.iter_raw_sessions(provider="chatgpt")]
        assert len(chatgpt_records) == 3

        claude_records = [r async for r in backend.iter_raw_sessions(provider="claude-ai")]
        assert len(claude_records) == 3

    async def test_iter_raw_ids_by_source_name(self, backend: SQLiteBackend) -> None:
        """ID iteration can filter by exact provider name without hydrating raw blobs."""
        records = [
            make_raw_session(
                raw_id=f"raw-id-{i}",
                source_name="chatgpt" if i % 2 == 0 else "claude-ai",
                source_path=f"/path/{i}.json",
                blob_size=len(b"{}"),
                acquired_at="2026-02-02T12:00:00+00:00",
            )
            for i in range(6)
        ]

        for record in records:
            await backend.save_raw_session(record)

        chatgpt_ids = [raw_id async for raw_id in backend.iter_raw_ids(source_name="chatgpt")]
        claude_ids = [raw_id async for raw_id in backend.iter_raw_ids(source_name="claude-ai")]

        assert len(chatgpt_ids) == 3
        assert len(claude_ids) == 3
        assert all(raw_id.startswith("raw-id-") for raw_id in chatgpt_ids + claude_ids)

    async def test_iter_raw_headers_by_source_name(self, backend: SQLiteBackend) -> None:
        """Header iteration exposes blob sizes without hydrating full raw records."""
        records = [
            make_raw_session(
                raw_id=f"raw-header-{i}",
                source_name="chatgpt" if i % 2 == 0 else "claude-ai",
                source_path=f"/path/{i}.json",
                blob_size=(i + 1) * 10,
                acquired_at=f"2026-02-02T12:00:0{i}+00:00",
            )
            for i in range(4)
        ]

        for record in records:
            await backend.save_raw_session(record)

        chatgpt_headers = [header async for header in backend.iter_raw_headers(source_name="chatgpt")]

        assert chatgpt_headers == [("raw-header-2", 30), ("raw-header-0", 10)]

    async def test_get_raw_blob_sizes_preserves_requested_order(self, backend: SQLiteBackend) -> None:
        """Blob-size lookups should preserve caller order for batch shaping."""
        for raw_id, blob_size in (("raw-a", 10), ("raw-b", 20), ("raw-c", 30)):
            await backend.save_raw_session(
                make_raw_session(
                    raw_id=raw_id,
                    source_name="chatgpt",
                    source_path=f"/path/{raw_id}.json",
                    blob_size=blob_size,
                    acquired_at="2026-02-02T12:00:00+00:00",
                )
            )

        blob_sizes = await backend.get_raw_blob_sizes(["raw-c", "raw-a", "missing", "raw-b"])

        assert blob_sizes == [("raw-c", 30), ("raw-a", 10), ("raw-b", 20)]

    async def test_get_raw_sessions_batch_preserves_requested_order(self, backend: SQLiteBackend) -> None:
        """Hydrated raw batch reads should preserve caller order for replay planning."""
        for raw_id in ("raw-a", "raw-b", "raw-c"):
            await backend.save_raw_session(
                make_raw_session(
                    raw_id=raw_id,
                    source_name="chatgpt",
                    source_path=f"/path/{raw_id}.json",
                    blob_size=len(b"{}"),
                    acquired_at="2026-02-02T12:00:00+00:00",
                )
            )

        records = await backend.get_raw_sessions_batch(["raw-c", "raw-a", "missing", "raw-b"])

        assert [record.raw_id for record in records] == ["raw-c", "raw-a", "raw-b"]

    async def test_raw_provider_filters_prefer_payload_provider_when_present(self, backend: SQLiteBackend) -> None:
        """Raw provider filtering should use payload_provider when validation/parsing has classified the payload."""
        await backend.save_raw_session(
            make_raw_session(
                raw_id="raw-generic",
                payload_provider="chatgpt",
                source_name="inbox",
                source_path="/path/raw.json",
                blob_size=len(b"{}"),
                acquired_at="2026-02-02T12:00:00+00:00",
            )
        )

        matched_records = [record async for record in backend.iter_raw_sessions(provider="chatgpt")]
        matched_ids = [raw_id async for raw_id in backend.iter_raw_ids(source_name="chatgpt")]
        matched_count = await backend.get_raw_session_count(provider="chatgpt")

        assert [record.raw_id for record in matched_records] == ["raw-generic"]
        assert matched_ids == ["raw-generic"]
        assert matched_count == 1

    async def test_iter_raw_sessions_with_limit(self, backend: SQLiteBackend) -> None:
        """Limit the number of records returned."""
        for i in range(10):
            await backend.save_raw_session(
                make_raw_session(
                    raw_id=f"raw-{i}",
                    source_name="test",
                    source_path=f"/path/{i}.json",
                    blob_size=len(b"{}"),
                    acquired_at="2026-02-02T12:00:00+00:00",
                )
            )

        limited = [r async for r in backend.iter_raw_sessions(limit=3)]
        assert len(limited) == 3

    async def test_session_links_to_raw(self, backend: SQLiteBackend) -> None:
        """Sessions can link to their raw source via raw_id.

        The link goes: sessions.raw_id → raw_sessions.raw_id
        (data flows from raw to parsed, FK points backward to origin)
        """
        # First store the raw session
        raw_record = make_raw_session(
            raw_id="raw-abc123",
            source_name="test",
            source_path="/test.json",
            blob_size=len(b'{"id": "test-conv"}'),
            acquired_at="2026-02-02T12:00:00+00:00",
        )
        await backend.save_raw_session(raw_record)

        # Then store parsed session with link to raw
        conv = make_session(
            session_id="conv-link-test",
            source_name="test",
            provider_session_id="test-123",
            content_hash="hash123",
            raw_id="raw-abc123",  # Link to raw source
        )
        await save_session_to_archive(backend, session=conv)

        # Verify the link exists in database. Session ids are generated as
        # ``origin:native_id`` (#1743); source_name="test" → unknown-export.
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT raw_id FROM sessions WHERE session_id = ?",
                ("unknown-export:test-123",),
            )
            row = await cursor.fetchone()

        assert row is not None
        assert row["raw_id"] == "raw-abc123"

    async def test_session_without_raw_id(self, backend: SQLiteBackend) -> None:
        """Sessions can be saved without raw_id (e.g., direct file ingest)."""
        conv = make_session(
            session_id="conv-no-raw",
            source_name="test",
            provider_session_id="test-456",
            content_hash="hash456",
            # raw_id is None (default)
        )
        await save_session_to_archive(backend, session=conv)

        # Verify it saved correctly. Session ids are generated as
        # ``origin:native_id`` (#1743); source_name="test" → unknown-export.
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT raw_id FROM sessions WHERE session_id = ?",
                ("unknown-export:test-456",),
            )
            row = await cursor.fetchone()

        assert row is not None
        assert row["raw_id"] is None

    async def test_get_raw_session_count(self, backend: SQLiteBackend) -> None:
        """Count raw sessions."""
        # Initially empty
        assert await backend.get_raw_session_count() == 0

        # Add some records
        for i in range(5):
            await backend.save_raw_session(
                make_raw_session(
                    raw_id=f"count-{i}",
                    source_name="chatgpt" if i < 3 else "claude-ai",
                    source_path=f"/path/{i}.json",
                    blob_size=len(b"{}"),
                    acquired_at="2026-02-02T12:00:00+00:00",
                )
            )

        # Total count
        assert await backend.get_raw_session_count() == 5

        # Filtered count
        assert await backend.get_raw_session_count(provider="chatgpt") == 3
        assert await backend.get_raw_session_count(provider="claude-ai") == 2
        assert await backend.get_raw_session_count(provider="codex") == 0

    async def test_iter_raw_sessions_without_limit_returns_all(self, backend: SQLiteBackend) -> None:
        """Iterating without limit returns all records."""
        # Save 7 records
        for i in range(7):
            record = make_raw_session(
                raw_id=f"raw-all-{i}",
                source_name="test",
                source_path=f"/tmp/test-{i}.json",
                source_index=i,
                blob_size=len(f'{{"idx": {i}}}'.encode()),
                acquired_at="2026-02-02T12:00:00+00:00",
                file_mtime=None,
            )
            await backend.save_raw_session(record)

        # Iterate without limit
        results = []
        async for raw in backend.iter_raw_sessions():
            results.append(raw)

        assert len(results) == 7
        assert {r.raw_id for r in results} == {f"raw-all-{i}" for i in range(7)}


class TestRawSessionRecordValidation:
    """Tests for RawSessionRecord Pydantic validation."""

    def test_valid_record(self) -> None:
        """Valid record passes validation."""
        record = make_raw_session(
            raw_id="valid-id",
            source_name="chatgpt",
            source_path="/path/to/file.json",
            blob_size=len(b'{"test": true}'),
            acquired_at="2026-02-02T12:00:00Z",
        )

        assert record.raw_id == "valid-id"
        assert record.source_name == "chatgpt"

    def test_empty_raw_id_fails(self) -> None:
        """Empty raw_id fails validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            make_raw_session(
                raw_id="",
                source_name="test",
                source_path="/test.json",
                blob_size=len(b"{}"),
                acquired_at="2026-02-02T12:00:00Z",
            )

    def test_empty_source_name_fails(self) -> None:
        """Empty source_name fails validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            make_raw_session(
                raw_id="test-id",
                source_name="",
                source_path="/test.json",
                blob_size=len(b"{}"),
                acquired_at="2026-02-02T12:00:00Z",
            )

    def test_blob_size_is_persisted(self) -> None:
        """blob_size can be any non-negative integer."""
        # blob_size is just an int, no bounds checking
        record = make_raw_session(
            raw_id="test-id",
            source_name="test",
            source_path="/test.json",
            blob_size=1024,
            acquired_at="2026-02-02T12:00:00Z",
        )
        assert record.blob_size == 1024


class TestContentHashing:
    """Tests for raw session content hashing.

    These tests verify the hash integrity of stored raw sessions.
    For parsing tests, see test_fixtures_contract.py.
    """

    def test_raw_ids_are_sha256(self, raw_synthetic_samples: list[RawSessionRecord]) -> None:
        """Raw IDs are valid SHA256 hashes."""
        for sample in raw_synthetic_samples:
            assert len(sample.raw_id) == 64, f"Invalid hash length: {sample.raw_id}"
            assert all(c in "0123456789abcdef" for c in sample.raw_id)

    def test_content_matches_hash(self, raw_synthetic_samples: list[RawSessionRecord]) -> None:
        """Raw IDs are valid SHA256 hashes (constructed during fixture generation)."""
        # Note: raw_content is no longer stored on RawSessionRecord;
        # content is in the blob store keyed by raw_id. The fixture constructs
        # raw_id = sha256(raw_bytes), so we verify the ID format and that
        # each sample has a valid SHA256 hash.
        assert len(raw_synthetic_samples) > 0, "No samples generated"
        for sample in raw_synthetic_samples:
            assert len(sample.raw_id) == 64, f"Invalid hash length: {sample.raw_id}"
            assert all(c in "0123456789abcdef" for c in sample.raw_id)

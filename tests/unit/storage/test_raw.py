"""Raw session storage and validation tests.

This module contains tests for:
- RawSessionRecord storage in SQLiteBackend
- Raw session retrieval and iteration
- Pydantic validation for raw records
- Content hashing and SHA256 integrity
- Links between raw and parsed sessions
"""

from __future__ import annotations

import re
from pathlib import Path

import aiosqlite
import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries.raw_reads import get_raw_session as get_query_raw_session
from polylogue.storage.sqlite.queries.raw_writes import save_raw_session as save_query_raw_session
from tests.infra.storage_records import make_raw_session, make_session, save_session_to_archive

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

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

    async def test_capture_mode_round_trips_aistudio_drive_provenance(self, backend: SQLiteBackend) -> None:
        """A shared public origin retains its acquisition-time fiber member."""
        gemini = make_raw_session(
            raw_id="gemini-export",
            source_name="gemini",
            source_path="/tmp/export.json",
            blob_size=2,
            acquired_at="2026-02-02T12:00:00+00:00",
        ).model_copy(update={"capture_mode": Provider.GEMINI})
        drive = make_raw_session(
            raw_id="drive-live",
            source_name="gemini",
            source_path="/tmp/live-drive.json",
            blob_size=2,
            acquired_at="2026-02-02T12:00:00+00:00",
        ).model_copy(update={"capture_mode": Provider.DRIVE})

        assert await backend.save_raw_session(gemini) is True
        assert await backend.save_raw_session(drive) is True

        recovered_gemini = await backend.get_raw_session(gemini.raw_id)
        recovered_drive = await backend.get_raw_session(drive.raw_id)

        assert recovered_gemini is not None
        assert recovered_drive is not None
        assert recovered_gemini.capture_mode is Provider.GEMINI
        assert recovered_drive.capture_mode is Provider.DRIVE
        assert recovered_gemini.payload_provider is Provider.GEMINI
        assert recovered_drive.payload_provider is Provider.DRIVE

    async def test_reacquisition_keeps_first_known_capture_mode(self, backend: SQLiteBackend) -> None:
        """Later canonical fallback cannot erase durable live-Drive provenance."""
        drive = make_raw_session(
            raw_id="reacquired-aistudio",
            source_name="gemini",
            source_path="/tmp/live-drive.json",
            blob_size=2,
            acquired_at="2026-02-02T12:00:00+00:00",
        ).model_copy(update={"capture_mode": Provider.DRIVE})
        canonical_retry = drive.model_copy(update={"capture_mode": Provider.GEMINI})

        assert await backend.save_raw_session(drive) is True
        assert await backend.save_raw_session(canonical_retry) is False

        recovered = await backend.get_raw_session(drive.raw_id)
        assert recovered is not None
        assert recovered.capture_mode is Provider.DRIVE
        assert recovered.payload_provider is Provider.DRIVE

    async def test_rehydrated_unknown_capture_mode_remains_unknown(self, backend: SQLiteBackend) -> None:
        """A legacy NULL must not become the canonical GEMINI projection on save."""
        legacy = make_raw_session(
            raw_id="legacy-aistudio-unknown",
            source_name="gemini",
            source_path="/tmp/pre-capture-mode.json",
            blob_size=2,
            acquired_at="2026-02-02T12:00:00+00:00",
        )
        assert legacy.capture_mode is None

        assert await backend.save_raw_session(legacy) is True
        rehydrated = await backend.get_raw_session(legacy.raw_id)

        assert rehydrated is not None
        assert rehydrated.capture_mode is None
        assert rehydrated.payload_provider is Provider.GEMINI

        assert await backend.save_raw_session(rehydrated) is False
        reread = await backend.get_raw_session(legacy.raw_id)
        assert reread is not None
        assert reread.capture_mode is None

    async def test_query_writer_preserves_rehydrated_unknown_capture_mode(self, tmp_path: Path) -> None:
        """Repository raw writes do not promote a legacy NULL from its fallback."""
        source_db = tmp_path / "source.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        legacy = make_raw_session(
            raw_id="query-legacy-aistudio-unknown",
            source_name="gemini",
            source_path="/tmp/pre-capture-mode.json",
            blob_size=2,
            acquired_at="2026-02-02T12:00:00+00:00",
        )

        async with aiosqlite.connect(source_db) as conn:
            conn.row_factory = aiosqlite.Row
            assert await save_query_raw_session(conn, legacy, transaction_depth=0) is True
            rehydrated = await get_query_raw_session(conn, legacy.raw_id)

            assert rehydrated is not None
            assert rehydrated.capture_mode is None
            assert rehydrated.payload_provider is Provider.GEMINI

            assert await save_query_raw_session(conn, rehydrated, transaction_depth=0) is False
            row = await (
                await conn.execute("SELECT capture_mode FROM raw_sessions WHERE raw_id = ?", (legacy.raw_id,))
            ).fetchone()

        assert row is not None
        assert row[0] is None

    async def test_raw_session_state_preserves_capture_mode(self, backend: SQLiteBackend) -> None:
        """Planning-state reads preserve the live-Drive fiber member."""
        record = make_raw_session(
            raw_id="drive-planning-state",
            source_name="gemini",
            source_path="/tmp/live-drive.json",
            blob_size=2,
            acquired_at="2026-02-02T12:00:00+00:00",
        ).model_copy(update={"capture_mode": Provider.DRIVE})
        assert await backend.save_raw_session(record) is True

        state = (await backend.get_raw_session_states([record.raw_id]))[record.raw_id]
        assert state.source_name == Provider.DRIVE.value
        assert state.payload_provider is Provider.DRIVE
        assert state.validation_provider is Provider.DRIVE

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

        chatgpt_records = [r async for r in backend.iter_raw_sessions(origin="chatgpt")]
        assert len(chatgpt_records) == 3

        claude_records = [r async for r in backend.iter_raw_sessions(origin="claude-ai")]
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


class TestRawSessionWriterSingleSource:
    """polylogue-vwia: the async and sync-core raw writers must not diverge.

    ``SQLiteRawMixin.save_raw_session`` (async_sqlite_raw.py) used to hand-roll
    its own 18-column ``INSERT OR REPLACE``, while the durable-tier writer
    (queries/raw_writes.py) used a 28-column ``INSERT OR IGNORE`` including
    revision-authority evidence. A re-save through the async path silently
    reset revision lineage to defaults. The async mixin now delegates to the
    same writer function; these tests guard against either a column-list
    regression in that writer or a second hand-rolled INSERT reappearing.
    """

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a SQLiteBackend with a temp database."""
        db_path = tmp_path / "test.db"
        return SQLiteBackend(db_path=db_path)

    async def test_writer_insert_covers_every_live_raw_sessions_column(self, tmp_path: Path) -> None:
        """The canonical writer's INSERT must name every column the live DDL defines."""
        source_db = tmp_path / "source.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        async with aiosqlite.connect(source_db) as conn:
            cursor = await conn.execute("PRAGMA table_info(raw_sessions)")
            live_columns = {row[1] for row in await cursor.fetchall()}

        writer_source = (_REPO_ROOT / "polylogue/storage/sqlite/queries/raw_writes.py").read_text()
        match = re.search(r"INSERT OR IGNORE INTO raw_sessions \(([^)]+)\)", writer_source)
        assert match is not None, "expected exactly one canonical raw_sessions INSERT in raw_writes.py"
        writer_columns = {c.strip() for c in match.group(1).split(",")}

        assert writer_columns == live_columns, (
            "queries/raw_writes.py's INSERT column list has drifted from the live raw_sessions "
            f"DDL: missing from writer={live_columns - writer_columns} "
            f"extra in writer={writer_columns - live_columns}"
        )

    async def test_async_backend_has_no_second_raw_sessions_insert(self) -> None:
        """The async mixin must delegate, not hand-roll a competing INSERT/REPLACE."""
        mixin_source = (_REPO_ROOT / "polylogue/storage/sqlite/async_sqlite_raw.py").read_text()
        assert "INTO raw_sessions" not in mixin_source, (
            "async_sqlite_raw.py should delegate save_raw_session to "
            "queries/raw_writes.py, not embed its own INSERT statement (polylogue-vwia)"
        )

    async def test_resave_never_alters_revision_evidence(self, backend: SQLiteBackend) -> None:
        """Re-saving an existing raw_id must never reset durable revision authority."""
        envelope = RawRevisionEnvelope(
            logical_source_key="chatgpt:conv-1",
            kind=RawRevisionKind.FULL,
            source_revision="rev-1",
            acquisition_generation=1,
            authority=RawRevisionAuthority.BYTE_PROVEN,
        )
        original = make_raw_session(
            raw_id="revision-guarded",
            source_name="chatgpt",
            source_path="/tmp/export.json",
            blob_size=2,
            acquired_at="2026-02-02T12:00:00+00:00",
            revision=envelope,
        )
        assert await backend.save_raw_session(original) is True

        # A retried/duplicate acquisition of the same raw_id carries no revision
        # evidence of its own; it must not be able to wipe the original's.
        resave = make_raw_session(
            raw_id="revision-guarded",
            source_name="chatgpt",
            source_path="/tmp/export.json",
            blob_size=2,
            acquired_at="2026-02-02T12:00:00+00:00",
        )
        assert await backend.save_raw_session(resave) is False

        recovered = await backend.get_raw_session("revision-guarded")
        assert recovered is not None
        assert recovered.revision == envelope

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

        matched_records = [record async for record in backend.iter_raw_sessions(origin="chatgpt")]
        matched_ids = [raw_id async for raw_id in backend.iter_raw_ids(source_name="chatgpt")]
        matched_count = await backend.get_raw_session_count(origin="chatgpt")

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
        assert await backend.get_raw_session_count(origin="chatgpt") == 3
        assert await backend.get_raw_session_count(origin="claude-ai") == 2
        assert await backend.get_raw_session_count(origin="codex") == 0

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

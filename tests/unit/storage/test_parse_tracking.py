"""Tests for parse tracking, validation status, and mtime-based acquisition.

Tests cover:
- mark_raw_parsed: success and failure marking
- Parse skip: raw records with parsed_at are not re-parsed
- Parse failure: error sets parse_error, leaves parsed_at NULL for retry
- mark_raw_validated: validation status and error tracking
- get_known_source_mtimes: returns {path: mtime} mapping
- reset_parse_status: clears tracking for force-reparse
- reset_validation_status: clears validation tracking
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.schema import (
    SCHEMA_VERSION,
    _ensure_schema,
)
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.storage_records import admit_raw_record

# ─── Backend method tests ──────────────────────────────────────────────────


class TestMarkRawParsed:
    """Tests for mark_raw_parsed backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def _save_raw(self, backend: SQLiteBackend, raw_id: str = "test-raw") -> None:
        record = RawSessionRecord(
            raw_id=raw_id,
            source_name="test",
            source_path="/test.json",
            canonical_source_path="/test.json",
            blob_size=len(b'{"test": true}'),
            acquired_at="2026-01-01T00:00:00Z",
            file_mtime="2026-01-01T00:00:00Z",
        )
        await admit_raw_record(backend, record)

    async def test_mark_success(self, backend: SQLiteBackend) -> None:
        """Marking as parsed sets parsed_at and clears parse_error."""
        await self._save_raw(backend)
        await backend.mark_raw_parsed("test-raw", payload_provider="chatgpt")

        rec = await backend.get_raw_session("test-raw")
        assert rec is not None
        assert rec.parsed_at is not None
        assert rec.parse_error is None
        assert rec.payload_provider == "chatgpt"

    async def test_mark_failure(self, backend: SQLiteBackend) -> None:
        """Marking with error sets parse_error, leaves parsed_at NULL."""
        await self._save_raw(backend)
        await backend.mark_raw_parsed("test-raw", error="JSON decode error")

        rec = await backend.get_raw_session("test-raw")
        assert rec is not None
        assert rec.parsed_at is None
        assert rec.parse_error == "JSON decode error"

    async def test_mark_success_after_failure(self, backend: SQLiteBackend) -> None:
        """Successful parse after failure clears the error."""
        await self._save_raw(backend)
        await backend.mark_raw_parsed("test-raw", error="first attempt failed")
        await backend.mark_raw_parsed("test-raw")  # Success

        rec = await backend.get_raw_session("test-raw")
        assert rec is not None
        assert rec.parsed_at is not None
        assert rec.parse_error is None

    async def test_long_parse_error_is_retained_whole(self, backend: SQLiteBackend) -> None:
        """A long parse error is evidence; the write boundary keeps every character."""
        await self._save_raw(backend)
        long_error = "x" * 5000
        await backend.mark_raw_parsed("test-raw", error=long_error)

        rec = await backend.get_raw_session("test-raw")
        assert rec is not None
        assert rec.parse_error == long_error


class TestRawBlobAddress:
    """Tests for raw row identity versus content-addressed blob identity."""

    async def test_save_and_read_preserves_distinct_blob_hash(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        blob_hash = "a" * 64

        await admit_raw_record(
            backend,
            RawSessionRecord(
                raw_id="raw-row-identity",
                blob_hash=blob_hash,
                source_name="test",
                source_path="/test.json",
                canonical_source_path="/test.json",
                blob_size=len(b'{"test": true}'),
                acquired_at="2026-01-01T00:00:00Z",
            ),
        )

        rec = await backend.get_raw_session("raw-row-identity")

        assert rec is not None
        assert rec.raw_id == "raw-row-identity"
        assert rec.blob_hash == blob_hash


class TestUpdateRawState:
    """Tests for unified raw state update methods."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def _save_raw(self, backend: SQLiteBackend, raw_id: str = "update-raw") -> None:
        await admit_raw_record(
            backend,
            RawSessionRecord(
                raw_id=raw_id,
                source_name="test",
                source_path="/test.json",
                canonical_source_path="/test.json",
                blob_size=len(b'{"test": true}'),
                acquired_at="2026-01-01T00:00:00Z",
                file_mtime="2026-01-01T00:00:00Z",
            ),
        )

    async def test_update_raw_state_applies_only_requested_fields(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend)
        await backend.update_raw_state(
            "update-raw",
            state=RawSessionStateUpdate(
                parsed_at="2026-01-02T00:00:00Z",
                parse_error=None,
                payload_provider="chatgpt",
            ),
        )

        rec = await backend.get_raw_session("update-raw")
        assert rec is not None
        assert rec.parsed_at == "2026-01-02T00:00:00+00:00"
        assert rec.parse_error is None
        assert rec.payload_provider == "chatgpt"
        assert rec.validation_status is None

    async def test_update_raw_state_supports_validation_fields(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend, raw_id="validate-update")
        await backend.update_raw_state(
            "validate-update",
            state=RawSessionStateUpdate(
                validation_status="passed",
                validation_error="",
                validation_drift_count=3,
                validation_provider="chatgpt",
                validation_mode="strict",
            ),
        )

        rec = await backend.get_raw_session("validate-update")
        assert rec is not None
        assert rec.validated_at is not None
        assert rec.validation_status == "passed"
        assert rec.validation_error == ""
        assert rec.validation_drift_count == 3
        assert rec.validation_provider == "chatgpt"
        assert rec.validation_mode == "strict"

    async def test_update_raw_state_retains_whole_error_fields(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend, raw_id="error-trunc")
        long_error = "x" * 5000
        await backend.update_raw_state(
            "error-trunc",
            state=RawSessionStateUpdate(
                parse_error=long_error,
                validation_error=long_error,
            ),
        )

        rec = await backend.get_raw_session("error-trunc")
        assert rec is not None
        assert rec.parse_error == long_error
        assert rec.validation_error == long_error

    @pytest.mark.parametrize("wall_clock_ms", [1000, 999])
    async def test_failed_validation_after_parse_advances_past_identical_or_backward_clock(
        self, backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch, wall_clock_ms: int
    ) -> None:
        """A failed revalidation cannot tie or precede the parse it supersedes."""
        from polylogue.storage.sqlite.queries import raw_state as raw_state_queries

        await self._save_raw(backend, raw_id="parse-then-failed-validation")
        await backend.update_raw_state(
            "parse-then-failed-validation",
            state=RawSessionStateUpdate(parsed_at="1970-01-01T00:00:01Z"),
        )
        monkeypatch.setattr(raw_state_queries, "_now_ms", lambda: wall_clock_ms)
        await backend.mark_raw_validated("parse-then-failed-validation", status="failed", error="rejected")

        with sqlite3.connect(backend._source_db_path) as conn:
            row = conn.execute(
                "SELECT parsed_at_ms, validated_at_ms, validation_status FROM raw_sessions WHERE raw_id = ?",
                ("parse-then-failed-validation",),
            ).fetchone()
        assert row == (1000, 1001, "failed")

    @pytest.mark.parametrize(
        ("parsed_at", "expected"),
        [
            ("1970-01-01T00:00:01Z", (1001, 1000, "failed")),
            ("1970-01-01T00:00:00.999Z", (1001, 1000, "failed")),
        ],
    )
    async def test_successful_parse_after_validation_advances_past_identical_or_backward_clock(
        self,
        backend: SQLiteBackend,
        monkeypatch: pytest.MonkeyPatch,
        parsed_at: str,
        expected: tuple[int, int, str],
    ) -> None:
        """A later parse wins even if its injected wall clock is older."""
        from polylogue.storage.sqlite.queries import raw_state as raw_state_queries

        await self._save_raw(backend, raw_id="validation-then-parse")
        monkeypatch.setattr(raw_state_queries, "_now_ms", lambda: 1000)
        await backend.mark_raw_validated("validation-then-parse", status="failed", error="rejected")
        await backend.update_raw_state(
            "validation-then-parse",
            state=RawSessionStateUpdate(parsed_at=parsed_at, parse_error=None),
        )

        with sqlite3.connect(backend._source_db_path) as conn:
            row = conn.execute(
                "SELECT parsed_at_ms, validated_at_ms, validation_status FROM raw_sessions WHERE raw_id = ?",
                ("validation-then-parse",),
            ).fetchone()
        assert row == expected

    async def test_malformed_parse_timestamp_cannot_clear_existing_parse_authority(
        self, backend: SQLiteBackend
    ) -> None:
        await self._save_raw(backend, raw_id="malformed-parse-timestamp")
        await backend.update_raw_state(
            "malformed-parse-timestamp",
            state=RawSessionStateUpdate(parsed_at="1970-01-01T00:00:01Z"),
        )

        with pytest.raises(ValueError, match="parsed_at must be a valid timestamp"):
            await backend.update_raw_state(
                "malformed-parse-timestamp",
                state=RawSessionStateUpdate(parsed_at="not-a-timestamp"),
            )

        with sqlite3.connect(backend._source_db_path) as conn:
            row = conn.execute(
                "SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", ("malformed-parse-timestamp",)
            ).fetchone()
        assert row == (1000,)


class TestMarkRawValidated:
    """Tests for mark_raw_validated backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def _save_raw(self, backend: SQLiteBackend, raw_id: str = "test-raw") -> None:
        record = RawSessionRecord(
            raw_id=raw_id,
            source_name="test",
            source_path="/test.json",
            canonical_source_path="/test.json",
            blob_size=len(b'{"test": true}'),
            acquired_at="2026-01-01T00:00:00Z",
            file_mtime="2026-01-01T00:00:00Z",
        )
        await admit_raw_record(backend, record)

    async def test_mark_passed(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend)
        await backend.mark_raw_validated(
            "test-raw",
            status="passed",
            drift_count=2,
            provider="chatgpt",
            mode="strict",
            payload_provider="chatgpt",
        )

        rec = await backend.get_raw_session("test-raw")
        assert rec is not None
        assert rec.validated_at is not None
        assert rec.validation_status == "passed"
        assert rec.validation_error is None
        assert rec.validation_drift_count == 2
        assert rec.validation_provider == "chatgpt"
        assert rec.validation_mode == "strict"
        assert rec.payload_provider == "chatgpt"

    async def test_mark_failed_retains_whole_error(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend)
        long_error = "x" * 5000
        await backend.mark_raw_validated(
            "test-raw",
            status="failed",
            error=long_error,
            provider="chatgpt",
            mode="strict",
        )

        rec = await backend.get_raw_session("test-raw")
        assert rec is not None
        assert rec.validation_status == "failed"
        assert rec.validation_error == long_error

    async def test_invalid_status_raises(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend)
        with pytest.raises(ValueError, match="Invalid validation status"):
            await backend.mark_raw_validated("test-raw", status="unknown")


class TestGetKnownSourceMtimes:
    """Tests for get_known_source_mtimes backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        # Reads refuse an uninitialized Index; construct the empty archive first.
        run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
        return SQLiteBackend(db_path=tmp_path / "index.db")

    async def test_returns_mtime_mapping(self, backend: SQLiteBackend) -> None:
        """Returns {source_path: file_mtime} for records with mtimes."""
        for i in range(3):
            await admit_raw_record(
                backend,
                RawSessionRecord(
                    raw_id=f"raw-{i}",
                    source_name="test",
                    source_path=f"/path/file{i}.json",
                    canonical_source_path=f"/path/file{i}.json",
                    blob_size=len(f'{{"i": {i}}}'.encode()),
                    acquired_at="2026-01-01T00:00:00Z",
                    file_mtime=f"2026-01-0{i + 1}T00:00:00Z",
                ),
            )

        mtimes = await backend.get_known_source_mtimes()
        assert len(mtimes) == 3
        assert mtimes["/path/file0.json"] == "2026-01-01T00:00:00+00:00"
        assert mtimes["/path/file2.json"] == "2026-01-03T00:00:00+00:00"

    async def test_excludes_null_mtimes(self, backend: SQLiteBackend) -> None:
        """Records without file_mtime are excluded from the mapping."""
        await admit_raw_record(
            backend,
            RawSessionRecord(
                raw_id="with-mtime",
                source_name="test",
                source_path="/path/a.json",
                canonical_source_path="/path/a.json",
                blob_size=len(b"{}"),
                acquired_at="2026-01-01T00:00:00Z",
                file_mtime="2026-01-01T00:00:00Z",
            ),
        )
        await admit_raw_record(
            backend,
            RawSessionRecord(
                raw_id="no-mtime",
                source_name="test",
                source_path="/path/b.json",
                canonical_source_path="/path/b.json",
                blob_size=len(b'{"b": 1}'),
                acquired_at="2026-01-01T00:00:00Z",
                file_mtime=None,
            ),
        )

        mtimes = await backend.get_known_source_mtimes()
        assert len(mtimes) == 1
        assert "/path/a.json" in mtimes
        assert "/path/b.json" not in mtimes

    async def test_empty_db(self, backend: SQLiteBackend) -> None:
        """Empty database returns empty dict."""
        mtimes = await backend.get_known_source_mtimes()
        assert mtimes == {}


class TestResetParseStatus:
    """Tests for reset_parse_status backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def _populate(self, backend: SQLiteBackend) -> None:
        """Create 3 raw records, mark 2 as parsed."""
        rows = [
            ("chatgpt", "inbox-a"),
            ("chatgpt", "inbox-b"),
            ("claude-ai", "inbox-a"),
        ]
        for i, (provider, source_name) in enumerate(rows):
            await admit_raw_record(
                backend,
                RawSessionRecord(
                    raw_id=f"raw-{i}",
                    source_name=source_name,
                    payload_provider=Provider.from_string(provider),
                    source_path=f"/path/{i}.json",
                    canonical_source_path=f"/path/{i}.json",
                    blob_size=len(f'{{"i": {i}}}'.encode()),
                    acquired_at="2026-01-01T00:00:00Z",
                ),
            )
        # Origin-scoped selection projects detected_provider (raw_state.py's
        # raw_provider_origin_sql), not payload_provider, so a row without a
        # detected provider reads as 'unknown-export' and matches no origin
        # scope. Record the parser classification the scope filters on.
        with sqlite3.connect(backend.db_path.with_name("source.db")) as conn:
            for raw_id, provider in (("raw-0", "chatgpt"), ("raw-1", "chatgpt"), ("raw-2", "claude-ai")):
                conn.execute("UPDATE raw_sessions SET detected_provider = ? WHERE raw_id = ?", (provider, raw_id))
            conn.commit()
        await backend.mark_raw_parsed("raw-0")
        await backend.mark_raw_parsed("raw-2")

    async def test_reset_all(self, backend: SQLiteBackend) -> None:
        """Reset all providers clears parsed_at for all parsed records."""
        await self._populate(backend)
        count = await backend.reset_parse_status()
        assert count == 2

        # Verify all records are now unparsed
        for i in range(3):
            rec = await backend.get_raw_session(f"raw-{i}")
            assert rec is not None
            assert rec.parsed_at is None

    async def test_reset_by_provider(self, backend: SQLiteBackend) -> None:
        """Reset specific provider only clears that provider's records."""
        await self._populate(backend)
        count = await backend.reset_parse_status(origin="chatgpt")
        assert count == 1  # Only raw-0 was chatgpt and parsed

        # chatgpt record is reset
        rec0 = await backend.get_raw_session("raw-0")
        assert rec0 is not None
        assert rec0.parsed_at is None

        # claude record is still parsed
        rec2 = await backend.get_raw_session("raw-2")
        assert rec2 is not None
        assert rec2.parsed_at is not None

    async def test_reset_by_source_scope(self, backend: SQLiteBackend) -> None:
        """Reset scoped to specific origins only clears matching parsed records."""
        await self._populate(backend)
        count = await backend.reset_parse_status(source_names=["chatgpt-export", "claude-ai-export"])
        assert count == 2

        rec0 = await backend.get_raw_session("raw-0")
        rec1 = await backend.get_raw_session("raw-1")
        rec2 = await backend.get_raw_session("raw-2")
        assert rec0 is not None and rec1 is not None and rec2 is not None
        assert rec0.parsed_at is None
        assert rec1.parsed_at is None
        assert rec2.parsed_at is None

    async def test_reset_by_beads_origin_uses_detected_provider_projection(self, backend: SQLiteBackend) -> None:
        """A Beads classification scopes rows even when acquisition was unknown."""
        await admit_raw_record(
            backend,
            RawSessionRecord(
                raw_id="historical-beads",
                source_name="historical",
                payload_provider=Provider.UNKNOWN,
                source_path="/captures/historical.json",
                canonical_source_path="/captures/historical.json",
                blob_size=2,
                acquired_at="2026-01-01T00:00:00Z",
            ),
        )
        await backend.mark_raw_parsed("historical-beads")
        with sqlite3.connect(backend.db_path.with_name("source.db")) as conn:
            conn.execute(
                "UPDATE raw_sessions SET detected_provider = 'beads' WHERE raw_id = ?",
                ("historical-beads",),
            )
            conn.commit()

        assert await backend.reset_parse_status(origin="beads-issue") == 1
        record = await backend.get_raw_session("historical-beads")
        assert record is not None
        assert record.parsed_at is None

    async def test_reset_returns_zero_when_nothing_to_reset(self, backend: SQLiteBackend) -> None:
        """Reset returns 0 when no records have parsed_at set."""
        await admit_raw_record(
            backend,
            RawSessionRecord(
                raw_id="unparsed",
                source_name="test",
                source_path="/test.json",
                canonical_source_path="/test.json",
                blob_size=len(b"{}"),
                acquired_at="2026-01-01T00:00:00Z",
            ),
        )
        count = await backend.reset_parse_status()
        assert count == 0


class TestResetValidationStatus:
    """Tests for reset_validation_status backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def _populate(self, backend: SQLiteBackend) -> None:
        rows = [
            ("chatgpt", "inbox-a"),
            ("chatgpt", "inbox-b"),
            ("claude-ai", "inbox-a"),
        ]
        for i, (provider, source_name) in enumerate(rows):
            await admit_raw_record(
                backend,
                RawSessionRecord(
                    raw_id=f"raw-{i}",
                    source_name=source_name,
                    payload_provider=Provider.from_string(provider),
                    source_path=f"/path/{i}.json",
                    canonical_source_path=f"/path/{i}.json",
                    blob_size=len(f'{{"i": {i}}}'.encode()),
                    acquired_at="2026-01-01T00:00:00Z",
                ),
            )
        await backend.mark_raw_validated(
            "raw-0",
            status="passed",
            provider="chatgpt",
            mode="strict",
            payload_provider="chatgpt",
        )
        await backend.mark_raw_validated(
            "raw-2",
            status="failed",
            error="bad schema",
            provider="claude-ai",
            mode="strict",
            payload_provider="claude-ai",
        )

    async def test_reset_all(self, backend: SQLiteBackend) -> None:
        await self._populate(backend)
        count = await backend.reset_validation_status()
        assert count == 2

        for i in range(3):
            rec = await backend.get_raw_session(f"raw-{i}")
            assert rec is not None
            assert rec.validated_at is None
            assert rec.validation_status is None
            assert rec.validation_error is None
        rec0 = await backend.get_raw_session("raw-0")
        rec2 = await backend.get_raw_session("raw-2")
        assert rec0 is not None and rec2 is not None
        assert rec0.payload_provider == "chatgpt"
        assert rec2.payload_provider == "claude-ai"

    async def test_reset_by_provider(self, backend: SQLiteBackend) -> None:
        await self._populate(backend)
        count = await backend.reset_validation_status(origin="chatgpt")
        assert count == 1

        rec0 = await backend.get_raw_session("raw-0")
        rec2 = await backend.get_raw_session("raw-2")
        assert rec0 is not None and rec2 is not None
        assert rec0.validation_status is None
        assert rec2.validation_status == "failed"
        assert rec0.payload_provider == "chatgpt"
        assert rec2.payload_provider == "claude-ai"

    async def test_reset_validation_by_source_scope(self, backend: SQLiteBackend) -> None:
        await self._populate(backend)
        count = await backend.reset_validation_status(source_names=["chatgpt-export", "claude-ai-export"])
        assert count == 2

        rec0 = await backend.get_raw_session("raw-0")
        rec1 = await backend.get_raw_session("raw-1")
        rec2 = await backend.get_raw_session("raw-2")
        assert rec0 is not None and rec1 is not None and rec2 is not None
        assert rec0.validation_status is None
        assert rec1.validation_status is None
        assert rec2.validation_status is None


class TestSourceParsingInput:
    """Resident source parsing consumes its input on each explicit read."""

    def test_repeated_source_read_keeps_session(self, tmp_path: Path) -> None:
        from polylogue.config import Source
        from polylogue.sources.source_parsing import iter_source_sessions_with_raw

        fixture = Path(__file__).parents[2] / "fixtures/origin-capability/chatgpt-export.json"
        test_file = tmp_path / "test.json"
        test_file.write_bytes(fixture.read_bytes())
        source = Source(name="chatgpt", path=tmp_path)
        first = list(iter_source_sessions_with_raw(source, capture_raw=True))
        second = list(iter_source_sessions_with_raw(source, capture_raw=True))
        assert first
        assert [session.provider_session_id for _, session in second] == [
            session.provider_session_id for _, session in first
        ]


# ─── Fresh schema test ─────────────────────────────────────────────────────


class TestFreshSchema:
    """Test that fresh databases have all v12+v13+v14+v15 features."""

    def test_fresh_db_has_parse_tracking_columns(self, tmp_path: Path) -> None:
        """A fresh database has parse+validation tracking and ordering columns."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        # raw_sessions lives in the source durability tier (#1743).
        conn.executescript(SOURCE_DDL)

        cursor = conn.execute("PRAGMA table_info(raw_sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "parsed_at_ms" in columns
        assert "parse_error" in columns
        assert "validated_at_ms" in columns
        assert "validation_status" in columns
        assert "validation_error" in columns
        assert "origin" in columns

        cursor = conn.execute("PRAGMA table_info(messages)")
        msg_columns = {row[1] for row in cursor.fetchall()}
        assert "position" in msg_columns

        cursor = conn.execute("PRAGMA table_info(sessions)")
        conv_columns = {row[1] for row in cursor.fetchall()}
        assert "updated_at_ms" in conv_columns

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION

        conn.close()

    def test_fresh_db_has_all_indices(self, tmp_path: Path) -> None:
        """A fresh database has all expected indices from the archive schema."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        # raw_sessions and its indices live in the source durability tier (#1743).
        conn.executescript(SOURCE_DDL)

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in cursor.fetchall()}

        # raw_sessions indices
        assert "idx_raw_sessions_origin" in indices
        assert "idx_raw_sessions_source_path" in indices
        assert "idx_raw_sessions_parse_ready" in indices

        # sessions indices
        assert "idx_sessions_origin_sort" in indices
        assert "idx_sessions_raw_id" in indices

        # messages indices
        assert "idx_messages_session_position" in indices

        # block indices
        assert "idx_blocks_session_position" in indices
        assert "idx_blocks_type" in indices

        conn.close()

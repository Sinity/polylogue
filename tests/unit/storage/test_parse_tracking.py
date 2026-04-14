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

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.schema import (
    SCHEMA_VERSION,
    _ensure_raw_source_mtime_index,
    _ensure_schema,
)
from polylogue.storage.state_views import RawConversationStateUpdate
from polylogue.storage.store import RawConversationRecord

# ─── Backend method tests ──────────────────────────────────────────────────


class TestMarkRawParsed:
    """Tests for mark_raw_parsed backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def _save_raw(self, backend: SQLiteBackend, raw_id: str = "test-raw") -> None:
        record = RawConversationRecord(
            raw_id=raw_id,
            provider_name="test",
            source_path="/test.json",
            blob_size=len(b'{"test": true}'),
            acquired_at="2026-01-01T00:00:00Z",
            file_mtime="2026-01-01T00:00:00Z",
        )
        await backend.save_raw_conversation(record)

    async def test_mark_success(self, backend: SQLiteBackend) -> None:
        """Marking as parsed sets parsed_at and clears parse_error."""
        await self._save_raw(backend)
        await backend.mark_raw_parsed("test-raw", payload_provider="chatgpt")

        rec = await backend.get_raw_conversation("test-raw")
        assert rec is not None
        assert rec.parsed_at is not None
        assert rec.parse_error is None
        assert rec.payload_provider == "chatgpt"

    async def test_mark_failure(self, backend: SQLiteBackend) -> None:
        """Marking with error sets parse_error, leaves parsed_at NULL."""
        await self._save_raw(backend)
        await backend.mark_raw_parsed("test-raw", error="JSON decode error")

        rec = await backend.get_raw_conversation("test-raw")
        assert rec is not None
        assert rec.parsed_at is None
        assert rec.parse_error == "JSON decode error"

    async def test_mark_success_after_failure(self, backend: SQLiteBackend) -> None:
        """Successful parse after failure clears the error."""
        await self._save_raw(backend)
        await backend.mark_raw_parsed("test-raw", error="first attempt failed")
        await backend.mark_raw_parsed("test-raw")  # Success

        rec = await backend.get_raw_conversation("test-raw")
        assert rec is not None
        assert rec.parsed_at is not None
        assert rec.parse_error is None

    async def test_error_truncation(self, backend: SQLiteBackend) -> None:
        """Long error messages are truncated to prevent DB bloat."""
        await self._save_raw(backend)
        long_error = "x" * 5000
        await backend.mark_raw_parsed("test-raw", error=long_error)

        rec = await backend.get_raw_conversation("test-raw")
        assert rec is not None
        assert len(rec.parse_error) == 2000  # type: ignore[arg-type]


class TestUpdateRawState:
    """Tests for unified raw state update methods."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def _save_raw(self, backend: SQLiteBackend, raw_id: str = "update-raw") -> None:
        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id=raw_id,
                provider_name="test",
                source_path="/test.json",
                blob_size=len(b'{"test": true}'),
                acquired_at="2026-01-01T00:00:00Z",
                file_mtime="2026-01-01T00:00:00Z",
            )
        )

    async def test_update_raw_state_applies_only_requested_fields(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend)
        await backend.update_raw_state(
            "update-raw",
            state=RawConversationStateUpdate(
                parsed_at="2026-01-02T00:00:00Z",
                parse_error=None,
                payload_provider="chatgpt",
            ),
        )

        rec = await backend.get_raw_conversation("update-raw")
        assert rec is not None
        assert rec.parsed_at == "2026-01-02T00:00:00Z"
        assert rec.parse_error is None
        assert rec.payload_provider == "chatgpt"
        assert rec.validation_status is None

    async def test_update_raw_state_supports_validation_fields(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend, raw_id="validate-update")
        await backend.update_raw_state(
            "validate-update",
            state=RawConversationStateUpdate(
                validation_status="passed",
                validation_error="",
                validation_drift_count=3,
                validation_provider="chatgpt",
                validation_mode="strict",
            ),
        )

        rec = await backend.get_raw_conversation("validate-update")
        assert rec is not None
        assert rec.validated_at is not None
        assert rec.validation_status == "passed"
        assert rec.validation_error == ""
        assert rec.validation_drift_count == 3
        assert rec.validation_provider == "chatgpt"
        assert rec.validation_mode == "strict"

    async def test_update_raw_state_truncates_error_fields(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend, raw_id="error-trunc")
        long_error = "x" * 5000
        await backend.update_raw_state(
            "error-trunc",
            state=RawConversationStateUpdate(
                parse_error=long_error,
                validation_error=long_error,
            ),
        )

        rec = await backend.get_raw_conversation("error-trunc")
        assert rec is not None
        assert rec.parse_error is not None
        assert len(rec.parse_error) == 2000
        assert rec.validation_error is not None
        assert len(rec.validation_error) == 2000


class TestMarkRawValidated:
    """Tests for mark_raw_validated backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def _save_raw(self, backend: SQLiteBackend, raw_id: str = "test-raw") -> None:
        record = RawConversationRecord(
            raw_id=raw_id,
            provider_name="test",
            source_path="/test.json",
            blob_size=len(b'{"test": true}'),
            acquired_at="2026-01-01T00:00:00Z",
            file_mtime="2026-01-01T00:00:00Z",
        )
        await backend.save_raw_conversation(record)

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

        rec = await backend.get_raw_conversation("test-raw")
        assert rec is not None
        assert rec.validated_at is not None
        assert rec.validation_status == "passed"
        assert rec.validation_error is None
        assert rec.validation_drift_count == 2
        assert rec.validation_provider == "chatgpt"
        assert rec.validation_mode == "strict"
        assert rec.payload_provider == "chatgpt"

    async def test_mark_failed_truncates_error(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend)
        long_error = "x" * 5000
        await backend.mark_raw_validated(
            "test-raw",
            status="failed",
            error=long_error,
            provider="chatgpt",
            mode="strict",
        )

        rec = await backend.get_raw_conversation("test-raw")
        assert rec is not None
        assert rec.validation_status == "failed"
        assert rec.validation_error is not None
        assert len(rec.validation_error) == 2000

    async def test_invalid_status_raises(self, backend: SQLiteBackend) -> None:
        await self._save_raw(backend)
        with pytest.raises(ValueError, match="Invalid validation status"):
            await backend.mark_raw_validated("test-raw", status="unknown")


class TestGetKnownSourceMtimes:
    """Tests for get_known_source_mtimes backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def test_returns_mtime_mapping(self, backend: SQLiteBackend) -> None:
        """Returns {source_path: file_mtime} for records with mtimes."""
        for i in range(3):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"raw-{i}",
                    provider_name="test",
                    source_path=f"/path/file{i}.json",
                    blob_size=len(f'{{"i": {i}}}'.encode()),
                    acquired_at="2026-01-01T00:00:00Z",
                    file_mtime=f"2026-01-0{i + 1}T00:00:00Z",
                )
            )

        mtimes = await backend.get_known_source_mtimes()
        assert len(mtimes) == 3
        assert mtimes["/path/file0.json"] == "2026-01-01T00:00:00Z"
        assert mtimes["/path/file2.json"] == "2026-01-03T00:00:00Z"

    async def test_excludes_null_mtimes(self, backend: SQLiteBackend) -> None:
        """Records without file_mtime are excluded from the mapping."""
        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="with-mtime",
                provider_name="test",
                source_path="/path/a.json",
                blob_size=len(b"{}"),
                acquired_at="2026-01-01T00:00:00Z",
                file_mtime="2026-01-01T00:00:00Z",
            )
        )
        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="no-mtime",
                provider_name="test",
                source_path="/path/b.json",
                blob_size=len(b'{"b": 1}'),
                acquired_at="2026-01-01T00:00:00Z",
                file_mtime=None,
            )
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
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"raw-{i}",
                    provider_name=provider,
                    source_name=source_name,
                    source_path=f"/path/{i}.json",
                    blob_size=len(f'{{"i": {i}}}'.encode()),
                    acquired_at="2026-01-01T00:00:00Z",
                )
            )
        await backend.mark_raw_parsed("raw-0")
        await backend.mark_raw_parsed("raw-2")

    async def test_reset_all(self, backend: SQLiteBackend) -> None:
        """Reset all providers clears parsed_at for all parsed records."""
        await self._populate(backend)
        count = await backend.reset_parse_status()
        assert count == 2

        # Verify all records are now unparsed
        for i in range(3):
            rec = await backend.get_raw_conversation(f"raw-{i}")
            assert rec is not None
            assert rec.parsed_at is None

    async def test_reset_by_provider(self, backend: SQLiteBackend) -> None:
        """Reset specific provider only clears that provider's records."""
        await self._populate(backend)
        count = await backend.reset_parse_status(provider="chatgpt")
        assert count == 1  # Only raw-0 was chatgpt and parsed

        # chatgpt record is reset
        rec0 = await backend.get_raw_conversation("raw-0")
        assert rec0 is not None
        assert rec0.parsed_at is None

        # claude record is still parsed
        rec2 = await backend.get_raw_conversation("raw-2")
        assert rec2 is not None
        assert rec2.parsed_at is not None

    async def test_reset_by_source_scope(self, backend: SQLiteBackend) -> None:
        """Reset specific sources only clears matching parsed records."""
        await self._populate(backend)
        count = await backend.reset_parse_status(source_names=["inbox-a"])
        assert count == 2

        rec0 = await backend.get_raw_conversation("raw-0")
        rec1 = await backend.get_raw_conversation("raw-1")
        rec2 = await backend.get_raw_conversation("raw-2")
        assert rec0 is not None and rec1 is not None and rec2 is not None
        assert rec0.parsed_at is None
        assert rec1.parsed_at is None
        assert rec2.parsed_at is None

    async def test_reset_returns_zero_when_nothing_to_reset(self, backend: SQLiteBackend) -> None:
        """Reset returns 0 when no records have parsed_at set."""
        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="unparsed",
                provider_name="test",
                source_path="/test.json",
                blob_size=len(b"{}"),
                acquired_at="2026-01-01T00:00:00Z",
            )
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
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"raw-{i}",
                    provider_name=provider,
                    source_name=source_name,
                    source_path=f"/path/{i}.json",
                    blob_size=len(f'{{"i": {i}}}'.encode()),
                    acquired_at="2026-01-01T00:00:00Z",
                )
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
            rec = await backend.get_raw_conversation(f"raw-{i}")
            assert rec is not None
            assert rec.validated_at is None
            assert rec.validation_status is None
            assert rec.validation_error is None
        rec0 = await backend.get_raw_conversation("raw-0")
        rec2 = await backend.get_raw_conversation("raw-2")
        assert rec0 is not None and rec2 is not None
        assert rec0.payload_provider == "chatgpt"
        assert rec2.payload_provider == "claude-ai"

    async def test_reset_by_provider(self, backend: SQLiteBackend) -> None:
        await self._populate(backend)
        count = await backend.reset_validation_status(provider="chatgpt")
        assert count == 1

        rec0 = await backend.get_raw_conversation("raw-0")
        rec2 = await backend.get_raw_conversation("raw-2")
        assert rec0 is not None and rec2 is not None
        assert rec0.validation_status is None
        assert rec2.validation_status == "failed"
        assert rec0.payload_provider == "chatgpt"
        assert rec2.payload_provider == "claude-ai"

    async def test_reset_validation_by_source_scope(self, backend: SQLiteBackend) -> None:
        await self._populate(backend)
        count = await backend.reset_validation_status(source_names=["inbox-a"])
        assert count == 2

        rec0 = await backend.get_raw_conversation("raw-0")
        rec1 = await backend.get_raw_conversation("raw-1")
        rec2 = await backend.get_raw_conversation("raw-2")
        assert rec0 is not None and rec1 is not None and rec2 is not None
        assert rec0.validation_status is None
        assert rec1.validation_status is None
        assert rec2.validation_status is None


# ─── Mtime skip integration test ──────────────────────────────────────────


class TestMtimeSkip:
    """Tests for mtime-based file skipping in iter_source_conversations_with_raw."""

    def test_unchanged_file_skipped(self, tmp_path: Path) -> None:
        """Files with matching mtime in known_mtimes are skipped."""
        from polylogue.config import Source
        from polylogue.sources.cursor import _get_file_mtime
        from polylogue.sources.source_parsing import iter_source_conversations_with_raw

        # Create a test JSON file
        test_file = tmp_path / "test.json"
        test_file.write_text(
            '{"title": "test", "mapping": {"1": {"id": "1", "message": {"author": {"role": "user"}, "content": {"parts": ["hello"]}, "create_time": 1000000}}}}'
        )

        source = Source(name="test", path=tmp_path)

        # First pass: capture all conversations (no known_mtimes)
        results_first = list(iter_source_conversations_with_raw(source, capture_raw=True))
        assert len(results_first) > 0

        # Get the actual mtime
        file_mtime = _get_file_mtime(test_file)
        known_mtimes = {str(test_file): file_mtime}

        # Second pass with known_mtimes: file should be skipped
        results_second = list(
            iter_source_conversations_with_raw(
                source,
                capture_raw=True,
                known_mtimes=known_mtimes,
            )
        )
        assert len(results_second) == 0

    def test_modified_file_not_skipped(self, tmp_path: Path) -> None:
        """Files with different mtime are NOT skipped."""
        from polylogue.config import Source
        from polylogue.sources.source_parsing import iter_source_conversations_with_raw

        test_file = tmp_path / "test.json"
        test_file.write_text(
            '{"title": "test", "mapping": {"1": {"id": "1", "message": {"author": {"role": "user"}, "content": {"parts": ["hello"]}, "create_time": 1000000}}}}'
        )

        source = Source(name="test", path=tmp_path)

        # Known mtimes with a DIFFERENT mtime than the actual file
        known_mtimes = {str(test_file): "1999-01-01T00:00:00Z"}

        results = list(
            iter_source_conversations_with_raw(
                source,
                capture_raw=True,
                known_mtimes=known_mtimes,
            )
        )
        assert len(results) > 0

    def test_no_known_mtimes_processes_all(self, tmp_path: Path) -> None:
        """Without known_mtimes, all files are processed normally."""
        from polylogue.config import Source
        from polylogue.sources.source_parsing import iter_source_conversations_with_raw

        test_file = tmp_path / "test.json"
        test_file.write_text(
            '{"title": "test", "mapping": {"1": {"id": "1", "message": {"author": {"role": "user"}, "content": {"parts": ["hello"]}, "create_time": 1000000}}}}'
        )

        source = Source(name="test", path=tmp_path)

        results = list(
            iter_source_conversations_with_raw(
                source,
                capture_raw=True,
                known_mtimes=None,
            )
        )
        assert len(results) > 0


# ─── Fresh schema test ─────────────────────────────────────────────────────


class TestFreshSchema:
    """Test that fresh databases have all v12+v13+v14+v15 features."""

    def test_fresh_db_has_parse_tracking_columns(self, tmp_path: Path) -> None:
        """A fresh database has parse+validation tracking and sort_key columns."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)

        cursor = conn.execute("PRAGMA table_info(raw_conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "parsed_at" in columns
        assert "parse_error" in columns
        assert "validated_at" in columns
        assert "validation_status" in columns
        assert "validation_error" in columns
        assert "payload_provider" in columns

        cursor = conn.execute("PRAGMA table_info(messages)")
        msg_columns = {row[1] for row in cursor.fetchall()}
        assert "sort_key" in msg_columns

        cursor = conn.execute("PRAGMA table_info(conversations)")
        conv_columns = {row[1] for row in cursor.fetchall()}
        assert "sort_key" in conv_columns

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION

        conn.close()

    def test_fresh_db_has_all_indices(self, tmp_path: Path) -> None:
        """A fresh database has all expected indices from the v1 schema."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in cursor.fetchall()}

        # raw_conversations indices
        assert "idx_raw_conv_source_mtime" in indices
        assert "idx_raw_conv_parse_ready" in indices
        assert "idx_raw_conv_payload_provider" in indices

        # conversations indices
        assert "idx_conversations_content_hash" in indices
        assert "idx_conversations_sortkey" in indices

        # messages indices
        assert "idx_messages_conversation_sortkey" in indices

        # content_blocks indices
        assert "idx_content_blocks_message" in indices
        assert "idx_content_blocks_conversation" in indices
        assert "idx_content_blocks_conv_type" in indices

        conn.close()

    @pytest.mark.parametrize(
        "ddl",
        [
            """
            CREATE INDEX IF NOT EXISTS idx_raw_conv_source_mtime
            ON raw_conversations(source_path, file_mtime)
            WHERE file_mtime IS NOT NULL;
            """,
            """
            CREATE INDEX idx_raw_conv_source_mtime
            ON raw_conversations(source_path, file_mtime)
            WHERE file_mtime IS NOT NULL;
            """,
        ],
    )
    def test_raw_source_mtime_index_upgrade_accepts_equivalent_sql(self, tmp_path: Path, ddl: str) -> None:
        """Equivalent DDL formatting should not trigger drop/recreate churn."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE raw_conversations (raw_id TEXT PRIMARY KEY, source_path TEXT, file_mtime TEXT)")
        conn.executescript(ddl)

        _ensure_raw_source_mtime_index(conn)

        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_raw_conv_source_mtime'"
        ).fetchone()
        assert row is not None
        assert row[0] is not None
        assert "WHERE file_mtime IS NOT NULL" in row[0]
        conn.close()

"""Tests for parse tracking and mtime-based acquisition skip (v12 schema).

Tests cover:
- Schema migration v11 → v12 (columns, index, backfill)
- mark_raw_parsed: success and failure marking
- Parse skip: raw records with parsed_at are not re-parsed
- Parse failure: error sets parse_error, leaves parsed_at NULL for retry
- get_known_source_mtimes: returns {path: mtime} mapping
- reset_parse_status: clears tracking for force-reparse
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.schema import (
    SCHEMA_VERSION,
    _ensure_schema,
    _migrate_v11_to_v12,
    _migrate_v12_to_v13,
    _run_migrations,
)
from polylogue.storage.store import RawConversationRecord

# ─── Migration tests ───────────────────────────────────────────────────────


class TestMigrationV11ToV12:
    """Tests for the v11 → v12 schema migration."""

    def _make_v11_db(self, tmp_path: Path) -> sqlite3.Connection:
        """Create a database at schema v11 (pre-migration)."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Apply schema at v11 by running _ensure_schema then rolling back to v11
        # Simpler: just create the raw_conversations table at v11 schema
        conn.executescript("""
            CREATE TABLE raw_conversations (
                raw_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                source_name TEXT,
                source_path TEXT NOT NULL,
                source_index INTEGER,
                raw_content BLOB NOT NULL,
                acquired_at TEXT NOT NULL,
                file_mtime TEXT
            );
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                provider_conversation_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                content_hash TEXT NOT NULL,
                provider_meta TEXT,
                metadata TEXT DEFAULT '{}',
                source_name TEXT,
                version INTEGER NOT NULL,
                parent_conversation_id TEXT,
                branch_type TEXT,
                raw_id TEXT REFERENCES raw_conversations(raw_id)
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                provider_message_id TEXT,
                role TEXT,
                text TEXT,
                timestamp TEXT,
                content_hash TEXT NOT NULL,
                provider_meta TEXT,
                version INTEGER NOT NULL,
                parent_message_id TEXT,
                branch_index INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id) ON DELETE CASCADE
            );
        """)
        conn.execute("PRAGMA user_version = 11")
        conn.commit()
        return conn

    def test_migration_adds_columns(self, tmp_path: Path) -> None:
        """Migration adds parsed_at and parse_error columns."""
        conn = self._make_v11_db(tmp_path)
        _migrate_v11_to_v12(conn)
        conn.commit()

        cursor = conn.execute("PRAGMA table_info(raw_conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "parsed_at" in columns
        assert "parse_error" in columns
        conn.close()

    def test_migration_creates_mtime_index(self, tmp_path: Path) -> None:
        """Migration creates the source_path+file_mtime composite index."""
        conn = self._make_v11_db(tmp_path)
        _migrate_v11_to_v12(conn)
        conn.commit()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_raw_conv_source_mtime'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_migration_backfills_parsed_at(self, tmp_path: Path) -> None:
        """Migration backfills parsed_at for raw records that have linked conversations."""
        conn = self._make_v11_db(tmp_path)

        # Insert raw records
        conn.execute(
            "INSERT INTO raw_conversations (raw_id, provider_name, source_path, raw_content, acquired_at) "
            "VALUES ('raw-linked', 'test', '/test.json', x'7b7d', '2026-01-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO raw_conversations (raw_id, provider_name, source_path, raw_content, acquired_at) "
            "VALUES ('raw-orphan', 'test', '/test2.json', x'7b7d', '2026-01-02T00:00:00Z')"
        )
        # Only link one of them
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, "
            "content_hash, version, raw_id) VALUES ('conv-1', 'test', 'test-1', 'hash1', 1, 'raw-linked')"
        )
        conn.commit()

        _migrate_v11_to_v12(conn)
        conn.commit()

        # Linked record should have parsed_at = acquired_at
        row = conn.execute(
            "SELECT parsed_at FROM raw_conversations WHERE raw_id = 'raw-linked'"
        ).fetchone()
        assert row["parsed_at"] == "2026-01-01T00:00:00Z"

        # Orphan record should have parsed_at = NULL (will be re-parsed)
        row = conn.execute(
            "SELECT parsed_at FROM raw_conversations WHERE raw_id = 'raw-orphan'"
        ).fetchone()
        assert row["parsed_at"] is None

        conn.close()

    def test_full_migration_path_v11_to_v15(self, tmp_path: Path) -> None:
        """Full migration from v11 through v12, v13, v14, and v15 via _run_migrations."""
        conn = self._make_v11_db(tmp_path)
        _run_migrations(conn, 11, 15)

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 15
        conn.close()


class TestMigrationV12ToV13:
    """Tests for the v12 → v13 schema migration (performance indices)."""

    def _make_v12_db(self, tmp_path: Path) -> sqlite3.Connection:
        """Create a database at schema v12."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        conn.executescript("""
            CREATE TABLE raw_conversations (
                raw_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                source_name TEXT,
                source_path TEXT NOT NULL,
                source_index INTEGER,
                raw_content BLOB NOT NULL,
                acquired_at TEXT NOT NULL,
                file_mtime TEXT,
                parsed_at TEXT,
                parse_error TEXT
            );
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                provider_conversation_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                content_hash TEXT NOT NULL,
                provider_meta TEXT,
                metadata TEXT DEFAULT '{}',
                source_name TEXT,
                version INTEGER NOT NULL,
                parent_conversation_id TEXT,
                branch_type TEXT,
                raw_id TEXT REFERENCES raw_conversations(raw_id)
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                provider_message_id TEXT,
                role TEXT,
                text TEXT,
                timestamp TEXT,
                content_hash TEXT NOT NULL,
                provider_meta TEXT,
                version INTEGER NOT NULL,
                parent_message_id TEXT,
                branch_index INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id) ON DELETE CASCADE
            );
        """)
        conn.execute("PRAGMA user_version = 12")
        conn.commit()
        return conn

    def test_migration_creates_unparsed_partial_index(self, tmp_path: Path) -> None:
        """Migration creates the partial index on unparsed raw records."""
        conn = self._make_v12_db(tmp_path)
        _migrate_v12_to_v13(conn)
        conn.commit()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_raw_conv_unparsed'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_migration_creates_content_hash_index(self, tmp_path: Path) -> None:
        """Migration creates the content_hash index on conversations."""
        conn = self._make_v12_db(tmp_path)
        _migrate_v12_to_v13(conn)
        conn.commit()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_conversations_content_hash'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_migration_creates_messages_composite_index(self, tmp_path: Path) -> None:
        """Migration creates the conversation_id+timestamp composite index on messages."""
        conn = self._make_v12_db(tmp_path)
        _migrate_v12_to_v13(conn)
        conn.commit()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_conversation_ts'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Running the migration twice doesn't error (IF NOT EXISTS)."""
        conn = self._make_v12_db(tmp_path)
        _migrate_v12_to_v13(conn)
        conn.commit()
        # Running again should be a no-op
        _migrate_v12_to_v13(conn)
        conn.commit()
        conn.close()


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
            raw_content=b'{"test": true}',
            acquired_at="2026-01-01T00:00:00Z",
            file_mtime="2026-01-01T00:00:00Z",
        )
        await backend.save_raw_conversation(record)

    async def test_mark_success(self, backend: SQLiteBackend) -> None:
        """Marking as parsed sets parsed_at and clears parse_error."""
        await self._save_raw(backend)
        await backend.mark_raw_parsed("test-raw")

        rec = await backend.get_raw_conversation("test-raw")
        assert rec is not None
        assert rec.parsed_at is not None
        assert rec.parse_error is None

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


class TestGetKnownSourceMtimes:
    """Tests for get_known_source_mtimes backend method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def test_returns_mtime_mapping(self, backend: SQLiteBackend) -> None:
        """Returns {source_path: file_mtime} for records with mtimes."""
        for i in range(3):
            await backend.save_raw_conversation(RawConversationRecord(
                raw_id=f"raw-{i}",
                provider_name="test",
                source_path=f"/path/file{i}.json",
                raw_content=f'{{"i": {i}}}'.encode(),
                acquired_at="2026-01-01T00:00:00Z",
                file_mtime=f"2026-01-0{i+1}T00:00:00Z",
            ))

        mtimes = await backend.get_known_source_mtimes()
        assert len(mtimes) == 3
        assert mtimes["/path/file0.json"] == "2026-01-01T00:00:00Z"
        assert mtimes["/path/file2.json"] == "2026-01-03T00:00:00Z"

    async def test_excludes_null_mtimes(self, backend: SQLiteBackend) -> None:
        """Records without file_mtime are excluded from the mapping."""
        await backend.save_raw_conversation(RawConversationRecord(
            raw_id="with-mtime",
            provider_name="test",
            source_path="/path/a.json",
            raw_content=b'{}',
            acquired_at="2026-01-01T00:00:00Z",
            file_mtime="2026-01-01T00:00:00Z",
        ))
        await backend.save_raw_conversation(RawConversationRecord(
            raw_id="no-mtime",
            provider_name="test",
            source_path="/path/b.json",
            raw_content=b'{"b": 1}',
            acquired_at="2026-01-01T00:00:00Z",
            file_mtime=None,
        ))

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
        for i, provider in enumerate(["chatgpt", "chatgpt", "claude"]):
            await backend.save_raw_conversation(RawConversationRecord(
                raw_id=f"raw-{i}",
                provider_name=provider,
                source_path=f"/path/{i}.json",
                raw_content=f'{{"i": {i}}}'.encode(),
                acquired_at="2026-01-01T00:00:00Z",
            ))
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

    async def test_reset_returns_zero_when_nothing_to_reset(self, backend: SQLiteBackend) -> None:
        """Reset returns 0 when no records have parsed_at set."""
        await backend.save_raw_conversation(RawConversationRecord(
            raw_id="unparsed",
            provider_name="test",
            source_path="/test.json",
            raw_content=b'{}',
            acquired_at="2026-01-01T00:00:00Z",
        ))
        count = await backend.reset_parse_status()
        assert count == 0


# ─── Mtime skip integration test ──────────────────────────────────────────


class TestMtimeSkip:
    """Tests for mtime-based file skipping in iter_source_conversations_with_raw."""

    def test_unchanged_file_skipped(self, tmp_path: Path) -> None:
        """Files with matching mtime in known_mtimes are skipped."""
        from polylogue.config import Source
        from polylogue.sources.source import _get_file_mtime, iter_source_conversations_with_raw

        # Create a test JSON file
        test_file = tmp_path / "test.json"
        test_file.write_text('{"title": "test", "mapping": {"1": {"id": "1", "message": {"author": {"role": "user"}, "content": {"parts": ["hello"]}, "create_time": 1000000}}}}')

        source = Source(name="test", path=tmp_path)

        # First pass: capture all conversations (no known_mtimes)
        results_first = list(iter_source_conversations_with_raw(source, capture_raw=True))
        assert len(results_first) > 0

        # Get the actual mtime
        file_mtime = _get_file_mtime(test_file)
        known_mtimes = {str(test_file): file_mtime}

        # Second pass with known_mtimes: file should be skipped
        results_second = list(iter_source_conversations_with_raw(
            source, capture_raw=True, known_mtimes=known_mtimes,
        ))
        assert len(results_second) == 0

    def test_modified_file_not_skipped(self, tmp_path: Path) -> None:
        """Files with different mtime are NOT skipped."""
        from polylogue.config import Source
        from polylogue.sources.source import iter_source_conversations_with_raw

        test_file = tmp_path / "test.json"
        test_file.write_text('{"title": "test", "mapping": {"1": {"id": "1", "message": {"author": {"role": "user"}, "content": {"parts": ["hello"]}, "create_time": 1000000}}}}')

        source = Source(name="test", path=tmp_path)

        # Known mtimes with a DIFFERENT mtime than the actual file
        known_mtimes = {str(test_file): "1999-01-01T00:00:00Z"}

        results = list(iter_source_conversations_with_raw(
            source, capture_raw=True, known_mtimes=known_mtimes,
        ))
        assert len(results) > 0

    def test_no_known_mtimes_processes_all(self, tmp_path: Path) -> None:
        """Without known_mtimes, all files are processed normally."""
        from polylogue.config import Source
        from polylogue.sources.source import iter_source_conversations_with_raw

        test_file = tmp_path / "test.json"
        test_file.write_text('{"title": "test", "mapping": {"1": {"id": "1", "message": {"author": {"role": "user"}, "content": {"parts": ["hello"]}, "create_time": 1000000}}}}')

        source = Source(name="test", path=tmp_path)

        results = list(iter_source_conversations_with_raw(
            source, capture_raw=True, known_mtimes=None,
        ))
        assert len(results) > 0


# ─── Fresh schema test ─────────────────────────────────────────────────────


class TestFreshSchema:
    """Test that fresh databases have all v12+v13+v14+v15 features."""

    def test_fresh_db_has_parse_tracking_columns(self, tmp_path: Path) -> None:
        """A fresh database has parsed_at, parse_error, and sort_key columns."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)

        cursor = conn.execute("PRAGMA table_info(raw_conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "parsed_at" in columns
        assert "parse_error" in columns

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
        """A fresh database has all v12+v13+v14+v15 indices."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in cursor.fetchall()}

        # v12 indices
        assert "idx_raw_conv_source_mtime" in indices

        # v13 indices
        assert "idx_raw_conv_unparsed" in indices
        assert "idx_conversations_content_hash" in indices

        # v14 index (replaces idx_messages_conversation_ts)
        assert "idx_messages_conversation_sortkey" in indices

        # v15 index
        assert "idx_conversations_sortkey" in indices
        conn.close()

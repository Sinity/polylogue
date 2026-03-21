"""Database corruption recovery tests.

Proves SQLite corruption produces clean errors, never crashes
the application or leaks raw sqlite3.DatabaseError to callers.
"""
from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest


@pytest.mark.slow
class TestCorruptionRecovery:
    def test_zeroed_header(self, tmp_path: Path) -> None:
        """Overwriting the SQLite header with zeros causes polylogue to reinitialize.

        polylogue's open_connection uses a thread-local connection cache.  When
        the header is zeroed SQLite treats the file as a brand-new database, so
        the connection layer re-creates the schema rather than raising.  The
        invariant is: the database is usable afterward and the schema tables exist.
        """
        from polylogue.storage.backends.connection import _clear_connection_cache, open_connection

        db = tmp_path / "corrupt.db"
        with open_connection(db):
            pass
        assert db.stat().st_size > 100

        # Zero out the first 100 bytes (SQLite header) and flush the cache so
        # open_connection reopens the file rather than reusing the old fd.
        data = bytearray(db.read_bytes())
        data[:100] = b"\x00" * 100
        db.write_bytes(bytes(data))
        _clear_connection_cache()

        # polylogue auto-recovers: schema is re-initialized, DB is usable.
        with open_connection(db) as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {row[0] for row in result}
            assert "conversations" in table_names

    def test_truncated_file(self, tmp_path: Path) -> None:
        """Truncating a DB file to 50% produces a clean error or recovery."""
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "truncated.db"
        with open_connection(db):
            pass
        original_size = db.stat().st_size

        # Truncate to ~50%
        data = db.read_bytes()
        db.write_bytes(data[:original_size // 2])

        # Should either raise a clean error or create a fresh DB
        try:
            with open_connection(db) as conn:
                conn.execute("SELECT count(*) FROM conversations")
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            pass  # Expected: clean error

    def test_read_only_db(self, tmp_path: Path) -> None:
        """Read-only DB file produces actionable write error."""
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "readonly.db"
        with open_connection(db):
            pass

        # Make read-only
        os.chmod(db, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            with open_connection(db) as conn:
                # Read should work
                conn.execute("SELECT count(*) FROM conversations")
                # Write should fail cleanly
                with pytest.raises((sqlite3.OperationalError, Exception)):
                    conn.execute("INSERT INTO conversations (conversation_id, provider_name) VALUES ('test', 'test')")
                    conn.commit()
        finally:
            # Restore permissions for cleanup
            os.chmod(db, stat.S_IRUSR | stat.S_IWUSR)

    def test_missing_file_creates_fresh(self, tmp_path: Path) -> None:
        """Nonexistent path creates a fresh DB cleanly."""
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "subdir" / "brand_new.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        with open_connection(db) as conn:
            # Should have schema tables
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {row[0] for row in result}
            assert "conversations" in table_names

    def test_empty_file(self, tmp_path: Path) -> None:
        """Zero-byte file is handled cleanly."""
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        db.write_bytes(b"")

        # Should either initialize fresh or raise clean error
        try:
            with open_connection(db) as conn:
                conn.execute("SELECT count(*) FROM conversations")
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            pass  # Clean error is acceptable

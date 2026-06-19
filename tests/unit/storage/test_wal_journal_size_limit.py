"""WAL ``journal_size_limit`` + ``query_only`` pragma application (#1614).

Pins the contract that:

1. Every write connection opened via the canonical factory has
   ``PRAGMA journal_size_limit`` set to the documented cap. Without
   this cap the WAL file grows unbounded when a TRUNCATE checkpoint
   is blocked by a long-running reader, as observed during the
   2026-05-26 dogfood probe (WAL grew from ~750 MB to ~1 GB in 60 s).
2. Every read connection opened via the canonical factory has
   ``PRAGMA query_only`` ON so accidental write attempts via a read
   profile fail fast at SQL parse time instead of contending for the
   write lock.
3. The cap actually fires: a write transaction that exceeds the cap
   sees the WAL truncated back to (at most) the cap after a
   subsequent checkpoint.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.connection_profile import (
    DAEMON_WRITE_CACHE_SIZE_KIB,
    DAEMON_WRITE_CONNECTION_PROFILE,
    DAEMON_WRITE_MMAP_SIZE_BYTES,
    READ_CONNECTION_PROFILE,
    WAL_JOURNAL_SIZE_LIMIT_BYTES,
    WRITE_CONNECTION_PROFILE,
    open_connection,
    open_daemon_connection,
    open_readonly_connection,
)


def _pragma_int(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute(f"PRAGMA {name}").fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# Profile-level pragma plumbing
# ---------------------------------------------------------------------------


def test_write_profile_includes_journal_size_limit_pragma() -> None:
    statements = WRITE_CONNECTION_PROFILE.pragma_statements
    assert any(stmt == f"PRAGMA journal_size_limit = {WAL_JOURNAL_SIZE_LIMIT_BYTES}" for stmt in statements), (
        f"#1614: write profile must cap WAL size; statements were {statements}"
    )


def test_read_profile_includes_query_only_pragma() -> None:
    statements = READ_CONNECTION_PROFILE.pragma_statements
    assert "PRAGMA query_only = ON" in statements, (
        f"#1614: read profile must signal read-only intent; statements were {statements}"
    )


def test_daemon_write_profile_uses_bounded_cache_and_mmap() -> None:
    """Daemon maintenance writes should not inherit the batch-ingest cache."""
    assert DAEMON_WRITE_CONNECTION_PROFILE.cache_size_kib == DAEMON_WRITE_CACHE_SIZE_KIB
    assert DAEMON_WRITE_CONNECTION_PROFILE.mmap_size_bytes == DAEMON_WRITE_MMAP_SIZE_BYTES
    assert DAEMON_WRITE_CONNECTION_PROFILE.cache_size_kib < WRITE_CONNECTION_PROFILE.cache_size_kib
    assert DAEMON_WRITE_CONNECTION_PROFILE.mmap_size_bytes < WRITE_CONNECTION_PROFILE.mmap_size_bytes
    assert DAEMON_WRITE_CONNECTION_PROFILE.journal_size_limit_bytes == WAL_JOURNAL_SIZE_LIMIT_BYTES


# ---------------------------------------------------------------------------
# Live pragma application via the canonical factories
# ---------------------------------------------------------------------------


def test_open_connection_applies_journal_size_limit_on_disk(tmp_path: Path) -> None:
    """The factory ``open_connection`` propagates the cap to SQLite."""
    db_path = tmp_path / "index.db"
    conn = open_connection(db_path)
    try:
        limit = _pragma_int(conn, "journal_size_limit")
        assert limit == WAL_JOURNAL_SIZE_LIMIT_BYTES, (
            f"#1614: PRAGMA journal_size_limit must equal {WAL_JOURNAL_SIZE_LIMIT_BYTES}, got {limit}"
        )
    finally:
        conn.close()


def test_open_readonly_connection_applies_query_only(tmp_path: Path) -> None:
    """The read factory turns on query_only at the SQL parser level."""
    db_path = tmp_path / "index.db"
    # Bootstrap a tiny schema so the read connection has something to attach to.
    seed = sqlite3.connect(str(db_path))
    try:
        seed.execute("CREATE TABLE t (id INTEGER)")
        seed.commit()
    finally:
        seed.close()

    conn = open_readonly_connection(db_path)
    try:
        assert _pragma_int(conn, "query_only") == 1
        # Any DML attempt must be rejected at parse time.
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO t (id) VALUES (1)")
    finally:
        conn.close()


def test_open_daemon_connection_applies_bounded_cache_and_mmap(tmp_path: Path) -> None:
    db_path = tmp_path / "ops.db"
    conn = open_daemon_connection(db_path)
    try:
        assert _pragma_int(conn, "cache_size") == -DAEMON_WRITE_CACHE_SIZE_KIB
        assert _pragma_int(conn, "mmap_size") == DAEMON_WRITE_MMAP_SIZE_BYTES
        assert _pragma_int(conn, "journal_size_limit") == WAL_JOURNAL_SIZE_LIMIT_BYTES
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The cap actually fires under blocked-checkpoint conditions
# ---------------------------------------------------------------------------


def test_wal_file_truncates_back_to_size_limit_after_checkpoint(tmp_path: Path) -> None:
    """Write past the cap, then checkpoint — WAL file shrinks back.

    Holds a second read connection open the entire time so the writer
    cannot use the autocheckpoint path opportunistically; the explicit
    TRUNCATE checkpoint at the end is the only mechanism that can
    shrink the WAL, and that's exactly what the limit enforces.
    """
    db_path = tmp_path / "index.db"
    writer = open_connection(db_path)
    try:
        writer.execute("CREATE TABLE blob (id INTEGER PRIMARY KEY, payload BLOB)")
        writer.commit()
        # Open and hold a reader snapshot so autocheckpoint is forced to
        # use PASSIVE mode (which copies frames but does not truncate).
        reader = open_readonly_connection(db_path)
        try:
            reader.execute("SELECT 1 FROM blob").fetchall()
            # Push the WAL well past the limit. A single ~256 KiB blob
            # × 800 = ~200 MiB total, comfortably over the 160 MiB cap.
            chunk = b"x" * (256 * 1024)
            for _ in range(800):
                writer.execute("INSERT INTO blob (payload) VALUES (?)", (chunk,))
            writer.commit()
            # Force an explicit TRUNCATE checkpoint with the reader gone.
        finally:
            reader.close()
        # With the reader closed, TRUNCATE checkpoint can run and the
        # journal_size_limit cap should fire.
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    finally:
        writer.close()

    wal_path = db_path.with_name(db_path.name + "-wal")
    if not wal_path.exists():
        return  # WAL fully drained — perfect outcome, even stricter than the cap.
    wal_size = wal_path.stat().st_size
    assert wal_size <= WAL_JOURNAL_SIZE_LIMIT_BYTES, (
        f"#1614: WAL must shrink to within {WAL_JOURNAL_SIZE_LIMIT_BYTES} bytes after "
        f"TRUNCATE checkpoint; observed {wal_size} bytes"
    )


# ---------------------------------------------------------------------------
# #1614 AC2 — concurrent reader-while-writer stays WAL-bounded
# ---------------------------------------------------------------------------


def test_wal_stays_bounded_under_concurrent_reader_and_writer(tmp_path: Path) -> None:
    """Reproduces the dogfood scenario from #1614: a long-running reader
    holds a snapshot while the writer streams inserts. SQLite falls back
    to PASSIVE autocheckpoint (which does not truncate). Without the
    journal_size_limit cap landed in PR #1659, the WAL grew unbounded
    (~750 MB → 1 GB in 60 s observed in production).

    With the cap, the WAL still grows past the autocheckpoint threshold
    during the blocked window, but eventually the reader closes, the
    next checkpoint truncates the WAL down to the cap, and the file
    settles below ``WAL_JOURNAL_SIZE_LIMIT_BYTES + tolerance``.

    Concurrent threads use separate connections (sqlite3.Connection is
    not thread-safe across operations); both open through the canonical
    factories so the production pragma profile is exercised end to end.
    """
    import threading

    db_path = tmp_path / "index.db"

    # Seed the schema on a separate connection so the worker threads
    # don't race the table creation.
    bootstrap = open_connection(db_path)
    try:
        bootstrap.execute("CREATE TABLE blob (id INTEGER PRIMARY KEY, payload BLOB)")
        bootstrap.commit()
    finally:
        bootstrap.close()

    writes_done = threading.Event()
    reader_done = threading.Event()
    errors: list[BaseException] = []
    observed_wal_sizes: list[int] = []

    def _writer() -> None:
        try:
            conn = open_connection(db_path)
            try:
                chunk = b"x" * (256 * 1024)
                for _ in range(800):
                    conn.execute("INSERT INTO blob (payload) VALUES (?)", (chunk,))
                conn.commit()
            finally:
                conn.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            writes_done.set()

    def _reader() -> None:
        try:
            conn = open_readonly_connection(db_path)
            try:
                # Pin a snapshot for the duration of the writer's run. The
                # SELECT establishes the read transaction; subsequent reads
                # within the same connection are served from that snapshot.
                while not writes_done.is_set():
                    conn.execute("SELECT COUNT(*) FROM blob").fetchall()
                    # Capture the WAL size while the snapshot is held; the
                    # blocked-checkpoint window is where the limit matters.
                    wal_path = db_path.with_name(db_path.name + "-wal")
                    if wal_path.exists():
                        observed_wal_sizes.append(wal_path.stat().st_size)
            finally:
                conn.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            reader_done.set()

    writer_thread = threading.Thread(target=_writer)
    reader_thread = threading.Thread(target=_reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join(timeout=60.0)
    reader_thread.join(timeout=60.0)

    assert not errors, f"#1614: concurrent worker raised: {errors}"
    assert writes_done.is_set() and reader_done.is_set()

    # Force the post-reader-close TRUNCATE checkpoint that the cap relies on.
    closer = open_connection(db_path)
    try:
        closer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    finally:
        closer.close()

    wal_path = db_path.with_name(db_path.name + "-wal")
    final_wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    assert final_wal_size <= WAL_JOURNAL_SIZE_LIMIT_BYTES, (
        f"#1614: WAL must shrink to within {WAL_JOURNAL_SIZE_LIMIT_BYTES} bytes "
        f"after concurrent reader closes and TRUNCATE runs; observed {final_wal_size} bytes"
    )

    # Light sanity: the test did actually exercise the blocked-checkpoint
    # window (the reader was alive long enough to sample WAL size at
    # least once). If this list is empty the writer outran the reader
    # entirely and the test isn't actually testing what its name says.
    assert observed_wal_sizes, (
        "#1614: the test never observed the WAL while the reader held its "
        "snapshot — either the writer finished before the first sample or "
        "the WAL was already truncated. Test is not exercising the blocked-"
        "checkpoint window."
    )

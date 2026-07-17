"""Canonical SQLite connection profiles and factory functions shared by sync and async backends.

Factories
---------
``open_connection(path)`` returns a read-write connection with write pragmas applied.
``open_daemon_connection(path)`` returns a read-write connection with a smaller
daemon/ops cache profile.
``open_readonly_connection(path)`` returns a uri=ro connection with read pragmas applied.
``connection_context(path)`` is a context manager for a single-use read-write connection.

These are lightweight one-shot wrappers around ``sqlite3.connect()``.  For the
thread-local cached connection used by the async runtime, use the factories in
``connection.py`` instead.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class SQLiteConnectionProfile:
    """SQLite timeout and PRAGMA profile for one connection role."""

    role: Literal["read", "write"]
    timeout_seconds: int
    busy_timeout_ms: int
    cache_size_kib: int
    mmap_size_bytes: int
    foreign_keys: bool = False
    journal_mode: str | None = None
    synchronous: str | None = None
    temp_store: str = "MEMORY"
    wal_autocheckpoint_pages: int | None = None
    journal_size_limit_bytes: int | None = None
    query_only: bool = False

    @property
    def pragma_statements(self) -> tuple[str, ...]:
        statements: list[str] = []
        if self.foreign_keys:
            statements.append("PRAGMA foreign_keys = ON")
        if self.journal_mode is not None:
            statements.append(f"PRAGMA journal_mode={self.journal_mode}")
        statements.extend(
            (
                f"PRAGMA busy_timeout = {self.busy_timeout_ms}",
                f"PRAGMA cache_size = -{self.cache_size_kib}",
            )
        )
        if self.synchronous is not None:
            statements.append(f"PRAGMA synchronous = {self.synchronous}")
        statements.extend(
            (
                f"PRAGMA mmap_size = {self.mmap_size_bytes}",
                f"PRAGMA temp_store = {self.temp_store}",
            )
        )
        if self.wal_autocheckpoint_pages is not None:
            statements.append(f"PRAGMA wal_autocheckpoint = {self.wal_autocheckpoint_pages}")
        if self.journal_size_limit_bytes is not None:
            statements.append(f"PRAGMA journal_size_limit = {self.journal_size_limit_bytes}")
        if self.query_only:
            statements.append("PRAGMA query_only = ON")
        return tuple(statements)


DB_TIMEOUT = 30
# Read busy_timeout. WAL readers normally don't block on a writer, but the
# brief window where a writer holds an exclusive lock (commit + TRUNCATE
# checkpoint) can exceed a second on a multi-GiB archive. A 1 s timeout turned
# that transient window into a hard "database is locked" error on interactive
# read surfaces (e.g. `polylogue find` during daemon ingest); 5 s lets the read
# wait out the checkpoint and succeed while staying far below the 30 s writer
# timeout, so reads remain responsive.
READ_DB_TIMEOUT = 5
WRITE_CACHE_SIZE_KIB = 131072  # 128 MiB
DAEMON_WRITE_CACHE_SIZE_KIB = 16384  # 16 MiB
READ_CACHE_SIZE_KIB = 32768  # 32 MiB
WRITE_MMAP_SIZE_BYTES = 1073741824  # 1 GiB
DAEMON_WRITE_MMAP_SIZE_BYTES = 67108864  # 64 MiB
READ_MMAP_SIZE_BYTES = 134217728  # 128 MiB
WAL_AUTOCHECKPOINT_PAGES = 10000
# #1614: soft cap on the WAL file. After any checkpoint that frees
# pages, SQLite truncates the WAL down to this size. Without this cap
# the WAL grows unbounded when a TRUNCATE checkpoint is blocked by a
# long-running reader — the dogfood probe reproducibly grew it from
# ~750 MB to ~1 GB in 60 s during catch-up. 160 MiB = 4x the
# autocheckpoint threshold (40 MiB), so a healthy autocheckpoint
# cycle does not trip the limit but a reader-blocked WAL eventually
# hits it and shrinks on the next successful checkpoint.
WAL_JOURNAL_SIZE_LIMIT_BYTES = 160 * 1024 * 1024

WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="NORMAL",
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
    journal_size_limit_bytes=WAL_JOURNAL_SIZE_LIMIT_BYTES,
)

DAEMON_WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=DAEMON_WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=DAEMON_WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="NORMAL",
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
    journal_size_limit_bytes=WAL_JOURNAL_SIZE_LIMIT_BYTES,
)

READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=READ_DB_TIMEOUT,
    busy_timeout_ms=READ_DB_TIMEOUT * 1000,
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    # #1614: explicit read-only signal. ``open_readonly_connection``
    # opens with the ``mode=ro`` URI flag which is already enforced
    # by SQLite at the file level, but the pragma additionally
    # rejects accidental writes via the same connection at SQL parse
    # time instead of waiting for the write lock.
    query_only=True,
)

DAEMON_WRITE_CONNECTION_PRAGMA_STATEMENTS = DAEMON_WRITE_CONNECTION_PROFILE.pragma_statements
WRITE_CONNECTION_PRAGMA_STATEMENTS = WRITE_CONNECTION_PROFILE.pragma_statements
READ_CONNECTION_PRAGMA_STATEMENTS = READ_CONNECTION_PROFILE.pragma_statements


# ---------------------------------------------------------------------------
# Lightweight factory functions — open + apply pragmas, no caching / schema / vec
# ---------------------------------------------------------------------------


_SIBLING_TIER_ATTACHMENTS: tuple[tuple[str, str], ...] = (
    ("source_tier", "source.db"),
    ("user_tier", "user.db"),
    ("embeddings", "embeddings.db"),
    ("ops_tier", "ops.db"),
)


def _attach_sibling_tiers(conn: sqlite3.Connection) -> None:
    """Attach sibling archive tiers to an ``index.db`` connection (idempotent).

    Lets one-shot sync connections resolve cross-tier tables (e.g. source.db's
    ``raw_sessions``/``blob_refs``) by unqualified name. SQLite resolves
    unqualified names to ``main`` first, so index-tier tables are unaffected;
    only sibling-only tables resolve to their attached tier.
    """
    main_path: str | None = None
    attached: set[str] = set()
    for row in conn.execute("PRAGMA database_list").fetchall():
        schema_name = str(row[1])
        if schema_name == "main":
            main_path = str(row[2]) if row[2] else None
        else:
            attached.add(schema_name)
    if not main_path:
        return
    main = Path(main_path)
    if main.name != "index.db":
        return
    root = main.parent
    for schema_name, filename in _SIBLING_TIER_ATTACHMENTS:
        if schema_name in attached:
            continue
        sibling = root / filename
        if sibling.exists():
            conn.execute(f"ATTACH DATABASE ? AS {schema_name}", (str(sibling),))


def open_connection(path: str | Path, *, timeout: float = DB_TIMEOUT) -> sqlite3.Connection:
    """Open a read-write SQLite connection with canonical write pragmas applied.

    This is a lightweight one-shot factory: it opens the file, applies the
    write-time PRAGMA profile, attaches sibling archive tiers (so cross-tier
    reads resolve), and returns the connection.  The caller owns the connection
    lifecycle (must close it).

    For the thread-local cached archive connection used by the async runtime,
    use ``connection_context`` from ``connection.py`` instead.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        for stmt in WRITE_CONNECTION_PRAGMA_STATEMENTS:
            conn.execute(stmt)
        _attach_sibling_tiers(conn)
    except BaseException:
        # A pragma can fail (e.g. a WAL-mode write pragma against a
        # lock-held database). Close the just-opened connection before
        # propagating so it is not orphaned by the caller's ``with``/``closing``.
        conn.close()
        raise
    return conn


def open_daemon_connection(
    path: str | Path,
    *,
    timeout: float = DB_TIMEOUT,
    busy_timeout_ms: int | None = None,
) -> sqlite3.Connection:
    """Open a read-write SQLite connection for daemon maintenance/ops writes.

    Long-running daemon loops write small status, cursor, telemetry, and
    maintenance rows. They should not inherit the full batch-ingest cache and
    mmap profile, because systemd charges their SQLite page cache to the
    service cgroup for the lifetime of the process.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        for stmt in DAEMON_WRITE_CONNECTION_PRAGMA_STATEMENTS:
            if busy_timeout_ms is not None and stmt.startswith("PRAGMA busy_timeout"):
                stmt = f"PRAGMA busy_timeout = {busy_timeout_ms}"
            conn.execute(stmt)
        _attach_sibling_tiers(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def open_readonly_connection(path: str | Path, *, timeout: float = READ_DB_TIMEOUT) -> sqlite3.Connection:
    """Open a read-only SQLite connection with canonical read pragmas applied.

    Uses ``file:...?mode=ro`` URI mode to guarantee no write locks are taken.
    Returns ``None`` / raises ``sqlite3.OperationalError`` if the database file
    does not exist.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    try:
        for stmt in READ_CONNECTION_PRAGMA_STATEMENTS:
            conn.execute(stmt)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def connection_context(path: str | Path, *, timeout: float = DB_TIMEOUT) -> Iterator[sqlite3.Connection]:
    """Context manager for a single-use read-write connection.

    Opens a connection with write pragmas, yields it, and closes on exit.
    """
    conn = open_connection(path, timeout=timeout)
    try:
        yield conn
    finally:
        conn.close()


__all__ = [
    "DB_TIMEOUT",
    "DAEMON_WRITE_CACHE_SIZE_KIB",
    "DAEMON_WRITE_CONNECTION_PRAGMA_STATEMENTS",
    "DAEMON_WRITE_CONNECTION_PROFILE",
    "DAEMON_WRITE_MMAP_SIZE_BYTES",
    "READ_CACHE_SIZE_KIB",
    "READ_CONNECTION_PRAGMA_STATEMENTS",
    "READ_CONNECTION_PROFILE",
    "READ_DB_TIMEOUT",
    "READ_MMAP_SIZE_BYTES",
    "SQLiteConnectionProfile",
    "WAL_AUTOCHECKPOINT_PAGES",
    "WRITE_CACHE_SIZE_KIB",
    "WRITE_CONNECTION_PRAGMA_STATEMENTS",
    "WRITE_CONNECTION_PROFILE",
    "WRITE_MMAP_SIZE_BYTES",
    "connection_context",
    "open_daemon_connection",
    "open_connection",
    "open_readonly_connection",
]

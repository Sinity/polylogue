"""Canonical SQLite schema runtime for sync and async backends.

Polylogue has no in-place schema upgrade chain. Databases are either at the
canonical :data:`SCHEMA_VERSION` (open as-is), fresh (bootstrap from
:data:`SCHEMA_DDL`), or rejected. The operator handles rejection by
re-ingesting from source; the runtime never patches an out-of-band shape
into the canonical one.
"""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.errors import SchemaVersionMismatchError
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_async, ensure_runtime_indexes_sync
from polylogue.storage.sqlite.schema_bootstrap import (
    SCHEMA_DDL,
    SCHEMA_VERSION,
    capture_schema_snapshot,
    capture_schema_snapshot_async,
    decide_schema_bootstrap,
    ensure_vec0_table,
    ensure_vec0_table_async,
    schema_version_mismatch_message,
)


def assert_supported_archive_layout(conn: sqlite3.Connection) -> None:
    """Reject archive layouts that cannot be written safely.

    Polylogue has no in-place schema upgrade chain; layout is determined entirely by
    the on-disk ``user_version``. Anything outside ``{0, SCHEMA_VERSION}`` is
    rejected and the operator re-ingests from source.
    """
    snapshot = capture_schema_snapshot(conn)
    if snapshot.current_version not in (0, SCHEMA_VERSION):
        raise SchemaVersionMismatchError(
            schema_version_mismatch_message(snapshot.current_version),
            current_version=snapshot.current_version,
            expected_version=SCHEMA_VERSION,
        )


def assert_readable_archive_layout(conn: sqlite3.Connection) -> None:
    """Read-only mode counterpart of :func:`assert_supported_archive_layout`."""
    assert_supported_archive_layout(conn)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the database is at the current schema version.

    Polylogue has no versioned in-place upgrade chain. Databases with a mismatched
    schema version are rejected; the operator re-ingests from source.
    """
    snapshot = capture_schema_snapshot(conn)
    decision = decide_schema_bootstrap(snapshot)

    if decision.action == "create_fresh":
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_DDL)
        ensure_vec0_table(conn)
        ensure_runtime_indexes_sync(conn)
        conn.execute("PRAGMA optimize")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        return

    if decision.action == "version_mismatch":
        current = decision.current_version or 0
        raise SchemaVersionMismatchError(
            schema_version_mismatch_message(current),
            current_version=current,
            expected_version=SCHEMA_VERSION,
        )

    # open_as_is — vec0 still needs to be ensured per-connection because the
    # extension may have been newly loaded since fresh init.
    ensure_vec0_table(conn)
    ensure_runtime_indexes_sync(conn)


async def ensure_schema_async(conn: aiosqlite.Connection) -> None:
    """Async counterpart of :func:`_ensure_schema`. Same policy."""
    snapshot = await capture_schema_snapshot_async(conn)
    decision = decide_schema_bootstrap(snapshot)

    if decision.action == "create_fresh":
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.executescript(SCHEMA_DDL)
        await ensure_vec0_table_async(conn)
        await ensure_runtime_indexes_async(conn)
        await conn.execute("PRAGMA optimize")
        await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await conn.commit()
        return

    if decision.action == "version_mismatch":
        current = decision.current_version or 0
        raise SchemaVersionMismatchError(
            schema_version_mismatch_message(current),
            current_version=current,
            expected_version=SCHEMA_VERSION,
        )

    await ensure_vec0_table_async(conn)
    await ensure_runtime_indexes_async(conn)


__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "_ensure_schema",
    "assert_readable_archive_layout",
    "assert_supported_archive_layout",
    "ensure_schema_async",
    "ensure_vec0_table",
]

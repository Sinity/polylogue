"""Schema bootstrap helpers for the async SQLite backend."""

from __future__ import annotations

import aiosqlite

from polylogue.errors import DatabaseError
from polylogue.storage.backends.schema_bootstrap import (
    SCHEMA_DDL,
    SCHEMA_VERSION,
    apply_schema_extension_plan_async,
    assert_supported_archive_layout_snapshot,
    build_current_schema_extension_plan,
    capture_schema_snapshot_async,
    ensure_vec0_table_async,
    schema_version_mismatch_message,
)


async def ensure_schema(conn: aiosqlite.Connection) -> None:
    """Ensure database schema exists and is at the current schema version."""
    snapshot = await capture_schema_snapshot_async(conn)

    if snapshot.current_version == 0:
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.executescript(SCHEMA_DDL)
        await ensure_vec0_table_async(conn)
        await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await conn.commit()
        return

    assert_supported_archive_layout_snapshot(snapshot)

    if snapshot.current_version != SCHEMA_VERSION:
        raise DatabaseError(schema_version_mismatch_message(snapshot.current_version))

    plan = build_current_schema_extension_plan(snapshot)
    await apply_schema_extension_plan_async(conn, plan)
    await ensure_vec0_table_async(conn)
    await conn.commit()


__all__ = ["ensure_schema"]

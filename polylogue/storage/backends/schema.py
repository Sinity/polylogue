"""Canonical SQLite schema runtime for sync and async backends."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.errors import DatabaseError
from polylogue.logging import get_logger
from polylogue.storage.backends.schema_bootstrap import (
    SCHEMA_DDL,
    SCHEMA_VERSION,
    SchemaExtensionPlan,
    SchemaSnapshot,
    apply_schema_extension_plan,
    apply_schema_extension_plan_async,
    assert_supported_archive_layout_snapshot,
    build_current_schema_extension_plan,
    capture_schema_snapshot,
    capture_schema_snapshot_async,
    ensure_vec0_table,
    ensure_vec0_table_async,
    schema_version_mismatch_message,
)

logger = get_logger(__name__)


def _ensure_raw_source_mtime_index(conn: sqlite3.Connection) -> None:
    snapshot = capture_schema_snapshot(conn)
    plan = build_current_schema_extension_plan(snapshot)
    for statement in plan.statements:
        if "idx_raw_conv_source_mtime" in statement:
            conn.execute(statement)


def assert_supported_archive_layout(conn: sqlite3.Connection) -> None:
    """Reject legacy archive layouts that the current runtime cannot write safely."""
    assert_supported_archive_layout_snapshot(capture_schema_snapshot(conn))


def _log_index_replacement(snapshot: SchemaSnapshot, plan: SchemaExtensionPlan) -> None:
    if snapshot.sql_for_index("idx_raw_conv_source_mtime") is not None and any(
        statement == "DROP INDEX IF EXISTS idx_raw_conv_source_mtime" for statement in plan.statements
    ):
        logger.info("Replacing idx_raw_conv_source_mtime with partial covering definition")


def _apply_extensions_for_snapshot(conn: sqlite3.Connection, snapshot: SchemaSnapshot) -> None:
    plan = build_current_schema_extension_plan(snapshot)
    _log_index_replacement(snapshot, plan)
    apply_schema_extension_plan(conn, plan)
    ensure_vec0_table(conn)
    conn.commit()


async def _apply_extensions_for_snapshot_async(conn: aiosqlite.Connection, snapshot: SchemaSnapshot) -> None:
    plan = build_current_schema_extension_plan(snapshot)
    _log_index_replacement(snapshot, plan)
    await apply_schema_extension_plan_async(conn, plan)
    await ensure_vec0_table_async(conn)
    await conn.commit()


def apply_current_schema_extensions(conn: sqlite3.Connection) -> None:
    _apply_extensions_for_snapshot(conn, capture_schema_snapshot(conn))


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the database is at the current schema version.

    Control flow mirrors ensure_schema_async exactly:
    fresh-create → version-mismatch rejection → extension application.
    """
    snapshot = capture_schema_snapshot(conn)

    if snapshot.current_version == 0:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_DDL)
        ensure_vec0_table(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        logger.debug("Created fresh schema v%s", SCHEMA_VERSION)
        return

    assert_supported_archive_layout_snapshot(snapshot)

    if snapshot.current_version != SCHEMA_VERSION:
        raise DatabaseError(schema_version_mismatch_message(snapshot.current_version))

    _apply_extensions_for_snapshot(conn, snapshot)


async def ensure_schema_async(conn: aiosqlite.Connection) -> None:
    """Ensure the async connection is at the current schema version.

    Control flow mirrors _ensure_schema exactly:
    fresh-create → version-mismatch rejection → extension application.
    """
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

    await _apply_extensions_for_snapshot_async(conn, snapshot)


__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "_ensure_raw_source_mtime_index",
    "_ensure_schema",
    "apply_current_schema_extensions",
    "assert_supported_archive_layout",
    "ensure_schema_async",
    "ensure_vec0_table",
]

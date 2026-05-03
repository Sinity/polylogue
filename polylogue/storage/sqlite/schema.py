"""Canonical SQLite schema runtime for sync and async backends."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.errors import DatabaseError
from polylogue.logging import get_logger
from polylogue.storage.sqlite.schema_bootstrap import (
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
    decide_schema_bootstrap,
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


def assert_readable_archive_layout(conn: sqlite3.Connection) -> None:
    """Reject archive layouts that cannot be interpreted safely in read-only mode."""
    snapshot = capture_schema_snapshot(conn)
    assert_supported_archive_layout_snapshot(snapshot)

    if snapshot.current_version != SCHEMA_VERSION:
        raise DatabaseError(schema_version_mismatch_message(snapshot.current_version))


def _log_index_replacement(snapshot: SchemaSnapshot, plan: SchemaExtensionPlan) -> None:
    if snapshot.sql_for_index("idx_raw_conv_source_mtime") is not None and any(
        statement == "DROP INDEX IF EXISTS idx_raw_conv_source_mtime" for statement in plan.statements
    ):
        logger.info("Replacing idx_raw_conv_source_mtime with partial covering definition")


def _apply_extensions_for_plan(
    conn: sqlite3.Connection,
    snapshot: SchemaSnapshot,
    plan: SchemaExtensionPlan,
) -> None:
    _log_index_replacement(snapshot, plan)
    apply_schema_extension_plan(conn, plan)
    ensure_vec0_table(conn)
    conn.commit()


async def _apply_extensions_for_plan_async(
    conn: aiosqlite.Connection,
    snapshot: SchemaSnapshot,
    plan: SchemaExtensionPlan,
) -> None:
    _log_index_replacement(snapshot, plan)
    await apply_schema_extension_plan_async(conn, plan)
    await ensure_vec0_table_async(conn)
    await conn.commit()


def _apply_version_upgrade_plan(
    conn: sqlite3.Connection,
    snapshot: SchemaSnapshot,
    plan: SchemaExtensionPlan,
) -> None:
    if plan.scripts and plan.statements:
        raise DatabaseError("Schema version upgrade plans must use statement-level SQL or scripts, not both")
    _log_index_replacement(snapshot, plan)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for statement in plan.statements:
            conn.execute(statement)
        for script in plan.scripts:
            conn.executescript(script)
        ensure_vec0_table(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    except Exception:
        conn.rollback()
        raise
    conn.commit()


async def _apply_version_upgrade_plan_async(
    conn: aiosqlite.Connection,
    snapshot: SchemaSnapshot,
    plan: SchemaExtensionPlan,
) -> None:
    if plan.scripts and plan.statements:
        raise DatabaseError("Schema version upgrade plans must use statement-level SQL or scripts, not both")
    _log_index_replacement(snapshot, plan)
    try:
        await conn.execute("BEGIN IMMEDIATE")
        for statement in plan.statements:
            await conn.execute(statement)
        for script in plan.scripts:
            await conn.executescript(script)
        await ensure_vec0_table_async(conn)
        await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    except Exception:
        await conn.rollback()
        raise
    await conn.commit()


def apply_current_schema_extensions(conn: sqlite3.Connection) -> None:
    snapshot = capture_schema_snapshot(conn)
    decision = decide_schema_bootstrap(snapshot)
    if decision.action == "apply_current_extensions" and decision.extension_plan is not None:
        _apply_extensions_for_plan(conn, snapshot, decision.extension_plan)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the database is at the current schema version.

    Polylogue has no versioned migration chain. Databases with a mismatched
    schema version are rejected rather than partially patched with additive DDL.
    """
    snapshot = capture_schema_snapshot(conn)

    decision = decide_schema_bootstrap(snapshot)

    if decision.action == "create_fresh":
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_DDL)
        ensure_vec0_table(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        logger.debug("Created fresh schema v%s", SCHEMA_VERSION)
        return

    if decision.action == "version_mismatch":
        raise DatabaseError(schema_version_mismatch_message(decision.current_version or 0))
        return

    if (
        decision.action in {"upgrade_v2_to_current", "upgrade_v3_to_v4", "upgrade_v4_to_v5"}
        and decision.extension_plan is not None
    ):
        _apply_version_upgrade_plan(conn, snapshot, decision.extension_plan)
        logger.info("Upgraded schema from v%s to v%s", decision.current_version, SCHEMA_VERSION)
        return

    if decision.extension_plan is not None:
        _apply_extensions_for_plan(conn, snapshot, decision.extension_plan)


async def ensure_schema_async(conn: aiosqlite.Connection) -> None:
    """Ensure the database is at the current schema version.

    Polylogue has no versioned migration chain. Databases with a mismatched
    schema version are rejected rather than partially patched with additive DDL.
    """
    snapshot = await capture_schema_snapshot_async(conn)

    decision = decide_schema_bootstrap(snapshot)

    if decision.action == "create_fresh":
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.executescript(SCHEMA_DDL)
        await ensure_vec0_table_async(conn)
        await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await conn.commit()
        return

    if decision.action == "version_mismatch":
        raise DatabaseError(schema_version_mismatch_message(decision.current_version or 0))
        return

    if (
        decision.action in {"upgrade_v2_to_current", "upgrade_v3_to_v4", "upgrade_v4_to_v5"}
        and decision.extension_plan is not None
    ):
        await _apply_version_upgrade_plan_async(conn, snapshot, decision.extension_plan)
        logger.info("Upgraded schema from v%s to v%s", decision.current_version, SCHEMA_VERSION)
        return

    if decision.extension_plan is not None:
        await _apply_extensions_for_plan_async(conn, snapshot, decision.extension_plan)


__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "_ensure_raw_source_mtime_index",
    "_ensure_schema",
    "apply_current_schema_extensions",
    "assert_readable_archive_layout",
    "assert_supported_archive_layout",
    "ensure_schema_async",
    "ensure_vec0_table",
]

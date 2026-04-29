"""Canonical SQLite schema runtime for sync and async backends."""

from __future__ import annotations

import sqlite3

import aiosqlite

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
    decide_schema_bootstrap,
    ensure_vec0_table,
    ensure_vec0_table_async,
    export_user_metadata,
    import_user_metadata,
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


def apply_current_schema_extensions(conn: sqlite3.Connection) -> None:
    snapshot = capture_schema_snapshot(conn)
    decision = decide_schema_bootstrap(snapshot)
    if decision.action == "apply_current_extensions" and decision.extension_plan is not None:
        _apply_extensions_for_plan(conn, snapshot, decision.extension_plan)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the database is at the current schema version.

    When the schema version has changed, user metadata (tags, summaries)
    is preserved across the wipe. Conversation data must be re-imported
    via ``polylogue run``.
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
        _handle_version_mismatch(conn, decision.current_version or 0)
        return

    if decision.extension_plan is not None:
        _apply_extensions_for_plan(conn, snapshot, decision.extension_plan)


def _handle_version_mismatch(conn: sqlite3.Connection, current_version: int) -> None:
    """Preserve user metadata across a schema version bump.

    Exports tags and summaries from the old-schema database, recreates
    with the current schema, and restores what was saved. Conversation
    data (messages, content blocks, FTS indexes, vectors) is not preserved
    and must be re-imported.
    """
    logger.warning(
        "Schema version mismatch: db=%s, expected=%s. Preserving user metadata.",
        current_version,
        SCHEMA_VERSION,
    )

    metadata = export_user_metadata(conn)
    tags = metadata.get("tags")
    summaries = metadata.get("summaries")
    tag_list: list[object] = tags if isinstance(tags, list) else []
    summary_list: list[object] = summaries if isinstance(summaries, list) else []

    if tag_list or summary_list:
        logger.info(
            "Exported user metadata: %s tags, %s summaries",
            len(tag_list),
            len(summary_list),
        )

    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_DDL)
    ensure_vec0_table(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()

    if tag_list or summary_list:
        restored = import_user_metadata(conn, metadata)
        conn.commit()
        logger.info("Restored %s user metadata rows", restored)

    logger.warning(
        "Schema upgraded from v%s to v%s. User metadata preserved; "
        "conversation data must be re-imported: polylogue run",
        current_version,
        SCHEMA_VERSION,
    )


async def ensure_schema_async(conn: aiosqlite.Connection) -> None:
    """Ensure the database is at the current schema version.

    When the schema version has changed, user metadata (tags, summaries)
    is preserved across the wipe. Conversation data must be re-imported
    via ``polylogue run``.
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
        await _handle_version_mismatch_async(conn, decision.current_version or 0)
        return

    if decision.extension_plan is not None:
        await _apply_extensions_for_plan_async(conn, snapshot, decision.extension_plan)


async def _handle_version_mismatch_async(conn: aiosqlite.Connection, current_version: int) -> None:
    """Preserve user metadata across a schema version bump (async variant).

    Exports tags and summaries before the wipe using the async connection,
    recreates the schema, then restores metadata through the same connection.
    """
    logger.warning(
        "Schema version mismatch: db=%s, expected=%s. Preserving user metadata.",
        current_version,
        SCHEMA_VERSION,
    )

    tags: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    try:
        cursor = await conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversation_tags'")
        if await cursor.fetchone():
            cursor = await conn.execute(
                "SELECT conversation_id, tag FROM conversation_tags ORDER BY conversation_id, tag"
            )
            rows = await cursor.fetchall()
            tags = [{"conversation_id": r[0], "tag": r[1]} for r in rows]
    except Exception:
        logger.warning("Failed to export tags", exc_info=True)

    try:
        cursor = await conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversations'")
        if await cursor.fetchone():
            cursor = await conn.execute(
                "SELECT conversation_id, summary FROM conversations WHERE summary IS NOT NULL AND summary != ''"
            )
            rows = await cursor.fetchall()
            summaries = [{"conversation_id": r[0], "summary": r[1]} for r in rows]
    except Exception:
        logger.warning("Failed to export summaries", exc_info=True)

    if tags or summaries:
        logger.info("Exported user metadata: %s tags, %s summaries", len(tags), len(summaries))

    await conn.execute("PRAGMA foreign_keys = ON")
    await conn.executescript(SCHEMA_DDL)
    await ensure_vec0_table_async(conn)
    await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    await conn.commit()

    restored = 0
    for tag_entry in tags:
        cid = tag_entry.get("conversation_id")
        tag = tag_entry.get("tag")
        if cid and tag:
            try:
                await conn.execute(
                    "INSERT OR IGNORE INTO conversation_tags (conversation_id, tag) VALUES (?, ?)",
                    (cid, tag),
                )
                restored += 1
            except Exception:
                pass

    for summary_entry in summaries:
        cid = summary_entry.get("conversation_id")
        summary = summary_entry.get("summary")
        if cid and summary:
            try:
                await conn.execute(
                    "UPDATE conversations SET summary = ? WHERE conversation_id = ?",
                    (summary, cid),
                )
                restored += 1
            except Exception:
                pass

    if restored:
        await conn.commit()
        logger.info("Restored %s user metadata rows", restored)

    logger.warning(
        "Schema upgraded from v%s to v%s. User metadata preserved; "
        "conversation data must be re-imported: polylogue run",
        current_version,
        SCHEMA_VERSION,
    )


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

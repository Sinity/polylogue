"""Bounded repair for archive FTS rows that are missing from shadow tables."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_sync
from polylogue.storage.fts.fts_lifecycle import (
    _triggers_present_sync,
    ensure_fts_index_sync,
)
from polylogue.storage.fts.sql import (
    ACTION_FTS_INDEX_DOC_COUNT_SQL,
    FTS_INDEX_DOC_COUNT_SQL,
    FTS_INDEXABLE_MESSAGE_COUNT_SQL,
    insert_all_missing_action_rows_sql,
    insert_missing_message_rows_sql,
    insert_missing_plain_message_rows_sql,
)

BOUNDED_REPAIR_PRAGMAS = (
    "PRAGMA temp_store = FILE",
    "PRAGMA cache_size = -32768",
    "PRAGMA mmap_size = 134217728",
)


@dataclass(frozen=True, slots=True)
class DanglingFtsRepairOutcome:
    repaired_count: int
    success: bool
    detail: str


def configure_bounded_repair_connection(conn: sqlite3.Connection) -> None:
    """Keep large maintenance SQL from using the interactive write profile's RAM budget."""
    for pragma in BOUNDED_REPAIR_PRAGMAS:
        conn.execute(pragma)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return bool(
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table_name,)).fetchone()
    )


def _message_trigger_names(*, content_blocks_exists: bool) -> tuple[str, ...]:
    if not content_blocks_exists:
        return ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")
    return (
        "messages_fts_ai",
        "messages_fts_ad",
        "messages_fts_au",
        "content_blocks_fts_ai",
        "content_blocks_fts_ad",
        "content_blocks_fts_au",
    )


def dry_run_dangling_fts_repair(conn: sqlite3.Connection) -> DanglingFtsRepairOutcome:
    msg_count = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0] or 0)
    fts_count = int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    diff = abs(msg_count - fts_count)
    if diff == 0:
        return DanglingFtsRepairOutcome(repaired_count=0, success=True, detail="FTS index in sync")
    return DanglingFtsRepairOutcome(
        repaired_count=diff,
        success=True,
        detail=f"Would: FTS sync: {msg_count:,} messages vs {fts_count:,} indexed ({diff:,} difference)",
    )


def insert_missing_message_fts_rows_sync(conn: sqlite3.Connection) -> int:
    """Insert globally missing message FTS rows without rebuilding existing rows."""
    ensure_fts_index_sync(conn)
    before = int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    if _table_exists(conn, "content_blocks"):
        conn.execute(insert_missing_message_rows_sql())
    else:
        conn.execute(insert_missing_plain_message_rows_sql())
    after = int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    return max(0, after - before)


def insert_missing_action_fts_rows_sync(conn: sqlite3.Connection) -> int:
    """Insert globally missing action-event FTS rows without rebuilding existing rows."""
    if not (_table_exists(conn, "action_events") and _table_exists(conn, "action_events_fts_docsize")):
        return 0
    ensure_fts_index_sync(conn)
    before = int(conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    conn.execute(insert_all_missing_action_rows_sql())
    after = int(conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    return max(0, after - before)


def repair_missing_fts_rows(conn: sqlite3.Connection) -> DanglingFtsRepairOutcome:
    """Repair missing message/action FTS rows without full invariant snapshots."""
    before_messages = int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    source_messages = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0] or 0)
    action_table_exists = _table_exists(conn, "action_events")
    before_actions = int(conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0) if action_table_exists else 0
    source_actions = (
        int(conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0] or 0) if action_table_exists else 0
    )

    message_excess = max(0, before_messages - source_messages)
    action_excess = max(0, before_actions - source_actions)
    if message_excess or action_excess:
        return DanglingFtsRepairOutcome(
            repaired_count=message_excess + action_excess,
            success=False,
            detail=(
                "FTS index has excess rows; targeted missing-row repair is unsafe "
                f"({message_excess:,} message rows, {action_excess:,} action rows)"
            ),
        )

    inserted_messages = insert_missing_message_fts_rows_sync(conn)
    inserted_actions = insert_missing_action_fts_rows_sync(conn)
    after_messages = int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    after_actions = int(conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0) if action_table_exists else 0
    message_ready = after_messages == source_messages and _triggers_present_sync(
        conn,
        _message_trigger_names(content_blocks_exists=_table_exists(conn, "content_blocks")),
    )
    action_ready = (not action_table_exists) or (
        after_actions == source_actions
        and _triggers_present_sync(conn, ("action_events_fts_ai", "action_events_fts_ad", "action_events_fts_au"))
    )

    record_fts_surface_state_sync(
        conn,
        surface="messages_fts",
        state=READY if message_ready else STALE,
        source_rows=source_messages,
        indexed_rows=after_messages,
        detail=None if message_ready else "message FTS count/trigger check failed after repair",
    )
    if action_table_exists:
        record_fts_surface_state_sync(
            conn,
            surface="action_events_fts",
            state=READY if action_ready else STALE,
            source_rows=source_actions,
            indexed_rows=after_actions,
            detail=None if action_ready else "action-event FTS count/trigger check failed after repair",
        )

    return DanglingFtsRepairOutcome(
        repaired_count=inserted_messages + inserted_actions,
        success=message_ready and action_ready,
        detail=(
            "FTS sync: repaired index "
            f"({after_messages:,}/{source_messages:,} messages, {after_actions:,}/{source_actions:,} actions)"
        ),
    )


__all__ = [
    "BOUNDED_REPAIR_PRAGMAS",
    "DanglingFtsRepairOutcome",
    "configure_bounded_repair_connection",
    "dry_run_dangling_fts_repair",
    "insert_missing_action_fts_rows_sync",
    "insert_missing_message_fts_rows_sync",
    "repair_missing_fts_rows",
]

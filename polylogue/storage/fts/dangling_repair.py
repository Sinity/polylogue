"""Bounded repair for archive FTS rows that are missing from shadow tables."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.storage.fts.freshness import READY, STALE, freshness_ready_record_trusted, record_fts_surface_state_sync
from polylogue.storage.fts.fts_lifecycle import (
    _triggers_present_sync,
    ensure_fts_index_sync,
)
from polylogue.storage.fts.sql import (
    FTS_INDEX_DOC_COUNT_SQL,
    FTS_INDEXABLE_MESSAGE_COUNT_SQL,
    insert_missing_message_rows_sql,
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


def _message_trigger_names() -> tuple[str, ...]:
    return (
        "messages_fts_ai",
        "messages_fts_ad",
        "messages_fts_au",
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
    conn.execute(insert_missing_message_rows_sql())
    after = int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    return max(0, after - before)


def _count_table(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] or 0)


def _repair_keyed_content_fts_sync(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    fts_table: str,
    key_column: str,
    insert_sql: str,
) -> tuple[int, int, int, int]:
    if not (_table_exists(conn, source_table) and _table_exists(conn, fts_table)):
        return (0, 0, 0, 0)
    before = _count_table(conn, f"{fts_table}_docsize")
    source = _count_table(conn, source_table)
    if before == source:
        return (0, source, before, 0)
    conn.execute(
        f"""
        DELETE FROM {fts_table}
        WHERE {key_column} NOT IN (SELECT {key_column} FROM {source_table})
        """
    )
    after_delete = _count_table(conn, f"{fts_table}_docsize")
    conn.execute(insert_sql)
    after = _count_table(conn, f"{fts_table}_docsize")
    return (max(0, after - after_delete), source, after, max(0, before - source))


def _repair_session_work_events_fts_rows_sync(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
    return _repair_keyed_content_fts_sync(
        conn,
        source_table="session_work_events",
        fts_table="session_work_events_fts",
        key_column="event_id",
        insert_sql="""
            INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
            SELECT swe.event_id, swe.session_id, swe.work_event_type, swe.search_text
            FROM session_work_events AS swe
            LEFT JOIN session_work_events_fts AS fts ON fts.event_id = swe.event_id
            WHERE fts.event_id IS NULL
        """,
    )


def _repair_threads_fts_rows_sync(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
    return _repair_keyed_content_fts_sync(
        conn,
        source_table="threads",
        fts_table="threads_fts",
        key_column="thread_id",
        insert_sql="""
            INSERT INTO threads_fts (thread_id, root_id, text)
            SELECT wt.thread_id, wt.thread_id AS root_id, wt.search_text
            FROM threads AS wt
            LEFT JOIN threads_fts AS fts ON fts.thread_id = wt.thread_id
            WHERE fts.thread_id IS NULL
        """,
    )


def _record_optional_derived_surface(
    conn: sqlite3.Connection,
    *,
    surface: str,
    source_rows: int,
    indexed_rows: int,
    triggers: tuple[str, ...],
) -> bool:
    if not _table_exists(conn, surface):
        return True
    ready = source_rows == indexed_rows and _triggers_present_sync(conn, triggers)
    record_fts_surface_state_sync(
        conn,
        surface=surface,
        state=READY if ready else STALE,
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        detail=None if ready else f"{surface} count/trigger check failed after repair",
    )
    return ready


def _source_table_for_surface(surface: str) -> str | None:
    return {
        "messages_fts": "messages",
        "session_work_events_fts": "session_work_events",
        "threads_fts": "threads",
    }.get(surface)


def _source_has_rows(conn: sqlite3.Connection, surface: str) -> bool | None:
    source_table = _source_table_for_surface(surface)
    if source_table is None or not _table_exists(conn, source_table):
        return None
    return conn.execute(f"SELECT 1 FROM {source_table} LIMIT 1").fetchone() is not None


def _freshness_record(conn: sqlite3.Connection, surface: str) -> dict[str, object] | None:
    if not _table_exists(conn, "fts_freshness_state"):
        return None
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(fts_freshness_state)").fetchall()}
    selected = ["state"]
    for name in ("source_rows", "indexed_rows", "missing_rows", "excess_rows", "duplicate_rows"):
        if name in columns:
            selected.append(name)
    row = conn.execute(f"SELECT {', '.join(selected)} FROM fts_freshness_state WHERE surface=?", (surface,)).fetchone()
    return None if row is None else dict(zip(selected, row, strict=True))


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return 0
    if value is None:
        return 0
    try:
        return int(str(value))
    except ValueError:
        return 0


def _ready_freshness_marker(conn: sqlite3.Connection, surface: str, triggers: tuple[str, ...]) -> bool:
    record = _freshness_record(conn, surface)
    if record is None or not _triggers_present_sync(conn, triggers):
        return False
    source_rows = _int_or_zero(record.get("source_rows"))
    indexed_rows = _int_or_zero(record.get("indexed_rows"))
    return freshness_ready_record_trusted(
        state=str(record["state"]),
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=_int_or_zero(record.get("missing_rows")),
        excess_rows=_int_or_zero(record.get("excess_rows")),
        duplicate_rows=_int_or_zero(record.get("duplicate_rows")),
        source_has_rows=_source_has_rows(conn, surface) if source_rows == 0 and indexed_rows == 0 else False,
    )


def repair_stale_fts_rows(conn: sqlite3.Connection) -> DanglingFtsRepairOutcome:
    """Repair only stale/unknown FTS surfaces when durable freshness is usable."""
    message_ready = _ready_freshness_marker(
        conn,
        "messages_fts",
        _message_trigger_names(),
    )
    if not message_ready:
        return repair_missing_fts_rows(conn)

    inserted_work_events, source_work_events, after_work_events, _excess_work_events = (
        _repair_session_work_events_fts_rows_sync(conn)
    )
    inserted_threads, source_threads, after_threads, _excess_threads = _repair_threads_fts_rows_sync(conn)
    work_event_ready = _record_optional_derived_surface(
        conn,
        surface="session_work_events_fts",
        source_rows=source_work_events,
        indexed_rows=after_work_events,
        triggers=("session_work_events_fts_ai", "session_work_events_fts_ad", "session_work_events_fts_au"),
    )
    thread_ready = _record_optional_derived_surface(
        conn,
        surface="threads_fts",
        source_rows=source_threads,
        indexed_rows=after_threads,
        triggers=("threads_fts_ai", "threads_fts_ad", "threads_fts_au"),
    )
    return DanglingFtsRepairOutcome(
        repaired_count=inserted_work_events + inserted_threads,
        success=work_event_ready and thread_ready,
        detail=(
            "FTS sync: repaired stale derived surfaces "
            f"({after_work_events:,}/{source_work_events:,} work events, {after_threads:,}/{source_threads:,} threads)"
        ),
    )


def repair_missing_fts_rows(conn: sqlite3.Connection) -> DanglingFtsRepairOutcome:
    """Repair missing message and derived FTS rows without full invariant snapshots."""
    before_messages = int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    source_messages = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0] or 0)
    message_excess = max(0, before_messages - source_messages)
    if message_excess:
        return DanglingFtsRepairOutcome(
            repaired_count=message_excess,
            success=False,
            detail=(
                f"FTS index has excess rows; targeted missing-row repair is unsafe ({message_excess:,} message rows)"
            ),
        )

    inserted_messages = insert_missing_message_fts_rows_sync(conn)
    inserted_work_events, source_work_events, after_work_events, _excess_work_events = (
        _repair_session_work_events_fts_rows_sync(conn)
    )
    inserted_threads, source_threads, after_threads, _excess_threads = _repair_threads_fts_rows_sync(conn)
    after_messages = int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0] or 0)
    message_ready = after_messages == source_messages and _triggers_present_sync(
        conn,
        _message_trigger_names(),
    )

    record_fts_surface_state_sync(
        conn,
        surface="messages_fts",
        state=READY if message_ready else STALE,
        source_rows=source_messages,
        indexed_rows=after_messages,
        detail=None if message_ready else "message FTS count/trigger check failed after repair",
    )
    work_event_ready = _record_optional_derived_surface(
        conn,
        surface="session_work_events_fts",
        source_rows=source_work_events,
        indexed_rows=after_work_events,
        triggers=("session_work_events_fts_ai", "session_work_events_fts_ad", "session_work_events_fts_au"),
    )
    thread_ready = _record_optional_derived_surface(
        conn,
        surface="threads_fts",
        source_rows=source_threads,
        indexed_rows=after_threads,
        triggers=("threads_fts_ai", "threads_fts_ad", "threads_fts_au"),
    )

    return DanglingFtsRepairOutcome(
        repaired_count=inserted_messages + inserted_work_events + inserted_threads,
        success=message_ready and work_event_ready and thread_ready,
        detail=(
            "FTS sync: repaired index "
            f"({after_messages:,}/{source_messages:,} messages, "
            f"{after_work_events:,}/{source_work_events:,} work events, {after_threads:,}/{source_threads:,} threads)"
        ),
    )


__all__ = [
    "BOUNDED_REPAIR_PRAGMAS",
    "DanglingFtsRepairOutcome",
    "configure_bounded_repair_connection",
    "dry_run_dangling_fts_repair",
    "insert_missing_message_fts_rows_sync",
    "repair_missing_fts_rows",
    "repair_stale_fts_rows",
]

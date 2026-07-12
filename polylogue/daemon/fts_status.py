"""FTS readiness projection for daemon status payloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.logging import get_logger
from polylogue.storage.fts.freshness import STALE, UNKNOWN, freshness_ready_record_trusted
from polylogue.storage.fts.fts_lifecycle import FtsInvariantSnapshot, FtsSurfaceInvariant, fts_invariant_snapshot_sync
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

_FTS_SURFACES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "messages_fts",
        "messages",
        "messages_fts",
        ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"),
    ),
    (
        "session_work_events_fts",
        "session_work_events",
        "session_work_events_fts",
        ("session_work_events_fts_ai", "session_work_events_fts_ad", "session_work_events_fts_au"),
    ),
    (
        "threads_fts",
        "threads",
        "threads_fts",
        ("threads_fts_ai", "threads_fts_ad", "threads_fts_au"),
    ),
)

_ARCHIVE_BLOCKS_FTS_TRIGGERS = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")
BOUNDED_MESSAGE_FTS_REPAIR_DETAIL = "bounded global messages_fts repair completed; exact counts skipped"


class FTSReadiness(BaseModel):
    indexed_surface: str = "messages_fts"
    messages_ready: bool = False
    session_work_events_ready: bool = False
    threads_ready: bool = False
    invariant_ready: bool = False
    message_indexed_count: int | None = 0
    message_indexable_count: int | None = 0
    coverage_pct: float | None = 0.0
    coverage_exact: bool = True
    surfaces: dict[str, dict[str, int | bool | str | None]] = Field(default_factory=dict)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _triggers_present(conn: sqlite3.Connection, trigger_names: tuple[str, ...]) -> bool:
    placeholders = ",".join("?" for _ in trigger_names)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        trigger_names,
    ).fetchall()
    present = {row[0] for row in rows}
    return all(name in present for name in trigger_names)


def _source_has_rows(conn: sqlite3.Connection, table_name: str) -> bool | None:
    try:
        row = conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1").fetchone()
    except sqlite3.Error as exc:
        logger.warning("fts source-rows probe failed for %s: %s", table_name, exc, exc_info=True)
        return None
    return row is not None


def _freshness_rows(conn: sqlite3.Connection) -> dict[str, dict[str, int | str | None]] | None:
    if not _table_exists(conn, "fts_freshness_state"):
        return None
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(fts_freshness_state)").fetchall()}
    numeric_columns = (
        "source_rows",
        "indexed_rows",
        "missing_rows",
        "excess_rows",
        "duplicate_rows",
    )
    selected = ["surface", "state"]
    selected.extend(name for name in numeric_columns if name in columns)
    if "detail" in columns:
        selected.append("detail")
    rows = conn.execute(f"SELECT {', '.join(selected)} FROM fts_freshness_state").fetchall()
    records: dict[str, dict[str, int | str | None]] = {}
    for row in rows:
        record = dict(zip(selected, row, strict=True))
        records[str(record["surface"])] = {
            "state": str(record["state"]),
            "source_rows": _int_or_zero(record.get("source_rows")),
            "indexed_rows": _int_or_zero(record.get("indexed_rows")),
            "missing_rows": _int_or_zero(record.get("missing_rows")),
            "excess_rows": _int_or_zero(record.get("excess_rows")),
            "duplicate_rows": _int_or_zero(record.get("duplicate_rows")),
            "detail": None if "detail" not in record or record["detail"] is None else str(record["detail"]),
        }
    return records


def _freshness_record(
    freshness: dict[str, dict[str, int | str | None]] | None,
    surface: str,
) -> dict[str, int | str | None] | None:
    if freshness is None:
        return None
    return freshness.get(surface)


def _surface_payload(surface: FtsSurfaceInvariant) -> dict[str, int | bool | str | None]:
    return {
        "source_exists": surface.source_exists,
        "exists": surface.exists,
        "source_rows": surface.source_rows,
        "indexed_rows": surface.indexed_rows,
        "triggers_present": surface.triggers_present,
        "missing_rows": surface.missing_rows,
        "excess_rows": surface.excess_rows,
        "duplicate_rows": surface.duplicate_rows,
        "ready": surface.ready,
        "exact": True,
    }


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


def _payload_int(surface: dict[str, int | bool | str | None], key: str) -> int:
    value = surface.get(key)
    return _int_or_zero(value)


def _archive_index_path_for(dbf: Path) -> Path | None:
    if dbf.name == "index.db":
        return dbf
    index_db = dbf.with_name("index.db")
    return index_db if index_db.exists() else None


def _archive_exact_blocks_surface(conn: sqlite3.Connection) -> dict[str, int | bool | str | None]:
    source_exists = _table_exists(conn, "blocks")
    exists = _table_exists(conn, "messages_fts")
    triggers_present = exists and _triggers_present(conn, _ARCHIVE_BLOCKS_FTS_TRIGGERS)
    if not source_exists:
        ready = not exists
        return {
            "source_exists": source_exists,
            "exists": exists,
            "source_rows": 0,
            "indexed_rows": 0,
            "triggers_present": triggers_present,
            "missing_rows": 0,
            "excess_rows": 0,
            "duplicate_rows": 0,
            "ready": ready,
            "exact": True,
        }
    # A block is FTS-indexable iff search_text != '' — that is exactly the
    # predicate messages_fts is populated from (storage/fts/sql.py). Counting
    # `text IS NOT NULL` here instead undercounts the source (tool_use /
    # tool_result blocks carry derived search_text but a NULL display text), so
    # indexed/source could exceed 100%.
    source_rows = int(conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0] or 0)
    docsize_exists = _table_exists(conn, "messages_fts_docsize")
    if not exists or not docsize_exists:
        return {
            "source_exists": source_exists,
            "exists": exists,
            "source_rows": source_rows,
            "indexed_rows": 0,
            "triggers_present": False,
            "missing_rows": source_rows,
            "excess_rows": 0,
            "duplicate_rows": 0,
            "ready": False,
            "exact": True,
        }
    indexed_rows = int(conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] or 0)
    missing_rows = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM blocks b
            LEFT JOIN messages_fts_docsize d ON d.id = b.rowid
            WHERE b.search_text != ''
              AND d.id IS NULL
            """
        ).fetchone()[0]
        or 0
    )
    excess_rows = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts_docsize d
            LEFT JOIN blocks b ON b.rowid = d.id
            WHERE b.rowid IS NULL
               OR b.search_text = ''
            """
        ).fetchone()[0]
        or 0
    )
    duplicate_rows = 0
    ready = (
        triggers_present
        and missing_rows == 0
        and excess_rows == 0
        and duplicate_rows == 0
        and source_rows == indexed_rows
    )
    return {
        "source_exists": source_exists,
        "exists": exists,
        "source_rows": source_rows,
        "indexed_rows": indexed_rows,
        "triggers_present": triggers_present,
        "missing_rows": missing_rows,
        "excess_rows": excess_rows,
        "duplicate_rows": duplicate_rows,
        "ready": ready,
        "exact": True,
    }


def _archive_blocks_surface(conn: sqlite3.Connection) -> dict[str, int | bool | str | None]:
    freshness_records = _freshness_rows(conn)
    freshness = _freshness_record(freshness_records, "messages_fts")
    source_exists = _table_exists(conn, "blocks")
    exists = _table_exists(conn, "messages_fts")
    triggers_present = exists and _triggers_present(conn, _ARCHIVE_BLOCKS_FTS_TRIGGERS)
    source_rows = 0 if freshness is None else _int_or_zero(freshness.get("source_rows"))
    indexed_rows = 0 if freshness is None else _int_or_zero(freshness.get("indexed_rows"))
    missing_rows = 0 if freshness is None else _int_or_zero(freshness.get("missing_rows"))
    excess_rows = 0 if freshness is None else _int_or_zero(freshness.get("excess_rows"))
    duplicate_rows = 0 if freshness is None else _int_or_zero(freshness.get("duplicate_rows"))
    recorded_state = None if freshness is None else str(freshness.get("state"))
    source_has_rows = (
        _source_has_rows(conn, "blocks")
        if source_exists and recorded_state == "ready" and source_rows == 0 and indexed_rows == 0
        else False
    )
    freshness_ready = (
        True
        if freshness_records is None
        else freshness_ready_record_trusted(
            state=recorded_state,
            source_rows=source_rows,
            indexed_rows=indexed_rows,
            missing_rows=missing_rows,
            excess_rows=excess_rows,
            duplicate_rows=duplicate_rows,
            source_has_rows=source_has_rows,
        )
    )
    freshness_state = recorded_state
    if recorded_state == "ready" and not freshness_ready:
        freshness_state = UNKNOWN if source_rows == 0 and indexed_rows == 0 and source_has_rows is not False else STALE
    ready = (exists and triggers_present and freshness_ready) if source_exists else not exists
    counts_available = None if freshness is None else freshness.get("detail") != BOUNDED_MESSAGE_FTS_REPAIR_DETAIL
    return {
        "source_exists": source_exists,
        "exists": exists,
        "source_rows": source_rows,
        "indexed_rows": indexed_rows,
        "triggers_present": triggers_present,
        "missing_rows": missing_rows,
        "excess_rows": excess_rows,
        "duplicate_rows": duplicate_rows,
        "ready": ready,
        "exact": False,
        "freshness_known": freshness_records is not None,
        "freshness_state": freshness_state,
        "freshness_recorded_state": recorded_state,
        "freshness_trusted": freshness_ready,
        "freshness_detail": None if freshness is None else freshness.get("detail"),
        "counts_available": counts_available,
    }


def _archive_readiness_payload(conn: sqlite3.Connection, *, exact: bool) -> dict[str, object] | None:
    if not _table_exists(conn, "blocks") and not _table_exists(conn, "messages_fts"):
        return None
    blocks = _archive_exact_blocks_surface(conn) if exact else _archive_blocks_surface(conn)
    effective_exact = exact
    if not exact and blocks.get("freshness_state") == UNKNOWN and _source_has_rows(conn, "messages_fts_docsize"):
        # The durable freshness cache is poisoned: the recorded state is "ready"
        # but the counts are the 0/0 shape while the source actually has rows
        # (a fresh-archive bootstrap recorded ready|0|0 over an empty archive and
        # trigger-maintained ingest then populated the index without a rebuild to
        # refresh the counts). Recompute exact counts ONLY when the FTS index is
        # verifiably populated (``messages_fts_docsize`` has rows) so coverage
        # reflects the populated index instead of reporting 0%. Merge the exact
        # counts over the cheap surface so the freshness_* diagnostics survive.
        #
        # When the index is empty or unverifiable, the source genuinely has
        # unindexed rows: leave the surface at freshness_state=UNKNOWN and do NOT
        # scan source/FTS shadow tables (the #1003 request-safe contract, asserted
        # by test_fts_readiness_rejects_zero_count_ready_freshness_when_source_has_rows).
        blocks = {**blocks, **_archive_exact_blocks_surface(conn)}
        effective_exact = True
    block_source_rows = _payload_int(blocks, "source_rows")
    block_indexed_rows = _payload_int(blocks, "indexed_rows")
    invariant_ready = bool(blocks["ready"])
    counts_available = bool(blocks.get("counts_available", True))
    return {
        "indexed_surface": "messages_fts",
        "messages_ready": invariant_ready,
        "session_work_events_ready": True,
        "threads_ready": True,
        "invariant_ready": invariant_ready,
        "message_indexed_count": block_indexed_rows if counts_available else None,
        "message_indexable_count": block_source_rows if counts_available else None,
        "coverage_pct": (
            round((block_indexed_rows / block_source_rows) * 100, 1)
            if block_source_rows > 0
            else (100.0 if invariant_ready else 0.0)
        )
        if counts_available
        else None,
        "coverage_exact": effective_exact,
        "surfaces": {"messages_fts": blocks},
    }


def _archive_readiness_info(index_db: Path, *, exact: bool) -> dict[str, object] | None:
    if not index_db.exists():
        return None
    try:
        conn = open_readonly_connection(index_db)
        try:
            return _archive_readiness_payload(conn, exact=exact)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("fts archive readiness query failed for %s: %s", index_db, exc, exc_info=True)
        return {
            "indexed_surface": "messages_fts",
            "messages_ready": False,
            "session_work_events_ready": True,
            "threads_ready": True,
            "invariant_ready": False,
            "coverage_pct": 0.0,
            "surfaces": {},
        }


def _exact_readiness_payload(snapshot: FtsInvariantSnapshot) -> dict[str, object]:
    surfaces = {surface.name: _surface_payload(surface) for surface in snapshot.surfaces}
    messages = snapshot.messages
    coverage_pct = round((messages.indexed_rows / messages.source_rows) * 100, 1) if messages.source_rows > 0 else 100.0
    return {
        "messages_ready": messages.ready,
        "session_work_events_ready": snapshot.session_work_events.ready,
        "threads_ready": snapshot.threads.ready,
        "invariant_ready": snapshot.ready,
        "message_indexed_count": messages.indexed_rows,
        "message_indexable_count": messages.source_rows,
        "coverage_pct": coverage_pct,
        "coverage_exact": True,
        "surfaces": surfaces,
    }


def fts_readiness_info(dbf: Path, *, exact: bool = False) -> dict[str, object]:
    """Return FTS readiness for health/status probes.

    The default is request-safe: it proves tables/triggers exist and, when
    the durable freshness table exists, requires each live surface to be
    marked ready. It never scans source or FTS shadow tables. Use
    ``exact=True`` for explicit diagnostics/repair jobs that can afford a
    full invariant scan.
    """
    if not dbf.exists():
        archive_index = _archive_index_path_for(dbf)
        if archive_index is not None:
            archive_info = _archive_readiness_info(archive_index, exact=exact)
            if archive_info is not None:
                return archive_info
        return {
            "messages_ready": False,
            "coverage_pct": 0.0,
        }
    try:
        conn = open_readonly_connection(dbf)
        try:
            if exact:
                conn.execute("BEGIN")
            archive_info = _archive_readiness_payload(conn, exact=exact)
            if archive_info is not None:
                return archive_info
            if exact:
                return _exact_readiness_payload(fts_invariant_snapshot_sync(conn))
            freshness_records = _freshness_rows(conn)
            surfaces: dict[str, dict[str, int | bool | str | None]] = {}
            for name, source_table, fts_table, triggers in _FTS_SURFACES:
                source_exists = _table_exists(conn, source_table)
                exists = _table_exists(conn, fts_table)
                triggers_present = exists and _triggers_present(conn, triggers)
                freshness = _freshness_record(freshness_records, name)
                source_rows = 0 if freshness is None else _int_or_zero(freshness.get("source_rows"))
                indexed_rows = 0 if freshness is None else _int_or_zero(freshness.get("indexed_rows"))
                missing_rows = 0 if freshness is None else _int_or_zero(freshness.get("missing_rows"))
                excess_rows = 0 if freshness is None else _int_or_zero(freshness.get("excess_rows"))
                duplicate_rows = 0 if freshness is None else _int_or_zero(freshness.get("duplicate_rows"))
                recorded_state = None if freshness is None else str(freshness.get("state"))
                source_has_rows = (
                    _source_has_rows(conn, source_table)
                    if source_exists and recorded_state == "ready" and source_rows == 0 and indexed_rows == 0
                    else False
                )
                freshness_ready = (
                    True
                    if freshness_records is None
                    else freshness_ready_record_trusted(
                        state=recorded_state,
                        source_rows=source_rows,
                        indexed_rows=indexed_rows,
                        missing_rows=missing_rows,
                        excess_rows=excess_rows,
                        duplicate_rows=duplicate_rows,
                        source_has_rows=source_has_rows,
                    )
                )
                freshness_state = recorded_state
                if recorded_state == "ready" and not freshness_ready:
                    freshness_state = (
                        UNKNOWN if source_rows == 0 and indexed_rows == 0 and source_has_rows is not False else STALE
                    )
                ready = (exists and triggers_present and freshness_ready) if source_exists else not exists
                surfaces[name] = {
                    "source_exists": source_exists,
                    "exists": exists,
                    "source_rows": source_rows,
                    "indexed_rows": indexed_rows,
                    "triggers_present": triggers_present,
                    "missing_rows": missing_rows,
                    "excess_rows": excess_rows,
                    "duplicate_rows": duplicate_rows,
                    "ready": ready,
                    "exact": False,
                    "freshness_known": freshness_records is not None,
                    "freshness_state": freshness_state,
                    "freshness_recorded_state": recorded_state,
                    "freshness_trusted": freshness_ready,
                    "freshness_detail": None if freshness is None else freshness.get("detail"),
                }
        finally:
            if exact:
                conn.rollback()
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("fts readiness query failed for %s: %s", dbf, exc, exc_info=True)
        return {
            "messages_ready": False,
            "session_work_events_ready": False,
            "threads_ready": False,
            "invariant_ready": False,
            "coverage_pct": 0.0,
            "surfaces": {},
        }

    messages = surfaces["messages_fts"]
    session_work_events = surfaces["session_work_events_fts"]
    threads = surfaces["threads_fts"]
    invariant_ready = all(bool(surface["ready"]) for surface in surfaces.values())
    message_source_rows = _payload_int(messages, "source_rows")
    message_indexed_rows = _payload_int(messages, "indexed_rows")
    return {
        "messages_ready": messages["ready"],
        "session_work_events_ready": session_work_events["ready"],
        "threads_ready": threads["ready"],
        "invariant_ready": invariant_ready,
        "message_indexed_count": message_indexed_rows,
        "message_indexable_count": message_source_rows,
        "coverage_pct": (
            round((message_indexed_rows / message_source_rows) * 100, 1)
            if message_source_rows > 0
            else (100.0 if invariant_ready else 0.0)
        ),
        "coverage_exact": False,
        "surfaces": surfaces,
    }

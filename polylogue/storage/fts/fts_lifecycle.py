"""Canonical FTS lifecycle operations shared across sync and async callers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import aiosqlite

from polylogue.core.sqlite_introspection import table_exists as _table_exists_sync
from polylogue.core.sqlite_introspection import table_exists_async as _table_exists_async
from polylogue.storage.fts.sql import (
    BLOCKS_FTS_TRIGGER_DDL,
    FTS_IDENTITY_REBUILD_SQL,
    FTS_INDEX_DOC_COUNT_SQL,
    FTS_INDEX_EXISTS_SQL,
    FTS_INDEXABLE_MESSAGE_COUNT_SQL,
    FTS_MESSAGES_IDENTITY_TABLE_SQL,
    FTS_MESSAGES_TABLE_SQL,
    FTS_REBUILD_SQL,
    FTS_TRIGGER_DDL,
    IndexedMessage,
    chunked,
    excess_message_rows_sql,
    insert_all_message_identity_rows_sql,
    insert_all_message_rows_sql,
    insert_missing_message_rows_range_sql,
    insert_missing_message_rows_sql,
    message_identity_mismatch_sql,
    repair_all_message_identity_rows_sql,
    repair_message_identity_rows_range_sql,
)
from polylogue.storage.sqlite.connection_profile import (
    BOUNDED_REPAIR_CACHE_SIZE_KIB,
    BOUNDED_REPAIR_MMAP_SIZE_BYTES,
)

_chunked = chunked
IndexedMessageLike: TypeAlias = tuple[str, str, str | None] | IndexedMessage


def _indexed_message_parts(message: IndexedMessageLike) -> tuple[str, str, str | None]:
    if isinstance(message, tuple):
        return message
    return message.message_id, message.session_id, message.text


def _row_int(row: sqlite3.Row | None, key: int | str) -> int:
    if row is None:
        return 0
    try:
        return int(row[key])
    except (TypeError, ValueError):
        return 0


def _status_int(status: dict[str, object], key: str) -> int:
    value = status.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _message_trigger_names_for_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    del conn
    return _BLOCKS_FTS_TRIGGER_NAMES


async def _message_trigger_names_for_async(conn: aiosqlite.Connection) -> tuple[str, ...]:
    del conn
    return _BLOCKS_FTS_TRIGGER_NAMES


_BLOCKS_FTS_TRIGGER_NAMES = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
)

_FTS_TRIGGER_NAMES = _BLOCKS_FTS_TRIGGER_NAMES

FTS_TRIGGER_NAMES = _FTS_TRIGGER_NAMES
"""Canonical FTS trigger set for all archive and insight search surfaces."""

DEFAULT_MISSING_MESSAGE_FTS_BATCH_ROWS = 50_000
"""Rowid window size for archive-wide missing message FTS repair."""

DEFAULT_EXCESS_MESSAGE_FTS_BATCH_ROWS = 5_000
"""Batch size for archive-wide excess message FTS row deletion."""


@dataclass(frozen=True, slots=True)
class FtsSurfaceInvariant:
    """Exact freshness status for one FTS-backed surface."""

    name: str
    source_exists: bool
    exists: bool
    source_rows: int
    indexed_rows: int
    triggers_present: bool
    missing_rows: int = 0
    excess_rows: int = 0
    duplicate_rows: int = 0
    # polylogue-1xc.12: rowid-reuse/changed-text/changed-recipe drift the
    # messages_fts_identity ledger catches that missing_rows/excess_rows
    # cannot -- both sides still balance when a stale rowid has silently
    # rebound to a different block. Zero for surfaces without an identity
    # ledger (only messages_fts has one today).
    identity_mismatch_rows: int = 0

    @property
    def ready(self) -> bool:
        if not self.source_exists:
            return not self.exists
        return (
            self.exists
            and self.triggers_present
            and self.missing_rows == 0
            and self.excess_rows == 0
            and self.duplicate_rows == 0
            and self.identity_mismatch_rows == 0
        )


@dataclass(frozen=True, slots=True)
class FtsInvariantSnapshot:
    """Exact freshness status for every active FTS-backed search surface."""

    messages: FtsSurfaceInvariant
    retired_action_surface: FtsSurfaceInvariant

    @property
    def ready(self) -> bool:
        return all(surface.ready for surface in self.surfaces)

    @property
    def surfaces(self) -> tuple[FtsSurfaceInvariant, ...]:
        return (self.messages, self.retired_action_surface)


def _triggers_present_sync(conn: sqlite3.Connection, names: tuple[str, ...]) -> bool:
    """Check whether every named trigger exists in sqlite_master."""
    placeholders = ", ".join("?" for _ in names)
    row = conn.execute(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        names,
    ).fetchone()
    return row is not None and row[0] == len(names)


async def _triggers_present_async(conn: aiosqlite.Connection, names: tuple[str, ...]) -> bool:
    """Check whether every named trigger exists in sqlite_master."""
    placeholders = ", ".join("?" for _ in names)
    cursor = await conn.execute(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        names,
    )
    row = await cursor.fetchone()
    return row is not None and row[0] == len(names)


# polylogue-a7xr.5: FTS trigger DDL is now sourced from storage/fts/sql.py as the single
# source of truth. Aliases below preserve backward compatibility with code that
# references the private _*_TRIGGER_DDL names.
_BLOCKS_FTS_TRIGGER_DDL = BLOCKS_FTS_TRIGGER_DDL
_FTS_TRIGGER_DDL = FTS_TRIGGER_DDL


def configure_bounded_fts_repair_connection(conn: sqlite3.Connection) -> None:
    """Apply the bounded profile retained for explicit bulk FTS maintenance."""
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute(f"PRAGMA cache_size = -{BOUNDED_REPAIR_CACHE_SIZE_KIB}")
    conn.execute(f"PRAGMA main.mmap_size = {BOUNDED_REPAIR_MMAP_SIZE_BYTES}")


def suspend_fts_triggers_sync(conn: sqlite3.Connection, *, mark_stale: bool = True) -> None:
    """Drop FTS triggers for bulk sync operations."""
    del mark_stale
    for name in _FTS_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def message_fts_triggers_present_sync(conn: sqlite3.Connection) -> bool:
    """Return true when the block-backed message FTS triggers are present."""
    return _triggers_present_sync(conn, _BLOCKS_FTS_TRIGGER_NAMES)


def suspend_message_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Drop only block-backed message FTS triggers inside the caller's transaction."""
    for name in _BLOCKS_FTS_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def restore_message_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Restore only block-backed message FTS triggers inside the caller's transaction."""
    if not _table_exists_sync(conn, "blocks") or not _table_exists_sync(conn, "messages_fts"):
        return
    for ddl in _BLOCKS_FTS_TRIGGER_DDL:
        conn.execute(ddl)


def _create_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Issue the ``CREATE TRIGGER IF NOT EXISTS`` DDL for existing surfaces."""
    for ddl in _fts_trigger_ddl_for_existing_surfaces_sync(conn):
        conn.executescript(ddl) if ";" in ddl else conn.execute(ddl)


def restore_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Re-create FTS triggers after an interrupted or completed bulk insert.

    This is the recovery form and it must never drop first.  The DDL is
    ``CREATE TRIGGER IF NOT EXISTS``, so a leading ``DROP`` adds nothing but a
    durable trigger-less window: DDL runs in autocommit, so a process death
    between the drop and the creates leaves ``index.db`` permanently without
    FTS triggers and every later block write silently unindexed
    (polylogue-u66s3).

    Callers that genuinely need trigger *definitions* replaced use
    ``replace_fts_triggers_sync``.
    """
    _create_fts_triggers_sync(conn)


def replace_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Drop and re-create every FTS trigger definition.

    The explicit rebuild path: use it only where stale trigger *bodies* must be
    replaced and the caller owns the resulting window.  Recovery paths use
    ``restore_fts_triggers_sync`` instead.
    """
    suspend_fts_triggers_sync(conn)
    _create_fts_triggers_sync(conn)


def ensure_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Create missing FTS triggers without dropping existing triggers.

    Steady-state archive writes must not create a dropped-trigger window.
    ``replace_fts_triggers_sync`` remains the explicit rebuild path for
    replacing trigger definitions; ``restore_fts_triggers_sync`` is the
    no-drop recovery path.

    Fast-path: when all expected triggers are already present, return
    immediately without issuing any ``executescript()`` calls.  Each
    ``executescript()`` call issues an implicit COMMIT that fragments the
    caller's WAL transaction into multiple smaller ones; avoiding it in
    steady state preserves the intended one-transaction boundary for
    ``commit_archive_write_effects`` (#1851).
    """
    if _triggers_present_sync(conn, _FTS_TRIGGER_NAMES):
        return
    for ddl in _fts_trigger_ddl_for_existing_surfaces_sync(conn):
        conn.executescript(ddl) if ";" in ddl else conn.execute(ddl)


def ensure_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Ensure the FTS5 tables and triggers exist on a sync SQLite connection."""
    conn.execute(FTS_MESSAGES_TABLE_SQL)
    conn.execute(FTS_MESSAGES_IDENTITY_TABLE_SQL)
    ensure_fts_triggers_sync(conn)


async def ensure_fts_index_async(conn: aiosqlite.Connection) -> None:
    """Ensure the FTS5 tables and triggers exist on an async SQLite connection."""
    await conn.execute(FTS_MESSAGES_TABLE_SQL)
    await conn.execute(FTS_MESSAGES_IDENTITY_TABLE_SQL)
    for ddl in await _fts_trigger_ddl_for_existing_surfaces_async(conn):
        if ";" in ddl:
            await conn.executescript(ddl)
        else:
            await conn.execute(ddl)


def _fts_trigger_ddl_for_existing_surfaces_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    ddl: list[str] = []
    if _table_exists_sync(conn, "blocks") and _table_exists_sync(conn, "messages_fts"):
        ddl.extend(_BLOCKS_FTS_TRIGGER_DDL)
    return tuple(ddl)


async def _fts_trigger_ddl_for_existing_surfaces_async(conn: aiosqlite.Connection) -> tuple[str, ...]:
    ddl: list[str] = []
    if await _table_exists_async(conn, "blocks") and await _table_exists_async(conn, "messages_fts"):
        ddl.extend(_BLOCKS_FTS_TRIGGER_DDL)
    return tuple(ddl)


def rebuild_messages_fts_content_sync(conn: sqlite3.Connection) -> None:
    """Clear and repopulate ``messages_fts`` content only (no identity ledger).

    Companion to :func:`rebuild_messages_fts_identity_sync`; callers that also
    need the identity ledger to converge call both. Split out (polylogue-t3gk)
    so a caller that only needs one surface refreshed -- e.g. an index
    fast-forward declaration naming only ``messages_fts_identity`` -- does not
    have to pay for a full content rescan it did not declare.
    """
    conn.execute(FTS_REBUILD_SQL)
    conn.execute(insert_all_message_rows_sql())


def rebuild_messages_fts_identity_sync(conn: sqlite3.Connection) -> None:
    """Clear and repopulate the ``messages_fts_identity`` ledger only.

    See :func:`rebuild_messages_fts_content_sync`.
    """
    conn.execute(FTS_IDENTITY_REBUILD_SQL)
    conn.execute(insert_all_message_identity_rows_sql())


def rebuild_fts_index_sync(
    conn: sqlite3.Connection,
    *,
    resume_from_empty_message_index: bool = False,
) -> None:
    """Rebuild the full FTS index from persisted archive rows.

    ``messages_fts`` is contentless, so the ordinary path clears it and
    repopulates it from the canonical message/content-block projection. An
    owned bulk-build generation may opt into ``resume_from_empty_message_index``:
    its FTS store is known to have been cleared before replay, so the existing
    paged missing-row writer can commit each chunk and resume an interrupted
    terminal pass without redoing already materialized FTS rows.
    """
    ensure_fts_index_sync(conn)
    if resume_from_empty_message_index:
        insert_missing_message_rows_batched_sync(conn)
    else:
        rebuild_messages_fts_content_sync(conn)
        rebuild_messages_fts_identity_sync(conn)


def reset_message_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Drop and recreate the block-backed message FTS surface.

    ``messages_fts`` is a contentless-delete FTS5 table.  Clearing it with
    delete-all is correct but can be catastrophically slow on a large archive
    because SQLite walks the virtual table and shadow tables row by row.  Global
    recovery owns the whole message surface, so it can reset the virtual table
    structurally, recreate the block triggers, and repopulate from ``blocks``.
    """
    for name in _BLOCKS_FTS_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP TABLE IF EXISTS messages_fts")
    conn.execute(FTS_MESSAGES_TABLE_SQL)
    conn.execute("DROP TABLE IF EXISTS messages_fts_identity")
    conn.execute(FTS_MESSAGES_IDENTITY_TABLE_SQL)
    if _table_exists_sync(conn, "blocks"):
        for ddl in _BLOCKS_FTS_TRIGGER_DDL:
            conn.execute(ddl)
        insert_missing_message_rows_batched_sync(conn)


def _blocks_content_hash_available_sync(conn: sqlite3.Connection) -> bool:
    """Whether ``blocks.content_hash`` exists (identity ledger source-hash input).

    Some low-level tests exercise ``messages_fts`` repair against a minimal
    hand-rolled ``blocks`` table (a handful of TEXT columns, no
    ``content_hash``) rather than the full archive schema -- the identity
    ledger is additive there: skip populating it rather than erroring, the
    same accommodation already made for other optional derived surfaces.
    """
    return any(str(row[1]) == "content_hash" for row in conn.execute("PRAGMA table_info(blocks)").fetchall())


def insert_missing_message_rows_batched_sync(
    conn: sqlite3.Connection,
    *,
    batch_rows: int = DEFAULT_MISSING_MESSAGE_FTS_BATCH_ROWS,
    measure_counts: bool = True,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> int:
    """Insert missing block-backed FTS rows in committed rowid windows."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")

    ensure_fts_index_sync(conn)
    identity_supported = _blocks_content_hash_available_sync(conn)
    before = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0) if measure_counts else 0
    max_rowid = _row_int(
        conn.execute(
            """
            SELECT COALESCE(MAX(rowid), 0)
            FROM blocks
            WHERE search_text != ''
            """
        ).fetchone(),
        0,
    )
    sql = insert_missing_message_rows_range_sql()
    identity_sql = repair_message_identity_rows_range_sql()
    lower = 0
    while lower < max_rowid:
        upper = min(lower + batch_rows, max_rowid)
        changes_before = conn.total_changes
        conn.execute(sql, (lower, upper))
        inserted = conn.total_changes - changes_before
        identity_changed = 0
        if identity_supported:
            identity_changes_before = conn.total_changes
            conn.execute(identity_sql, (lower, upper))
            identity_changed = conn.total_changes - identity_changes_before
        if inserted or identity_changed:
            conn.commit()
        if progress_callback is not None:
            progress_callback(lower, upper, max(0, inserted))
        lower = upper

    if not measure_counts:
        return 0
    after = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
    return max(0, after - before)


def delete_excess_message_rows_batched_sync(
    conn: sqlite3.Connection,
    *,
    batch_rows: int = DEFAULT_EXCESS_MESSAGE_FTS_BATCH_ROWS,
    progress_callback: Callable[[int], None] | None = None,
) -> int:
    """Delete FTS rows whose canonical ``blocks`` row is no longer indexable."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")

    ensure_fts_index_sync(conn)
    deleted_total = 0
    while True:
        rows = conn.execute(excess_message_rows_sql(batch_rows)).fetchall()
        rowids = [int(row[0]) for row in rows]
        if not rowids:
            break
        placeholders = ", ".join("?" for _ in rowids)
        changes_before = conn.total_changes
        conn.execute(f"DELETE FROM messages_fts WHERE rowid IN ({placeholders})", tuple(rowids))
        conn.execute(f"DELETE FROM messages_fts_identity WHERE rowid IN ({placeholders})", tuple(rowids))
        deleted = max(0, conn.total_changes - changes_before)
        deleted_total += deleted
        if deleted:
            conn.commit()
        if progress_callback is not None:
            progress_callback(deleted)
        if len(rowids) < batch_rows:
            break
    return deleted_total


def reconcile_message_fts_rows_once_sync(conn: sqlite3.Connection) -> tuple[int, int]:
    """Reconcile the global message FTS surface with bounded scan count.

    The old numeric-rowid window loop is safe for compact tables, but a live
    archive can retain a sparse rowid space after many full replacements.
    Each window then repeats an expensive join against the FTS docsize shadow
    table.  A global debt repair is already an explicit maintenance action, so
    use one set-based missing-row pass and one set-based identity pass instead.
    Existing excess rows are still removed through the contentless-FTS-safe
    batched delete primitive.
    """
    ensure_fts_index_sync(conn)
    deleted = delete_excess_message_rows_batched_sync(conn)
    before = conn.total_changes
    conn.execute(insert_missing_message_rows_sql())
    inserted = max(0, conn.total_changes - before)
    if _blocks_content_hash_available_sync(conn):
        conn.execute(repair_all_message_identity_rows_sql())
    return inserted, deleted


async def rebuild_fts_index_async(
    conn: aiosqlite.Connection,
    *,
    session_ids: Sequence[str] | None = None,
    progress_callback: Callable[[int, str | None], None] | None = None,
    progress_desc: Callable[[int, int], str] | None = None,
) -> None:
    """Rebuild the full FTS index from persisted archive rows.

    A whole-archive rebuild owns enough state to publish the same exact
    invariant snapshot as the synchronous lifecycle.  Scoped repairs remain
    bounded observations and therefore stay stale until an exact pass.
    """
    await ensure_fts_index_async(conn)
    if session_ids is not None:
        await repair_fts_index_async(
            conn,
            session_ids,
            progress_callback=progress_callback,
            progress_desc=progress_desc,
        )
        return
    # Keep the exact scan on aiosqlite's owning worker thread.  The sync
    # lifecycle is the canonical full-rebuild path and couples the rebuild to
    # its transaction-bound freshness publication.
    await conn._execute(rebuild_fts_index_sync, conn._conn)  # type: ignore[no-untyped-call]


def repair_message_fts_index_sync(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    record_exact_snapshot: bool = True,
) -> None:
    """Repair message FTS rows for the supplied sessions.

    The supplied sessions are a bounded scope, so this operation never
    publishes a global READY verdict. ``record_exact_snapshot`` is retained
    for caller compatibility but no longer authorizes an archive-wide scan.
    """
    if not session_ids:
        return
    from polylogue.storage.fts.derivation import replace_fts_partition_sync

    for session_id in dict.fromkeys(session_ids):
        replace_fts_partition_sync(conn, session_id)
    del record_exact_snapshot


def repair_fts_index_sync(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Repair FTS rows for the supplied sessions from persisted rows."""
    ensure_fts_index_sync(conn)
    repair_message_fts_index_sync(conn, session_ids)


async def repair_fts_index_async(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    *,
    progress_callback: Callable[[int, str | None], None] | None = None,
    progress_desc: Callable[[int, int], str] | None = None,
) -> None:
    """Repair FTS rows for the supplied sessions from persisted rows."""
    await ensure_fts_index_async(conn)
    if not session_ids:
        return
    await conn._execute(repair_message_fts_index_sync, conn._conn, tuple(session_ids))  # type: ignore[no-untyped-call]
    if progress_callback is not None:
        total = len(session_ids)
        progress_callback(total, progress_desc(total, total) if progress_desc is not None else None)


def replace_fts_rows_for_messages_sync(
    conn: sqlite3.Connection,
    messages: Sequence[IndexedMessageLike],
) -> None:
    """Replace FTS rows for the supplied persisted message sessions."""
    ensure_fts_index_sync(conn)
    if not messages:
        return

    session_ids = sorted({_indexed_message_parts(message)[1] for message in messages})
    repair_message_fts_index_sync(conn, session_ids, record_exact_snapshot=False)


def fts_index_status_sync(conn: sqlite3.Connection) -> dict[str, object]:
    """Return existence and document counts for the sync FTS index."""
    row = conn.execute(FTS_INDEX_EXISTS_SQL).fetchone()
    exists = bool(row)
    count = 0
    if exists:
        count = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
    return {"exists": exists, "count": int(count), "action_exists": False, "action_count": 0}


async def fts_index_status_async(conn: aiosqlite.Connection) -> dict[str, object]:
    """Return existence and document counts for the async FTS index."""
    row = await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone()
    exists = bool(row)
    count = 0
    if exists:
        count_row = await (await conn.execute(FTS_INDEX_DOC_COUNT_SQL)).fetchone()
        count = count_row[0] if count_row else 0
    return {"exists": exists, "count": int(count), "action_exists": False, "action_count": 0}


def message_fts_readiness_sync(
    conn: sqlite3.Connection,
    *,
    verify_total_rows: bool = True,
) -> dict[str, int | bool]:
    """Inspect the canonical message/FTS relation used by search.

    ``verify_total_rows`` remains call-compatible but cannot select a weaker
    readiness proxy.  A count-only comparison misses wrong identities and
    orphan residue; a recorded freshness row is merely telemetry.  The domain
    adapter's global inspection is the one authoritative classifier for both
    search admission and daemon convergence.
    """
    del verify_total_rows
    from polylogue.storage.fts.derivation import GLOBAL_PARTITION, FtsDerivationAdapter

    inspection = FtsDerivationAdapter().inspect_partition(conn, GLOBAL_PARTITION)
    return {
        "exists": bool(conn.execute(FTS_INDEX_EXISTS_SQL).fetchone()),
        "indexed_rows": inspection.present_rows,
        "total_rows": inspection.required_rows,
        "ready": inspection.valid,
        "triggers_present": inspection.triggers_compatible,
    }


def message_fts_search_readiness_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    """Return retrieval readiness measured for the relation used by search."""
    return message_fts_readiness_sync(conn)


async def message_fts_readiness_async(
    conn: aiosqlite.Connection,
    *,
    verify_total_rows: bool = True,
) -> dict[str, int | bool]:
    """Async form of the same authoritative message FTS inspection."""
    del verify_total_rows
    result = await conn._execute(message_fts_readiness_sync, conn._conn)  # type: ignore[no-untyped-call]
    return cast(dict[str, int | bool], result)


async def message_fts_search_readiness_async(conn: aiosqlite.Connection) -> dict[str, int | bool]:
    """Async retrieval readiness measured for the relation used by search."""
    return await message_fts_readiness_async(conn)


# A caller-visible hint must never presume the reader knows whether a daemon
# is currently running (polylogue-roax): "Run polylogued run" read as a
# broken contract when the operator's daemon was already up and convergence
# just hadn't caught the drift yet. State the actual remedy (convergence
# self-heals while a daemon is running) and the fallback (start one) instead
# of a command that may already be satisfied.
MESSAGE_SEARCH_REPAIR_HINT = (
    "This repairs automatically while `polylogued run` is active (daemon convergence "
    "repairs FTS drift within a few convergence cycles); if no daemon is running, start one."
)


def check_fts_readiness(readiness: Mapping[str, object], repair_hint: str = MESSAGE_SEARCH_REPAIR_HINT) -> None:
    """Raise DatabaseError unless the FTS index is exactly ready."""
    from polylogue.core.errors import DatabaseError

    if not bool(readiness["exists"]):
        raise DatabaseError(f"Search index not built. {repair_hint}")
    if bool(readiness["ready"]):
        return
    raise DatabaseError(f"Search index is incomplete. {repair_hint}")


def _trigger_invariant_sync(
    conn: sqlite3.Connection,
    *,
    name: str,
    source_table_name: str,
    table_name: str,
    source_sql: str,
    indexed_sql: str,
    trigger_names: tuple[str, ...],
    missing_sql: str | None = None,
    excess_sql: str | None = None,
    duplicate_sql: str | None = None,
    identity_sql: str | None = None,
) -> FtsSurfaceInvariant:
    source_exists = _table_exists_sync(conn, source_table_name)
    exists = _table_exists_sync(conn, table_name)
    source_rows = _row_int(conn.execute(source_sql).fetchone(), 0) if source_exists else 0
    indexed_rows = _row_int(conn.execute(indexed_sql).fetchone(), 0) if exists else 0
    missing_rows = _row_int(conn.execute(missing_sql).fetchone(), 0) if source_exists and exists and missing_sql else 0
    excess_rows = _row_int(conn.execute(excess_sql).fetchone(), 0) if source_exists and exists and excess_sql else 0
    duplicate_rows = _row_int(conn.execute(duplicate_sql).fetchone(), 0) if exists and duplicate_sql else 0
    identity_mismatch_rows = (
        _row_int(conn.execute(identity_sql).fetchone(), 0)
        if source_exists and exists and identity_sql and _table_exists_sync(conn, "messages_fts_identity")
        else 0
    )
    return FtsSurfaceInvariant(
        name=name,
        source_exists=source_exists,
        exists=exists,
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        triggers_present=exists and _triggers_present_sync(conn, trigger_names),
        missing_rows=missing_rows,
        excess_rows=excess_rows,
        duplicate_rows=duplicate_rows,
        identity_mismatch_rows=identity_mismatch_rows,
    )


def fts_invariant_snapshot_sync(conn: sqlite3.Connection) -> FtsInvariantSnapshot:
    """Return exact freshness status for every active FTS search surface.

    The snapshot spans multiple aggregate queries. Start an explicit read
    transaction when the caller has not already opened one so live ingest
    commits cannot make source counts and FTS shadow counts describe
    different moments in time.
    """
    if conn.in_transaction:
        return _fts_invariant_snapshot_sync(conn)
    conn.execute("BEGIN")
    try:
        snapshot = _fts_invariant_snapshot_sync(conn)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return snapshot


def _fts_invariant_snapshot_sync(conn: sqlite3.Connection) -> FtsInvariantSnapshot:
    """Return exact freshness status for every active FTS search surface."""
    if _table_exists_sync(conn, "blocks") or _table_exists_sync(conn, "messages_fts"):
        message_surface = _trigger_invariant_sync(
            conn,
            name="messages_fts",
            source_table_name="blocks",
            table_name="messages_fts",
            source_sql="SELECT COUNT(*) FROM blocks WHERE search_text != ''",
            indexed_sql="SELECT COUNT(*) FROM messages_fts_docsize",
            trigger_names=_BLOCKS_FTS_TRIGGER_NAMES,
            missing_sql="""
                SELECT COUNT(*)
                FROM blocks AS b
                LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                WHERE b.search_text != '' AND d.id IS NULL
            """,
            excess_sql="""
                SELECT COUNT(*)
                FROM messages_fts_docsize AS d
                LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
                WHERE b.rowid IS NULL
            """,
            identity_sql=message_identity_mismatch_sql(),
        )
    else:
        message_surface = _messages_fts_invariant_sync(conn)
    return FtsInvariantSnapshot(
        messages=message_surface,
        retired_action_surface=_absent_optional_surface("retired_action_surface"),
    )


def _absent_optional_surface(name: str) -> FtsSurfaceInvariant:
    return FtsSurfaceInvariant(
        name=name,
        source_exists=False,
        exists=False,
        source_rows=0,
        indexed_rows=0,
        triggers_present=False,
    )


def _messages_fts_invariant_sync(conn: sqlite3.Connection) -> FtsSurfaceInvariant:
    """Return the block-backed message FTS invariant."""
    return _trigger_invariant_sync(
        conn,
        name="messages_fts",
        source_table_name="blocks",
        table_name="messages_fts",
        source_sql=FTS_INDEXABLE_MESSAGE_COUNT_SQL,
        indexed_sql=FTS_INDEX_DOC_COUNT_SQL,
        trigger_names=_BLOCKS_FTS_TRIGGER_NAMES,
        missing_sql="""
            SELECT COUNT(*)
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE d.id IS NULL AND b.search_text != ''
        """,
        excess_sql="""
            SELECT COUNT(*)
            FROM messages_fts_docsize AS d
            LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
            WHERE b.rowid IS NULL
        """,
        identity_sql=message_identity_mismatch_sql(),
    )


__all__ = [
    "FtsInvariantSnapshot",
    "FtsSurfaceInvariant",
    "FTS_TRIGGER_NAMES",
    "_BLOCKS_FTS_TRIGGER_DDL",
    "_chunked",
    "check_fts_readiness",
    "configure_bounded_fts_repair_connection",
    "ensure_fts_index_async",
    "ensure_fts_index_sync",
    "ensure_fts_triggers_sync",
    "fts_index_status_async",
    "fts_index_status_sync",
    "fts_invariant_snapshot_sync",
    "message_fts_readiness_async",
    "message_fts_readiness_sync",
    "message_fts_search_readiness_async",
    "message_fts_search_readiness_sync",
    "message_fts_triggers_present_sync",
    "delete_excess_message_rows_batched_sync",
    "insert_missing_message_rows_batched_sync",
    "rebuild_fts_index_async",
    "rebuild_fts_index_sync",
    "rebuild_messages_fts_content_sync",
    "rebuild_messages_fts_identity_sync",
    "repair_fts_index_async",
    "repair_fts_index_sync",
    "repair_message_fts_index_sync",
    "reconcile_message_fts_rows_once_sync",
    "reset_message_fts_index_sync",
    "replace_fts_rows_for_messages_sync",
    "restore_message_fts_triggers_sync",
    "replace_fts_triggers_sync",
    "restore_fts_triggers_sync",
    "suspend_message_fts_triggers_sync",
    "suspend_fts_triggers_sync",
]

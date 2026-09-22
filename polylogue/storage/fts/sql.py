"""SQL helpers and shared support for FTS lifecycle operations."""

from __future__ import annotations

from typing import Protocol

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked
from polylogue.storage.fts.pl_fold import pl_fold_sql_expr

# polylogue-9jsi: "unicode61 remove_diacritics 2" folds combining-mark
# diacritics (o accent, a ogonek, z dot, ...) symmetrically for indexed text
# and MATCH query text -- see polylogue/storage/fts/pl_fold.py for the full
# rationale. This tokenizer string is one of two canonical DDL sites (the
# other is the CREATE VIRTUAL TABLE messages_fts definition embedded in
# polylogue/storage/sqlite/archive_tiers/index.py); a drift-lock test keeps
# them identical.
FTS_UNICODE_TOKENIZER = "unicode61 remove_diacritics 2"

# polylogue-wohv: the table declares ONLY the indexed `text` column. It
# previously also declared block_id/message_id/session_id/block_type
# UNINDEXED, which a contentless table (content='') discards at insert --
# SQLite stores nothing for them and `SELECT block_id FROM messages_fts`
# returns NULL even though the INSERT supplied a value. Every reader already
# resolves identity by joining `blocks` on rowid (search/query_builders.py,
# search/runtime.py, archive_tiers/archive.py) or by the
# `messages_fts_identity` ledger below, so the four columns bought nothing
# and the DDL implied a retrievability the storage mode forbids. Do NOT add
# `contentless_unindexed=1` to persist them instead: the blocks join is
# needed for display fields regardless, so persisting would only grow the
# index.
FTS_MESSAGES_TABLE_SQL = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        text,
        content='',
        contentless_delete=1,
        tokenize='{FTS_UNICODE_TOKENIZER}'
    );
"""

# polylogue-1xc.12: identity recipe version consumed by messages_fts_identity
# rows (see FTS_MESSAGES_IDENTITY_TABLE_SQL below). Bump this string -- not
# INDEX_SCHEMA_VERSION -- whenever tokenizer/fold semantics change in a way
# that invalidates previously-ledgered rows without changing table shape;
# exact reconciliation compares a ledgered row's stored recipe_id against the
# CURRENT value of this constant, so an archive that never rebuilt after a
# recipe bump shows up as drift instead of silently serving results folded
# under the stale recipe.
FTS_MESSAGES_IDENTITY_RECIPE_ID = "messages_fts.v1:unicode61-remove_diacritics2+pl_fold"

# polylogue-1xc.12: rowid-keyed shadow ledger binding each `messages_fts`
# rowid to the block_id it was populated from. `messages_fts` is a
# CONTENTLESS FTS5 table (content=''), so it cannot carry identity itself:
# an UNINDEXED column there would be write-only and never retrievable by a
# later SELECT, which is why polylogue-wohv removed the four that used to be
# declared. SQLite reuses freed rowids (deleting
# the highest-rowid block then inserting a new one commonly gets the SAME
# rowid back -- exactly what a full-session-replace does), so a bare rowid
# cannot prove which block a `messages_fts` row currently represents. Count-
# only reconciliation (source_rows == indexed_rows) is blind to this: both
# sides still balance even when a stale rowid has silently rebound to a
# different block. This ledger makes block identity legible again so exact
# reconciliation can join on rowid AND block_id, not rowid alone.
# `source_hash` reuses the existing `blocks.content_hash` evidence hash (see
# storage/sqlite/archive_tiers/index.py) as the source-identity component,
# and `recipe_id` is FTS_MESSAGES_IDENTITY_RECIPE_ID as the recipe-identity
# component -- the same subject/source/recipe separation
# storage/derivation_identity.py formalizes for polylogue-wmsc's
# DerivationKey, applied here as a lightweight per-row ledger (not a full
# DerivationKey digest -- too expensive to compute per trigger-fired row) and
# never as a shared cross-domain table: FTS keeps its own ledger and repair
# lifecycle.
FTS_MESSAGES_IDENTITY_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS messages_fts_identity (
        rowid       INTEGER PRIMARY KEY,
        block_id    TEXT NOT NULL UNIQUE,
        source_hash BLOB,
        recipe_id   TEXT NOT NULL
    ) STRICT;
"""

# polylogue-crwl6: the block columns the message-FTS surface actually reduces.
# ``search_text`` and ``block_id`` are generated columns and can never appear in
# a trigger's ``UPDATE OF`` list, so the projection names their *inputs*:
# ``text``/``tool_name``/``tool_input`` compose ``search_text`` (see
# ``BLOCKS_SPEC``), ``message_id``/``position`` compose ``block_id``, and
# ``content_hash`` is the identity ledger's ``source_hash``. ``session_id``
# names the partition a row belongs to. A write to any other block column --
# ``tool_outcome``, ``semantic_type`` -- cannot move FTS membership or identity:
# the ``messages_fts_au`` arm still rewrites the row, but with identical text and
# identical identity, so the readiness binding below stays true.
FTS_BLOCK_INPUT_PROJECTION_COLUMNS: tuple[str, ...] = (
    "session_id",
    "message_id",
    "position",
    "text",
    "tool_name",
    "tool_input",
    "content_hash",
)

# polylogue-crwl6: the message-FTS domain's readiness binding, colocated with
# ``messages_fts``/``messages_fts_identity`` rather than in any cross-domain
# freshness ledger.
#
# A present row is not a flag saying "FTS is fresh". It states the *result of
# one authoritative global inspection* together with the row counts the two
# output relations held when that inspection ran. Two independent facts then
# make it evidence rather than a claim:
#
#   * the input side -- ``blocks`` -- is proven unmoved by the retirement
#     triggers declared in ``archive_tiers/index.py``: any insert, delete, or
#     update of a projected column deletes this row inside the writer's own
#     transaction. Those triggers are deliberately NOT part of
#     ``BLOCKS_FTS_TRIGGER_DDL``: the bulk-write paths suspend *those* by name
#     (``suspend_message_fts_triggers_sync``), and a bulk write is exactly when
#     the binding must not survive.
#   * the output side -- ``messages_fts`` -- is a virtual table, and SQLite
#     cannot carry a trigger on one. So the binding records the two shadow
#     relation row counts instead, and inspection re-counts them. That is two
#     narrow scans rather than the five archive-wide joins against ``blocks``
#     that ``inspect_partition`` performs for the global key.
#
# ``recipe_id`` makes a fold/tokenizer recipe bump invalidate every existing
# binding without touching a row, exactly as it already does for
# ``messages_fts_identity``.
FTS_READINESS_BINDING_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS messages_fts_readiness_binding (
        surface       TEXT PRIMARY KEY,
        recipe_id     TEXT NOT NULL,
        source_rows   INTEGER NOT NULL,
        indexed_rows  INTEGER NOT NULL,
        identity_rows INTEGER NOT NULL
    ) STRICT;
"""

FTS_READINESS_BINDING_SURFACE = "messages_fts"

FTS_INDEX_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
FTS_INDEX_DOC_COUNT_SQL = "SELECT COUNT(*) FROM messages_fts_docsize"
FTS_INDEXABLE_MESSAGE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM blocks
    WHERE search_text != ''
"""
FTS_REBUILD_SQL = """
    DELETE FROM messages_fts
"""

# polylogue-crd8: dedicated guard name gating the messages_fts per-row trigger
# BODIES (not their existence -- see FTS_BULK_SESSION_WRITE_GUARD docstring
# below for why this must never be the shared 'session-write' guard name).
FTS_BULK_SESSION_WRITE_GUARD = "fts-bulk-session-write"

_FTS_BULK_GUARD_NOT_SET = (
    f"NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = '{FTS_BULK_SESSION_WRITE_GUARD}')"
)

# FTS trigger DDL for message/block FTS maintenance.
#
# polylogue-1xc.12: each arm also maintains messages_fts_identity in the SAME
# trigger body as its messages_fts write, so the two can never observe
# different block/rowid bindings -- one atomic statement sequence per event,
# not a second pass. ad/au explicitly DELETE the identity row by rowid before
# any re-insert so a reused rowid never inherits a stale block_id.
#
# polylogue-miwv: the identity INSERTs use ``INSERT OR REPLACE`` (not a bare
# INSERT), not merely the rowid-scoped delete above. ``messages_fts_identity``
# has TWO independent uniqueness constraints -- the `rowid` INTEGER PRIMARY
# KEY and the `block_id TEXT ... UNIQUE` column -- and the ad/au DELETE only
# ever targets the rowid side. Any write path that leaves a stale
# ``(old_rowid, block_id=X)`` ledger row behind (a delete that ran with FTS
# triggers/companions suppressed and no matching bulk cleanup for that exact
# rowid -- e.g. an interrupted bulk-guarded delete, or an orphan the next
# repair pass hasn't reached yet) means a LATER insert for a *different* new
# rowid computing that same block_id X hits `UNIQUE constraint failed:
# messages_fts_identity.block_id`, aborting the write outright -- reproduced
# directly in ``tests/unit/storage/test_fts_identity_ledger.py::
# TestIdentityLedgerBlockIdCollisionRepro``. ``INSERT OR REPLACE`` evicts
# whichever pre-existing row (by either constraint) is in the way before
# inserting, so the incoming write always wins and self-heals the ledger
# instead of aborting.
BLOCKS_FTS_TRIGGER_DDL = [
    f"""CREATE TRIGGER IF NOT EXISTS messages_fts_ai
       AFTER INSERT ON blocks WHEN new.search_text != '' AND {_FTS_BULK_GUARD_NOT_SET} BEGIN
           INSERT INTO messages_fts(rowid, text)
           VALUES (new.rowid, {pl_fold_sql_expr("new.search_text")});
           INSERT OR REPLACE INTO messages_fts_identity(rowid, block_id, source_hash, recipe_id)
           VALUES (new.rowid, new.block_id, new.content_hash, '{FTS_MESSAGES_IDENTITY_RECIPE_ID}');
       END""",
    f"""CREATE TRIGGER IF NOT EXISTS messages_fts_ad
       AFTER DELETE ON blocks WHEN old.search_text != '' AND {_FTS_BULK_GUARD_NOT_SET} BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
           DELETE FROM messages_fts_identity WHERE rowid = old.rowid;
       END""",
    f"""CREATE TRIGGER IF NOT EXISTS messages_fts_au
       AFTER UPDATE ON blocks WHEN {_FTS_BULK_GUARD_NOT_SET} BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
           DELETE FROM messages_fts_identity WHERE rowid = old.rowid;
           INSERT INTO messages_fts(rowid, text)
           SELECT new.rowid, {pl_fold_sql_expr("new.search_text")}
           WHERE new.search_text != '';
           INSERT OR REPLACE INTO messages_fts_identity(rowid, block_id, source_hash, recipe_id)
           SELECT new.rowid, new.block_id, new.content_hash, '{FTS_MESSAGES_IDENTITY_RECIPE_ID}'
           WHERE new.search_text != '';
       END""",
]

# polylogue-eizc: THREAD_FTS_TRIGGER_DDL / threads_fts dropped in
# INDEX_SCHEMA_VERSION 63 -- zero application-layer consumers (see
# lifecycle.py's v63 declaration and archive_tiers/index.py's threads_fts
# removal comment).

# polylogue-cuxz.7: SESSION_WORK_EVENT_FTS_TRIGGER_DDL / session_work_events_fts
# dropped in INDEX_SCHEMA_VERSION 105 together with the session_work_events
# table the triggers maintained.

# Combined trigger DDL for all FTS surfaces.
FTS_TRIGGER_DDL = list(BLOCKS_FTS_TRIGGER_DDL)


class IndexedMessage(Protocol):
    message_id: str
    session_id: str
    text: str | None


def delete_session_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    # DELETE on external-content FTS5 raises ``database disk image is malformed``
    # when a rowid is not present in the FTS index.  Filter the deletion to
    # rowids that actually have a docsize shadow entry (i.e., are indexed).
    return f"""
        DELETE FROM messages_fts
        WHERE rowid IN (
            SELECT blocks.rowid
            FROM blocks
            WHERE blocks.session_id IN ({placeholders})
        )
        AND rowid IN (SELECT id FROM messages_fts_docsize)
    """


def insert_session_rows_sql(chunk_size: int) -> str:
    values = ", ".join("(?)" for _ in range(chunk_size))
    return f"""
        WITH raw_target_sessions(session_id) AS (
            VALUES {values}
        ),
        target_sessions AS (
            SELECT DISTINCT session_id
            FROM raw_target_sessions
        )
        INSERT INTO messages_fts (rowid, text)
        SELECT b.rowid, {pl_fold_sql_expr("b.search_text")}
        FROM blocks AS b
        JOIN target_sessions AS target
          ON target.session_id = b.session_id
        WHERE b.search_text != ''
    """


def insert_all_message_rows_sql() -> str:
    return f"""
        INSERT INTO messages_fts (rowid, text)
        SELECT rowid, {pl_fold_sql_expr("search_text")}
        FROM blocks
        WHERE search_text != ''
    """


def insert_missing_message_rows_sql() -> str:
    return f"""
        WITH missing(rowid, search_text) AS (
            SELECT b.rowid, b.search_text
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE d.id IS NULL AND b.search_text != ''
        )
        INSERT INTO messages_fts (rowid, text)
        SELECT rowid, {pl_fold_sql_expr("search_text")}
        FROM missing
    """


def insert_missing_message_rows_range_sql() -> str:
    return f"""
        WITH missing(rowid, search_text) AS (
            SELECT b.rowid, b.search_text
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE d.id IS NULL
              AND b.search_text != ''
              AND b.rowid > ?
              AND b.rowid <= ?
        )
        INSERT INTO messages_fts (rowid, text)
        SELECT rowid, {pl_fold_sql_expr("search_text")}
        FROM missing
    """


def excess_message_rows_sql(limit: int) -> str:
    return f"""
        SELECT d.id
        FROM messages_fts_docsize AS d
        LEFT JOIN blocks AS b
          ON b.rowid = d.id
         AND b.search_text != ''
        WHERE b.rowid IS NULL
        LIMIT {max(1, int(limit))}
    """


# polylogue-1xc.12: identity-ledger companions to the messages_fts bulk SQL
# above. Every place that bulk-writes/deletes messages_fts rows outside the
# per-row triggers (rebuild, batched missing/excess repair, session-scoped
# repair) pairs its call with the matching function here so
# messages_fts_identity never lags messages_fts for those paths. `FTS_REBUILD_SQL`
# (``DELETE FROM messages_fts``) has no companion constant; use
# ``FTS_IDENTITY_REBUILD_SQL`` alongside it.
FTS_IDENTITY_REBUILD_SQL = "DELETE FROM messages_fts_identity"


def insert_all_message_identity_rows_sql() -> str:
    """Bulk (re)populate ``messages_fts_identity`` from ``blocks``.

    Companion to :func:`insert_all_message_rows_sql`; callers clear the
    table first with :data:`FTS_IDENTITY_REBUILD_SQL`, matching the
    ``messages_fts`` rebuild shape. ``INSERT OR REPLACE`` (polylogue-miwv,
    see :data:`BLOCKS_FTS_TRIGGER_DDL`'s note) is defense-in-depth here even
    though the documented clear-first contract already makes the table empty
    at call time -- a caller that violates that precondition self-heals
    instead of aborting the rebuild.
    """
    return f"""
        INSERT OR REPLACE INTO messages_fts_identity (rowid, block_id, source_hash, recipe_id)
        SELECT rowid, block_id, content_hash, '{FTS_MESSAGES_IDENTITY_RECIPE_ID}'
        FROM blocks
        WHERE search_text != ''
    """


def delete_session_identity_rows_sql(chunk_size: int) -> str:
    """Companion to :func:`delete_session_rows_sql` for ``messages_fts_identity``."""
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"""
        DELETE FROM messages_fts_identity
        WHERE rowid IN (
            SELECT blocks.rowid
            FROM blocks
            WHERE blocks.session_id IN ({placeholders})
        )
    """


def insert_session_identity_rows_sql(chunk_size: int) -> str:
    """Companion to :func:`insert_session_rows_sql` for ``messages_fts_identity``.

    ``INSERT OR REPLACE`` (polylogue-miwv, see :data:`BLOCKS_FTS_TRIGGER_DDL`'s
    note): :func:`delete_session_identity_rows_sql` only removes ledger rows
    whose rowid matches a block *currently* in ``blocks`` for this session --
    an orphaned ledger row left behind by some other path (rowid no longer
    backed by any live block) is invisible to that delete. If this session's
    fresh block set happens to compute a ``block_id`` that orphan still
    holds, a bare INSERT hits ``UNIQUE constraint failed:
    messages_fts_identity.block_id`` and aborts the whole write. REPLACE
    evicts the stale holder instead.
    """
    values = ", ".join("(?)" for _ in range(chunk_size))
    return f"""
        WITH raw_target_sessions(session_id) AS (
            VALUES {values}
        ),
        target_sessions AS (
            SELECT DISTINCT session_id
            FROM raw_target_sessions
        )
        INSERT OR REPLACE INTO messages_fts_identity (rowid, block_id, source_hash, recipe_id)
        SELECT b.rowid, b.block_id, b.content_hash, '{FTS_MESSAGES_IDENTITY_RECIPE_ID}'
        FROM blocks AS b
        JOIN target_sessions AS target
          ON target.session_id = b.session_id
        WHERE b.search_text != ''
    """


def repair_message_identity_rows_range_sql() -> str:
    """UPSERT ``messages_fts_identity`` for indexed rows in a bounded rowid window.

    Unlike :func:`insert_missing_message_rows_range_sql` (INSERT-only, for
    rows entirely absent from ``messages_fts``), this also *overwrites* an
    existing identity row whose ``block_id``/``source_hash``/``recipe_id``
    no longer matches the current block -- the exact rowid-reuse and
    changed-text/changed-recipe cases polylogue-1xc.12 exists to catch and
    self-heal. Scoped to already-indexed rows (present in
    ``messages_fts_docsize``) within ``(?, ?]`` so a bounded repair pass
    over a huge archive stays bounded.

    Two ``ON CONFLICT`` clauses (polylogue-miwv, SQLite 3.35+ multi-target
    UPSERT -- this project already requires 3.43+ for FTS5
    ``contentless_delete``): the first repairs the common rowid-reuse case
    in place, preserving the original "only write if actually different"
    optimization via its ``WHERE``. The second handles the class the single-
    clause form could not -- an existing row holding this exact ``block_id``
    at a *different*, now-stale rowid (an orphan some other path left
    behind) -- by moving that row's identity onto the correct rowid instead
    of raising ``UNIQUE constraint failed: messages_fts_identity.block_id``
    and aborting the whole repair pass.
    """
    return f"""
        INSERT INTO messages_fts_identity (rowid, block_id, source_hash, recipe_id)
        SELECT b.rowid, b.block_id, b.content_hash, '{FTS_MESSAGES_IDENTITY_RECIPE_ID}'
        FROM blocks AS b
        JOIN messages_fts_docsize AS d ON d.id = b.rowid
        WHERE b.search_text != ''
          AND b.rowid > ?
          AND b.rowid <= ?
        ON CONFLICT(rowid) DO UPDATE SET
            block_id = excluded.block_id,
            source_hash = excluded.source_hash,
            recipe_id = excluded.recipe_id
        WHERE messages_fts_identity.block_id != excluded.block_id
           OR messages_fts_identity.source_hash IS NOT excluded.source_hash
           OR messages_fts_identity.recipe_id != excluded.recipe_id
        ON CONFLICT(block_id) DO UPDATE SET
            rowid = excluded.rowid,
            source_hash = excluded.source_hash,
            recipe_id = excluded.recipe_id
    """


def repair_all_message_identity_rows_sql() -> str:
    """Return the set-based identity-ledger reconciliation statement.

    Global recovery must not turn sparse SQLite rowids into millions of
    numeric windows.  This is the same conflict policy as
    :func:`repair_message_identity_rows_range_sql`, but evaluates the archive
    once while the caller owns a global FTS repair transaction.
    """
    return f"""
        INSERT INTO messages_fts_identity (rowid, block_id, source_hash, recipe_id)
        SELECT b.rowid, b.block_id, b.content_hash, '{FTS_MESSAGES_IDENTITY_RECIPE_ID}'
        FROM blocks AS b
        JOIN messages_fts_docsize AS d ON d.id = b.rowid
        WHERE b.search_text != ''
        ON CONFLICT(rowid) DO UPDATE SET
            block_id = excluded.block_id,
            source_hash = excluded.source_hash,
            recipe_id = excluded.recipe_id
        WHERE messages_fts_identity.block_id != excluded.block_id
           OR messages_fts_identity.source_hash IS NOT excluded.source_hash
           OR messages_fts_identity.recipe_id != excluded.recipe_id
        ON CONFLICT(block_id) DO UPDATE SET
            rowid = excluded.rowid,
            source_hash = excluded.source_hash,
            recipe_id = excluded.recipe_id
    """


def message_identity_mismatch_sql() -> str:
    """Exact rowid+block_id+source+recipe identity CONFLICT check for ``messages_fts``.

    Two independent failure classes, summed: (1) an indexed row
    (``messages_fts_docsize`` joined with a still-indexable ``blocks`` row)
    whose identity ledger entry EXISTS but is bound to a different
    ``block_id``, or carries a stale ``source_hash``/``recipe_id`` -- the
    rowid-reuse/changed-text/changed-recipe cases count-only reconciliation
    cannot see because both sides still balance; (2) an identity ledger row
    left over for a rowid no longer present in ``messages_fts_docsize`` at
    all (an orphan, e.g. from a partial/interrupted write).

    Deliberately NOT counted: an indexed row with NO identity ledger entry
    at all. Every per-row trigger arm and bulk companion writes the ledger
    alongside its ``messages_fts`` write; as of polylogue-miwv,
    ``storage/sqlite/archive_tiers/write.py``'s non-bulk full-session-replace
    fast path (``delete_session_rows_sql``/``insert_session_rows_sql``
    called directly, outside this module -- see the polylogue-1xc.12 note in
    ``docs/internals.md``) also pairs each call with its identity companion
    inline, so this coverage gap no longer accrues on ordinary writes. The
    boundary still exists deliberately, as defense-in-depth for archives
    written before polylogue-1xc.12 introduced the ledger (a missing entry
    there is provably safe on its own: nothing reads block identity FROM
    this ledger except this reconciliation query itself, and the next
    trigger-fired mutation at that rowid -- delete or update -- creates a
    correct fresh row regardless of whether one existed before). Only a
    PRESENT-but-WRONG entry is the dangerous rowid-reuse signature this
    check exists to catch -- a consumer trusting the ledger would see a real
    but incorrect binding, not an absence. Counting missing rows here would
    make ``ready`` permanently false on any archive with pre-ledger history
    still in its rowid space, which defeats the point of a readiness signal.
    """
    return f"""
        SELECT
            (
                SELECT COUNT(*)
                FROM messages_fts_docsize AS d
                JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
                JOIN messages_fts_identity AS i ON i.rowid = d.id
                WHERE i.block_id != b.block_id
                   OR i.source_hash IS NOT b.content_hash
                   OR i.recipe_id != '{FTS_MESSAGES_IDENTITY_RECIPE_ID}'
            )
            +
            (
                SELECT COUNT(*)
                FROM messages_fts_identity AS i
                LEFT JOIN messages_fts_docsize AS d ON d.id = i.rowid
                WHERE d.id IS NULL
            )
    """


__all__ = [
    "BLOCKS_FTS_TRIGGER_DDL",
    "FTS_BULK_SESSION_WRITE_GUARD",
    "FTS_IDENTITY_REBUILD_SQL",
    "FTS_INDEXABLE_MESSAGE_COUNT_SQL",
    "FTS_INDEX_DOC_COUNT_SQL",
    "FTS_BLOCK_INPUT_PROJECTION_COLUMNS",
    "FTS_INDEX_EXISTS_SQL",
    "FTS_MESSAGES_IDENTITY_RECIPE_ID",
    "FTS_MESSAGES_IDENTITY_TABLE_SQL",
    "FTS_MESSAGES_TABLE_SQL",
    "FTS_READINESS_BINDING_SURFACE",
    "FTS_READINESS_BINDING_TABLE_SQL",
    "FTS_REBUILD_SQL",
    "FTS_TRIGGER_DDL",
    "FTS_UNICODE_TOKENIZER",
    "IndexedMessage",
    "chunked",
    "delete_session_identity_rows_sql",
    "delete_session_rows_sql",
    "excess_message_rows_sql",
    "insert_all_message_identity_rows_sql",
    "insert_all_message_rows_sql",
    "insert_missing_message_rows_range_sql",
    "insert_missing_message_rows_sql",
    "insert_session_identity_rows_sql",
    "insert_session_rows_sql",
    "message_identity_mismatch_sql",
    "repair_message_identity_rows_range_sql",
    "repair_all_message_identity_rows_sql",
]

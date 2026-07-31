"""Runtime executor for declared, clone-safe index-tier fast-forward plans.

Writer module: index.

``polylogue.storage.sqlite.lifecycle`` declares, for a bounded set of
index-tier schema-version gaps, a contiguous chain of
:class:`~polylogue.storage.sqlite.lifecycle.IndexDeltaDeclaration` entries
that are provably clone-safe (no raw reparse, no consumer-visible semantic
change). Before this module (polylogue-t3gk), nothing consumed that
declaration: ``initialize_archive_database`` forced the same full-rebuild
``RuntimeError`` for every version gap, benign or not -- the live 2026-07-21
incident this closes (a freshly promoted v42 archive could not be opened by
v43 code even though the v43 delta is declared clone-safe).

This module is the missing open-path executor. It applies one declared
:class:`~polylogue.storage.sqlite.lifecycle.IndexFastForwardPlan`'s
operations, one declaration at a time, bumping ``PRAGMA user_version`` and
committing after each declaration -- so a crash mid-chain resumes cleanly on
the next open (the plan is rebuilt from whatever ``user_version`` is on
disk, and every operation is idempotent to re-run against its own already-
converged shape).

Every operation dispatches off :class:`FastForwardOperationKind`, never off a
declaration's version number: a newly declared benign delta becomes
executable purely by adding a correctly typed ``FastForwardOperation`` to
``INDEX_DELTA_DECLARATIONS`` in ``lifecycle.py`` -- no executor change
required for a same-shape delta.
"""

from __future__ import annotations

import re
import sqlite3
import uuid

from polylogue.storage.fts.fts_lifecycle import ensure_fts_index_sync, rebuild_fts_index_sync
from polylogue.storage.fts.sql import insert_all_message_identity_rows_sql, insert_all_message_rows_sql
from polylogue.storage.sqlite.lifecycle import (
    FastForwardOperation,
    FastForwardOperationKind,
    IndexFastForwardPlan,
    resolve_canonical_index_objects,
)

# messages_fts / messages_fts_identity are maintained by the SAME per-row
# triggers (BLOCKS_FTS_TRIGGER_DDL) -- a REBUILD_FTS operation naming either
# table always drops+recreates both declared triggers first.
_MESSAGES_FTS_SURFACE_TABLES = frozenset({"messages_fts", "messages_fts_identity"})
# session_work_events_fts / threads_fts have no declared fast-forward plan
# that narrows to just one of them today; converge every FTS surface via the
# shared full rebuild rather than duplicating its private per-surface helpers
# here (see fts_lifecycle.rebuild_fts_index_sync).
_OTHER_FTS_SURFACE_TABLES = frozenset({"session_work_events_fts", "threads_fts"})

# polylogue-rlvj: fts_freshness_state's v51 REPLACE_TABLE (the new
# state='ready' => balanced-counters CHECK) is, unlike every prior
# CONSTRAINT_ONLY delta here, not guaranteed satisfied by pre-existing rows.
# The exact bug the CHECK exists to make unrepresentable
# (daemon/convergence_stages.py's now-fixed
# `_mark_message_fts_ready_after_targeted_repair`) could already have written
# a poisoned `messages_fts` row -- state='ready' with source_rows !=
# indexed_rows -- on any archive that ran the old code, and the live archive
# audited when this fix landed had exactly that row. A blind shared-column
# copy into the new, stricter schema would raise a CHECK violation and abort
# the whole fast-forward for those archives. The `_REPLACE_TABLE_SANITIZERS`
# entry below downgrades any such row to 'stale' (an honest "unknown, needs
# reconvergence" verdict, not a lost measurement -- the daemon's ordinary
# convergence stages recompute a real value on the next pass) against the OLD
# table, which has no such CHECK, before the generic copy runs.
_REPLACE_TABLE_SANITIZERS: dict[str, str] = {
    "fts_freshness_state": """
        UPDATE "fts_freshness_state"
        SET state = 'stale',
            detail = 'downgraded during index schema fast-forward: a prior '
                     || '''ready'' row failed the new missing_rows=0/excess_rows=0/'
                     || 'duplicate_rows=0/source_rows=indexed_rows invariant'
        WHERE state = 'ready'
          AND NOT (
              missing_rows = 0
              AND excess_rows = 0
              AND duplicate_rows = 0
              AND source_rows = indexed_rows
          )
    """,
}


def _apply_drop_table(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{name}"')


def _apply_replace_view(conn: sqlite3.Connection, name: str, canonical_sql: str) -> None:
    conn.execute(f'DROP VIEW IF EXISTS "{name}"')
    conn.execute(canonical_sql)


def _apply_create_index(conn: sqlite3.Connection, canonical_sql: str) -> None:
    # Canonical index DDL is always "CREATE INDEX IF NOT EXISTS" -- safe to
    # re-run unconditionally, including on an already-converged archive.
    conn.execute(canonical_sql)


def _apply_replace_table(conn: sqlite3.Connection, name: str, canonical_sql: str) -> None:
    """Copy-forward a table onto its canonical shape, preserving shared columns.

    Every declared ``REPLACE_TABLE`` delta is, by ``lifecycle.py``'s own
    per-declaration documentation, a constraint-widening or dead-column-
    removal change: no surviving column changes meaning. Copying the columns
    common to both the existing and canonical shape reproduces the declared
    v33/v36/v38/v41 deltas without a raw reparse -- widened CHECK
    constraints apply to the copied rows for free (SQLite validates on
    insert into the new table), and dropped columns are simply excluded from
    the copy.
    """
    existing_columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{name}")').fetchall()]
    if not existing_columns:
        # Table absent entirely (e.g. a from-scratch fixture) -- canonical
        # DDL already uses IF NOT EXISTS, so a plain create is equivalent.
        conn.execute(canonical_sql)
        return
    sanitizer = _REPLACE_TABLE_SANITIZERS.get(name)
    if sanitizer is not None and {
        "state",
        "missing_rows",
        "excess_rows",
        "duplicate_rows",
        "source_rows",
        "indexed_rows",
    } <= set(existing_columns):
        conn.execute(sanitizer)
    scratch = sqlite3.connect(":memory:")
    try:
        scratch.execute(canonical_sql)
        canonical_columns = [str(row[1]) for row in scratch.execute(f'PRAGMA table_info("{name}")').fetchall()]
    finally:
        scratch.close()
    shared_columns = [column for column in canonical_columns if column in existing_columns]
    if not shared_columns:
        raise RuntimeError(f"fast-forward replace-table {name!r} shares no columns with its canonical shape")
    temporary_name = f"{name}__fast_forward_{uuid.uuid4().hex[:8]}"
    create_temporary, substitutions = re.subn(
        rf'(?i)^CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?"?{re.escape(name)}"?',
        f'CREATE TABLE "{temporary_name}"',
        canonical_sql,
        count=1,
    )
    if substitutions == 0:
        raise RuntimeError(f"could not derive a temporary CREATE TABLE for fast-forward replace of {name!r}")
    quoted_columns = ", ".join(f'"{column}"' for column in shared_columns)
    conn.execute(create_temporary)
    conn.execute(f'INSERT INTO "{temporary_name}" ({quoted_columns}) SELECT {quoted_columns} FROM "{name}"')
    conn.execute(f'DROP TABLE "{name}"')
    conn.execute(f'ALTER TABLE "{temporary_name}" RENAME TO "{name}"')


def _apply_rebuild_fts(conn: sqlite3.Connection, operation: FastForwardOperation) -> None:
    table_names = {name for object_type, name in operation.objects if object_type == "table"}
    trigger_names = [name for object_type, name in operation.objects if object_type == "trigger"]
    for trigger_name in trigger_names:
        conn.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')
    if table_names & _MESSAGES_FTS_SURFACE_TABLES:
        # Recreates messages_fts / messages_fts_identity tables (idempotent,
        # IF NOT EXISTS) and the just-dropped triggers with their CURRENT
        # bodies -- this is exactly how a declared trigger-body refresh (e.g.
        # v43's messages_fts_identity ledger maintenance) reaches disk.
        ensure_fts_index_sync(conn)
        if "messages_fts" in table_names:
            # messages_fts is contentless (content='') -- clearing it requires
            # a bulk delete-all, then a full repopulate from blocks.
            conn.execute("DELETE FROM messages_fts")
            conn.execute(insert_all_message_rows_sql())
        if "messages_fts_identity" in table_names:
            conn.execute("DELETE FROM messages_fts_identity")
            conn.execute(insert_all_message_identity_rows_sql())
    if table_names & _OTHER_FTS_SURFACE_TABLES:
        rebuild_fts_index_sync(conn)


def _apply_operation(
    conn: sqlite3.Connection,
    operation: FastForwardOperation,
    canonical: dict[tuple[str, str], str],
) -> None:
    if operation.kind is FastForwardOperationKind.REBUILD_FTS:
        _apply_rebuild_fts(conn, operation)
        return
    for object_type, name in operation.objects:
        if operation.kind is FastForwardOperationKind.DROP_TABLE:
            _apply_drop_table(conn, name)
        elif operation.kind is FastForwardOperationKind.REPLACE_VIEW:
            _apply_replace_view(conn, name, canonical[(object_type, name)])
        elif operation.kind is FastForwardOperationKind.CREATE_INDEX:
            _apply_create_index(conn, canonical[(object_type, name)])
        elif operation.kind is FastForwardOperationKind.REPLACE_TABLE:
            _apply_replace_table(conn, name, canonical[(object_type, name)])
        else:  # pragma: no cover - exhaustive over FastForwardOperationKind
            raise RuntimeError(f"unhandled fast-forward operation kind: {operation.kind!r}")


def apply_index_fast_forward(conn: sqlite3.Connection, plan: IndexFastForwardPlan) -> None:
    """Apply every declaration in ``plan`` to ``conn``, one declaration at a time.

    Each declaration's operations run inside one ``BEGIN IMMEDIATE`` /
    ``COMMIT`` transaction, followed by bumping ``PRAGMA user_version`` to
    that declaration's version in the SAME transaction. A crash or exception
    before commit rolls back to the previous ``user_version``; the next call
    (rebuilding the plan from whatever ``user_version`` is actually on disk)
    re-applies that declaration from scratch. Every operation kind is
    idempotent against its own already-converged shape, so re-applying a
    declaration that partially landed before a crash is safe.

    Callers must build ``plan`` via
    :func:`polylogue.storage.sqlite.lifecycle.index_fast_forward_plan`, which
    already refuses non-contiguous spans and spans containing a
    ``SEMANTIC_REPARSE`` declaration.
    """
    if not plan.eligible_for_sql_fast_forward:
        raise RuntimeError("index fast-forward plan is not eligible for SQL fast-forward")
    _canonical_lookup_kinds = (
        FastForwardOperationKind.REPLACE_TABLE,
        FastForwardOperationKind.REPLACE_VIEW,
        FastForwardOperationKind.CREATE_INDEX,
    )
    objects_needing_canonical_sql = tuple(
        dict.fromkeys(
            object_ref
            for declaration in plan.declarations
            for operation in declaration.operations
            if operation.kind in _canonical_lookup_kinds
            for object_ref in operation.objects
        )
    )
    canonical = resolve_canonical_index_objects(objects_needing_canonical_sql)
    for declaration in plan.declarations:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for operation in declaration.operations:
                _apply_operation(conn, operation, canonical)
            conn.execute(f"PRAGMA user_version = {declaration.version}")
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()


__all__ = ["apply_index_fast_forward"]

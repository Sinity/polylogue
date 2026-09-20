"""FTS5 automerge tuning for the daemon write path (#1851).

FTS5's default ``automerge=8`` setting merges segments during every write
that accumulates ≥8 level-0 segments.  On a mature archive the existing
segments are hundreds of MB; each small ingest batch adds one tiny
level-0 segment and may trigger a merge of the large existing segments,
writing 8–12 MiB to the WAL regardless of the actual append size.

The fix is two-part:

1. ``automerge=0`` is persisted for each FTS surface so that no merge is
   ever triggered by a write.  The setting lives in the FTS5 ``%_config``
   table and survives connection closure.

2. A periodic background pass calls ``merge=N`` (bounded work-units) for
   each surface, amortising merge cost over time instead of paying it on
   every write.

Part 1 is *not* a startup-only action.  A wiped archive has no ``index.db``
at daemon startup -- it is created later and lazily by the ingest write path
-- so a one-shot startup pass configures nothing and the whole first
post-wipe rebuild runs at FTS5's default ``automerge=8`` (measured on 120K
docs in 80-doc commits, the per-session commit shape: 10.27s load at
``automerge=8`` versus 6.30s at ``automerge=0``, with no query-latency
difference).  The periodic pass therefore *ensures* the setting on every
surface it finds before merging, so the tuning converges from any starting
state and for any surface created after startup.
"""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from pathlib import Path

from polylogue.logging import DEBUG, WARNING, emit


def _fts_surfaces(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Return every FTS5 virtual table present in *conn*, read from its schema.

    Derived, never hand-listed: the tier DDL that declares an FTS5 surface is
    the sole owner of this set, so a newly declared surface is tuned by this
    module the moment it exists. A hand-maintained copy here previously
    omitted a declared surface, leaving it on FTS5's default automerge and
    paying unbounded inline merges on the live-ingest writer.
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE '%USING fts5%' COLLATE NOCASE ORDER BY name"
    ).fetchall()
    return tuple(str(name) for (name,) in rows)


# Work-unit budget per periodic merge call.  500 units bounds each call
# to roughly 2–4 MiB of WAL writes so the periodic merge never becomes an
# unbounded write event of its own.
_PERIODIC_MERGE_WORK_UNITS = 500


def _stored_automerge(conn: sqlite3.Connection, surface: str) -> str | None:
    """Return the persisted ``automerge`` value for *surface*, or ``None``.

    ``None`` means the key has never been written, which is FTS5's default
    ``automerge=8`` — the state this module exists to leave behind.
    """
    try:
        row = conn.execute(f"SELECT v FROM {surface}_config WHERE k = 'automerge'").fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else str(row[0])


def configure_fts_automerge_sync(conn: sqlite3.Connection) -> list[str]:
    """Ensure ``automerge=0`` on each present FTS surface; return those surfaces.

    Idempotent and convergent: a surface already at ``0`` is left alone and no
    write (nor commit) is issued for it, so this is cheap enough to call on
    every periodic maintenance pass rather than once at startup.  Surfaces not
    yet present in the schema (fresh archive before first ingest) are silently
    skipped — which is exactly why this must not be a one-shot startup call.

    The FTS5 ``%_config`` table stores the setting durably, so once a surface
    reports ``0`` here the tuning survives connection and process cycles.
    """
    ensured: list[str] = []
    changed: list[str] = []
    for surface in _fts_surfaces(conn):
        with suppress(sqlite3.OperationalError):
            if _stored_automerge(conn, surface) != "0":
                conn.execute(f"INSERT INTO {surface}({surface}, rank) VALUES('automerge', 0)")
                changed.append(surface)
            ensured.append(surface)
    if changed:
        conn.commit()
        emit("daemon.fts.automerge_configured", outcome="ok", rows=len(changed))
    return ensured


def run_periodic_fts_merge_sync(db: Path) -> None:
    """Run a bounded FTS5 merge pass on every present surface.

    Called from the daemon's periodic maintenance loop.  Each surface gets
    at most ``_PERIODIC_MERGE_WORK_UNITS`` work units, bounding the WAL
    cost to a predictable ceiling instead of merging all pending segments
    in one unbounded pass.

    Also ensures ``automerge=0`` on every surface it finds (see the module
    docstring): the periodic loop re-checks the index on each iteration, so it
    is the one place the tuning cannot be silently skipped by an index that
    did not exist yet at daemon startup.

    Skips gracefully if:
    - ``db`` does not exist (fresh startup race).
    - A surface is absent (not yet created by a first ingest).
    - SQLite is busy (another writer holds the lock).
    """
    if not db.exists():
        return
    conn: sqlite3.Connection | None = None
    try:
        from polylogue.storage.sqlite.connection_profile import open_connection

        conn = open_connection(db, timeout=5.0)
        # Converge the tuning first: on a wiped archive the index did not
        # exist at daemon startup, so this pass is the only thing that will
        # ever take the surfaces off FTS5's default automerge=8.
        configure_fts_automerge_sync(conn)
        merged: list[str] = []
        for surface in _fts_surfaces(conn):
            with suppress(sqlite3.OperationalError):
                conn.execute(f"INSERT INTO {surface}({surface}, rank) VALUES('merge', {_PERIODIC_MERGE_WORK_UNITS})")
                merged.append(surface)
        if merged:
            conn.commit()
            emit(
                "daemon.fts.periodic_merge",
                level=DEBUG,
                outcome="ok",
                rows=len(merged),
                limit=_PERIODIC_MERGE_WORK_UNITS,
                path=db,
            )
    except sqlite3.OperationalError as exc:
        emit(
            "daemon.fts.periodic_merge_skipped",
            level=DEBUG,
            outcome="skipped",
            reason="database_busy",
            path=db,
            error_detail=str(exc),
        )
    except Exception as exc:
        emit(
            "daemon.fts.periodic_merge_failed",
            level=WARNING,
            outcome="degraded",
            reason="merge_failed",
            path=db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
    finally:
        if conn is not None:
            with suppress(Exception):
                conn.close()


__all__ = [
    "configure_fts_automerge_sync",
    "run_periodic_fts_merge_sync",
]

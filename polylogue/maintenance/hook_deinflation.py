"""One-time retroactive repair for hook-events-ingested-as-sessions inflation.

Historically ``sources/hooks.py`` persisted each spooled hook event through
``write_source_raw_session`` with a session origin, minting a ``raw_sessions``
row per hook that materialized into an empty standalone index session
(polylogue-31r1). The going-forward fix (``write_source_hook_event``) stops
creating those rows; this repair removes the ones already accumulated.

Correctness invariants:

* A hook raw is identified ONLY by living under the hook spool
  (``source_path LIKE '%/hooks/pending/%'``); on the live archive that predicate
  is an exact 1:1 with ``raw_hook_events`` (64,896 == 64,896). Genuinely-empty
  real sessions (content quarantined / zero-turn conversations) do NOT match and
  are never touched.
* ``raw_hook_events`` rows and ``raw_payload`` blob refs have no FK to
  ``raw_sessions`` and are deliberately KEPT — the hook evidence and its bytes
  survive; only the spurious session row is removed. The four source FKs into
  ``raw_sessions`` are all ``ON DELETE CASCADE`` (parser census etc.), so the
  delete self-cleans without orphans and does not touch raw-authority plans or
  blockers (hook raws never appear there).
* The index tier is rebuildable; its per-session aux tables cascade from
  ``sessions`` via FK, so a single ``DELETE FROM sessions`` is complete.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_HOOK_RAW_PREDICATE = "source_path LIKE '%/hooks/pending/%'"


@dataclass(frozen=True, slots=True)
class HookDeinflationReport:
    """Counts describing a hook-de-inflation repair (dry-run or applied)."""

    hook_raw_sessions: int
    hook_index_sessions: int
    raw_hook_events_before: int
    raw_hook_events_after: int
    hook_blob_refs_retained: int
    applied: bool


def _hook_raw_ids(source_conn: sqlite3.Connection) -> list[str]:
    return [str(row[0]) for row in source_conn.execute(f"SELECT raw_id FROM raw_sessions WHERE {_HOOK_RAW_PREDICATE}")]


def _load_ids(conn: sqlite3.Connection, table: str, ids: list[str]) -> None:
    conn.execute(f"CREATE TEMP TABLE {table} (id TEXT PRIMARY KEY)")
    conn.executemany(f"INSERT OR IGNORE INTO {table}(id) VALUES (?)", ((rid,) for rid in ids))


def repair_hook_session_inflation(archive_root: Path, *, dry_run: bool = True) -> HookDeinflationReport:
    """Delete hook-derived session rows from both tiers, keeping hook evidence.

    Returns a report with before/after counts. ``dry_run=True`` (default)
    computes the counts without mutating anything.
    """
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.is_file():
        raise FileNotFoundError(source_db)

    with sqlite3.connect(source_db) as source_conn:
        source_conn.execute("PRAGMA foreign_keys = ON")
        hook_raw_ids = _hook_raw_ids(source_conn)
        raw_hook_events_before = int(source_conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone()[0])

        # Index-side target session ids: sessions whose source raw is a hook raw
        # AND which carry no content (belt-and-suspenders against ever deleting a
        # session that somehow materialized messages).
        index_session_ids: list[str] = []
        if index_db.is_file() and hook_raw_ids:
            with sqlite3.connect(index_db) as idx_ro:
                _load_ids(idx_ro, "_hook_raw_ids", hook_raw_ids)
                index_session_ids = [
                    str(row[0])
                    for row in idx_ro.execute(
                        "SELECT session_id FROM sessions "
                        "WHERE raw_id IN (SELECT id FROM _hook_raw_ids) AND message_count = 0"
                    )
                ]

        if dry_run:
            hook_blob_refs = int(
                source_conn.execute(
                    "SELECT COUNT(*) FROM blob_refs WHERE ref_type='raw_payload' "
                    "AND ref_id IN (SELECT raw_id FROM raw_sessions WHERE " + _HOOK_RAW_PREDICATE + ")"
                ).fetchone()[0]
            )
            return HookDeinflationReport(
                hook_raw_sessions=len(hook_raw_ids),
                hook_index_sessions=len(index_session_ids),
                raw_hook_events_before=raw_hook_events_before,
                raw_hook_events_after=raw_hook_events_before,
                hook_blob_refs_retained=hook_blob_refs,
                applied=False,
            )

    # --- Apply: index first (rebuildable), then durable source. ---
    if index_session_ids:
        with sqlite3.connect(index_db) as idx:
            idx.execute("PRAGMA foreign_keys = ON")
            _load_ids(idx, "_del_sessions", index_session_ids)
            idx.execute("DELETE FROM sessions WHERE session_id IN (SELECT id FROM _del_sessions)")

    with sqlite3.connect(source_db) as source_conn:
        source_conn.execute("PRAGMA foreign_keys = ON")
        # Retain the blob refs (bytes stay rooted for GC, matching the new
        # write_source_hook_event path) and raw_hook_events. Delete only the
        # spurious session rows; FK cascade removes their parser-census rows.
        source_conn.execute(f"DELETE FROM raw_sessions WHERE {_HOOK_RAW_PREDICATE}")
        raw_hook_events_after = int(source_conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone()[0])
        hook_blob_refs = int(
            source_conn.execute(
                "SELECT COUNT(*) FROM blob_refs WHERE ref_type='raw_payload' AND source_path LIKE '%/hooks/pending/%'"
            ).fetchone()[0]
        )

    return HookDeinflationReport(
        hook_raw_sessions=len(hook_raw_ids),
        hook_index_sessions=len(index_session_ids),
        raw_hook_events_before=raw_hook_events_before,
        raw_hook_events_after=raw_hook_events_after,
        hook_blob_refs_retained=hook_blob_refs,
        applied=True,
    )

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

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveLocation

# A raw is a verified hook raw ONLY when it lives under the hook spool AND a
# raw_hook_events row was written from the same spool file (same source_path).
# The path pattern alone is not sufficient authority to delete a durable row: a
# legitimate raw whose path merely contains the substring would otherwise be
# removed even without hook evidence (Codex review, PR #3265).
_HOOK_RAW_PREDICATE = (
    "source_path LIKE '%/hooks/pending/%' AND source_path IN (SELECT source_path FROM raw_hook_events)"
)

# Raw-authority tables reference raws by JSON string (input_raw_ids_json), not by
# FK, so deleting a raw does NOT cascade-clean its frontier plans/blockers/census
# rows. A plan whose EVERY input raw is gone is unprocessable and must be pruned,
# or the daemon reconciler throws ("duplicate strategy did not reach its typed
# terminal postcondition") on the dangling plan (live incident 2026-07-22, first
# convergence pass after the hook de-inflation). These are children of
# raw_authority_plans with NO ACTION (RESTRICT) FKs, so delete children first.
# Set-based (single pass, LEFT JOIN on the raw_id PK): expand each plan's input
# raws, GROUP BY plan, keep plans whose inputs are ALL missing. The equivalent
# correlated-subquery form ran ~26 billion ops (>1h) on the live archive; this
# form completes in ~0.3s. A plan with empty inputs produces no json_each rows
# and is correctly excluded (it is not orphaned-by-missing-raw).
_PURELY_ORPHANED_PLANS_SQL = """
    SELECT p.plan_id
    FROM raw_authority_plans p, json_each(p.input_raw_ids_json) j
    LEFT JOIN raw_sessions r ON r.raw_id = j.value
    GROUP BY p.plan_id
    HAVING SUM(CASE WHEN r.raw_id IS NULL THEN 1 ELSE 0 END) > 0
       AND SUM(CASE WHEN r.raw_id IS NOT NULL THEN 1 ELSE 0 END) = 0
"""


@dataclass(frozen=True, slots=True)
class HookDeinflationReport:
    """Counts describing a hook-de-inflation repair (dry-run or applied)."""

    hook_raw_sessions: int
    hook_index_sessions: int
    raw_hook_events_before: int
    raw_hook_events_after: int
    hook_blob_refs_retained: int
    orphaned_authority_plans: int
    applied: bool


def _hook_raw_ids(source_conn: sqlite3.Connection) -> list[str]:
    return [str(row[0]) for row in source_conn.execute(f"SELECT raw_id FROM raw_sessions WHERE {_HOOK_RAW_PREDICATE}")]


def _load_ids(conn: sqlite3.Connection, table: str, ids: list[str]) -> None:
    conn.execute(f"CREATE TEMP TABLE {table} (id TEXT PRIMARY KEY)")
    conn.executemany(f"INSERT OR IGNORE INTO {table}(id) VALUES (?)", ((rid,) for rid in ids))


def _count_orphaned_authority_plans(conn: sqlite3.Connection) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM ({_PURELY_ORPHANED_PLANS_SQL})").fetchone()[0])


def _delete_orphaned_authority(conn: sqlite3.Connection) -> int:
    """Prune raw-authority plans whose every input raw is gone, plus their
    blocker/census children. Children first (NO ACTION FKs to plans).

    ``_orphan_plans`` is created WITH a primary-key index on ``plan_id``: the
    child tables (census_plans ~405k, census_post_plans ~324k rows) don't index
    ``plan_id`` as a leading column, so each ``DELETE ... WHERE plan_id IN
    (SELECT plan_id FROM _orphan_plans)`` full-scans the child once and needs an
    O(log) membership probe per row. Without the PK the probe degrades to a full
    scan of the 64k-row orphan set — ~26 billion ops, observed as a >1h hang on
    the live archive."""
    conn.execute("CREATE TEMP TABLE _orphan_plans (plan_id TEXT PRIMARY KEY)")
    conn.execute(f"INSERT INTO _orphan_plans (plan_id) {_PURELY_ORPHANED_PLANS_SQL}")
    count = int(conn.execute("SELECT COUNT(*) FROM _orphan_plans").fetchone()[0])
    # Existing child indexes on plan_id are partial (WHERE resolved_at_ms IS NULL
    # / WHERE selected = 1) or non-leading (PK is (census_id, plan_id)), so the FK
    # RESTRICT check on the parent plan delete would full-scan each child per plan
    # (~51 billion ops, the observed >1h hang). Full temporary plan_id indexes make
    # both the IN-delete and the RESTRICT check index-driven; dropped before commit
    # so the durable schema is unchanged.
    child_indexes = {
        "_tmp_hookdeflate_blk_plan": "raw_authority_blockers",
        "_tmp_hookdeflate_cp_plan": "raw_authority_census_plans",
        "_tmp_hookdeflate_cpp_plan": "raw_authority_census_post_plans",
    }
    for index_name, table in child_indexes.items():
        conn.execute(f"CREATE INDEX {index_name} ON {table}(plan_id)")
    for table in child_indexes.values():
        conn.execute(f"DELETE FROM {table} WHERE plan_id IN (SELECT plan_id FROM _orphan_plans)")
    conn.execute("DELETE FROM raw_authority_plans WHERE plan_id IN (SELECT plan_id FROM _orphan_plans)")
    for index_name in child_indexes:
        conn.execute(f"DROP INDEX {index_name}")
    conn.execute("DROP TABLE _orphan_plans")
    return count


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
                # Currently-orphaned plans (e.g. a prior incomplete repair left
                # some); apply also prunes any this run newly orphans.
                orphaned_authority_plans=_count_orphaned_authority_plans(source_conn),
                applied=False,
            )

    # --- Apply: refuse to race the sole-writer daemon. ---
    # The repair mutates two tiers in separate transactions after collecting
    # candidate ids; a live daemon could ingest/materialize in between, deleting
    # source rows for sessions it just created or leaving a cross-tier snapshot
    # the repair never examined (Codex review, PR #3265). Require the daemon to
    # be stopped, matching every other offline maintenance mutation.
    active_config = Config(
        archive_root=archive_root,
        render_root=render_root(),
        sources=[],
        db_path=ArchiveLocation.resolve(archive_root).active_index_path,
    )
    if reason := offline_maintenance_block_reason(active_config, active=True, dry_run=False):
        raise RuntimeError(reason)

    # --- Index first (rebuildable), then durable source. ---
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
        # Prune raw-authority plans/blockers/census now dangling on the deleted
        # raws (no FK to raw_sessions, so no cascade did this). Runs in the same
        # transaction, after the raw delete, so "orphaned" is computed against
        # the post-delete raw set.
        orphaned_authority = _delete_orphaned_authority(source_conn)
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
        orphaned_authority_plans=orphaned_authority,
        applied=True,
    )

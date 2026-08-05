"""Read-only classification: re-key orphaned hook-event blob refs.

polylogue-tfzw0: before the ``hook_payload`` ref_type existed (source schema
v22), ``write_source_hook_event`` wrote every hook event's durable blob ref as
``ref_type='raw_payload'`` keyed by a synthetic ``raw_id`` -- a value that
never corresponds to any ``raw_sessions`` row, because a hook event
deliberately never mints one (polylogue-31r1). Every such row is therefore an
orphan under the corrected liveness join (``storage/blob_gc.py``,
``_blob_refs_still_live``): its ``ref_type`` says "join against
``raw_sessions``", but nothing in ``raw_sessions`` ever matches. Measured at
73,427 rows / ~1.94 GiB on the live archive.

This module classifies which of those orphans can be *provably* re-keyed to
the ``raw_hook_events`` row they actually belong to, without touching
anything. The synthetic id a hook event's blob ref used
(``deterministic_raw_session_id(origin, source_path, source_index=0,
blob_hash, native_id)``, see ``archive_tiers/source_write.py``) is
recomputable from data already on hand: the orphaned ref's own ``blob_hash``
plus a candidate ``raw_hook_events`` row's ``origin``/``native_id`` (source
tier does not store hook events' ``origin`` on ``blob_refs`` itself, so
candidates are drawn from ``raw_hook_events`` rows sharing the ref's
``source_path`` -- the value ``write_source_hook_event`` always writes
identically to both the ref and the hook-event row it accompanies). A ref
whose recomputed id matches its own ``ref_id`` exactly is provably the blob
for that hook event; anything else (no match, or more than one candidate
matching) is left alone rather than guessed at.

Read-only: only ``polylogue.maintenance.hook_payload_ref_reconciliation_apply``
mutates the archive, and only for the confirmed, non-ambiguous matches this
module reports.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id


@dataclass(frozen=True, slots=True)
class HookPayloadRefReconciliationCandidate:
    """One orphaned ``raw_payload`` blob ref provably re-keyable to a hook event."""

    blob_hash: bytes
    orphaned_ref_id: str
    source_path: str | None
    size_bytes: int
    acquired_at_ms: int
    hook_event_id: str


@dataclass(frozen=True, slots=True)
class HookPayloadRefReconciliationPlan:
    """Read-only projection of the orphaned raw_payload-ref population.

    ``scanned_count`` is every ``raw_payload`` blob ref whose ``ref_id`` does
    not match any ``raw_sessions`` row (the orphan population this module
    targets). ``matched`` is the provable, unambiguous subset;
    ``unmatched_count`` covers everything else (no candidate hook event
    shares its ``source_path``, or more than one candidate's recomputed id
    collided -- both left untouched rather than guessed at).
    """

    scanned_count: int
    matched: tuple[HookPayloadRefReconciliationCandidate, ...]
    unmatched_count: int

    @property
    def matched_bytes(self) -> int:
        return sum(candidate.size_bytes for candidate in self.matched)


_MATCH_TABLE = "hook_payload_ref_reconciliation_matches"
_ORPHAN_TABLE = "hook_payload_ref_reconciliation_orphans"
_HOOK_EVIDENCE_TABLE = "hook_payload_ref_reconciliation_hook_evidence"
_HOOK_TABLE = "hook_payload_ref_reconciliation_hooks"
_IDENTITY_TABLE = "hook_payload_ref_reconciliation_identity_matches"
_AMBIGUOUS_TABLE = "hook_payload_ref_reconciliation_ambiguous"


def _deterministic_raw_session_id_udf(
    origin: object, source_path: object, source_index: object, blob_hash: object, native_id: object
) -> str | None:
    if origin is None or blob_hash is None:
        return None
    try:
        if not isinstance(source_index, (int, str)) or not isinstance(blob_hash, bytes):
            return None
        return deterministic_raw_session_id(
            str(origin),
            "" if source_path is None else str(source_path),
            int(source_index),
            blob_hash,
            None if native_id is None else str(native_id),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _create_match_stage(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
    """Stage exact hook matches in SQLite, avoiding a Python cross-product."""

    conn.create_function("polylogue_deterministic_raw_session_id", 5, _deterministic_raw_session_id_udf)
    for table in (_MATCH_TABLE, _ORPHAN_TABLE, _HOOK_EVIDENCE_TABLE, _HOOK_TABLE, _IDENTITY_TABLE, _AMBIGUOUS_TABLE):
        conn.execute(f"DROP TABLE IF EXISTS temp.{table}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_ORPHAN_TABLE} AS
        SELECT b.blob_hash, b.ref_id, b.source_path, b.size_bytes, b.acquired_at_ms
        FROM blob_refs AS b
        WHERE b.ref_type = 'raw_payload'
          AND NOT EXISTS (SELECT 1 FROM raw_sessions AS r WHERE r.raw_id = b.ref_id)
        """
    )
    conn.execute(f"CREATE INDEX {_ORPHAN_TABLE}_source_path ON {_ORPHAN_TABLE}(source_path, blob_hash)")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_HOOK_EVIDENCE_TABLE} AS
        SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
        FROM (
            SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
            FROM raw_hook_events AS h
            JOIN (
                SELECT DISTINCT source_path, blob_hash
                FROM {_ORPHAN_TABLE}
                WHERE source_path IS NOT NULL
            ) AS o
              ON o.source_path IS h.source_path
             AND o.blob_hash = h.blob_hash
            UNION
            SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
            FROM raw_hook_events AS h
            JOIN (
                SELECT DISTINCT source_path
                FROM {_ORPHAN_TABLE}
                WHERE source_path IS NOT NULL
            ) AS o
              ON o.source_path IS h.source_path
            WHERE h.blob_hash IS NULL
        ) AS h
        """
    )
    conn.execute(f"CREATE INDEX {_HOOK_EVIDENCE_TABLE}_source_hash ON {_HOOK_EVIDENCE_TABLE}(source_path, blob_hash)")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_HOOK_TABLE} AS
        SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
        FROM {_HOOK_EVIDENCE_TABLE} AS h
        WHERE NOT EXISTS (
            SELECT 1 FROM blob_refs AS b
            WHERE b.ref_type = 'hook_payload'
              AND b.ref_id = h.hook_event_id
        )
        """
    )
    conn.execute(f"CREATE INDEX {_HOOK_TABLE}_source_path ON {_HOOK_TABLE}(source_path, blob_hash)")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_IDENTITY_TABLE} AS
        SELECT o.blob_hash, o.ref_id, h.hook_event_id, h.blob_hash AS hook_blob_hash
        FROM {_ORPHAN_TABLE} AS o
        JOIN {_HOOK_EVIDENCE_TABLE} AS h
          ON h.source_path IS o.source_path
         AND h.blob_hash = o.blob_hash
         AND polylogue_deterministic_raw_session_id(
               h.origin, o.source_path, 0, o.blob_hash, h.native_id
             ) = o.ref_id
        UNION ALL
        SELECT o.blob_hash, o.ref_id, h.hook_event_id, h.blob_hash AS hook_blob_hash
        FROM {_ORPHAN_TABLE} AS o
        JOIN (
            SELECT source_path
            FROM {_ORPHAN_TABLE}
            GROUP BY source_path
            HAVING COUNT(*) = 1
        ) AS unique_orphan_paths
          ON unique_orphan_paths.source_path IS o.source_path
        JOIN (
            SELECT source_path
            FROM {_HOOK_EVIDENCE_TABLE}
            WHERE blob_hash IS NULL
            GROUP BY source_path
            HAVING COUNT(*) = 1
        ) AS unique_legacy_paths
          ON unique_legacy_paths.source_path IS o.source_path
        JOIN {_HOOK_EVIDENCE_TABLE} AS h
          ON h.source_path IS o.source_path
         AND h.blob_hash IS NULL
         AND polylogue_deterministic_raw_session_id(
               h.origin, o.source_path, 0, o.blob_hash, h.native_id
             ) = o.ref_id
        """
    )
    conn.execute(f"CREATE INDEX {_IDENTITY_TABLE}_candidate ON {_IDENTITY_TABLE}(blob_hash, ref_id)")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_AMBIGUOUS_TABLE} (
            blob_hash BLOB NOT NULL,
            orphaned_ref_id TEXT NOT NULL,
            PRIMARY KEY (blob_hash, orphaned_ref_id)
        ) STRICT
        """
    )
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {_AMBIGUOUS_TABLE} (blob_hash, orphaned_ref_id)
        SELECT blob_hash, ref_id
        FROM {_IDENTITY_TABLE}
        GROUP BY blob_hash, ref_id
        HAVING COUNT(DISTINCT hook_event_id) > 1
        """
    )
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {_AMBIGUOUS_TABLE} (blob_hash, orphaned_ref_id)
        SELECT o.blob_hash, o.ref_id
        FROM {_ORPHAN_TABLE} AS o
        WHERE EXISTS (
            SELECT 1 FROM {_HOOK_TABLE} AS h
            WHERE h.source_path IS o.source_path AND h.blob_hash IS NULL
        )
          AND (
            (SELECT COUNT(*) FROM {_ORPHAN_TABLE} AS o2 WHERE o2.source_path IS o.source_path) > 1
            OR (SELECT COUNT(*) FROM {_HOOK_TABLE} AS h2
                WHERE h2.source_path IS o.source_path AND h2.blob_hash IS NULL) > 1
          )
        GROUP BY o.blob_hash, o.ref_id
        """
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE {_MATCH_TABLE} (
            blob_hash BLOB NOT NULL,
            orphaned_ref_id TEXT NOT NULL,
            source_path TEXT,
            size_bytes INTEGER NOT NULL,
            acquired_at_ms INTEGER NOT NULL,
            hook_event_id TEXT NOT NULL,
            PRIMARY KEY (blob_hash, orphaned_ref_id)
        ) STRICT
        """
    )
    conn.execute(
        f"""
        INSERT INTO {_MATCH_TABLE} (
            blob_hash, orphaned_ref_id, source_path, size_bytes, acquired_at_ms, hook_event_id
        )
        SELECT o.blob_hash, o.ref_id, o.source_path, o.size_bytes, o.acquired_at_ms, MIN(h.hook_event_id)
        FROM {_ORPHAN_TABLE} AS o
        JOIN {_HOOK_TABLE} AS h
          ON h.source_path IS o.source_path
         AND h.blob_hash = o.blob_hash
         AND polylogue_deterministic_raw_session_id(h.origin, o.source_path, 0, o.blob_hash, h.native_id) = o.ref_id
        WHERE NOT EXISTS (
                SELECT 1 FROM {_AMBIGUOUS_TABLE} AS a
                WHERE a.blob_hash = o.blob_hash AND a.orphaned_ref_id = o.ref_id
              )
        GROUP BY o.blob_hash, o.ref_id, o.source_path, o.size_bytes, o.acquired_at_ms
        HAVING COUNT(*) = 1
        """
    )
    # Legacy rows have no hook blob_hash, so a shared source path cannot be
    # used as a bounded identity key. Leave those paths untouched unless both
    # sides contain exactly one row. This preserves the ambiguity rule without
    # evaluating an orphan-by-hook cross-product.
    conn.execute(
        f"""
        INSERT INTO {_MATCH_TABLE} (
            blob_hash, orphaned_ref_id, source_path, size_bytes, acquired_at_ms, hook_event_id
        )
        SELECT o.blob_hash, o.ref_id, o.source_path, o.size_bytes, o.acquired_at_ms, h.hook_event_id
        FROM {_ORPHAN_TABLE} AS o
        JOIN {_HOOK_TABLE} AS h
          ON h.source_path IS o.source_path
         AND h.blob_hash IS NULL
        WHERE NOT EXISTS (
                SELECT 1 FROM {_AMBIGUOUS_TABLE} AS a
                WHERE a.blob_hash = o.blob_hash AND a.orphaned_ref_id = o.ref_id
              )
          AND (SELECT COUNT(*) FROM {_ORPHAN_TABLE} AS o2 WHERE o2.source_path IS o.source_path) = 1
          AND (SELECT COUNT(*) FROM {_HOOK_TABLE} AS h2
               WHERE h2.source_path IS h.source_path AND h2.blob_hash IS NULL) = 1
          AND polylogue_deterministic_raw_session_id(h.origin, o.source_path, 0, o.blob_hash, h.native_id) = o.ref_id
        """
    )
    scanned_count = int(conn.execute(f"SELECT COUNT(*) FROM {_ORPHAN_TABLE}").fetchone()[0])
    matched_count = int(conn.execute(f"SELECT COUNT(*) FROM {_MATCH_TABLE}").fetchone()[0])
    matched_bytes = int(conn.execute(f"SELECT COALESCE(SUM(size_bytes), 0) FROM {_MATCH_TABLE}").fetchone()[0])
    ambiguous_count = int(conn.execute(f"SELECT COUNT(*) FROM {_AMBIGUOUS_TABLE}").fetchone()[0])
    return scanned_count, matched_count, matched_bytes, ambiguous_count


def plan_hook_payload_ref_reconciliation(conn: sqlite3.Connection) -> HookPayloadRefReconciliationPlan:
    """Classify orphaned ``raw_payload`` blob refs against candidate hook events.

    Read-only: issues only ``SELECT`` statements. Safe against a read-only
    connection.
    """
    if not _table_exists(conn, "blob_refs") or not _table_exists(conn, "raw_hook_events"):
        return HookPayloadRefReconciliationPlan(scanned_count=0, matched=(), unmatched_count=0)

    scanned_count, matched_count, _matched_bytes, _ambiguous_count = _create_match_stage(conn)
    if not scanned_count:
        return HookPayloadRefReconciliationPlan(scanned_count=0, matched=(), unmatched_count=0)

    matched = tuple(
        HookPayloadRefReconciliationCandidate(
            blob_hash=bytes(row[0]),
            orphaned_ref_id=str(row[1]),
            source_path=str(row[2]) if row[2] is not None else None,
            size_bytes=int(row[3]),
            acquired_at_ms=int(row[4]),
            hook_event_id=str(row[5]),
        )
        for row in conn.execute(
            f"""
            SELECT blob_hash, orphaned_ref_id, source_path, size_bytes, acquired_at_ms, hook_event_id
            FROM {_MATCH_TABLE}
            ORDER BY orphaned_ref_id, blob_hash
            """
        )
    )

    return HookPayloadRefReconciliationPlan(
        scanned_count=scanned_count,
        matched=matched,
        unmatched_count=scanned_count - matched_count,
    )


__all__ = [
    "HookPayloadRefReconciliationCandidate",
    "HookPayloadRefReconciliationPlan",
    "plan_hook_payload_ref_reconciliation",
]

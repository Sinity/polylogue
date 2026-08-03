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


def _orphaned_raw_payload_refs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT blob_hash, ref_id, source_path, size_bytes, acquired_at_ms
        FROM blob_refs
        WHERE ref_type = 'raw_payload'
          AND NOT EXISTS (SELECT 1 FROM raw_sessions WHERE raw_sessions.raw_id = blob_refs.ref_id)
        """
    ).fetchall()


def _hook_events_missing_blob_hash(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT hook_event_id, origin, native_id, source_path
        FROM raw_hook_events
        WHERE blob_hash IS NULL
        """
    ).fetchall()


def plan_hook_payload_ref_reconciliation(conn: sqlite3.Connection) -> HookPayloadRefReconciliationPlan:
    """Classify orphaned ``raw_payload`` blob refs against candidate hook events.

    Read-only: issues only ``SELECT`` statements. Safe against a read-only
    connection.
    """
    conn.row_factory = sqlite3.Row
    if not _table_exists(conn, "blob_refs") or not _table_exists(conn, "raw_hook_events"):
        return HookPayloadRefReconciliationPlan(scanned_count=0, matched=(), unmatched_count=0)

    orphaned_refs = _orphaned_raw_payload_refs(conn)
    if not orphaned_refs:
        return HookPayloadRefReconciliationPlan(scanned_count=0, matched=(), unmatched_count=0)

    candidates_by_source_path: dict[str | None, list[sqlite3.Row]] = {}
    for row in _hook_events_missing_blob_hash(conn):
        candidates_by_source_path.setdefault(row["source_path"], []).append(row)

    matched: list[HookPayloadRefReconciliationCandidate] = []
    unmatched_count = 0
    for ref in orphaned_refs:
        blob_hash = bytes(ref["blob_hash"])
        candidates = candidates_by_source_path.get(ref["source_path"], ())
        hits: list[str] = []
        for candidate in candidates:
            recomputed = deterministic_raw_session_id(
                candidate["origin"],
                ref["source_path"] or "",
                0,
                blob_hash,
                candidate["native_id"],
            )
            if recomputed == ref["ref_id"]:
                hits.append(candidate["hook_event_id"])
        if len(hits) == 1:
            matched.append(
                HookPayloadRefReconciliationCandidate(
                    blob_hash=blob_hash,
                    orphaned_ref_id=ref["ref_id"],
                    source_path=ref["source_path"],
                    size_bytes=int(ref["size_bytes"]),
                    acquired_at_ms=int(ref["acquired_at_ms"]),
                    hook_event_id=hits[0],
                )
            )
        else:
            # Zero hits (no candidate hook event shares this source_path, or
            # none recomputes to this exact ref_id) or more than one hit (a
            # genuine collision -- recompute ambiguity) are both left
            # unmatched rather than guessed at.
            unmatched_count += 1

    return HookPayloadRefReconciliationPlan(
        scanned_count=len(orphaned_refs),
        matched=tuple(matched),
        unmatched_count=unmatched_count,
    )


__all__ = [
    "HookPayloadRefReconciliationCandidate",
    "HookPayloadRefReconciliationPlan",
    "plan_hook_payload_ref_reconciliation",
]

"""Read-only classification: quarantined raws whose membership already decided.

polylogue-lb39z (Phase 1, item 2): the membership pipeline classifies most raw
revisions correctly via ``session_revision_membership.classify_membership_
revisions`` and records the verdict in ``raw_session_memberships.decision`` --
but nothing propagates that verdict back onto ``raw_sessions.revision_
authority``. A live, read-only audit measured 15,737 quarantined
``raw_sessions`` rows (42.95GB) whose membership row already says
``applied`` / ``superseded_equivalent`` / ``superseded_prefix`` with
membership ``revision_authority='byte_proven'`` -- a decided fact sitting
one column away from where the rest of the archive reads authority from.

This module only classifies. :mod:`polylogue.maintenance.raw_membership_writeback_apply`
is the "act" half, following the same dry-run-by-default,
verified-backup-required-to-apply, immutable-receipt pattern as the
already-merged ``raw_live_source_reconciliation_apply`` (polylogue-u19l).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

#: The only membership decisions this write-back ever acts on. Every one of
#: these means the membership pipeline already established this raw's
#: content is either the accepted revision or a proven-superseded prior
#: revision of it -- never ``ambiguous`` (a real, unresolved conflict) or
#: ``deferred`` (not yet decided) or ``NULL`` (no membership evidence at all,
#: see item 3 / append-chain backfill for that population instead).
WRITE_BACK_DECISIONS: tuple[str, ...] = ("applied", "superseded_equivalent", "superseded_prefix")

#: SQLite ``CASE`` ordering matching the report's own precedence when a raw
#: has more than one membership row (rare: a retired-sibling reparse under a
#: different logical key, see ``raw_membership_retired_full_revision_siblings``).
#: Prefer the strongest, most-specific decision deterministically.
_DECISION_PRECEDENCE_SQL = (
    "CASE m.decision "
    "WHEN 'applied' THEN 0 "
    "WHEN 'superseded_equivalent' THEN 1 "
    "WHEN 'superseded_prefix' THEN 2 "
    "ELSE 3 END"
)


@dataclass(frozen=True, slots=True)
class WriteBackCandidate:
    """One quarantined raw whose membership already decided its fate."""

    raw_id: str
    logical_source_key: str
    provider_session_id: str
    decision: str
    blob_size: int


@dataclass(frozen=True, slots=True)
class RawMembershipWriteBackPlan:
    scanned_count: int
    candidates: tuple[WriteBackCandidate, ...]
    candidate_bytes: int


def plan_raw_membership_writeback(conn: sqlite3.Connection, *, limit: int | None = None) -> RawMembershipWriteBackPlan:
    """Classify quarantined ``raw_sessions`` rows with an already-decided membership verdict.

    Never mutates anything. Only ever selects a membership row whose
    ``decision`` is in :data:`WRITE_BACK_DECISIONS` *and* whose own
    ``revision_authority`` is already ``'byte_proven'`` -- genuinely decided,
    byte-proven membership evidence, never a raw's own unresolved/ambiguous
    membership row.
    """
    scanned_count = int(
        conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE revision_authority = 'quarantined'").fetchone()[0]
    )
    placeholders = ", ".join("?" for _ in WRITE_BACK_DECISIONS)
    query = f"""
        SELECT raw_id, logical_source_key, provider_session_id, decision, blob_size
        FROM (
            SELECT
                r.raw_id AS raw_id,
                m.logical_source_key AS logical_source_key,
                m.provider_session_id AS provider_session_id,
                m.decision AS decision,
                r.blob_size AS blob_size,
                ROW_NUMBER() OVER (
                    PARTITION BY r.raw_id ORDER BY {_DECISION_PRECEDENCE_SQL}
                ) AS rn
            FROM raw_sessions AS r
            JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
            WHERE r.revision_authority = 'quarantined'
              AND m.revision_authority = 'byte_proven'
              AND m.decision IN ({placeholders})
        )
        WHERE rn = 1
        ORDER BY raw_id
    """
    if limit is not None:
        query += " LIMIT ?"
        rows = conn.execute(query, (*WRITE_BACK_DECISIONS, limit)).fetchall()
    else:
        rows = conn.execute(query, WRITE_BACK_DECISIONS).fetchall()
    candidates = tuple(
        WriteBackCandidate(
            raw_id=str(raw_id),
            logical_source_key=str(logical_source_key),
            provider_session_id=str(provider_session_id),
            decision=str(decision),
            blob_size=int(blob_size),
        )
        for raw_id, logical_source_key, provider_session_id, decision, blob_size in rows
    )
    return RawMembershipWriteBackPlan(
        scanned_count=scanned_count,
        candidates=candidates,
        candidate_bytes=sum(candidate.blob_size for candidate in candidates),
    )


__all__ = [
    "WRITE_BACK_DECISIONS",
    "RawMembershipWriteBackPlan",
    "WriteBackCandidate",
    "plan_raw_membership_writeback",
]

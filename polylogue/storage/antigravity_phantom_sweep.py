"""Read-only sweep: find ``antigravity-session`` phantom-metadata sessions.

polylogue-eo81 / polylogue-msia: every ``antigravity-session`` row (116/116,
measured 2026-07-31) was materialized from a per-artifact
``~/.gemini/antigravity/brain/<uuid>/*.md.metadata.json`` sidecar -- one
single-message "session" per metadata file, none of which hold real
conversation content (the real content lives in ``conversations/*.pb``
trajectories, acquired by a separate fix, PR #3441/polylogue-eo81). These
degraded fragments are already tagged at ingest time
(``session_tags.tag = 'degraded:brain-metadata-fragment'``, PR #1856) --
this module's read-only job is to turn that existing tag into an explicit,
countable purge candidate list, cross-checked against the raw payload each
session's ``raw_id`` points at so a caller can see exactly what would be
removed before anything is mutated.

Separate from this report: an explicit, ``--apply``-gated actuator
(``devtools/antigravity_phantom_purge_apply.py``) that actually deletes the
flagged sessions (``ArchiveStore.delete_sessions``) and persists a
``raw_artifacts`` classification for their raw rows
(``materialize_artifact_observations_for_raw_ids``). This module only
classifies and counts; it never mutates either tier.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from polylogue.core.enums import Origin

_PHANTOM_TAG = "degraded:brain-metadata-fragment"


@dataclass(frozen=True, slots=True)
class AntigravityPhantomCandidate:
    session_id: str
    raw_id: str | None
    message_count: int
    source_path: str | None
    """``raw_sessions.source_path`` for ``raw_id``, when the raw row is still
    present in ``source.db``. ``None`` if ``raw_id`` is unset or the raw row
    is missing (reported honestly rather than assumed)."""
    blob_size: int | None


@dataclass(slots=True)
class AntigravityPhantomSweepPlan:
    scanned_count: int = 0
    candidates: list[AntigravityPhantomCandidate] = field(default_factory=list)
    missing_raw_row_count: int = 0
    """Candidates whose ``raw_id`` no longer resolves to a ``source.db`` row
    (e.g. already GC'd). Still included in ``candidates`` -- the session-tag
    evidence alone is sufficient to purge the session -- but counted
    separately so the report is honest about incomplete cross-checks."""

    @property
    def raw_bytes(self) -> int:
        return sum(c.blob_size or 0 for c in self.candidates)

    def session_ids(self) -> tuple[str, ...]:
        return tuple(c.session_id for c in self.candidates)

    def raw_ids(self) -> tuple[str, ...]:
        return tuple(c.raw_id for c in self.candidates if c.raw_id is not None)


def scan_antigravity_phantom_sessions(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection | None = None,
    *,
    limit: int | None = None,
) -> AntigravityPhantomSweepPlan:
    """Read-only: list ``antigravity-session`` rows tagged as brain-metadata
    fragments, cross-checked against their raw payload when *source_conn* is
    given.

    *index_conn* and *source_conn* should both be opened read-only
    (``?mode=ro``) by the caller -- this function never writes.
    """
    index_conn.row_factory = sqlite3.Row
    query = """
        SELECT s.session_id, s.raw_id, s.message_count
        FROM sessions AS s
        JOIN session_tags AS t ON t.session_id = s.session_id
        WHERE s.origin = ? AND t.tag = ?
        ORDER BY s.session_id
    """
    params: list[object] = [Origin.ANTIGRAVITY_SESSION.value, _PHANTOM_TAG]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    rows = index_conn.execute(query, params).fetchall()

    raw_by_id: dict[str, sqlite3.Row] = {}
    if source_conn is not None:
        source_conn.row_factory = sqlite3.Row
        raw_ids = [row["raw_id"] for row in rows if row["raw_id"] is not None]
        for batch_start in range(0, len(raw_ids), 250):
            batch = raw_ids[batch_start : batch_start + 250]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            found = source_conn.execute(
                f"SELECT raw_id, source_path, blob_size FROM raw_sessions WHERE raw_id IN ({placeholders})",
                batch,
            ).fetchall()
            for raw_row in found:
                raw_by_id[raw_row["raw_id"]] = raw_row

    plan = AntigravityPhantomSweepPlan()
    for row in rows:
        plan.scanned_count += 1
        raw_id = row["raw_id"]
        raw_row = raw_by_id.get(raw_id) if raw_id is not None else None
        if raw_id is not None and source_conn is not None and raw_row is None:
            plan.missing_raw_row_count += 1
        plan.candidates.append(
            AntigravityPhantomCandidate(
                session_id=row["session_id"],
                raw_id=raw_id,
                message_count=row["message_count"],
                source_path=raw_row["source_path"] if raw_row is not None else None,
                blob_size=raw_row["blob_size"] if raw_row is not None else None,
            )
        )
    return plan


__all__ = [
    "AntigravityPhantomCandidate",
    "AntigravityPhantomSweepPlan",
    "scan_antigravity_phantom_sessions",
]

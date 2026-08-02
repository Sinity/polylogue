"""Read-only sweep: find ``agent-*.meta.json`` subagent-sidecar phantom sessions.

polylogue-ioz7 (direct fix for the polylogue-b508 audit finding): before
2026-07-28, a per-subagent metadata sidecar file
(``~/.claude/projects/<project>/<session-uuid>/subagents/agent-<id>.meta.json``)
was materialized into the archive as its own, empty ``claude-code-session``
row -- a duplicate phantom standing beside the real subagent transcript
(ingested separately and correctly under the same ``agent-<id>`` native id,
without the ``.meta`` suffix). The producer bug is already fixed (raws
acquired after 2026-07-28 correctly produce no session); this module answers,
read-only, "which already-materialized index sessions are this exact
residue" -- reusing the bead's own repro predicate as the classifier:

    message_count = 0  AND  raw_sessions.source_path LIKE '%.meta.json'

Live-archive verification (2026-08-02, read-only): exactly 4,945 sessions
match, and all 4,945 also have ``native_id LIKE 'agent-%.meta'`` -- the two
independent predicates agree completely, matching polylogue-b508's own
audit numbers exactly.

Separate from this module: an explicit, ``--apply``-gated actuator
(``polylogue/maintenance/agent_meta_sidecar_purge_apply.py`` +
``devtools/agent_meta_sidecar_purge_apply.py``) that actually deletes the
flagged ``sessions`` rows from ``index.db`` (the rebuildable tier) --
``raw_sessions`` rows and blobs in ``source.db`` are never touched, and
re-ingest cannot resurrect a purged row because the producer no longer
materializes this shape. This module only classifies and counts; it never
mutates either tier.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

_AGENT_META_NATIVE_ID_PATTERN = "agent-%.meta"
_AGENT_META_SOURCE_PATH_PATTERN = "%.meta.json"


@dataclass(frozen=True, slots=True)
class AgentMetaSidecarCandidate:
    session_id: str
    origin: str
    native_id: str
    raw_id: str
    source_path: str
    blob_size: int
    native_id_matches_agent_meta_shape: bool
    """True when ``native_id`` also matches ``agent-<hash>.meta`` -- reported
    for confidence, not required for the match: the bead's own repro
    predicate is ``message_count=0 AND source_path LIKE '%.meta.json'``
    alone. Live verification found the two predicates agree on all 4,945
    live rows; this field lets the report flag it if a future row ever
    disagrees, rather than assuming silently."""


@dataclass(slots=True)
class AgentMetaSidecarSweepPlan:
    scanned_count: int = 0
    candidates: list[AgentMetaSidecarCandidate] = field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def candidate_bytes(self) -> int:
        return sum(c.blob_size for c in self.candidates)

    @property
    def shape_mismatch_count(self) -> int:
        """Candidates matching the source_path predicate but not the native_id
        shape -- always inspect these before relying on --apply; zero on the
        live archive as of 2026-08-02."""
        return sum(1 for c in self.candidates if not c.native_id_matches_agent_meta_shape)

    def by_origin(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            counts[candidate.origin] = counts.get(candidate.origin, 0) + 1
        return counts


def scan_agent_meta_sidecar_sessions(
    index_conn: sqlite3.Connection,
    source_db_path: Path,
    *,
    limit: int | None = None,
) -> AgentMetaSidecarSweepPlan:
    """Read-only: classify every zero-message session against the bead's repro predicate.

    *index_conn* should be a connection to ``index.db`` (``?mode=ro`` is the
    caller's responsibility, matching the sibling binary-artifact sweep).
    *source_db_path* is attached read-only for the duration of this call and
    detached before returning. This function never writes to either tier.
    """
    index_conn.row_factory = sqlite3.Row
    index_conn.execute("ATTACH DATABASE ? AS src", (f"file:{source_db_path}?mode=ro",))
    try:
        query = """
            SELECT
                s.session_id AS session_id,
                s.origin AS origin,
                s.native_id AS native_id,
                r.raw_id AS raw_id,
                r.source_path AS source_path,
                r.blob_size AS blob_size
            FROM sessions s
            JOIN src.raw_sessions r ON r.raw_id = s.raw_id
            WHERE s.message_count = 0
              AND r.source_path LIKE ?
            ORDER BY s.session_id
        """
        params: tuple[object, ...] = (_AGENT_META_SOURCE_PATH_PATTERN,)
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        rows = index_conn.execute(query, params).fetchall()

        plan = AgentMetaSidecarSweepPlan(scanned_count=len(rows))
        for row in rows:
            native_id = str(row["native_id"]) if row["native_id"] is not None else ""
            plan.candidates.append(
                AgentMetaSidecarCandidate(
                    session_id=row["session_id"],
                    origin=row["origin"],
                    native_id=native_id,
                    raw_id=row["raw_id"],
                    source_path=row["source_path"],
                    blob_size=int(row["blob_size"]) if row["blob_size"] is not None else 0,
                    native_id_matches_agent_meta_shape=_native_id_matches_agent_meta_shape(native_id),
                )
            )
        return plan
    finally:
        index_conn.execute("DETACH DATABASE src")


def _native_id_matches_agent_meta_shape(native_id: str) -> bool:
    if not native_id.startswith("agent-") or not native_id.endswith(".meta"):
        return False
    middle = native_id[len("agent-") : -len(".meta")]
    return len(middle) > 0


__all__ = [
    "AgentMetaSidecarCandidate",
    "AgentMetaSidecarSweepPlan",
    "scan_agent_meta_sidecar_sessions",
]

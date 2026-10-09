"""Scalar session-profile analytics on the caller's selected read snapshot."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from typing import Literal, cast

from polylogue.analysis.archive_rollups import ABANDONMENT_SEVERITY_RANK, iso_week_bucket_key
from polylogue.core.errors import DatabaseError
from polylogue.storage.sqlite.queries.mappers_support import _parse_json

ProfileAnalyticsMode = Literal["counts", "workflow", "abandoned"]

PROFILE_WALLCLOCK_SQL = (
    "(SELECT MAX(m.occurred_at_ms) - MIN(m.occurred_at_ms) "
    "FROM messages m WHERE m.session_id = s.session_id AND m.occurred_at_ms IS NOT NULL)"
)


def read_profile_analytics(
    conn: sqlite3.Connection,
    relation: str,
    params: Sequence[object],
    *,
    mode: ProfileAnalyticsMode,
    group_by: str,
    min_severity: str,
    limit: int,
    checkpoint: Callable[[], None],
) -> dict[str, object]:
    """Reduce native fields; hydrate only the selected abandoned evidence page."""
    checkpoint()
    shape = "COALESCE(NULLIF(sp.workflow_shape, ''), 'unknown')"
    state = "COALESCE(NULLIF(sp.terminal_state, ''), 'unknown')"
    if mode == "counts":
        dimensions = {"workflow_shape": shape, "terminal_state": state, "origin": "s.origin"}
        if group_by not in dimensions:
            raise ValueError(f"Unknown group_by: {group_by!r}. Supported: workflow_shape, terminal_state, origin.")
        buckets: dict[str, int] = {}
        with closing(
            conn.execute(f"SELECT {dimensions[group_by]} AS dimension, COUNT(*) {relation} GROUP BY dimension", params)
        ) as rows:
            for row in rows:
                checkpoint()
                buckets[str(row[0])] = int(row[1])
        return {"group_by": group_by, "total_sessions": sum(buckets.values()), "buckets": buckets}

    if mode == "workflow":
        if group_by not in {"week", "origin", "project"}:
            raise ValueError("group_by must be one of week, origin, project")
        buckets_by_shape: dict[str, dict[str, int]] = {}
        total = 0
        extra = (
            ", json_extract(sp.evidence_payload_json, '$.cwd_paths') AS cwd_paths_json" if group_by == "project" else ""
        )
        with closing(
            conn.execute(
                f"SELECT s.session_id, s.origin, {shape}, sp.canonical_session_date {extra} {relation}", params
            )
        ) as rows:
            for row in rows:
                checkpoint()
                total += 1
                keys: tuple[str, ...]
                if group_by == "origin":
                    keys = (str(row[1]),)
                elif group_by == "week":
                    keys = (iso_week_bucket_key(row[3]),)
                else:
                    paths = _parse_json(row[4], field="cwd_paths", record_id=str(row[0]))
                    if paths is not None and (
                        not isinstance(paths, list) or not all(isinstance(path, str) for path in paths)
                    ):
                        raise DatabaseError(f"Invalid stored cwd_paths for {row[0]}")
                    keys = tuple(cast(list[str], paths or [])) or ("unattributed",)
                for key in keys:
                    bucket = buckets_by_shape.setdefault(key, {})
                    bucket[str(row[2])] = bucket.get(str(row[2]), 0) + 1
        return {"group_by": group_by, "total_sessions": total, "buckets": buckets_by_shape}

    if mode != "abandoned":
        raise ValueError(f"profile analytics mode is not declared: {mode!r}")
    if min_severity not in ABANDONMENT_SEVERITY_RANK:
        raise ValueError("min_severity must be one of " + ", ".join(sorted(ABANDONMENT_SEVERITY_RANK)))
    minimum = ABANDONMENT_SEVERITY_RANK[min_severity]
    selected_params = tuple(params)
    if minimum:
        states = [name for name, rank in ABANDONMENT_SEVERITY_RANK.items() if rank >= minimum]
        clause = " AND " if " WHERE " in relation else " WHERE "
        relation += clause + f"{state} IN ({','.join('?' for _ in states)})"
        selected_params += tuple(states)
    with closing(conn.execute(f"SELECT COUNT(*) {relation}", selected_params)) as cursor:
        total = int(cursor.fetchone()[0])
    # Preserve the public Python slice semantics, including negative limits,
    # while binding a finite SQLite window and decoding only returned evidence.
    stop = slice(None, limit).indices(total)[1]
    items: list[dict[str, object]] = []
    with closing(
        conn.execute(
            f"SELECT s.session_id, s.origin, s.title, {state}, sp.terminal_state_confidence, "
            f"{shape}, sp.canonical_session_date, sp.terminal_state_evidence_json {relation} "
            "ORDER BY COALESCE(sp.canonical_session_date, '') DESC, s.sort_key_ms DESC, s.session_id LIMIT ?",
            (*selected_params, stop),
        )
    ) as rows:
        for row in rows:
            checkpoint()
            evidence = _parse_json(row[7], field="terminal_state_evidence_json", record_id=str(row[0]))
            if not isinstance(evidence, dict):
                raise DatabaseError(f"Invalid stored terminal_state_evidence for {row[0]}")
            items.append(
                {
                    "session_id": str(row[0]),
                    "origin": str(row[1]),
                    "title": row[2],
                    "terminal_state": str(row[3]),
                    "terminal_state_confidence": float(row[4] or 0.0),
                    "workflow_shape": str(row[5]),
                    "canonical_session_date": row[6],
                    "evidence": evidence,
                }
            )
    return {"total": total, "items": items}

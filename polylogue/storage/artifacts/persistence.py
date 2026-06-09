"""Persistence helpers for durable artifact observations.

Observations are materialized into ``raw_artifacts`` (source tier) by inspecting
the raw payloads recorded in ``raw_sessions``. The read model returns the
freshly-inspected in-memory records so callers see full inspection fidelity
(resolved schema package, wire format) even though ``raw_artifacts`` only
durably stores the subset of columns it declares (#1743).
"""

from __future__ import annotations

import sqlite3

from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.runtime import ArtifactObservationRecord
from polylogue.storage.sqlite.queries.artifacts import (
    RAW_ARTIFACT_UPSERT_SQL,
    artifact_observation_params,
)
from polylogue.storage.sqlite.queries.mappers import _row_to_raw_session


def _upsert_artifact_observation(
    conn: sqlite3.Connection,
    record: ArtifactObservationRecord,
) -> bool:
    exists = conn.execute(
        "SELECT 1 FROM raw_artifacts WHERE artifact_id = ?",
        (record.observation_id,),
    ).fetchone()
    conn.execute(RAW_ARTIFACT_UPSERT_SQL, artifact_observation_params(record))
    return exists is None


def materialize_artifact_observations(
    conn: sqlite3.Connection,
) -> list[ArtifactObservationRecord]:
    """Inspect every raw session, refresh ``raw_artifacts``, and return records.

    The returned records carry the full inspection result (including the
    resolved schema package and wire format that ``raw_artifacts`` does not
    store); the durable table is refreshed as a side effect.
    """
    records: list[ArtifactObservationRecord] = []
    last_rowid = 0
    while True:
        rows = conn.execute(
            """
            SELECT r.rowid AS raw_rowid, r.*
            FROM raw_sessions r
            WHERE r.rowid > ?
            ORDER BY r.rowid
            LIMIT 250
            """,
            (last_rowid,),
        ).fetchall()
        if not rows:
            break
        for row in rows:
            last_rowid = max(last_rowid, int(row["raw_rowid"]))
            raw_record = _row_to_raw_session(row)
            observation = inspect_raw_artifact(raw_record)
            _upsert_artifact_observation(conn, observation)
            records.append(observation)
        conn.commit()
    return records


def ensure_artifact_observations(
    conn: sqlite3.Connection,
    *,
    providers: list[str] | None = None,
    refresh_resolutions: bool = False,
) -> int:
    """Refresh durable artifact observations from raw records; return the count."""
    del providers, refresh_resolutions
    return len(materialize_artifact_observations(conn))


__all__ = ["ensure_artifact_observations", "materialize_artifact_observations"]

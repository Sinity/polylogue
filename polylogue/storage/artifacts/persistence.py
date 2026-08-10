"""Persistence helpers for durable artifact observations.

Observations are materialized into ``raw_artifacts`` (source tier) by inspecting
the raw payloads recorded in ``raw_sessions``. The read model returns the
freshly-inspected in-memory records so callers see full inspection fidelity
(resolved schema package, wire format) even though ``raw_artifacts`` only
durably stores the subset of columns it declares (#1743).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

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
    cursor = conn.execute(RAW_ARTIFACT_UPSERT_SQL, artifact_observation_params(record))
    return cursor.rowcount == 1


def upsert_artifact_observations(
    conn: sqlite3.Connection,
    records: Sequence[ArtifactObservationRecord],
) -> int:
    """Upsert a caller-selected observation batch without committing.

    The raw-authority census uses this narrow primitive so its apply path can
    hold one source-tier transaction and prove that no raw authority or blob
    relation was touched. Callers own transaction boundaries.
    """
    return sum(_upsert_artifact_observation(conn, record) for record in records)


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


def materialize_artifact_observations_for_raw_ids(
    conn: sqlite3.Connection,
    raw_ids: Sequence[str],
) -> list[ArtifactObservationRecord]:
    """Inspect and persist ``raw_artifacts`` rows for exactly the given raw ids.

    Scoped sibling of :func:`materialize_artifact_observations` for callers
    that already know precisely which raw rows they care about (e.g. a
    targeted phantom-session sweep) and want to avoid paying for a full
    ``raw_sessions`` census just to refresh a handful of rows. Unknown raw
    ids are silently skipped (no row to inspect); duplicates are de-duplicated.
    """
    unique_ids = list(dict.fromkeys(raw_ids))
    if not unique_ids:
        return []
    conn.row_factory = sqlite3.Row
    records: list[ArtifactObservationRecord] = []
    for batch_start in range(0, len(unique_ids), 250):
        batch = unique_ids[batch_start : batch_start + 250]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT rowid AS raw_rowid, * FROM raw_sessions WHERE raw_id IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows:
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


__all__ = [
    "ensure_artifact_observations",
    "materialize_artifact_observations",
    "materialize_artifact_observations_for_raw_ids",
    "upsert_artifact_observations",
]

"""Durable artifact observation writes for the async SQLite backend.

Artifact classifications persist in the ``raw_artifacts`` source-tier table
(#1743). Only the columns that table declares are durable; richer inspection
facts the model still carries (``wire_format``, ``bundle_scope``, resolved
schema-package fields, ``file_mtime``) are recomputed on read from the raw
payload and are intentionally not stored.
"""

from __future__ import annotations

from datetime import datetime

import aiosqlite

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.runtime import ArtifactObservationRecord

__all__ = [
    "RAW_ARTIFACT_UPSERT_SQL",
    "artifact_observation_params",
    "save_artifact_observation",
]


RAW_ARTIFACT_UPSERT_SQL = """
INSERT INTO raw_artifacts (
    artifact_id,
    raw_id,
    origin,
    source_path,
    source_index,
    artifact_kind,
    support_status,
    classification_reason,
    parse_as_session,
    schema_eligible,
    malformed_jsonl_lines,
    decode_error,
    cohort_id,
    link_group_key,
    sidecar_agent_type,
    first_observed_at_ms,
    last_observed_at_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(artifact_id) DO UPDATE SET
    raw_id = excluded.raw_id,
    origin = excluded.origin,
    source_path = excluded.source_path,
    source_index = excluded.source_index,
    artifact_kind = excluded.artifact_kind,
    support_status = excluded.support_status,
    classification_reason = excluded.classification_reason,
    parse_as_session = excluded.parse_as_session,
    schema_eligible = excluded.schema_eligible,
    malformed_jsonl_lines = excluded.malformed_jsonl_lines,
    decode_error = excluded.decode_error,
    cohort_id = excluded.cohort_id,
    link_group_key = excluded.link_group_key,
    sidecar_agent_type = excluded.sidecar_agent_type,
    last_observed_at_ms = excluded.last_observed_at_ms
"""


def _iso_to_ms(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return int(datetime.fromisoformat(value).timestamp() * 1000)


def artifact_observation_params(record: ArtifactObservationRecord) -> tuple[object, ...]:
    """Project an artifact observation onto the ``raw_artifacts`` column order."""
    provider = record.payload_provider or Provider.from_string(record.source_name or "")
    origin_value = origin_from_provider(provider).value
    return (
        record.observation_id,
        record.raw_id,
        origin_value,
        record.source_path,
        record.source_index if record.source_index is not None else 0,
        record.artifact_kind,
        record.support_status.value,
        record.classification_reason,
        int(record.parse_as_session),
        int(record.schema_eligible),
        record.malformed_jsonl_lines,
        record.decode_error,
        record.cohort_id,
        record.link_group_key,
        record.sidecar_agent_type,
        _iso_to_ms(record.first_observed_at),
        _iso_to_ms(record.last_observed_at),
    )


async def save_artifact_observation(
    conn: aiosqlite.Connection,
    record: ArtifactObservationRecord,
    transaction_depth: int,
) -> bool:
    """Persist or refresh one artifact observation row in ``raw_artifacts``."""
    exists_cursor = await conn.execute(
        "SELECT 1 FROM raw_artifacts WHERE artifact_id = ?",
        (record.observation_id,),
    )
    existed = await exists_cursor.fetchone() is not None

    await conn.execute(RAW_ARTIFACT_UPSERT_SQL, artifact_observation_params(record))
    if transaction_depth == 0:
        await conn.commit()
    return not existed

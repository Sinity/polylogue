"""Durable artifact observation writes for the async SQLite backend.

Artifact classifications persist in the ``raw_artifacts`` source-tier table
(#1743). Only the columns that table declares are durable; richer inspection
facts the model still carries (``wire_format``, ``bundle_scope``, resolved
schema-package fields, ``file_mtime``) are recomputed on read from the raw
payload and are intentionally not stored.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import aiosqlite

from polylogue.core.enums import ArtifactSupportStatus, Provider
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_EVIDENCE_KINDS,
    validated_raw_failure_evidence_kind,
)
from polylogue.core.sources import origin_from_provider
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.runtime import ArtifactObservationRecord

__all__ = [
    "RAW_ARTIFACT_UPSERT_SQL",
    "artifact_observation_params",
    "save_artifact_observation",
    "save_raw_failure_evidence",
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
    if record.artifact_kind == "hook_event":
        await _materialize_hook_events(conn, record)
    if transaction_depth == 0:
        await conn.commit()
    return not existed


async def save_raw_failure_evidence(
    conn: aiosqlite.Connection,
    raw_id: str,
    *,
    artifact_kind: str,
    support_status: ArtifactSupportStatus | str,
    outcome_code: str,
    retryable: bool | None,
    evidence_ref: str | None,
    remediation: str | None,
    diagnostic: str | None,
    transaction_depth: int,
) -> None:
    """Persist one typed worker disposition beside its retained raw bytes.

    The raw row supplies the origin and exact source coordinate inside the
    caller's source-tier transaction.  Failure disposition details live in
    the existing structured ``classification_reason`` carrier, while the
    diagnostic remains available through ``decode_error``.
    """
    status = ArtifactSupportStatus.from_string(str(support_status))
    evidence_kind = validated_raw_failure_evidence_kind(
        artifact_kind,
        status,
        validation_failed=False,
    )
    if evidence_kind is None:
        raise ValueError(f"invalid closed raw-failure evidence pair: {artifact_kind!r}/{status.value!r}")

    raw_cursor = await conn.execute(
        """
        SELECT origin, source_path, source_index, acquired_at_ms
        FROM raw_sessions
        WHERE raw_id = ?
        """,
        (raw_id,),
    )
    raw_row = await raw_cursor.fetchone()
    if raw_row is None:
        raise KeyError(raw_id)
    origin, source_path, source_index, acquired_at_ms = raw_row

    failure_kind_placeholders = ", ".join("?" for _ in RAW_FAILURE_EVIDENCE_KINDS)
    existing_cursor = await conn.execute(
        f"""
        SELECT artifact_id
        FROM raw_artifacts
        WHERE raw_id = ?
          AND origin = ?
          AND source_path = ?
          AND source_index = ?
          AND artifact_kind IN ({failure_kind_placeholders})
        LIMIT 1
        """,
        (raw_id, origin, source_path, source_index, *sorted(RAW_FAILURE_EVIDENCE_KINDS)),
    )
    existing_row = await existing_cursor.fetchone()
    artifact_id = (
        str(existing_row[0])
        if existing_row is not None
        else "raw-failure:" + hashlib.sha256(f"{raw_id}:{origin}:{source_path}:{source_index}".encode()).hexdigest()
    )
    classification_reason = json.dumps(
        {
            "diagnostic": diagnostic,
            "evidence_ref": evidence_ref,
            "outcome_code": outcome_code,
            "remediation": remediation,
            "retryable": retryable,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    await conn.execute(
        RAW_ARTIFACT_UPSERT_SQL,
        (
            artifact_id,
            raw_id,
            origin,
            source_path,
            source_index,
            evidence_kind.value,
            evidence_kind.support_status.value,
            classification_reason,
            int(evidence_kind.lifecycle == "deferred"),
            int(evidence_kind.lifecycle == "deferred"),
            0,
            diagnostic,
            None,
            None,
            None,
            acquired_at_ms,
            acquired_at_ms,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()


def _hook_observed_at_ms(value: object, fallback: str) -> int:
    candidate = value if isinstance(value, str) else fallback
    try:
        return _iso_to_ms(candidate)
    except (TypeError, ValueError):
        return _iso_to_ms(fallback)


def _hook_records_from_blob(blob_hash: str) -> list[dict[str, object]]:
    path = get_blob_store().blob_path(blob_hash)
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if not all(isinstance(candidate.get(key), str) for key in ("event_type", "session_id", "provider")):
                    continue
                records.append(candidate)
    return records


async def _materialize_hook_events(
    conn: aiosqlite.Connection,
    record: ArtifactObservationRecord,
) -> None:
    """Project classified hook JSONL into the existing source-tier relation."""

    cursor = await conn.execute("SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (record.raw_id,))
    row = await cursor.fetchone()
    if row is None or not row[0]:
        return
    blob_hash = str(row[0]).lower()
    try:
        hook_records = _hook_records_from_blob(blob_hash)
    except (OSError, ValueError, UnicodeDecodeError):
        return
    for index, hook_record in enumerate(hook_records):
        provider = str(hook_record["provider"])
        if provider not in {"claude-code", "codex"}:
            continue
        origin = origin_from_provider(Provider.from_string(provider)).value
        event_type = str(hook_record["event_type"])
        session_native_id = str(hook_record["session_id"])
        digest_input = json.dumps(
            hook_record,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        hook_event_id = (
            "hook:"
            + hashlib.sha256(
                record.source_path.encode("utf-8") + b"\0" + str(index).encode("ascii") + b"\0" + digest_input
            ).hexdigest()
        )
        await conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id, source_path,
                event_type, payload_json, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hook_event_id) DO UPDATE SET
                origin = excluded.origin,
                native_id = excluded.native_id,
                session_native_id = excluded.session_native_id,
                source_path = excluded.source_path,
                event_type = excluded.event_type,
                payload_json = excluded.payload_json,
                observed_at_ms = excluded.observed_at_ms
            """,
            (
                hook_event_id,
                origin,
                f"{session_native_id}:{event_type}:{index}",
                session_native_id,
                record.source_path,
                event_type,
                json.dumps(hook_record, ensure_ascii=False, default=str),
                _hook_observed_at_ms(hook_record.get("timestamp"), record.last_observed_at),
            ),
        )

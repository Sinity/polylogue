"""Raw-capture writer/read helpers for source.db."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from polylogue.core.enums import ArtifactSupportStatus, Origin, ValidationMode, ValidationStatus


@dataclass(frozen=True, slots=True)
class ArchiveSourceBlobRef:
    """A compact raw blob reference row."""

    blob_hash: bytes
    raw_id: str | None = None
    ref_type: str = "raw_payload"
    source_path: str | None = None
    size_bytes: int | None = None
    acquired_at_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ArchiveSourceArtifact:
    """A compact raw-artifact row."""

    artifact_id: str
    origin: Origin | str
    source_path: str
    artifact_kind: str
    classification_reason: str
    support_status: ArtifactSupportStatus | str = ArtifactSupportStatus.UNKNOWN
    parse_as_session: bool = False
    schema_eligible: bool = False
    first_observed_at_ms: int = 0
    last_observed_at_ms: int = 0
    source_index: int = 0
    malformed_jsonl_lines: int = 0
    decode_error: str | None = None
    cohort_id: str | None = None
    link_group_key: str | None = None
    sidecar_agent_type: str | None = None
    native_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveRawArtifactEnvelope:
    """Read-back view of one raw artifact classification row."""

    artifact_id: str
    raw_id: str
    origin: str
    source_path: str
    source_index: int
    artifact_kind: str
    support_status: str
    classification_reason: str
    parse_as_session: bool
    schema_eligible: bool
    malformed_jsonl_lines: int
    decode_error: str | None
    cohort_id: str | None
    link_group_key: str | None
    sidecar_agent_type: str | None
    first_observed_at_ms: int
    last_observed_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveHookEvent:
    """A compact hook-event row."""

    hook_event_id: str
    origin: Origin | str
    source_path: str
    event_type: str
    payload: dict[str, object]
    observed_at_ms: int
    native_id: str | None = None
    session_native_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveHistorySidecar:
    """A persisted history.jsonl-style sidecar payload."""

    sidecar_id: str
    origin: str
    source_path: str
    payload: dict[str, object]
    observed_at_ms: int
    content_hash: bytes


@dataclass(frozen=True, slots=True)
class ArchiveRawSessionEnvelope:
    """Compact read-back view of one raw-session row."""

    raw_id: str
    origin: str
    native_id: str | None
    source_path: str
    source_index: int
    blob_hash: bytes
    blob_size: int
    acquired_at_ms: int
    file_mtime_ms: int | None
    parsed_at_ms: int | None
    parse_error: str | None
    validated_at_ms: int | None
    validation_status: str | None
    validation_error: str | None
    validation_drift_count: int
    validation_mode: str | None
    detection_warnings: tuple[str, ...]
    blob_refs: tuple[ArchiveSourceBlobRef, ...]
    artifact_ids: tuple[str, ...]
    hook_event_ids: tuple[str, ...]
    history_sidecar_ids: tuple[str, ...]


def deterministic_blob_hash(payload: bytes) -> bytes:
    """Deterministic SHA-256 hash for a raw payload."""
    return hashlib.sha256(payload).digest()


def deterministic_raw_session_id(
    origin: Origin | str,
    source_path: str,
    source_index: int,
    blob_hash: bytes,
    native_id: str | None = None,
) -> str:
    """Deterministic text identifier for an archive raw session."""
    origin_value = _enum_value(origin)
    if origin_value is None:
        raise ValueError("origin is required for raw session ids")
    digest = hashlib.sha256()
    digest.update(origin_value.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(source_path.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(str(source_index).encode("utf-8"))
    digest.update(b"\0")
    digest.update(blob_hash)
    digest.update(b"\0")
    digest.update((native_id or "").encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def deterministic_history_sidecar_id(origin: Origin | str, source_path: str, content_hash: bytes) -> str:
    """Deterministic text identifier for one history sidecar observation."""
    origin_value = _enum_value(origin)
    if origin_value is None:
        raise ValueError("origin is required for history sidecar ids")
    digest = hashlib.sha256()
    digest.update(origin_value.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(source_path.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(content_hash)
    return digest.hexdigest()


def write_history_sidecar(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    source_path: str,
    payload: dict[str, object],
    observed_at_ms: int,
    sidecar_id: str | None = None,
) -> str:
    """Persist a history sidecar payload in the archive raw tier."""
    origin_value = _enum_value(origin)
    payload_json = _json_dumps(payload)
    content_hash = deterministic_blob_hash(payload_json.encode("utf-8", errors="surrogatepass"))
    resolved_sidecar_id = sidecar_id or deterministic_history_sidecar_id(origin, source_path, content_hash)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO history_sidecars (
                sidecar_id, origin, source_path, payload_json, observed_at_ms, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (resolved_sidecar_id, origin_value, source_path, payload_json, observed_at_ms, content_hash),
        )
    return resolved_sidecar_id


def write_source_raw_session(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    source_path: str,
    source_index: int,
    payload: bytes,
    acquired_at_ms: int,
    native_id: str | None = None,
    raw_id: str | None = None,
    file_mtime_ms: int | None = None,
    parsed_at_ms: int | None = None,
    parse_error: str | None = None,
    validated_at_ms: int | None = None,
    validation_status: ValidationStatus | str | None = None,
    validation_error: str | None = None,
    validation_drift_count: int = 0,
    validation_mode: ValidationMode | str | None = None,
    detection_warnings: tuple[str, ...] = (),
    additional_blob_refs: tuple[ArchiveSourceBlobRef, ...] = (),
    artifact: ArchiveSourceArtifact | None = None,
    hook_event: ArchiveHookEvent | None = None,
) -> str:
    """Insert one raw session and its required raw-payload blob reference."""
    conn.execute("PRAGMA foreign_keys = ON")
    origin_value = _enum_value(origin)
    blob_hash = deterministic_blob_hash(payload)
    blob_size = len(payload)
    resolved_raw_id = raw_id or deterministic_raw_session_id(
        origin,
        source_path,
        source_index,
        blob_hash,
        native_id,
    )

    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error,
                validated_at_ms, validation_status, validation_error, validation_drift_count,
                validation_mode, detection_warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_raw_id,
                origin_value,
                native_id,
                source_path,
                source_index,
                blob_hash,
                blob_size,
                acquired_at_ms,
                file_mtime_ms,
                parsed_at_ms,
                parse_error,
                validated_at_ms,
                _enum_value(validation_status),
                validation_error,
                validation_drift_count,
                _enum_value(validation_mode),
                _json_dumps(detection_warnings),
            ),
        )
        _insert_blob_ref(
            conn,
            ArchiveSourceBlobRef(
                blob_hash=blob_hash,
                raw_id=resolved_raw_id,
                ref_type="raw_payload",
                source_path=source_path,
                size_bytes=blob_size,
                acquired_at_ms=acquired_at_ms,
            ),
        )
        for blob_ref in additional_blob_refs:
            _insert_blob_ref(
                conn,
                ArchiveSourceBlobRef(
                    blob_hash=blob_ref.blob_hash,
                    raw_id=resolved_raw_id,
                    ref_type=blob_ref.ref_type,
                    source_path=blob_ref.source_path,
                    size_bytes=blob_ref.size_bytes or 0,
                    acquired_at_ms=blob_ref.acquired_at_ms or acquired_at_ms,
                ),
            )
        if artifact is not None:
            _insert_artifact(conn, resolved_raw_id, artifact)
        if hook_event is not None:
            _insert_hook_event(conn, resolved_raw_id, hook_event)

        return resolved_raw_id


def write_source_raw_session_blob_ref(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    source_path: str,
    source_index: int,
    blob_hash: bytes,
    blob_size: int,
    acquired_at_ms: int,
    native_id: str | None = None,
    raw_id: str | None = None,
) -> str:
    """Insert one raw session that already has a materialized raw-payload blob."""
    if len(blob_hash) != 32:
        raise ValueError("blob_hash must be a 32-byte SHA-256 digest")
    conn.execute("PRAGMA foreign_keys = ON")
    origin_value = _enum_value(origin)
    resolved_raw_id = raw_id or deterministic_raw_session_id(
        origin,
        source_path,
        source_index,
        blob_hash,
        native_id,
    )
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_raw_id,
                origin_value,
                native_id,
                source_path,
                source_index,
                blob_hash,
                blob_size,
                acquired_at_ms,
            ),
        )
        _insert_blob_ref(
            conn,
            ArchiveSourceBlobRef(
                blob_hash=blob_hash,
                raw_id=resolved_raw_id,
                ref_type="raw_payload",
                source_path=source_path,
                size_bytes=blob_size,
                acquired_at_ms=acquired_at_ms,
            ),
        )
    return resolved_raw_id


def read_archive_raw_session_envelope(conn: sqlite3.Connection, raw_id: str) -> ArchiveRawSessionEnvelope:
    """Read a compact envelope for one raw source session."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT
            raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
            acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error, validated_at_ms,
            validation_status, validation_error, validation_drift_count, validation_mode,
            detection_warnings_json
        FROM raw_sessions
        WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    if row is None:
        raise KeyError(raw_id)

    blob_refs = tuple(
        ArchiveSourceBlobRef(
            blob_hash=row["blob_hash"],
            raw_id=row["raw_id"],
            ref_type=row["ref_type"],
            source_path=row["source_path"],
            size_bytes=row["size_bytes"],
            acquired_at_ms=row["acquired_at_ms"],
        )
        for row in conn.execute(
            """
            SELECT blob_hash, ref_id AS raw_id, ref_type, source_path, size_bytes, acquired_at_ms
            FROM blob_refs
            WHERE ref_id = ?
            ORDER BY ref_type, source_path
            """,
            (raw_id,),
        ).fetchall()
    )
    artifact_ids = tuple(
        row["artifact_id"]
        for row in conn.execute(
            """
            SELECT artifact_id FROM raw_artifacts WHERE raw_id = ? ORDER BY artifact_id
            """,
            (raw_id,),
        ).fetchall()
    )
    hook_event_ids = tuple(
        row["hook_event_id"]
        for row in conn.execute(
            """
            SELECT hook_event_id
            FROM raw_hook_events
            WHERE origin = ? AND session_native_id = ? AND source_path = ?
            ORDER BY observed_at_ms
            """,
            (row["origin"], row["native_id"], row["source_path"]),
        ).fetchall()
    )
    history_sidecar_ids = tuple(
        row["sidecar_id"]
        for row in conn.execute(
            """
            SELECT sidecar_id
            FROM history_sidecars
            WHERE origin = ? AND source_path = ?
            ORDER BY observed_at_ms, sidecar_id
            """,
            (row["origin"], row["source_path"]),
        ).fetchall()
    )

    return ArchiveRawSessionEnvelope(
        raw_id=row["raw_id"],
        origin=row["origin"],
        native_id=row["native_id"],
        source_path=row["source_path"],
        source_index=row["source_index"],
        blob_hash=row["blob_hash"],
        blob_size=row["blob_size"],
        acquired_at_ms=row["acquired_at_ms"],
        file_mtime_ms=row["file_mtime_ms"],
        parsed_at_ms=row["parsed_at_ms"],
        parse_error=row["parse_error"],
        validated_at_ms=row["validated_at_ms"],
        validation_status=row["validation_status"],
        validation_error=row["validation_error"],
        validation_drift_count=row["validation_drift_count"],
        validation_mode=row["validation_mode"],
        detection_warnings=tuple(json.loads(row["detection_warnings_json"] or "[]")),
        blob_refs=blob_refs,
        artifact_ids=artifact_ids,
        hook_event_ids=hook_event_ids,
        history_sidecar_ids=history_sidecar_ids,
    )


def read_history_sidecar(conn: sqlite3.Connection, sidecar_id: str) -> ArchiveHistorySidecar:
    """Read one persisted history sidecar payload."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT sidecar_id, origin, source_path, payload_json, observed_at_ms, content_hash
        FROM history_sidecars
        WHERE sidecar_id = ?
        """,
        (sidecar_id,),
    ).fetchone()
    if row is None:
        raise KeyError(sidecar_id)
    return ArchiveHistorySidecar(
        sidecar_id=row["sidecar_id"],
        origin=row["origin"],
        source_path=row["source_path"],
        payload=_json_loads(row["payload_json"]),
        observed_at_ms=row["observed_at_ms"],
        content_hash=row["content_hash"],
    )


def read_raw_artifact(conn: sqlite3.Connection, artifact_id: str) -> ArchiveRawArtifactEnvelope:
    """Read one raw artifact classification row."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT
            artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
            support_status, classification_reason, parse_as_session, schema_eligible,
            malformed_jsonl_lines, decode_error, cohort_id, link_group_key, sidecar_agent_type,
            first_observed_at_ms, last_observed_at_ms
        FROM raw_artifacts
        WHERE artifact_id = ?
        """,
        (artifact_id,),
    ).fetchone()
    if row is None:
        raise KeyError(artifact_id)
    return _raw_artifact_from_row(row)


def list_raw_artifacts(
    conn: sqlite3.Connection,
    *,
    raw_id: str | None = None,
    origin: Origin | str | None = None,
) -> tuple[ArchiveRawArtifactEnvelope, ...]:
    """Return raw artifact rows ordered by source identity."""
    conn.row_factory = sqlite3.Row
    query = """
        SELECT
            artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
            support_status, classification_reason, parse_as_session, schema_eligible,
            malformed_jsonl_lines, decode_error, cohort_id, link_group_key, sidecar_agent_type,
            first_observed_at_ms, last_observed_at_ms
        FROM raw_artifacts
    """
    clauses: list[str] = []
    params: list[object] = []
    if raw_id is not None:
        clauses.append("raw_id = ?")
        params.append(raw_id)
    if origin is not None:
        clauses.append("origin = ?")
        params.append(_enum_value(origin))
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY source_path, source_index, artifact_id"
    return tuple(_raw_artifact_from_row(row) for row in conn.execute(query, tuple(params)).fetchall())


def read_hook_event(conn: sqlite3.Connection, hook_event_id: str) -> ArchiveHookEvent:
    """Read one raw hook event row."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT hook_event_id, origin, source_path, event_type, payload_json, observed_at_ms,
            native_id, session_native_id
        FROM raw_hook_events
        WHERE hook_event_id = ?
        """,
        (hook_event_id,),
    ).fetchone()
    if row is None:
        raise KeyError(hook_event_id)
    return _hook_event_from_row(row)


def list_hook_events(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str | None = None,
    session_native_id: str | None = None,
) -> tuple[ArchiveHookEvent, ...]:
    """Return raw hook event rows ordered by observation time."""
    conn.row_factory = sqlite3.Row
    query = """
        SELECT hook_event_id, origin, source_path, event_type, payload_json, observed_at_ms,
            native_id, session_native_id
        FROM raw_hook_events
    """
    clauses: list[str] = []
    params: list[object] = []
    if origin is not None:
        clauses.append("origin = ?")
        params.append(_enum_value(origin))
    if session_native_id is not None:
        clauses.append("session_native_id = ?")
        params.append(session_native_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY observed_at_ms, hook_event_id"
    return tuple(_hook_event_from_row(row) for row in conn.execute(query, tuple(params)).fetchall())


def _insert_blob_ref(conn: sqlite3.Connection, ref: ArchiveSourceBlobRef) -> None:
    if ref.raw_id is None or ref.size_bytes is None or ref.acquired_at_ms is None:
        raise ValueError("raw_id, size_bytes, and acquired_at_ms are required for blob refs")
    conn.execute(
        """
        INSERT OR REPLACE INTO blob_refs (
            blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ref.blob_hash,
            ref.raw_id,
            ref.ref_type,
            ref.source_path,
            ref.size_bytes,
            ref.acquired_at_ms,
        ),
    )


def _insert_artifact(conn: sqlite3.Connection, raw_id: str, artifact: ArchiveSourceArtifact) -> None:
    conn.execute(
        """
        INSERT INTO raw_artifacts (
            artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
            support_status, classification_reason, parse_as_session, schema_eligible,
            malformed_jsonl_lines, decode_error, cohort_id, link_group_key, sidecar_agent_type,
            first_observed_at_ms, last_observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact.artifact_id,
            raw_id,
            _enum_value(artifact.origin),
            artifact.source_path,
            artifact.source_index,
            artifact.artifact_kind,
            _enum_value(artifact.support_status),
            artifact.classification_reason,
            int(artifact.parse_as_session),
            int(artifact.schema_eligible),
            artifact.malformed_jsonl_lines,
            artifact.decode_error,
            artifact.cohort_id,
            artifact.link_group_key,
            artifact.sidecar_agent_type,
            artifact.first_observed_at_ms,
            artifact.last_observed_at_ms,
        ),
    )


def _insert_hook_event(conn: sqlite3.Connection, raw_id: str, hook_event: ArchiveHookEvent) -> None:
    del raw_id
    conn.execute(
        """
        INSERT INTO raw_hook_events (
            hook_event_id, origin, native_id, session_native_id, source_path, event_type,
            payload_json, observed_at_ms
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
            hook_event.hook_event_id,
            _enum_value(hook_event.origin),
            hook_event.native_id,
            hook_event.session_native_id,
            hook_event.source_path,
            hook_event.event_type,
            _json_dumps(hook_event.payload),
            hook_event.observed_at_ms,
        ),
    )


def _raw_artifact_from_row(row: sqlite3.Row) -> ArchiveRawArtifactEnvelope:
    return ArchiveRawArtifactEnvelope(
        artifact_id=row["artifact_id"],
        raw_id=row["raw_id"],
        origin=row["origin"],
        source_path=row["source_path"],
        source_index=row["source_index"],
        artifact_kind=row["artifact_kind"],
        support_status=row["support_status"],
        classification_reason=row["classification_reason"],
        parse_as_session=bool(row["parse_as_session"]),
        schema_eligible=bool(row["schema_eligible"]),
        malformed_jsonl_lines=row["malformed_jsonl_lines"],
        decode_error=row["decode_error"],
        cohort_id=row["cohort_id"],
        link_group_key=row["link_group_key"],
        sidecar_agent_type=row["sidecar_agent_type"],
        first_observed_at_ms=row["first_observed_at_ms"],
        last_observed_at_ms=row["last_observed_at_ms"],
    )


def _hook_event_from_row(row: sqlite3.Row) -> ArchiveHookEvent:
    return ArchiveHookEvent(
        hook_event_id=row["hook_event_id"],
        origin=row["origin"],
        source_path=row["source_path"],
        event_type=row["event_type"],
        payload=_json_loads(row["payload_json"]),
        observed_at_ms=row["observed_at_ms"],
        native_id=row["native_id"],
        session_native_id=row["session_native_id"],
    )


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_loads(raw_json: str | bytes) -> dict[str, object]:
    if isinstance(raw_json, bytes):
        raw_json = raw_json.decode("utf-8")
    loaded = json.loads(raw_json or "{}")
    return loaded if isinstance(loaded, dict) else {}


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


__all__ = [
    "ArchiveHistorySidecar",
    "ArchiveHookEvent",
    "ArchiveRawArtifactEnvelope",
    "ArchiveRawSessionEnvelope",
    "ArchiveSourceArtifact",
    "ArchiveSourceBlobRef",
    "deterministic_blob_hash",
    "deterministic_history_sidecar_id",
    "deterministic_raw_session_id",
    "list_hook_events",
    "list_raw_artifacts",
    "read_history_sidecar",
    "read_hook_event",
    "read_raw_artifact",
    "read_archive_raw_session_envelope",
    "write_history_sidecar",
    "write_source_raw_session",
    "write_source_raw_session_blob_ref",
]

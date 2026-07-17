"""Durable source-tier publication outbox primitives.

All synchronous functions accept an already-open ``sqlite3.Connection`` and
never commit or roll back.  ``stage_payload_async`` follows the same rule for
the async source backend used by ingest.  This lets raw-session acceptance,
exact material bytes, and the idempotent obligation share one source.db
transaction without pretending a WAL transaction spans source.db/index.db.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Protocol, cast, runtime_checkable

from polylogue.sinex.models import (
    ObligationStatus,
    PublicationMode,
    PublicationObligation,
    PublicationPayload,
    PublicationReceipt,
    ReceiptState,
)

_COLUMNS = (
    "object_id",
    "protocol_version",
    "revision_id",
    "manifest_digest",
    "mode",
    "status",
    "attempt_count",
    "last_attempt_at_ms",
    "last_receipt_state",
    "last_error",
    "created_at_ms",
    "updated_at_ms",
    "retired_at_ms",
    "next_attempt_at_ms",
)


class PublicationPayloadConflictError(RuntimeError):
    """The same publication key was presented with different exact bytes."""


class PublicationPayloadInvalidError(ValueError):
    """A staged payload is malformed or does not match its declared digest."""


@runtime_checkable
class _AsyncCursor(Protocol):
    async def fetchone(self) -> object | None: ...

    async def fetchall(self) -> list[object]: ...


@runtime_checkable
class AsyncSqlConnection(Protocol):
    async def execute(self, sql: str, parameters: Sequence[object] = ()) -> _AsyncCursor: ...


Key = tuple[str, str, str, str]


def _key(obligation: PublicationObligation | PublicationPayload) -> Key:
    return (
        obligation.object_id,
        obligation.protocol_version,
        obligation.revision_id,
        obligation.manifest_digest,
    )


def _validate_payload(payload: PublicationPayload) -> None:
    if not payload.object_id or not payload.protocol_version or not payload.revision_id:
        raise PublicationPayloadInvalidError("publication identity fields must be non-empty")
    actual_manifest_digest = hashlib.sha256(payload.manifest_bytes).hexdigest()
    if actual_manifest_digest != payload.manifest_digest:
        raise PublicationPayloadInvalidError("manifest_digest does not match the exact staged manifest bytes")
    names: set[str] = set()
    for name, _segment in payload.segments:
        path = PurePosixPath(name)
        if not name or "\x00" in name or path.is_absolute() or ".." in path.parts:
            raise PublicationPayloadInvalidError(f"unsafe protocol segment name: {name!r}")
        if name in names:
            raise PublicationPayloadInvalidError(f"duplicate protocol segment name: {name!r}")
        names.add(name)


def _row_to_obligation(row: sqlite3.Row) -> PublicationObligation:
    return PublicationObligation(
        object_id=str(row["object_id"]),
        protocol_version=str(row["protocol_version"]),
        revision_id=str(row["revision_id"]),
        manifest_digest=str(row["manifest_digest"]),
        mode=PublicationMode.from_string(str(row["mode"])),
        status=ObligationStatus.from_string(str(row["status"])),
        attempt_count=int(row["attempt_count"]),
        last_attempt_at_ms=(int(row["last_attempt_at_ms"]) if row["last_attempt_at_ms"] is not None else None),
        last_receipt_state=(
            ReceiptState(str(row["last_receipt_state"])) if row["last_receipt_state"] is not None else None
        ),
        last_error=(str(row["last_error"]) if row["last_error"] is not None else None),
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
        retired_at_ms=(int(row["retired_at_ms"]) if row["retired_at_ms"] is not None else None),
        next_attempt_at_ms=(int(row["next_attempt_at_ms"]) if row["next_attempt_at_ms"] is not None else None),
    )


def record_obligation(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    protocol_version: str,
    revision_id: str,
    manifest_digest: str,
    mode: PublicationMode,
    now_ms: int,
) -> PublicationObligation:
    """Idempotently create an obligation and monotonically elevate its mode."""
    if mode is PublicationMode.OFF:
        raise ValueError("record_obligation must not be called in off mode")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT INTO sinex_publication_obligations (
            object_id, protocol_version, revision_id, manifest_digest, mode,
            status, attempt_count, created_at_ms, updated_at_ms,
            next_attempt_at_ms
        ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
        ON CONFLICT(object_id, protocol_version, revision_id, manifest_digest)
        DO UPDATE SET
            mode = CASE
                WHEN sinex_publication_obligations.mode = 'mirror'
                 AND excluded.mode = 'primary' THEN 'primary'
                ELSE sinex_publication_obligations.mode
            END,
            updated_at_ms = CASE
                WHEN sinex_publication_obligations.mode = 'mirror'
                 AND excluded.mode = 'primary' THEN excluded.updated_at_ms
                ELSE sinex_publication_obligations.updated_at_ms
            END
        """,
        (object_id, protocol_version, revision_id, manifest_digest, mode.value, now_ms, now_ms, now_ms),
    )
    existing = get_obligation(
        conn,
        object_id=object_id,
        protocol_version=protocol_version,
        revision_id=revision_id,
        manifest_digest=manifest_digest,
    )
    assert existing is not None
    return existing


def _assert_existing_payload_matches(
    *,
    manifest_bytes: bytes,
    manifest_sha256: str,
    manifest_size_bytes: int,
    segment_count: int,
    total_size_bytes: int,
    payload: PublicationPayload,
) -> None:
    if (
        manifest_bytes != payload.manifest_bytes
        or manifest_sha256 != payload.manifest_digest
        or manifest_size_bytes != len(payload.manifest_bytes)
        or segment_count != len(payload.segments)
        or total_size_bytes != payload.size_bytes
    ):
        raise PublicationPayloadConflictError(
            "publication key already exists with different exact payload bytes or metadata"
        )


def stage_payload(
    conn: sqlite3.Connection,
    *,
    payload: PublicationPayload,
    mode: PublicationMode,
    now_ms: int,
) -> PublicationObligation:
    """Stage exact bytes and obligation in the caller's source.db transaction."""
    _validate_payload(payload)
    obligation = record_obligation(
        conn,
        object_id=payload.object_id,
        protocol_version=payload.protocol_version,
        revision_id=payload.revision_id,
        manifest_digest=payload.manifest_digest,
        mode=mode,
        now_ms=now_ms,
    )
    key = _key(payload)
    conn.execute(
        """
        INSERT OR IGNORE INTO sinex_publication_payloads (
            object_id, protocol_version, revision_id, manifest_digest,
            manifest_bytes, manifest_sha256, manifest_size_bytes,
            segment_count, total_size_bytes, staged_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            *key,
            payload.manifest_bytes,
            payload.manifest_digest,
            len(payload.manifest_bytes),
            len(payload.segments),
            payload.size_bytes,
            now_ms,
        ),
    )
    row = conn.execute(
        """
        SELECT manifest_bytes, manifest_sha256, manifest_size_bytes,
               segment_count, total_size_bytes
        FROM sinex_publication_payloads
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
        """,
        key,
    ).fetchone()
    assert row is not None
    _assert_existing_payload_matches(
        manifest_bytes=bytes(row[0]),
        manifest_sha256=str(row[1]),
        manifest_size_bytes=int(row[2]),
        segment_count=int(row[3]),
        total_size_bytes=int(row[4]),
        payload=payload,
    )
    for position, (name, segment_bytes) in enumerate(payload.segments):
        digest = hashlib.sha256(segment_bytes).hexdigest()
        conn.execute(
            """
            INSERT OR IGNORE INTO sinex_publication_segments (
                object_id, protocol_version, revision_id, manifest_digest,
                position, segment_name, segment_bytes, segment_sha256, size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*key, position, name, segment_bytes, digest, len(segment_bytes)),
        )
        existing = conn.execute(
            """
            SELECT segment_name, segment_bytes, segment_sha256, size_bytes
            FROM sinex_publication_segments
            WHERE object_id = ? AND protocol_version = ? AND revision_id = ?
              AND manifest_digest = ? AND position = ?
            """,
            (*key, position),
        ).fetchone()
        assert existing is not None
        if (
            str(existing[0]),
            bytes(existing[1]),
            str(existing[2]),
            int(existing[3]),
        ) != (name, segment_bytes, digest, len(segment_bytes)):
            raise PublicationPayloadConflictError(
                f"publication key already exists with different segment bytes at position={position}"
            )
    return obligation


async def stage_payload_async(
    conn: AsyncSqlConnection,
    *,
    payload: PublicationPayload,
    mode: PublicationMode,
    now_ms: int,
) -> None:
    """Async equivalent of :func:`stage_payload`, without transaction ownership."""
    if mode is PublicationMode.OFF:
        raise ValueError("stage_payload_async must not be called in off mode")
    _validate_payload(payload)
    key = _key(payload)
    await conn.execute(
        """
        INSERT INTO sinex_publication_obligations (
            object_id, protocol_version, revision_id, manifest_digest, mode,
            status, attempt_count, created_at_ms, updated_at_ms,
            next_attempt_at_ms
        ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
        ON CONFLICT(object_id, protocol_version, revision_id, manifest_digest)
        DO UPDATE SET
            mode = CASE
                WHEN sinex_publication_obligations.mode = 'mirror'
                 AND excluded.mode = 'primary' THEN 'primary'
                ELSE sinex_publication_obligations.mode
            END,
            updated_at_ms = CASE
                WHEN sinex_publication_obligations.mode = 'mirror'
                 AND excluded.mode = 'primary' THEN excluded.updated_at_ms
                ELSE sinex_publication_obligations.updated_at_ms
            END
        """,
        (*key, mode.value, now_ms, now_ms, now_ms),
    )
    await conn.execute(
        """
        INSERT OR IGNORE INTO sinex_publication_payloads (
            object_id, protocol_version, revision_id, manifest_digest,
            manifest_bytes, manifest_sha256, manifest_size_bytes,
            segment_count, total_size_bytes, staged_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            *key,
            payload.manifest_bytes,
            payload.manifest_digest,
            len(payload.manifest_bytes),
            len(payload.segments),
            payload.size_bytes,
            now_ms,
        ),
    )
    cursor = await conn.execute(
        """
        SELECT manifest_bytes, manifest_sha256, manifest_size_bytes,
               segment_count, total_size_bytes
        FROM sinex_publication_payloads
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
        """,
        key,
    )
    row = await cursor.fetchone()
    assert row is not None
    values: tuple[object, ...] = tuple(row)  # type: ignore[arg-type]
    _assert_existing_payload_matches(
        manifest_bytes=bytes(cast(bytes, values[0])),
        manifest_sha256=str(values[1]),
        manifest_size_bytes=int(cast(int, values[2])),
        segment_count=int(cast(int, values[3])),
        total_size_bytes=int(cast(int, values[4])),
        payload=payload,
    )
    for position, (name, segment_bytes) in enumerate(payload.segments):
        digest = hashlib.sha256(segment_bytes).hexdigest()
        await conn.execute(
            """
            INSERT OR IGNORE INTO sinex_publication_segments (
                object_id, protocol_version, revision_id, manifest_digest,
                position, segment_name, segment_bytes, segment_sha256, size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*key, position, name, segment_bytes, digest, len(segment_bytes)),
        )
        cursor = await conn.execute(
            """
            SELECT segment_name, segment_bytes, segment_sha256, size_bytes
            FROM sinex_publication_segments
            WHERE object_id = ? AND protocol_version = ? AND revision_id = ?
              AND manifest_digest = ? AND position = ?
            """,
            (*key, position),
        )
        existing = await cursor.fetchone()
        assert existing is not None
        existing_values: tuple[object, ...] = tuple(existing)  # type: ignore[arg-type]
        if (
            str(existing_values[0]),
            bytes(cast(bytes, existing_values[1])),
            str(existing_values[2]),
            int(cast(int, existing_values[3])),
        ) != (name, segment_bytes, digest, len(segment_bytes)):
            raise PublicationPayloadConflictError(
                f"publication key already exists with different segment bytes at position={position}"
            )


def get_obligation(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    protocol_version: str,
    revision_id: str,
    manifest_digest: str,
) -> PublicationObligation | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        f"""
        SELECT {", ".join(_COLUMNS)} FROM sinex_publication_obligations
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
        """,
        (object_id, protocol_version, revision_id, manifest_digest),
    ).fetchone()
    return _row_to_obligation(row) if row is not None else None


def list_obligations(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[ObligationStatus, ...] | None = None,
    object_id: str | None = None,
    object_ids: Sequence[str] | None = None,
    due_at_ms: int | None = None,
    limit: int | None = None,
) -> tuple[PublicationObligation, ...]:
    """List obligations with deterministic ordering and optional due filter."""
    conn.row_factory = sqlite3.Row
    clauses: list[str] = []
    params: list[object] = []
    if statuses is not None:
        placeholders = ",".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        params.extend(status.value for status in statuses)
    if object_id is not None:
        clauses.append("object_id = ?")
        params.append(object_id)
    if object_ids is not None:
        unique_ids = tuple(dict.fromkeys(str(value) for value in object_ids if value))
        if not unique_ids:
            return ()
        placeholders = ",".join("?" for _ in unique_ids)
        clauses.append(f"object_id IN ({placeholders})")
        params.extend(unique_ids)
    if due_at_ms is not None:
        clauses.append("COALESCE(next_attempt_at_ms, created_at_ms) <= ?")
        params.append(due_at_ms)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = ""
    if limit is not None:
        if limit < 1:
            return ()
        limit_sql = " LIMIT ?"
        params.append(limit)
    rows = conn.execute(
        f"""
        SELECT {", ".join(_COLUMNS)} FROM sinex_publication_obligations
        {where}
        ORDER BY COALESCE(next_attempt_at_ms, created_at_ms), created_at_ms, object_id
        {limit_sql}
        """,
        params,
    ).fetchall()
    return tuple(_row_to_obligation(row) for row in rows)


def load_payload(conn: sqlite3.Connection, obligation: PublicationObligation) -> PublicationPayload:
    key = _key(obligation)
    row = conn.execute(
        """
        SELECT manifest_bytes, manifest_sha256, manifest_size_bytes,
               segment_count, total_size_bytes
        FROM sinex_publication_payloads
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
        """,
        key,
    ).fetchone()
    if row is None:
        raise PublicationPayloadInvalidError("obligation has no durable staged payload")
    segment_rows = conn.execute(
        """
        SELECT segment_name, segment_bytes, segment_sha256, size_bytes
        FROM sinex_publication_segments
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
        ORDER BY position
        """,
        key,
    ).fetchall()
    manifest_bytes = bytes(row[0])
    payload = PublicationPayload(
        object_id=obligation.object_id,
        protocol_version=obligation.protocol_version,
        revision_id=obligation.revision_id,
        manifest_digest=obligation.manifest_digest,
        manifest_bytes=manifest_bytes,
        segments=tuple((str(item[0]), bytes(item[1])) for item in segment_rows),
    )
    _validate_payload(payload)
    if str(row[1]) != hashlib.sha256(manifest_bytes).hexdigest():
        raise PublicationPayloadInvalidError("staged manifest digest reconciliation failed")
    if int(row[2]) != len(manifest_bytes):
        raise PublicationPayloadInvalidError("staged manifest size reconciliation failed")
    if int(row[3]) != len(payload.segments):
        raise PublicationPayloadInvalidError("staged segment count does not reconcile with payload metadata")
    if int(row[4]) != payload.size_bytes:
        raise PublicationPayloadInvalidError("staged total size reconciliation failed")
    for item, (_name, segment_bytes) in zip(segment_rows, payload.segments, strict=True):
        if str(item[2]) != hashlib.sha256(segment_bytes).hexdigest() or int(item[3]) != len(segment_bytes):
            raise PublicationPayloadInvalidError("staged segment digest/size reconciliation failed")
    return payload


def mark_publishing(
    conn: sqlite3.Connection,
    obligation: PublicationObligation,
    *,
    now_ms: int,
    lease_until_ms: int,
) -> PublicationObligation:
    """Durably lease one retryable row before invoking the transport."""
    cursor = conn.execute(
        """
        UPDATE sinex_publication_obligations
        SET status = 'publishing', updated_at_ms = ?, next_attempt_at_ms = ?
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
          AND status IN ('pending', 'publishing', 'durable_debt')
          AND COALESCE(next_attempt_at_ms, created_at_ms) <= ?
        """,
        (now_ms, lease_until_ms, *_key(obligation), now_ms),
    )
    if cursor.rowcount != 1:
        current = get_obligation(
            conn,
            object_id=obligation.object_id,
            protocol_version=obligation.protocol_version,
            revision_id=obligation.revision_id,
            manifest_digest=obligation.manifest_digest,
        )
        if current is None:
            raise RuntimeError("publication obligation disappeared while acquiring lease")
        return current
    updated = get_obligation(
        conn,
        object_id=obligation.object_id,
        protocol_version=obligation.protocol_version,
        revision_id=obligation.revision_id,
        manifest_digest=obligation.manifest_digest,
    )
    assert updated is not None
    return updated


def mark_attempt(
    conn: sqlite3.Connection,
    obligation: PublicationObligation,
    *,
    status: ObligationStatus,
    receipt: PublicationReceipt | None,
    error_code: str | None,
    now_ms: int,
    next_attempt_at_ms: int | None,
) -> PublicationObligation:
    """Persist one attempt outcome and append its secret-safe receipt history."""
    receipt_state = receipt.state if receipt is not None else None
    receipt_detail = receipt.detail if receipt is not None else ""
    retired_at_ms = now_ms if status in (ObligationStatus.CONFIRMED, ObligationStatus.REJECTED) else None
    next_attempt = None if retired_at_ms is not None else next_attempt_at_ms
    conn.execute(
        """
        UPDATE sinex_publication_obligations
        SET status = ?, attempt_count = attempt_count + 1,
            last_attempt_at_ms = ?, last_receipt_state = ?, last_error = ?,
            updated_at_ms = ?, retired_at_ms = ?, next_attempt_at_ms = ?
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
        """,
        (
            status.value,
            now_ms,
            receipt_state.value if receipt_state is not None else None,
            error_code,
            now_ms,
            retired_at_ms,
            next_attempt,
            *_key(obligation),
        ),
    )
    updated = get_obligation(
        conn,
        object_id=obligation.object_id,
        protocol_version=obligation.protocol_version,
        revision_id=obligation.revision_id,
        manifest_digest=obligation.manifest_digest,
    )
    assert updated is not None
    conn.execute(
        """
        INSERT INTO sinex_publication_receipts (
            object_id, protocol_version, revision_id, manifest_digest,
            attempt_number, request_id, receipt_state, receipt_detail,
            error_code, received_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            *_key(obligation),
            updated.attempt_count,
            obligation.request_id,
            receipt_state.value if receipt_state is not None else None,
            receipt_detail,
            error_code,
            now_ms,
        ),
    )
    return updated


def reset_retryable(
    conn: sqlite3.Connection,
    *,
    now_ms: int,
    object_ids: Sequence[str] | None = None,
    include_rejected: bool = False,
) -> int:
    """Operator-safe redrive primitive used by restart/ops reset paths."""
    statuses = ["pending", "publishing", "durable_debt"]
    if include_rejected:
        statuses.append("rejected")
    status_placeholders = ",".join("?" for _ in statuses)
    params: list[object] = [now_ms, now_ms, *statuses]
    object_clause = ""
    if object_ids is not None:
        ids = tuple(dict.fromkeys(str(value) for value in object_ids if value))
        if not ids:
            return 0
        object_clause = f" AND object_id IN ({','.join('?' for _ in ids)})"
        params.extend(ids)
    cursor = conn.execute(
        f"""
        UPDATE sinex_publication_obligations
        SET status = 'pending', retired_at_ms = NULL,
            next_attempt_at_ms = ?, updated_at_ms = ?
        WHERE status IN ({status_placeholders}){object_clause}
        """,
        params,
    )
    return int(cursor.rowcount)


__all__ = [
    "AsyncSqlConnection",
    "PublicationPayloadConflictError",
    "PublicationPayloadInvalidError",
    "get_obligation",
    "list_obligations",
    "load_payload",
    "mark_attempt",
    "mark_publishing",
    "record_obligation",
    "reset_retryable",
    "stage_payload",
    "stage_payload_async",
]

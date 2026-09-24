"""Immutable source-owned accepted marker inputs. Delivery belongs to user.db."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast


class AcceptedMarkerInputRefusedError(ValueError):
    """A prepared carrier is invalid or conflicts with an accepted identity."""


class _Cursor(Protocol):
    async def fetchone(self) -> object: ...

    async def fetchall(self) -> Iterable[object]: ...


class _Connection(Protocol):
    def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> Awaitable[_Cursor]: ...


def _encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _stored_payload(value: object) -> bytes:
    """Narrow SQLite's dynamically typed BLOB result before comparing it."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise AcceptedMarkerInputRefusedError("retained accepted marker payload is not a BLOB")


@dataclass(frozen=True, slots=True)
class PreparedAcceptedMarkerInput:
    raw_id: str
    identity: str
    payload: bytes
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class AcceptedMarkerInput:
    stream_id: str
    sequence: int
    batch: PreparedAcceptedMarkerInput


def retained_marker_input_sync(
    conn: sqlite3.Connection, request_key: str
) -> tuple[str, PreparedAcceptedMarkerInput, str | None] | None:
    """Return exact source-owned pending/accepted bytes for one request."""
    row = conn.execute(
        "SELECT 'pending', raw_id, carrier_digest, payload, expected_incarnation_id "
        "FROM pending_accepted_marker_inputs "
        "WHERE request_key = ? UNION ALL "
        "SELECT 'accepted', raw_id, payload_sha256, payload, index_incarnation_id "
        "FROM accepted_marker_inputs WHERE identity = ?",
        (request_key, request_key),
    ).fetchone()
    if row is None:
        return None
    state, raw_id, digest, payload_value, expected_incarnation_id = cast(tuple[str, str, str, object, str | None], row)
    payload = _stored_payload(payload_value)
    try:
        value = json.loads(payload)
        batch = PreparedAcceptedMarkerInput(raw_id, request_key, payload, digest)
        _validate(batch)
    except (ValueError, TypeError, KeyError) as exc:
        raise AcceptedMarkerInputRefusedError("retained accepted marker carrier is malformed") from exc
    if not isinstance(value, dict) or value.get("identity") != request_key:
        raise AcceptedMarkerInputRefusedError("retained accepted marker request key disagrees with payload")
    return state, batch, expected_incarnation_id


def prepare_accepted_marker_input(
    raw_id: str,
    sessions: Sequence[dict[str, object]],
    *,
    request_facts: dict[str, object] | None = None,
    request_sessions: Sequence[dict[str, object]] | None = None,
) -> PreparedAcceptedMarkerInput:
    """Seal one interpreted raw input, retaining empty candidate batches too.

    ``request_sessions`` is the complete normalized parse before append or
    lineage slicing. It defines replay identity; ``sessions`` contains the
    exact selected write carriers and may legitimately differ after a retry.
    """
    ordered = list(sessions)
    bindings = [{key: value for key, value in session.items() if key != "candidates"} for session in ordered]
    request_bindings = list(request_sessions if request_sessions is not None else bindings)
    facts = dict(request_facts or {})
    identity = hashlib.sha256(
        _encode({"format": 3, "raw_id": raw_id, "request_facts": facts, "sessions": request_bindings})
    ).hexdigest()
    payload = _encode(
        {
            "format": 3,
            "raw_id": raw_id,
            "identity": identity,
            "request_facts": facts,
            "request_sessions": request_bindings,
            "sessions": ordered,
        }
    )
    return PreparedAcceptedMarkerInput(raw_id, identity, payload, hashlib.sha256(payload).hexdigest())


def _validate(batch: PreparedAcceptedMarkerInput) -> None:
    try:
        value = json.loads(batch.payload)
        if not isinstance(value, dict):
            raise TypeError("carrier must be an object")
        raw_id = value.get("raw_id")
        sessions = value.get("sessions")
        if not isinstance(raw_id, str) or not isinstance(sessions, list):
            raise TypeError("carrier identity or sessions are invalid")
        if any(not isinstance(session, dict) for session in sessions):
            raise TypeError("carrier session must be an object")
        request_facts = value.get("request_facts")
        request_sessions = value.get("request_sessions")
        if not isinstance(request_facts, dict):
            raise TypeError("carrier request facts are invalid")
        if not isinstance(request_sessions, list) or any(not isinstance(session, dict) for session in request_sessions):
            raise TypeError("carrier request sessions are invalid")
        rebuilt = prepare_accepted_marker_input(
            raw_id, sessions, request_facts=request_facts, request_sessions=request_sessions
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise AcceptedMarkerInputRefusedError("invalid accepted marker carrier") from exc
    if not batch.raw_id or rebuilt != batch:
        raise AcceptedMarkerInputRefusedError("accepted marker carrier identity or bytes disagree")


def persist_pending_marker_input_sync(
    conn: sqlite3.Connection,
    batch: PreparedAcceptedMarkerInput,
    *,
    expected_incarnation_id: str,
) -> str:
    """Insert or verify a pending carrier in the caller's source transaction."""
    _validate(batch)
    if len(expected_incarnation_id) != 36:
        raise AcceptedMarkerInputRefusedError("pending marker carrier has an invalid index incarnation")
    accepted = conn.execute(
        "SELECT payload_sha256, payload FROM accepted_marker_inputs WHERE identity = ?",
        (batch.identity,),
    ).fetchone()
    if accepted is not None:
        digest, payload = cast(tuple[str, object], accepted)
        if digest != batch.payload_sha256 or _stored_payload(payload) != batch.payload:
            raise AcceptedMarkerInputRefusedError("accepted marker request conflicts with retained carrier")
        return "accepted"
    row = conn.execute(
        "SELECT carrier_digest, expected_incarnation_id, payload FROM pending_accepted_marker_inputs "
        "WHERE request_key = ?",
        (batch.identity,),
    ).fetchone()
    if row is not None:
        if tuple(row) != (batch.payload_sha256, expected_incarnation_id, batch.payload):
            raise AcceptedMarkerInputRefusedError("pending marker request conflicts with retained carrier")
        return "pending-existing"
    conn.execute(
        "INSERT INTO pending_accepted_marker_inputs(request_key, raw_id, carrier_digest, "
        "expected_incarnation_id, payload) VALUES (?, ?, ?, ?, ?)",
        (batch.identity, batch.raw_id, batch.payload_sha256, expected_incarnation_id, batch.payload),
    )
    return "pending-new"


async def append_accepted_marker_input(
    conn: _Connection,
    batch: PreparedAcceptedMarkerInput,
    *,
    index_incarnation_id: str | None = None,
) -> int:
    """Append within the caller's source acceptance transaction; never commit.

    The unique interpretation identity makes identical replay a no-op. Sequence
    allocation is SQLite-owned and survives restart independently of index.db.
    """
    _validate(batch)
    if index_incarnation_id is not None and len(index_incarnation_id) != 36:
        raise AcceptedMarkerInputRefusedError("accepted marker carrier has an invalid index incarnation")
    cursor = await conn.execute(
        "SELECT sequence, payload, payload_sha256, index_incarnation_id FROM accepted_marker_inputs WHERE identity = ?",
        (batch.identity,),
    )
    existing = await cursor.fetchone()
    if existing is not None:
        sequence, payload, digest, retained_incarnation = cast(tuple[int, object, str, str | None], existing)
        if (
            _stored_payload(payload) != batch.payload
            or digest != batch.payload_sha256
            or retained_incarnation != index_incarnation_id
        ):
            raise AcceptedMarkerInputRefusedError("conflicting replay of accepted marker input")
        return int(sequence)
    await conn.execute(
        "INSERT OR IGNORE INTO accepted_marker_stream(singleton, stream_id) VALUES (1, ?)",
        (str(uuid.uuid4()),),
    )
    cursor = await conn.execute(
        "INSERT INTO accepted_marker_inputs(identity, raw_id, payload, index_incarnation_id, payload_sha256) "
        "VALUES (?, ?, ?, ?, ?) RETURNING sequence",
        (batch.identity, batch.raw_id, batch.payload, index_incarnation_id, batch.payload_sha256),
    )
    row = await cursor.fetchone()
    assert row is not None
    return int(cast(tuple[int], row)[0])


async def persist_pending_accepted_marker_input(
    conn: _Connection,
    batch: PreparedAcceptedMarkerInput,
    *,
    expected_incarnation_id: str,
) -> None:
    """Durably retain exact carrier bytes before the index transaction commits."""
    _validate(batch)
    if len(expected_incarnation_id) != 36:
        raise AcceptedMarkerInputRefusedError("pending marker carrier has an invalid index incarnation")
    cursor = await conn.execute(
        "SELECT payload_sha256, payload FROM accepted_marker_inputs WHERE identity = ?",
        (batch.identity,),
    )
    accepted = await cursor.fetchone()
    if accepted is not None:
        digest, payload = cast(tuple[str, object], accepted)
        if digest != batch.payload_sha256 or _stored_payload(payload) != batch.payload:
            raise AcceptedMarkerInputRefusedError("accepted marker request conflicts with retained carrier")
        return
    cursor = await conn.execute(
        "SELECT carrier_digest, expected_incarnation_id, payload FROM pending_accepted_marker_inputs "
        "WHERE request_key = ?",
        (batch.identity,),
    )
    row = await cursor.fetchone()
    if row is not None:
        digest, retained_incarnation, payload = cast(tuple[str, str, object], row)
        if (
            digest != batch.payload_sha256
            or retained_incarnation != expected_incarnation_id
            or _stored_payload(payload) != batch.payload
        ):
            raise AcceptedMarkerInputRefusedError("pending marker request conflicts with retained carrier")
        return
    await conn.execute(
        "INSERT INTO pending_accepted_marker_inputs(request_key, raw_id, carrier_digest, "
        "expected_incarnation_id, payload) VALUES (?, ?, ?, ?, ?)",
        (batch.identity, batch.raw_id, batch.payload_sha256, expected_incarnation_id, batch.payload),
    )


async def finalize_pending_accepted_marker_input(conn: _Connection, batch: PreparedAcceptedMarkerInput) -> int:
    """Append accepted bytes and remove their pending copy in the caller's transaction."""
    _validate(batch)
    cursor = await conn.execute(
        "SELECT carrier_digest, expected_incarnation_id, payload FROM pending_accepted_marker_inputs "
        "WHERE request_key = ?",
        (batch.identity,),
    )
    row = await cursor.fetchone()
    if row is None:
        accepted_cursor = await conn.execute(
            "SELECT sequence, payload_sha256, payload, index_incarnation_id "
            "FROM accepted_marker_inputs WHERE identity = ?",
            (batch.identity,),
        )
        accepted = await accepted_cursor.fetchone()
        if accepted is not None:
            sequence, digest, payload, _incarnation_id = cast(tuple[int, str, object, str | None], accepted)
            if digest == batch.payload_sha256 and _stored_payload(payload) == batch.payload:
                return int(sequence)
        raise AcceptedMarkerInputRefusedError("pending marker carrier is absent for this request")
    digest, expected_incarnation_id, payload = cast(tuple[str, str, object], row)
    if digest != batch.payload_sha256 or _stored_payload(payload) != batch.payload:
        raise AcceptedMarkerInputRefusedError("pending marker carrier differs from this request")
    sequence = await append_accepted_marker_input(conn, batch, index_incarnation_id=expected_incarnation_id)
    await conn.execute("DELETE FROM pending_accepted_marker_inputs WHERE request_key = ?", (batch.identity,))
    return sequence


async def read_accepted_marker_inputs(
    conn: _Connection, *, after_sequence: int = 0, limit: int = 100
) -> tuple[AcceptedMarkerInput, ...]:
    """Read a bounded, validated page without changing source or delivery state."""
    if after_sequence < 0 or not 1 <= limit <= 1000:
        raise ValueError("marker stream requires a nonnegative position and a limit from 1 to 1000")
    cursor = await conn.execute(
        "SELECT s.stream_id, i.sequence, i.raw_id, i.identity, i.payload, i.payload_sha256 "
        "FROM accepted_marker_inputs i CROSS JOIN accepted_marker_stream s "
        "WHERE s.singleton = 1 AND i.sequence > ? ORDER BY i.sequence LIMIT ?",
        (after_sequence, limit),
    )
    result = []
    for row in await cursor.fetchall():
        stream_id, sequence, raw_id, identity, payload, digest = cast(tuple[str, int, str, str, bytes, str], row)
        batch = PreparedAcceptedMarkerInput(raw_id, identity, payload, digest)
        _validate(batch)
        result.append(AcceptedMarkerInput(stream_id, sequence, batch))
    return tuple(result)

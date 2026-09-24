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


class AcceptedMarkerInputExcisedError(AcceptedMarkerInputRefusedError):
    """A durable excision tombstone forbids restoring one marker carrier."""


class MixedAcceptedMarkerInputError(AcceptedMarkerInputRefusedError):
    """One sealed carrier would mix excised and retained sessions."""


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


@dataclass(frozen=True, slots=True)
class MarkerInputExcisionTarget:
    """Content-free identity needed to tombstone one source marker carrier."""

    identity: str
    raw_id: str
    carrier_digest: str
    state: str
    stream_id: str | None = None
    accepted_sequence: int | None = None
    #: The carrier was erased during an interrupted earlier source phase; its
    #: terminal evidence still owns index-witness cleanup on retry.
    tombstoned: bool = False


def marker_input_session_ids(batch: PreparedAcceptedMarkerInput) -> frozenset[str]:
    """Return complete request and selected session membership of one carrier."""
    _validate(batch)
    value = json.loads(batch.payload)
    assert isinstance(value, dict)  # established by _validate
    session_ids: set[str] = set()
    for collection_name in ("request_sessions", "sessions"):
        collection = value[collection_name]
        assert isinstance(collection, list)  # established by _validate
        for session in collection:
            assert isinstance(session, dict)  # established by _validate
            session_id = session.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise AcceptedMarkerInputRefusedError("marker carrier has an invalid session membership")
            session_ids.add(session_id)
    return frozenset(session_ids)


def marker_input_excision_targets_sync(
    conn: sqlite3.Connection,
    *,
    target_session_ids: frozenset[str],
    target_raw_ids: frozenset[str],
) -> tuple[MarkerInputExcisionTarget, ...]:
    """Resolve wholly-targeted carriers and fail closed for sealed mixed bytes."""
    rows = conn.execute(
        "SELECT 'pending', request_key, raw_id, carrier_digest, payload, NULL, NULL "
        "FROM pending_accepted_marker_inputs UNION ALL "
        "SELECT 'accepted', identity, raw_id, payload_sha256, payload, sequence, "
        "(SELECT stream_id FROM accepted_marker_stream WHERE singleton = 1) "
        "FROM accepted_marker_inputs"
    ).fetchall()
    targets: list[MarkerInputExcisionTarget] = []
    for state, identity, raw_id, digest, payload_value, sequence, stream_id in rows:
        batch = PreparedAcceptedMarkerInput(str(raw_id), str(identity), _stored_payload(payload_value), str(digest))
        session_ids = marker_input_session_ids(batch)
        if not (session_ids & target_session_ids) and batch.raw_id not in target_raw_ids:
            continue
        retained_session_ids = session_ids - target_session_ids
        if retained_session_ids:
            raise MixedAcceptedMarkerInputError(
                "accepted marker carrier mixes excised and retained sessions: "
                + ", ".join(sorted(retained_session_ids))
            )
        targets.append(
            MarkerInputExcisionTarget(
                identity=batch.identity,
                raw_id=batch.raw_id,
                carrier_digest=batch.payload_sha256,
                state=str(state),
                stream_id=None if stream_id is None else str(stream_id),
                accepted_sequence=None if sequence is None else int(sequence),
            )
        )
    if target_raw_ids:
        placeholders = ",".join("?" for _ in target_raw_ids)
        for identity, raw_id, digest, state, stream_id, sequence in conn.execute(
            f"SELECT identity, raw_id, carrier_digest, state, stream_id, accepted_sequence "
            f"FROM excised_marker_inputs WHERE raw_id IN ({placeholders})",
            tuple(sorted(target_raw_ids)),
        ).fetchall():
            if any(target.identity == str(identity) for target in targets):
                continue
            targets.append(
                MarkerInputExcisionTarget(
                    identity=str(identity),
                    raw_id=str(raw_id),
                    carrier_digest=str(digest),
                    state=str(state),
                    stream_id=None if stream_id is None else str(stream_id),
                    accepted_sequence=None if sequence is None else int(sequence),
                    tombstoned=True,
                )
            )
    return tuple(targets)


def excise_marker_input_targets_sync(
    conn: sqlite3.Connection, targets: Iterable[MarkerInputExcisionTarget], *, excised_at_ms: int
) -> dict[str, int]:
    """Tombstone first, then erase source payloads through the explicit path."""
    counts = {"pending": 0, "accepted": 0}
    for target in targets:
        if target.tombstoned:
            # A source-first crash erased the payload but not its rebuildable
            # witness. Preserve the original carrier count in the first
            # durable receipt written by the retry.
            counts[target.state] += 1
            continue
        conn.execute(
            "INSERT INTO excised_marker_inputs("
            "identity, raw_id, carrier_digest, state, stream_id, accepted_sequence, excised_at_ms"
            ") VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(identity) DO NOTHING",
            (
                target.identity,
                target.raw_id,
                target.carrier_digest,
                target.state,
                target.stream_id,
                target.accepted_sequence,
                excised_at_ms,
            ),
        )
        if target.state == "pending":
            cursor = conn.execute(
                "DELETE FROM pending_accepted_marker_inputs "
                "WHERE request_key = ? AND raw_id = ? AND carrier_digest = ?",
                (target.identity, target.raw_id, target.carrier_digest),
            )
        elif target.state == "accepted":
            cursor = conn.execute(
                "DELETE FROM accepted_marker_inputs WHERE identity = ? AND raw_id = ? AND payload_sha256 = ?",
                (target.identity, target.raw_id, target.carrier_digest),
            )
        else:
            raise AcceptedMarkerInputRefusedError(f"unknown marker carrier state: {target.state}")
        counts[target.state] += max(cursor.rowcount, 0)
    return counts


def _assert_marker_input_not_excised_sync(conn: sqlite3.Connection, identity: str, raw_id: str | None = None) -> None:
    if raw_id is None:
        row = conn.execute("SELECT 1 FROM excised_marker_inputs WHERE identity = ?", (identity,)).fetchone()
    else:
        row = conn.execute(
            "SELECT 1 FROM excised_marker_inputs WHERE identity = ? OR raw_id = ? LIMIT 1",
            (identity, raw_id),
        ).fetchone()
    if row is not None:
        raise AcceptedMarkerInputExcisedError("accepted marker carrier was excised and cannot be restored")


async def _assert_marker_input_not_excised(conn: _Connection, identity: str, raw_id: str | None = None) -> None:
    if raw_id is None:
        cursor = await conn.execute("SELECT 1 FROM excised_marker_inputs WHERE identity = ?", (identity,))
    else:
        cursor = await conn.execute(
            "SELECT 1 FROM excised_marker_inputs WHERE identity = ? OR raw_id = ? LIMIT 1",
            (identity, raw_id),
        )
    if await cursor.fetchone() is not None:
        raise AcceptedMarkerInputExcisedError("accepted marker carrier was excised and cannot be restored")


def retained_marker_input_sync(
    conn: sqlite3.Connection, request_key: str
) -> tuple[str, PreparedAcceptedMarkerInput, str | None] | None:
    """Return exact source-owned pending/accepted bytes for one request."""
    _assert_marker_input_not_excised_sync(conn, request_key)
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
    _assert_marker_input_not_excised_sync(conn, batch.identity, batch.raw_id)
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
    await _assert_marker_input_not_excised(conn, batch.identity, batch.raw_id)
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
    await _assert_marker_input_not_excised(conn, batch.identity, batch.raw_id)
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
    await _assert_marker_input_not_excised(conn, batch.identity, batch.raw_id)
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

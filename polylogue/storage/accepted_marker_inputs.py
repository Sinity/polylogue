"""Immutable source-owned accepted marker inputs. Delivery belongs to user.db."""

from __future__ import annotations

import hashlib
import json
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


def prepare_accepted_marker_input(raw_id: str, sessions: Sequence[dict[str, object]]) -> PreparedAcceptedMarkerInput:
    """Seal one interpreted raw input, retaining empty candidate batches too."""
    ordered = sorted(sessions, key=lambda item: str(item["session_id"]))
    bindings = [{key: value for key, value in session.items() if key != "candidates"} for session in ordered]
    identity = hashlib.sha256(_encode({"format": 1, "raw_id": raw_id, "sessions": bindings})).hexdigest()
    payload = _encode({"format": 1, "raw_id": raw_id, "identity": identity, "sessions": ordered})
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
        rebuilt = prepare_accepted_marker_input(raw_id, sessions)
    except (ValueError, TypeError, KeyError) as exc:
        raise AcceptedMarkerInputRefusedError("invalid accepted marker carrier") from exc
    if not batch.raw_id or rebuilt != batch:
        raise AcceptedMarkerInputRefusedError("accepted marker carrier identity or bytes disagree")


async def append_accepted_marker_input(conn: _Connection, batch: PreparedAcceptedMarkerInput) -> int:
    """Append within the caller's source acceptance transaction; never commit.

    The unique interpretation identity makes identical replay a no-op. Sequence
    allocation is SQLite-owned and survives restart independently of index.db.
    """
    _validate(batch)
    cursor = await conn.execute(
        "SELECT sequence, payload, payload_sha256 FROM accepted_marker_inputs WHERE identity = ?",
        (batch.identity,),
    )
    existing = await cursor.fetchone()
    if existing is not None:
        sequence, payload, digest = cast(tuple[int, bytes, str], existing)
        if payload != batch.payload or digest != batch.payload_sha256:
            raise AcceptedMarkerInputRefusedError("conflicting replay of accepted marker input")
        return int(sequence)
    await conn.execute(
        "INSERT OR IGNORE INTO accepted_marker_stream(singleton, stream_id) VALUES (1, ?)",
        (str(uuid.uuid4()),),
    )
    cursor = await conn.execute(
        "INSERT INTO accepted_marker_inputs(identity, raw_id, payload, payload_sha256) "
        "VALUES (?, ?, ?, ?) RETURNING sequence",
        (batch.identity, batch.raw_id, batch.payload, batch.payload_sha256),
    )
    row = await cursor.fetchone()
    assert row is not None
    return int(cast(tuple[int], row)[0])


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

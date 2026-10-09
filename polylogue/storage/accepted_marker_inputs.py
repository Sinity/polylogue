"""Immutable source-owned accepted marker inputs. Delivery belongs to user.db."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import uuid
from collections.abc import Awaitable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, BinaryIO, Protocol, cast

import ijson

if TYPE_CHECKING:
    from polylogue.storage.accepted_marker_producer import PreparedAcceptedMarkerCarrier
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation


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
    batch: AcceptedMarkerInputReference


@dataclass(frozen=True, slots=True)
class AcceptedMarkerInputReference:
    """Payload-free coordinates for one immutable Source carrier."""

    raw_id: str
    identity: str
    payload_sha256: str


@dataclass(slots=True)
class VerifiedAcceptedMarkerPayload:
    """A validated Source payload copied to a private seekable spool."""

    reference: AcceptedMarkerInputReference
    payload_file: BinaryIO
    candidate_count: int = 0
    retirement_count: int = 0

    def close(self) -> None:
        self.payload_file.close()

    def iter_items(self, prefix: str) -> Iterator[object]:
        yield from _iter_json_items(self.payload_file, prefix)

    def iter_session_ids(self) -> Iterator[str]:
        """Yield complete request and selected session membership without arrays."""
        for prefix in ("request_sessions.item.session_id", "sessions.item.session_id"):
            for value in self.iter_items(prefix):
                if not isinstance(value, str) or not value:
                    raise AcceptedMarkerInputRefusedError("marker carrier has an invalid session membership")
                yield value


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


def stage_accepted_marker_input(seal: PreparedIndexMutation, carrier: PreparedAcceptedMarkerCarrier) -> None:
    """Add one generated immutable marker root to an original Source seal.

    The caller owns the surrounding ``seal.source_producer()`` and later
    publishes its exact Source permit. A matching prior root is idempotent;
    a conflicting root or excision tombstone is a typed refusal.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    try:
        batch = carrier.batch
        if not batch.raw_id or len(batch.identity) != 64 or len(batch.payload_sha256) != 64 or carrier.byte_length < 0:
            raise AcceptedMarkerInputRefusedError("invalid prepared accepted marker carrier metadata")
        with seal.source_rows(
            "SELECT raw_id,payload_sha256,length(payload) FROM accepted_marker_inputs WHERE identity=?",
            (batch.identity,),
        ) as rows:
            existing = rows.fetchone()
        if existing is None:
            with seal.original_rows(
                "source",
                "SELECT raw_id,payload_sha256,length(payload) FROM accepted_marker_inputs WHERE identity=?",
                (batch.identity,),
            ) as rows:
                existing = rows.fetchone()
        if existing is not None:
            if tuple(existing) != (batch.raw_id, batch.payload_sha256, carrier.byte_length):
                raise AcceptedMarkerInputRefusedError("conflicting replay of accepted marker input")
            for _chunk in carrier.verified_chunks():
                pass
            return
        with seal.source_rows(
            "SELECT 1 FROM excised_marker_inputs WHERE identity=? OR raw_id=? LIMIT 1",
            (batch.identity, batch.raw_id),
        ) as rows:
            excised = rows.fetchone()
        if excised is None:
            with seal.original_rows(
                "source",
                "SELECT 1 FROM excised_marker_inputs WHERE identity=? OR raw_id=? LIMIT 1",
                (batch.identity, batch.raw_id),
            ) as rows:
                excised = rows.fetchone()
        if excised is not None:
            raise AcceptedMarkerInputExcisedError("accepted marker carrier was excised and cannot be restored")
        with seal.source_rows("SELECT stream_id FROM accepted_marker_stream WHERE singleton=1") as rows:
            stream_row = rows.fetchone()
        if stream_row is None:
            with seal.original_rows("source", "SELECT stream_id FROM accepted_marker_stream WHERE singleton=1") as rows:
                stream_row = rows.fetchone()
        if stream_row is None:
            import uuid

            stream_cells = {
                "singleton": seal.retain_literal_scalar(1),
                "stream_id": seal.retain_literal_scalar(str(uuid.uuid4())),
            }
            expressions: list[str] = []
            parameters: list[object] = []
            for column in ("singleton", "stream_id"):
                expression, operands = seal.source_literal_expression(stream_cells[column])
                expressions.append(expression)
                parameters.extend(operands)
            singleton = stream_cells["singleton"]
            with seal.source_statement(
                "INSERT INTO accepted_marker_stream(singleton,stream_id) VALUES (" + ",".join(expressions) + ")",
                tuple(parameters),
                table="accepted_marker_stream",
                writable_targets=(("accepted_marker_stream", (singleton,)),),
                prepared_cells=stream_cells,
            ):
                pass

        payload_cell = seal.retain_literal_stream("blob", carrier.byte_length, carrier.verified_chunks())
        cells = {
            "identity": seal.retain_literal_scalar(batch.identity),
            "raw_id": seal.retain_literal_scalar(batch.raw_id),
            "payload": payload_cell,
            "index_incarnation_id": seal.retain_literal_scalar(None),
            "payload_sha256": seal.retain_literal_scalar(batch.payload_sha256),
        }
        payload_expressions: list[str] = []
        payload_parameters: list[object] = [None]
        for column in ("identity", "raw_id", "payload", "index_incarnation_id", "payload_sha256"):
            expression, operands = seal.source_literal_expression(cells[column])
            payload_expressions.append(expression)
            payload_parameters.extend(operands)
        sql = (
            "INSERT INTO accepted_marker_inputs(sequence, identity, raw_id, payload, index_incarnation_id, payload_sha256) "
            "VALUES (?, " + ", ".join(payload_expressions) + ") RETURNING sequence"
        )
        with seal.source_statement(
            sql,
            tuple(payload_parameters),
            table="accepted_marker_inputs",
            writable_targets=(),
            prepared_cells=cells,
            allocation_parameter=0,
            generated_primary_key=True,
        ) as inserted:
            if inserted.fetchone() is None:
                raise ReferenceSealError("accepted marker root omitted its generated sequence")
    except (sqlite3.DataError, OverflowError) as exc:
        if "too big" in str(exc).lower() or isinstance(exc, OverflowError):
            raise AcceptedMarkerInputRefusedError(
                "accepted marker carrier exceeds SQLite's physical value limit"
            ) from exc
        raise
    finally:
        carrier.close()


def _iter_json_items(payload: BinaryIO, prefix: str) -> Iterator[object]:
    payload.seek(0)
    try:
        for value in ijson.items(payload, prefix, use_float=False):
            yield _marker_json_numbers(value)
    except (ijson.JSONError, UnicodeError, ValueError) as exc:
        raise AcceptedMarkerInputRefusedError("invalid accepted marker carrier") from exc


def _marker_json_numbers(value: object) -> object:
    """Restore producer JSON floats while keeping arbitrary integer tokens exact.

    The producer emits Python floats with stdlib JSON. ijson's decimal mode
    preserves the integer domain; decimal tokens are those original floats,
    including integral floats such as 1.0 that must not reserialize as 1.
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_marker_json_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _marker_json_numbers(item) for key, item in value.items()}
    return value


def _one_json_item(payload: BinaryIO, prefix: str) -> object:
    items = _iter_json_items(payload, prefix)
    first = next(items, _MISSING)
    if first is _MISSING or next(items, _MISSING) is not _MISSING:
        raise AcceptedMarkerInputRefusedError("accepted marker carrier has invalid top-level metadata")
    return first


_MISSING = object()


def _validate_marker_payload(payload: BinaryIO, reference: AcceptedMarkerInputReference) -> tuple[int, int]:
    """Validate one format-3 carrier without constructing its arrays."""
    try:
        root_keys: set[str] = set()
        root_open = False
        root_closed = False
        request_array = False
        selected_array = False
        request_session_id = False
        selected_session_id = False
        for prefix, event, value in _iter_json_events(payload):
            if prefix == "" and event == "start_map":
                if root_open or root_closed:
                    raise ValueError("carrier has more than one root")
                root_open = True
            elif prefix == "" and event == "map_key":
                key = cast(str, value)
                if key in root_keys:
                    raise ValueError("carrier repeats a top-level key")
                root_keys.add(key)
            elif prefix == "" and event == "end_map":
                root_closed = True
            elif prefix == "format":
                if event != "number" or value != 3:
                    raise ValueError("carrier format is unsupported")
            elif prefix in {"raw_id", "identity"} and event != "string":
                raise ValueError("carrier metadata is not text")
            elif prefix == "request_facts" and event not in {"start_map", "map_key", "end_map"}:
                raise ValueError("carrier request facts are not an object")
            elif prefix == "request_sessions" and event == "start_array":
                request_array = True
            elif prefix == "sessions" and event == "start_array":
                selected_array = True
            elif prefix == "request_sessions.item" and event == "start_map":
                request_session_id = False
            elif prefix == "request_sessions.item" and event not in {"start_map", "map_key", "end_map"}:
                raise ValueError("request session is not an object")
            elif prefix == "request_sessions.item.session_id" and event == "string" and value:
                request_session_id = True
            elif prefix == "request_sessions.item" and event == "end_map":
                if not request_session_id:
                    raise ValueError("request session has no canonical session id")
            elif prefix == "sessions.item" and event == "start_map":
                selected_session_id = False
            elif prefix == "sessions.item" and event not in {"start_map", "map_key", "end_map"}:
                raise ValueError("selected session is not an object")
            elif prefix == "sessions.item.session_id" and event == "string" and value:
                selected_session_id = True
            elif prefix == "sessions.item.candidates" and event == "start_array":
                pass
            elif prefix == "sessions.item.candidates" and event not in {"start_array", "end_array"}:
                raise ValueError("selected session candidates are not an array")
            elif prefix == "sessions.item.retired_assertions" and event == "start_array":
                pass
            elif prefix == "sessions.item.retired_assertions" and event not in {"start_array", "end_array"}:
                raise ValueError("selected session retirements are not an array")
            elif prefix == "sessions.item" and event == "end_map":
                if not selected_session_id:
                    raise ValueError("selected session carrier has no canonical session id")

        if (
            not root_open
            or not root_closed
            or root_keys
            != {
                "format",
                "raw_id",
                "identity",
                "request_facts",
                "request_sessions",
                "sessions",
            }
        ):
            raise ValueError("carrier root fields are incomplete")
        if not request_array or not selected_array:
            raise ValueError("carrier session arrays are absent")
        raw_id = _one_json_item(payload, "raw_id")
        identity = _one_json_item(payload, "identity")
        facts = _one_json_item(payload, "request_facts")
        if (
            not isinstance(raw_id, str)
            or raw_id != reference.raw_id
            or not isinstance(identity, str)
            or identity != reference.identity
            or not isinstance(facts, dict)
        ):
            raise ValueError("carrier identity facts disagree with Source coordinates")

        digest = hashlib.sha256()
        digest.update(b'{"format":3,"raw_id":')
        digest.update(_encode(raw_id))
        digest.update(b',"request_facts":')
        digest.update(_encode(facts))
        digest.update(b',"sessions":[')
        first = True
        for request_session in _iter_json_items(payload, "request_sessions.item"):
            if not isinstance(request_session, dict):
                raise ValueError("request session is not an object")
            session_id = request_session.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("request session has no canonical session id")
            if not first:
                digest.update(b",")
            first = False
            digest.update(_encode(request_session))
        digest.update(b"]}")
        if digest.hexdigest() != identity:
            raise ValueError("carrier identity digest disagrees with request facts")

        # Traverse candidates and retirements independently so neither array
        # is retained while the other is decoded. Their detailed lowering
        # schema is checked by the marker-domain decoder at publication.
        candidate_count = sum(1 for _ in _iter_json_items(payload, "sessions.item.candidates.item"))
        retirement_count = 0
        for retired in _iter_json_items(payload, "sessions.item.retired_assertions.item"):
            if not isinstance(retired, str) or not retired:
                raise ValueError("retired assertion id is invalid")
            retirement_count += 1
    except (ijson.JSONError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AcceptedMarkerInputRefusedError):
            raise
        raise AcceptedMarkerInputRefusedError("invalid accepted marker carrier") from exc
    finally:
        payload.seek(0)
    return candidate_count, retirement_count


def _iter_json_events(payload: BinaryIO) -> Iterator[tuple[str, str, object]]:
    payload.seek(0)
    try:
        for prefix, event, value in ijson.parse(payload, use_float=False):
            yield prefix, event, _marker_json_numbers(value)
    except (ijson.JSONError, UnicodeError, ValueError) as exc:
        raise AcceptedMarkerInputRefusedError("invalid accepted marker carrier") from exc
    finally:
        payload.seek(0)


def verified_marker_payload_from_blob(
    conn: sqlite3.Connection, accepted: AcceptedMarkerInput
) -> VerifiedAcceptedMarkerPayload:
    """Copy one Source BLOB incrementally, then verify its digest and identity."""
    row = conn.execute(
        "SELECT rowid,raw_id,payload_sha256 FROM accepted_marker_inputs WHERE sequence=? AND identity=?",
        (accepted.sequence, accepted.batch.identity),
    ).fetchone()
    if row is None or tuple(row[1:]) != (accepted.batch.raw_id, accepted.batch.payload_sha256):
        raise AcceptedMarkerInputRefusedError("accepted marker coordinates changed before consumption")
    spool = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
    digest = hashlib.sha256()
    try:
        with conn.blobopen("accepted_marker_inputs", "payload", int(row[0]), readonly=True) as source:
            while chunk := source.read(1024 * 1024):
                spool.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != accepted.batch.payload_sha256:
            raise AcceptedMarkerInputRefusedError("accepted marker payload digest disagrees with Source")
        spool.seek(0)
        candidate_count, retirement_count = _validate_marker_payload(spool, accepted.batch)
        spool.seek(0)
        return VerifiedAcceptedMarkerPayload(accepted.batch, spool, candidate_count, retirement_count)
    except BaseException:
        spool.close()
        raise


def iter_marker_input_session_ids(payload: BinaryIO, reference: AcceptedMarkerInputReference) -> Iterator[str]:
    """Validate a caller-owned marker payload and yield its membership IDs."""
    payload.seek(0)
    digest = hashlib.sha256()
    while chunk := payload.read(1024 * 1024):
        digest.update(chunk)
    if digest.hexdigest() != reference.payload_sha256:
        raise AcceptedMarkerInputRefusedError("accepted marker payload digest disagrees with Source")
    _validate_marker_payload(payload, reference)
    for prefix in ("request_sessions.item.session_id", "sessions.item.session_id"):
        for value in _iter_json_items(payload, prefix):
            if not isinstance(value, str) or not value:
                raise AcceptedMarkerInputRefusedError("marker carrier has an invalid session membership")
            yield value


def marker_input_excision_targets_sync(
    conn: sqlite3.Connection,
    *,
    target_session_ids: frozenset[str],
    target_raw_ids: frozenset[str],
) -> tuple[MarkerInputExcisionTarget, ...]:
    """Resolve wholly-targeted carriers and fail closed for sealed mixed bytes."""
    targets: list[MarkerInputExcisionTarget] = []
    tables = (
        (
            "pending",
            "pending_accepted_marker_inputs",
            "SELECT rowid,request_key,raw_id,carrier_digest,NULL,NULL FROM pending_accepted_marker_inputs",
        ),
        (
            "accepted",
            "accepted_marker_inputs",
            "SELECT rowid,identity,raw_id,payload_sha256,sequence,"
            "(SELECT stream_id FROM accepted_marker_stream WHERE singleton=1) FROM accepted_marker_inputs",
        ),
    )
    for state, table, sql in tables:
        rows = conn.execute(sql)
        try:
            while row := rows.fetchone():
                rowid, identity, raw_id, digest, sequence, stream_id = row
                reference = AcceptedMarkerInputReference(str(raw_id), str(identity), str(digest))
                saw_target = False
                saw_retained = False
                try:
                    with conn.blobopen(table, "payload", int(rowid), readonly=True) as payload:
                        for session_id in iter_marker_input_session_ids(cast(BinaryIO, payload), reference):
                            if session_id in target_session_ids:
                                saw_target = True
                            else:
                                saw_retained = True
                except (sqlite3.Error, OSError) as exc:
                    raise AcceptedMarkerInputRefusedError("cannot read accepted marker carrier for excision") from exc
                if not (saw_target or reference.raw_id in target_raw_ids):
                    continue
                if saw_retained:
                    raise MixedAcceptedMarkerInputError("accepted marker carrier mixes excised and retained sessions")
                targets.append(
                    MarkerInputExcisionTarget(
                        identity=reference.identity,
                        raw_id=reference.raw_id,
                        carrier_digest=reference.payload_sha256,
                        state=state,
                        stream_id=None if stream_id is None else str(stream_id),
                        accepted_sequence=None if sequence is None else int(sequence),
                    )
                )
        finally:
            rows.close()
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
        reference = AcceptedMarkerInputReference(batch.raw_id, batch.identity, batch.payload_sha256)
        if hashlib.sha256(batch.payload).hexdigest() != batch.payload_sha256:
            raise ValueError("carrier payload digest disagrees")
        _validate_marker_payload(io.BytesIO(batch.payload), reference)
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, AcceptedMarkerInputRefusedError):
            raise
        raise AcceptedMarkerInputRefusedError("invalid accepted marker carrier") from exc
    if not batch.raw_id or len(batch.identity) != 64 or len(batch.payload_sha256) != 64:
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


def replace_uncommitted_pending_marker_input_sync(
    conn: sqlite3.Connection,
    batch: PreparedAcceptedMarkerInput,
    *,
    retained_digest: str,
    expected_incarnation_id: str,
) -> None:
    """Replace a pending carrier whose index transaction provably never committed.

    A pending carrier is the prepare half of a two-tier commit. Its caller
    holds the Index write transaction and establishes that the corresponding
    publication did not commit in this incarnation, so the retained bytes
    were never accepted or delivered. They describe a rolled-back (or
    replaced-index) attempt, and the current interpretation supersedes them.
    Accepted carriers are never replaced.
    """
    _validate(batch)
    _assert_marker_input_not_excised_sync(conn, batch.identity, batch.raw_id)
    if len(expected_incarnation_id) != 36:
        raise AcceptedMarkerInputRefusedError("pending marker carrier has an invalid index incarnation")
    cursor = conn.execute(
        "UPDATE pending_accepted_marker_inputs SET carrier_digest = ?, expected_incarnation_id = ?, payload = ? "
        "WHERE request_key = ? AND raw_id = ? AND carrier_digest = ?",
        (
            batch.payload_sha256,
            expected_incarnation_id,
            batch.payload,
            batch.identity,
            batch.raw_id,
            retained_digest,
        ),
    )
    if cursor.rowcount != 1:
        raise AcceptedMarkerInputRefusedError("pending marker carrier changed while it was being replaced")


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


_ACCEPTED_MARKER_PAGE_SQL = (
    "SELECT s.stream_id, i.sequence, i.raw_id, i.identity, i.payload_sha256 "
    "FROM accepted_marker_inputs i CROSS JOIN accepted_marker_stream s "
    "WHERE s.singleton = 1 AND i.sequence > ? ORDER BY i.sequence LIMIT ?"
)


def _marker_page_parameters(after_sequence: int, limit: int) -> tuple[int, int]:
    if after_sequence < 0 or not 1 <= limit <= 1000:
        raise ValueError("marker stream requires a nonnegative position and a limit from 1 to 1000")
    return after_sequence, limit


def _decode_marker_page(rows: Iterable[object]) -> tuple[AcceptedMarkerInput, ...]:
    result = []
    for row in rows:
        stream_id, sequence, raw_id, identity, digest = cast(tuple[str, int, str, str, str], row)
        if not raw_id or len(identity) != 64 or len(digest) != 64:
            raise AcceptedMarkerInputRefusedError("accepted marker row has invalid coordinates")
        batch = AcceptedMarkerInputReference(raw_id, identity, digest)
        result.append(AcceptedMarkerInput(stream_id, sequence, batch))
    return tuple(result)


def read_accepted_marker_inputs_sync(
    conn: sqlite3.Connection, *, after_sequence: int = 0, limit: int = 100
) -> tuple[AcceptedMarkerInput, ...]:
    """Read payload-free Source coordinates on the native connection's owning thread."""
    cursor = conn.execute(_ACCEPTED_MARKER_PAGE_SQL, _marker_page_parameters(after_sequence, limit))
    try:
        return _decode_marker_page(cursor.fetchall())
    finally:
        cursor.close()


async def read_accepted_marker_inputs(
    conn: _Connection, *, after_sequence: int = 0, limit: int = 100
) -> tuple[AcceptedMarkerInput, ...]:
    """Read a bounded page of payload-free coordinates without changing state."""
    cursor = await conn.execute(_ACCEPTED_MARKER_PAGE_SQL, _marker_page_parameters(after_sequence, limit))
    return _decode_marker_page(await cursor.fetchall())

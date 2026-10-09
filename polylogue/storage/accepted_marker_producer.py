"""Prepare and seal marker history for each accepted retained revision.

Marker candidates come from the exact canonical writer rows that are about to
be published.  The source carrier records both the complete normalized parse
and the selected write, so delivery is independent of a later index
projection.  A carrier is streamed through a private temporary file and then
through the reference seal's literal owner; count does not determine resident
carrier memory.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol

from polylogue.core.identity_law import block_id as archive_block_id
from polylogue.markers.lowering import assertion_id_for_marker, iter_candidates_for_block
from polylogue.storage.accepted_marker_inputs import AcceptedMarkerInputRefusedError
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation


class _Hash(Protocol):
    def update(self, value: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class AcceptedMarkerCarrierIdentity:
    raw_id: str
    identity: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedAcceptedMarkerCarrier:
    """One sealed source carrier backed by a seekable private file."""

    batch: AcceptedMarkerCarrierIdentity
    payload_file: BinaryIO
    byte_length: int

    def chunks(self, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("marker carrier read size must be positive")
        self.payload_file.seek(0)
        while chunk := self.payload_file.read(chunk_size):
            yield chunk

    def verified_chunks(self, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        digest = hashlib.sha256()
        length = 0
        for chunk in self.chunks(chunk_size):
            digest.update(chunk)
            length += len(chunk)
            yield chunk
        if length != self.byte_length or digest.hexdigest() != self.batch.payload_sha256:
            raise AcceptedMarkerInputRefusedError("prepared accepted marker carrier changed before Source publication")

    def close(self) -> None:
        self.payload_file.close()


def marker_candidates_for_prepared_write_stream(prepared: PreparedSessionWrite) -> Iterator[dict[str, object]]:
    """Yield candidates in canonical block order without collecting the batch."""
    columns = tuple(column.name for column in BLOCKS_SPEC.insert_columns)
    # Assertion IDs can repeat across distinct marker coordinates. Keep this
    # deduplication on a private disk-backed index so long sessions stay
    # bounded by the current block and candidate.
    with tempfile.TemporaryDirectory(prefix="polylogue-marker-seen-") as directory:
        seen = sqlite3.connect(Path(directory) / "seen.sqlite3")
        try:
            seen.execute("CREATE TABLE ids (assertion_id TEXT PRIMARY KEY) WITHOUT ROWID")
            for values in prepared.rows.block_rows:
                row = dict(zip(columns, values, strict=True))
                text = row["text"]
                if not isinstance(text, str):
                    continue
                message_id = str(row["message_id"])
                block_id = archive_block_id(
                    message_id,
                    content_identity=str(row["content_identity"]),
                    content_occurrence=int(row["content_occurrence"]),
                )
                for candidate in iter_candidates_for_block(message_id, block_id, text):
                    assertion_id = assertion_id_for_marker(candidate)
                    if assertion_id is not None:
                        prior_changes = seen.total_changes
                        seen.execute("INSERT OR IGNORE INTO ids VALUES (?)", (assertion_id,))
                        if seen.total_changes == prior_changes:
                            continue
                    value = asdict(candidate)
                    value["assertion_kind"] = candidate.assertion_kind.value if candidate.assertion_kind else None
                    yield value
        finally:
            seen.close()


def prepare_accepted_marker_carrier(
    *,
    raw_id: str,
    request_facts: Mapping[str, object],
    request_sessions: Callable[[], Iterable[Mapping[str, object]]],
    prepared_sessions: Iterable[tuple[str, PreparedSessionWrite, Iterable[object]]],
) -> PreparedAcceptedMarkerCarrier:
    """Stream one accepted raw's complete request and selected write carrier.

    ``request_sessions`` must be repeatable. It yields candidate-free normalized
    session bindings on each pass. ``prepared_sessions`` yields the canonical
    output for each selected session as ``(session_id, write, retired_ids)``.
    """
    if not raw_id:
        raise AcceptedMarkerInputRefusedError("accepted marker carrier has an empty raw id")
    facts = dict(request_facts)
    identity = accepted_marker_input_identity(raw_id=raw_id, request_facts=facts, request_sessions=request_sessions)

    # Ownership transfers to the returned carrier, which closes it after its
    # seal staging attempt. A lexical context here would close before staging.
    payload_file = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
    payload_hash = hashlib.sha256()
    payload_length = 0

    def emit(value: bytes) -> None:
        nonlocal payload_length
        payload_file.write(value)
        payload_hash.update(value)
        payload_length += len(value)

    def emit_json(value: object) -> None:
        for part in _json_chunks(value):
            emit(part)

    try:
        emit(b'{"format":3,"identity":')
        emit_json(identity)
        emit(b',"raw_id":')
        emit_json(raw_id)
        emit(b',"request_facts":')
        emit_json(facts)
        emit(b',"request_sessions":[')
        first = True
        for session in request_sessions():
            _emit_comma(emit, first)
            first = False
            emit_json(_request_binding(session))
        emit(b'],"sessions":[')
        first = True
        for session_id, prepared, retired_ids in prepared_sessions:
            if not session_id:
                raise AcceptedMarkerInputRefusedError("accepted marker session has an empty id")
            _emit_comma(emit, first)
            first = False
            emit(b'{"candidates":[')
            candidate_first = True
            for candidate in marker_candidates_for_prepared_write_stream(prepared):
                _emit_comma(emit, candidate_first)
                candidate_first = False
                emit_json(candidate)
            emit(b'],"retired_assertions":[')
            retired_first = True
            for assertion_id in retired_ids:
                if not isinstance(assertion_id, str) or not assertion_id:
                    raise AcceptedMarkerInputRefusedError("retired marker assertion id is invalid")
                _emit_comma(emit, retired_first)
                retired_first = False
                emit_json(assertion_id)
            emit(b'],"session_id":')
            emit_json(session_id)
            emit(b"}")
        emit(b"]}")
        payload_file.flush()
        payload_file.seek(0)
        batch = AcceptedMarkerCarrierIdentity(raw_id, identity, payload_hash.hexdigest())
        return PreparedAcceptedMarkerCarrier(batch, payload_file, payload_length)
    except BaseException:
        payload_file.close()
        raise


def _without_candidates(value: Mapping[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "candidates"}


def _request_binding(value: Mapping[str, object]) -> dict[str, object]:
    binding = _without_candidates(value)
    session_id = binding.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise AcceptedMarkerInputRefusedError("accepted marker request session has no canonical session id")
    return binding


def _json_chunks(value: object) -> Iterator[bytes]:
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    for part in encoder.iterencode(value):
        yield part.encode("utf-8")


def _write_identity_prefix(digest: _Hash, raw_id: str, facts: dict[str, object]) -> None:
    # Keep the long `sessions` member out of json.dumps' object graph while
    # preserving the existing canonical identity format exactly.
    digest.update(b'{"format":3,"raw_id":')
    for part in _json_chunks(raw_id):
        digest.update(part)
    digest.update(b',"request_facts":')
    for part in _json_chunks(facts):
        digest.update(part)
    digest.update(b',"sessions":[')


def _write_json(digest: _Hash, value: object) -> None:
    for part in _json_chunks(value):
        digest.update(part)


def _write_comma(digest: _Hash, first: bool) -> None:
    if not first:
        digest.update(b",")


def _emit_comma(emit: Callable[[bytes], None], first: bool) -> None:
    if not first:
        emit(b",")


def accepted_marker_input_identity(
    *, raw_id: str, request_facts: Mapping[str, object], request_sessions: Callable[[], Iterable[Mapping[str, object]]]
) -> str:
    """Compute one accepted carrier identity from replayable normalized facts."""
    if not raw_id:
        raise AcceptedMarkerInputRefusedError("accepted marker carrier has an empty raw id")
    facts = dict(request_facts)
    required_facts = {
        "blob_hash",
        "provider",
        "revision_kind",
        "source_path",
        "parser_fingerprint",
        "marker_recipe",
    }
    if not required_facts <= facts.keys() or any(
        not isinstance(facts[key], str) or not facts[key] for key in required_facts
    ):
        raise AcceptedMarkerInputRefusedError("accepted marker request facts are incomplete")
    identity_hash = hashlib.sha256()
    _write_identity_prefix(identity_hash, raw_id, facts)
    first = True
    for session in request_sessions():
        _write_comma(identity_hash, first)
        first = False
        _write_json(identity_hash, _request_binding(session))
    identity_hash.update(b"]}")
    return identity_hash.hexdigest()


def accepted_marker_input_is_durable(
    seal: PreparedIndexMutation,
    *,
    raw_id: str,
    request_facts: Mapping[str, object],
    request_sessions: Callable[[], Iterable[Mapping[str, object]]],
) -> bool:
    """Check whether this exact normalized accepted identity already has a Source root.

    This lets replay preserve an earlier immutable carrier without preparing
    its canonical writer rows again. It never repairs or reconstructs a root.
    """
    identity = accepted_marker_input_identity(
        raw_id=raw_id, request_facts=request_facts, request_sessions=request_sessions
    )
    with seal.original_rows(
        "source", "SELECT raw_id FROM accepted_marker_inputs WHERE identity=?", (identity,)
    ) as rows:
        existing = rows.fetchone()
    if existing is None:
        return False
    if str(existing[0]) != raw_id:
        raise AcceptedMarkerInputRefusedError("accepted marker identity belongs to a different raw revision")
    return True


def stage_accepted_marker_carrier(seal: PreparedIndexMutation, carrier: PreparedAcceptedMarkerCarrier) -> None:
    """Stage one immutable generated Source root in the original seal."""
    from polylogue.storage.accepted_marker_inputs import stage_accepted_marker_input

    stage_accepted_marker_input(seal, carrier)

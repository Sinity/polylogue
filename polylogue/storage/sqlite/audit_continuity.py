"""Replayable cross-tier write-ahead control for durable ``audit.db`` writes.

SQLite cannot atomically commit transactions spanning source.db and audit.db.
The source control row is therefore the authoritative write-ahead command:
prepare it in source.db, commit the audit mutation plus its head, then promote
the source head.  Startup can complete the first two crash windows because the
pending row contains the exact typed command, and it rejects an audit image
whose head regressed after source promotion.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import sqlite3
import stat
import threading
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeVar, cast

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.audit_leaf import (
    AuditLeafError,
    open_verified_audit_connection,
    open_verified_audit_read_connection,
    open_verified_sqlite_read_connection,
    open_verified_sqlite_write_connection,
)
from polylogue.storage.sqlite.literal_cells import LITERAL_CHUNK_BYTES, owned_literal_stream, stream_literal_blob

_FORMAT = "polylogue.audit-continuity-command.v1"
EXCISION_SOURCE_COMMIT_KIND = "record_excision_source_commit"
AUDIT_CONTINUITY_GENESIS_HEAD_SHA256 = "3230fdd585a4fd2d71b7d720bcfe5d697ff120fdb32aecde394e89d407c7198f"
_T = TypeVar("_T")

_COORDINATOR_LOCK_GUARD = threading.Lock()
_COORDINATOR_LOCKS: dict[str, threading.RLock] = {}


def _coordinator_lock(archive_root: Path) -> threading.RLock:
    """Serialize one archive's cross-tier continuity state machine in-process."""

    key = str(archive_root.resolve())
    with _COORDINATOR_LOCK_GUARD:
        lock = _COORDINATOR_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _COORDINATOR_LOCKS[key] = lock
        return lock


class AuditContinuityError(RuntimeError):
    """Audit and source durable control state cannot prove one continuity head."""


class AuditContinuityPendingError(AuditContinuityError):
    """A reader cannot acknowledge an in-flight or unreconciled transition."""


class AuditContinuityUnknownMutationError(AuditContinuityError):
    """A prepared command names a mutation kind this runtime does not declare.

    Only another runtime can have prepared it. Replaying it here would guess
    its effect, and clearing it would drop an effect that may have committed,
    so reconciliation refuses and the command stays pending for that runtime.
    """

    def __init__(self, kind: str) -> None:
        super().__init__(
            f"pending audit continuity command has kind {kind!r}, which this runtime does not declare; "
            "open this archive with the runtime that prepared it"
        )
        self.kind = kind


@dataclass(frozen=True, slots=True)
class AuditMutation:
    """One typed audit command with generated identity and replay inputs."""

    kind: str
    mutation_id: str
    created_at_ms: int
    payload: Mapping[str, object] | CanonicalAuditLiteral

    def command(self) -> dict[str, object]:
        if isinstance(self.payload, CanonicalAuditLiteral):
            raise AuditContinuityError("native command must use the canonical continuity composer")
        return {
            "kind": self.kind,
            "mutation_id": self.mutation_id,
            "created_at_ms": self.created_at_ms,
            "payload": dict(self.payload),
        }

    @property
    def mapping_payload(self) -> Mapping[str, object]:
        if isinstance(self.payload, CanonicalAuditLiteral):
            raise AuditContinuityError("native completion payload cannot enter an ordinary mutation decoder")
        return self.payload

    @classmethod
    def from_command(cls, raw: object) -> AuditMutation:
        if not isinstance(raw, dict):
            raise AuditContinuityError("pending audit continuity command is not an object")
        kind = raw.get("kind")
        mutation_id = raw.get("mutation_id")
        created_at_ms = raw.get("created_at_ms")
        payload = raw.get("payload")
        if (
            not isinstance(kind, str)
            or not kind
            or not isinstance(mutation_id, str)
            or not mutation_id
            or not isinstance(created_at_ms, int)
            or created_at_ms < 0
            or not isinstance(payload, dict)
        ):
            raise AuditContinuityError("pending audit continuity command is malformed")
        return cls(kind=kind, mutation_id=mutation_id, created_at_ms=created_at_ms, payload=payload)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalAuditLiteral:
    """Exact canonical UTF-8 bytes borrowed from their original native owner.

    The factory must check owner currency on every stream acquisition. Closing
    its generator settles its actual readers; this carrier never owns a path.
    """

    byte_length: int
    sha256: str
    chunks: Callable[[], Generator[bytes, None, None]]

    def verified_chunks(self) -> Generator[bytes, None, None]:
        digest = hashlib.sha256()
        length = 0
        with owned_literal_stream(self.chunks()) as stream:
            for chunk in stream:
                check_compute_cancelled()
                if not isinstance(chunk, bytes):
                    raise AuditContinuityError("canonical audit literal requires byte chunks")
                for offset in range(0, len(chunk), LITERAL_CHUNK_BYTES):
                    check_compute_cancelled()
                    part = chunk[offset : offset + LITERAL_CHUNK_BYTES]
                    length += len(part)
                    if length > self.byte_length:
                        raise AuditContinuityError("canonical audit literal exceeds its declared length")
                    digest.update(part)
                    yield part

        if length != self.byte_length or digest.hexdigest() != self.sha256:
            raise AuditContinuityError("canonical audit literal checksum or length mismatch")


def _literal_from_bytes(value: bytes) -> CanonicalAuditLiteral:
    def chunks() -> Generator[bytes, None, None]:
        for offset in range(0, len(value), LITERAL_CHUNK_BYTES):
            yield value[offset : offset + LITERAL_CHUNK_BYTES]

    return CanonicalAuditLiteral(len(value), hashlib.sha256(value).hexdigest(), chunks)


def _compose_literal(parts: tuple[bytes | CanonicalAuditLiteral, ...]) -> CanonicalAuditLiteral:
    def chunks() -> Generator[bytes, None, None]:
        for part in parts:
            if isinstance(part, bytes):
                yield part
            else:
                yield from part.verified_chunks()

    digest = hashlib.sha256()
    length = 0
    with owned_literal_stream(chunks()) as stream:
        for chunk in stream:
            length += len(chunk)
            digest.update(chunk)
    return CanonicalAuditLiteral(length, digest.hexdigest(), chunks)


def _slice_literal(literal: CanonicalAuditLiteral, start: int, length: int) -> CanonicalAuditLiteral:
    def chunks() -> Generator[bytes, None, None]:
        offset = 0
        with owned_literal_stream(literal.verified_chunks()) as stream:
            for chunk in stream:
                end = offset + len(chunk)
                if end > start and offset < start + length:
                    yield chunk[max(0, start - offset) : min(len(chunk), start + length - offset)]
                offset = end

    digest = hashlib.sha256()
    with owned_literal_stream(chunks()) as stream:
        for chunk in stream:
            digest.update(chunk)
    return CanonicalAuditLiteral(length, digest.hexdigest(), chunks)


def write_canonical_audit_literal(
    connection: sqlite3.Connection, table: str, column: str, rowid: int, literal: CanonicalAuditLiteral
) -> None:
    """Write one TEXT value through the exact existing native SQL creator."""
    from polylogue.storage.sqlite.literal_cells import SQLiteLiteralWriteError, write_literal_text

    if (table, column) not in {
        ("operation_events", "detail_json"),
        ("audit_continuity_control", "pending_payload_json"),
    }:
        raise AuditContinuityError("canonical audit literal has no declared destination")
    try:
        write_literal_text(
            connection, table, column, rowid, byte_length=literal.byte_length, chunks=literal.verified_chunks
        )
    except SQLiteLiteralWriteError as error:
        raise AuditContinuityError("canonical audit " + str(error)) from error


_SOURCE_COUNT_KEYS = tuple(
    sorted(
        (
            "source_blob_refs",
            "source_raw_rows",
            "source_raw_existence_changes",
            "source_hook_events",
            "source_fact_rows",
            "source_sidecar_rows",
            "source_container_members",
            "source_container_items",
            "source_materials",
            "source_marker_inputs_pending",
            "source_marker_inputs_accepted",
            "source_publication_reservations",
        )
    )
)


class ExcisionSourceReceiptVisitor(Protocol):
    """Stage bounded completion fields on the caller's existing native owner."""

    def begin_embedding_intent(self) -> None: ...
    def embedding_incarnation(self, value: tuple[int, int] | None) -> None: ...
    def embedding_namespace_header(self, value: tuple[int, int, int] | None) -> None: ...
    def embedding_namespace_link_chunk(self, chunk: bytes) -> None: ...
    def embedding_output(self, meta_present: bool, retire: bool, vector_hash: str, vector_present: bool) -> None: ...
    def embedding_presence(self, present: bool) -> None: ...
    def begin_embedding_row(self, ordinal: int) -> None: ...
    def begin_embedding_cell(self, byte_length: int) -> None: ...
    def embedding_literal_hex_chunk(self, chunk: bytes) -> None: ...
    def embedding_cell_storage_class(self, storage_class: str) -> None: ...
    def end_embedding_cell(self) -> None: ...
    def embedding_row_identity(self, table: str, row_address: int | str) -> None: ...
    def end_embedding_row(self) -> None: ...
    def embedding_schema_version(self, version: int | None) -> None: ...
    def end_embedding_intent(self) -> None: ...
    def begin_target(self, ordinal: int) -> None: ...
    def source_count(self, key: str, value: int) -> None: ...
    def blob_hash(self, value: str, *, removed: bool) -> None: ...
    def session_literal_chunk(self, chunk: bytes) -> None: ...
    def end_target(self) -> None: ...


class _ExcisionCanonicalReader:
    """Read only the declared completion grammar, retaining no receipt arrays.

    Ordinary commands retain their existing domain decoder. This reader is
    deliberately restricted to the completion envelope and its fixed fields.
    """

    def __init__(self, stream: Generator[bytes, None, None]) -> None:
        self.stream = stream
        self.chunk = b""
        self.index = 0
        self.offset = 0
        self.embeddings_range: tuple[int, int] | None = None

    def peek(self) -> int | None:
        while self.index == len(self.chunk):
            self.chunk = next(self.stream, b"")
            self.index = 0
            if not self.chunk:
                return None
        return self.chunk[self.index]

    def take(self, token: bytes) -> None:
        for byte in token:
            if self.peek() != byte:
                raise AuditContinuityError("excision continuity literal is malformed or noncanonical")
            self.index += 1
            self.offset += 1

    def number(self, *, maximum: int = (1 << 63) - 1) -> int:
        first = self.peek()
        value = 0
        count = 0
        while (byte := self.peek()) is not None and 48 <= byte <= 57:
            value = value * 10 + byte - 48
            count += 1
            self.take(bytes((byte,)))
            if value > maximum:
                raise AuditContinuityError("excision continuity integer exceeds its declared physical representation")
        if not count or (count > 1 and first == 48):
            raise AuditContinuityError("excision continuity integer is noncanonical")
        return value

    def generated_identity(self, prefix: str) -> str:
        # AuditRepository creates these identities from token_urlsafe(18):
        # exactly 24 unescaped ASCII characters, independent of input size.
        self.take(b'"' + prefix.encode("ascii"))
        token = bytearray()
        for _ in range(24):
            byte = self.peek()
            if byte is None or byte not in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-":
                raise AuditContinuityError("excision continuity generated identity is malformed")
            token.append(byte)
            self.take(bytes((byte,)))
        self.take(b'"')
        return prefix + token.decode("ascii")

    def hash_value(self) -> str:
        self.take(b'"')
        token = bytearray()
        for _ in range(64):
            byte = self.peek()
            if byte is None or byte not in b"0123456789abcdef":
                raise AuditContinuityError("excision continuity hash is malformed")
            token.append(byte)
            self.take(bytes((byte,)))
        self.take(b'"')
        return token.decode("ascii")

    def text(self, *, emit: Callable[[bytes], None] | None = None) -> None:
        self.take(b'"')
        if emit is not None:
            emit(b'"')
        decoder = codecs.getincrementaldecoder("utf-8")()
        while True:
            byte = self.peek()
            if byte is None:
                raise AuditContinuityError("excision continuity string ended early")
            if byte == 34:
                self.take(b'"')
                decoder.decode(b"", final=True)
                if emit is not None:
                    emit(b'"')
                break
            if byte == 92:
                self.take(b"\\")
                escape = self.peek()
                if escape is not None and escape in b'"\\bfnrt':
                    token = b"\\" + bytes((escape,))
                    self.take(token[1:])
                elif escape == 117:
                    token = b"\\u"
                    self.take(b"u")
                    for _ in range(4):
                        digit = self.peek()
                        if digit is None or digit not in b"0123456789abcdef":
                            raise AuditContinuityError("excision continuity escape is noncanonical")
                        token += bytes((digit,))
                        self.take(bytes((digit,)))
                    value = int(token[2:], 16)
                    if value >= 32 or value in (8, 9, 10, 12, 13):
                        raise AuditContinuityError("excision continuity escape is noncanonical")
                else:
                    raise AuditContinuityError("excision continuity escape is invalid")
                decoder.decode(b"", final=True)
                decoder = codecs.getincrementaldecoder("utf-8")()
                if emit is not None:
                    emit(token)
                continue
            # Consume a complete native chunk run, checking UTF-8 without
            # retaining even one complete session identifier in Python.
            end = self.index
            while end < len(self.chunk) and self.chunk[end] not in (34, 92) and self.chunk[end] >= 32:
                end += 1
            if end == self.index:
                raise AuditContinuityError("excision continuity string has a control byte")
            part = self.chunk[self.index : end]
            decoder.decode(part)
            if emit is not None:
                emit(part)
            self.offset += len(part)
            self.index = end

    def hashes(self, visitor: ExcisionSourceReceiptVisitor | None = None, *, removed: bool = False) -> None:
        self.take(b"[")
        previous = ""
        if self.peek() != 93:
            while True:
                value = self.hash_value()
                if value <= previous:
                    raise AuditContinuityError("excision receipt hashes are not canonical sorted unique hashes")
                previous = value
                if visitor is not None:
                    visitor.blob_hash(value, removed=removed)
                if self.peek() != 44:
                    break
                self.take(b",")
        self.take(b"]")

    def boolean(self) -> bool:
        if self.peek() == 116:
            self.take(b"true")
            return True
        self.take(b"false")
        return False

    def signed_number(self) -> int:
        if self.peek() != 45:
            return self.number()
        self.take(b"-")
        value = self.number(maximum=1 << 63)
        if value == 0:
            raise AuditContinuityError("excision intent rowid is noncanonical negative zero")
        return -value

    def declared_text(self, values: tuple[str, ...]) -> str:
        self.take(b'"')
        choices = tuple(value.encode("ascii") for value in values)
        token = b""
        while self.peek() != 34:
            byte = self.peek()
            if byte is None:
                raise AuditContinuityError("excision intent declaration ended early")
            token += bytes((byte,))
            if not any(value.startswith(token) for value in choices):
                raise AuditContinuityError("excision intent names an undeclared literal shape")
            self.take(bytes((byte,)))
        self.take(b'"')
        if token not in choices:
            raise AuditContinuityError("excision intent names an incomplete declaration")
        return token.decode("ascii")

    def literal_hex(
        self, byte_length: int, visitor: ExcisionSourceReceiptVisitor | None, *, retain_key: bool = False
    ) -> bytes | None:
        self.take(b'"')
        remaining = 2 * byte_length
        key = bytearray() if retain_key and byte_length == 64 else None
        while remaining:
            if self.peek() is None:
                raise AuditContinuityError("excision intent literal ended early")
            count = min(remaining, len(self.chunk) - self.index)
            part = self.chunk[self.index : self.index + count]
            if any(byte not in b"0123456789abcdef" for byte in part):
                raise AuditContinuityError("excision intent literal is not exact lowercase hex")
            self.index += count
            self.offset += count
            remaining -= count
            if key is not None:
                key.extend(part)
            if visitor is not None:
                visitor.embedding_literal_hex_chunk(part)
        self.take(b'"')
        return None if key is None else bytes(key)

    def embedding_intent(self, visitor: ExcisionSourceReceiptVisitor | None) -> None:
        """Stream the six declared relations' original cell bytes as intent."""
        from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import EMBEDDINGS_TABLE_SPECS
        from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION

        tables = (
            "embedding_derivation_state",
            "embedding_failures",
            "embedding_status",
            "message_embedding_refs",
            "message_embeddings",
            "message_embeddings_meta",
        )
        cell_counts = {
            table: 3 if table == "message_embeddings" else len(EMBEDDINGS_TABLE_SPECS[table].all_columns)
            for table in tables
        }
        if visitor is not None:
            visitor.begin_embedding_intent()
        self.take(b'{"incarnation":')
        incarnation = None
        if self.peek() == 110:
            self.take(b"null")
        else:
            self.take(b"[")
            dev = self.number(maximum=(1 << 64) - 1)
            self.take(b",")
            ino = self.number(maximum=(1 << 64) - 1)
            self.take(b"]")
            incarnation = (dev, ino)
        if visitor is not None:
            visitor.embedding_incarnation(incarnation)
        self.take(b',"namespace":')
        namespace = None
        if self.peek() == 110:
            self.take(b"null")
            if visitor is not None:
                visitor.embedding_namespace_header(None)
        else:
            self.take(b"[")
            dev = self.number(maximum=(1 << 64) - 1)
            self.take(b",")
            ino = self.number(maximum=(1 << 64) - 1)
            self.take(b",")
            mode = self.number(maximum=(1 << 32) - 1)
            self.take(b",")
            namespace = (dev, ino, mode)
            if visitor is not None:
                visitor.embedding_namespace_header(namespace)
            if self.peek() == 110:
                self.take(b"null")
                if visitor is not None:
                    visitor.embedding_namespace_link_chunk(b"null")
            else:
                self.text(emit=None if visitor is None else visitor.embedding_namespace_link_chunk)
            self.take(b"]")
        self.take(b',"outputs":[')
        output_count = 0
        previous_hash = ""
        if self.peek() != 93:
            while True:
                self.take(b'{"meta_present":')
                meta_present = self.boolean()
                self.take(b',"retire":')
                retire = self.boolean()
                self.take(b',"vector_derivation_hash":')
                vector_hash = self.hash_value()
                self.take(b',"vector_present":')
                vector_present = self.boolean()
                self.take(b"}")
                if vector_hash <= previous_hash:
                    raise AuditContinuityError("excision intent outputs are not sorted unique identities")
                previous_hash = vector_hash
                if visitor is not None:
                    visitor.embedding_output(meta_present, retire, vector_hash, vector_present)
                output_count += 1
                if self.peek() != 44:
                    break
                self.take(b",")
        self.take(b'],"present":')
        present = self.boolean()
        if visitor is not None:
            visitor.embedding_presence(present)
        self.take(b',"rows":[')
        row_count = 0
        previous_row: tuple[str, int | str] | None = None
        if self.peek() != 93:
            while True:
                if visitor is not None:
                    visitor.begin_embedding_row(row_count)
                self.take(b'{"cells":[')
                cells: list[tuple[str, int]] = []
                first_literal: bytes | None = None
                if self.peek() != 93:
                    while True:
                        self.take(b'{"byte_length":')
                        length = self.number()
                        if visitor is not None:
                            visitor.begin_embedding_cell(length)
                        self.take(b',"literal_hex":')
                        retained = self.literal_hex(length, visitor, retain_key=not cells)
                        if not cells:
                            first_literal = retained
                        self.take(b',"storage_class":')
                        kind = self.declared_text(("null", "integer", "real", "text", "blob"))
                        self.take(b"}")
                        if kind == "null" and length != 0 or kind in {"integer", "real"} and length != 8:
                            raise AuditContinuityError("excision intent fixed cell has a wrong byte length")
                        cells.append((kind, length))
                        if len(cells) > max(cell_counts.values()):
                            raise AuditContinuityError("excision intent row exceeds every declared relation shape")
                        if visitor is not None:
                            visitor.embedding_cell_storage_class(kind)
                            visitor.end_embedding_cell()
                        if self.peek() != 44:
                            break
                        self.take(b",")
                self.take(b'],"row_address":{')
                if self.peek() != 34:
                    raise AuditContinuityError("excision intent row address is malformed")
                address_kind = self.declared_text(("physical_rowid", "vector_derivation_hash"))
                self.take(b":")
                address: int | str = self.signed_number() if address_kind == "physical_rowid" else self.hash_value()
                self.take(b'},"table":')
                table = self.declared_text(tables)
                self.take(b"}")
                if (table == "message_embeddings") != (address_kind == "vector_derivation_hash"):
                    raise AuditContinuityError("excision intent row address differs from its actual relation")
                if table == "message_embeddings" and (
                    not cells
                    or cells[0] != ("text", 64)
                    or first_literal != str(address).encode("ascii").hex().encode("ascii")
                ):
                    raise AuditContinuityError(
                        "excision intent vector address differs from its exact original key cell"
                    )
                out_of_order = previous_row is not None and table < previous_row[0]
                if previous_row is not None and table == previous_row[0]:
                    # The declared table fixes the address branch on both rows.
                    out_of_order = (
                        cast(str, address) <= cast(str, previous_row[1])
                        if table == "message_embeddings"
                        else cast(int, address) <= cast(int, previous_row[1])
                    )
                if len(cells) != cell_counts[table] or out_of_order:
                    raise AuditContinuityError("excision intent relation rows have a wrong shape or order")
                previous_row = (table, address)
                if visitor is not None:
                    visitor.embedding_row_identity(table, address)
                    visitor.end_embedding_row()
                row_count += 1
                if self.peek() != 44:
                    break
                self.take(b",")
        self.take(b'],"schema_version":')
        version = None
        if self.peek() == 110:
            self.take(b"null")
        else:
            version = self.number()
        self.take(b"}")
        if present:
            if incarnation is None or namespace is None or version != EMBEDDINGS_SCHEMA_VERSION:
                raise AuditContinuityError("present excision Embeddings intent lacks its original tier identity")
        elif incarnation is not None or namespace is not None or version is not None or output_count or row_count:
            raise AuditContinuityError("absent excision Embeddings intent carries invented tier evidence")
        if visitor is not None:
            visitor.embedding_schema_version(version)
            visitor.end_embedding_intent()

    def payload(self, visitor: ExcisionSourceReceiptVisitor | None = None) -> tuple[str, str, str]:
        self.take(b'{"attempt_id":')
        attempt_id = self.generated_identity("attempt:")
        self.take(b',"embeddings_intent":')
        intent_start = self.offset
        self.embedding_intent(visitor)
        self.embeddings_range = (intent_start, self.offset)
        self.take(b',"operation_id":')
        operation_id = self.generated_identity("operation:")
        self.take(b',"plan_hash":')
        plan_hash = self.hash_value()
        if (
            not attempt_id
            or not operation_id
            or len(plan_hash) != 64
            or any(c not in "0123456789abcdef" for c in plan_hash)
        ):
            raise AuditContinuityError("excision Source completion has malformed attempt identity")
        self.take(b',"targets":[')
        target_ordinal = 0
        if self.peek() != 93:
            while True:
                if visitor is not None:
                    visitor.begin_target(target_ordinal)
                self.take(b'{"counts":{')
                for ordinal, key in enumerate(_SOURCE_COUNT_KEYS):
                    self.take((b"," if ordinal else b"") + _canonical_json(key).encode() + b":")
                    count = self.number()
                    if visitor is not None:
                        visitor.source_count(key, count)
                self.take(b'},"removed_blob_hashes":')
                self.hashes(visitor, removed=True)
                self.take(b',"session_id":')
                self.text(emit=None if visitor is None else visitor.session_literal_chunk)
                self.take(b',"shared_blob_hashes":')
                self.hashes(visitor, removed=False)
                self.take(b"}")
                if visitor is not None:
                    visitor.end_target()
                target_ordinal += 1
                if self.peek() != 44:
                    break
                self.take(b",")
        self.take(b"]}")
        return operation_id, attempt_id, plan_hash


def scan_excision_source_completion(
    literal: CanonicalAuditLiteral, visitor: ExcisionSourceReceiptVisitor | None = None
) -> tuple[str, str, str]:
    """Visit one complete canonical receipt, yielding only staged observations.

    A visitor must keep its callbacks disposable until this function returns:
    validation/checksum/owner failures invalidate every partially visited row.
    Session chunks are the exact canonical JSON string, including its quotes;
    native SQL can decode it without hydrating an identifier in Python.
    """
    try:
        with owned_literal_stream(literal.verified_chunks()) as stream:
            reader = _ExcisionCanonicalReader(stream)
            identity = reader.payload(visitor)
            if reader.peek() is not None:
                raise AuditContinuityError("excision continuity payload has trailing bytes")
            return identity
    except UnicodeError as error:
        raise AuditContinuityError("excision continuity payload has invalid UTF-8") from error


def excision_embeddings_intent_literal(literal: CanonicalAuditLiteral) -> CanonicalAuditLiteral:
    """Borrow exact intent bytes from the same fully validated canonical receipt."""
    try:
        with owned_literal_stream(literal.verified_chunks()) as stream:
            reader = _ExcisionCanonicalReader(stream)
            reader.payload()
            if reader.peek() is not None or reader.embeddings_range is None:
                raise AuditContinuityError("excision completion lacks its exact canonical intent range")
            start, end = reader.embeddings_range
    except UnicodeError as error:
        raise AuditContinuityError("excision continuity payload has invalid UTF-8") from error

    def chunks() -> Generator[bytes, None, None]:
        offset = 0
        with owned_literal_stream(literal.verified_chunks()) as stream:
            for chunk in stream:
                stop = offset + len(chunk)
                if stop > start and offset < end:
                    yield chunk[max(0, start - offset) : min(len(chunk), end - offset)]
                offset = stop
        # Continue consuming the enclosing original stream even after this
        # range ends, retaining its checksum, currency and close obligations.

    digest = hashlib.sha256()
    with owned_literal_stream(chunks()) as stream:
        for chunk in stream:
            digest.update(chunk)
    return CanonicalAuditLiteral(end - start, digest.hexdigest(), chunks)


def excision_completion_identity(literal: CanonicalAuditLiteral) -> tuple[str, str, str]:
    return scan_excision_source_completion(literal)


@dataclass(frozen=True, slots=True)
class PreparedAuditContinuityCommand:
    mutation: AuditMutation
    prior_generation: int
    prior_head_sha256: str
    next_generation: int
    next_head_sha256: str
    command_sha256: str
    literal: CanonicalAuditLiteral
    source_currency_check: Callable[[], None] | None = None

    @property
    def byte_length(self) -> int:
        return self.literal.byte_length

    @property
    def sha256(self) -> str:
        return self.literal.sha256

    def chunks(self) -> Generator[bytes, None, None]:
        yield from self.literal.verified_chunks()


def prepared_audit_continuity_command(
    mutation: AuditMutation, *, prior_generation: int, prior_head_sha256: str
) -> PreparedAuditContinuityCommand:
    """Compose the sole canonical Source command from scalar or native operands."""
    if (
        type(mutation.created_at_ms) is not int
        or not 0 <= mutation.created_at_ms <= 9223372036854775807
        or type(prior_generation) is not int
        or not 0 <= prior_generation < 9223372036854775807
        or not mutation.kind
        or not mutation.mutation_id
        or len(prior_head_sha256) != 64
        or any(c not in "0123456789abcdef" for c in prior_head_sha256)
    ):
        raise AuditContinuityError("audit continuity command has malformed identity or head")
    payload = mutation.payload
    if isinstance(payload, CanonicalAuditLiteral):
        if mutation.kind != EXCISION_SOURCE_COMMIT_KIND:
            raise AuditContinuityError("native audit payload has no declared domain decoder")
        _operation, attempt, _plan = excision_completion_identity(payload)
        if mutation.mutation_id != f"excision-source:{attempt}":
            raise AuditContinuityError("excision command differs from its completion attempt")
    else:
        if mutation.kind == EXCISION_SOURCE_COMMIT_KIND:
            raise AuditContinuityError("excision completion requires its original native literal")
        payload = _literal_from_bytes(_canonical_json(dict(payload)).encode("utf-8"))
    command = _compose_literal(
        (
            b'{"created_at_ms":' + _canonical_json(mutation.created_at_ms).encode(),
            b',"kind":' + _canonical_json(mutation.kind).encode("utf-8"),
            b',"mutation_id":' + _canonical_json(mutation.mutation_id).encode("utf-8"),
            b',"payload":',
            payload,
            b"}",
        )
    )
    next_head = _sha256({"previous_head_sha256": prior_head_sha256, "command_sha256": command.sha256})
    literal = _compose_literal(
        (
            b'{"command":',
            command,
            b',"command_sha256":' + _canonical_json(command.sha256).encode(),
            b',"format":' + _canonical_json(_FORMAT).encode(),
            b',"next_generation":' + _canonical_json(prior_generation + 1).encode(),
            b',"next_head_sha256":' + _canonical_json(next_head).encode(),
            b',"prior_generation":' + _canonical_json(prior_generation).encode(),
            b',"prior_head_sha256":' + _canonical_json(prior_head_sha256).encode(),
            b"}",
        )
    )
    return PreparedAuditContinuityCommand(
        mutation, prior_generation, prior_head_sha256, prior_generation + 1, next_head, command.sha256, literal
    )


@contextmanager
def _open_source_read_connection(path: Path) -> Iterator[sqlite3.Connection]:
    try:
        with open_verified_sqlite_read_connection(path) as connection:
            yield connection
    except AuditLeafError as exc:
        raise AuditContinuityError(f"cannot safely read source continuity tier: {path}: {exc}") from exc


@contextmanager
def _open_source_write_connection(path: Path) -> Iterator[sqlite3.Connection]:
    try:
        with open_verified_sqlite_write_connection(path) as connection:
            yield connection
    except AuditLeafError as exc:
        raise AuditContinuityError(f"cannot safely write source continuity tier: {path}: {exc}") from exc


def _entry_is_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise AuditContinuityError(f"cannot inspect audit continuity tier entry: {path}") from exc
    return False


class AuditContinuityCoordinator:
    """Coordinate typed audit commands through source.db's durable WAL row."""

    def __init__(
        self,
        archive_root: Path,
        *,
        phase_hook: Callable[[str, AuditMutation], None] | None = None,
    ) -> None:
        self.archive_root = archive_root.resolve()
        self.source_path = self.archive_root / "source.db"
        self.audit_path = self.archive_root / "audit.db"
        self._phase_hook = phase_hook
        self._execution_lock = _coordinator_lock(self.archive_root)

    def execute(self, mutation: AuditMutation, apply: Callable[[sqlite3.Connection, AuditMutation], _T]) -> _T:
        """Prepare one command, commit audit bytes, then promote source control."""

        with self._execution_lock:
            return self._execute_serialized(mutation, apply)

    def _execute_serialized(
        self, mutation: AuditMutation, apply: Callable[[sqlite3.Connection, AuditMutation], _T]
    ) -> _T:
        """Run the complete source-to-audit transition under one archive lock."""

        self._phase("before_source_prepare", mutation)
        prepared = self._prepare(mutation)
        self._phase("after_source_prepare", mutation)
        try:
            result = self._apply_prepared(prepared, apply)
        except Exception:
            # _apply_prepared has exited its audit transaction before this
            # handler runs. Clear this exact source WAL entry only when the
            # audit head still proves no commit happened, so validation rejects
            # cannot wedge every later audit mutation.
            self._abort_prepared(prepared)
            raise
        self._phase("after_audit_commit", mutation)
        self._promote(prepared)
        self._phase("after_source_promotion", mutation)
        return result

    def reconcile(self, apply: Callable[[sqlite3.Connection, AuditMutation], object]) -> None:
        """Deterministically complete a pending command or reject a stale audit image."""

        with self._execution_lock:
            self._reconcile_serialized(apply)

    @contextmanager
    def settled_read(self, *, wait_for_lock: bool = False) -> Iterator[dict[str, int]]:
        """Observe audit receipts only after the source control head agrees.

        Completion controls refuse transient contention. Independent recovery
        discovery may wait for this exact lock, checking the original compute
        cancellation while it waits. A persisted pending head still refuses.
        """
        if wait_for_lock:
            # This interval is a cancellation wake boundary, not an outcome
            # deadline: slow valid continuity work remains serviceable.
            while True:
                check_compute_cancelled()
                if self._execution_lock.acquire(timeout=0.05):
                    break
        elif not self._execution_lock.acquire(blocking=False):
            raise AuditContinuityPendingError("audit continuity transition is in flight")
        try:
            if wait_for_lock:
                check_compute_cancelled()
            if not self.is_available():
                raise AuditContinuityPendingError("audit continuity requires owner reconciliation")
            if self._has_pending():
                raise AuditContinuityPendingError("audit continuity requires owner reconciliation")
            self._assert_committed_head_matches_audit()
            with (
                _open_source_read_connection(self.source_path) as source,
                open_verified_audit_read_connection(self.audit_path) as audit,
            ):
                yield {
                    "source": int(source.execute("PRAGMA user_version").fetchone()[0]),
                    "audit": int(audit.execute("PRAGMA user_version").fetchone()[0]),
                }
        finally:
            self._execution_lock.release()

    def _reconcile_serialized(self, apply: Callable[[sqlite3.Connection, AuditMutation], object]) -> None:
        """Recover continuity without racing an in-flight writer."""

        if not self.is_available():
            return
        with self._pending() as prepared:
            if prepared is None:
                self._assert_committed_head_matches_audit()
                return
            self._apply_prepared(prepared, apply)
        # No variable literal is needed for promotion. Close the Source
        # snapshot first, including in archives using rollback journals.
        self._promote(prepared)

    def is_available(self) -> bool:
        """Return whether both continuity tiers exist; a present tier lacking its half is damage."""

        if _entry_is_absent(self.source_path) or _entry_is_absent(self.audit_path):
            return False
        try:
            with (
                _open_source_read_connection(self.source_path) as source,
                open_verified_audit_read_connection(self.audit_path) as audit,
            ):
                # Every durable tier this runtime can open was created with its
                # continuity half (fresh v1 has no pre-continuity schema), so a
                # present tier without it is damage, never a standby window
                # (polylogue-h6yuj).
                if not self._has_table(source, "audit_continuity_control"):
                    raise AuditContinuityError("current source schema is missing audit continuity control")
                if not self._has_table(audit, "audit_continuity_head"):
                    raise AuditContinuityError("current audit schema is missing audit continuity head")
                source.execute("SELECT 1 FROM audit_continuity_control WHERE singleton = 1").fetchone()
                audit.execute("SELECT 1 FROM audit_continuity_head WHERE singleton = 1").fetchone()
                if self._is_populated_genesis_audit(source, audit):
                    # Every audit write advances the head, so a populated
                    # journal still at genesis was written around continuity.
                    raise AuditContinuityError("populated audit journal has only genesis continuity heads")
        except AuditLeafError as exc:
            raise AuditContinuityError(str(exc)) from exc
        except sqlite3.OperationalError as exc:
            raise AuditContinuityError("cannot inspect audit continuity compatibility state") from exc
        except sqlite3.DatabaseError as exc:
            raise AuditContinuityError("cannot inspect audit continuity compatibility state") from exc
        return True

    def runtime_probe(self) -> str:
        """Exercise the coordinator's released-schema or compatibility state."""

        if not self.is_available():
            return "standby until source.db and audit.db both install continuity control"
        if self._has_pending():
            raise AuditContinuityError("runtime probe found an unreconciled audit continuity command")
        self._assert_committed_head_matches_audit()
        return "reconciled matching source/audit continuity heads"

    def _phase(self, name: str, mutation: AuditMutation) -> None:
        if self._phase_hook is not None:
            self._phase_hook(name, mutation)

    def _prepare(self, mutation: AuditMutation) -> PreparedAuditContinuityCommand:
        if mutation.kind == EXCISION_SOURCE_COMMIT_KIND:
            raise AuditContinuityError("excision completion must commit atomically with its original Source effects")
        self._require_paths()
        with _open_source_write_connection(self.source_path) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT committed_generation, committed_head_sha256, pending_mutation_id FROM audit_continuity_control WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise AuditContinuityError("source audit continuity control is missing")
            if row[2] is not None:
                raise AuditContinuityError("another audit continuity mutation is already pending")
            prepared = prepared_audit_continuity_command(
                mutation, prior_generation=int(row[0]), prior_head_sha256=str(row[1])
            )
            if mutation.kind == "accept_ingest":
                # Prove the manifest's retained inputs here, but publish it in
                # _promote. A prepare that writes durable source rows cannot be
                # rolled back by _abort_prepared, which is what made a failed
                # accept_ingest wedge every later audit mutation and read.
                from polylogue.storage.sqlite.archive_tiers.source_items import (
                    FrozenSourceManifest,
                    source_manifest_from_dict,
                    validate_frozen_source_manifest,
                    validate_sealed_source_manifest,
                )

                manifest = source_manifest_from_dict(cast(Mapping[str, object], mutation.payload).get("manifest"))
                if isinstance(manifest, FrozenSourceManifest):
                    validate_frozen_source_manifest(conn, manifest)
                else:
                    validate_sealed_source_manifest(conn, manifest)
            conn.execute(
                """
                UPDATE audit_continuity_control
                SET pending_mutation_id = ?, pending_payload_json = ?, pending_payload_sha256 = ?, prepared_at_ms = ?
                WHERE singleton = 1 AND pending_mutation_id IS NULL
                """,
                (mutation.mutation_id, "{}", prepared.sha256, mutation.created_at_ms),
            )
            with connection_cursor(conn, "SELECT rowid FROM audit_continuity_control WHERE singleton=1") as cursor:
                rowid = int(cursor.fetchone()[0])
            write_canonical_audit_literal(
                conn, "audit_continuity_control", "pending_payload_json", rowid, prepared.literal
            )
            conn.commit()
        return prepared

    def _has_pending(self) -> bool:
        with (
            _open_source_read_connection(self.source_path) as conn,
            connection_cursor(
                conn, "SELECT pending_mutation_id FROM audit_continuity_control WHERE singleton=1"
            ) as cursor,
        ):
            row = cursor.fetchone()
        if row is None:
            raise AuditContinuityError("source audit continuity control is missing")
        return row[0] is not None

    @contextmanager
    def _pending(self) -> Iterator[PreparedAuditContinuityCommand | None]:
        """Keep the original pending literal's read snapshot alive through replay."""
        self._require_paths()
        with _open_source_read_connection(self.source_path) as conn:
            conn.execute("BEGIN")
            with connection_cursor(
                conn,
                "SELECT rowid,committed_generation,committed_head_sha256,pending_mutation_id,"
                "pending_payload_sha256,typeof(pending_payload_json),length(CAST(pending_payload_json AS BLOB)),"
                "CASE WHEN json_valid(pending_payload_json) THEN json_extract(pending_payload_json,'$.command.kind') END "
                "FROM audit_continuity_control WHERE singleton=1",
            ) as cursor:
                row = cursor.fetchone()
            if row is None:
                raise AuditContinuityError("source audit continuity control is missing")
            if row[3] is None:
                yield None
                return
            if row[5] != "text":
                raise AuditContinuityError("source audit continuity pending command is malformed")
            from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_on_current_thread

            owner = next(
                (owner for owner in retained_native_sql_owners_on_current_thread() if owner.connection is conn), None
            )
            if owner is None:
                raise AuditContinuityError("pending command lacks its original native Source owner")
            active = True

            def require_current_source() -> None:
                if not active:
                    raise AuditContinuityError("pending continuity literal outlived its original snapshot")
                if owner.require_connection() is not conn or owner.leaf is None:
                    raise AuditContinuityError("pending continuity lost its original Source namespace owner")
                try:
                    owner.leaf.assert_unchanged()
                except AuditLeafError as error:
                    raise AuditContinuityError("pending continuity Source namespace changed") from error

            def chunks() -> Generator[bytes, None, None]:
                require_current_source()
                with owner.readonly_blob("audit_continuity_control", "pending_payload_json", int(row[0])) as blob:
                    yield from stream_literal_blob(blob, int(row[6]), check_compute_cancelled)
                require_current_source()

            literal = CanonicalAuditLiteral(int(row[6]), str(row[4]), chunks)
            try:
                if row[7] is None:
                    raise AuditContinuityError("source audit continuity pending command is invalid JSON")
                if row[7] == EXCISION_SOURCE_COMMIT_KIND:
                    prepared = self._read_excision_prepared(literal)
                else:
                    # Ordinary domain commands already own a Python payload.
                    # The unbounded completion receipt never enters this path.
                    with owned_literal_stream(literal.verified_chunks()) as stream:
                        for _chunk in stream:
                            pass
                    with connection_cursor(
                        conn, "SELECT pending_payload_json FROM audit_continuity_control WHERE singleton=1"
                    ) as cursor:
                        raw = cursor.fetchone()
                    try:
                        value = json.loads(raw[0])
                        mutation = AuditMutation.from_command(value["command"])
                        prepared = prepared_audit_continuity_command(
                            mutation,
                            prior_generation=value["prior_generation"],
                            prior_head_sha256=value["prior_head_sha256"],
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        raise AuditContinuityError("source audit continuity pending command is invalid JSON") from error
                    if _sha256(value) != literal.sha256 or prepared.sha256 != literal.sha256:
                        raise AuditContinuityError("source audit continuity pending command checksum mismatch")
                if (prepared.prior_generation, prepared.prior_head_sha256, prepared.mutation.mutation_id) != (
                    row[1],
                    row[2],
                    row[3],
                ):
                    raise AuditContinuityError("source pending command does not bind its committed head and mutation")
                prepared = replace(prepared, source_currency_check=require_current_source)
                self._validate_prepared(prepared)
                yield prepared
            finally:
                active = False

    @staticmethod
    def _read_excision_prepared(literal: CanonicalAuditLiteral) -> PreparedAuditContinuityCommand:
        try:
            with owned_literal_stream(literal.verified_chunks()) as stream:
                reader = _ExcisionCanonicalReader(stream)
                reader.take(b'{"command":')
                command_start = reader.offset
                reader.take(b'{"created_at_ms":')
                created = reader.number()
                reader.take(b',"kind":')
                reader.take(_canonical_json(EXCISION_SOURCE_COMMIT_KIND).encode())
                kind = EXCISION_SOURCE_COMMIT_KIND
                reader.take(b',"mutation_id":')
                mutation_id = reader.generated_identity("excision-source:attempt:")
                reader.take(b',"payload":')
                payload_start = reader.offset
                _operation, attempt, _plan = reader.payload()
                payload_length = reader.offset - payload_start
                reader.take(b"}")
                command_length = reader.offset - command_start
                reader.take(b',"command_sha256":')
                command_hash = reader.hash_value()
                reader.take(b',"format":')
                reader.take(_canonical_json(_FORMAT).encode())
                reader.take(b',"next_generation":')
                next_generation = reader.number()
                reader.take(b',"next_head_sha256":')
                next_head = reader.hash_value()
                reader.take(b',"prior_generation":')
                prior_generation = reader.number()
                reader.take(b',"prior_head_sha256":')
                prior_head = reader.hash_value()
                reader.take(b"}")
                if reader.peek() is not None:
                    raise AuditContinuityError("excision continuity command format mismatch")
            if mutation_id != f"excision-source:{attempt}":
                raise AuditContinuityError("excision command differs from its completion attempt")
            command = _slice_literal(literal, command_start, command_length)
            if command.sha256 != command_hash:
                raise AuditContinuityError("audit continuity command payload checksum mismatch")
            payload = _slice_literal(literal, payload_start, payload_length)
            return PreparedAuditContinuityCommand(
                AuditMutation(kind, mutation_id, created, payload),
                prior_generation,
                prior_head,
                next_generation,
                next_head,
                command_hash,
                literal,
            )
        except UnicodeError as error:
            raise AuditContinuityError("excision continuity command has invalid UTF-8") from error

    def _apply_prepared(
        self,
        prepared: PreparedAuditContinuityCommand,
        apply: Callable[[sqlite3.Connection, AuditMutation], _T],
    ) -> _T:
        self._validate_prepared(prepared)
        mutation = prepared.mutation
        with open_verified_audit_connection(self.audit_path) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise AuditContinuityError("audit continuity head is missing")
            current = (int(row[0]), str(row[1]), row[2])
            prior = (prepared.prior_generation, prepared.prior_head_sha256)
            target = (prepared.next_generation, prepared.next_head_sha256)
            if current[:2] == target and current[2] == mutation.mutation_id:
                if prepared.source_currency_check is not None:
                    prepared.source_currency_check()
                conn.commit()
                return cast(_T, None)
            if current[:2] != prior:
                raise AuditContinuityError("audit continuity head does not match the prepared source command")
            result = apply(conn, mutation)
            conn.execute(
                "UPDATE audit_continuity_head SET generation = ?, head_sha256 = ?, mutation_id = ?, advanced_at_ms = ? WHERE singleton = 1",
                (*target, mutation.mutation_id, mutation.created_at_ms),
            )
            # The pending snapshot is still held here. Descriptor lifetime
            # alone cannot authorize publication after its namespace retires.
            if prepared.source_currency_check is not None:
                prepared.source_currency_check()
            conn.commit()
            return result

    def _promote(self, prepared: PreparedAuditContinuityCommand) -> None:
        mutation = prepared.mutation
        with _open_source_write_connection(self.source_path) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if mutation.kind == "accept_ingest":
                # The audit commit that accepts this manifest is durable, so
                # publishing the generation here joins the promotion's own
                # transaction. Republishing an identical manifest is a no-op,
                # which keeps reconcile's replay of this step idempotent.
                from polylogue.storage.sqlite.archive_tiers.source_items import (
                    FrozenSourceManifest,
                    publish_frozen_source_manifest,
                    publish_sealed_source_manifest,
                    source_manifest_from_dict,
                )

                manifest = source_manifest_from_dict(cast(Mapping[str, object], mutation.payload).get("manifest"))
                if isinstance(manifest, FrozenSourceManifest):
                    publish_frozen_source_manifest(conn, manifest, prepared_at_ms=mutation.created_at_ms)
                else:
                    publish_sealed_source_manifest(conn, manifest, prepared_at_ms=mutation.created_at_ms)
            cursor = conn.execute(
                """
                UPDATE audit_continuity_control
                SET committed_generation = ?, committed_head_sha256 = ?,
                    pending_mutation_id = NULL, pending_payload_json = NULL,
                    pending_payload_sha256 = NULL, prepared_at_ms = NULL
                WHERE singleton = 1 AND pending_mutation_id = ? AND pending_payload_sha256 = ?
                """,
                (
                    prepared.next_generation,
                    prepared.next_head_sha256,
                    mutation.mutation_id,
                    prepared.sha256,
                ),
            )
            if cursor.rowcount != 1:
                raise AuditContinuityError("source audit continuity promotion lost its prepared command")
            conn.commit()

    def _abort_prepared(self, prepared: PreparedAuditContinuityCommand) -> None:
        """Discard a rejected WAL command after proving its audit transaction rolled back."""

        mutation = prepared.mutation
        if mutation.kind == EXCISION_SOURCE_COMMIT_KIND:
            raise AuditContinuityError("committed excision Source effects cannot be aborted as an Audit prepare")
        prior = (prepared.prior_generation, prepared.prior_head_sha256)
        target = (prepared.next_generation, prepared.next_head_sha256)
        with open_verified_audit_connection(self.audit_path) as audit:
            audit.execute("BEGIN IMMEDIATE")
            row = audit.execute(
                "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise AuditContinuityError("audit continuity head is missing while aborting a prepared command")
            current = (int(row[0]), str(row[1]), row[2])
            if current[:2] == target and current[2] == mutation.mutation_id:
                # The audit commit did land. Keep the WAL command for normal
                # promotion instead of mistaking an ambiguous failure for rollback.
                return
            if current[:2] != prior:
                raise AuditContinuityError("cannot abort prepared command after an unrelated audit head change")
            with _open_source_write_connection(self.source_path) as source, source:
                source.execute("BEGIN IMMEDIATE")
                cursor = source.execute(
                    """
                UPDATE audit_continuity_control
                SET pending_mutation_id = NULL, pending_payload_json = NULL,
                    pending_payload_sha256 = NULL, prepared_at_ms = NULL
                WHERE singleton = 1 AND committed_generation = ? AND committed_head_sha256 = ?
                  AND pending_mutation_id = ? AND pending_payload_sha256 = ?
                """,
                    (prior[0], prior[1], mutation.mutation_id, prepared.sha256),
                )
                if cursor.rowcount != 1:
                    raise AuditContinuityError("source audit continuity abort lost its prepared command")
                source.commit()

    def _assert_committed_head_matches_audit(self) -> None:
        with (
            _open_source_read_connection(self.source_path) as source,
            open_verified_audit_read_connection(self.audit_path) as audit,
        ):
            source_row = source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control WHERE singleton = 1"
            ).fetchone()
            audit_row = audit.execute(
                "SELECT generation, head_sha256 FROM audit_continuity_head WHERE singleton = 1"
            ).fetchone()
        if source_row is None or audit_row is None:
            raise AuditContinuityError("audit continuity control row is missing")
        if (int(source_row[0]), str(source_row[1])) != (int(audit_row[0]), str(audit_row[1])):
            raise AuditContinuityError("audit continuity head regressed or was replaced after source promotion")

    def _validate_prepared(self, prepared: PreparedAuditContinuityCommand) -> None:
        if prepared.source_currency_check is not None:
            prepared.source_currency_check()
        if type(prepared.prior_generation) is not int or type(prepared.next_generation) is not int:
            raise AuditContinuityError("audit continuity command generations are malformed")
        if prepared.prior_generation < 0 or prepared.next_generation != prepared.prior_generation + 1:
            raise AuditContinuityError("audit continuity command generation is non-monotonic")
        expected = prepared_audit_continuity_command(
            prepared.mutation, prior_generation=prepared.prior_generation, prior_head_sha256=prepared.prior_head_sha256
        )
        if (expected.sha256, expected.command_sha256, expected.next_head_sha256) != (
            prepared.sha256,
            prepared.command_sha256,
            prepared.next_head_sha256,
        ):
            raise AuditContinuityError("audit continuity command checksum mismatch")
        with owned_literal_stream(prepared.chunks()) as stream:
            for _chunk in stream:
                pass

    @staticmethod
    def _has_table(connection: sqlite3.Connection, name: str) -> bool:
        return (
            connection.execute("SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?", (name,)).fetchone()
            is not None
        )

    def _is_populated_genesis_audit(self, source: sqlite3.Connection, audit: sqlite3.Connection) -> bool:
        source_head = source.execute(
            "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control WHERE singleton = 1"
        ).fetchone()
        audit_head = audit.execute(
            "SELECT generation, head_sha256 FROM audit_continuity_head WHERE singleton = 1"
        ).fetchone()
        if source_head != (0, AUDIT_CONTINUITY_GENESIS_HEAD_SHA256) or audit_head != (
            0,
            AUDIT_CONTINUITY_GENESIS_HEAD_SHA256,
        ):
            return False
        tables = tuple(
            str(row[0])
            for row in audit.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "AND name != 'audit_continuity_head' ORDER BY name"
            )
        )
        for name in tables:
            quoted_name = name.replace('"', '""')
            if audit.execute(f'SELECT 1 FROM "{quoted_name}" LIMIT 1').fetchone() is not None:
                return True
        return False

    def _require_paths(self) -> None:
        for path in (self.source_path, self.audit_path):
            try:
                metadata = path.lstat()
            except FileNotFoundError as exc:
                raise AuditContinuityError("audit continuity requires initialized source.db and audit.db") from exc
            except OSError as exc:
                raise AuditContinuityError(f"cannot inspect audit continuity tier entry: {path}") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise AuditContinuityError(f"audit continuity tier entry is not an owned regular file: {path}")


__all__ = [
    "AuditContinuityCoordinator",
    "AuditContinuityError",
    "AuditContinuityUnknownMutationError",
    "AuditMutation",
    "CanonicalAuditLiteral",
    "ExcisionSourceReceiptVisitor",
    "scan_excision_source_completion",
    "PreparedAuditContinuityCommand",
    "excision_completion_identity",
    "prepared_audit_continuity_command",
]

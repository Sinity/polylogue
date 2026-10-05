"""Durable source-generation accounting and idempotent item transitions.

Writer module: source.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import INGEST_OUTCOME_RETRYABLE, IngestOutcome, Origin
from polylogue.core.provider_identity import captured_hermes_profile_key, profile_root_for_artifact
from polylogue.pipeline.ingest_outcomes import bounded_diagnostic
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.connection_profile import scratch_connection_context

from .common import require_vocabulary
from .source_attachments import (
    SourceAttachment,
    _preflight_source_attachments,
    record_source_attachments,
    source_attachment_census,
)

if TYPE_CHECKING:
    from polylogue.security.excision_policy import ExcisionPolicySnapshot


@dataclass(frozen=True, slots=True)
class SourceItemAdmission:
    """Exact frozen-input membership joined to one raw admission."""

    source_generation_id: str
    source_item_id: str
    record_coordinate: str
    entry_ordinal: int | None = None
    split_index: int | None = None
    addressing_mode: str | None = None
    content_identity: str | None = None


class AcquisitionDisposition(StrEnum):
    """Storage-owned source-item state machine, checked by its writer.

    Durable source DDL intentionally does not mirror this vocabulary. Add a
    state only with its transition, retryability, reconciliation, and reader
    behavior reviewed together.
    """

    PENDING = "pending"
    ADMITTED = "admitted"
    NON_SESSION = "non_session"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    CORRUPT = "corrupt"
    UNKNOWN_BLOCKING = "unknown_blocking"


class SourceItemMemberDisposition(StrEnum):
    """Why a central-directory member was not admitted as raw evidence.

    This storage-local audit vocabulary is validated at
    ``record_source_item_member_disposition``. Extend it only when the ZIP
    admission policy and enumeration/reconciliation reader understand the new
    outcome; durable source DDL remains vocabulary-free.
    """

    REFUSED = "refused"
    UNSELECTED = "unselected"


_MAX_MEMBER_DIAGNOSTIC_CHARS = 4080


@dataclass(frozen=True, slots=True)
class SourceItem:
    source_generation_id: str
    source_item_id: str
    logical_coordinate: str
    addressing_mode: str
    disposition: AcquisitionDisposition
    outcome_code: IngestOutcome
    stage: str
    revision: int
    retryable: bool | None
    raw_id: str | None
    blob_hash: bytes | None


@dataclass(frozen=True, slots=True)
class CapturedSourceInputIdentity:
    """Original opened input namespace retained with the exact accepted blob."""

    canonical_source_path: str
    semantic_source_path: str
    profile_root: str
    profile_key: str
    profile_source_path: str

    def __post_init__(self) -> None:
        coordinate_fields: tuple[object, ...] = (
            self.canonical_source_path,
            self.semantic_source_path,
            self.profile_root,
            self.profile_source_path,
        )
        for value in coordinate_fields:
            if (
                not isinstance(value, str)
                or not Path(value).is_absolute()
                or ".." in Path(value).parts
                or "\0" in value
            ):
                raise ValueError("captured source input coordinates must be absolute")
        if captured_hermes_profile_key(Path(self.profile_root)) != self.profile_key:
            raise ValueError("captured source input profile key does not match its accepted namespace")
        try:
            Path(self.profile_source_path).relative_to(self.profile_root)
        except ValueError as exc:
            raise ValueError("captured profile member is outside its accepted namespace") from exc

    def member_profile_identity(self, member_name: str) -> tuple[Path, Path] | None:
        """Interpret a relative member only in this accepted namespace."""
        member = PurePosixPath(member_name)
        if member.is_absolute() or ".." in member.parts:
            return None
        profile_path = Path(self.profile_source_path).parent.joinpath(*member.parts)
        return profile_root_for_artifact(profile_path), profile_path

    def to_dict(self) -> dict[str, str]:
        return {
            "canonical_source_path": self.canonical_source_path,
            "semantic_source_path": self.semantic_source_path,
            "profile_root": self.profile_root,
            "profile_key": self.profile_key,
            "profile_source_path": self.profile_source_path,
        }

    @classmethod
    def from_dict(cls, payload: object) -> CapturedSourceInputIdentity:
        fields = {"canonical_source_path", "semantic_source_path", "profile_root", "profile_key", "profile_source_path"}
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or any(not isinstance(value, str) for value in payload.values())
        ):
            raise ValueError("invalid captured source input identity")
        return cls(**payload)


def _captured_input_json(identity: CapturedSourceInputIdentity | None) -> str | None:
    return json.dumps(identity.to_dict(), ensure_ascii=False, separators=(",", ":")) if identity is not None else None


def _captured_input_from_json(payload: str | None) -> CapturedSourceInputIdentity | None:
    return CapturedSourceInputIdentity.from_dict(json.loads(payload)) if payload is not None else None


@dataclass(frozen=True, slots=True)
class FrozenSourceInput:
    """One retained physical input, never an actuator payload or bearer secret."""

    coordinate: str
    source_path: str
    blob_hash: str
    publication_receipt_id: str
    captured_identity: CapturedSourceInputIdentity | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate": self.coordinate,
            "source_path": self.source_path,
            "blob_hash": self.blob_hash,
            "publication_receipt_id": self.publication_receipt_id,
            "captured_identity": self.captured_identity.to_dict() if self.captured_identity is not None else None,
        }


@dataclass(frozen=True, slots=True)
class RetainedSourceInput:
    """An accepted source reference; publication authority was already consumed."""

    source_item_id: str
    coordinate: str
    source_path: str
    blob_hash: str
    enumeration_complete: bool
    stage: str
    revision: int
    captured_identity: CapturedSourceInputIdentity | None = None


@dataclass(frozen=True, slots=True)
class RetainedSourceGeneration:
    source_generation_id: str
    enumeration_fingerprint: str
    inputs: tuple[RetainedSourceInput, ...]
    item_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceItemMemberDispositionRecord:
    source_generation_id: str
    source_item_id: str
    entry_ordinal: int
    member_name: str
    disposition: SourceItemMemberDisposition
    diagnostic: str
    observed_at_ms: int


def retained_source_generation_header(conn: sqlite3.Connection, source_generation_id: str) -> RetainedSourceGeneration:
    """Read the accepted decoder and count without materializing input members."""
    row = conn.execute(
        "SELECT item_count FROM source_generations WHERE source_generation_id=?", (source_generation_id,)
    ).fetchone()
    if row is None or type(row[0]) is not int or row[0] < 1:
        raise ValueError("accepted source generation is absent or empty")
    first = page_retained_source_inputs(conn, source_generation_id, limit=1)
    if not first:
        raise ValueError("accepted source generation has no retained input")
    fingerprint = first[0][1]
    _require_digest(fingerprint, "enumeration_fingerprint")
    return RetainedSourceGeneration(source_generation_id, fingerprint, (), row[0])


def page_retained_source_inputs(
    conn: sqlite3.Connection, source_generation_id: str, *, after: tuple[str, str] | None = None, limit: int = 256
) -> tuple[tuple[RetainedSourceInput, str], ...]:
    """Read a bounded accepted input page with a stable two-column keyset."""
    if not 1 <= limit <= 256:
        raise ValueError("accepted input page limit must be between 1 and 256")
    rows = conn.execute(
        "SELECT source_item_id, logical_coordinate, source_path, blob_hash, enumeration_fingerprint, "
        "enumerated_at_ms, stage, revision, captured_input_identity FROM source_items WHERE source_generation_id=? "
        "AND (logical_coordinate, source_item_id) > (?, ?) ORDER BY logical_coordinate, source_item_id LIMIT ?",
        (source_generation_id, *(after or ("", "")), limit),
    ).fetchall()
    result: list[tuple[RetainedSourceInput, str]] = []
    for row in rows:
        if row[2] is None or not isinstance(row[3], bytes) or len(row[3]) != 32 or row[4] is None:
            raise ValueError("accepted source input lost its retained physical identity")
        result.append(
            (
                RetainedSourceInput(
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    row[3].hex(),
                    row[5] is not None,
                    str(row[6]),
                    int(row[7]),
                    _captured_input_from_json(row[8]),
                ),
                str(row[4]),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FrozenSourceManifest:
    """Inline evidence for an opened ZIP or an immutable historical command.

    New general acceptance plans use the staged manifest owner and carry only
    its sealed reference. Historical command decoding preserves these exact
    fields and their digest without rediscovering source inputs.
    """

    source_generation_id: str
    enumeration_fingerprint: str
    inputs: tuple[FrozenSourceInput, ...]
    source_name: str | None = None

    def __post_init__(self) -> None:
        _require_digest(self.enumeration_fingerprint, "enumeration_fingerprint")
        if not self.source_generation_id or not self.inputs:
            raise ValueError("frozen manifest requires a generation and nonempty input set")
        if len(self.inputs) > 1:
            with scratch_connection_context(prefix="polylogue-manifest-", filename="coordinates.db") as keys:
                keys.execute("PRAGMA journal_mode=DELETE")
                keys.execute("BEGIN")
                keys.execute("CREATE TABLE coordinates(value TEXT PRIMARY KEY) WITHOUT ROWID")
                for item in self.inputs:
                    try:
                        keys.execute("INSERT INTO coordinates VALUES (?)", (item.coordinate,))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("frozen manifest coordinates must be distinct") from exc
        if self.source_name is not None and (not self.source_name.strip()):
            raise ValueError("frozen source name must be nonempty and bounded")
        for item in self.inputs:
            _require_digest(item.blob_hash, "input blob_hash")
            if not item.coordinate.strip() or not item.source_path.strip() or not item.publication_receipt_id:
                raise ValueError("frozen input requires exact coordinates and publication receipt")

    @property
    def manifest_digest(self) -> str:
        # A publication receipt proves retention, not input identity. Exclude
        # its independently generated ID from this immutable content digest.
        digest = hashlib.sha256()
        digest.update(
            ("[" + json.dumps(self.enumeration_fingerprint, ensure_ascii=False, separators=(",", ":")) + ",[").encode()
        )
        for index, item in enumerate(self.inputs):
            fields: list[object] = [item.coordinate, item.source_path, item.blob_hash]
            if item.captured_identity is not None:
                fields.append(item.captured_identity.to_dict())
            digest.update(
                (("," if index else "") + json.dumps(fields, ensure_ascii=False, separators=(",", ":"))).encode()
            )
        digest.update(b"]")
        if self.source_name is not None:
            digest.update(
                (
                    ',["source_name",' + json.dumps(self.source_name, ensure_ascii=False, separators=(",", ":")) + "]"
                ).encode()
            )
        digest.update(b"]")
        return digest.hexdigest()

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "source_generation_id": self.source_generation_id,
            "enumeration_fingerprint": self.enumeration_fingerprint,
            "inputs": [item.to_dict() for item in self.inputs],
        }
        if self.source_name is not None:
            value["source_name"] = self.source_name
        return value

    @classmethod
    def from_dict(cls, value: object) -> FrozenSourceManifest:
        required = {"source_generation_id", "enumeration_fingerprint", "inputs"}
        if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"source_name"}:
            raise ValueError("invalid frozen source manifest fields")
        if not isinstance(value["source_generation_id"], str) or not isinstance(value["enumeration_fingerprint"], str):
            raise ValueError("invalid frozen source manifest identity")
        source_name = value.get("source_name")
        if source_name is not None and not isinstance(source_name, str):
            raise ValueError("invalid frozen source name")
        raw_inputs = value["inputs"]
        if not isinstance(raw_inputs, list):
            raise ValueError("frozen source inputs must be a list")
        inputs = []
        for item in raw_inputs:
            fields = {"coordinate", "source_path", "blob_hash", "publication_receipt_id"}
            if (
                not isinstance(item, dict)
                or not fields <= set(item)
                or set(item) - fields - {"captured_identity"}
                or any(not isinstance(item[field], str) for field in fields)
            ):
                raise ValueError("invalid frozen source input fields")
            captured = item.get("captured_identity")
            inputs.append(
                FrozenSourceInput(
                    item["coordinate"],
                    item["source_path"],
                    item["blob_hash"],
                    item["publication_receipt_id"],
                    CapturedSourceInputIdentity.from_dict(captured) if captured is not None else None,
                )
            )
        return cls(value["source_generation_id"], value["enumeration_fingerprint"], tuple(inputs), source_name)


@dataclass(frozen=True, slots=True)
class SealedSourceManifestRef:
    """Bounded authority for a complete prepared denominator in source.db."""

    source_generation_id: str
    enumeration_fingerprint: str
    manifest_digest: str
    custody_digest: str
    input_count: int
    source_name: str | None = None

    def __post_init__(self) -> None:
        if not self.source_generation_id or self.input_count < 1:
            raise ValueError("sealed manifest requires a nonempty generation")
        for name in ("enumeration_fingerprint", "manifest_digest", "custody_digest"):
            _require_digest(getattr(self, name), name)
        if self.source_name is not None and (not self.source_name.strip()):
            raise ValueError("sealed source name must be bounded and nonempty")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_generation_id": self.source_generation_id,
            "enumeration_fingerprint": self.enumeration_fingerprint,
            "manifest_digest": self.manifest_digest,
            "custody_digest": self.custody_digest,
            "input_count": self.input_count,
            "source_name": self.source_name,
        }

    @classmethod
    def from_dict(cls, value: object) -> SealedSourceManifestRef:
        fields = {
            "source_generation_id",
            "enumeration_fingerprint",
            "manifest_digest",
            "custody_digest",
            "input_count",
            "source_name",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("invalid sealed source manifest reference")
        if any(type(value[field]) is not str for field in fields - {"input_count", "source_name"}):
            raise ValueError("invalid sealed source manifest identity")
        if type(value["input_count"]) is not int or (
            value["source_name"] is not None and type(value["source_name"]) is not str
        ):
            raise ValueError("invalid sealed source manifest count or name")
        return cls(**value)


def source_manifest_from_dict(value: object) -> FrozenSourceManifest | SealedSourceManifestRef:
    """Decode old bounded audit commands and new sealed references exactly."""
    if isinstance(value, dict) and "inputs" in value:
        return FrozenSourceManifest.from_dict(value)
    return SealedSourceManifestRef.from_dict(value)


def begin_prepared_source_manifest(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    publisher_id: str,
    enumeration_fingerprint: str,
    source_name: str | None,
) -> None:
    if not conn.in_transaction or not source_generation_id or not publisher_id:
        raise ValueError("prepared source manifest needs a source transaction and generation")
    _require_digest(enumeration_fingerprint, "enumeration_fingerprint")
    if source_name is not None and (not source_name.strip()):
        raise ValueError("invalid source name")
    conn.execute(
        "INSERT INTO prepared_source_manifests(source_generation_id, publisher_id, enumeration_fingerprint, source_name) "
        "VALUES (?, ?, ?, ?)",
        (source_generation_id, publisher_id, enumeration_fingerprint, source_name),
    )


def abort_prepared_source_manifest(conn: sqlite3.Connection, *, source_generation_id: str, publisher_id: str) -> None:
    """Release this attempt's staged rows and reservations before acceptance.

    The publisher belongs to this ingest execution alone. A promoted generation
    is never abortable, including after an ambiguous audit interruption.
    """
    if not conn.in_transaction or not source_generation_id or not publisher_id:
        raise ValueError("prepared source abort requires a transaction and identities")
    header = conn.execute(
        "SELECT publisher_id FROM prepared_source_manifests WHERE source_generation_id=?", (source_generation_id,)
    ).fetchone()
    if header is not None and header[0] != publisher_id:
        raise ValueError("prepared source publisher identity changed")
    if (
        conn.execute(
            "SELECT 1 FROM source_generations WHERE source_generation_id=?", (source_generation_id,)
        ).fetchone()
        is not None
    ):
        raise ValueError("accepted source generation cannot be aborted")
    conn.execute("DELETE FROM prepared_source_manifest_members WHERE source_generation_id=?", (source_generation_id,))
    conn.execute("DELETE FROM prepared_source_manifests WHERE source_generation_id=?", (source_generation_id,))
    conn.execute("DELETE FROM blob_publication_reservations WHERE publisher_id=?", (publisher_id,))


def append_prepared_source_inputs(
    conn: sqlite3.Connection, source_generation_id: str, start_ordinal: int, inputs: tuple[FrozenSourceInput, ...]
) -> None:
    if not conn.in_transaction or not 1 <= len(inputs) <= 256:
        raise ValueError("prepared source input batch must be bounded by 256")
    row = conn.execute(
        "SELECT input_count, sealed_at_ms FROM prepared_source_manifests WHERE source_generation_id=?",
        (source_generation_id,),
    ).fetchone()
    if row is None or row[1] is not None or int(row[0]) != start_ordinal:
        raise ValueError("prepared source input batch is not the next unsealed page")
    for offset, item in enumerate(inputs):
        _require_digest(item.blob_hash, "input blob_hash")
        if not item.coordinate.strip() or not item.source_path.strip() or not item.publication_receipt_id:
            raise ValueError("prepared source input is incomplete")
        conn.execute(
            "INSERT INTO prepared_source_manifest_members(source_generation_id, ordinal, coordinate, source_path, "
            "blob_hash, publication_receipt_id, captured_input_identity) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source_generation_id,
                start_ordinal + offset,
                item.coordinate,
                item.source_path,
                bytes.fromhex(item.blob_hash),
                item.publication_receipt_id,
                _captured_input_json(item.captured_identity),
            ),
        )
    conn.execute(
        "UPDATE prepared_source_manifests SET input_count=? WHERE source_generation_id=?",
        (start_ordinal + len(inputs), source_generation_id),
    )


def prepare_source_manifest(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    publisher_id: str,
    enumeration_fingerprint: str,
    inputs: Iterable[FrozenSourceInput],
    source_name: str | None,
    sealed_at_ms: int,
    check_stop: Callable[[], None] | None = None,
) -> SealedSourceManifestRef:
    """Stage public inputs pagewise before constructing new acceptance plans.

    The caller owns the source transaction and publication reservations. This
    uses the same manifest rows and seal as daemon discovery; no inline audit
    command or input-count ceiling is created.
    """
    if check_stop is not None:
        check_stop()
    begin_prepared_source_manifest(
        conn,
        source_generation_id=source_generation_id,
        publisher_id=publisher_id,
        enumeration_fingerprint=enumeration_fingerprint,
        source_name=source_name,
    )
    iterator = iter(inputs)
    ordinal = 0
    while page := tuple(islice(iterator, 256)):
        if check_stop is not None:
            check_stop()
        append_prepared_source_inputs(conn, source_generation_id, ordinal, page)
        ordinal += len(page)
    if check_stop is not None:
        check_stop()
    return seal_prepared_source_manifest(conn, source_generation_id, sealed_at_ms=sealed_at_ms)


def _prepared_manifest_digests(
    conn: sqlite3.Connection, generation_id: str, fingerprint: str, source_name: str | None
) -> tuple[str, str, int]:
    semantic = hashlib.sha256()
    semantic.update(("[" + json.dumps(fingerprint, ensure_ascii=False, separators=(",", ":")) + ",[").encode())
    custody = hashlib.sha256()
    custody.update(
        json.dumps([generation_id, fingerprint, source_name], ensure_ascii=False, separators=(",", ":")).encode()
    )
    count = 0
    for ordinal, coordinate, source_path, blob_hash, receipt_id, captured in conn.execute(
        "SELECT ordinal, coordinate, source_path, blob_hash, publication_receipt_id, captured_input_identity "
        "FROM prepared_source_manifest_members WHERE source_generation_id=? ORDER BY ordinal",
        (generation_id,),
    ):
        if type(ordinal) is not int or ordinal != count or not isinstance(blob_hash, bytes) or len(blob_hash) != 32:
            raise ValueError("prepared source manifest has a missing or malformed ordinal")
        item: list[object] = [str(coordinate), str(source_path), blob_hash.hex()]
        identity = _captured_input_from_json(captured)
        if identity is not None:
            item.append(identity.to_dict())
        semantic.update((("," if count else "") + json.dumps(item, ensure_ascii=False, separators=(",", ":"))).encode())
        custody.update(
            ("\n" + json.dumps([ordinal, *item, str(receipt_id)], ensure_ascii=False, separators=(",", ":"))).encode()
        )
        count += 1
    semantic.update(b"]")
    if source_name is not None:
        semantic.update(
            (',["source_name",' + json.dumps(source_name, ensure_ascii=False, separators=(",", ":")) + "]").encode()
        )
    semantic.update(b"]")
    return semantic.hexdigest(), custody.hexdigest(), count


def seal_prepared_source_manifest(
    conn: sqlite3.Connection, generation_id: str, *, sealed_at_ms: int
) -> SealedSourceManifestRef:
    if not conn.in_transaction:
        raise ValueError("prepared source seal requires a source transaction")
    row = conn.execute(
        "SELECT enumeration_fingerprint, source_name, input_count, sealed_at_ms FROM prepared_source_manifests WHERE source_generation_id=?",
        (generation_id,),
    ).fetchone()
    if row is None or row[3] is not None or int(row[2]) < 1:
        raise ValueError("prepared source manifest is absent, empty, or already sealed")
    semantic, custody, count = _prepared_manifest_digests(conn, generation_id, str(row[0]), row[1])
    if count != int(row[2]):
        raise ValueError("prepared source manifest count differs from its rows")
    ref = SealedSourceManifestRef(generation_id, str(row[0]), semantic, custody, count, row[1])
    _require_prepared_reservations(conn, ref)
    conn.execute(
        "UPDATE prepared_source_manifests SET semantic_digest=?, custody_digest=?, sealed_at_ms=? WHERE source_generation_id=?",
        (semantic, custody, sealed_at_ms, generation_id),
    )
    return ref


def _require_prepared_reservations(conn: sqlite3.Connection, ref: SealedSourceManifestRef) -> None:
    missing = conn.execute(
        "SELECT 1 FROM prepared_source_manifest_members m LEFT JOIN blob_publication_reservations r "
        "ON r.publication_id=m.publication_receipt_id AND r.blob_hash=m.blob_hash "
        "WHERE m.source_generation_id=? AND r.publication_id IS NULL LIMIT 1",
        (ref.source_generation_id,),
    ).fetchone()
    if missing is not None:
        raise ValueError("sealed input publication reservation is missing or mismatched")


def validate_sealed_source_manifest(conn: sqlite3.Connection, ref: SealedSourceManifestRef) -> None:
    if not conn.in_transaction:
        raise ValueError("sealed source manifest requires a source transaction")
    row = conn.execute(
        "SELECT enumeration_fingerprint, source_name, input_count, semantic_digest, custody_digest, sealed_at_ms "
        "FROM prepared_source_manifests WHERE source_generation_id=?",
        (ref.source_generation_id,),
    ).fetchone()
    if (
        row is None
        or tuple(row[:5])
        != (ref.enumeration_fingerprint, ref.source_name, ref.input_count, ref.manifest_digest, ref.custody_digest)
        or row[5] is None
    ):
        raise ValueError("sealed source manifest reference differs from durable header")
    semantic, custody, count = _prepared_manifest_digests(
        conn, ref.source_generation_id, ref.enumeration_fingerprint, ref.source_name
    )
    if (semantic, custody, count) != (ref.manifest_digest, ref.custody_digest, ref.input_count):
        raise ValueError("sealed source manifest rows differ from its digests")
    accepted = conn.execute(
        "SELECT manifest_digest, item_count FROM source_generations WHERE source_generation_id=?",
        (ref.source_generation_id,),
    ).fetchone()
    if accepted is None:
        _require_prepared_reservations(conn, ref)
    elif tuple(accepted) != (ref.manifest_digest, ref.input_count):
        raise ValueError("accepted source generation differs from sealed manifest")
    else:
        accepted_count = conn.execute(
            "SELECT COUNT(*) FROM source_items WHERE source_generation_id=?",
            (ref.source_generation_id,),
        ).fetchone()[0]
        mismatch = conn.execute(
            "SELECT 1 FROM prepared_source_manifest_members m LEFT JOIN source_items i "
            "ON i.source_generation_id=m.source_generation_id AND i.logical_coordinate=m.coordinate "
            "WHERE m.source_generation_id=? AND (i.source_item_id IS NULL OR i.source_path<>m.source_path "
            "OR i.blob_hash<>m.blob_hash OR i.enumeration_fingerprint<>? "
            "OR i.captured_input_identity IS NOT m.captured_input_identity) LIMIT 1",
            (ref.source_generation_id, ref.enumeration_fingerprint),
        ).fetchone()
        if int(accepted_count) != ref.input_count or mismatch is not None:
            raise ValueError("accepted source inputs differ from sealed manifest")


def publish_sealed_source_manifest(
    conn: sqlite3.Connection, ref: SealedSourceManifestRef, *, prepared_at_ms: int
) -> None:
    """Publish all accepted members and consume custody in one source transaction."""
    from polylogue.storage.blob_publication import consume_blob_publication_receipt

    validate_sealed_source_manifest(conn, ref)
    existing = conn.execute(
        "SELECT 1 FROM source_generations WHERE source_generation_id=?",
        (ref.source_generation_id,),
    ).fetchone()
    if existing is not None:
        actual_count = conn.execute(
            "SELECT COUNT(*) FROM source_items WHERE source_generation_id=?",
            (ref.source_generation_id,),
        ).fetchone()[0]
        if int(actual_count) != ref.input_count:
            raise ValueError("accepted source generation is incomplete")
        return
    conn.execute(
        "INSERT INTO source_generations(source_generation_id, manifest_digest, addressing_mode, item_count, created_at_ms) "
        "VALUES (?, ?, 'physical-file-v1', ?, ?)",
        (ref.source_generation_id, ref.manifest_digest, ref.input_count, prepared_at_ms),
    )
    for _ordinal, coordinate, source_path, blob_hash, receipt_id, captured in conn.execute(
        "SELECT ordinal, coordinate, source_path, blob_hash, publication_receipt_id, captured_input_identity "
        "FROM prepared_source_manifest_members WHERE source_generation_id=? ORDER BY ordinal",
        (ref.source_generation_id,),
    ):
        item_id = source_item_id(
            source_generation_id=ref.source_generation_id,
            logical_coordinate=str(coordinate),
            addressing_mode="physical-file-v1",
        )
        conn.execute(
            "INSERT INTO source_items(source_generation_id, source_item_id, logical_coordinate, addressing_mode, "
            "source_path, disposition, outcome_code, stage, observed_at_ms, updated_at_ms, blob_hash, enumeration_fingerprint, "
            "captured_input_identity) "
            "VALUES (?, ?, ?, 'physical-file-v1', ?, 'pending', 'interrupted', 'manifest', ?, ?, ?, ?, ?)",
            (
                ref.source_generation_id,
                item_id,
                coordinate,
                source_path,
                prepared_at_ms,
                prepared_at_ms,
                blob_hash,
                ref.enumeration_fingerprint,
                _captured_input_json(_captured_input_from_json(captured)),
            ),
        )
        consume_blob_publication_receipt(conn, str(receipt_id), blob_hash)


def validate_frozen_source_manifest(conn: sqlite3.Connection, manifest: FrozenSourceManifest) -> None:
    """Prove every frozen input still holds its publication reservation.

    Read-only on purpose.  The source-WAL prepare transaction must be able to
    roll back to nothing, so the prepare phase only *checks* the manifest; the
    durable publication happens in :func:`publish_frozen_source_manifest`
    during promotion, once the audit commit that accepts it is durable.
    """
    if not conn.in_transaction:
        raise ValueError("frozen manifest requires the source-WAL prepare transaction")
    for item in manifest.inputs:
        if (
            conn.execute(
                "SELECT 1 FROM blob_publication_reservations WHERE publication_id=? AND blob_hash=?",
                (item.publication_receipt_id, bytes.fromhex(item.blob_hash)),
            ).fetchone()
            is None
        ):
            raise ValueError("frozen input publication reservation is missing or mismatched")


def publish_frozen_source_manifest(
    conn: sqlite3.Connection, manifest: FrozenSourceManifest, *, prepared_at_ms: int
) -> None:
    """Join the source-WAL promotion; no independent commit or raw admission."""
    from polylogue.storage.blob_publication import consume_blob_publication_receipt

    if not conn.in_transaction:
        raise ValueError("frozen manifest requires the source-WAL promotion transaction")
    publish_source_generation(
        conn,
        source_generation_id=manifest.source_generation_id,
        manifest_digest=manifest.manifest_digest,
        addressing_mode="physical-file-v1",
        coordinates=tuple(item.coordinate for item in manifest.inputs),
        source_paths={item.coordinate: item.source_path for item in manifest.inputs},
        input_blob_hashes={item.coordinate: bytes.fromhex(item.blob_hash) for item in manifest.inputs},
        captured_input_identities={item.coordinate: item.captured_identity for item in manifest.inputs},
        enumeration_fingerprint=manifest.enumeration_fingerprint,
        observed_at_ms=prepared_at_ms,
        commit=False,
    )
    for item in manifest.inputs:
        consume_blob_publication_receipt(conn, item.publication_receipt_id, bytes.fromhex(item.blob_hash))


def acquired_zip_manifest(
    *,
    blob_hash: str,
    publication_receipt_id: str,
    captured_identity: CapturedSourceInputIdentity,
    enumeration_fingerprint: str,
    source_name: str | None,
) -> FrozenSourceManifest:
    """Freeze an ordinary opened ZIP without a pass or clock identity."""
    item = FrozenSourceInput(
        '["physical-input-v1",0]',
        captured_identity.semantic_source_path,
        blob_hash,
        publication_receipt_id,
        captured_identity,
    )
    candidate = FrozenSourceManifest("ordinary-zip", enumeration_fingerprint, (item,), source_name)
    return FrozenSourceManifest(
        "ordinary-zip:" + candidate.manifest_digest,
        enumeration_fingerprint,
        (item,),
        source_name,
    )


def publish_acquired_zip_input(
    conn: sqlite3.Connection,
    manifest: FrozenSourceManifest,
    *,
    observed_at_ms: int,
) -> str:
    """Authenticate and publish one input inside the caller's transaction."""
    if len(manifest.inputs) != 1:
        raise ValueError("ordinary ZIP acquisition requires exactly one opened input")
    validate_frozen_source_manifest(conn, manifest)
    publish_frozen_source_manifest(conn, manifest, prepared_at_ms=observed_at_ms)
    return source_item_id(
        source_generation_id=manifest.source_generation_id,
        logical_coordinate=manifest.inputs[0].coordinate,
        addressing_mode="physical-file-v1",
    )


def source_item_id(*, source_generation_id: str, logical_coordinate: str, addressing_mode: str) -> str:
    """Derive identity only from generation-bound manifest coordinates."""
    payload = json.dumps(
        [source_generation_id, addressing_mode, logical_coordinate],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def publish_source_generation(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    manifest_digest: str,
    addressing_mode: str,
    coordinates: tuple[str, ...],
    observed_at_ms: int,
    origin: Origin | str | None = None,
    source_paths: Mapping[str, str] | None = None,
    attachments: tuple[SourceAttachment, ...] = (),
    policy_snapshot: ExcisionPolicySnapshot | None = None,
    input_blob_hashes: Mapping[str, bytes] | None = None,
    captured_input_identities: Mapping[str, CapturedSourceInputIdentity | None] | None = None,
    enumeration_fingerprint: str | None = None,
    commit: bool = True,
) -> tuple[str, ...]:
    """Publish every manifest coordinate before any read/decode/admission work."""
    if len(manifest_digest) != 64 or any(c not in "0123456789abcdef" for c in manifest_digest):
        raise ValueError("manifest_digest must be a SHA-256 hex digest")
    if len(set(coordinates)) != len(coordinates) or any(not coordinate.strip() for coordinate in coordinates):
        raise ValueError("manifest coordinates must be distinct and nonempty")
    if enumeration_fingerprint is not None:
        _require_digest(enumeration_fingerprint, "enumeration_fingerprint")
        if input_blob_hashes is None or set(input_blob_hashes) != set(coordinates):
            raise ValueError("enumerated manifest requires every frozen input blob")
    if input_blob_hashes is not None and (
        set(input_blob_hashes) != set(coordinates)
        or any(type(value) is not bytes or len(value) != 32 for value in input_blob_hashes.values())
    ):
        raise ValueError("input_blob_hashes must bind every coordinate to a SHA-256 blob")
    if captured_input_identities is not None and set(captured_input_identities) != set(coordinates):
        raise ValueError("captured input identities must bind every coordinate")
    origin_value = require_vocabulary(origin, Origin, field="origin") if origin is not None else None
    _preflight_source_attachments(attachments)
    ids = tuple(
        source_item_id(source_generation_id=source_generation_id, logical_coordinate=c, addressing_mode=addressing_mode)
        for c in coordinates
    )
    existing = conn.execute(
        "SELECT manifest_digest, addressing_mode, item_count FROM source_generations WHERE source_generation_id=?",
        (source_generation_id,),
    ).fetchone()
    if existing is not None and tuple(existing) != (manifest_digest, addressing_mode, len(coordinates)):
        raise ValueError(f"source generation manifest changed: {source_generation_id}")
    if existing is not None:
        expected = {
            (
                item_id,
                coordinate,
                (input_blob_hashes or {}).get(coordinate),
                enumeration_fingerprint,
                _captured_input_json((captured_input_identities or {}).get(coordinate)),
            )
            for item_id, coordinate in zip(ids, coordinates, strict=True)
        }
        actual = {
            tuple(row)
            for row in conn.execute(
                "SELECT source_item_id, logical_coordinate, blob_hash, enumeration_fingerprint, captured_input_identity "
                "FROM source_items WHERE source_generation_id=?",
                (source_generation_id,),
            )
        }
        if actual != expected:
            raise ValueError(f"source generation input binding changed: {source_generation_id}")
    conn.execute(
        """INSERT INTO source_generations(source_generation_id, manifest_digest, addressing_mode, item_count, created_at_ms)
           VALUES (?, ?, ?, ?, ?) ON CONFLICT(source_generation_id) DO NOTHING""",
        (source_generation_id, manifest_digest, addressing_mode, len(coordinates), observed_at_ms),
    )
    for coordinate, item_id in zip(coordinates, ids, strict=True):
        conn.execute(
            """INSERT INTO source_items(
                 source_generation_id, source_item_id, logical_coordinate, addressing_mode,
                 origin, source_path, disposition, outcome_code, stage, observed_at_ms, updated_at_ms,
                 blob_hash, enumeration_fingerprint, captured_input_identity)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 'interrupted', 'manifest', ?, ?, ?, ?, ?)
               ON CONFLICT(source_generation_id, source_item_id) DO NOTHING""",
            (
                source_generation_id,
                item_id,
                coordinate,
                addressing_mode,
                origin_value,
                (source_paths or {}).get(coordinate),
                observed_at_ms,
                observed_at_ms,
                (input_blob_hashes or {}).get(coordinate),
                enumeration_fingerprint,
                _captured_input_json((captured_input_identities or {}).get(coordinate)),
            ),
        )
    if policy_snapshot is not None:
        # ``excision_policy_projections`` is canonical source DDL. An ordinary
        # write must never create a durable shape: it leaves the table absent
        # from every archive this generation was not written into, and forces
        # readers to probe sqlite_schema instead of querying (polylogue-j264r).
        conn.execute(
            """INSERT INTO excision_policy_projections(
               source_generation_id, policy_digest, user_generation, audit_generation,
               audit_head, assertion_refs_json, generated_at_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_generation_id) DO UPDATE SET
                 policy_digest=excluded.policy_digest,
                 user_generation=excluded.user_generation,
                 audit_generation=excluded.audit_generation,
                 audit_head=excluded.audit_head,
                 assertion_refs_json=excluded.assertion_refs_json,
                 generated_at_ms=excluded.generated_at_ms""",
            (
                source_generation_id,
                policy_snapshot.digest,
                policy_snapshot.user_generation,
                policy_snapshot.audit_generation,
                policy_snapshot.audit_head,
                json.dumps(policy_snapshot.assertion_refs, separators=(",", ":")),
                observed_at_ms,
            ),
        )
    record_source_attachments(
        conn,
        source_generation_id=source_generation_id,
        attachments=attachments,
        observed_at_ms=observed_at_ms,
        commit=False,
    )
    if commit:
        conn.commit()
    return ids


def seal_source_generation(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    sealed_at_ms: int,
    commit: bool = True,
) -> None:
    """Seal only a complete, payload-backed manifest; otherwise fail closed."""
    existing = conn.execute(
        "SELECT sealed_at_ms FROM source_generations WHERE source_generation_id=?",
        (source_generation_id,),
    ).fetchone()
    if existing is None:
        raise KeyError(f"unknown source generation: {source_generation_id}")
    if existing[0] is not None:
        raise ValueError(f"source generation is already sealed: {source_generation_id}")
    census = source_generation_census(conn, source_generation_id)
    if not census["sealable"]:
        raise ValueError(f"source generation is not sealable: {source_generation_id}")
    attachment_table = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='source_attachments'"
    ).fetchone()
    if attachment_table is not None:
        attachment_census = source_attachment_census(conn, source_generation_id)
        if not attachment_census["sealable"]:
            raise ValueError(f"source generation has pending attachments: {source_generation_id}")
    conn.execute(
        "UPDATE source_generations SET sealed_at_ms=? WHERE source_generation_id=?",
        (sealed_at_ms, source_generation_id),
    )
    if commit:
        conn.commit()


def transition_source_item(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    request_id: str,
    disposition: AcquisitionDisposition,
    outcome_code: IngestOutcome,
    stage: str,
    observed_at_ms: int,
    retryable: bool | None = None,
    diagnostic: str | None = None,
    evidence_ref: str | None = None,
    content_fingerprint: str | None = None,
    source_fingerprint: str | None = None,
    parser_fingerprint: str | None = None,
    policy_fingerprint: str | None = None,
    raw_id: str | None = None,
    blob_hash: bytes | None = None,
    commit: bool = True,
    expected_revision: int | None = None,
) -> int:
    """Advance a fact; a retry of the most recent transition is a no-op.

    The command journal owns historical request deduplication. Callers joining
    a domain publication transaction use ``commit=False`` and can reject stale
    competing item updates with ``expected_revision``.
    """
    # These vocabularies are owned by the source-item state machine and
    # IngestOutcome. SQLite intentionally carries no enum registry; normalize
    # and reject at this public write boundary before even the idempotent path.
    disposition_value = require_vocabulary(disposition, AcquisitionDisposition, field="disposition")
    outcome_value = require_vocabulary(outcome_code, IngestOutcome, field="outcome_code")
    if retryable is None:
        retryable = INGEST_OUTCOME_RETRYABLE[IngestOutcome(outcome_value)]
    row = conn.execute(
        "SELECT revision, request_id, blob_hash, enumeration_fingerprint FROM source_items "
        "WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if row is None:
        raise KeyError(f"unmanifested source item: {source_generation_id}/{source_item_id}")
    if row[1] == request_id:
        return int(row[0])
    if expected_revision is not None and int(row[0]) != expected_revision:
        raise ValueError("source item revision changed")
    if row[3] is not None and blob_hash is not None and blob_hash != row[2]:
        raise ValueError("frozen source input blob cannot change")
    revision = int(row[0]) + 1
    conn.execute(
        """UPDATE source_items SET disposition=?, outcome_code=?, stage=?, retryable=?, diagnostic=?,
           evidence_ref=?, content_fingerprint=COALESCE(?,content_fingerprint),
           source_fingerprint=COALESCE(?,source_fingerprint), parser_fingerprint=COALESCE(?,parser_fingerprint),
           policy_fingerprint=COALESCE(?,policy_fingerprint), raw_id=COALESCE(?,raw_id),
           blob_hash=COALESCE(?,blob_hash), revision=?, request_id=?, observed_at_ms=?, updated_at_ms=?
           WHERE source_generation_id=? AND source_item_id=?""",
        (
            disposition_value,
            outcome_value,
            stage,
            None if retryable is None else int(retryable),
            bounded_diagnostic(diagnostic, max_len=4096),
            evidence_ref,
            content_fingerprint,
            source_fingerprint,
            parser_fingerprint,
            policy_fingerprint,
            raw_id,
            blob_hash,
            revision,
            request_id,
            observed_at_ms,
            observed_at_ms,
            source_generation_id,
            source_item_id,
        ),
    )
    if commit:
        conn.commit()
    return revision


def _require_digest(value: str, field: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a SHA-256 hex digest")


def record_source_item_raw_member(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    record_coordinate: str,
    raw_id: str,
    raw_blob_hash: bytes,
) -> None:
    """Join successful raw admission in the caller's source transaction.

    An existing edge whose raw was retired is immutable historical evidence,
    not permission to reacquire it. Call this for deduplicated admissions too.
    """
    if not conn.in_transaction:
        raise ValueError("raw membership requires the raw admission transaction")
    if not record_coordinate.strip() or not raw_id or type(raw_blob_hash) is not bytes or len(raw_blob_hash) != 32:
        raise ValueError("raw membership requires a coordinate, raw id and SHA-256 blob")
    item = conn.execute(
        "SELECT enumeration_fingerprint, enumerated_at_ms FROM source_items "
        "WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if item is None or item[0] is None:
        raise ValueError("raw membership requires a frozen enumerated input")
    existing = conn.execute(
        "SELECT raw_id, raw_blob_hash FROM source_item_raw_members "
        "WHERE source_generation_id=? AND source_item_id=? AND record_coordinate=?",
        (source_generation_id, source_item_id, record_coordinate),
    ).fetchone()
    if existing is not None:
        if existing[0] is None:
            raise ValueError("source member raw was retired; readmission is forbidden")
        if tuple(existing) != (raw_id, raw_blob_hash):
            raise ValueError("source record membership changed")
        return
    if item[1] is not None:
        raise ValueError("completed source enumeration cannot gain members")
    raw = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
    if raw is None or raw[0] != raw_blob_hash:
        raise ValueError("source member requires matching admitted raw evidence")
    conn.execute(
        "INSERT INTO source_item_raw_members "
        "(source_generation_id, source_item_id, record_coordinate, raw_id, raw_blob_hash) VALUES (?, ?, ?, ?, ?)",
        (source_generation_id, source_item_id, record_coordinate, raw_id, raw_blob_hash),
    )


def complete_source_item_enumeration(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    enumeration_fingerprint: str,
    record_coordinates: Iterable[str],
    enumerated_at_ms: int,
    member_ordinals: Iterable[int] | None = None,
    member_count: int | None = None,
    check_stop: Callable[[], None] | None = None,
) -> str:
    """Validate exhausted decoder evidence against the same durable membership.

    Call only after normal EOF and integrity validation. Temporary denominator
    rows belong to this validation and never establish acquisition membership.
    Exact sets and canonical digests are compared in SQLite order, without
    retaining the complete decoder output or central directory in Python.
    """
    if not conn.in_transaction:
        raise ValueError("enumeration completion requires a source transaction")

    def checkpoint() -> None:
        if check_stop is not None:
            check_stop()

    checkpoint()
    _require_digest(enumeration_fingerprint, "enumeration_fingerprint")
    item = conn.execute(
        "SELECT enumeration_fingerprint, enumerated_record_count, enumeration_digest, enumerated_at_ms, "
        "enumerated_member_count, enumeration_member_digest "
        "FROM source_items WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if item is None or item[0] != enumeration_fingerprint:
        raise ValueError("source enumeration decoder binding changed")
    count, digest, member_count, member_digest, retired_count = _measure_source_item_enumeration(
        conn,
        source_generation_id=source_generation_id,
        source_item_id=source_item_id,
        record_coordinates=record_coordinates,
        member_ordinals=member_ordinals,
        member_count=member_count,
        check_stop=check_stop,
    )
    binding = (source_generation_id, source_item_id)
    checkpoint()
    if retired_count:
        raise ValueError("source enumeration contains retired raw members")
    if item[3] is not None:
        if tuple(item[1:3]) != (count, digest) or tuple(item[4:]) != (member_count, member_digest):
            raise ValueError("completed source enumeration changed")
        return digest
    conn.execute(
        "UPDATE source_items SET enumerated_record_count=?, enumeration_digest=?, enumerated_at_ms=?, "
        "enumerated_member_count=?, enumeration_member_digest=? "
        "WHERE source_generation_id=? AND source_item_id=?",
        (count, digest, enumerated_at_ms, member_count, member_digest, *binding),
    )
    return digest


def _measure_source_item_enumeration(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    record_coordinates: Iterable[str],
    member_ordinals: Iterable[int] | None,
    member_count: int | None,
    check_stop: Callable[[], None] | None = None,
) -> tuple[int, str, int, str, int]:
    """Measure exact membership on the caller's snapshot without publishing it."""

    def checkpoint() -> None:
        if check_stop is not None:
            check_stop()

    checkpoint()
    binding = (source_generation_id, source_item_id)
    with scratch_connection_context(prefix="polylogue-enumeration-", filename="denominator.db") as denominator:
        # Regular indexed tables spill to this owner's private file even when
        # the connection profile intentionally keeps unrelated TEMP tables in
        # memory. Source membership is read from the caller's live transaction.
        # The scratch factory's MEMORY journal would retain an update journal
        # for the full ordinal set. Select a disk journal before any SQL data
        # or transaction exists; never change TEMP storage under the caller.
        denominator.execute("PRAGMA journal_mode=DELETE")
        denominator.execute("PRAGMA temp_store=FILE")
        denominator.execute("BEGIN")
        denominator.execute("CREATE TABLE records(coordinate TEXT PRIMARY KEY) WITHOUT ROWID")
        denominator.execute(
            "CREATE TABLE ordinals(ordinal INTEGER PRIMARY KEY, accepted INTEGER, "
            "member_name TEXT, disposition TEXT, diagnostic TEXT)"
        )
        coordinates: Iterable[object] = record_coordinates
        for coordinate in coordinates:
            checkpoint()
            if not isinstance(coordinate, str) or not coordinate.strip():
                raise ValueError("enumeration coordinates must be distinct and nonempty")
            try:
                denominator.execute("INSERT INTO records VALUES (?)", (coordinate,))
            except sqlite3.IntegrityError as exc:
                raise ValueError("enumeration coordinates must be distinct and nonempty") from exc
        count = int(denominator.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        record_hash = hashlib.sha256(b"[")
        seen = 0
        retired_count = 0
        is_zip = False
        cursor = conn.execute(
            "SELECT record_coordinate, raw_blob_hash, raw_id FROM source_item_raw_members "
            "WHERE source_generation_id=? AND source_item_id=? ORDER BY record_coordinate",
            binding,
        )
        try:
            for coordinate, blob_hash, raw_id in cursor:
                checkpoint()
                retired_count += raw_id is None
                if denominator.execute("SELECT 1 FROM records WHERE coordinate=?", (coordinate,)).fetchone() is None:
                    raise ValueError("source enumeration has missing or unexpected raw members")
                is_zip |= coordinate.startswith('["zip-v2"')
                if seen:
                    record_hash.update(b",")
                record_hash.update(
                    json.dumps((coordinate, blob_hash.hex()), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                )
                seen += 1
        finally:
            cursor.close()
        if seen != count:
            raise ValueError("source enumeration has missing or unexpected raw members")
        record_hash.update(b"]")
        digest = record_hash.hexdigest()
        if member_count is None and is_zip:
            raise ValueError("ZIP source enumeration requires its central-directory denominator")
        member_hash = hashlib.sha256(b"[")
        first = True

        def emit_member(value: object) -> None:
            nonlocal first
            checkpoint()
            if not first:
                member_hash.update(b",")
            member_hash.update(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            first = False

        if member_count is not None:
            if type(member_count) is not int or member_count < 0 or member_ordinals is None:
                raise ValueError("source member denominator requires non-negative count and ordinals")
            for ordinal in member_ordinals:
                checkpoint()
                if type(ordinal) is not int or ordinal < 0:
                    raise ValueError("source member ordinals must be distinct and non-negative")
                try:
                    denominator.execute("INSERT INTO ordinals(ordinal) VALUES (?)", (ordinal,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("source member ordinals must be distinct and non-negative") from exc
            extent = denominator.execute("SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM ordinals").fetchone()
            if extent[0] != member_count or (member_count and extent[1:] != (0, member_count - 1)):
                raise ValueError("source enumeration has missing or unexpected central-directory members")
            cursor = conn.execute(
                "SELECT c.entry_ordinal FROM source_item_raw_members m JOIN raw_container_coordinates c "
                "ON c.raw_id=m.raw_id WHERE m.source_generation_id=? AND m.source_item_id=?",
                binding,
            )
            try:
                for row in cursor:
                    checkpoint()
                    updated = denominator.execute(
                        "UPDATE ordinals SET accepted=1 WHERE ordinal=? AND (accepted IS NULL OR accepted=1)",
                        (row[0],),
                    )
                    if updated.rowcount != 1:
                        raise ValueError("source enumeration has overlapping or missing central-directory dispositions")
            finally:
                cursor.close()
            cursor = conn.execute(
                "SELECT entry_ordinal, member_name, disposition, diagnostic FROM source_item_member_dispositions "
                "WHERE source_generation_id=? AND source_item_id=?",
                binding,
            )
            try:
                for ordinal, name, disposition, diagnostic in cursor:
                    checkpoint()
                    updated = denominator.execute(
                        "UPDATE ordinals SET accepted=0,member_name=?,disposition=?,diagnostic=? "
                        "WHERE ordinal=? AND accepted IS NULL",
                        (name, disposition, diagnostic, ordinal),
                    )
                    if updated.rowcount != 1:
                        raise ValueError("source enumeration has overlapping or missing central-directory dispositions")
            finally:
                cursor.close()
            if denominator.execute("SELECT 1 FROM ordinals WHERE accepted IS NULL LIMIT 1").fetchone() is not None:
                raise ValueError("source enumeration has overlapping or missing central-directory dispositions")
            cursor = denominator.execute("SELECT ordinal FROM ordinals WHERE accepted=1 ORDER BY ordinal")
            try:
                for row in cursor:
                    emit_member((int(row[0]), "accepted", "", ""))
            finally:
                cursor.close()
            cursor = denominator.execute(
                "SELECT ordinal,member_name,disposition,diagnostic FROM ordinals WHERE accepted=0 ORDER BY ordinal"
            )
            try:
                for row in cursor:
                    emit_member((int(row[0]), str(row[1]), str(row[2]), str(row[3])))
            finally:
                cursor.close()
        else:
            member_count = count
            for ordinal in range(count):
                emit_member(ordinal)
        member_hash.update(b"]")
        member_digest = member_hash.hexdigest()
        checkpoint()
        return count, digest, member_count, member_digest, retired_count


def retained_completed_source_item_for_raw(source_read: CompletedSourceItemRead, raw_id: str) -> tuple[str, str]:
    """Select an exact completed set, never an acquisition clock or group rank.

    A raw can occur in multiple accepted inputs. They prove one interchangeable
    group only when the decoder binding, complete enumeration digests and
    their counts agree.
    The deterministic returned coordinate does not choose newer authority.
    """
    from polylogue.core.raw_failure_evidence import RetainedZipMembershipUnprovedError

    selected: tuple[str, str] | None = None
    signature: tuple[object, ...] | None = None
    with source_read.completed_source_item_rows(raw_id) as cursor:
        for row in cursor:
            check_compute_cancelled()
            if row[2] is None:
                raise RetainedZipMembershipUnprovedError("completed ZIP input has no decoder binding")
            _require_digest(str(row[2]), "enumeration_fingerprint")
            current = tuple(row[2:])
            if any(value is None for value in current):
                raise RetainedZipMembershipUnprovedError("completed ZIP input has incomplete enumeration evidence")
            if signature is not None and signature != current:
                raise RetainedZipMembershipUnprovedError("raw belongs to different completed ZIP acquisition sets")
            signature = current
            if selected is None:
                selected = (str(row[0]), str(row[1]))
    if selected is None:
        raise RetainedZipMembershipUnprovedError("retained ZIP raw has no proved complete acquisition group")
    return selected


def record_source_item_member_disposition(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    entry_ordinal: int,
    member_name: str,
    disposition: SourceItemMemberDisposition,
    diagnostic: str,
    observed_at_ms: int,
) -> None:
    """Persist one refused/unselected central-directory member idempotently."""
    value = require_vocabulary(disposition, SourceItemMemberDisposition, field="member disposition")
    if entry_ordinal < 0:
        raise ValueError("source member ordinal must be non-negative")
    if not member_name:
        raise ValueError("source member name must be non-empty")
    item = conn.execute(
        "SELECT enumerated_at_ms FROM source_items WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if item is None:
        raise KeyError(f"unmanifested source item: {source_generation_id}/{source_item_id}")
    admitted = conn.execute(
        "SELECT 1 FROM source_item_raw_members m "
        "JOIN raw_container_coordinates c ON c.raw_id=m.raw_id "
        "WHERE m.source_generation_id=? AND m.source_item_id=? AND c.entry_ordinal=?",
        (source_generation_id, source_item_id, entry_ordinal),
    ).fetchone()
    if admitted is not None:
        raise ValueError("source member already has an admitted raw record")
    # The exact member name is evidence. Only display diagnostics are shortened;
    # central-directory ordinal and complete name remain available for replay.
    bounded = bounded_diagnostic(diagnostic, max_len=_MAX_MEMBER_DIAGNOSTIC_CHARS) or ""
    row = conn.execute(
        "SELECT member_name, disposition, diagnostic, observed_at_ms FROM source_item_member_dispositions "
        "WHERE source_generation_id=? AND source_item_id=? AND entry_ordinal=?",
        (source_generation_id, source_item_id, entry_ordinal),
    ).fetchone()
    if row is not None:
        if tuple(row[:3]) != (member_name, value, bounded):
            raise ValueError("source member disposition changed")
        return
    if item[0] is not None:
        raise ValueError("completed source enumeration cannot gain member dispositions")
    conn.execute(
        "INSERT INTO source_item_member_dispositions "
        "(source_generation_id, source_item_id, entry_ordinal, member_name, disposition, diagnostic, observed_at_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source_generation_id, source_item_id, entry_ordinal, member_name, value, bounded, observed_at_ms),
    )


def source_generation_census(conn: sqlite3.Connection, source_generation_id: str) -> dict[str, int | bool]:
    cursor = conn.execute(
        "SELECT * FROM source_item_reconciliation WHERE source_generation_id=?", (source_generation_id,)
    )
    row = cursor.fetchone()
    if row is None:
        raise KeyError(source_generation_id)
    names = [column[0] for column in cursor.description or ()]
    return {
        name: (bool(value) if name == "sealable" else (value if name == "source_generation_id" else int(value or 0)))
        for name, value in zip(names, row, strict=True)
    }


__all__ = [
    "AcquisitionDisposition",
    "SourceItemMemberDisposition",
    "SourceItemMemberDispositionRecord",
    "SourceItem",
    "FrozenSourceInput",
    "FrozenSourceManifest",
    "CapturedSourceInputIdentity",
    "acquired_zip_manifest",
    "publish_acquired_zip_input",
    "SourceItemAdmission",
    "retained_completed_source_item_for_raw",
    "complete_source_item_enumeration",
    "publish_source_generation",
    "validate_frozen_source_manifest",
    "publish_frozen_source_manifest",
    "record_source_item_raw_member",
    "record_source_item_member_disposition",
    "seal_source_generation",
    "source_generation_census",
    "source_item_id",
    "transition_source_item",
]


class CompletedSourceItemRead(Protocol):
    """The canonical completed-group metadata on one owned Source snapshot."""

    def completed_source_item_rows(self, raw_id: str) -> AbstractContextManager[sqlite3.Cursor]: ...


_COMPLETED_SOURCE_ITEM_SQL = (
    "SELECT i.source_generation_id, i.source_item_id, i.enumeration_fingerprint, "
    "i.enumerated_record_count, i.enumeration_digest, i.enumerated_member_count, i.enumeration_member_digest "
    "FROM source_item_raw_members m JOIN source_items i "
    "ON i.source_generation_id=m.source_generation_id AND i.source_item_id=m.source_item_id "
    "JOIN raw_sessions r ON r.raw_id=m.raw_id AND r.blob_hash=m.raw_blob_hash "
    "WHERE m.raw_id=? AND i.enumerated_at_ms IS NOT NULL "
    "ORDER BY i.source_generation_id, i.source_item_id"
)


@dataclass(frozen=True, slots=True)
class ConnectionCompletedSourceItemRead:
    connection: sqlite3.Connection

    def completed_source_item_rows(self, raw_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        return connection_cursor(self.connection, _COMPLETED_SOURCE_ITEM_SQL, (raw_id,))


if TYPE_CHECKING:
    from polylogue.security.excision_policy import ExcisionPolicySnapshot

"""Raw-capture writer/read helpers for source.db.

Writer module: source.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol, cast, get_args

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    canonical_authority_logical_key,
    raw_authority_parser_fingerprint,
)
from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider, ValidationMode, ValidationStatus
from polylogue.core.raw_coordinates import (
    CapturedZipMemberCoordinate,
    MemberAddressingMode,
    captured_zip_coordinate_receipt,
    read_captured_zip_coordinate_receipt,
)
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_EVIDENCE_KINDS,
    terminal_carrier_overwrite_predicate,
)
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.security.excision_policy import ExcisionPolicyError, ExcisionPolicySnapshot
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.sqlite.archive_tiers.common import require_vocabulary
from polylogue.storage.sqlite.raw_state_update import compile_raw_state_update, raw_state_parameter

_TERMINAL_CARRIER_GUARD_SQL = f"\nWHERE NOT {terminal_carrier_overwrite_predicate()}\n"


class ContentExcisedError(ExcisionPolicyError):
    """Raised when acquire attempts to re-store a durably excised blob.

    The archive can forget on purpose (polylogue-27m): once a blob hash is
    recorded in ``excised_content``, ordinary re-ingest of unmodified source
    bytes must not resurrect it, even across an ``index.db`` rebuild. There
    are two acquire-time raw-session write functions that gate on this --
    ``write_source_raw_session`` (payload held in memory) and
    ``write_source_raw_session_blob_ref`` (payload already published as a
    blob, not held in memory; the memory-bounded path for multi-GiB files).
    The daemon reaches both: ``polylogue import`` submits the ``ingest``
    operation, which admits through ``raw_admission``, and the live watcher
    picks one per record in
    ``sources.live.batch.LiveBatchProcessor._ingest_full_records_archive``.
    ``operations.canonical_archive_ingest.ingest_one_shot_archive`` drives that
    batch in-process for tooling (demo seeding, probes, tests), not as an
    import route. Both must gate identically or the blob-ref route silently resurrects
    excised content that arrives via the streaming path (polylogue-27m fix
    round). Callers at the live batch orchestration layer catch this and skip the
    one file (count it, continue the batch) rather than aborting the whole
    run.
    """

    def __init__(self, *, blob_hash: bytes, source_path: str) -> None:
        self.blob_hash = blob_hash
        self.source_path = source_path
        super().__init__(
            f"content at {source_path!r} (blob_hash={blob_hash.hex()}) was durably excised; refusing to re-acquire"
        )


class HookEventConflictError(RuntimeError):
    """A hook carrier or logical event attempted an immutable identity change."""


PENDING_RAW_LOGICAL_SOURCE_PREFIX = "pending-raw:"

# Storage-local blob ownership categories. These values identify the durable
# relation that keeps bytes live; adding one requires reviewing blob liveness
# ownership and the source DDL migration together.
BlobRefType = Literal["raw_payload", "attachment", "sidecar", "hook_payload"]
# Coordinate format is a durable encoding contract. A new format needs a
# reviewed reader/reacquisition path and source migration, not just a new tag.
ContainerCoordinateFormat = Literal["zip-v2"]
# Carrier role controls which source is eligible to publish hook-carrier
# authority; extend only with a reviewed source selection policy.
HookCarrierRole = Literal["primary-writable", "legacy-read-only"]
_BLOB_REF_TYPES = get_args(BlobRefType)
_CONTAINER_COORDINATE_FORMATS = get_args(ContainerCoordinateFormat)
_HOOK_CARRIER_ROLES = get_args(HookCarrierRole)


def _is_raw_failure_artifact_kind(artifact_kind: object) -> bool:
    value = getattr(artifact_kind, "value", artifact_kind)
    return str(value) in RAW_FAILURE_EVIDENCE_KINDS


def pending_raw_logical_source_key(*, origin: Origin | str, source_path: str, source_index: int, raw_id: str) -> str:
    """Return the typed identity used until a parser proves the session key."""
    origin_value = require_vocabulary(origin, Origin, field="origin")
    return f"{PENDING_RAW_LOGICAL_SOURCE_PREFIX}{origin_value}:{source_index}:{source_path}:{raw_id}"


def is_blob_hash_excised(conn: sqlite3.Connection, blob_hash: bytes, *, schema: str = "main") -> bool:
    """Return ``True`` if ``blob_hash`` is recorded in the durable excision ledger.

    A cheap primary-key lookup; a no-op fast path when the table does not
    exist (an archive whose ``source.db`` predates migration 010) or is
    empty (the overwhelming common case -- excision is rare).
    """
    if schema not in {"main", "source"}:
        raise ValueError(f"unsupported source schema: {schema}")
    if not _table_exists(conn, "excised_content", schema=schema):
        return False
    with connection_cursor(
        conn,
        f"SELECT 1 FROM {schema}.excised_content WHERE removed_hash = ? AND hash_kind = 'blob_hash' LIMIT 1",
        (blob_hash,),
    ) as cursor:
        return cursor.fetchone() is not None


def _assert_excision_policy(
    blob_hash: bytes,
    *,
    source_path: str,
    policy_snapshot: ExcisionPolicySnapshot | None,
) -> None:
    if policy_snapshot is None or policy_snapshot.allows(blob_hash):
        return
    # Both acquire-time excision gates must refuse in the same vocabulary. The
    # policy snapshot resolves the durable excision set before the write, so it
    # fires first and its bare ``ExcisionPolicyError`` used to escape the batch
    # orchestrators, which catch only ``ContentExcisedError`` -- one excised
    # file then rolled back and aborted the whole reingest batch instead of
    # being skipped and counted. Raise the class those callers handle.
    raise ContentExcisedError(blob_hash=blob_hash, source_path=source_path)


def record_excised_blob_hash(
    conn: sqlite3.Connection,
    *,
    blob_hash: bytes,
    reason: str,
    actor: str,
    prior_revision: str | None = None,
    span: tuple[int, int] | None = None,
    excised_at_ms: int,
) -> None:
    """Idempotently record a durable removed-content marker.

    ``INSERT ... ON CONFLICT DO NOTHING``: re-excising the same blob hash
    (e.g. a retried apply) must not overwrite the original reason/actor/
    timestamp of record.
    """
    span_start, span_end = span if span is not None else (None, None)
    conn.execute(
        """
        INSERT INTO excised_content (
            removed_hash, hash_kind, reason, actor, prior_revision, span_start, span_end, excised_at_ms
        ) VALUES (?, 'blob_hash', ?, ?, ?, ?, ?, ?)
        ON CONFLICT(removed_hash, hash_kind) DO NOTHING
        """,
        (blob_hash, reason, actor, prior_revision, span_start, span_end, excised_at_ms),
    )


@dataclass(frozen=True, slots=True)
class ArchiveSourceBlobRef:
    """A compact raw blob reference row."""

    blob_hash: bytes
    raw_id: str | None = None
    ref_type: BlobRefType = "raw_payload"
    source_path: str | None = None
    size_bytes: int | None = None
    acquired_at_ms: int | None = None
    publication_receipt_id: str | None = None


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
class ArchiveRawSessionEnvelope:
    """Compact read-back view of one raw-session row."""

    raw_id: str
    origin: str
    capture_mode: str | None
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


CaptureModeResolutionStatus = Literal["unknown", "unambiguous", "ambiguous"]


@dataclass(frozen=True, slots=True)
class CaptureModeResolution:
    """Every acquisition mode ever observed for one raw, explicitly disambiguated.

    ``raw_sessions.capture_mode`` (v8) is a convenience cache of only the
    *first* known acquisition mode for a raw_id -- later writes update it
    only when it is still NULL. Because ``raw_id`` is content-derived, a
    GEMINI export and a live DRIVE acquisition of byte-identical bytes
    collide on the same raw_id even though their acquisition mechanism
    genuinely differs, and the cache alone cannot tell a caller whether the
    value it holds is the only fact on record or one of several (polylogue-
    buns). ``status`` makes that explicit: ``"unknown"`` when no observation
    was ever recorded, ``"unambiguous"`` when exactly one distinct mode was,
    ``"ambiguous"`` when more than one was. ``modes`` is ordered by first
    observation time.
    """

    raw_id: str
    status: CaptureModeResolutionStatus
    modes: tuple[Provider, ...]


def record_capture_mode_observation(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    capture_mode: Provider | str | None,
    observed_at_ms: int,
) -> None:
    """Append one durable (raw_id, capture_mode) observation.

    Idempotent: repeating an already-recorded pair is a no-op (its original
    ``first_observed_at_ms`` is kept), but a genuinely new mode for this
    raw_id gets its own row. This is what lets a later
    :func:`read_capture_mode_resolution` distinguish "only one acquisition
    mode was ever observed" from "several were, and the cache only kept the
    first" -- content-hash dedup already collapses the bytes to one raw_id,
    but must not also collapse distinct acquisition evidence about how those
    bytes were obtained (polylogue-buns AC1/AC2). A ``None`` capture_mode is
    a no-op: unknown provenance is not itself an observation.
    """
    value = require_vocabulary(capture_mode, Provider, field="capture_mode") if capture_mode is not None else None
    if value is None:
        return
    conn.execute(
        """
        INSERT INTO raw_capture_observations (raw_id, capture_mode, first_observed_at_ms)
        VALUES (?, ?, ?)
        ON CONFLICT(raw_id, capture_mode) DO NOTHING
        """,
        (raw_id, value, observed_at_ms),
    )


def record_raw_container_coordinate(
    conn: sqlite3.Connection,
    raw_id: str,
    *,
    coordinate_format: ContainerCoordinateFormat,
    entry_ordinal: int,
    split_index: int,
    addressing_mode: MemberAddressingMode | str | None,
    content_identity: str | None = None,
    captured_coordinate: CapturedZipMemberCoordinate | None = None,
    manage_transaction: bool = True,
) -> None:
    """Persist the actual immutable acquired member namespace and reading.

    A coordinate-less operational string cannot establish member identity.
    Whole-member and split-element readings remain distinct even at slot zero.
    """
    if entry_ordinal < 0 or split_index < 0:
        raise ValueError("container entry ordinal and split index must be non-negative")
    coordinate_format_value = require_vocabulary(
        coordinate_format, _CONTAINER_COORDINATE_FORMATS, field="coordinate_format"
    )
    if captured_coordinate is None:
        from polylogue.core.raw_failure_evidence import RetainedZipMembershipUnprovedError

        raise RetainedZipMembershipUnprovedError("ZIP coordinate publication requires its captured member receipt")
    if content_identity is not None:
        if len(content_identity) != 64:
            raise ValueError("content_identity must be a 64-character digest")
        try:
            bytes.fromhex(content_identity)
        except ValueError as exc:
            raise ValueError("content_identity must be hexadecimal") from exc
    mode = (
        require_vocabulary(addressing_mode, MemberAddressingMode, field="addressing_mode")
        if addressing_mode is not None
        else None
    )
    receipt = captured_zip_coordinate_receipt(captured_coordinate)
    if (
        captured_coordinate.entry_ordinal != entry_ordinal
        or captured_coordinate.split_index != split_index
        or captured_coordinate.addressing_mode.value != mode
    ):
        raise ValueError("captured ZIP receipt differs from its durable address")
    with conn if manage_transaction else nullcontext():
        conn.execute(
            """
            INSERT OR IGNORE INTO raw_container_coordinates (
                raw_id, coordinate_format, entry_ordinal, split_index, addressing_mode, content_identity, captured_coordinate
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (raw_id, coordinate_format_value, entry_ordinal, split_index, mode, content_identity, receipt),
        )
        stored = conn.execute(
            """
            SELECT coordinate_format, entry_ordinal, split_index, addressing_mode, content_identity, captured_coordinate
            FROM raw_container_coordinates
            WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchone()
        stored_tuple = tuple(stored) if stored is not None else None
        expected = (coordinate_format, entry_ordinal, split_index)
        if stored_tuple is None or stored_tuple[:3] != expected:
            raise ValueError(f"raw container coordinate changed for {raw_id}")
        if stored_tuple[5] != receipt:
            raise ValueError(f"captured ZIP coordinate changed or missing for {raw_id}")
        if stored_tuple[3] != mode:
            raise ValueError(f"raw container addressing mode changed for {raw_id}")
        if content_identity is not None and stored_tuple[4] not in {None, content_identity}:
            raise ValueError(f"raw container content identity changed for {raw_id}")
        if content_identity is not None and stored_tuple[4] is None:
            conn.execute(
                "UPDATE raw_container_coordinates SET content_identity = ? WHERE raw_id = ?",
                (content_identity, raw_id),
            )

        # Keep the durable raw row in step with its coordinate evidence.
        raw_identity = conn.execute(
            "SELECT addressing_mode, content_identity FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
        if raw_identity is None:
            raise ValueError(f"raw session missing for {raw_id}")
        if raw_identity[0] not in {None, mode}:
            raise ValueError(f"raw addressing mode changed for {raw_id}")
        if content_identity is not None and raw_identity[1] not in {None, content_identity}:
            raise ValueError(f"raw content identity changed for {raw_id}")
        conn.execute(
            """
            UPDATE raw_sessions
               SET addressing_mode = COALESCE(addressing_mode, ?),
                   content_identity = COALESCE(content_identity, ?)
             WHERE raw_id = ?
            """,
            (mode, content_identity, raw_id),
        )


def read_raw_captured_zip_coordinate(conn: sqlite3.Connection, raw_id: str) -> CapturedZipMemberCoordinate | None:
    """Read immutable acquired member evidence without reopening its namespace."""
    with _ConnectionSourceProducer(conn)._statement(_RAW_CAPTURED_ZIP_COORDINATE_SQL, (raw_id,)) as rows:
        return _raw_captured_zip_coordinate_from_row(rows.fetchone())


def read_capture_mode_resolution(conn: sqlite3.Connection, raw_id: str) -> CaptureModeResolution:
    """Read every acquisition mode ever observed for ``raw_id``, explicitly ambiguous or not.

    This is the durable-evidence read surface (AC2): unlike
    ``raw_sessions.capture_mode``, which silently returns only the first-ever
    known mode, this tells a caller whether that value is the sole fact on
    record or one of several colliding observations.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT capture_mode FROM raw_capture_observations
        WHERE raw_id = ?
        ORDER BY first_observed_at_ms, capture_mode
        """,
        (raw_id,),
    ).fetchall()
    modes = tuple(Provider(require_vocabulary(row["capture_mode"], Provider, field="capture_mode")) for row in rows)
    status: CaptureModeResolutionStatus
    if not modes:
        status = "unknown"
    elif len(modes) == 1:
        status = "unambiguous"
    else:
        status = "ambiguous"
    return CaptureModeResolution(raw_id=raw_id, status=status, modes=modes)


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
    origin_value = require_vocabulary(origin, Origin, field="origin")
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


def _revision_values(revision: RawRevisionEnvelope) -> tuple[object, ...]:
    return (
        revision.logical_source_key,
        revision.kind.value,
        revision.source_revision,
        revision.predecessor_source_revision,
        revision.predecessor_raw_id,
        revision.baseline_raw_id,
        revision.append_start_offset,
        revision.append_end_offset,
        revision.acquisition_generation,
        revision.authority.value,
    )


def _is_compatible_classification_refinement(
    existing: tuple[object, ...],
    revision: RawRevisionEnvelope,
) -> bool:
    """Accept retry of a provisional envelope already refined by classification."""
    proposed = _revision_values(revision)
    if revision.authority is not RawRevisionAuthority.QUARANTINED:
        return False
    if existing[9] != RawRevisionAuthority.BYTE_PROVEN.value:
        return False
    # Classification may fill the raw parent/baseline and generation, but it
    # cannot change the acquired stream identity or its byte boundaries.
    for index in (0, 1, 2, 3, 6, 7):
        if existing[index] != proposed[index]:
            return False
    return all(proposed[index] is None or existing[index] == proposed[index] for index in (4, 5))


def _assert_existing_raw_identity(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    origin: str,
    native_id: str | None,
    source_path: str,
    canonical_source_path: str,
    captured_profile_key: str | None,
    new_profile_receipt: bool = False,
    source_index: int,
    blob_hash: bytes,
    blob_size: int,
    revision: RawRevisionEnvelope | None,
) -> None:
    row = conn.execute(
        """
        SELECT origin, native_id, source_path, source_index, blob_hash, blob_size,
               logical_source_key, revision_kind, source_revision,
               predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, acquisition_generation,
               revision_authority, canonical_source_path
        FROM raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"raw insert conflict lost its retained row: {raw_id}")
    values = tuple(row)
    # Bytes and provenance coordinates are the acquisition evidence and must match
    # exactly. ``origin`` is a derived classification: a decoded ingest can sniff a
    # ZIP as a whole and stamp every member, while a source-only replay of the same
    # bytes sees one opaque member and can only say ``unknown-export``. Comparing it
    # as evidence made those routes mutually exclusive over identical bytes. A
    # refinement away from ``unknown-export`` is admitted; two confident but
    # different origins remain a genuine contradiction.
    if (
        values[1:6] != (native_id, source_path, source_index, blob_hash, blob_size)
        or values[-1] != canonical_source_path
    ):
        raise ValueError(f"raw id is already bound to different acquisition evidence: {raw_id}")
    stored_origin = values[0]
    unknown_origin = Origin.UNKNOWN_EXPORT.value
    if stored_origin != origin and unknown_origin not in (stored_origin, origin):
        raise ValueError(
            f"raw id is already bound to a conflicting origin: {raw_id} (stored={stored_origin!r}, incoming={origin!r})"
        )
    if revision is not None and values[6:-1] != _revision_values(revision):
        raise ValueError(f"raw id is already bound to a different revision envelope: {raw_id}")
    record_raw_profile_identity(
        conn, raw_id=raw_id, profile_key=captured_profile_key, allow_new_receipt=new_profile_receipt
    )


def require_profile_identity_key(value: str) -> str:
    """Validate the existing Hermes profile qualifier at its durable boundary."""
    if len(value) != 12 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("profile identity must be a 12-character lowercase hexadecimal qualifier")
    return value


def record_raw_profile_identity(
    conn: sqlite3.Connection, *, raw_id: str, profile_key: str | None, allow_new_receipt: bool = False
) -> None:
    """Retain acquisition's qualifier without changing an existing raw receipt."""
    if profile_key is None:
        return
    profile_key = require_profile_identity_key(profile_key)
    existing = conn.execute(
        "SELECT profile_key FROM raw_profile_identity_receipts WHERE raw_id = ?", (raw_id,)
    ).fetchone()
    if existing is not None and existing[0] != profile_key:
        raise ValueError(f"raw id is already bound to a different profile identity: {raw_id}")
    if existing is None and not allow_new_receipt:
        raise ValueError(f"retained raw is missing its original profile identity receipt: {raw_id}")
    conn.execute(
        "INSERT INTO raw_profile_identity_receipts(raw_id, profile_key) VALUES (?, ?) ON CONFLICT(raw_id) DO NOTHING",
        (raw_id, profile_key),
    )


def read_raw_profile_identity(conn: sqlite3.Connection, raw_id: str) -> str | None:
    """Read only the retained receipt; absence never permits path discovery."""
    with _ConnectionSourceProducer(conn)._statement(_RAW_PROFILE_IDENTITY_SQL, (raw_id,)) as rows:
        return _raw_profile_identity_from_row(rows.fetchone())


def _backfill_raw_file_mtime(conn: sqlite3.Connection, *, raw_id: str, file_mtime_ms: int | None) -> None:
    """Fill an unknown source mtime without changing established evidence."""
    if file_mtime_ms is not None:
        conn.execute(
            "UPDATE raw_sessions SET file_mtime_ms = ? WHERE raw_id = ? AND file_mtime_ms IS NULL",
            (file_mtime_ms, raw_id),
        )


def apply_source_raw_state_update(
    conn: sqlite3.Connection,
    raw_id: str,
    *,
    state: RawSessionStateUpdate,
    manage_transaction: bool = True,
) -> None:
    """Apply the canonical typed producer on its actual Source transaction."""
    _apply_source_raw_state_update(
        _ConnectionSourceProducer(conn), raw_id, state=state, manage_transaction=manage_transaction
    )


def refine_raw_origin(conn: sqlite3.Connection, *, raw_id: str, origin: Origin | str) -> None:
    """Replace a placeholder ``unknown-export`` origin with a confident one.

    Origin is a derived classification rather than acquisition evidence, and
    routes derive it with different information: a decoded ingest sniffs a ZIP
    archive as a whole, while a source-only replay of the same bytes sees one
    opaque member. When the better-informed route re-observes bytes already
    admitted under the placeholder, the row converges upward. Guarded on the
    placeholder so a confident origin is never silently overwritten; a genuine
    contradiction is raised by the identity assertions instead.
    """
    origin_value = require_vocabulary(origin, Origin, field="origin")
    if origin_value == Origin.UNKNOWN_EXPORT.value:
        return
    conn.execute(
        "UPDATE raw_sessions SET origin = ? WHERE raw_id = ? AND origin = ?",
        (origin_value, raw_id, Origin.UNKNOWN_EXPORT.value),
    )


def _assert_additional_blob_refs_admissible(
    conn: sqlite3.Connection,
    additional_blob_refs: tuple[ArchiveSourceBlobRef, ...],
    *,
    policy_snapshot: ExcisionPolicySnapshot | None,
) -> None:
    """Gate every sibling attachment/sidecar reference on the excision ledger.

    Session excision records every sibling attachment and sidecar hash that
    no other session still references, not just the session payload's (a
    hash another session shares is never marked, so this gate does not
    refuse that session's own re-ingest). Checking only the primary hash left
    ``additional_blob_refs`` as the one unguarded way back in: a parsed-session
    ingest carrying an excised attachment inserted its ``blob_refs`` row and
    consumed its publication receipt, making bytes the operator durably excised
    live again under a new raw record. ``write_source_blob_refs`` already
    refuses per reference and names the hash; this is the same gate for the
    references the raw-session writers take inline.

    Runs before the writers' transaction so a refusal leaves no side effect.
    """
    for ref in additional_blob_refs:
        ref_source_path = ref.source_path or f"blob_ref:{ref.ref_type}"
        _assert_excision_policy(ref.blob_hash, source_path=ref_source_path, policy_snapshot=policy_snapshot)
        if is_blob_hash_excised(conn, ref.blob_hash):
            raise ContentExcisedError(blob_hash=ref.blob_hash, source_path=ref_source_path)


def write_source_blob_refs(
    conn: sqlite3.Connection,
    raw_id: str,
    refs: Callable[[], Iterable[ArchiveSourceBlobRef]],
) -> None:
    """Attach already-published blobs in the caller's Source transaction.

    The repeatable cursor preflights and writes closed pages without retaining
    every reference. The caller commits or rolls back the complete operation;
    this helper never ends its transaction.

    Gated on the durable excision ledger like every other writer in this
    module (``write_source_raw_session``,
    ``write_source_raw_session_blob_ref`` -- including the sibling references
    they take through ``additional_blob_refs``). This was the one
    blob-reference
    writer that was not: the Drive attachment convergence stage re-downloads
    an ``unfetched`` reference, hashes it back to the exact excised hash, and
    used this function to recreate a live ``blob_refs`` row for content the
    operator durably excised.

    The refusal is per reference and names the hash, so a caller reports
    which item it did not attach rather than dropping it silently.
    """
    _write_source_blob_refs(_ConnectionSourceProducer(conn), raw_id, refs)


def _require_canonical_source_path(canonical_source_path: object) -> None:
    """Refuse a raw row without its frozen canonical source path.

    Live selection and frontier admission match raws by the canonical path
    acquisition froze; a row without one makes that authority unavailable for
    the whole archive, so the write boundary refuses it instead.
    """
    if not isinstance(canonical_source_path, str) or not canonical_source_path:
        raise ValueError("a retained raw requires the canonical source path acquisition froze")


def write_source_raw_session(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    capture_mode: Provider | str | None = None,
    source_path: str,
    canonical_source_path: str,
    captured_profile_key: str | None = None,
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
    blob_publication_receipt_id: str | None = None,
    additional_blob_refs: tuple[ArchiveSourceBlobRef, ...] = (),
    artifact: ArchiveSourceArtifact | None = None,
    hook_event: ArchiveHookEvent | None = None,
    revision: RawRevisionEnvelope | None = None,
    manage_transaction: bool = True,
    policy_snapshot: ExcisionPolicySnapshot | None = None,
) -> str:
    """Insert one raw session and its required raw-payload blob reference.

    By default the write runs in its own transaction (``with conn:``) committed
    on success. A bulk caller that batches many sessions into one transaction —
    to amortize the per-commit fsync and WAL churn that dominate re-ingest I/O —
    passes ``manage_transaction=False`` and owns the surrounding commit and any
    rollback-on-error itself.
    """
    _require_canonical_source_path(canonical_source_path)
    conn.execute("PRAGMA foreign_keys = ON")
    origin_value = require_vocabulary(origin, Origin, field="origin")
    if origin_value is None:
        raise ValueError("origin is required for raw sessions")
    blob_hash = deterministic_blob_hash(payload)
    _assert_excision_policy(blob_hash, source_path=source_path, policy_snapshot=policy_snapshot)
    if is_blob_hash_excised(conn, blob_hash):
        raise ContentExcisedError(blob_hash=blob_hash, source_path=source_path)
    _assert_additional_blob_refs_admissible(conn, additional_blob_refs, policy_snapshot=policy_snapshot)
    blob_size = len(payload)
    resolved_raw_id = raw_id or deterministic_raw_session_id(
        origin,
        source_path,
        source_index,
        blob_hash,
        native_id,
    )

    with conn if manage_transaction else nullcontext():
        raw_inserted = _insert_raw_session(
            _ConnectionSourceProducer(conn),
            resolved_raw_id,
            columns=_RAW_SESSION_INSERT_COLUMNS,
            values=(
                resolved_raw_id,
                origin_value,
                require_vocabulary(capture_mode, Provider, field="capture_mode") if capture_mode is not None else None,
                native_id,
                source_path,
                canonical_source_path,
                source_index,
                blob_hash,
                blob_size,
                acquired_at_ms,
                file_mtime_ms,
                parsed_at_ms,
                parse_error,
                validated_at_ms,
                require_vocabulary(validation_status, ValidationStatus, field="validation_status")
                if validation_status is not None
                else None,
                validation_error,
                validation_drift_count,
                require_vocabulary(validation_mode, ValidationMode, field="validation_mode")
                if validation_mode is not None
                else None,
                _json_dumps(detection_warnings),
                revision.logical_source_key if revision else None,
                revision.kind.value if revision else "unknown",
                revision.source_revision if revision else None,
                revision.predecessor_source_revision if revision else None,
                revision.predecessor_raw_id if revision else None,
                revision.baseline_raw_id if revision else None,
                revision.append_start_offset if revision else None,
                revision.append_end_offset if revision else None,
                revision.acquisition_generation if revision else None,
                revision.authority.value if revision else "quarantined",
            ),
        )
        # Validate the complete acquisition identity before the NULL-only mtime
        # backfill or any observation/blob reference side effects. This keeps
        # caller-owned transactions atomic when a raw_id conflicts.
        _assert_existing_raw_identity(
            conn,
            raw_id=resolved_raw_id,
            origin=origin_value,
            native_id=native_id,
            source_path=source_path,
            canonical_source_path=canonical_source_path,
            captured_profile_key=captured_profile_key,
            new_profile_receipt=raw_inserted,
            source_index=source_index,
            blob_hash=blob_hash,
            blob_size=blob_size,
            revision=revision,
        )
        _backfill_raw_file_mtime(conn, raw_id=resolved_raw_id, file_mtime_ms=file_mtime_ms)
        if capture_mode is not None:
            conn.execute(
                "UPDATE raw_sessions SET capture_mode = ? WHERE raw_id = ? AND capture_mode IS NULL",
                (require_vocabulary(capture_mode, Provider, field="capture_mode"), resolved_raw_id),
            )
        record_capture_mode_observation(
            conn,
            raw_id=resolved_raw_id,
            capture_mode=capture_mode,
            observed_at_ms=acquired_at_ms,
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
                publication_receipt_id=blob_publication_receipt_id,
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
                    publication_receipt_id=blob_ref.publication_receipt_id,
                ),
            )
        if artifact is not None:
            _insert_artifact(conn, resolved_raw_id, artifact)
        if hook_event is not None:
            # This hook event shares the session's own raw-payload blob (the
            # same bytes just inserted above as a real raw_sessions row), so
            # it needs no blob ref of its own -- unlike write_source_hook_event
            # below, resolved_raw_id here is a genuine raw_sessions row.
            _insert_hook_event(conn, hook_event, blob_hash=blob_hash)

        return resolved_raw_id


def write_source_hook_event(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    source_path: str,
    payload: bytes,
    acquired_at_ms: int,
    raw_id: str,
    hook_event: ArchiveHookEvent,
    blob_publication_receipt_id: str | None = None,
    carrier_source_id: str = "representative-hook-source",
    carrier_relative_path: str | None = None,
    carrier_role: str = "primary-writable",
    manage_transaction: bool = True,
    policy_snapshot: ExcisionPolicySnapshot | None = None,
) -> str:
    """Persist a hook event WITHOUT minting a session.

    A hook event (PreToolUse / PostToolUse / UserPromptSubmit / …) is evidence
    *within* an existing session (identified by ``hook_event.session_native_id``),
    never a conversation of its own. It therefore gets a durable ``hook_payload``
    blob ref (so blob GC + copy-forward treat it like any other retained bytes)
    and a ``raw_hook_events`` row, but — unlike :func:`write_source_raw_session` —
    NO ``raw_sessions`` row. Writing a ``raw_sessions`` row per hook is exactly
    the defect that inflated the archive with tens of thousands of empty
    session shells (polylogue-31r1); ``raw_hook_events`` has no FK to
    ``raw_sessions``, so the linkage stands on its own via ``session_native_id``.

    ``raw_id`` is a content-derived identifier kept for the caller's return
    value / logging only. It is deliberately NOT used as the blob ref's
    ``ref_id`` (polylogue-tfzw0): no ``raw_sessions`` row is ever minted here,
    so a ``raw_id``-keyed ref could never join to a live referent and blob GC
    would retain it forever, uncounted. The blob ref is keyed by
    ``ref_type='hook_payload'`` / ``ref_id=hook_event.hook_event_id`` instead,
    which really is this row's primary key in ``raw_hook_events``.
    """
    carrier_role_value = cast(
        HookCarrierRole, require_vocabulary(carrier_role, _HOOK_CARRIER_ROLES, field="carrier_role")
    )
    conn.execute("PRAGMA foreign_keys = ON")
    if require_vocabulary(origin, Origin, field="origin") is None:
        raise ValueError("origin is required for hook events")
    blob_hash = deterministic_blob_hash(payload)
    _assert_excision_policy(blob_hash, source_path=source_path, policy_snapshot=policy_snapshot)
    if is_blob_hash_excised(conn, blob_hash):
        raise ContentExcisedError(blob_hash=blob_hash, source_path=source_path)
    blob_size = len(payload)
    relative_path = carrier_relative_path or source_path
    with conn if manage_transaction else nullcontext():
        _insert_hook_event(conn, hook_event, blob_hash=blob_hash)
        _insert_blob_ref(
            conn,
            ArchiveSourceBlobRef(
                blob_hash=blob_hash,
                raw_id=hook_event.hook_event_id,
                ref_type="hook_payload",
                source_path=source_path,
                size_bytes=blob_size,
                acquired_at_ms=acquired_at_ms,
                publication_receipt_id=blob_publication_receipt_id,
            ),
        )
        _insert_hook_event_carrier(
            conn,
            source_id=carrier_source_id,
            relative_path=relative_path,
            hook_event=hook_event,
            blob_hash=blob_hash,
            role=carrier_role_value,
            admitted_at_ms=acquired_at_ms,
        )
    return raw_id


@dataclass(frozen=True, slots=True)
class CarrierHookEvent:
    """One hook event and its byte coordinate inside its NDJSON carrier."""

    byte_offset: int
    line_bytes: int
    event: ArchiveHookEvent


def hook_carrier_coordinate(relative_path: str, byte_offset: int, hook_event_id: str) -> str:
    """The durable carrier coordinate of one event inside an NDJSON carrier.

    ``hook_event_carriers`` is keyed by ``(source_id, relative_path)`` because
    a file-per-event spool made the file the event. A carrier holds many
    events, so the coordinate that identifies one of them is the file *plus*
    the byte offset its line starts at and the event identity. Growth cannot
    renumber existing events, and recreating the path cannot alias a new
    event onto an old event's physical position.
    """

    # The producer can recreate a lost carrier at the same day/PID path.
    # A different event at the old byte position retains its own coordinate;
    # unchanged events keep the same coordinate through append and replay.
    return f"{relative_path}#{byte_offset:012d}:{hook_event_id}"


def hook_event_payload_digest(event: ArchiveHookEvent) -> bytes:
    """Digest exactly the canonical payload the Source hook writer stores."""
    return hashlib.sha256(_json_dumps(event.payload).encode()).digest()


def write_source_hook_event_batch(
    conn: sqlite3.Connection,
    *,
    carrier_source_id: str,
    carrier_relative_path: str,
    carrier_role: str,
    carrier_blob_hash: bytes,
    carrier_source_path: str,
    events: Sequence[CarrierHookEvent],
    acquired_at_ms: int,
    blob_publication_receipt_id: str | None = None,
    manage_transaction: bool = True,
    policy_snapshot: ExcisionPolicySnapshot | None = None,
) -> int:
    """Persist every hook event carried by one NDJSON carrier, in one transaction.

    The carrier's bytes are already durable -- they were published once as the
    carrier's own raw acquisition -- so nothing here writes a blob. Each event
    takes a ``hook_payload`` reference to that one carrier blob, which keeps
    blob liveness and excision reaching hook evidence exactly as they did when
    every event had a blob of its own, without paying a blob write, a
    publication reservation and two fsyncs per event.

    No ``raw_sessions`` row is written for an event: a hook event is evidence
    within a session, never a session (polylogue-31r1). The carrier itself has
    a raw row, because the carrier is an acquired artifact.
    """

    carrier_role_value = cast(
        HookCarrierRole, require_vocabulary(carrier_role, _HOOK_CARRIER_ROLES, field="carrier_role")
    )
    # Validate the complete batch before its first event write; callers may
    # already own a transaction and catch the refusal locally.
    for carried in events:
        require_vocabulary(carried.event.origin, Origin, field="hook_event.origin")
    conn.execute("PRAGMA foreign_keys = ON")
    _assert_excision_policy(carrier_blob_hash, source_path=carrier_source_path, policy_snapshot=policy_snapshot)
    if is_blob_hash_excised(conn, carrier_blob_hash):
        raise ContentExcisedError(blob_hash=carrier_blob_hash, source_path=carrier_source_path)
    from polylogue.storage.blob_publication import consume_blob_publication_receipt

    coordinates = [
        hook_carrier_coordinate(carrier_relative_path, carried.byte_offset, carried.event.hook_event_id)
        for carried in events
    ]
    with conn if manage_transaction else nullcontext():
        # Every read and write is one statement per page or per batch, not per
        # event: Source statements are re-authorized at every prepare, so a
        # per-event statement family made the authorizer the route's cost.
        recorded_events = _recorded_hook_events(conn, [carried.event.hook_event_id for carried in events])
        recorded_carriers = _recorded_hook_carriers(conn, carrier_source_id, coordinates)
        event_rows: list[tuple[object, ...]] = []
        ref_rows: list[tuple[object, ...]] = []
        carrier_rows: list[tuple[object, ...]] = []
        for carried, coordinate in zip(events, coordinates, strict=True):
            event = carried.event
            # A carrier grows, so a later revision retains a superset of an
            # earlier one's bytes under a different blob hash. The event is the
            # same evidence either way, and its FIRST-observed carrier blob is
            # what the archive already recorded -- exactly the convention
            # source-tier v36 set when it added this relation. Re-materializing
            # therefore keeps the recorded blob rather than conflicting on it;
            # a genuine disagreement about the event's own content still raises.
            recorded_event = recorded_events.get(event.hook_event_id)
            recorded_carrier = recorded_carriers.get(coordinate)
            observed_blob_hash = (
                recorded_event[6]
                if recorded_event is not None and recorded_event[6] is not None
                else (recorded_carrier[1] if recorded_carrier is not None else None)
            )
            blob_hash = observed_blob_hash or carrier_blob_hash
            payload_json = _json_dumps(event.payload)
            incoming_event = (
                require_vocabulary(event.origin, Origin, field="hook_event.origin"),
                event.native_id,
                event.session_native_id,
                event.event_type,
                payload_json,
                event.observed_at_ms,
                blob_hash,
            )
            if recorded_event is not None:
                if recorded_event != incoming_event:
                    raise HookEventConflictError(f"hook event conflict for {event.hook_event_id}")
            else:
                recorded_events[event.hook_event_id] = incoming_event
                event_rows.append((event.hook_event_id, *incoming_event[:3], event.source_path, *incoming_event[3:]))
            ref_rows.append(
                (
                    blob_hash,
                    event.hook_event_id,
                    "hook_payload",
                    carrier_source_path,
                    carried.line_bytes,
                    acquired_at_ms,
                )
            )
            incoming_carrier = (
                event.hook_event_id,
                blob_hash,
                hook_event_payload_digest(event),
                carrier_role_value,
            )
            if recorded_carrier is not None:
                if recorded_carrier != incoming_carrier:
                    raise HookEventConflictError(f"hook carrier conflict for {carrier_source_id}:{coordinate}")
            else:
                recorded_carriers[coordinate] = incoming_carrier
                carrier_rows.append((carrier_source_id, coordinate, *incoming_carrier, acquired_at_ms))
        conn.executemany(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id, source_path, event_type,
                payload_json, observed_at_ms, blob_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            event_rows,
        )
        # Hook coordinates keep their first observation.
        conn.executemany(
            "INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            ref_rows,
        )
        for consumed_blob_hash in dict.fromkeys(row[0] for row in ref_rows):
            consume_blob_publication_receipt(conn, blob_publication_receipt_id, cast(bytes, consumed_blob_hash))
        conn.executemany(
            "INSERT INTO hook_event_carriers "
            "(source_id, relative_path, hook_event_id, blob_hash, payload_digest, carrier_role, admitted_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            carrier_rows,
        )
    return len(events)


#: Page size of one recorded-identity read; paging bounds the statement's
#: variable count, never the batch it reads.
_HOOK_IDENTITY_PAGE = 500


def _recorded_hook_events(conn: sqlite3.Connection, hook_event_ids: Sequence[str]) -> dict[str, tuple[object, ...]]:
    """The recorded semantics of every listed hook event that already exists."""
    recorded: dict[str, tuple[object, ...]] = {}
    distinct = tuple(dict.fromkeys(hook_event_ids))
    for offset in range(0, len(distinct), _HOOK_IDENTITY_PAGE):
        page = distinct[offset : offset + _HOOK_IDENTITY_PAGE]
        placeholders = ",".join("?" for _ in page)
        for row in conn.execute(
            "SELECT hook_event_id, origin, native_id, session_native_id, event_type, payload_json, "
            f"observed_at_ms, blob_hash FROM raw_hook_events WHERE hook_event_id IN ({placeholders})",
            page,
        ):
            recorded[str(row[0])] = (
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                None if row[7] is None else bytes(row[7]),
            )
    return recorded


def _recorded_hook_carriers(
    conn: sqlite3.Connection, source_id: str, relative_paths: Sequence[str]
) -> dict[str, tuple[object, ...]]:
    """The recorded carrier coordinate of every listed path that already exists."""
    recorded: dict[str, tuple[object, ...]] = {}
    distinct = tuple(dict.fromkeys(relative_paths))
    for offset in range(0, len(distinct), _HOOK_IDENTITY_PAGE):
        page = distinct[offset : offset + _HOOK_IDENTITY_PAGE]
        placeholders = ",".join("?" for _ in page)
        for row in conn.execute(
            "SELECT relative_path, hook_event_id, blob_hash, payload_digest, carrier_role FROM hook_event_carriers "
            f"WHERE source_id = ? AND relative_path IN ({placeholders})",
            (source_id, *page),
        ):
            recorded[str(row[0])] = (row[1], bytes(row[2]), bytes(row[3]), row[4])
    return recorded


def delete_source_hook_event(
    conn: sqlite3.Connection,
    hook_event_id: str,
    *,
    manage_transaction: bool = True,
) -> bool:
    """Delete one hook event and its owned payload ref as one source-tier write.

    Hook events do not have a foreign key to ``blob_refs`` because the source
    tier stores the reference by ``ref_type``/``ref_id``. Keep their delete
    route paired with the reference cleanup so a removed event cannot leave a
    durable ``hook_payload`` row behind to pin its blob.
    """
    conn.execute("PRAGMA foreign_keys = ON")
    with conn if manage_transaction else nullcontext():
        conn.execute("DELETE FROM hook_event_carriers WHERE hook_event_id = ?", (hook_event_id,))
        cursor = conn.execute("DELETE FROM raw_hook_events WHERE hook_event_id = ?", (hook_event_id,))
        conn.execute(
            "DELETE FROM blob_refs WHERE ref_type = 'hook_payload' AND ref_id = ?",
            (hook_event_id,),
        )
    return cursor.rowcount == 1


def write_source_raw_session_blob_ref(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    capture_mode: Provider | str | None = None,
    source_path: str,
    canonical_source_path: str,
    captured_profile_key: str | None = None,
    source_index: int,
    blob_hash: bytes,
    blob_size: int,
    acquired_at_ms: int,
    file_mtime_ms: int | None = None,
    native_id: str | None = None,
    raw_id: str | None = None,
    blob_publication_receipt_id: str | None = None,
    additional_blob_refs: tuple[ArchiveSourceBlobRef, ...] = (),
    artifact: ArchiveSourceArtifact | None = None,
    revision: RawRevisionEnvelope | None = None,
    manage_transaction: bool = True,
    policy_snapshot: ExcisionPolicySnapshot | None = None,
) -> str:
    """Insert one raw session that already has a materialized raw-payload blob.

    See :func:`write_source_raw_session` for the ``manage_transaction`` contract.
    """
    if len(blob_hash) != 32:
        raise ValueError("blob_hash must be a 32-byte SHA-256 digest")
    _require_canonical_source_path(canonical_source_path)
    conn.execute("PRAGMA foreign_keys = ON")
    origin_value = require_vocabulary(origin, Origin, field="origin")
    if origin_value is None:
        raise ValueError("origin is required for raw sessions")
    _assert_excision_policy(blob_hash, source_path=source_path, policy_snapshot=policy_snapshot)
    if is_blob_hash_excised(conn, blob_hash):
        raise ContentExcisedError(blob_hash=blob_hash, source_path=source_path)
    _assert_additional_blob_refs_admissible(conn, additional_blob_refs, policy_snapshot=policy_snapshot)
    resolved_raw_id = raw_id or deterministic_raw_session_id(
        origin,
        source_path,
        source_index,
        blob_hash,
        native_id,
    )
    with conn if manage_transaction else nullcontext():
        raw_inserted = _insert_raw_session(
            _ConnectionSourceProducer(conn),
            resolved_raw_id,
            columns=_RAW_SESSION_BLOB_REF_INSERT_COLUMNS,
            values=(
                resolved_raw_id,
                origin_value,
                require_vocabulary(capture_mode, Provider, field="capture_mode") if capture_mode is not None else None,
                native_id,
                source_path,
                canonical_source_path,
                source_index,
                blob_hash,
                blob_size,
                acquired_at_ms,
                file_mtime_ms,
                revision.logical_source_key if revision else None,
                revision.kind.value if revision else "unknown",
                revision.source_revision if revision else None,
                revision.predecessor_source_revision if revision else None,
                revision.predecessor_raw_id if revision else None,
                revision.baseline_raw_id if revision else None,
                revision.append_start_offset if revision else None,
                revision.append_end_offset if revision else None,
                revision.acquisition_generation if revision else None,
                revision.authority.value if revision else "quarantined",
            ),
        )
        # Validate the complete acquisition identity before the NULL-only mtime
        # backfill or any observation/blob reference side effects. This keeps
        # caller-owned transactions atomic when a raw_id conflicts.
        _assert_existing_raw_identity(
            conn,
            raw_id=resolved_raw_id,
            origin=origin_value,
            native_id=native_id,
            source_path=source_path,
            canonical_source_path=canonical_source_path,
            captured_profile_key=captured_profile_key,
            new_profile_receipt=raw_inserted,
            source_index=source_index,
            blob_hash=blob_hash,
            blob_size=blob_size,
            revision=revision,
        )
        _backfill_raw_file_mtime(conn, raw_id=resolved_raw_id, file_mtime_ms=file_mtime_ms)
        if capture_mode is not None:
            conn.execute(
                "UPDATE raw_sessions SET capture_mode = ? WHERE raw_id = ? AND capture_mode IS NULL",
                (require_vocabulary(capture_mode, Provider, field="capture_mode"), resolved_raw_id),
            )
        record_capture_mode_observation(
            conn,
            raw_id=resolved_raw_id,
            capture_mode=capture_mode,
            observed_at_ms=acquired_at_ms,
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
                publication_receipt_id=blob_publication_receipt_id,
            ),
        )
        for additional_ref in additional_blob_refs:
            _insert_blob_ref(
                conn,
                ArchiveSourceBlobRef(
                    blob_hash=additional_ref.blob_hash,
                    raw_id=resolved_raw_id,
                    ref_type=additional_ref.ref_type,
                    source_path=additional_ref.source_path,
                    size_bytes=additional_ref.size_bytes or 0,
                    acquired_at_ms=additional_ref.acquired_at_ms or acquired_at_ms,
                    publication_receipt_id=additional_ref.publication_receipt_id,
                ),
            )
        if artifact is not None:
            _insert_artifact(conn, resolved_raw_id, artifact)
    return resolved_raw_id


def bind_source_raw_revision(
    conn: sqlite3.Connection,
    raw_id: str,
    revision: RawRevisionEnvelope,
    *,
    manage_transaction: bool = True,
) -> None:
    """Bind acquisition evidence after a payload proves a single session identity.

    ``manage_transaction=False`` batches multiple raws' binds into one
    caller-managed commit window (polylogue-amg1) -- the caller must call
    ``conn.commit()`` (or ``conn.rollback()`` on failure) itself.
    """
    _bind_source_raw_revision(_ConnectionSourceProducer(conn), raw_id, revision, manage_transaction=manage_transaction)


def read_archive_raw_session_envelope(conn: sqlite3.Connection, raw_id: str) -> ArchiveRawSessionEnvelope:
    """Read a compact envelope for one raw source session."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT
            raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash, blob_size,
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

    # Source DDL deliberately leaves durable membership out of its CHECKs.
    # Refuse malformed historical/direct-SQL rows at the typed hydration
    # boundary before exposing the envelope to callers.
    origin = require_vocabulary(row["origin"], Origin, field="origin")
    capture_mode = (
        require_vocabulary(row["capture_mode"], Provider, field="capture_mode")
        if row["capture_mode"] is not None
        else None
    )
    validation_status = (
        require_vocabulary(row["validation_status"], ValidationStatus, field="validation_status")
        if row["validation_status"] is not None
        else None
    )
    validation_mode = (
        require_vocabulary(row["validation_mode"], ValidationMode, field="validation_mode")
        if row["validation_mode"] is not None
        else None
    )

    blob_refs = tuple(
        ArchiveSourceBlobRef(
            blob_hash=row["blob_hash"],
            raw_id=row["raw_id"],
            ref_type=cast(BlobRefType, require_vocabulary(row["ref_type"], _BLOB_REF_TYPES, field="ref_type")),
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
    return ArchiveRawSessionEnvelope(
        raw_id=row["raw_id"],
        origin=origin,
        capture_mode=capture_mode,
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
        validation_status=validation_status,
        validation_error=row["validation_error"],
        validation_drift_count=row["validation_drift_count"],
        validation_mode=validation_mode,
        detection_warnings=tuple(json.loads(row["detection_warnings_json"] or "[]")),
        blob_refs=blob_refs,
        artifact_ids=artifact_ids,
        hook_event_ids=hook_event_ids,
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
    _write_blob_ref(_ConnectionSourceProducer(conn), ref)


def _insert_artifact(conn: sqlite3.Connection, raw_id: str, artifact: ArchiveSourceArtifact) -> None:
    _write_artifact(_ConnectionSourceProducer(conn), raw_id, artifact)


def upsert_raw_artifact(
    conn: sqlite3.Connection,
    raw_id: str,
    artifact: ArchiveSourceArtifact,
    *,
    manage_transaction: bool = True,
) -> None:
    """Publish the ordinary artifact producer on its existing Source transaction."""
    _upsert_raw_artifact(_ConnectionSourceProducer(conn), raw_id, artifact, manage_transaction=manage_transaction)


def _insert_hook_event(
    conn: sqlite3.Connection,
    hook_event: ArchiveHookEvent,
    *,
    blob_hash: bytes | None = None,
) -> None:
    existing = conn.execute(
        "SELECT origin, native_id, session_native_id, source_path, event_type, payload_json, observed_at_ms, blob_hash "
        "FROM raw_hook_events WHERE hook_event_id = ?",
        (hook_event.hook_event_id,),
    ).fetchone()
    incoming = (
        require_vocabulary(hook_event.origin, Origin, field="hook_event.origin"),
        hook_event.native_id,
        hook_event.session_native_id,
        hook_event.event_type,
        _json_dumps(hook_event.payload),
        hook_event.observed_at_ms,
        blob_hash,
    )
    if existing is not None:
        existing_semantics = (existing[0], existing[1], existing[2], existing[4], existing[5], existing[6], existing[7])
        if existing_semantics != incoming:
            raise HookEventConflictError(f"hook event conflict for {hook_event.hook_event_id}")
        return
    conn.execute(
        """
        INSERT INTO raw_hook_events (
            hook_event_id, origin, native_id, session_native_id, source_path, event_type,
            payload_json, observed_at_ms, blob_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (hook_event.hook_event_id, incoming[0], incoming[1], incoming[2], hook_event.source_path, *incoming[3:]),
    )


def _insert_hook_event_carrier(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    relative_path: str,
    hook_event: ArchiveHookEvent,
    blob_hash: bytes,
    role: str,
    admitted_at_ms: int,
) -> None:
    if role not in {"primary-writable", "legacy-read-only"}:
        raise ValueError(f"invalid hook carrier role: {role}")
    existing = conn.execute(
        "SELECT hook_event_id, blob_hash, payload_digest, carrier_role FROM hook_event_carriers "
        "WHERE source_id = ? AND relative_path = ?",
        (source_id, relative_path),
    ).fetchone()
    payload_digest = hashlib.sha256(_json_dumps(hook_event.payload).encode()).digest()
    incoming = (hook_event.hook_event_id, blob_hash, payload_digest, role)
    if existing is not None:
        if tuple(existing) != incoming:
            raise HookEventConflictError(f"hook carrier conflict for {source_id}:{relative_path}")
        return
    conn.execute(
        "INSERT INTO hook_event_carriers "
        "(source_id, relative_path, hook_event_id, blob_hash, payload_digest, carrier_role, admitted_at_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source_id, relative_path, hook_event.hook_event_id, blob_hash, payload_digest, role, admitted_at_ms),
    )


def _raw_artifact_from_row(row: sqlite3.Row) -> ArchiveRawArtifactEnvelope:
    return ArchiveRawArtifactEnvelope(
        artifact_id=row["artifact_id"],
        raw_id=row["raw_id"],
        origin=require_vocabulary(row["origin"], Origin, field="artifact.origin"),
        source_path=row["source_path"],
        source_index=row["source_index"],
        artifact_kind=row["artifact_kind"],
        support_status=require_vocabulary(
            row["support_status"], ArtifactSupportStatus, field="artifact.support_status"
        ),
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
        origin=require_vocabulary(row["origin"], Origin, field="hook_event.origin"),
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
    "ArchiveHookEvent",
    "ArchiveRawArtifactEnvelope",
    "ArchiveRawSessionEnvelope",
    "ArchiveSourceArtifact",
    "ArchiveSourceBlobRef",
    "CaptureModeResolution",
    "CarrierHookEvent",
    "ContentExcisedError",
    "PENDING_RAW_LOGICAL_SOURCE_PREFIX",
    "deterministic_blob_hash",
    "hook_carrier_coordinate",
    "hook_event_payload_digest",
    "deterministic_raw_session_id",
    "is_blob_hash_excised",
    "list_hook_events",
    "list_raw_artifacts",
    "read_capture_mode_resolution",
    "read_hook_event",
    "read_raw_artifact",
    "read_archive_raw_session_envelope",
    "record_capture_mode_observation",
    "record_raw_container_coordinate",
    "record_excised_blob_hash",
    "pending_raw_logical_source_key",
    "upsert_raw_artifact",
    "write_source_hook_event_batch",
    "write_source_raw_session",
    "write_source_raw_session_blob_ref",
]


_RAW_CAPTURED_ZIP_COORDINATE_SQL = "SELECT captured_coordinate FROM raw_container_coordinates WHERE raw_id = ?"


def _raw_captured_zip_coordinate_from_row(row: sqlite3.Row | None) -> CapturedZipMemberCoordinate | None:
    if row is None:
        return None
    if row[0] is None:
        from polylogue.core.raw_failure_evidence import RetainedZipMembershipUnprovedError

        raise RetainedZipMembershipUnprovedError("retained ZIP input lacks its captured namespace/member receipt")
    if not isinstance(row[0], str):
        raise ValueError("captured ZIP coordinate receipt must be text")
    return read_captured_zip_coordinate_receipt(row[0])


_RAW_APPEND_LOGICAL_KEY_SQL = "SELECT logical_source_key FROM raw_sessions WHERE raw_id=? AND revision_kind='append'"


def _raw_append_logical_key_from_row(row: sqlite3.Row | None) -> str | None:
    return None if row is None or row[0] is None else str(row[0])


def read_raw_append_logical_key(conn: sqlite3.Connection, raw_id: str) -> str | None:
    """Read the retained append envelope's declared session key."""
    with _ConnectionSourceProducer(conn)._statement(_RAW_APPEND_LOGICAL_KEY_SQL, (raw_id,)) as rows:
        return _raw_append_logical_key_from_row(rows.fetchone())


_RAW_PROFILE_IDENTITY_SQL = "SELECT profile_key FROM raw_profile_identity_receipts WHERE raw_id = ?"


def _raw_profile_identity_from_row(row: sqlite3.Row | None) -> str | None:
    return None if row is None else require_profile_identity_key(row[0])


class SourceRawStateProducer(Protocol):
    """The typed raw-state writer's literal and exact raw-PK effect boundary."""

    def state_transaction(self, manage_transaction: bool) -> AbstractContextManager[object]: ...
    def state_literal(self, value: object) -> tuple[str, tuple[object, ...]]: ...
    def state_write(self, raw_id: str, sql: str, parameters: tuple[object, ...]) -> int: ...


def _apply_source_raw_state_update(
    producer: SourceRawStateProducer,
    raw_id: str,
    *,
    state: RawSessionStateUpdate,
    manage_transaction: bool = True,
) -> None:
    """Build identical typed SQL with the owning host's exact literal operands."""
    set_clauses, compiled_params = compile_raw_state_update(
        state,
        now_ms=int(datetime.now(UTC).timestamp() * 1000),
        literal=producer.state_literal,
    )
    if not set_clauses:
        return
    raw_expression, raw_parameters = producer.state_literal(raw_id)
    with producer.state_transaction(manage_transaction):
        changed = producer.state_write(
            raw_id,
            f"UPDATE raw_sessions SET {', '.join(set_clauses)} WHERE raw_id = {raw_expression}",
            (*compiled_params, *raw_parameters),
        )
        if changed != 1:
            raise KeyError(raw_id)


def _write_source_blob_refs(
    producer: SourceBlobReferenceProducer,
    raw_id: str,
    refs: Callable[[], Iterable[ArchiveSourceBlobRef]],
) -> None:
    """Share the canonical complete preflight and reference/receipt writes."""
    # Preflight the complete batch so invalid storage-local categories cannot
    # leave earlier refs written in a caller-owned transaction.
    for ref in refs():
        require_vocabulary(ref.ref_type, _BLOB_REF_TYPES, field="ref_type")
        if ref.size_bytes is None or ref.acquired_at_ms is None:
            raise ValueError("size_bytes and acquired_at_ms are required for blob refs")
        if producer.blob_ref_is_excised(ref.blob_hash):
            raise ContentExcisedError(
                blob_hash=ref.blob_hash, source_path=ref.source_path or f"blob_ref:{ref.ref_type}"
            )
    for ref in refs():
        if producer.blob_ref_is_excised(ref.blob_hash):
            raise ContentExcisedError(
                blob_hash=ref.blob_hash,
                source_path=ref.source_path or f"blob_ref:{ref.ref_type}",
            )
        _write_blob_ref(
            producer,
            ArchiveSourceBlobRef(
                blob_hash=ref.blob_hash,
                raw_id=raw_id,
                ref_type=ref.ref_type,
                source_path=ref.source_path,
                size_bytes=ref.size_bytes,
                acquired_at_ms=ref.acquired_at_ms,
                publication_receipt_id=ref.publication_receipt_id,
            ),
        )


class SourceRawSessionInsertProducer(Protocol):
    """The two canonical raw acquisition INSERTs and their exact operands."""

    def raw_insert_literal(self, value: object) -> tuple[str, tuple[object, ...]]: ...
    def raw_insert(
        self,
        raw_id: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]: ...


_RAW_SESSION_INSERT_COLUMNS = (
    "raw_id",
    "origin",
    "capture_mode",
    "native_id",
    "source_path",
    "canonical_source_path",
    "source_index",
    "blob_hash",
    "blob_size",
    "acquired_at_ms",
    "file_mtime_ms",
    "parsed_at_ms",
    "parse_error",
    "validated_at_ms",
    "validation_status",
    "validation_error",
    "validation_drift_count",
    "validation_mode",
    "detection_warnings_json",
    "logical_source_key",
    "revision_kind",
    "source_revision",
    "predecessor_source_revision",
    "predecessor_raw_id",
    "baseline_raw_id",
    "append_start_offset",
    "append_end_offset",
    "acquisition_generation",
    "revision_authority",
)

_RAW_SESSION_BLOB_REF_INSERT_COLUMNS = (
    "raw_id",
    "origin",
    "capture_mode",
    "native_id",
    "source_path",
    "canonical_source_path",
    "source_index",
    "blob_hash",
    "blob_size",
    "acquired_at_ms",
    "file_mtime_ms",
    "logical_source_key",
    "revision_kind",
    "source_revision",
    "predecessor_source_revision",
    "predecessor_raw_id",
    "baseline_raw_id",
    "append_start_offset",
    "append_end_offset",
    "acquisition_generation",
    "revision_authority",
)


def _insert_raw_session(
    producer: SourceRawSessionInsertProducer,
    raw_id: str,
    *,
    columns: tuple[str, ...],
    values: tuple[object, ...],
) -> bool:
    if columns not in {_RAW_SESSION_INSERT_COLUMNS, _RAW_SESSION_BLOB_REF_INSERT_COLUMNS}:
        raise ValueError("raw acquisition requires its canonical INSERT columns")
    if len(values) != len(columns) or values[0] != raw_id:
        raise ValueError("raw acquisition values must match its declared identity and columns")
    operands = tuple(producer.raw_insert_literal(value) for value in values)
    expressions = ", ".join(expression for expression, _parameters in operands)
    parameters = tuple(parameter for _expression, params in operands for parameter in params)
    # SQLite allocates this NULL operand ordinarily and during preparation;
    # the original witness retains its exact root INSERT rowid for replay.
    with producer.raw_insert(
        raw_id,
        f"INSERT INTO raw_sessions (rowid, {', '.join(columns)}) VALUES (?, {expressions}) "
        "ON CONFLICT(raw_id) DO NOTHING",
        (None, *parameters),
    ) as cursor:
        return cursor.rowcount == 1


class RawRevisionBindingProducer(Protocol):
    """The existing raw bind's two declared read/write families."""

    def binding_transaction(self, *, manage_transaction: bool) -> AbstractContextManager[object]: ...

    def binding_literal(self, value: object) -> tuple[str, tuple[object, ...]]: ...

    def update_binding(
        self,
        raw_id: str,
        sql: str,
        parameters: tuple[object, ...],
        *,
        parser_singleton_witness: PreparedParserSingletonWitness | None = None,
    ) -> int: ...

    def read_binding(self, raw_id: str, sql: str, parameters: tuple[object, ...]) -> tuple[object, ...] | None: ...


@dataclass(frozen=True, slots=True)
class _ConnectionSourceProducer:
    connection: sqlite3.Connection

    @contextmanager
    def _statement(self, sql: str, parameters: tuple[object, ...] = ()) -> Generator[sqlite3.Cursor, None, None]:
        with connection_cursor(self.connection, sql, parameters) as cursor:
            yield cursor

    def binding_transaction(self, *, manage_transaction: bool) -> AbstractContextManager[object]:
        return self.connection if manage_transaction else nullcontext()

    def binding_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return "?", (value,)

    def update_binding(
        self,
        raw_id: str,
        sql: str,
        parameters: tuple[object, ...],
        *,
        parser_singleton_witness: PreparedParserSingletonWitness | None = None,
    ) -> int:
        if parser_singleton_witness is not None:
            raise ValueError("a prepared singleton witness requires its original prepared Source producer")
        with self._statement(sql, parameters) as cursor:
            return cursor.rowcount

    def read_binding(self, raw_id: str, sql: str, parameters: tuple[object, ...]) -> tuple[object, ...] | None:
        with self._statement(sql, parameters) as cursor:
            row = cursor.fetchone()
        return None if row is None else tuple(row)

    def membership_decision_write(
        self,
        raw_id: str,
        logical_source_key: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> None:
        with self._statement(sql, parameters):
            pass

    def raw_insert_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return self.binding_literal(value)

    def raw_insert(
        self,
        raw_id: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        return self._statement(sql, parameters)

    def state_transaction(self, manage_transaction: bool) -> AbstractContextManager[object]:
        return self.binding_transaction(manage_transaction=manage_transaction)

    def state_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return raw_state_parameter(value)

    def state_write(self, raw_id: str, sql: str, parameters: tuple[object, ...]) -> int:
        return self.update_binding(raw_id, sql, parameters)

    def artifact_transaction(self, manage_transaction: bool) -> AbstractContextManager[object]:
        return self.connection if manage_transaction else nullcontext()

    def artifact_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return "?", (value,)

    def artifact_coordinate_rows(
        self,
        raw_id: str,
        artifact: ArchiveSourceArtifact,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        sql, parameters = _artifact_coordinate_query(raw_id, artifact, columns="a.artifact_id, a.raw_id")
        return self._statement(sql, parameters)

    def artifact_observation_rows(
        self,
        raw_id: str,
        *,
        receipt: bool,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        sql, parameters = _artifact_observation_query(raw_id, receipt=receipt)
        return self._statement(sql, parameters)

    def artifact_validation_failed(self, raw_id: str) -> bool:
        with self._statement(_ARTIFACT_VALIDATION_STATUS_SQL, (raw_id,)) as rows:
            row = rows.fetchone()
        return row is not None and str(row[0] or "") == "failed"

    def artifact_write(
        self,
        sql: str,
        parameters: tuple[object, ...],
        artifact_id: str,
        *,
        allocation: bool,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        return self._statement(sql, parameters)

    def blob_ref_is_excised(self, blob_hash: bytes) -> bool:
        return is_blob_hash_excised(self.connection, blob_hash)

    def blob_ref_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return self.binding_literal(value)

    def blob_ref_write(
        self,
        ref: ArchiveSourceBlobRef,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        return self._statement(sql, parameters)

    def consume_reference_receipt(self, ref: ArchiveSourceBlobRef) -> None:
        from polylogue.storage.blob_publication import consume_blob_publication_receipt

        consume_blob_publication_receipt(self.connection, ref.publication_receipt_id, ref.blob_hash)


_RAW_REVISION_BINDING_COLUMNS = (
    "logical_source_key, revision_kind, source_revision, predecessor_source_revision, "
    "predecessor_raw_id, baseline_raw_id, append_start_offset, append_end_offset, "
    "acquisition_generation, revision_authority"
)

_RAW_REVISION_BINDING_SQL = f"SELECT {_RAW_REVISION_BINDING_COLUMNS} FROM raw_sessions WHERE raw_id=?"

_RAW_SINGLETON_REVISION_BINDING_SQL = (
    f"SELECT {_RAW_REVISION_BINDING_COLUMNS}, "
    "EXISTS(SELECT 1 FROM raw_session_memberships WHERE raw_id=raw_sessions.raw_id) "
    "FROM raw_sessions WHERE raw_id=?"
)


@dataclass(frozen=True, slots=True)
class PreparedParserSingletonRevision:
    """The canonical singleton's predicted binding, before any Source write."""

    raw_id: str
    before_binding: tuple[object, ...]
    revision: RawRevisionEnvelope


@dataclass(frozen=True, slots=True)
class PreparedParserSingletonWitness:
    """One original producer's already-prepared singleton identity operand."""

    seal: PreparedIndexMutation
    producer_identity: object
    raw_id: str
    blob_hash: bytes
    parser_fingerprint: str
    logical_source_key: str
    before_binding: tuple[object, ...]
    prepared_output: Sequence[ParsedSession]


def _prepare_parser_singleton_revision(
    producer: RawRevisionBindingProducer,
    raw_id: str,
    logical_source_key: str,
) -> PreparedParserSingletonRevision | None:
    """Refine a parser-proved singleton while preserving acquired FULL bytes.

    Pending acquisition is already FULL. Its source revision and byte-chain
    coordinates remain the actual acquired envelope, rather than being
    replaced by the fallback identity used for an UNKNOWN raw.
    """
    if canonical_authority_logical_key(logical_source_key) != logical_source_key:
        raise ValueError("parser singleton binding requires its canonical logical source key")
    existing = producer.read_binding(raw_id, _RAW_SINGLETON_REVISION_BINDING_SQL, (raw_id,))
    if existing is None:
        raise ValueError(f"parser revision binding found no raw row: {raw_id}")
    # Membership authority was prepared by its own canonical census. A
    # receipt refresh must not promote that normalized raw back to FULL.
    if bool(existing[10]):
        return None
    kind = RawRevisionKind(str(existing[1]))
    pending = isinstance(existing[0], str) and existing[0].startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX)
    if kind is RawRevisionKind.UNKNOWN:
        if tuple(existing[:10]) != (None, "unknown", None, None, None, None, None, None, None, "quarantined"):
            raise ValueError("parser singleton binding requires the original unclassified raw envelope")
        revision = RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=RawRevisionKind.FULL,
            source_revision=raw_id,
            acquisition_generation=0,
            authority=RawRevisionAuthority.QUARANTINED,
        )
    elif kind is RawRevisionKind.FULL and pending:
        if not isinstance(existing[2], str):
            raise ValueError("pending FULL parser input has no acquired source revision")
        generation = existing[8]
        if type(generation) is not int:
            raise ValueError("pending FULL parser input has no canonical integer generation")
        revision = RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=kind,
            source_revision=existing[2],
            predecessor_source_revision=cast(str | None, existing[3]),
            predecessor_raw_id=cast(str | None, existing[4]),
            baseline_raw_id=cast(str | None, existing[5]),
            append_start_offset=cast(int | None, existing[6]),
            append_end_offset=cast(int | None, existing[7]),
            acquisition_generation=generation,
            authority=RawRevisionAuthority(str(existing[9])),
        )
    else:
        return None
    return PreparedParserSingletonRevision(raw_id, tuple(existing[:10]), revision)


def prepare_parser_singleton_witness(
    seal: PreparedIndexMutation,
    prepared: PreparedParserSingletonRevision,
    *,
    prepared_output: Sequence[ParsedSession],
    parser_fingerprint: str,
) -> PreparedParserSingletonWitness:
    """Bind the actual prepared output to the pinned original Raw and phase."""
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.prepared_jsonl import PreparedSessionSequence

    if not isinstance(prepared_output, PreparedSessionSequence):
        raise ValueError("parser singleton witness requires the original prepared artifact output")
    prepared_output.artifact.verify_files(full=False)
    if prepared_output.artifact.error is not None:
        raise ValueError("parser singleton witness cannot accept a failed prepared artifact")
    if parser_fingerprint != raw_authority_parser_fingerprint():
        raise ValueError("parser singleton witness requires the current parser fingerprint")
    if len(prepared_output) != 1 or not isinstance(prepared_output[0], ParsedSession):
        raise ValueError("parser singleton witness requires exactly one prepared session")
    session = prepared_output[0]
    observed_key = canonical_authority_logical_key(f"{session.source_name.value}:{session.provider_session_id}")
    if observed_key != prepared.revision.logical_source_key:
        raise ValueError("parser singleton witness does not match its prepared output")
    producer_identity = seal.source_producer_identity
    if producer_identity is None:
        raise ValueError("parser singleton witness requires its active original producer")
    with seal.original_rows(
        "source",
        f"SELECT {_RAW_REVISION_BINDING_COLUMNS}, blob_hash FROM raw_sessions WHERE raw_id=?",
        (prepared.raw_id,),
    ) as rows:
        original = rows.fetchone()
    if original is None or tuple(original[:10]) != prepared.before_binding:
        raise ValueError("parser singleton witness does not match the original Raw binding")
    blob_hash = original[10]
    if not isinstance(blob_hash, bytes) or len(blob_hash) != 32:
        raise ValueError("parser singleton witness requires its original acquired blob identity")
    if prepared_output.artifact.blob_hash != blob_hash.hex():
        raise ValueError("parser singleton witness output belongs to different acquired bytes")
    return PreparedParserSingletonWitness(
        seal,
        producer_identity,
        prepared.raw_id,
        blob_hash,
        parser_fingerprint,
        observed_key,
        prepared.before_binding,
        prepared_output,
    )


def _bind_parser_singleton_revision(
    producer: RawRevisionBindingProducer,
    raw_id: str,
    logical_source_key: str,
    *,
    witness: PreparedParserSingletonWitness,
) -> bool:
    """Apply the canonical prediction only with its exact prepared operand."""
    prepared = _prepare_parser_singleton_revision(producer, raw_id, logical_source_key)
    if prepared is None:
        return False
    if (
        witness.raw_id != raw_id
        or witness.before_binding != prepared.before_binding
        or witness.logical_source_key != logical_source_key
        or witness.parser_fingerprint != raw_authority_parser_fingerprint()
        or witness.producer_identity is not witness.seal.source_producer_identity
        or len(witness.prepared_output) != 1
        or canonical_authority_logical_key(
            f"{witness.prepared_output[0].source_name.value}:{witness.prepared_output[0].provider_session_id}"
        )
        != logical_source_key
    ):
        raise ValueError("parser singleton binding lacks its exact original prepared witness")
    _bind_source_raw_revision(
        producer,
        raw_id,
        prepared.revision,
        manage_transaction=False,
        parser_singleton_witness=witness,
    )
    return True


def _bind_source_raw_revision(
    producer: RawRevisionBindingProducer,
    raw_id: str,
    revision: RawRevisionEnvelope,
    *,
    manage_transaction: bool,
    parser_singleton_witness: PreparedParserSingletonWitness | None = None,
) -> None:
    """Run the same canonical bind against its actual declared host."""
    with producer.binding_transaction(manage_transaction=manage_transaction):
        operands = tuple(
            producer.binding_literal(value)
            for value in (*_revision_values(revision), raw_id, f"{PENDING_RAW_LOGICAL_SOURCE_PREFIX}%")
        )
        (
            logical_key,
            kind,
            source_revision,
            predecessor_revision,
            predecessor_raw,
            baseline_raw,
            append_start,
            append_end,
            generation,
            authority,
            raw_key,
            pending_prefix,
        ) = (expression for expression, _parameters in operands)
        binding_options = (
            {"parser_singleton_witness": parser_singleton_witness} if parser_singleton_witness is not None else {}
        )
        updated_row_count = producer.update_binding(
            raw_id,
            f"""
            UPDATE raw_sessions
            SET logical_source_key = {logical_key}, revision_kind = {kind}, source_revision = {source_revision},
                predecessor_source_revision = {predecessor_revision}, predecessor_raw_id = {predecessor_raw},
                baseline_raw_id = {baseline_raw}, append_start_offset = {append_start},
                append_end_offset = {append_end}, acquisition_generation = {generation}, revision_authority = {authority}
            WHERE raw_id = {raw_key}
              AND (
                  (
                      revision_kind = 'unknown'
                      AND revision_authority = 'quarantined'
                      AND logical_source_key IS NULL
                      AND source_revision IS NULL
                  )
                  OR logical_source_key LIKE {pending_prefix}
              )
            """,
            tuple(parameter for _expression, parameters in operands for parameter in parameters),
            **binding_options,
        )
        if updated_row_count != 1:
            existing = producer.read_binding(raw_id, _RAW_REVISION_BINDING_SQL, (raw_id,))
            if existing is None:
                raise ValueError(f"raw revision bind found no raw row: {raw_id}")
            existing_values = tuple(existing)
            if existing_values == _revision_values(revision) or _is_compatible_classification_refinement(
                existing_values, revision
            ):
                return
            field_names = (
                "logical_source_key",
                "revision_kind",
                "source_revision",
                "predecessor_source_revision",
                "predecessor_raw_id",
                "baseline_raw_id",
                "append_start_offset",
                "append_end_offset",
                "acquisition_generation",
                "revision_authority",
            )
            proposed_values = _revision_values(revision)
            differing = ", ".join(
                f"{name}: stored={stored!r} proposed={proposed!r}"
                for name, stored, proposed in zip(field_names, existing_values, proposed_values, strict=True)
                if stored != proposed
            )
            raise ValueError(f"raw revision is already authoritative and differs for {raw_id}: {differing}")


class SourceBlobReferenceProducer(Protocol):
    def blob_ref_is_excised(self, blob_hash: bytes) -> bool: ...
    def blob_ref_literal(self, value: object) -> tuple[str, tuple[object, ...]]: ...
    def blob_ref_write(
        self,
        ref: ArchiveSourceBlobRef,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]: ...
    def consume_reference_receipt(self, ref: ArchiveSourceBlobRef) -> None: ...


def _write_blob_ref(producer: SourceBlobReferenceProducer, ref: ArchiveSourceBlobRef) -> None:
    """Retain the canonical reference and consume its exact publication receipt."""
    ref_type = require_vocabulary(ref.ref_type, _BLOB_REF_TYPES, field="ref_type")
    if ref.raw_id is None or ref.size_bytes is None or ref.acquired_at_ms is None:
        raise ValueError("raw_id, size_bytes, and acquired_at_ms are required for blob refs")
    operands = tuple(
        producer.blob_ref_literal(value)
        for value in (
            ref.blob_hash,
            ref.raw_id,
            ref_type,
            ref.source_path,
            ref.size_bytes,
            ref.acquired_at_ms,
        )
    )
    expressions = ", ".join(expression for expression, _parameters in operands)
    parameters = tuple(parameter for _expression, values in operands for parameter in values)
    # Hook coordinates keep their first observation. Other references retain
    # the existing physical REPLACE ordering used by raw receipt selection.
    sql = (
        "INSERT INTO blob_refs (rowid, blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms) "
        f"VALUES (?, {expressions}) ON CONFLICT DO NOTHING"
        if ref_type == "hook_payload"
        else "INSERT OR REPLACE INTO blob_refs (rowid, blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms) "
        f"VALUES (?, {expressions})"
    )
    with producer.blob_ref_write(ref, sql, (None, *parameters)):
        pass
    producer.consume_reference_receipt(ref)


class SourceArtifactProducer(Protocol):
    """The existing artifact producer's exact coordinate and observation inputs."""

    def artifact_transaction(self, manage_transaction: bool) -> AbstractContextManager[object]: ...
    def artifact_literal(self, value: object) -> tuple[str, tuple[object, ...]]: ...
    def artifact_coordinate_rows(
        self,
        raw_id: str,
        artifact: ArchiveSourceArtifact,
    ) -> AbstractContextManager[sqlite3.Cursor]: ...
    def artifact_observation_rows(
        self,
        raw_id: str,
        *,
        receipt: bool,
    ) -> AbstractContextManager[sqlite3.Cursor]: ...
    def artifact_validation_failed(self, raw_id: str) -> bool: ...
    def artifact_write(
        self,
        sql: str,
        parameters: tuple[object, ...],
        artifact_id: str,
        *,
        allocation: bool,
    ) -> AbstractContextManager[sqlite3.Cursor]: ...


_ARTIFACT_VALIDATION_STATUS_SQL = "SELECT validation_status FROM raw_sessions WHERE raw_id=?"


def _artifact_coordinate_query(
    raw_id: str,
    artifact: ArchiveSourceArtifact,
    *,
    columns: str,
) -> tuple[str, tuple[object, ...]]:
    # Both hosts select the identical complete matching coordinate family.
    if columns not in {"a.rowid", "a.artifact_id, a.raw_id"}:
        raise ValueError("artifact coordinate read requires its declared columns")
    origin = require_vocabulary(artifact.origin, Origin, field="artifact.origin")
    if _is_raw_failure_artifact_kind(artifact.artifact_kind):
        predicate = "a.raw_id = ? AND a.origin = ? AND a.source_path = ? AND a.source_index = ?"
        parameters: tuple[object, ...] = (raw_id, origin, artifact.source_path, artifact.source_index)
    else:
        predicate = "a.origin = ? AND a.source_path = ? AND a.source_index = ? AND a.artifact_kind NOT IN ("
        predicate += ", ".join("?" for _ in RAW_FAILURE_EVIDENCE_KINDS) + ")"
        parameters = (origin, artifact.source_path, artifact.source_index, *sorted(RAW_FAILURE_EVIDENCE_KINDS))
    return f"SELECT {columns} FROM raw_artifacts AS a WHERE {predicate}", parameters


def _artifact_observation_query(raw_id: str, *, receipt: bool) -> tuple[str, tuple[object, ...]]:
    return (
        "SELECT acquired_at_ms, rowid FROM blob_refs "
        "WHERE ref_id = ? AND ref_type = 'raw_payload' ORDER BY rowid DESC LIMIT 1"
        if receipt
        else "SELECT acquired_at_ms, rowid FROM raw_sessions WHERE raw_id = ?",
        (raw_id,),
    )


def _write_artifact(producer: SourceArtifactProducer, raw_id: str, artifact: ArchiveSourceArtifact) -> None:
    values = (
        artifact.artifact_id,
        raw_id,
        require_vocabulary(artifact.origin, Origin, field="artifact.origin"),
        artifact.source_path,
        artifact.source_index,
        artifact.artifact_kind,
        require_vocabulary(artifact.support_status, ArtifactSupportStatus, field="artifact.support_status"),
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
    )
    literals = tuple(producer.artifact_literal(value) for value in values)
    expressions = ", ".join(expression for expression, _parameters in literals)
    parameters = (None, *(parameter for _expression, operands in literals for parameter in operands))
    with producer.artifact_write(
        f"""
        INSERT INTO raw_artifacts (
            rowid, artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
            support_status, classification_reason, parse_as_session, schema_eligible,
            malformed_jsonl_lines, decode_error, cohort_id, link_group_key, sidecar_agent_type,
            first_observed_at_ms, last_observed_at_ms
        ) VALUES (?, {expressions})
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
        # A terminal failure carrier is the durable statement that these
        # bytes will never become a session, and the raw-frontier gate reads
        # it to settle the path. Re-observing the coordinate re-derives an
        # ordinary path classification, which must not take the row back.
        + _TERMINAL_CARRIER_GUARD_SQL,
        parameters,
        artifact.artifact_id,
        allocation=True,
    ):
        pass


def _upsert_raw_artifact(
    producer: SourceArtifactProducer,
    raw_id: str,
    artifact: ArchiveSourceArtifact,
    *,
    manage_transaction: bool = True,
) -> None:
    """Use the canonical coordinate winner on ordinary or selected prepared state."""
    with producer.artifact_transaction(manage_transaction):
        with producer.artifact_coordinate_rows(raw_id, artifact) as artifact_rows:
            existing = artifact_rows.fetchone()
        if existing is not None:
            # Original receipt insertion order remains the carrier authority.
            if str(existing[1]) != raw_id:
                with producer.artifact_observation_rows(raw_id, receipt=True) as rows:
                    incoming_receipt = rows.fetchone()
                with producer.artifact_observation_rows(str(existing[1]), receipt=True) as rows:
                    existing_receipt = rows.fetchone()
                if (incoming_receipt is None) != (existing_receipt is None):
                    raise RuntimeError(
                        "cannot compare artifact observation order across incompatible raw-payload receipt coverage"
                    )
                if incoming_receipt is None:
                    with producer.artifact_observation_rows(raw_id, receipt=False) as rows:
                        incoming_observation = rows.fetchone()
                    with producer.artifact_observation_rows(str(existing[1]), receipt=False) as rows:
                        existing_observation = rows.fetchone()
                else:
                    incoming_observation = incoming_receipt
                    existing_observation = existing_receipt
                if incoming_observation is None:
                    raise KeyError(raw_id)
                if existing_observation is None:
                    raise KeyError(str(existing[1]))
                if int(existing_observation[1]) >= int(incoming_observation[1]):
                    timestamp_expression, timestamp_parameters = producer.artifact_literal(
                        artifact.first_observed_at_ms
                    )
                    id_expression, id_parameters = producer.artifact_literal(str(existing[0]))
                    with producer.artifact_write(
                        "UPDATE raw_artifacts SET first_observed_at_ms = "
                        f"MIN(first_observed_at_ms, {timestamp_expression}) WHERE artifact_id = {id_expression}",
                        (*timestamp_parameters, *id_parameters),
                        str(existing[0]),
                        allocation=False,
                    ):
                        pass
                    return
            artifact = replace(artifact, artifact_id=str(existing[0]))
        _write_artifact(producer, raw_id, artifact)


if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

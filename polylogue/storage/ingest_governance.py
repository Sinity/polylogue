"""Prepared membership-governance compute and publication seam.

The daemon owns admission, scheduling, cancellation, and terminal source
markers.  This module deliberately owns neither side of that boundary.  It
only turns an already-retained raw census or one logical membership cohort
into immutable, writer-revalidated work.  Compute can therefore run on the
shared parse worker while the single writer only publishes evidence that still
describes its current source and index state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.revision_authority import HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL, RawRevisionKind
from polylogue.archive.session_revision_membership import (
    MembershipClassification,
    MembershipRevision,
    classify_membership_revisions,
)
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.pipeline.ids import SessionRevisionProjection, session_revision_projection
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.blob_store import BlobStore, PreparedBlob
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    raw_revision_descriptor,
)
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef
from polylogue.storage.sqlite.archive_tiers.write import PreparedRows, prepare_session_rows

ParseRetainedRaw = Callable[[Any, str], Sequence[ParsedSession] | None]


class RawCensusStatus(StrEnum):
    """The complete parser result carried to source-census publication."""

    COMPLETE = "complete"
    NON_SESSION = "non_session"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RawCensusBinding:
    """The durable census row observed while preparing one raw."""

    parser_fingerprint: str | None
    status: str | None
    member_count: int | None
    censused_at_ms: int | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class RawDescriptorBinding:
    """All source-row facts that make prepared parser work applicable."""

    raw_id: str
    provider: Provider
    blob_hash: str
    source_path: str
    revision_kind: RawRevisionKind
    blob_size: int
    logical_source_key: str | None
    source_revision: str | None
    predecessor_raw_id: str | None
    baseline_raw_id: str | None
    predecessor_source_revision: str | None
    append_start_offset: int | None
    append_end_offset: int | None
    revision_authority: str
    acquisition_generation: int | None
    census: RawCensusBinding
    membership: tuple[object, ...] | None


@dataclass(frozen=True, slots=True)
class AcceptedHeadBinding:
    """The head and the indexed session frontier it claims."""

    accepted_raw_id: str | None
    accepted_source_revision: str | None
    accepted_content_hash: bytes | None
    accepted_frontier_kind: str | None
    accepted_frontier: int | None
    acquisition_generation: int | None
    append_end_offset: int | None
    session_id: str | None
    persisted_raw_id: str | None
    persisted_content_hash: bytes | None
    persisted_message_count: int | None
    persisted_updated_at_ms: int | None


@dataclass(frozen=True, slots=True)
class AppendFrontierBinding:
    """A possibly-quarantined append can still prevent a head replacement."""

    raw_id: str
    predecessor_source_revision: str | None
    source_revision: str | None
    append_start_offset: int | None
    append_end_offset: int | None
    revision_authority: str


@dataclass(frozen=True, slots=True)
class PreparedAttachmentBlob:
    """Attachment bytes staged by compute but not published until writer admission."""

    position: int
    prepared_blob: PreparedBlob | None
    precomputed_blob: tuple[str, int] | None


@dataclass(frozen=True, slots=True)
class PreparedRawCensus:
    """One raw's complete normalized parser census, without terminal state."""

    raw_id: str
    parser_fingerprint: str
    descriptor: RawDescriptorBinding
    sessions: tuple[ParsedSession, ...] | None
    projections: tuple[SessionRevisionProjection, ...]
    logical_keys: tuple[str, ...]
    status: RawCensusStatus
    censused_at_ms: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PreparedIngestCohort:
    """A precise cohort plus every fact the writer must revalidate."""

    logical_source_key: str
    blob_root: str
    request_owned_complete_raw_ids: tuple[str, ...]
    selector_raw_ids: tuple[str, ...]
    member_bindings: tuple[RawDescriptorBinding, ...]
    existing_head: AcceptedHeadBinding
    append_frontier: tuple[AppendFrontierBinding, ...]
    convertible_full_raw_ids: tuple[str, ...]
    retirement_censuses: tuple[PreparedRawCensus, ...]
    parsed_by_raw_id: dict[str, ParsedSession]
    projections_by_raw_id: dict[str, SessionRevisionProjection]
    classification: MembershipClassification | None
    prepared_rows_by_raw_id: dict[str, PreparedRows]
    prepared_attachment_blobs: tuple[PreparedAttachmentBlob, ...]
    affected_session_ids: tuple[str, ...]
    acquired_at_ms: int


@dataclass(frozen=True, slots=True)
class CensusPublication:
    """Source-census publication outcome; it never means terminal parsing."""

    published: bool
    reprepare_required: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CohortPublication:
    """Membership publication outcome, including a typed reprepare signal."""

    published: bool
    reprepare_required: bool
    session_id: str | None = None
    retired_raw_ids: tuple[str, ...] = ()
    reprepare_logical_source_keys: tuple[str, ...] = ()
    reason: str | None = None


def _logical_key(session: ParsedSession) -> str:
    return f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"


def _normalized_sessions(archive: Any, raw_id: str, sessions: Sequence[ParsedSession]) -> tuple[ParsedSession, ...]:
    fallback_timestamp = archive.raw_revision_file_mtime(raw_id)
    return tuple(normalize_session_timestamps(session, fallback_timestamp=fallback_timestamp) for session in sessions)


def _census_binding(archive: Any, raw_id: str) -> RawCensusBinding:
    row = (
        archive._ensure_source_conn()
        .execute(
            """
        SELECT parser_fingerprint, status, member_count, censused_at_ms, detail
        FROM raw_membership_census WHERE raw_id = ?
        """,
            (raw_id,),
        )
        .fetchone()
    )
    if row is None:
        return RawCensusBinding(None, None, None, None, None)
    return RawCensusBinding(str(row[0]), str(row[1]), int(row[2]), int(row[3]), None if row[4] is None else str(row[4]))


def _raw_binding(archive: Any, raw_id: str, *, logical_source_key: str | None = None) -> RawDescriptorBinding:
    provider, blob_hash, source_path, revision_kind, blob_size = raw_revision_descriptor(archive, raw_id)
    conn = archive._ensure_source_conn()
    row = conn.execute(
        """
        SELECT logical_source_key, source_revision, predecessor_raw_id, baseline_raw_id,
               predecessor_source_revision, append_start_offset, append_end_offset,
               revision_authority, acquisition_generation
        FROM raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    if row is None:
        raise KeyError(raw_id)
    membership: tuple[object, ...] | None = None
    if logical_source_key is not None:
        membership_row = conn.execute(
            """
            SELECT provider_session_id, source_revision, normalized_content_hash, message_count,
                   decision, revision_authority, acquisition_generation
            FROM raw_session_memberships
            WHERE raw_id = ? AND logical_source_key = ?
            """,
            (raw_id, logical_source_key),
        ).fetchone()
        membership = None if membership_row is None else tuple(membership_row)
    return RawDescriptorBinding(
        raw_id=raw_id,
        provider=provider,
        blob_hash=blob_hash,
        source_path=source_path,
        revision_kind=revision_kind,
        blob_size=blob_size,
        logical_source_key=None if row[0] is None else str(row[0]),
        source_revision=None if row[1] is None else str(row[1]),
        predecessor_raw_id=None if row[2] is None else str(row[2]),
        baseline_raw_id=None if row[3] is None else str(row[3]),
        predecessor_source_revision=None if row[4] is None else str(row[4]),
        append_start_offset=None if row[5] is None else int(row[5]),
        append_end_offset=None if row[6] is None else int(row[6]),
        revision_authority=str(row[7]),
        acquisition_generation=None if row[8] is None else int(row[8]),
        census=_census_binding(archive, raw_id),
        membership=membership,
    )


def _head_binding(archive: Any, logical_source_key: str) -> AcceptedHeadBinding:
    head = archive._conn.execute(
        """
        SELECT accepted_raw_id, accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier, acquisition_generation,
               append_end_offset, session_id
        FROM raw_revision_heads WHERE logical_source_key = ?
        """,
        (logical_source_key,),
    ).fetchone()
    if head is None:
        return AcceptedHeadBinding(None, None, None, None, None, None, None, None, None, None, None, None)
    session_id = str(head[7])
    persisted = archive._conn.execute(
        """
        SELECT raw_id, content_hash, message_count, updated_at_ms
        FROM sessions WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return AcceptedHeadBinding(
        accepted_raw_id=str(head[0]),
        accepted_source_revision=None if head[1] is None else str(head[1]),
        accepted_content_hash=None if head[2] is None else bytes(head[2]),
        accepted_frontier_kind=None if head[3] is None else str(head[3]),
        accepted_frontier=None if head[4] is None else int(head[4]),
        acquisition_generation=None if head[5] is None else int(head[5]),
        append_end_offset=None if head[6] is None else int(head[6]),
        session_id=session_id,
        persisted_raw_id=None if persisted is None or persisted[0] is None else str(persisted[0]),
        persisted_content_hash=None if persisted is None or persisted[1] is None else bytes(persisted[1]),
        persisted_message_count=None if persisted is None or persisted[2] is None else int(persisted[2]),
        persisted_updated_at_ms=None if persisted is None or persisted[3] is None else int(persisted[3]),
    )


def _append_frontier(archive: Any, logical_source_key: str) -> tuple[AppendFrontierBinding, ...]:
    rows = (
        archive._ensure_source_conn()
        .execute(
            """
        SELECT raw_id, predecessor_source_revision, source_revision,
               append_start_offset, append_end_offset, revision_authority
        FROM raw_sessions
        WHERE logical_source_key = ?
          AND revision_kind = 'append'
          AND predecessor_source_revision IS NOT NULL
        ORDER BY raw_id
        """,
            (logical_source_key,),
        )
        .fetchall()
    )
    return tuple(
        AppendFrontierBinding(
            raw_id=str(row[0]),
            predecessor_source_revision=None if row[1] is None else str(row[1]),
            source_revision=None if row[2] is None else str(row[2]),
            append_start_offset=None if row[3] is None else int(row[3]),
            append_end_offset=None if row[4] is None else int(row[4]),
            revision_authority=str(row[5]),
        )
        for row in rows
    )


def prepare_raw_census(
    reader_archive: Any,
    raw_id: str,
    *,
    parser_fingerprint: str,
    parse_retained_raw: ParseRetainedRaw,
    censused_at_ms: int,
    detail: str = "",
) -> PreparedRawCensus:
    """Parse one retained raw away from the writer and bind its exact input."""
    descriptor = _raw_binding(reader_archive, raw_id)
    parsed = parse_retained_raw(reader_archive, raw_id)
    if parsed is None:
        return PreparedRawCensus(
            raw_id, parser_fingerprint, descriptor, None, (), (), RawCensusStatus.FAILED, censused_at_ms, detail
        )
    sessions = _normalized_sessions(reader_archive, raw_id, parsed)
    projections = tuple(session_revision_projection(session) for session in sessions)
    return PreparedRawCensus(
        raw_id=raw_id,
        parser_fingerprint=parser_fingerprint,
        descriptor=descriptor,
        sessions=sessions,
        projections=projections,
        logical_keys=tuple(sorted({_logical_key(session) for session in sessions})),
        status=RawCensusStatus.COMPLETE if sessions else RawCensusStatus.NON_SESSION,
        censused_at_ms=censused_at_ms,
        detail=detail,
    )


def publish_raw_census(writer_archive: Any, prepared: PreparedRawCensus) -> CensusPublication:
    """Publish a prepared source census only if its retained input is unchanged."""
    if _raw_binding(writer_archive, prepared.raw_id) != prepared.descriptor:
        return CensusPublication(False, True, "raw descriptor or prior census changed")
    writer_archive.replace_raw_membership_census(
        prepared.raw_id,
        None if prepared.sessions is None else list(prepared.sessions),
        parser_fingerprint=prepared.parser_fingerprint,
        censused_at_ms=prepared.censused_at_ms,
        detail=prepared.detail,
        projections=prepared.projections,
    )
    return CensusPublication(True, False)


def _selector_raw_ids(
    archive: Any,
    logical_source_key: str,
    request_owned_complete_raw_ids: Sequence[str],
) -> tuple[str, ...]:
    selected: set[str] = set()
    for raw_id in request_owned_complete_raw_ids:
        selected.update(archive.raw_membership_raw_ids(logical_source_key, include_complete_raw_id=raw_id))
    selected.update(archive.raw_membership_retired_full_revision_siblings(logical_source_key))
    head_raw_id = archive.raw_revision_head_raw_id(logical_source_key)
    if head_raw_id is not None:
        selected.add(head_raw_id)
    return tuple(sorted(selected))


def _membership_revision(
    archive: Any, raw_id: str, session: ParsedSession, projection: SessionRevisionProjection
) -> MembershipRevision:
    browser_snapshot_fidelity: Literal["dom", "native"] | None = None
    if (
        NATIVE_BROWSER_CAPTURE_INGEST_FLAG in session.ingest_flags
        or COMPACT_BROWSER_CAPTURE_INGEST_FLAG in session.ingest_flags
    ):
        browser_snapshot_fidelity = "native"
    elif DOM_FALLBACK_INGEST_FLAG in session.ingest_flags:
        browser_snapshot_fidelity = "dom"
    observed_at_ms, _receipt_order = archive.raw_revision_observation_order(raw_id)
    return MembershipRevision(
        raw_id=raw_id,
        projection=projection,
        provider_updated_at=session.updated_at,
        observed_at_ms=observed_at_ms,
        browser_snapshot_fidelity=browser_snapshot_fidelity,
        provider_message_ids=frozenset(
            message.provider_message_id for message in session.messages if message.provider_message_id is not None
        ),
        provider_attachment_ids=frozenset(attachment.provider_attachment_id for attachment in session.attachments),
    )


def _session_for_key(
    archive: Any,
    raw_id: str,
    logical_source_key: str,
    parse_retained_raw: ParseRetainedRaw,
) -> ParsedSession:
    sessions = _normalized_sessions(archive, raw_id, parse_retained_raw(archive, raw_id) or ())
    matches = [session for session in sessions if _logical_key(session) == logical_source_key]
    if len(matches) != 1:
        raise RuntimeError(f"membership {raw_id}:{logical_source_key} no longer parses uniquely")
    return matches[0]


def _prepare_attachment_blobs(reader_archive: Any, session: ParsedSession) -> tuple[PreparedAttachmentBlob, ...]:
    """Stage inline attachment bytes without publishing or acquiring a write lease."""
    blob_store = BlobStore(reader_archive.archive_root / "blob")
    return tuple(
        PreparedAttachmentBlob(
            position=position,
            prepared_blob=(
                None if attachment.inline_bytes is None else blob_store.prepare_from_bytes(attachment.inline_bytes)
            ),
            precomputed_blob=attachment.precomputed_blob,
        )
        for position, attachment in enumerate(session.attachments)
        if attachment.inline_bytes is not None or attachment.precomputed_blob is not None
    )


def prepare_ingest_cohort(
    reader_archive: Any,
    *,
    logical_source_key: str,
    accepted_raw_ids: Sequence[str],
    parser_fingerprint: str,
    parse_retained_raw: ParseRetainedRaw,
    acquired_at_ms: int,
) -> PreparedIngestCohort:
    """Prepare one exact live membership cohort without taking the writer."""
    request_owned = tuple(dict.fromkeys(accepted_raw_ids))
    convertible = tuple(reader_archive.convertible_full_revision_raw_ids(logical_source_key))
    selector = _selector_raw_ids(reader_archive, logical_source_key, request_owned)
    head = _head_binding(reader_archive, logical_source_key)
    append_frontier = _append_frontier(reader_archive, logical_source_key)

    if convertible:
        retirements: list[PreparedRawCensus] = []
        for raw_id in convertible:
            census = prepare_raw_census(
                reader_archive,
                raw_id,
                parser_fingerprint=parser_fingerprint,
                parse_retained_raw=parse_retained_raw,
                censused_at_ms=acquired_at_ms,
                detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
            )
            if census.status is not RawCensusStatus.COMPLETE or census.sessions is None:
                raise RuntimeError(f"convertible full revision {raw_id} does not have a complete parser census")
            if sum(1 for session in census.sessions if _logical_key(session) == logical_source_key) != 1:
                raise RuntimeError(f"membership {raw_id}:{logical_source_key} no longer parses uniquely")
            retirements.append(census)
        return PreparedIngestCohort(
            logical_source_key=logical_source_key,
            blob_root=str(reader_archive.archive_root / "blob"),
            request_owned_complete_raw_ids=request_owned,
            selector_raw_ids=selector,
            member_bindings=tuple(
                _raw_binding(reader_archive, raw_id, logical_source_key=logical_source_key) for raw_id in selector
            ),
            existing_head=head,
            append_frontier=append_frontier,
            convertible_full_raw_ids=convertible,
            retirement_censuses=tuple(retirements),
            parsed_by_raw_id={},
            projections_by_raw_id={},
            classification=None,
            prepared_rows_by_raw_id={},
            prepared_attachment_blobs=(),
            affected_session_ids=tuple(
                sorted(
                    {
                        str(make_session_id(session.source_name, session.provider_session_id))
                        for item in retirements
                        for session in item.sessions or ()
                    }
                )
            ),
            acquired_at_ms=acquired_at_ms,
        )

    parsed_by_raw_id: dict[str, ParsedSession] = {}
    projections_by_raw_id: dict[str, SessionRevisionProjection] = {}
    revisions: list[MembershipRevision] = []
    bindings: list[RawDescriptorBinding] = []
    for raw_id in selector:
        session = _session_for_key(reader_archive, raw_id, logical_source_key, parse_retained_raw)
        projection = session_revision_projection(session)
        parsed_by_raw_id[raw_id] = session
        projections_by_raw_id[raw_id] = projection
        revisions.append(_membership_revision(reader_archive, raw_id, session, projection))
        bindings.append(_raw_binding(reader_archive, raw_id, logical_source_key=logical_source_key))
    classification = (
        classify_membership_revisions(revisions, existing_accepted_raw_id=head.accepted_raw_id) if revisions else None
    )
    prepared_rows: dict[str, PreparedRows] = {}
    attachment_blobs: tuple[PreparedAttachmentBlob, ...] = ()
    if classification is not None and classification.accepted_raw_ids:
        accepted_raw_id = classification.accepted_raw_ids[-1]
        accepted_session = parsed_by_raw_id[accepted_raw_id]
        prepared_rows[accepted_raw_id] = prepare_session_rows(accepted_session)
        attachment_blobs = _prepare_attachment_blobs(reader_archive, accepted_session)
    return PreparedIngestCohort(
        logical_source_key=logical_source_key,
        blob_root=str(reader_archive.archive_root / "blob"),
        request_owned_complete_raw_ids=request_owned,
        selector_raw_ids=selector,
        member_bindings=tuple(bindings),
        existing_head=head,
        append_frontier=append_frontier,
        convertible_full_raw_ids=(),
        retirement_censuses=(),
        parsed_by_raw_id=parsed_by_raw_id,
        projections_by_raw_id=projections_by_raw_id,
        classification=classification,
        prepared_rows_by_raw_id=prepared_rows,
        prepared_attachment_blobs=attachment_blobs,
        affected_session_ids=tuple(
            sorted(
                {
                    str(make_session_id(session.source_name, session.provider_session_id))
                    for session in parsed_by_raw_id.values()
                }
            )
        ),
        acquired_at_ms=acquired_at_ms,
    )


def _cohort_still_current(writer_archive: Any, prepared: PreparedIngestCohort) -> str | None:
    if str(writer_archive.archive_root / "blob") != prepared.blob_root:
        return "prepared attachment blob root changed"
    if (
        tuple(writer_archive.convertible_full_revision_raw_ids(prepared.logical_source_key))
        != prepared.convertible_full_raw_ids
    ):
        return "convertible full-revision route changed"
    if (
        _selector_raw_ids(writer_archive, prepared.logical_source_key, prepared.request_owned_complete_raw_ids)
        != prepared.selector_raw_ids
    ):
        return "eligible membership selector changed"
    if _head_binding(writer_archive, prepared.logical_source_key) != prepared.existing_head:
        return "accepted head or persisted session frontier changed"
    if _append_frontier(writer_archive, prepared.logical_source_key) != prepared.append_frontier:
        return "byte append frontier changed"
    current_bindings = tuple(
        _raw_binding(writer_archive, binding.raw_id, logical_source_key=prepared.logical_source_key)
        for binding in prepared.member_bindings
    )
    if current_bindings != prepared.member_bindings:
        return "raw descriptor, membership, or census changed"
    for census in prepared.retirement_censuses:
        if _raw_binding(writer_archive, census.raw_id) != census.descriptor:
            return "convertible full-revision descriptor or census changed"
    return None


def _retirement_has_active_byte_descendant(writer_archive: Any, raw_ids: Sequence[str]) -> bool:
    """Check every proposed retirement before changing any source census row."""
    if not raw_ids:
        return False
    placeholders = ", ".join("?" for _ in raw_ids)
    row = (
        writer_archive._ensure_source_conn()
        .execute(
            f"""
        SELECT 1
        FROM raw_sessions
        WHERE raw_id NOT IN ({placeholders})
          AND (predecessor_raw_id IN ({placeholders}) OR baseline_raw_id IN ({placeholders}))
        LIMIT 1
        """,
            (*raw_ids, *raw_ids, *raw_ids),
        )
        .fetchone()
    )
    return row is not None


def _retirement_order(writer_archive: Any, raw_ids: Sequence[str]) -> tuple[str, ...]:
    """Retire a contained full-revision chain from descendants to baselines."""
    if not raw_ids:
        return ()
    placeholders = ", ".join("?" for _ in raw_ids)
    rows = (
        writer_archive._ensure_source_conn()
        .execute(
            f"""
        SELECT raw_id, predecessor_raw_id, baseline_raw_id
        FROM raw_sessions WHERE raw_id IN ({placeholders})
        """,
            tuple(raw_ids),
        )
        .fetchall()
    )
    pending = {str(row[0]) for row in rows}
    predecessors = {
        str(row[0]): {str(value) for value in row[1:] if value is not None and str(value) in pending} for row in rows
    }
    ordered: list[str] = []
    while pending:
        # A predecessor relation only constrains a child that remains pending.
        # Once a descendant has been retired, its retained relation must not
        # keep its baseline from becoming the next leaf.
        leaves = sorted(raw_id for raw_id in pending if not any(raw_id in predecessors[child] for child in pending))
        if not leaves:
            raise RuntimeError("convertible full-revision retirement contains a byte-lineage cycle")
        ordered.extend(leaves)
        pending.difference_update(leaves)
        for parents in predecessors.values():
            parents.difference_update(leaves)
    return tuple(ordered)


def discard_prepared_ingest_cohort(prepared: PreparedIngestCohort) -> None:
    """Discard staged attachment bytes after cancellation or a deferred publish."""
    blob_store = BlobStore(Path(prepared.blob_root))
    for attachment in prepared.prepared_attachment_blobs:
        if attachment.prepared_blob is not None:
            blob_store.discard_prepared(attachment.prepared_blob)


def _writer_preacquired_attachments(
    writer_archive: Any,
    prepared: PreparedIngestCohort,
) -> tuple[dict[int, tuple[bytes | None, int, str]], tuple[ArchiveSourceBlobRef, ...]]:
    """Queue compute-staged blobs; only the writer later reserves and publishes them."""
    if prepared.classification is None or not prepared.classification.accepted_raw_ids:
        return {}, ()
    if str(writer_archive.archive_root / "blob") != prepared.blob_root:
        raise RuntimeError("prepared attachment blob root does not match the writer archive")
    publisher = writer_archive._blob_publisher
    if publisher is None:
        raise RuntimeError("membership publication requires a writable blob publisher")
    accepted_raw_id = prepared.classification.accepted_raw_ids[-1]
    accepted_session = prepared.parsed_by_raw_id[accepted_raw_id]
    binding = next(binding for binding in prepared.member_bindings if binding.raw_id == accepted_raw_id)
    attachments: dict[int, tuple[bytes | None, int, str]] = {}
    refs: list[ArchiveSourceBlobRef] = []
    for item in prepared.prepared_attachment_blobs:
        attachment = accepted_session.attachments[item.position]
        if item.prepared_blob is None:
            if item.precomputed_blob is None:
                raise RuntimeError("prepared attachment has neither staged nor precomputed bytes")
            hash_hex, size = item.precomputed_blob
            attachments[id(attachment)] = (bytes.fromhex(hash_hex), size, "acquired")
            continue
        hash_hex, size = publisher.queue_prepared(item.prepared_blob)
        attachments[id(attachment)] = (bytes.fromhex(hash_hex), size, "acquired")
        refs.append(
            ArchiveSourceBlobRef(
                blob_hash=bytes.fromhex(hash_hex),
                ref_type="attachment",
                source_path=binding.source_path,
                size_bytes=size,
                acquired_at_ms=prepared.acquired_at_ms,
                publication_receipt_id=publisher.receipt_id(hash_hex),
            )
        )
    return attachments, tuple(refs)


def publish_ingest_cohort(
    writer_archive: Any,
    prepared: PreparedIngestCohort,
    *,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "membership_replay",
    manage_transaction: bool = True,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    defer_fts: bool = False,
) -> CohortPublication:
    """Publish only a still-current cohort; never reparse or reopen attachments."""
    reason = _cohort_still_current(writer_archive, prepared)
    if reason is not None:
        discard_prepared_ingest_cohort(prepared)
        return CohortPublication(False, True, reason=reason)
    if prepared.retirement_censuses:
        retirement_raw_ids = tuple(census.raw_id for census in prepared.retirement_censuses)
        if _retirement_has_active_byte_descendant(writer_archive, retirement_raw_ids):
            discard_prepared_ingest_cohort(prepared)
            return CohortPublication(False, True, reason="convertible full revision has an active byte descendant")
        # One source transaction makes the all-or-nothing retirement boundary
        # explicit.  It remains a source-only action; the next preparation
        # must still classify and commit its own index/head receipt.
        with writer_archive._ensure_source_conn():
            censuses_by_raw_id = {census.raw_id: census for census in prepared.retirement_censuses}
            for raw_id in _retirement_order(writer_archive, retirement_raw_ids):
                census = censuses_by_raw_id[raw_id]
                writer_archive.replace_raw_membership_census(
                    census.raw_id,
                    list(census.sessions or ()),
                    parser_fingerprint=census.parser_fingerprint,
                    censused_at_ms=census.censused_at_ms,
                    detail=census.detail,
                    retire_full_revision_governance=True,
                    projections=census.projections,
                    manage_transaction=False,
                )
        return CohortPublication(
            False,
            True,
            retired_raw_ids=retirement_raw_ids,
            reprepare_logical_source_keys=tuple(
                sorted({key for census in prepared.retirement_censuses for key in census.logical_keys})
            ),
            reason="full revisions retired; reprepare membership cohort",
        )
    if prepared.classification is None:
        discard_prepared_ingest_cohort(prepared)
        return CohortPublication(False, True, reason="no eligible membership members")
    attachments, attachment_refs = _writer_preacquired_attachments(writer_archive, prepared)
    session_id = writer_archive.apply_raw_membership_classification(
        prepared.logical_source_key,
        prepared.classification,
        prepared.parsed_by_raw_id,
        prepared.projections_by_raw_id,
        acquired_at_ms=prepared.acquired_at_ms,
        stage_timings_s=stage_timings_s,
        stage_timing_prefix=stage_timing_prefix,
        manage_transaction=manage_transaction,
        bulk_fts=bulk_fts,
        bulk_build=bulk_build,
        defer_fts=defer_fts,
        preacquired_attachment_blobs=attachments,
        preacquired_attachment_refs=attachment_refs,
        prepared_by_raw_id=prepared.prepared_rows_by_raw_id,
    )
    return CohortPublication(True, False, session_id=session_id)


__all__ = [
    "AcceptedHeadBinding",
    "AppendFrontierBinding",
    "CensusPublication",
    "CohortPublication",
    "ParseRetainedRaw",
    "PreparedIngestCohort",
    "PreparedAttachmentBlob",
    "PreparedRawCensus",
    "RawCensusBinding",
    "RawCensusStatus",
    "RawDescriptorBinding",
    "discard_prepared_ingest_cohort",
    "prepare_ingest_cohort",
    "prepare_raw_census",
    "publish_ingest_cohort",
    "publish_raw_census",
]

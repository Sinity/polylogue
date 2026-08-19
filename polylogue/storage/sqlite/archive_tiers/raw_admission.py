"""Raw-admission chokepoint: the single typed decision point for ``raw_sessions`` writes.

polylogue-1fijp (aggz Invariant 2): :func:`admit_raw_observation` is designed
to become the sole creator of ``raw_sessions`` rows, mirroring
``write.py``'s ``write_parsed_session_to_archive`` on the index side. Given
freshly (and atomically -- see :mod:`polylogue.sources.atomic_read`) read
bytes, origin evidence, and the prior accepted head for the payload's
logical source key, it resolves exactly one of five typed, exhaustive arms
-- there is no nullable "unknown, figure it out later" limbo:

1. ``SKIP_DUPLICATE`` -- the bytes are byte-equal to the accepted head.
   Nothing new to record; the caller gets the existing raw_id back.
2. ``APPEND`` -- the bytes extend the accepted head as a strict byte-prefix
   (the common case for tailed logs). Recorded as ``revision_kind=append``
   with a real ``predecessor_raw_id``.
3. ``SUPERSEDE`` -- the incoming bytes are themselves a strict byte-prefix
   of the accepted head. This is the reverse of (2): the newly acquired
   read is a proven-dominated, strictly older/shorter state than what is
   already on record (e.g. a source file briefly reverted, or this read
   raced a rewrite and caught an earlier generation than a sibling read
   already captured). It is recorded as a ``FULL`` revision with
   ``authority=BYTE_PROVEN`` and ``predecessor_raw_id`` pointing at the
   fuller head it is provably subsumed by -- a *proven* relationship, not
   a guess, so it does not go through arm 5's ambiguity handling.

   NOTE ON INTERPRETATION: the bead text describing this arm ("head is
   byte-prefix of prior full -> supersede") admits more than one reading.
   This module's interpretation -- new-shorter-bytes proven subsumed by an
   already-accepted fuller head -- is the one implemented and tested here.
   It has not been independently confirmed against operator intent beyond
   the bead text itself; flag for confirmation before treating this arm's
   exact semantics as load-bearing for anything beyond "don't silently
   regress the head".
4. ``ARTIFACT`` -- the payload is not conversational content (the omsw
   artifact taxonomy's ``ArtifactClassification.parse_as_session is
   False``: tool-results/*.json, subagents/workflows/*/journal.jsonl,
   file-history-snapshot, etc). A ``raw_sessions`` row is still written
   (it remains the acquisition-evidence ledger for every observed byte,
   session or not -- ``raw_artifacts.raw_id`` has a ``NOT NULL REFERENCES
   raw_sessions`` foreign key, so an artifact row cannot exist without
   one), but its classification is attached immediately as a
   ``raw_artifacts`` row using the SAME deterministic
   ``artifact_observation_id`` scheme the offline
   ``materialize_artifact_observations`` sweep uses, so a later sweep
   upserts onto the same row rather than colliding. The one guarantee
   this arm gives is the one arm 4 names: this bytes-classified-as-
   non-conversational content structurally never reaches any code path
   that could create an ``index.db`` ``sessions`` row -- this module has
   no import of any index-tier writer.
5. ``REFUSED_AMBIGUOUS`` -- neither equal, a forward extension, nor a
   backward prefix. Per the operator's 2026-08-03 correction on the bead,
   re-acquisition here is OPPORTUNISTIC, never doctrinal: if a caller
   supplies a ``reacquire`` callback, it is invoked at most once and, if it
   returns bytes that resolve one of arms 1-3, that outcome wins. If the
   source is absent/unreadable (callback returns ``None``) or the
   re-attempt is itself still ambiguous, this falls through to a typed
   refusal row (``revision_kind=unknown``, ``revision_authority=
   quarantined``) carrying a machine-readable reason -- never a silent
   drop, and never a decision that depends on the source file continuing
   to exist.

The bootstrap case -- no prior head at all for this logical source key --
is not one of the five named arms (all five presuppose a prior head to
compare against); it is handled as a distinct ``BASELINE`` outcome: the
first-ever observation is recorded as a ``FULL``/``ASSERTED`` revision.

The ``SHARED_GROUPED`` outcome is likewise outside the five arms. One
grouped payload (a resume-fork chain inside a single Claude Code/Codex
JSONL, a Gemini/Drive bundle) parses into many sessions that all share the
identical captured bytes, so its raw row is keyed by the physical
acquisition coordinate alone and carries no ``native_id`` and no revision
envelope -- which sessions it speaks for lives in
``raw_session_memberships``. The arm exists so that absence is a decision
this module makes and names, rather than a caller-side bypass.

The ``POST_PARSE_PENDING`` outcome is a separate acquisition-preservation
contract for live and streaming callers that must record bytes before parsing
can finish. It writes a deterministic raw row with a typed, quarantined
pending key. The later parser bind replaces that envelope with the provider's
accepted revision, while a retry of the same observation is idempotent.

This module composes the existing low-level writers in
``source_write.py`` (``write_source_raw_session``,
``deterministic_blob_hash``) rather than re-implementing raw-row
persistence -- it decides *which* revision envelope to write, not how to
write it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from polylogue.archive.artifact_taxonomy import ArtifactClassification
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    ArchiveSourceBlobRef,
    deterministic_blob_hash,
    deterministic_raw_session_id,
    pending_raw_logical_source_key,
    write_source_raw_session,
    write_source_raw_session_blob_ref,
)


class RawAdmissionArm(StrEnum):
    """The exhaustive, typed outcome of one :func:`admit_raw_observation` call."""

    BASELINE = "baseline"
    SKIP_DUPLICATE = "skip_duplicate"
    APPEND = "append"
    SUPERSEDE = "supersede"
    ARTIFACT = "artifact"
    REFUSED_AMBIGUOUS = "refused_ambiguous"
    POST_PARSE_PENDING = "post_parse_pending"
    SHARED_GROUPED = "shared_grouped"


_ByteRelation = Literal["duplicate", "append", "supersede", "ambiguous"]


@dataclass(frozen=True, slots=True)
class PriorRawHead:
    """The currently-accepted head raw for one logical source key.

    Callers resolve this themselves (this module makes no assumption about
    how "head" is determined for a given provider/acquisition mode) and
    pass its bytes in so the byte-relation comparison in arms 1-3/5 is a
    pure, synchronous, in-memory decision -- no additional DB or blob-store
    read happens inside :func:`admit_raw_observation` itself.
    """

    raw_id: str
    source_revision: str
    payload: bytes
    baseline_raw_id: str | None = None
    acquisition_generation: int = 0


@dataclass(frozen=True, slots=True)
class RawAdmissionResult:
    """The outcome of one admission decision."""

    arm: RawAdmissionArm
    raw_id: str
    artifact_id: str | None = None
    refusal_reason: str | None = None
    reacquire_attempted: bool = False
    reacquire_changed_outcome: bool = False


def _classify_bytes(payload: bytes, prior_head_payload: bytes) -> _ByteRelation:
    if payload == prior_head_payload:
        return "duplicate"
    if len(payload) > len(prior_head_payload) and payload.startswith(prior_head_payload):
        return "append"
    if len(payload) < len(prior_head_payload) and prior_head_payload.startswith(payload):
        return "supersede"
    return "ambiguous"


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _assert_existing_raw_observation_identity(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    origin: Origin | str,
    native_id: str | None,
    source_path: str,
    source_index: int,
    blob_hash: bytes,
    blob_size: int,
) -> bool:
    """Return whether an explicit raw id already names the same observation.

    Acquisition evidence is what was observed and where: the bytes
    (``blob_hash``/``blob_size``) and the provenance coordinates
    (``native_id``/``source_path``/``source_index``). Those must match exactly --
    a raw id bound to different bytes or a different location is a substitution
    hazard and stays fatal.

    ``origin`` is not acquisition evidence. It is a derived classification, and
    routes legitimately derive it with different amounts of information: a
    decoded ZIP ingest can sniff the archive as a whole and stamp every member
    ``chatgpt-export``, while a source-only replay of the same bytes sees only an
    opaque member and can honestly say no more than ``unknown-export``. Treating
    that as an evidence conflict made the two routes mutually exclusive over
    identical bytes. A refinement away from ``unknown-export`` is therefore
    admitted and upgrades the stored row; only a contradiction between two
    confident origins is fatal.
    """
    row = conn.execute(
        "SELECT origin, native_id, source_path, source_index, blob_hash, blob_size FROM raw_sessions WHERE raw_id = ?",
        (raw_id,),
    ).fetchone()
    if row is None:
        return False
    stored_origin = row[0]
    if tuple(row[1:]) != (native_id, source_path, source_index, blob_hash, blob_size):
        raise ValueError(f"raw id is already bound to different acquisition evidence: {raw_id}")

    incoming_origin = _enum_value(origin)
    if stored_origin == incoming_origin:
        return True

    unknown = Origin.UNKNOWN_EXPORT.value
    if incoming_origin == unknown:
        # The less-informed route re-observing the same bytes. Keep the sharper
        # classification already on the row.
        return True
    if stored_origin == unknown:
        # A better-informed re-observation of bytes admitted under the placeholder.
        # The mutation goes through the declared source-tier writer rather than
        # being issued here.
        from polylogue.storage.sqlite.archive_tiers.source_write import refine_raw_origin

        refine_raw_origin(conn, raw_id=raw_id, origin=origin)
        return True
    raise ValueError(
        f"raw id is already bound to a conflicting origin: {raw_id} "
        f"(stored={stored_origin!r}, incoming={incoming_origin!r})"
    )


def admit_raw_observation(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    capture_mode: Provider | str | None = None,
    source_path: str,
    source_index: int = 0,
    payload: bytes,
    acquired_at_ms: int,
    native_id: str | None = None,
    logical_source_key: str | None = None,
    prior_head: PriorRawHead | None = None,
    raw_id: str | None = None,
    post_parse: bool = False,
    artifact: ArtifactClassification | None = None,
    grouped: bool = False,
    blob_publication_receipt_id: str | None = None,
    additional_blob_refs: tuple[ArchiveSourceBlobRef, ...] = (),
    reacquire: Callable[[], bytes | None] | None = None,
    manage_transaction: bool = True,
) -> RawAdmissionResult:
    """Decide and apply exactly one typed resolution arm for one raw observation.

    ``logical_source_key`` is required for the ``BASELINE``/``ARTIFACT``
    outcomes that do not compare against a prior head. ``post_parse=True`` is
    the separate pre-parse preservation contract and derives a typed pending
    key instead of accepting a nullable revision.

    ``additional_blob_refs`` forwards through to the underlying
    ``write_source_raw_session`` call for the session-content arms (BASELINE/
    APPEND/SUPERSEDE/REFUSED_AMBIGUOUS) -- e.g. attachment blobs preacquired
    by the caller alongside the primary payload. It is not accepted by the
    ``ARTIFACT`` arm: a non-conversational artifact payload has no parsed
    attachments of its own.

    ``grouped=True`` selects the ``SHARED_GROUPED`` arm -- one raw row shared
    by every session parsed out of one grouped payload (a resume-fork chain in
    a single Claude Code/Codex JSONL, a Gemini/Drive bundle). Its identity is
    the physical acquisition coordinate alone, so ``native_id`` is forced NULL
    and no per-session revision envelope is attached: which sessions this raw
    speaks for is recorded in ``raw_session_memberships`` by the caller, and a
    revision chain over a payload that N sessions share has no single subject
    to be about. That absence is now a named arm rather than, as before, a
    bare ``write_source_raw_session`` call that bypassed this module.
    """
    if post_parse and (logical_source_key is not None or prior_head is not None or artifact is not None):
        raise ValueError("post-parse admission cannot combine a logical key, prior head, or artifact")
    if not post_parse and not logical_source_key:
        raise ValueError("logical_source_key is required for raw admission")
    if grouped:
        if post_parse or artifact is not None:
            raise ValueError("grouped admission cannot combine post-parse or artifact resolution")
        if prior_head is not None:
            raise ValueError(
                "grouped admission has no revision chain to compare against: a shared raw's "
                "identity is its acquisition coordinate, so no prior head applies"
            )

    if post_parse:
        blob_hash = deterministic_blob_hash(payload)
        resolved_raw_id = raw_id or deterministic_raw_session_id(
            origin,
            source_path,
            source_index,
            blob_hash,
            native_id,
        )
        pending_key = pending_raw_logical_source_key(
            origin=origin,
            source_path=source_path,
            source_index=source_index,
            raw_id=resolved_raw_id,
        )
        if _assert_existing_raw_observation_identity(
            conn,
            raw_id=resolved_raw_id,
            origin=origin,
            native_id=native_id,
            source_path=source_path,
            source_index=source_index,
            blob_hash=blob_hash,
            blob_size=len(payload),
        ):
            return RawAdmissionResult(arm=RawAdmissionArm.POST_PARSE_PENDING, raw_id=resolved_raw_id)
        admitted_raw_id = write_source_raw_session(
            conn,
            origin=origin,
            capture_mode=capture_mode,
            source_path=source_path,
            source_index=source_index,
            payload=payload,
            acquired_at_ms=acquired_at_ms,
            native_id=native_id,
            raw_id=resolved_raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
            additional_blob_refs=additional_blob_refs,
            revision=RawRevisionEnvelope(
                logical_source_key=pending_key,
                kind=RawRevisionKind.FULL,
                source_revision=blob_hash.hex(),
                acquisition_generation=0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
            manage_transaction=manage_transaction,
        )
        return RawAdmissionResult(arm=RawAdmissionArm.POST_PARSE_PENDING, raw_id=admitted_raw_id)

    assert logical_source_key is not None

    if artifact is not None and not artifact.parse_as_session:
        return _admit_artifact(
            conn,
            origin=origin,
            capture_mode=capture_mode,
            source_path=source_path,
            source_index=source_index,
            payload=payload,
            acquired_at_ms=acquired_at_ms,
            native_id=native_id,
            raw_id=raw_id,
            artifact=artifact,
            blob_publication_receipt_id=blob_publication_receipt_id,
            manage_transaction=manage_transaction,
        )

    if grouped:
        admitted_raw_id = write_source_raw_session(
            conn,
            origin=origin,
            capture_mode=capture_mode,
            source_path=source_path,
            source_index=source_index,
            payload=payload,
            acquired_at_ms=acquired_at_ms,
            native_id=None,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
            additional_blob_refs=additional_blob_refs,
            revision=None,
            manage_transaction=manage_transaction,
        )
        return RawAdmissionResult(arm=RawAdmissionArm.SHARED_GROUPED, raw_id=admitted_raw_id)

    if prior_head is None:
        raw_id = write_source_raw_session(
            conn,
            origin=origin,
            capture_mode=capture_mode,
            source_path=source_path,
            source_index=source_index,
            payload=payload,
            acquired_at_ms=acquired_at_ms,
            native_id=native_id,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
            additional_blob_refs=additional_blob_refs,
            revision=RawRevisionEnvelope(
                logical_source_key=logical_source_key,
                kind=RawRevisionKind.FULL,
                source_revision=deterministic_blob_hash(payload).hex(),
                acquisition_generation=0,
                authority=RawRevisionAuthority.ASSERTED,
            ),
            manage_transaction=manage_transaction,
        )
        return RawAdmissionResult(arm=RawAdmissionArm.BASELINE, raw_id=raw_id)

    relation = _classify_bytes(payload, prior_head.payload)
    reacquire_attempted = False
    reacquire_changed_outcome = False
    resolved_payload = payload

    if relation == "ambiguous" and reacquire is not None:
        reacquire_attempted = True
        candidate = reacquire()
        if candidate is not None:
            candidate_relation = _classify_bytes(candidate, prior_head.payload)
            if candidate_relation != "ambiguous":
                reacquire_changed_outcome = True
                relation = candidate_relation
                resolved_payload = candidate
            else:
                # Still ambiguous, but the refusal row should reflect the
                # freshest bytes actually observed, not the stale first read.
                resolved_payload = candidate

    if relation == "duplicate":
        return RawAdmissionResult(
            arm=RawAdmissionArm.SKIP_DUPLICATE,
            raw_id=prior_head.raw_id,
            reacquire_attempted=reacquire_attempted,
            reacquire_changed_outcome=reacquire_changed_outcome,
        )

    if relation == "append":
        raw_id = write_source_raw_session(
            conn,
            origin=origin,
            capture_mode=capture_mode,
            source_path=source_path,
            source_index=source_index,
            payload=resolved_payload,
            acquired_at_ms=acquired_at_ms,
            native_id=native_id,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
            additional_blob_refs=additional_blob_refs,
            revision=RawRevisionEnvelope(
                logical_source_key=logical_source_key,
                kind=RawRevisionKind.APPEND,
                source_revision=deterministic_blob_hash(resolved_payload).hex(),
                predecessor_source_revision=prior_head.source_revision,
                predecessor_raw_id=prior_head.raw_id,
                baseline_raw_id=prior_head.baseline_raw_id or prior_head.raw_id,
                append_start_offset=len(prior_head.payload),
                append_end_offset=len(resolved_payload),
                acquisition_generation=prior_head.acquisition_generation + 1,
                authority=RawRevisionAuthority.ASSERTED,
            ),
            manage_transaction=manage_transaction,
        )
        return RawAdmissionResult(
            arm=RawAdmissionArm.APPEND,
            raw_id=raw_id,
            reacquire_attempted=reacquire_attempted,
            reacquire_changed_outcome=reacquire_changed_outcome,
        )

    if relation == "supersede":
        raw_id = write_source_raw_session(
            conn,
            origin=origin,
            capture_mode=capture_mode,
            source_path=source_path,
            source_index=source_index,
            payload=resolved_payload,
            acquired_at_ms=acquired_at_ms,
            native_id=native_id,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
            additional_blob_refs=additional_blob_refs,
            revision=RawRevisionEnvelope(
                logical_source_key=logical_source_key,
                kind=RawRevisionKind.FULL,
                source_revision=deterministic_blob_hash(resolved_payload).hex(),
                predecessor_raw_id=prior_head.raw_id,
                baseline_raw_id=prior_head.baseline_raw_id or prior_head.raw_id,
                acquisition_generation=prior_head.acquisition_generation,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
            manage_transaction=manage_transaction,
        )
        return RawAdmissionResult(
            arm=RawAdmissionArm.SUPERSEDE,
            raw_id=raw_id,
            reacquire_attempted=reacquire_attempted,
            reacquire_changed_outcome=reacquire_changed_outcome,
        )

    # Still ambiguous after the (opportunistic, at-most-one) reacquire
    # attempt -- or no reacquire callback was supplied at all. Never persist
    # a nullable "figure it out later" row: this is a typed refusal with a
    # machine-readable reason, not a bare quarantine.
    refusal_reason = (
        "reacquire_still_ambiguous"
        if reacquire_attempted and reacquire_changed_outcome is False and resolved_payload != payload
        else "reacquire_unavailable_or_absent"
        if reacquire_attempted
        else "no_byte_relation_to_prior_head"
    )
    raw_id = write_source_raw_session(
        conn,
        origin=origin,
        capture_mode=capture_mode,
        source_path=source_path,
        source_index=source_index,
        payload=resolved_payload,
        acquired_at_ms=acquired_at_ms,
        native_id=native_id,
        raw_id=raw_id,
        blob_publication_receipt_id=blob_publication_receipt_id,
        additional_blob_refs=additional_blob_refs,
        revision=RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=RawRevisionKind.UNKNOWN,
            source_revision=deterministic_blob_hash(resolved_payload).hex(),
            acquisition_generation=prior_head.acquisition_generation + 1,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
        manage_transaction=manage_transaction,
    )
    return RawAdmissionResult(
        arm=RawAdmissionArm.REFUSED_AMBIGUOUS,
        raw_id=raw_id,
        refusal_reason=refusal_reason,
        reacquire_attempted=reacquire_attempted,
        reacquire_changed_outcome=reacquire_changed_outcome,
    )


def admit_raw_blob_observation(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    capture_mode: Provider | str | None = None,
    source_path: str,
    source_index: int,
    blob_hash: bytes,
    blob_size: int,
    acquired_at_ms: int,
    native_id: str | None = None,
    raw_id: str | None = None,
    blob_publication_receipt_id: str | None = None,
) -> RawAdmissionResult:
    """Admit a prepublished, memory-bounded raw blob pending post-parse identity."""
    if len(blob_hash) != 32:
        raise ValueError("blob_hash must be a 32-byte SHA-256 digest")
    resolved_raw_id = raw_id or deterministic_raw_session_id(
        origin,
        source_path,
        source_index,
        blob_hash,
        native_id,
    )
    if _assert_existing_raw_observation_identity(
        conn,
        raw_id=resolved_raw_id,
        origin=origin,
        native_id=native_id,
        source_path=source_path,
        source_index=source_index,
        blob_hash=blob_hash,
        blob_size=blob_size,
    ):
        return RawAdmissionResult(arm=RawAdmissionArm.POST_PARSE_PENDING, raw_id=resolved_raw_id)
    pending_key = pending_raw_logical_source_key(
        origin=origin,
        source_path=source_path,
        source_index=source_index,
        raw_id=resolved_raw_id,
    )
    admitted_raw_id = write_source_raw_session_blob_ref(
        conn,
        origin=origin,
        capture_mode=capture_mode,
        source_path=source_path,
        source_index=source_index,
        blob_hash=blob_hash,
        blob_size=blob_size,
        acquired_at_ms=acquired_at_ms,
        native_id=native_id,
        raw_id=resolved_raw_id,
        blob_publication_receipt_id=blob_publication_receipt_id,
        revision=RawRevisionEnvelope(
            logical_source_key=pending_key,
            kind=RawRevisionKind.FULL,
            source_revision=blob_hash.hex(),
            acquisition_generation=0,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
        manage_transaction=True,
    )
    return RawAdmissionResult(arm=RawAdmissionArm.POST_PARSE_PENDING, raw_id=admitted_raw_id)


def admit_raw_artifact_blob_observation(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    capture_mode: Provider | str | None = None,
    source_path: str,
    source_index: int,
    blob_hash: bytes,
    blob_size: int,
    acquired_at_ms: int,
    raw_id: str | None = None,
    classification: ArtifactClassification,
    blob_publication_receipt_id: str | None = None,
) -> RawAdmissionResult:
    """Admit a prepublished non-session artifact with typed authority."""
    if classification.parse_as_session:
        raise ValueError("artifact admission requires a non-session classification")
    if len(blob_hash) != 32:
        raise ValueError("blob_hash must be a 32-byte SHA-256 digest")
    artifact_id = artifact_observation_id(
        source_name=_enum_value(origin),
        source_path=source_path,
        source_index=source_index,
    )
    admitted_raw_id = write_source_raw_session_blob_ref(
        conn,
        origin=origin,
        capture_mode=capture_mode,
        source_path=source_path,
        source_index=source_index,
        blob_hash=blob_hash,
        blob_size=blob_size,
        acquired_at_ms=acquired_at_ms,
        raw_id=raw_id,
        blob_publication_receipt_id=blob_publication_receipt_id,
        artifact=ArchiveSourceArtifact(
            artifact_id=artifact_id,
            origin=origin,
            source_path=source_path,
            artifact_kind=classification.cohort,
            classification_reason=classification.reason,
            support_status=ArtifactSupportStatus.UNKNOWN,
            parse_as_session=False,
            schema_eligible=classification.schema_eligible,
            first_observed_at_ms=acquired_at_ms,
            last_observed_at_ms=acquired_at_ms,
            source_index=source_index,
        ),
        revision=None,
        manage_transaction=True,
    )
    return RawAdmissionResult(arm=RawAdmissionArm.ARTIFACT, raw_id=admitted_raw_id, artifact_id=artifact_id)


@dataclass(frozen=True, slots=True)
class ReconstructedRawRow:
    """Every column of a raw row rebuilt from already-proven evidence."""

    raw_id: str
    origin: str
    capture_mode: str | None
    native_id: str | None
    source_path: str
    source_index: int
    blob_hash: bytes
    blob_size: int
    acquired_at_ms: int
    logical_source_key: str
    source_revision: str
    baseline_raw_id: str


def insert_reconstructed_raw_row(
    conn: sqlite3.Connection,
    row: ReconstructedRawRow,
    *,
    schema: str = "main",
) -> None:
    """Insert a raw row RECONSTRUCTED from proven evidence -- not an observation.

    This is the one named exemption from :func:`admit_raw_observation`, and it
    is architectural rather than a not-yet-migrated call site.

    ``admit_raw_observation`` decides *what an observation means*: it takes
    freshly read bytes plus the accepted head and resolves which revision
    envelope those bytes earn. Every input to that decision is missing here.
    A copy-forward repair (``storage/repair.py``'s browser-origin
    reconstruction) is not observing a source at all -- the source it would
    observe is gone. It is rewriting evidence the archive already holds and
    has already adjudicated, under a corrected identity, from a repair plan
    whose contents were proven before this call. Handing those bytes to the
    chokepoint would ask it to re-derive a verdict that is an input here, and
    it would derive a *different* one: with no prior head for the corrected
    logical key it would resolve BASELINE/``ASSERTED``, discarding the
    ``BYTE_PROVEN`` authority the repair plan established.

    So the exemption is not "this write skips the typed arms". It is that the
    arms have already been resolved, durably, elsewhere -- which is why every
    envelope column is a required field of :class:`ReconstructedRawRow`
    instead of a nullable this function could leave unset. The invariant the
    chokepoint exists for (typed resolution, no nullable limbo) holds on this
    path by construction; what does not apply is the resolution *step*.

    Callers must be repair/migration paths writing under an explicit,
    receipt-emitting plan. Acquisition routes must use
    :func:`admit_raw_observation`.
    """
    if schema not in {"main", "source"}:
        raise ValueError(f"unsupported source schema: {schema}")
    if len(row.blob_hash) != 32:
        raise ValueError("blob_hash must be a 32-byte SHA-256 digest")

    columns = {str(info[1]) for info in conn.execute(f"PRAGMA {schema}.table_info(raw_sessions)")}
    names = [
        "raw_id",
        "origin",
        "native_id",
        "source_path",
        "source_index",
        "blob_hash",
        "blob_size",
        "acquired_at_ms",
        "logical_source_key",
        "revision_kind",
        "source_revision",
        "baseline_raw_id",
        "acquisition_generation",
        "revision_authority",
    ]
    values: list[object] = [
        row.raw_id,
        row.origin,
        row.native_id,
        row.source_path,
        row.source_index,
        row.blob_hash,
        row.blob_size,
        row.acquired_at_ms,
        row.logical_source_key,
        RawRevisionKind.FULL.value,
        row.source_revision,
        row.baseline_raw_id,
        0,
        RawRevisionAuthority.BYTE_PROVEN.value,
    ]
    if "capture_mode" in columns:
        names.insert(2, "capture_mode")
        values.insert(2, row.capture_mode)
    conn.execute(
        f"INSERT INTO {schema}.raw_sessions ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
        values,
    )


def _admit_artifact(
    conn: sqlite3.Connection,
    *,
    origin: Origin | str,
    capture_mode: Provider | str | None,
    source_path: str,
    source_index: int,
    payload: bytes,
    acquired_at_ms: int,
    native_id: str | None,
    raw_id: str | None,
    artifact: ArtifactClassification,
    blob_publication_receipt_id: str | None,
    manage_transaction: bool,
) -> RawAdmissionResult:
    origin_value = _enum_value(origin)
    artifact_id = artifact_observation_id(
        source_name=origin_value,
        source_path=source_path,
        source_index=source_index,
    )
    raw_id = write_source_raw_session(
        conn,
        origin=origin,
        capture_mode=capture_mode,
        source_path=source_path,
        source_index=source_index,
        payload=payload,
        acquired_at_ms=acquired_at_ms,
        native_id=native_id,
        raw_id=raw_id,
        blob_publication_receipt_id=blob_publication_receipt_id,
        # Artifacts are acquisition-evidence, not part of any revision
        # chain: no logical_source_key/predecessor tracking applies to a
        # non-conversational sidecar payload.
        revision=None,
        artifact=ArchiveSourceArtifact(
            artifact_id=artifact_id,
            origin=origin,
            source_path=source_path,
            artifact_kind=artifact.cohort,
            classification_reason=artifact.reason,
            support_status=ArtifactSupportStatus.UNKNOWN,
            parse_as_session=False,
            schema_eligible=artifact.schema_eligible,
            first_observed_at_ms=acquired_at_ms,
            last_observed_at_ms=acquired_at_ms,
            source_index=source_index,
        ),
        manage_transaction=manage_transaction,
    )
    return RawAdmissionResult(arm=RawAdmissionArm.ARTIFACT, raw_id=raw_id, artifact_id=artifact_id)


__all__ = [
    "PriorRawHead",
    "RawAdmissionArm",
    "RawAdmissionResult",
    "ReconstructedRawRow",
    "admit_raw_artifact_blob_observation",
    "admit_raw_blob_observation",
    "admit_raw_observation",
    "insert_reconstructed_raw_row",
]

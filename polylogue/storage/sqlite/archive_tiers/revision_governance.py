"""Raw revision & membership governance: the authority over which raw bytes win.

Writer module: index, source.
Twin-write contract: raw-revision-replay.

Extracted from ``archive_tiers/archive.py`` (polylogue-1r9c hotspot-map slice,
2026-07-30 defect-concentration cut — PRs #3394/#3396/#3397/#3398/#3401 all
landed in this cluster or its callers). ``archive.py``'s documented contract
is "every SELECT-shaped query surface (sessions, messages, blocks, insights
reads, search)" (see ``docs/architecture-hotspots.md``); this module owns a
different, write-authority concern that used to live inside that file.

## What this module owns

Given one *logical source key* (one conversation as re-exported/re-captured
possibly many times), this module decides:

- which raw bytes are the authoritative full snapshot vs. an appendable tail
  vs. a duplicate vs. a genuine, unresolved conflict
  (``prepare_raw_revision_byte_classification``, ``raw_revision_replay_plan``,
  ``_raw_revision_candidates``, ``_authorize_full_snapshot_fold``);
- whether a parsed session is allowed to overwrite ``sessions`` at all, given
  the raw's recorded membership decision and revision-authority state
  (``_write_parsed_precedence_result``, ``apply_raw_revision_replay``,
  ``apply_prepared_membership_index``);
- how a raw's membership in a revision cohort is computed, persisted, and
  queried (``replace_raw_membership_census``, ``raw_membership_*``);
- bookkeeping for a raw's own parse lifecycle
  (``finalize_raw_parse_state``, ``mark_raw_parse_failed/succeeded``) and the
  narrow prepared raw-write path that hands a parsed session to this
  authority (``_index_parsed_for_retained_raw``).

## What this module refuses

- It does not decide *comparison identity* for a session (which fields make
  two acquisitions "the same conversation") — that lives in
  ``polylogue.pipeline.ids`` and ``polylogue.archive.session_revision_membership``,
  untouched here on purpose (live lane, see polylogue-aggz).
- It does not read sessions/messages/blocks back out — that stays the query
  surface's job in ``archive_tiers/archive.py``.
- It does not own hook-event ingest (``write_hook_event`` stays in
  ``archive.py`` — a hook is evidence linked to a session, never itself a raw
  revision candidate; polylogue-31r1).

## The connection interface

Revision binding and persisted replay reads take ``RawRevisionSourceHost``.
Functions that lower Index changes take ``RawRevisionGovernanceHost`` and require
the Store's exact mutation scope. Both protocols use the caller's actual tier
connections. ``ArchiveStore`` owns
a persistent lazy ``source.db`` connection plus in-flight write-batch state
(pending blob receipts, pending raw-parse-state flushes, the blob publisher)
as instance attributes; the governance surface needs a subset of that state
but must not gain silent access to the other ~9,000 lines of read-surface
internals that live alongside it. ``RawRevisionGovernanceHost`` is a
``Protocol`` extending Source authority with the actual Index connection,
mutation scope and pending publication state. ``ArchiveStore`` satisfies both
interfaces structurally. The Drive Source adapter implements only Source
binding and replay reads; it carries no placeholder Index handle.

This was chosen over two alternatives: (a) passing the raw ``sqlite3.Connection``
alone — insufficient, because several functions need the lazily-opened
source.db connection, the blob publisher, and the pending-raw-parse-state
batch, not just one connection; (b) a mixin ``ArchiveStore`` inherits from —
rejected because inheritance gives every moved method unrestricted `self`
access to all of ``ArchiveStore``'s state, which is exactly the "reach back
into internals" this module is supposed to make impossible to do by
accident. The Protocol makes the dependency surface an explicit, readable
contract instead of "whatever `self` happens to have".

Raw convergence prepares byte authority with
``prepare_raw_revision_byte_classification`` on its original Source seal.
Publication validates the selected blob identities and applies that seal's
Source statements. ``ArchiveStore`` exposes the persisted replay plan, not a
separate direct classifier or frozen-source remediation route.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Protocol, cast

from polylogue.archive.revision_authority import (
    HistoricalRawRevisionStream,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
    canonical_authority_logical_key,
    classify_historical_full_revision_streams,
    is_work_event_raw_id,
    parser_census_identity_measurement,
    raw_authority_parser_fingerprint,
    raw_receipt_order_sql,
)
from polylogue.core.compute_cancel import check_compute_cancelled, compute_cancel_requested
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    REVISION_HEAD_ROW_FIELDS,
    FullRevisionReplacementAuthorization,
    FullSnapshotFoldAuthorization,
    RevisionApplicationReceipt,
    assert_session_fts_exact_sync,
    prepare_revision_application_head,
    record_revision_application_sync,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    PENDING_RAW_LOGICAL_SOURCE_PREFIX,
    ArchiveSourceArtifact,
    ArchiveSourceBlobRef,
    PreparedParserSingletonWitness,
    RawRevisionBindingProducer,
    SourceArtifactProducer,
    SourceRawStateProducer,
    _apply_source_raw_state_update,
    _artifact_coordinate_query,
    _artifact_observation_query,
    _bind_parser_singleton_revision,
    _prepare_parser_singleton_revision,
    _revision_values,
    _upsert_raw_artifact,
    apply_source_raw_state_update,
    bind_source_raw_revision,
    prepare_parser_singleton_witness,
    write_source_raw_session,
    write_source_raw_session_blob_ref,
)
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveWriteOutcome,
    BeforeIndexInput,
    ConnectionSessionSourceRead,
    MembershipHeadSourceRead,
    PreparedSessionRows,
    PreparedSessionWrite,
    PreparedSessionWriteRefusedError,
    SessionSourceRead,
    _retain_stale_session_observations,
    recorded_attachment_owner_gaps,
    replace_parser_ingest_flag_tags,
    session_revision_row_values,
    upsert_parser_ingest_flag_tags,
    write_parsed_session_to_archive,
)

from .source_items import SourceItemAdmission

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession

from polylogue.archive.artifact_taxonomy import ArtifactClassification
from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_FLAGS
from polylogue.archive.revision_replay import (
    ApplicationDecision,
    RevisionCandidate,
    RevisionReplayPlan,
    plan_revision_replay,
)
from polylogue.archive.session_revision_membership import MembershipClassification, MembershipDecision
from polylogue.core.codex_append import strip_codex_legacy_append_header
from polylogue.core.enums import Origin, Provider
from polylogue.core.errors import RawCASFrontierError
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
    MissingProfileIdentityError,
    RawFailureEvidenceKind,
    RetainedZipMembershipUnprovedError,
    raw_failure_classification_reason,
)
from polylogue.core.sources import origin_from_provider, provider_from_origin
from polylogue.core.timestamp_authority import (
    normalize_session_timestamps,
    session_evidence_timestamps,
)
from polylogue.pipeline.ids import (
    SessionRevisionProjection,
    bound_session_content_hash,
    session_content_hash,
    session_revision_projection,
)
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.security.excision_policy import ExcisionPolicySnapshot, build_excision_policy_snapshot
from polylogue.storage.attachment_reasons import AttachmentOwnerResolutionReason
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.fts.derivation import converge_fts_partition_sync
from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.sqlite.archive_tiers.ingest_precedence import (
    BrowserCapturePrecedence,
    browser_capture_precedence,
    record_capture_gap_event,
    record_source_outage_events,
    revision_authority_refuses_write,
    session_has_parser_ingest_flag,
    should_skip_stale_replace,
    stored_message_count,
)
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    RawAdmissionArm,
    RawAdmissionResult,
    admit_raw_artifact_blob_observation,
    admit_raw_blob_observation,
    admit_raw_observation,
)
from polylogue.storage.sqlite.reference_seal import IndexMutationScope


class ActiveByteRevisionChainError(RuntimeError):
    """A byte-identical revision chain cannot admit a conflicting sibling."""


class MembershipReplayConflictError(RawCASFrontierError):
    """Membership replay refused to move an accepted head this pass.

    Carried by a membership head plan (``prepare_membership_head_plan``)
    when it cannot safely retire or replace the currently accepted
    ``raw_revision_heads`` row for a
    logical identity (an unrelated accepted head, or a head with unresolved
    byte-append evidence still hanging off it). This is a **transient,
    retry-eligible** refusal, not proof the raw is unparseable: a later pass
    over the same durable raw bytes can succeed once sibling evidence
    resolves or the accepted head itself changes (polylogue-5iz4).

    A dedicated subclass exists so the raw-failure boundary can persist a
    structured retryable evidence kind. The free-form ``parse_error`` remains
    a diagnostic only and is not an authorization signal for replay.
    """


class PreparedRawClassificationStaleError(RuntimeError):
    """Off-writer byte classification no longer describes the durable source."""


def _blob_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _is_frozen_candidate(store: RawRevisionGovernanceHost) -> bool:
    """Return whether this host is the owned inactive-generation adapter."""
    return bool(getattr(store, "_inactive_candidate_durable_read_only", False))


def _policy_snapshot_for_store(store: RawRevisionGovernanceHost) -> ExcisionPolicySnapshot | None:
    archive_root = getattr(store, "archive_root", None)
    if archive_root is None:
        return None
    return build_excision_policy_snapshot(Path(archive_root))


class RawRevisionSourceHost(Protocol):
    """The actual Source authority used by classification and revision binding."""

    archive_root: Path
    _blob_publisher: ArchiveBlobPublisher | None

    def _ensure_source_conn(self) -> sqlite3.Connection: ...


class RawRevisionGovernanceHost(RawRevisionSourceHost, Protocol):
    """The narrow slice of ``ArchiveStore`` this module is allowed to touch.

    ``ArchiveStore`` satisfies this structurally (duck typing) — it is never
    declared as implementing it explicitly, so this module never imports
    ``ArchiveStore`` and no import cycle can form.
    """

    _conn: sqlite3.Connection
    index_db_path: Path

    def index_mutation_scope(self) -> AbstractContextManager[IndexMutationScope]: ...

    def open_raw_revision_material(
        self,
        raw_id: str,
    ) -> AbstractContextManager[tuple[Provider, BinaryIO, str, RawRevisionKind]]: ...

    _write_lease_archive_root: Path
    _inactive_candidate_durable_read_only: bool
    _pending_raw_parse_states: list[tuple[str, RawSessionStateUpdate]]

    def commit(self) -> None: ...

    def _preacquire_attachment_blobs(
        self,
        session: ParsedSession,
        *,
        source_path: str,
        acquired_at_ms: int,
    ) -> tuple[
        dict[Any, tuple[bytes | None, int, str]],
        tuple[ArchiveSourceBlobRef, ...],
    ]: ...

    @staticmethod
    def _write_counts(session: ParsedSession) -> dict[str, int]: ...

    @staticmethod
    def _skipped_counts(session: ParsedSession, *, session_events: int = 0) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class ArchiveRawParsedWriteResult:
    """Result of one raw acquisition plus parsed-session write."""

    raw_id: str
    session_id: str
    content_changed: bool
    counts: dict[str, int]
    # A legitimate unchanged derivation can still settle its original Source
    # acknowledgement. Refusal is a distinct actual writer decision.
    publication_refused: bool = False
    # The ordinary raw replay route must retain the writer's typed owner
    # decisions.  An empty tuple means every written attachment got a ref;
    # entries are ``(attachment_id, reason)`` for deliberate non-links.
    unresolved_attachment_owners: tuple[tuple[str, AttachmentOwnerResolutionReason], ...] = ()


def _source_integer(value: object) -> int:
    """Require the integer emitted by the canonical Source schema or query."""
    if type(value) is not int:
        raise ValueError("canonical Source integer operand has another storage type")
    return value


def _source_scalar(value: object) -> None | int | float | str | bytes:
    if value is None or isinstance(value, (int, float, str, bytes)):
        return value
    raise ValueError("canonical Source statement operand is not a SQLite scalar")


class _PreparedAttachmentChain(Mapping[object, tuple[bytes | None, int, str]]):
    """Read original attachment views in newest-first order without copying them."""

    def __init__(self, mappings: Iterable[Mapping[object, tuple[bytes | None, int, str]]]) -> None:
        self._mappings = tuple(mappings)

    def __getitem__(self, key: object) -> tuple[bytes | None, int, str]:
        for mapping in self._mappings:
            try:
                return mapping[key]
            except KeyError:
                continue
        raise KeyError(key)

    def __iter__(self) -> Iterator[object]:
        seen: set[object] = set()
        for mapping in self._mappings:
            for key in mapping:
                if key not in seen:
                    seen.add(key)
                    yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)


def _bind_retained_enrichment(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    session_id: str,
    prepared_write: PreparedSessionWrite,
) -> None:
    """Consume the current evidence captured by this original session carrier."""
    from polylogue.sources.revision_backfill import record_session_enrichment_binding

    binding = prepared_write.enrichment_binding
    if binding is None:
        return
    if prepared_write.session_id != session_id:
        raise PreparedSessionWriteRefusedError("enrichment publication names another prepared session")
    record_session_enrichment_binding(
        store._conn,
        session_id=session_id,
        carried_key=session.enrichment_evidence_key,
        current_key=binding[1],
    )


def _work_events_already_stored(conn: sqlite3.Connection, session_id: str, session: ParsedSession) -> bool:
    """Whether every event a work-event raw carries is already on its session."""
    return bool(session.session_events) and all(
        conn.execute(
            """
            SELECT 1 FROM session_events
            WHERE session_id = ? AND event_type = ? AND json_extract(payload_json, '$.event_id') = ?
            LIMIT 1
            """,
            (session_id, event.event_type, str(event.payload.get("event_id"))),
        ).fetchone()
        is not None
        for event in session.session_events
    )


def _write_parsed_precedence_result(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    raw_id: str,
    source_index: int,
    stage_timings_s: dict[str, float] | None,
    stage_timing_prefix: str,
    manage_transaction: bool,
    preacquired_attachment_blobs: Mapping[object, tuple[bytes | None, int, str]] | None = None,
    revision_authoritative: bool = False,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    defer_fts_rebuild: bool = False,
    prepared: PreparedSessionRows | None = None,
    prepared_required: bool = False,
    prepared_write: PreparedSessionWrite | None = None,
    content_hash: str | None = None,
) -> ArchiveRawParsedWriteResult:
    if prepared_write is None:
        raise PreparedSessionWriteRefusedError("retained publication requires its original prepared session write")
    session = normalize_session_timestamps(session, fallback_timestamp=prepared_write.fallback_timestamp)
    session_id = str(make_session_id(session.source_name, session.provider_session_id))
    if content_hash is None:
        content_hash = str(bound_session_content_hash(session) or session_content_hash(session))
    else:
        try:
            supplied_hash = bytes.fromhex(content_hash)
        except ValueError as exc:
            raise ValueError("supplied session content_hash must be hexadecimal text") from exc
        if len(supplied_hash) != 32:
            raise ValueError("supplied session content_hash must be a SHA-256 digest")
        content_hash = supplied_hash.hex()
    existing_row = (
        None
        if fresh_build
        else store._conn.execute(
            "SELECT content_hash, raw_id, updated_at_ms, parser_fingerprint, lowering_fingerprint "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    )
    if is_work_event_raw_id(raw_id) and _work_events_already_stored(store._conn, session_id, session):
        # An event-only write keeps the transcript's content hash, so the
        # hash cannot show that this event is already recorded; its id can.
        return ArchiveRawParsedWriteResult(
            raw_id=raw_id,
            session_id=session_id,
            content_changed=False,
            counts=store._skipped_counts(session),
        )
    existing_raw_id = str(existing_row["raw_id"] or "") if existing_row is not None else ""
    existing_hash = existing_row["content_hash"] if existing_row is not None else None
    existing_hash_hex = existing_hash.hex() if isinstance(existing_hash, bytes) else str(existing_hash or "")
    content_unchanged = existing_row is not None and existing_hash_hex == content_hash
    if content_unchanged:
        incoming_aliases = {
            str(value).strip()
            for value in (session.provider_session_id, *session.provider_session_aliases)
            if str(value).strip()
        }
        stored_aliases = {
            str(row[0])
            for row in store._conn.execute(
                "SELECT provider_value FROM session_identity_claims "
                "WHERE claimant_session_id = ? AND identity_namespace = 'provider-session'",
                (session_id,),
            ).fetchall()
        }
        # Alias claims are excluded from content identity but still drive
        # lineage resolution. Force the normal writer when their set changes
        # so it refreshes claims and invalidates affected child links.
        content_unchanged = incoming_aliases == stored_aliases
    existing_is_dom_fallback = False
    incoming_is_dom_fallback = DOM_FALLBACK_INGEST_FLAG in session.ingest_flags
    existing_has_native_browser_payload = False
    incoming_has_native_browser_payload = any(flag in session.ingest_flags for flag in NATIVE_BROWSER_CAPTURE_FLAGS)
    current_stored_message_count = 0
    browser_precedence: BrowserCapturePrecedence = "default"

    writer_outcomes: list[ArchiveWriteOutcome] = []

    def write_with_reparse_receipt(*, force_replace: bool) -> None:
        """Keep a reparse receipt and its session replacement in one index txn."""
        from polylogue.storage.sqlite.reference_seal import current_index_mutation_scope

        if not manage_transaction and current_index_mutation_scope() is None:
            raise RuntimeError("manage_transaction=False requires the caller's live Index mutation scope")
        with store.index_mutation_scope() as mutation_scope:
            write_parsed_session_to_archive(
                store._conn,
                session,
                content_hash=content_hash,
                raw_id=raw_id,
                child_source_path=prepared_write.context.child_source_path,
                fallback_timestamp=prepared_write.fallback_timestamp,
                merge_append=source_index < 0,
                force_replace=force_replace,
                stage_timings_s=stage_timings_s,
                stage_timing_prefix=stage_timing_prefix,
                preacquired_attachment_blobs=preacquired_attachment_blobs,
                # The helper owns the transaction boundary when requested;
                # otherwise the caller owns it. Do not let the session
                # writer's own ``with conn`` commit an outer transaction.
                manage_transaction=False,
                mutation_scope=mutation_scope,
                bulk_fts=bulk_fts,
                bulk_build=bulk_build,
                fresh_build=fresh_build,
                fresh_build_batch=fresh_build_batch,
                defer_fts_rebuild=defer_fts_rebuild,
                prepared_write=prepared_write,
                write_outcome=writer_outcomes,
                # Lineage, hook-parent and dispatch-sidecar evidence live in
                # the source tier. Live ingest hands the writer this handle; a
                # replay of the same raws must too, or it rebuilds edges
                # without the witnesses the first acquisition bound them by.
                source_read=ConnectionSessionSourceRead(store._ensure_source_conn()),
            )
            if not writer_outcomes:
                raise RuntimeError("session publication returned no actual write outcome")
            if writer_outcomes[-1].wrote:
                record_prepared_accepted_head_reparse_receipt(
                    store._conn,
                    prepared_write,
                    content_hash=content_hash,
                    decided_at_ms=int(time.time() * 1000),
                )
                _bind_retained_enrichment(store, session, session_id=session_id, prepared_write=prepared_write)

    def browser_capture_refusal() -> ArchiveRawParsedWriteResult | None:
        nonlocal existing_is_dom_fallback, existing_has_native_browser_payload
        nonlocal current_stored_message_count, browser_precedence
        existing_is_dom_fallback = session_has_parser_ingest_flag(
            store._conn,
            session_id,
            DOM_FALLBACK_INGEST_FLAG,
        )
        existing_has_native_browser_payload = session_has_parser_ingest_flag(
            store._conn,
            session_id,
            NATIVE_BROWSER_CAPTURE_FLAGS,
        )
        current_stored_message_count = stored_message_count(store._conn, session_id)
        lower_precedence_fallback = incoming_is_dom_fallback and not existing_is_dom_fallback
        browser_precedence = browser_capture_precedence(
            existing_is_dom_fallback=existing_is_dom_fallback,
            incoming_is_dom_fallback=incoming_is_dom_fallback,
            existing_has_native_payload=existing_has_native_browser_payload,
            incoming_has_native_payload=incoming_has_native_browser_payload,
            stored_message_count=current_stored_message_count,
            incoming_message_count=len(session.messages),
        )
        if browser_precedence == "skip":
            session_event_count = 0
            if lower_precedence_fallback:
                record_capture_gap_event(
                    store._conn,
                    session_id=session_id,
                    existing_raw_id=existing_raw_id,
                    incoming_raw_id=raw_id,
                    stored_message_count=current_stored_message_count,
                    incoming_message_count=len(session.messages),
                )
                session_event_count = 1
            session_event_count += record_source_outage_events(
                store._conn,
                session_id=session_id,
                events=session.session_events,
            )
            if manage_transaction:
                store._conn.commit()
            return ArchiveRawParsedWriteResult(
                raw_id=raw_id,
                session_id=session_id,
                content_changed=False,
                publication_refused=True,
                counts=store._skipped_counts(session, session_events=session_event_count),
            )
        return None

    if content_unchanged:
        from polylogue.storage.sqlite.archive_tiers.session_suppression import session_write_is_suppressed

        # An unchanged transcript is still refused when the operator hid it.
        # This applies to ordinary and authoritative publication alike.
        content_unchanged = not session_write_is_suppressed(store._conn, session_id)
    if revision_authoritative and content_unchanged:
        from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
        from polylogue.storage.sqlite.archive_tiers.write import prepared_session_storage_lineage_matches

        # Executable identity, aliases and physical prefix representation are
        # separate obligations from semantic content identity.
        assert existing_row is not None
        content_unchanged = (
            not is_work_event_raw_id(raw_id)
            and tuple(existing_row[name] for name in ("content_hash", "raw_id", "updated_at_ms"))
            == prepared_write.predecessor
            and existing_row["parser_fingerprint"]
            == parser_fingerprint_for_origin(origin_from_provider(session.source_name))
            and existing_row["lowering_fingerprint"] == lowering_fingerprint()
            and prepared_write.cross_acquisition_union is None
            and prepared_session_storage_lineage_matches(store._conn, prepared_write)
        )
    if (
        revision_authoritative
        and incoming_is_dom_fallback
        and source_index >= 0
        and existing_raw_id
        and raw_id
        and existing_raw_id != raw_id
        and session_has_parser_ingest_flag(store._conn, session_id, NATIVE_BROWSER_CAPTURE_FLAGS)
    ):
        refusal = browser_capture_refusal()
        if refusal is not None:
            return refusal
    if revision_authoritative and not content_unchanged:
        write_with_reparse_receipt(force_replace=source_index >= 0 and not fresh_build)
        # The writer refuses a session the operator tombstoned in user.db.
        # Authoritative replay must not then claim it changed archive content:
        # the run's receipt has to show the refusal, not a phantom write.
        outcome = writer_outcomes[-1]
        return ArchiveRawParsedWriteResult(
            raw_id=raw_id,
            session_id=session_id,
            content_changed=outcome.wrote,
            counts=store._write_counts(session) if outcome.wrote else store._skipped_counts(session),
            publication_refused=outcome.stale_skipped or outcome.suppression_skipped,
            unresolved_attachment_owners=(writer_outcomes[-1].unresolved_attachment_owners if writer_outcomes else ()),
        )
    # A work event annotates its session; it is never a competing revision of
    # the transcript, so the transcript's accepted head cannot refuse it.
    if (
        not revision_authoritative
        and not is_work_event_raw_id(raw_id)
        and revision_authority_refuses_write(
            store._conn,
            store._ensure_source_conn(),
            session_id=session_id,
            raw_id=raw_id,
            provider_session_id=session.provider_session_id,
        )
    ):
        with store.index_mutation_scope():
            _retain_stale_session_observations(store._conn, session_id, session)
        return ArchiveRawParsedWriteResult(
            raw_id=raw_id,
            session_id=session_id,
            content_changed=False,
            publication_refused=True,
            counts=store._skipped_counts(session),
        )

    if not revision_authoritative and source_index >= 0 and existing_raw_id and raw_id and existing_raw_id != raw_id:
        refusal = browser_capture_refusal()
        if refusal is not None:
            return refusal

    _incoming_created_at_ms, incoming_freshness_ms = session_evidence_timestamps(session)
    if incoming_freshness_ms is None:
        incoming_freshness_ms = _incoming_created_at_ms
    if (
        not revision_authoritative
        and source_index >= 0
        and browser_precedence != "replace"
        and existing_row is not None
        and incoming_freshness_ms is not None
    ):
        existing_updated_at_ms = existing_row["updated_at_ms"]
        existing_updated_at_int = int(existing_updated_at_ms) if existing_updated_at_ms is not None else None
        if should_skip_stale_replace(
            incoming_freshness_ms=incoming_freshness_ms,
            existing_updated_at_ms=existing_updated_at_int,
        ):
            # This early return bypasses the ordinary index write transaction;
            # make stale observation repair durable for direct governance calls.
            with store.index_mutation_scope():
                _retain_stale_session_observations(store._conn, session_id, session)
            return ArchiveRawParsedWriteResult(
                raw_id=raw_id,
                session_id=session_id,
                content_changed=False,
                publication_refused=True,
                counts=store._skipped_counts(session),
            )

    if content_unchanged:
        if revision_authoritative:
            from polylogue.storage.sqlite.archive_tiers.write import (
                _resolve_session_graph,
                _write_session_link,
                validate_prepared_session_lineage,
            )
            from polylogue.storage.sqlite.delegation_facts import refresh_delegation_facts_for_sessions
            from polylogue.storage.sqlite.reference_seal import note_current_lineage_change

            source_read = ConnectionSessionSourceRead(store._ensure_source_conn())
            context = prepared_write.context
            pending_hash = bytes.fromhex(content_hash)
            if (
                prepared_write.session_id != session_id
                or prepared_write.input_content_hash != pending_hash
                or prepared_write.merge_append != (source_index < 0)
                or prepared_write.rows.session_content_hash != pending_hash
            ):
                raise PreparedSessionWriteRefusedError("prepared write is stale or has a different pending input")
            validate_prepared_session_lineage(store._conn, session, prepared_write, source_read=source_read)
            _write_session_link(
                store._conn,
                session_id,
                context.effective_session,
                branch_point_message_id=context.branch_point_message_id,
                branch_point_content_address=context.branch_point_content_address,
                inheritance=context.lineage_inheritance,
                source_read=source_read,
                prior_links=True,
                child_source_path=context.child_source_path,
            )
            note_current_lineage_change(store._conn, session_id)
            graph_changed_ids = _resolve_session_graph(
                store._conn,
                session_id,
                session.provider_session_id,
                origin_from_provider(session.source_name).value,
                bulk_fts=bulk_fts,
                bulk_build=bulk_build,
                source_read=source_read,
            )
            if not bulk_build:
                refresh_delegation_facts_for_sessions(store._conn, {session_id, *graph_changed_ids})
        if browser_precedence == "replace":
            replace_parser_ingest_flag_tags(store._conn, session_id, session.ingest_flags)
        elif session.ingest_flags:
            upsert_parser_ingest_flag_tags(store._conn, session_id, session.ingest_flags)
        raw_link_changed = False
        if raw_id and raw_id != existing_raw_id:
            cursor = store._conn.execute(
                "UPDATE sessions SET raw_id = ? WHERE session_id = ? AND (raw_id IS NULL OR raw_id != ?)",
                (raw_id, session_id, raw_id),
            )
            raw_link_changed = bool(cursor.rowcount)
        fts_repaired = converge_fts_partition_sync(store._conn, session_id)
        # Unchanged content enriched against current evidence is still an
        # accepted derivation from that evidence; bind it, or an evidence
        # move that happens not to change the output re-derives forever.
        _bind_retained_enrichment(store, session, session_id=session_id, prepared_write=prepared_write)
        if manage_transaction:
            store._conn.commit()
        counts = store._skipped_counts(session)
        counts["raw_links"] = int(raw_link_changed)
        counts["_fts_repair"] = int(fts_repaired)
        return ArchiveRawParsedWriteResult(
            raw_id=raw_id,
            session_id=session_id,
            content_changed=False,
            counts=counts,
            # Unchanged content leaves the same attachments unowned; report the
            # gaps its last write recorded rather than none.
            unresolved_attachment_owners=recorded_attachment_owner_gaps(store._conn, session_id),
        )

    write_with_reparse_receipt(force_replace=browser_precedence == "replace")
    if writer_outcomes[-1].stale_skipped or writer_outcomes[-1].suppression_skipped:
        return ArchiveRawParsedWriteResult(
            raw_id=raw_id,
            session_id=session_id,
            content_changed=False,
            publication_refused=True,
            counts=store._skipped_counts(session),
        )
    counts = store._write_counts(session)
    if (
        existing_raw_id
        and raw_id
        and existing_raw_id != raw_id
        and existing_is_dom_fallback
        and not incoming_is_dom_fallback
    ):
        record_capture_gap_event(
            store._conn,
            session_id=session_id,
            existing_raw_id=existing_raw_id,
            incoming_raw_id=raw_id,
            stored_message_count=current_stored_message_count,
            incoming_message_count=len(session.messages),
        )
        counts["session_events"] += 1
        if manage_transaction:
            store._conn.commit()
    return ArchiveRawParsedWriteResult(
        raw_id=raw_id,
        session_id=session_id,
        content_changed=True,
        counts=counts,
        unresolved_attachment_owners=(writer_outcomes[-1].unresolved_attachment_owners if writer_outcomes else ()),
    )


def write_raw_payload(
    store: RawRevisionGovernanceHost,
    *,
    provider: Provider,
    capture_mode: Provider | None = None,
    payload: bytes,
    source_path: str,
    canonical_source_path: str,
    captured_profile_key: str | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
    source_item: SourceItemAdmission | None = None,
    addressing_mode: str | None = None,
    content_identity: str | None = None,
    acquired_at_ms: int,
    file_mtime_ms: int | None = None,
    source_index: int = 0,
    raw_id: str | None = None,
    native_id: str | None = None,
    blob_publication_receipt_id: str | None = None,
    revision: RawRevisionEnvelope | None = None,
    post_parse: bool = False,
) -> str:
    """Commit raw bytes before attempting to parse or index them.

    ``native_id``, when known, records the provider session identity this
    raw evidence belongs to (polylogue-u19l: used by Codex append-mode live
    captures, whose record stream has no ``session_meta`` of its own to
    self-describe it). It is sidecar metadata, not part of the hashed
    payload, so replay (``sources/revision_backfill.py``'s
    ``parse_retained_raw_sessions``) can recover the same identity it was
    written with without the identity ever having been spliced into the
    stored bytes.

    ``post_parse=False`` does NOT go through the raw-admission chokepoint: it
    writes a bare row with whatever ``revision`` envelope the caller supplies
    (possibly none). polylogue-1fijp surveyed this branch rather than migrating
    it, because no live acquisition route reaches it -- every production
    acquisition caller (``sources/live/append_ingest.py``,
    ``sources/live/batch.py``) passes ``post_parse=True`` and resolves the
    ``POST_PARSE_PENDING`` arm. What remains on this branch is fixture seeding:
    ``durable_change_train.py``'s temp-archive self-probes and ``tests/infra``
    corpus builders, which need to plant specific row shapes that admission
    would normalize away. Treat a NEW production caller of this branch as a
    chokepoint bypass, not as precedent.
    """
    policy_snapshot = _policy_snapshot_for_store(store)
    # DRIVE is a deliberate compatibility carrier for the non-injective
    # AISTUDIO_DRIVE origin mapping.  Preserve it when callers use the legacy
    # direct-provider API; every other omitted route remains unknown.
    observed_capture_mode = (
        capture_mode if capture_mode is not None else (Provider.DRIVE if provider is Provider.DRIVE else None)
    )
    if store._blob_publisher is None:
        raise RuntimeError("raw archive writes require a writable archive publisher")
    if blob_publication_receipt_id is None:
        raw_hash, _raw_size = store._blob_publisher.write_from_bytes(payload)
        blob_publication_receipt_id = store._blob_publisher.receipt_id(raw_hash)
    store._blob_publisher.flush()
    if captured_zip_coordinate is not None:
        if not post_parse or revision is not None:
            raise ValueError("captured ZIP payload requires post-parse admission")
        return write_raw_blob_ref(
            store,
            provider=provider,
            capture_mode=capture_mode,
            post_parse=True,
            blob_hash_hex=hashlib.sha256(payload).hexdigest(),
            blob_size=len(payload),
            source_path=source_path,
            canonical_source_path=canonical_source_path,
            captured_profile_key=captured_profile_key,
            captured_zip_coordinate=captured_zip_coordinate,
            source_item=source_item,
            addressing_mode=addressing_mode,
            content_identity=content_identity,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            source_index=source_index,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
        )
    if post_parse:
        if revision is not None:
            raise ValueError("post-parse raw admission cannot receive a revision envelope")
        admission = admit_raw_observation(
            store._ensure_source_conn(),
            origin=origin_from_provider(provider),
            # A missing capture mode is an unknown observation.  Do not
            # manufacture route provenance from provider identity; callers
            # that genuinely observed a provider-shaped route pass it
            # explicitly.
            capture_mode=observed_capture_mode,
            source_path=source_path,
            canonical_source_path=canonical_source_path,
            captured_profile_key=captured_profile_key,
            source_index=source_index,
            payload=payload,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            native_id=native_id,
            raw_id=raw_id,
            post_parse=True,
            blob_publication_receipt_id=blob_publication_receipt_id,
            manage_transaction=True,
            policy_snapshot=policy_snapshot,
        )
        if admission.arm is not RawAdmissionArm.POST_PARSE_PENDING:
            raise RuntimeError(f"unexpected post-parse raw admission arm: {admission.arm!r}")
        return admission.raw_id
    if revision is not None:
        # Revision-bearing calls are production admissions, rather than fixture
        # seeding. Give the chokepoint the caller's already-proven baseline
        # envelope so it creates the row with that authority atomically.
        admission = admit_raw_observation(
            store._ensure_source_conn(),
            origin=origin_from_provider(provider),
            capture_mode=observed_capture_mode,
            source_path=source_path,
            canonical_source_path=canonical_source_path,
            captured_profile_key=captured_profile_key,
            source_index=source_index,
            payload=payload,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            native_id=native_id,
            raw_id=raw_id,
            logical_source_key=revision.logical_source_key,
            baseline_revision=revision,
            blob_publication_receipt_id=blob_publication_receipt_id,
            manage_transaction=True,
            policy_snapshot=policy_snapshot,
        )
        return admission.raw_id
    return write_source_raw_session(
        store._ensure_source_conn(),
        origin=origin_from_provider(provider),
        capture_mode=observed_capture_mode,
        source_path=source_path,
        canonical_source_path=canonical_source_path,
        captured_profile_key=captured_profile_key,
        source_index=source_index,
        payload=payload,
        acquired_at_ms=acquired_at_ms,
        file_mtime_ms=file_mtime_ms,
        raw_id=raw_id,
        native_id=native_id,
        blob_publication_receipt_id=blob_publication_receipt_id,
        revision=revision,
        manage_transaction=True,
        policy_snapshot=policy_snapshot,
    )


def write_raw_blob_ref(
    store: RawRevisionGovernanceHost,
    *,
    provider: Provider,
    capture_mode: Provider | None = None,
    blob_hash_hex: str,
    blob_size: int,
    source_path: str,
    canonical_source_path: str,
    captured_profile_key: str | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
    source_item: SourceItemAdmission | None = None,
    addressing_mode: str | None = None,
    content_identity: str | None = None,
    acquired_at_ms: int,
    file_mtime_ms: int | None = None,
    source_index: int = 0,
    raw_id: str | None = None,
    blob_publication_receipt_id: str | None = None,
    revision: RawRevisionEnvelope | None = None,
    post_parse: bool = False,
) -> str:
    """Commit a prepublished raw blob reference before parsing it.

    See :func:`write_raw_payload` for why the ``post_parse=False`` branch is a
    surveyed fixture-seeding path rather than a migrated admission route
    (polylogue-1fijp); the only caller reaching it here is test infrastructure.
    """
    policy_snapshot = _policy_snapshot_for_store(store)
    observed_capture_mode = (
        capture_mode if capture_mode is not None else (Provider.DRIVE if provider is Provider.DRIVE else None)
    )
    if store._blob_publisher is not None:
        store._blob_publisher.flush()
    if post_parse:
        if revision is not None:
            raise ValueError("post-parse raw admission cannot receive a revision envelope")
        admission = admit_raw_blob_observation(
            store._ensure_source_conn(),
            origin=origin_from_provider(provider),
            capture_mode=observed_capture_mode,
            source_path=source_path,
            canonical_source_path=canonical_source_path,
            captured_profile_key=captured_profile_key,
            captured_zip_coordinate=captured_zip_coordinate,
            source_item=source_item,
            addressing_mode=addressing_mode,
            content_identity=content_identity,
            source_index=source_index,
            blob_hash=bytes.fromhex(blob_hash_hex),
            blob_size=blob_size,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
            policy_snapshot=policy_snapshot,
        )
        # A re-imported source member whose accepted raw already holds these
        # exact bytes is the idempotent no-op arm: admission verified the
        # identity and content and returns the existing raw id.
        if admission.arm not in {RawAdmissionArm.POST_PARSE_PENDING, RawAdmissionArm.SKIP_DUPLICATE}:
            raise RuntimeError(f"unexpected post-parse blob admission arm: {admission.arm!r}")
        return admission.raw_id
    if captured_zip_coordinate is not None:
        raise ValueError("captured ZIP intake requires ordinary typed admission")
    return write_source_raw_session_blob_ref(
        store._ensure_source_conn(),
        origin=origin_from_provider(provider),
        capture_mode=observed_capture_mode,
        source_path=source_path,
        canonical_source_path=canonical_source_path,
        captured_profile_key=captured_profile_key,
        source_index=source_index,
        blob_hash=bytes.fromhex(blob_hash_hex),
        blob_size=blob_size,
        acquired_at_ms=acquired_at_ms,
        file_mtime_ms=file_mtime_ms,
        raw_id=raw_id,
        blob_publication_receipt_id=blob_publication_receipt_id,
        revision=revision,
        manage_transaction=True,
        policy_snapshot=policy_snapshot,
    )


def admit_raw_artifact_payload(
    store: RawRevisionGovernanceHost,
    *,
    provider: Provider,
    payload: bytes,
    source_path: str,
    canonical_source_path: str,
    captured_profile_key: str | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
    source_item: SourceItemAdmission | None = None,
    addressing_mode: str | None = None,
    content_identity: str | None = None,
    acquired_at_ms: int,
    file_mtime_ms: int | None = None,
    classification: ArtifactClassification,
    source_index: int = 0,
    raw_id: str | None = None,
    blob_publication_receipt_id: str | None = None,
) -> RawAdmissionResult:
    """Commit a non-conversational artifact payload through the raw-admission chokepoint.

    polylogue-1fijp arm 4: routes a payload already classified as
    ``not classification.parse_as_session`` (the omsw artifact taxonomy --
    tool-results/*.json, subagents/workflows/*/journal.jsonl,
    file-history-snapshot, etc) through :func:`admit_raw_observation` so its
    ``raw_artifacts`` classification is attached at admission time rather
    than deferred to the next offline ``materialize_artifact_observations``
    sweep. A ``raw_sessions`` row is still written (the acquisition-evidence
    ledger; ``raw_artifacts.raw_id`` has a NOT NULL FK to it), but this arm
    guarantees the payload never reaches any index-tier writer.
    """
    policy_snapshot = _policy_snapshot_for_store(store)
    if store._blob_publisher is None:
        raise RuntimeError("raw archive writes require a writable archive publisher")
    if blob_publication_receipt_id is None:
        raw_hash, _raw_size = store._blob_publisher.write_from_bytes(payload)
        blob_publication_receipt_id = store._blob_publisher.receipt_id(raw_hash)
    store._blob_publisher.flush()
    if captured_zip_coordinate is not None:
        return admit_raw_artifact_blob_ref(
            store,
            provider=provider,
            classification=classification,
            blob_hash_hex=hashlib.sha256(payload).hexdigest(),
            blob_size=len(payload),
            source_path=source_path,
            canonical_source_path=canonical_source_path,
            captured_profile_key=captured_profile_key,
            captured_zip_coordinate=captured_zip_coordinate,
            source_item=source_item,
            addressing_mode=addressing_mode,
            content_identity=content_identity,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            source_index=source_index,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
        )
    origin = origin_from_provider(provider)
    result = admit_raw_observation(
        store._ensure_source_conn(),
        origin=origin,
        capture_mode=provider,
        source_path=source_path,
        canonical_source_path=canonical_source_path,
        captured_profile_key=captured_profile_key,
        source_index=source_index,
        payload=payload,
        acquired_at_ms=acquired_at_ms,
        file_mtime_ms=file_mtime_ms,
        raw_id=raw_id,
        logical_source_key=f"{origin.value}:{source_path}",
        prior_head=None,
        artifact=classification,
        blob_publication_receipt_id=blob_publication_receipt_id,
        manage_transaction=True,
        policy_snapshot=policy_snapshot,
    )
    # Acquisition retains the typed artifact and original bytes. The resident
    # Source phase records its parser receipt on the original prepared witness.
    return result


def admit_raw_artifact_blob_ref(
    store: RawRevisionGovernanceHost,
    *,
    provider: Provider,
    blob_hash_hex: str,
    blob_size: int,
    source_path: str,
    canonical_source_path: str,
    captured_profile_key: str | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
    source_item: SourceItemAdmission | None = None,
    addressing_mode: str | None = None,
    content_identity: str | None = None,
    acquired_at_ms: int,
    file_mtime_ms: int | None = None,
    classification: ArtifactClassification,
    source_index: int = 0,
    raw_id: str | None = None,
    blob_publication_receipt_id: str | None = None,
) -> RawAdmissionResult:
    """Admit a prepublished non-session artifact without a pending envelope."""
    policy_snapshot = _policy_snapshot_for_store(store)
    if store._blob_publisher is not None:
        store._blob_publisher.flush()
    result = admit_raw_artifact_blob_observation(
        store._ensure_source_conn(),
        origin=origin_from_provider(provider),
        capture_mode=provider,
        source_path=source_path,
        canonical_source_path=canonical_source_path,
        captured_profile_key=captured_profile_key,
        captured_zip_coordinate=captured_zip_coordinate,
        source_item=source_item,
        addressing_mode=addressing_mode,
        content_identity=content_identity,
        source_index=source_index,
        blob_hash=bytes.fromhex(blob_hash_hex),
        blob_size=blob_size,
        acquired_at_ms=acquired_at_ms,
        file_mtime_ms=file_mtime_ms,
        raw_id=raw_id,
        classification=classification,
        blob_publication_receipt_id=blob_publication_receipt_id,
        policy_snapshot=policy_snapshot,
    )
    # Acquisition retains the typed artifact and original bytes. The resident
    # Source phase records its parser receipt on the original prepared witness.
    return result


def bind_raw_revision(
    store: RawRevisionSourceHost, raw_id: str, revision: RawRevisionEnvelope, *, manage_transaction: bool = True
) -> None:
    """Bind acquisition evidence; ``manage_transaction=False`` batches (polylogue-amg1)."""
    bind_source_raw_revision(store._ensure_source_conn(), raw_id, revision, manage_transaction=manage_transaction)


def promote_reconstructed_legacy_append_revisions(
    store: RawRevisionGovernanceHost,
    revisions: Sequence[tuple[str, RawRevisionEnvelope]],
    *,
    source_prefix_sha256: str,
    source_size: int,
    source_mtime_ns: int,
    source_ctime_ns: int,
    observed_at_ms: int,
) -> None:
    """Promote a byte-proven legacy append chain reconstructed from retained bytes."""
    if not revisions:
        return
    conn = store._ensure_source_conn()
    with conn:
        for raw_id, revision in revisions:
            if (
                revision.kind is not RawRevisionKind.APPEND
                or revision.authority is not RawRevisionAuthority.BYTE_PROVEN
            ):
                raise ValueError("legacy reconstruction can only promote byte-proven append revisions")
            cursor = conn.execute(
                """
                UPDATE raw_sessions
                SET logical_source_key = ?, revision_kind = ?, source_revision = ?,
                    predecessor_source_revision = ?, predecessor_raw_id = ?, baseline_raw_id = ?, append_start_offset = ?,
                    append_end_offset = ?, acquisition_generation = ?, revision_authority = ?,
                    revision_authority_evidence = 'live_source_verification_v1'
                WHERE raw_id = ?
                  AND logical_source_key = ?
                  AND revision_kind = 'unknown'
                  AND revision_authority = 'quarantined'
                  AND source_revision IS NOT NULL
                  AND predecessor_source_revision IS NULL
                  AND predecessor_raw_id IS NULL
                  AND baseline_raw_id IS NULL
                  AND append_start_offset IS NULL
                  AND append_end_offset IS NULL
                """,
                (*_revision_values(revision), raw_id, revision.logical_source_key),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"legacy append revision is no longer eligible for promotion: {raw_id}")
            conn.execute("DELETE FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,))
            row = conn.execute(
                """
                SELECT source_path, blob_hash, blob_size
                FROM raw_sessions
                WHERE raw_id = ?
                """,
                (raw_id,),
            ).fetchone()
            if row is None or revision.append_start_offset is None or revision.append_end_offset is None:
                raise ValueError(f"legacy append revision receipt is unavailable: {raw_id}")
            conn.execute(
                """
                INSERT INTO raw_legacy_append_resynthesis_receipts (
                    raw_id, logical_source_key, source_path, blob_hash, blob_size,
                    append_start_offset, append_end_offset, matched_after_codex_header_strip,
                    previous_revision_authority, source_prefix_sha256, source_size,
                    source_mtime_ns, source_ctime_ns, observed_at_ms, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'quarantined', ?, ?, ?, ?, ?, '')
                """,
                (
                    raw_id,
                    revision.logical_source_key,
                    row[0],
                    row[1],
                    row[2],
                    revision.append_start_offset,
                    revision.append_end_offset,
                    int(row[2]) != revision.append_end_offset - revision.append_start_offset,
                    source_prefix_sha256,
                    source_size,
                    source_mtime_ns,
                    source_ctime_ns,
                    observed_at_ms,
                ),
            )


def release_provisional_full_revisions(store: RawRevisionGovernanceHost, raw_ids: Sequence[str]) -> None:
    """Undo census-time bindings when index authority rejects adoption.

    Only backfill-created full envelopes use ``source_revision=raw_id``;
    asserted/live envelopes and append authority cannot match this guard.
    """
    if not raw_ids:
        return
    placeholders = ",".join("?" for _ in raw_ids)
    conn = store._ensure_source_conn()
    with conn:
        conn.execute(
            f"""
            UPDATE raw_sessions
            SET logical_source_key = NULL,
                revision_kind = 'unknown',
                source_revision = NULL,
                predecessor_source_revision = NULL,
                predecessor_raw_id = NULL,
                baseline_raw_id = NULL,
                append_start_offset = NULL,
                append_end_offset = NULL,
                acquisition_generation = NULL,
                revision_authority = 'quarantined'
            WHERE raw_id IN ({placeholders})
              AND revision_kind = 'full'
              AND source_revision = raw_id
            """,
            tuple(raw_ids),
        )


def raw_full_revision_generation(store: RawRevisionGovernanceHost, logical_source_key: str) -> int:
    """Allocate the next generation from durable, authoritative evidence."""
    row = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT MAX(acquisition_generation)
        FROM raw_sessions
        WHERE logical_source_key = ? AND revision_authority != 'quarantined'
        """,
            (logical_source_key,),
        )
        .fetchone()
    )
    return int(row[0]) + 1 if row is not None and row[0] is not None else 0


def raw_append_revision_parent(
    store: RawRevisionGovernanceHost,
    logical_source_key: str,
    start_offset: int,
    predecessor_revision: str | None,
) -> tuple[str, str, int] | None:
    """Return a unique byte-contiguous predecessor and its baseline."""
    if predecessor_revision is None:
        return None
    rows = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT raw_id, CASE WHEN revision_kind = 'full' THEN raw_id ELSE baseline_raw_id END, acquisition_generation
        FROM raw_sessions
        WHERE logical_source_key = ? AND source_revision = ?
          AND revision_authority != 'quarantined'
          AND ((revision_kind = 'full' AND ? = blob_size)
               OR (revision_kind = 'append' AND append_end_offset = ?))
        ORDER BY acquisition_generation DESC
        LIMIT 2
        """,
            (logical_source_key, predecessor_revision, start_offset, start_offset),
        )
        .fetchall()
    )
    if len(rows) != 1:
        return None
    row = rows[0]
    return str(row[0]), str(row[1]), int(row[2]) + 1


def raw_legacy_append_resynthesis_receipt(store: RawRevisionGovernanceHost, raw_id: str) -> tuple[str, int] | None:
    """Return the live-source proof recorded when one legacy append was promoted."""
    row = (
        store._ensure_source_conn()
        .execute(
            """
            SELECT source_prefix_sha256, matched_after_codex_header_strip
            FROM raw_legacy_append_resynthesis_receipts
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        .fetchone()
    )
    return (str(row[0]), int(row[1])) if row is not None else None


def raw_membership_retired_full_revision_siblings(
    store: RawRevisionSourceHost, logical_source_key: str
) -> tuple[str, ...]:
    """Return raws previously retired from full-revision byte governance for this key.

    ``replace_raw_membership_census(..., retire_full_revision_governance=True)``
    nulls the retired raw's ``raw_sessions.logical_source_key`` and sets
    ``revision_authority='quarantined'`` -- it becomes invisible both to
    ``prepare_raw_revision_byte_classification``'s own byte-row query and to
    ``raw_membership_rebuild_raw_ids``'s deliberately byte-proven-only
    filter (polylogue-lkrc/#2822 guards a different hazard: reopening a
    quarantined member against an already-established head). Its
    ``raw_session_memberships`` row, keyed by the raw's own parsed
    logical identity, survives retirement together with a
    ``raw_membership_census.detail`` marker naming this specific
    transition, so a later-arriving sibling for the same identity can
    still be told this identity has known, unresolved ambiguous
    evidence (polylogue-52l2) instead of being evaluated alone.

    Matches the typed quarantine authority; ``detail`` is display evidence.
    """
    rows = (
        store._ensure_source_conn()
        .execute(_RETIRED_MEMBERSHIP_SIBLINGS_SQL, _retired_membership_siblings_parameters(logical_source_key))
        .fetchall()
    )
    return tuple(str(row[0]) for row in rows)


# A retired member is proven by its census's typed quarantine authority, or,
# when the census code has no typed translation (NULL), by the retired raw
# row itself: retirement leaves it keyless, of unknown kind and quarantined.
# An unknown census code never reads as "not retired".
_RETIRED_MEMBERSHIP_SIBLINGS_SQL = """
    SELECT m.raw_id
    FROM raw_session_memberships AS m
    JOIN raw_membership_census AS c ON c.raw_id = m.raw_id
    JOIN raw_sessions AS r ON r.raw_id = m.raw_id
    WHERE m.logical_source_key = ?
      AND (
          c.revision_authority = ?
          OR (
              c.revision_authority IS NULL
              AND r.logical_source_key IS NULL
              AND r.revision_kind = 'unknown'
              AND r.revision_authority = ?
          )
      )
    ORDER BY m.raw_id
"""


def _retired_membership_siblings_parameters(logical_source_key: str) -> tuple[str, str, str]:
    quarantined = RawRevisionAuthority.QUARANTINED.value
    return (logical_source_key, quarantined, quarantined)


def _raw_revision_authority(store: RawRevisionGovernanceHost, raw_id: str) -> str | None:
    row = (
        store._ensure_source_conn()
        .execute("SELECT revision_authority FROM raw_sessions WHERE raw_id = ?", (raw_id,))
        .fetchone()
    )
    return None if row is None or row[0] is None else str(row[0])


def raw_revision_replay_plan(store: RawRevisionSourceHost, logical_source_key: str) -> RevisionReplayPlan:
    return plan_revision_replay(_raw_revision_candidates(store, logical_source_key))


def _raw_revision_candidates(store: RawRevisionSourceHost, logical_source_key: str) -> list[RevisionCandidate]:
    rows = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT raw_id, revision_kind, source_revision, acquisition_generation,
               revision_authority, blob_size, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, predecessor_source_revision
        FROM raw_sessions
        WHERE logical_source_key = ? AND source_revision IS NOT NULL
        """,
            (logical_source_key,),
        )
        .fetchall()
    )
    return [
        RevisionCandidate(
            raw_id=str(row[0]),
            logical_source_key=logical_source_key,
            kind=RawRevisionKind(str(row[1])),
            source_revision=str(row[2]),
            acquisition_generation=int(row[3]),
            authority=RawRevisionAuthority(str(row[4])),
            blob_size=int(row[5]),
            predecessor_source_revision=str(row[10]) if row[10] is not None else None,
            predecessor_raw_id=str(row[6]) if row[6] is not None else None,
            baseline_raw_id=str(row[7]) if row[7] is not None else None,
            append_start_offset=int(row[8]) if row[8] is not None else None,
            append_end_offset=int(row[9]) if row[9] is not None else None,
        )
        for row in rows
    ]


def _authorize_full_snapshot_fold(
    store: RetainedRevisionBytesRead,
    *,
    existing_head: tuple[object, ...],
    full_candidate: RevisionCandidate,
    candidates: Mapping[str, RevisionCandidate],
) -> FullSnapshotFoldAuthorization | None:
    """Prove one full raw is exactly the accepted byte-append chain.

    The caller supplies the actual selected raw byte reader and head inputs;
    failure intentionally yields no authority and leaves ordinary CAS
    semantics in force.  Every byte, offset, source revision, and raw
    predecessor edge is checked instead of trusting parser-normalized
    content hashes, which are segmentation-sensitive for Codex JSONL.
    """
    if (
        full_candidate.kind is not RawRevisionKind.FULL
        or full_candidate.authority is not RawRevisionAuthority.BYTE_PROVEN
        or str(existing_head[4]) != "byte"
        or str(existing_head[1]) not in candidates
    ):
        return None
    accepted_head = candidates[str(existing_head[1])]
    frontier = int(cast(int | str | bytes, existing_head[5]))
    if (
        accepted_head.kind is not RawRevisionKind.APPEND
        or accepted_head.authority is not RawRevisionAuthority.BYTE_PROVEN
        or accepted_head.source_revision != str(existing_head[2])
        or accepted_head.append_end_offset != frontier
    ):
        return None
    _full_digest, full_size = _raw_revision_payload_digest_and_size(store, full_candidate.raw_id)
    if full_size != frontier:
        return None

    tail_raw_ids: list[str] = []
    current = accepted_head
    baseline_raw_id = current.baseline_raw_id
    expected_end = frontier
    visited: set[str] = set()
    while current.kind is RawRevisionKind.APPEND:
        check_compute_cancelled()
        if (
            current.raw_id in visited
            or current.authority is not RawRevisionAuthority.BYTE_PROVEN
            or current.baseline_raw_id != baseline_raw_id
            or current.predecessor_raw_id is None
            or current.predecessor_source_revision is None
            or current.append_start_offset is None
            or current.append_end_offset != expected_end
        ):
            return None
        visited.add(current.raw_id)
        assert current.append_end_offset is not None
        assert current.append_start_offset is not None
        tail_start = _append_payload_start_offset(store, current)
        if tail_start is None:
            return None
        tail_digest, tail_size = _raw_revision_payload_digest_and_size(
            store,
            current.raw_id,
            start_offset=tail_start,
        )
        if tail_size != current.append_end_offset - current.append_start_offset:
            return None
        predecessor = candidates.get(current.predecessor_raw_id)
        if (
            predecessor is None
            or predecessor.source_revision != current.predecessor_source_revision
            or current.source_revision != append_source_revision(predecessor.source_revision, tail_digest)
        ):
            return None
        predecessor_end = (
            predecessor.blob_size if predecessor.kind is RawRevisionKind.FULL else predecessor.append_end_offset
        )
        if predecessor_end != current.append_start_offset:
            return None
        tail_raw_ids.append(current.raw_id)
        expected_end = current.append_start_offset
        current = predecessor
    if (
        current.kind is not RawRevisionKind.FULL
        or current.authority is not RawRevisionAuthority.BYTE_PROVEN
        or current.raw_id != baseline_raw_id
        or current.blob_size != expected_end
    ):
        return None
    _baseline_digest, baseline_size = _raw_revision_payload_digest_and_size(store, current.raw_id)
    if baseline_size != current.blob_size or not _raw_revision_matches_segments(
        store,
        full_candidate.raw_id,
        [current, *(candidates[raw_id] for raw_id in reversed(tail_raw_ids))],
    ):
        return None
    return FullSnapshotFoldAuthorization(
        logical_source_key=full_candidate.logical_source_key,
        session_id=str(existing_head[0]),
        accepted_append_raw_id=str(existing_head[1]),
        accepted_append_source_revision=str(existing_head[2]),
        accepted_append_content_hash=cast(bytes, existing_head[3]),
        frontier=frontier,
        full_raw_id=full_candidate.raw_id,
        full_source_revision=full_candidate.source_revision,
    )


def raw_revision_descriptor(
    store: RawRevisionGovernanceHost, raw_id: str
) -> tuple[Provider, str, str, RawRevisionKind, int]:
    """Return one retained revision's identity without materializing its blob."""
    row = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT origin, detected_provider, capture_mode, lower(hex(blob_hash)), source_path, revision_kind, blob_size
        FROM raw_sessions WHERE raw_id = ?
        """,
            (raw_id,),
        )
        .fetchone()
    )
    if row is None:
        raise KeyError(raw_id)
    return (
        (
            Provider.from_string(str(row[1]))
            if row[1] is not None
            else provider_from_origin(Origin.from_string(str(row[0])), family_hint=row[2])
        ),
        str(row[3]),
        str(row[4]),
        RawRevisionKind(str(row[5])),
        int(row[6]),
    )


def raw_native_id(store: RawRevisionGovernanceHost, raw_id: str) -> str | None:
    """Return the identity hint recorded for a raw revision, if any.

    polylogue-u19l: populated for Codex append-mode captures whose own
    record stream carries no ``session_meta`` of its own (an append delta),
    so replay can recover the provider session identity without it having
    been spliced into the hashed/stored payload bytes (see
    ``write_raw_payload``'s ``native_id`` and ``sources/live/batch.py``'s
    ``_append_payload_for_provider``). ``None`` for every raw row that
    doesn't carry one, including all historical rows written before this.
    """
    row = (
        store._ensure_source_conn().execute("SELECT native_id FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    )
    if row is None:
        return None
    value = row[0]
    return value if isinstance(value, str) and value.strip() else None


@contextmanager
def open_raw_revision_material(
    store: RawRevisionGovernanceHost, raw_id: str
) -> Iterator[tuple[Provider, BinaryIO, str, RawRevisionKind]]:
    """Open a retained revision for bounded streaming consumption."""
    provider, blob_hash, source_path, kind, _blob_size = raw_revision_descriptor(store, raw_id)
    payload_store = _retained_blob_store(store)
    with payload_store.open(blob_hash) as payload:
        yield provider, payload, source_path, kind


@contextmanager
def open_raw_container_material(
    store: RawRevisionGovernanceHost, captured_zip_coordinate: CapturedZipMemberCoordinate
) -> Iterator[BinaryIO | None]:
    """Open the accepted physical container of one ZIP member raw, if retained."""
    payload_store = _retained_blob_store(store)
    if not payload_store.exists(captured_zip_coordinate.container_blob_hash):
        yield None
        return
    with payload_store.open(captured_zip_coordinate.container_blob_hash) as container:
        yield container


def raw_revision_material(
    store: RawRevisionGovernanceHost, raw_id: str
) -> tuple[Provider, bytes, str, RawRevisionKind]:
    """Read one retained revision with its parsing identity.

    Use ``open_raw_revision_material`` for potentially large blobs.
    """
    provider, blob_hash, source_path, kind, _blob_size = raw_revision_descriptor(store, raw_id)
    payload_store = _retained_blob_store(store)
    return provider, payload_store.read_all(blob_hash), source_path, kind


def _retained_blob_store(store: RawRevisionSourceHost) -> BlobStore:
    return store._blob_publisher or BlobStore(store.archive_root / "blob")


def blob_path_for_hash(store: RawRevisionGovernanceHost, blob_hash: str) -> Path | None:
    """Return the real on-disk path for a content-addressed blob, if materialized.

    Some payload shapes (Hermes state.db/verification_evidence.db) need a
    real filesystem path -- ``sqlite3.connect`` cannot open in-memory
    bytes. Returns ``None`` when the blob is not (yet) materialized on
    disk so callers fall back to a bounded temp-file spill instead of
    trusting an unverified path.
    """
    path = _retained_blob_store(store).blob_path(blob_hash)
    return path if path.exists() else None


def _raw_revision_payload_digest_and_size(
    store: RetainedRevisionBytesRead,
    raw_id: str,
    *,
    start_offset: int = 0,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with store.open_raw_revision_material(raw_id) as (_provider, payload, _source_path, _kind):
        payload.seek(start_offset)
        while chunk := payload.read(1024 * 1024):
            check_compute_cancelled()
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _append_payload_start_offset(store: RetainedRevisionBytesRead, candidate: RevisionCandidate) -> int | None:
    """Return the retained-payload offset for one byte-governed append."""
    if candidate.append_start_offset is None or candidate.append_end_offset is None:
        return None
    expected_size = candidate.append_end_offset - candidate.append_start_offset
    if candidate.blob_size == expected_size:
        return 0
    if candidate.blob_size < expected_size:
        return None
    with store.open_raw_revision_material(candidate.raw_id) as (provider, payload, _source_path, _kind):
        if provider is not Provider.CODEX:
            return None
        header_size = candidate.blob_size - expected_size
        header = payload.read(header_size)
    if len(header) != header_size or strip_codex_legacy_append_header(header + b"\0") != b"\0":
        return None
    return header_size


def _raw_revision_matches_segments(
    store: RetainedRevisionBytesRead, full_raw_id: str, segments: Sequence[RevisionCandidate | str]
) -> bool:
    with store.open_raw_revision_material(full_raw_id) as (_provider, full, _source_path, _kind):
        for segment in segments:
            check_compute_cancelled()
            start_offset: int | None
            if isinstance(segment, str):
                raw_id = segment
                start_offset = 0
            else:
                raw_id = segment.raw_id
                start_offset = (
                    _append_payload_start_offset(store, segment) if segment.kind is RawRevisionKind.APPEND else 0
                )
            if start_offset is None:
                return False
            with store.open_raw_revision_material(raw_id) as (_provider, payload, _source_path, _kind):
                payload.seek(start_offset)
                while chunk := payload.read(1024 * 1024):
                    check_compute_cancelled()
                    if full.read(len(chunk)) != chunk:
                        return False
        return full.read(1) == b""


def pending_raw_revision_logical_keys(store: RawRevisionGovernanceHost) -> tuple[str, ...]:
    rows = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT DISTINCT logical_source_key
        FROM raw_sessions
        WHERE logical_source_key IS NOT NULL AND parsed_at_ms IS NULL
        ORDER BY logical_source_key
        """
        )
        .fetchall()
    )
    return tuple(str(row[0]) for row in rows)


def raw_revision_rebuild_logical_keys(
    store: RawRevisionGovernanceHost,
    raw_ids: list[str] | None,
) -> tuple[str, ...]:
    """Expand requested raws only to complete same-source-path cohorts."""
    conn = store._ensure_source_conn()
    if raw_ids is None:
        return tuple(
            str(row[0])
            for row in conn.execute(
                """
                SELECT DISTINCT logical_source_key FROM raw_sessions
                WHERE logical_source_key IS NOT NULL ORDER BY logical_source_key
                """
            )
        )
    selected = tuple(dict.fromkeys(raw_ids))
    if not selected:
        return ()
    placeholders = ",".join("?" for _ in selected)
    source_paths = tuple(
        str(row[0])
        for row in conn.execute(
            f"SELECT DISTINCT source_path FROM raw_sessions WHERE raw_id IN ({placeholders})",
            selected,
        )
    )
    if not source_paths:
        return ()
    path_placeholders = ",".join("?" for _ in source_paths)
    return tuple(
        str(row[0])
        for row in conn.execute(
            f"""
            SELECT DISTINCT logical_source_key FROM raw_sessions
            WHERE source_path IN ({path_placeholders})
              AND logical_source_key IS NOT NULL
            ORDER BY logical_source_key
            """,
            source_paths,
        )
    )


def raw_membership_census_rows(
    store: RawRevisionGovernanceHost, raw_ids: Sequence[str] | None = None
) -> tuple[tuple[str, int, bool, int], ...]:
    """Return retained raws and whether durable evidence says they are non-sessions."""
    conn = store._ensure_source_conn()
    columns = """
        r.raw_id,
        r.source_index,
        (
            EXISTS(SELECT 1 FROM raw_artifacts AS a WHERE a.raw_id = r.raw_id AND a.parse_as_session = 0)
            OR EXISTS(
                SELECT 1 FROM raw_membership_census AS c
                WHERE c.raw_id = r.raw_id
                  AND c.parser_fingerprint = ?
                  AND c.status = 'non_session'
                  AND r.parsed_at_ms IS NOT NULL
                  AND r.parse_error IS NULL
            )
        ),
        r.rowid
    """
    if raw_ids is None:
        rows = conn.execute(
            f"SELECT {columns} FROM raw_sessions AS r ORDER BY r.raw_id", (raw_authority_parser_fingerprint(),)
        ).fetchall()
    elif raw_ids:
        placeholders = ",".join("?" for _ in raw_ids)
        rows = conn.execute(
            f"SELECT {columns} FROM raw_sessions AS r WHERE r.raw_id IN ({placeholders}) ORDER BY r.raw_id",
            (raw_authority_parser_fingerprint(), *raw_ids),
        ).fetchall()
    else:
        rows = []
    return tuple((str(row[0]), int(row[1]), bool(row[2]), int(row[3])) for row in rows)


def raw_payload_sizes(store: RawRevisionGovernanceHost, raw_ids: Sequence[str]) -> dict[str, int]:
    if not raw_ids:
        return {}
    placeholders = ",".join("?" for _ in raw_ids)
    rows = store._ensure_source_conn().execute(
        f"SELECT raw_id, blob_size FROM raw_sessions WHERE raw_id IN ({placeholders})",
        tuple(raw_ids),
    )
    return {str(row[0]): int(row[1] or 0) for row in rows}


_RAW_BYTE_REVISION_DEPENDENTS_FILTER_SQL = """WHERE raw_id != ?
  AND (predecessor_raw_id = ? OR baseline_raw_id = ?)
"""

RAW_BYTE_REVISION_DEPENDENTS_SQL = (
    "\nSELECT 1 FROM raw_sessions\n" + _RAW_BYTE_REVISION_DEPENDENTS_FILTER_SQL + "LIMIT 1\n"
)


def has_raw_byte_revision_dependents(conn: sqlite3.Connection, raw_id: str) -> bool:
    """Read dependency authority from the caller's exact Source snapshot."""
    cursor = conn.execute(RAW_BYTE_REVISION_DEPENDENTS_SQL, (raw_id, raw_id, raw_id))
    try:
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def replace_raw_membership_census(
    seal: PreparedIndexMutation,
    raw_id: str,
    sessions: Sequence[ParsedSession] | None,
    *,
    parser_fingerprint: str,
    censused_at_ms: int,
    detail: str = "",
    revision_authority: RawRevisionAuthority | None,
    retire_full_revision_governance: bool = False,
    projections: Sequence[SessionRevisionProjection] | None = None,
) -> None:
    """Prepare complete membership replacement on the original Source tape.

    The parent owns the original read window and selected Source phase;
    publication commits the retained effects on the actual Source writer.
    """
    if projections is not None and (sessions is None or len(projections) != len(sessions)):
        raise ValueError("prepared membership projections must align with the complete parser census")
    check_compute_cancelled()
    _load_parser_census_source_inputs(seal, raw_id)
    raw_key = seal.retain_literal_scalar(raw_id)
    raw_expression, raw_parameters = seal.source_literal_expression(raw_key)
    if retire_full_revision_governance:
        with seal.source_rows(
            "SELECT logical_source_key, revision_kind FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ) as census_rows:
            revision = census_rows.fetchone()
        if revision is None:
            raise RuntimeError(f"membership census raw is missing: {raw_id}")
        if _prepared_raw_has_byte_revision_dependents(seal, raw_id):
            raise ActiveByteRevisionChainError("an active byte-revision chain cannot move to membership governance")
        # Authority is supplied by the producer; detail is display text.
        census_authority = revision_authority
        if sessions and census_authority is not RawRevisionAuthority.QUARANTINED:
            # A retirement that leaves membership rows behind is only observable
            # through its census authority: the retired raw loses its
            # ``logical_source_key`` and goes ``quarantined``, so
            # ``raw_membership_retired_full_revision_siblings`` and
            # the prepared same-path sibling guard find it by the
            # typed quarantined authority alone. An unrecognized marker with no
            # typed authority is not a harmless label -- it makes the retirement
            # invisible, and a later-arriving sibling for the
            # same identity is then accepted as an unconditional singleton
            # byte-proven baseline, which is exactly the polylogue-52l2 hazard the
            # marker exists to prevent. Refuse the write instead of letting an
            # unknown source value read back as success (polylogue-sze30 AC2).
            #
            # A census with no surviving membership row (a non-session artifact or
            # retained-state export) has no logical identity to be ambiguous
            # about, so its detail stays free explanatory prose.
            raise ValueError(
                "full-revision retirement with membership rows requires a recognized governance marker "
                "with quarantined revision authority"
            )
        _retire_prepared_full_revision_binding(seal, raw_key)
    with seal.source_statement(
        f"DELETE FROM raw_session_memberships WHERE raw_id = {raw_expression}",
        raw_parameters,
        table="raw_session_memberships",
        writable_targets=_prepared_membership_write_targets(seal, raw_id),
    ):
        pass
    if sessions is not None:
        for index, session in enumerate(sessions):
            projection = projections[index] if projections is not None else session_revision_projection(session)
            logical_key = canonical_authority_logical_key(f"{session.source_name.value}:{session.provider_session_id}")
            check_compute_cancelled()
            logical_key_cell = seal.retain_literal_scalar(logical_key)
            member_expressions, member_parameters = _prepared_source_operands(
                seal,
                raw_id,
                logical_key,
                session.provider_session_id,
                projection.session_hash.hex(),
                projection.session_hash,
                len(projection.message_hashes),
            )
            seal.source_allocation_dependencies("raw_session_memberships")
            with seal.source_statement(
                f"""
                INSERT INTO raw_session_memberships (
                    rowid, raw_id, logical_source_key, provider_session_id,
                    source_revision, normalized_content_hash, message_count
                ) VALUES (?, {member_expressions[0]}, {member_expressions[1]}, {member_expressions[2]},
                          {member_expressions[3]}, {member_expressions[4]}, {member_expressions[5]})
                """,
                (None, *member_parameters),
                table="raw_session_memberships",
                writable_targets=(("raw_session_memberships", (raw_key, logical_key_cell)),),
                prepared_cells={"raw_id": raw_key, "logical_source_key": logical_key_cell},
                allocation_parameter=0,
            ):
                pass
    status: Literal["failed", "non_session", "complete"] = (
        "failed" if sessions is None else ("non_session" if not sessions else "complete")
    )
    record_prepared_membership_census_receipt(
        seal,
        raw_id,
        parser_fingerprint=parser_fingerprint,
        status=status,
        member_count=len(sessions or []),
        censused_at_ms=censused_at_ms,
        detail=detail,
        revision_authority=revision_authority,
    )
    record_current_parser_source_census(seal, raw_id, parser_sessions=sessions)


def refine_prepared_raw_origin(seal: PreparedIndexMutation, raw_id: str, origin: Origin) -> None:
    """Stage the parsed origin over acquisition's ``unknown-export`` placeholder.

    A file acquired before detection (a browser capture) carries the
    placeholder until its census parses it. The parsed session names the
    origin; the row converges to it, as ``refine_raw_origin`` does for a
    better-informed re-acquisition. Guarded on the placeholder, so a
    confident origin is never overwritten.
    """
    if origin is Origin.UNKNOWN_EXPORT:
        return
    check_compute_cancelled()
    _load_parser_census_source_inputs(seal, raw_id)
    with seal.source_rows("SELECT origin FROM raw_sessions WHERE raw_id = ?", (raw_id,)) as rows:
        row = rows.fetchone()
    if row is None:
        raise RuntimeError(f"origin refinement names an absent raw: {raw_id}")
    if str(row[0]) != Origin.UNKNOWN_EXPORT.value:
        return
    raw_key = seal.retain_literal_scalar(raw_id)
    raw_expression, raw_parameters = seal.source_literal_expression(raw_key)
    origin_expression, origin_parameters = seal.source_literal_expression(seal.retain_literal_scalar(origin.value))
    placeholder_expression, placeholder_parameters = seal.source_literal_expression(
        seal.retain_literal_scalar(Origin.UNKNOWN_EXPORT.value)
    )
    with seal.source_statement(
        f"UPDATE raw_sessions SET origin = {origin_expression} "
        f"WHERE raw_id = {raw_expression} AND origin = {placeholder_expression}",
        (*origin_parameters, *raw_parameters, *placeholder_parameters),
        table="raw_sessions",
        writable_targets=(("raw_sessions", (raw_key,)),),
    ):
        pass


def _retire_prepared_full_revision_binding(seal: PreparedIndexMutation, raw_key: KnownTierCell) -> None:
    """Stage the one shape of a full revision leaving byte governance."""
    raw_expression, raw_parameters = seal.source_literal_expression(raw_key)
    with seal.source_statement(
        f"""
        UPDATE raw_sessions
        SET logical_source_key = NULL,
            revision_kind = 'unknown',
            source_revision = NULL,
            predecessor_raw_id = NULL,
            baseline_raw_id = NULL,
            append_start_offset = NULL,
            append_end_offset = NULL,
            acquisition_generation = NULL,
            revision_authority = 'quarantined',
            predecessor_source_revision = NULL
        WHERE raw_id = {raw_expression}
        """,
        raw_parameters,
        table="raw_sessions",
        writable_targets=(("raw_sessions", (raw_key,)),),
    ):
        pass


def _release_membership_decided_byte_binding(seal: PreparedIndexMutation, raw_id: str, logical_source_key: str) -> bool:
    """Retire a full revision's byte binding once membership decides it.

    Conversion keeps a full's byte binding beside its census until the
    membership classifier rules (``prepare_revision_source_membership_conversion``).
    A terminal decision for the same identity hands the raw to membership
    governance; keeping the binding would rebuild it through both authorities.
    A full that still carries byte-append dependents keeps its chain.
    """
    check_compute_cancelled()
    _load_parser_census_source_inputs(seal, raw_id)
    with seal.source_rows(
        "SELECT logical_source_key, revision_kind FROM raw_sessions WHERE raw_id = ?", (raw_id,)
    ) as rows:
        binding = rows.fetchone()
    if binding is None:
        raise RuntimeError(f"membership decision names an absent raw: {raw_id}")
    if binding[0] != logical_source_key or binding[1] != RawRevisionKind.FULL.value:
        return False
    if _prepared_raw_has_byte_revision_dependents(seal, raw_id):
        return False
    _retire_prepared_full_revision_binding(seal, seal.retain_literal_scalar(raw_id))
    return True


def record_prepared_membership_census_receipt(
    seal: PreparedIndexMutation,
    raw_id: str,
    *,
    parser_fingerprint: str,
    status: Literal["failed", "non_session", "complete"],
    member_count: int,
    censused_at_ms: int,
    detail: str,
    revision_authority: RawRevisionAuthority | None,
) -> None:
    """Retain the original census receipt without replacing its durable members."""
    check_compute_cancelled()
    _load_parser_census_source_inputs(seal, raw_id)
    raw_key = seal.retain_literal_scalar(raw_id)
    census_expressions, census_parameters = _prepared_source_operands(
        seal,
        raw_id,
        parser_fingerprint,
        status,
        member_count,
        censused_at_ms,
        detail,
        revision_authority,
    )
    seal.source_allocation_dependencies("raw_membership_census")
    with seal.source_statement(
        f"""
        INSERT INTO raw_membership_census (
            rowid, raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail, revision_authority
        ) VALUES (?, {census_expressions[0]}, {census_expressions[1]}, {census_expressions[2]},
                  {census_expressions[3]}, {census_expressions[4]}, {census_expressions[5]}, {census_expressions[6]})
        ON CONFLICT(raw_id) DO UPDATE SET
            parser_fingerprint=excluded.parser_fingerprint,
            status=excluded.status,
            member_count=excluded.member_count,
            censused_at_ms=excluded.censused_at_ms,
            detail=excluded.detail,
            revision_authority=excluded.revision_authority
        """,
        (None, *census_parameters),
        table="raw_membership_census",
        writable_targets=(("raw_membership_census", (raw_key,)),),
        prepared_cells={"raw_id": raw_key},
        allocation_parameter=0,
    ):
        pass


def record_current_parser_source_census(
    seal: PreparedIndexMutation,
    raw_id: str,
    *,
    parser_sessions: Sequence[ParsedSession] | None = None,
    inherited_logical_keys: Sequence[str] | None = None,
) -> None:
    """Prepare one current-parser receipt on the original Source witness.

    The caller owns the original read window and Source producer phase.
    Publication applies this same captured tape on its dedicated Source
    writer; this function neither opens a writer nor commits preparation.

    Ordinary admissions have a typed raw logical key; grouped imports instead
    establish their keys through ``raw_session_memberships``. The parsed
    identities must match that durable authority before this writer records a
    complete receipt for replay promotion. A byte-prefix revision may inherit
    its identity from the independently parsed head, which is recorded through
    ``inherited_logical_keys`` without parsing the already-proven prefix again.
    """
    check_compute_cancelled()
    _load_parser_census_source_inputs(seal, raw_id)
    if parser_sessions is not None and inherited_logical_keys is not None:
        raise ValueError("parser census cannot combine parsed and inherited identities")
    with seal.source_rows(
        """
        SELECT logical_source_key, revision_kind,
               EXISTS(SELECT 1 FROM raw_artifacts WHERE raw_id = raw_sessions.raw_id AND parse_as_session = 0),
               source_index
        FROM raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ) as census_rows:
        raw = census_rows.fetchone()
    if raw is None:
        raise RuntimeError(f"parser census raw is missing: {raw_id}")
    if parser_sessions is not None and len(parser_sessions) == 1:
        session = parser_sessions[0]
        singleton_producer = _PreparedSourceProducer(seal)
        singleton_key = canonical_authority_logical_key(f"{session.source_name.value}:{session.provider_session_id}")
        singleton_revision = _prepare_parser_singleton_revision(singleton_producer, raw_id, singleton_key)
        if singleton_revision is not None:
            singleton_witness = prepare_parser_singleton_witness(
                seal,
                singleton_revision,
                prepared_output=parser_sessions,
                parser_fingerprint=raw_authority_parser_fingerprint(),
            )
            _bind_parser_singleton_revision(
                singleton_producer,
                raw_id,
                singleton_key,
                witness=singleton_witness,
            )
        with seal.source_rows(
            """
            SELECT logical_source_key, revision_kind,
                   EXISTS(SELECT 1 FROM raw_artifacts WHERE raw_id = raw_sessions.raw_id AND parse_as_session = 0),
                   source_index
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id,),
        ) as census_rows:
            raw = census_rows.fetchone()
    with seal.source_rows(
        """
        SELECT status, revision_authority, detail FROM raw_membership_census
        WHERE raw_id = ? AND parser_fingerprint = ?
        """,
        (raw_id, raw_authority_parser_fingerprint()),
    ) as census_rows:
        membership_census = census_rows.fetchone()
    typed_non_session = bool(raw[2])
    # An append fragment has byte authority rather than semantic membership.
    # Its parser receipt therefore proves a complete, empty identity set;
    # recording failure would keep the component pending on every replay.
    byte_governed_fragment = (
        int(raw[3]) < 0
        and membership_census is not None
        and str(membership_census[0]) == "failed"
        and str(membership_census[1]) == RawRevisionAuthority.BYTE_PROVEN.value
    )
    parser_confirmed_non_session = membership_census is not None and str(membership_census[0]) == "non_session"
    iter_ids = getattr(parser_sessions, "iter_session_ids", None)
    if parser_sessions is not None:
        observed = (
            iter_ids()
            if callable(iter_ids)
            else (f"{session.source_name.value}:{session.provider_session_id}" for session in parser_sessions)
        )
    else:
        observed = inherited_logical_keys
    # Measurement consumes this indexed identity stream into its existing
    # disk owner. Close the selected reader before retaining literal chunks:
    # selected reads and descriptor writes share the Native witness.
    with ExitStack() as measurement_lifetime:
        with seal.source_rows(
            "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=? ORDER BY logical_source_key",
            (raw_id,),
        ) as memberships:
            measured = measurement_lifetime.enter_context(
                parser_census_identity_measurement(
                    raw_logical_key=raw[0],
                    revision_kind=raw[1],
                    membership_logical_keys=(row[0] for row in memberships),
                    observed_logical_keys=observed,
                    observed_are_receipt=inherited_logical_keys is not None,
                    inherit_durable_keys=(
                        observed is None
                        and (typed_non_session or parser_confirmed_non_session or byte_governed_fragment)
                    ),
                    check_stop=check_compute_cancelled,
                )
            )
        complete = measured.complete(
            typed_non_session=typed_non_session,
            parser_confirmed_non_session=parser_confirmed_non_session,
            byte_governed_fragment=byte_governed_fragment,
        )
        with measured.keys_json_stream(
            sqlite_encoding=parser_sessions is not None, check_stop=check_compute_cancelled
        ) as (byte_length, chunks):
            logical_keys_cell = seal.retain_literal_stream("text", byte_length, chunks)
        check_compute_cancelled()
    detail = (
        "parser-observed: append fragment governed by byte revision authority"
        if byte_governed_fragment and complete
        else "parser-observed: typed non-session admission established no parser identity"
        if typed_non_session and complete
        else "parser-observed: membership census established durable authority identity"
        if membership_census is not None and complete
        else "parser-observed: inherited from byte-proven revision head"
        if inherited_logical_keys is not None and complete
        else "parser-observed: parser identity matches durable authority bindings"
        if complete
        else (
            str(membership_census[2])
            if membership_census is not None and membership_census[2] is not None
            else "current parser produced no durable authority identity"
        )
    )
    logical_keys_expression, literal_parameters = seal.source_literal_expression(logical_keys_cell)
    raw_key = seal.retain_literal_scalar(raw_id)
    receipt_expressions, receipt_parameters = _prepared_source_operands(
        seal,
        raw_id,
        raw_authority_parser_fingerprint(),
        "complete" if complete else "failed",
    )
    detail_expression, detail_parameters = seal.source_literal_expression(seal.retain_literal_scalar(detail))
    seal.source_allocation_dependencies("raw_authority_parser_census")
    with seal.source_statement(
        f"""
        INSERT INTO raw_authority_parser_census (
            rowid, raw_id, parser_fingerprint, status, logical_keys_json, detail
        ) VALUES (?, {receipt_expressions[0]}, {receipt_expressions[1]}, {receipt_expressions[2]},
                  {logical_keys_expression}, {detail_expression})
        ON CONFLICT(raw_id) DO UPDATE SET
            parser_fingerprint = excluded.parser_fingerprint,
            status = excluded.status,
            logical_keys_json = excluded.logical_keys_json,
            detail = excluded.detail
        """,
        (None, *receipt_parameters, *literal_parameters, *detail_parameters),
        table="raw_authority_parser_census",
        writable_targets=(("raw_authority_parser_census", (raw_key,)),),
        prepared_cells={"raw_id": raw_key, "logical_keys_json": logical_keys_cell},
        allocation_parameter=0,
    ):
        pass
    check_compute_cancelled()


def convertible_full_revision_raw_ids(store: RawRevisionGovernanceHost, logical_source_key: str) -> tuple[str, ...]:
    """Return a full-only byte cohort that can join semantic membership."""
    rows = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT raw_id, revision_kind
        FROM raw_sessions
        WHERE logical_source_key = ?
        ORDER BY raw_id
        """,
            (logical_source_key,),
        )
        .fetchall()
    )
    if not rows or any(str(row[1]) != RawRevisionKind.FULL.value for row in rows):
        return ()
    return tuple(str(row[0]) for row in rows)


def pending_raw_envelope_has_membership_authority(conn: sqlite3.Connection, logical_source_key: str) -> bool:
    """Whether this pending-raw envelope is governed per session."""
    if not logical_source_key.startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX):
        return False
    with _governance_read_rows(conn, _PENDING_ENVELOPE_MEMBERSHIP_SQL, (logical_source_key,)) as rows:
        return rows.fetchone() is not None


def raw_has_membership_governed_pending_envelope(conn: sqlite3.Connection, raw_id: str) -> bool:
    """Whether this raw keeps a pending envelope beside its memberships."""
    with _governance_read_rows(
        conn,
        _RAW_PENDING_MEMBERSHIP_SQL,
        (raw_id, len(PENDING_RAW_LOGICAL_SOURCE_PREFIX), PENDING_RAW_LOGICAL_SOURCE_PREFIX),
    ) as rows:
        return rows.fetchone() is not None


def membership_key_has_pending_envelope_member(conn: sqlite3.Connection, logical_source_key: str) -> bool:
    """Whether this logical key includes a pending-envelope member."""
    with _governance_read_rows(
        conn,
        _MEMBERSHIP_PENDING_ENVELOPE_SQL,
        (logical_source_key, len(PENDING_RAW_LOGICAL_SOURCE_PREFIX), PENDING_RAW_LOGICAL_SOURCE_PREFIX),
    ) as rows:
        return rows.fetchone() is not None


def expand_raw_membership_selection(
    store: RawRevisionGovernanceHost, raw_ids: list[str] | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Expand scheduling hints to the complete transitive membership cohort."""
    return expand_raw_membership_selection_sync(store._ensure_source_conn(), raw_ids)


def raw_membership_selection_components_sync(
    conn: sqlite3.Connection,
    raw_ids: list[str],
) -> tuple[tuple[str, ...], ...]:
    """Partition scheduling hints with one bulk authority-graph snapshot.

    Re-expanding every direct candidate separately turns a large backlog
    into thousands of overlapping recursive SQL walks.  Source paths and
    logical membership keys are both undirected authority edges, so build
    their connected components once and project only components containing
    a scheduling hint.
    """
    hints = tuple(dict.fromkeys(raw_ids))
    if not hints:
        return ()

    parent: dict[str, str] = {}

    def find(raw_id: str) -> str:
        root = parent.setdefault(raw_id, raw_id)
        while root != parent[root]:
            parent[root] = parent[parent[root]]
            root = parent[root]
        while raw_id != root:
            next_raw_id = parent[raw_id]
            parent[raw_id] = root
            raw_id = next_raw_id
        return root

    def join(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_path: dict[str, str] = {}
    by_key: dict[str, str] = {}
    for raw_id, source_path, logical_source_key in conn.execute(
        "SELECT raw_id, source_path, logical_source_key FROM raw_sessions"
    ):
        raw_text = str(raw_id)
        find(raw_text)
        path = str(source_path or "")
        if path:
            prior = by_path.setdefault(path, raw_text)
            join(prior, raw_text)
        if logical_source_key is not None:
            key = str(logical_source_key)
            prior = by_key.setdefault(key, raw_text)
            join(prior, raw_text)
    for raw_id, logical_source_key in conn.execute("SELECT raw_id, logical_source_key FROM raw_session_memberships"):
        raw_text = str(raw_id)
        find(raw_text)
        key = str(logical_source_key)
        prior = by_key.setdefault(key, raw_text)
        join(prior, raw_text)

    members: dict[str, list[str]] = {}
    for raw_id in parent:
        members.setdefault(find(raw_id), []).append(raw_id)
    selected_roots = {find(raw_id) for raw_id in hints if raw_id in parent}
    components = [tuple(sorted(members[root])) for root in selected_roots]
    return tuple(sorted(components, key=lambda component: component[0]))


def raw_membership_selection_components(
    store: RawRevisionGovernanceHost, raw_ids: list[str]
) -> tuple[tuple[str, ...], ...]:
    return raw_membership_selection_components_sync(store._ensure_source_conn(), raw_ids)


_MEMBERSHIP_EXPANSION_BATCH = 400


def expand_raw_membership_selection_sync(
    conn: sqlite3.Connection,
    raw_ids: list[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Expand scheduling hints with the canonical path/membership predicates."""
    from polylogue.storage.sqlite.archive_tiers.write import ConnectionSessionSourceRead

    if raw_ids is None:
        with _governance_read_rows(conn, "SELECT raw_id FROM raw_sessions") as rows:
            selected = tuple(str(row[0]) for row in rows)
    else:
        selected = tuple(raw_ids)
    return _expand_raw_membership_selection(ConnectionSessionSourceRead(conn), selected)


def raw_membership_raw_ids(
    store: RawRevisionSourceHost,
    logical_source_key: str,
    *,
    include_complete_raw_ids: frozenset[str] = frozenset(),
    source_generation_id: str | None = None,
) -> tuple[str, ...]:
    """Return byte-proven candidates plus the complete raws being classified.

    A newly censused live snapshot has not received a membership decision
    yet, so it is deliberately quarantined until this classification
    completes. Admit only those caller-owned complete censuses alongside
    established byte-proven evidence; do not reopen unrelated quarantined
    members from a prior failed or ambiguous replay.

    The caller owns a whole request's accepted raws, so ownership is applied
    once per logical source key rather than once per accepted raw: a request
    accepting A raws across K keys costs K reads, not A*K. SQL still bounds
    the read to this one logical source key -- whose undecided candidates are
    bounded by the key itself -- and ownership is a Python set test, so no
    variable-size bind list and no re-encoded accepted set ever reaches
    SQLite. The read is always current: it is the mutable membership and
    census evidence that publication revalidation must observe moving.
    """
    rows = (
        store._ensure_source_conn()
        .execute(
            """
            SELECT m.raw_id, m.revision_authority,
                   EXISTS(SELECT 1 FROM source_item_raw_members sm
                          WHERE sm.raw_id=m.raw_id AND sm.source_generation_id=?) AS generation_owned
            FROM raw_session_memberships AS m
            LEFT JOIN raw_membership_census AS c ON c.raw_id = m.raw_id
            WHERE m.logical_source_key = ?
              AND (
                m.revision_authority = 'byte_proven'
                OR (c.status = 'complete' AND m.decision IS NULL)
              )
            ORDER BY m.raw_id
            """,
            (source_generation_id, logical_source_key),
        )
        .fetchall()
    )
    return tuple(
        str(row[0])
        for row in rows
        if row[1] == "byte_proven" or bool(row[2]) or str(row[0]) in include_complete_raw_ids
    )


def raw_revision_acquired_at_ms(store: RawRevisionGovernanceHost, raw_id: str) -> int:
    """Return the durable acquisition order for one retained raw revision."""
    row = (
        store._ensure_source_conn()
        .execute(
            "SELECT acquired_at_ms FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        )
        .fetchone()
    )
    if row is None:
        raise KeyError(f"unknown raw revision {raw_id}")
    return int(row[0])


def raw_revision_file_mtime(store: RawRevisionGovernanceHost, raw_id: str) -> str | None:
    """Return retained acquisition file mtime for replay fallback authority."""
    row = (
        store._ensure_source_conn()
        .execute(
            "SELECT file_mtime_ms FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        )
        .fetchone()
    )
    if row is None:
        raise KeyError(f"unknown raw revision {raw_id}")
    if row[0] is None:
        return None
    return datetime.fromtimestamp(int(row[0]) / 1000, UTC).isoformat()


def raw_revision_observed_at_ms(store: RawRevisionGovernanceHost, raw_id: str) -> int:
    """Return the latest durable observation receipt for a retained raw.

    ``raw_sessions.acquired_at_ms`` is deliberately immutable because the raw
    id is content-derived.  Re-observing identical bytes refreshes the
    ``blob_refs`` raw-payload receipt instead, which is the ordering authority
    for replaying mutable state snapshots.
    """
    return raw_revision_observation_order(store, raw_id)[0]


def raw_revision_observation_order(store: RawRevisionGovernanceHost, raw_id: str) -> tuple[int, int]:
    """Return the latest observation's timestamp and its durable receipt order.

    The latest observation is the newest ``raw_payload`` receipt by its
    monotonic insertion order (``rowid``), never by its wall-clock stamp: a
    clock rollback between two observations must not reorder them. Callers
    order by the second element; the timestamp is reported, not ranked. A raw
    with no receipt ranks oldest (order 0), as in ``raw_receipt_order_sql``:
    ``raw_sessions.rowid`` is another sequence and cannot be compared with a
    receipt's.
    """
    conn = store._ensure_source_conn()
    row = conn.execute(
        """
        SELECT acquired_at_ms, rowid
        FROM blob_refs
        WHERE ref_id = ? AND ref_type = 'raw_payload'
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (raw_id,),
    ).fetchone()
    if row is not None:
        return int(row[0]), int(row[1])
    row = conn.execute("SELECT acquired_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown raw revision {raw_id}")
    return int(row[0]), 0


def raw_membership_rebuild_raw_ids(store: RawRevisionGovernanceHost, logical_source_key: str) -> tuple[str, ...]:
    """Return census candidates excluding quarantined full rows with another authority key."""
    rows = (
        store._ensure_source_conn()
        .execute(
            """
            SELECT m.raw_id
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r ON r.raw_id = m.raw_id
            WHERE m.logical_source_key = ? AND r.revision_authority = 'byte_proven'
            ORDER BY m.raw_id
            """,
            (logical_source_key,),
        )
        .fetchall()
    )
    return tuple(str(row[0]) for row in rows)


def raw_revision_head_raw_id(store: RawRevisionGovernanceHost, logical_source_key: str) -> str | None:
    """Return the currently indexed accepted raw for one logical session."""
    row = store._conn.execute(
        "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
        (logical_source_key,),
    ).fetchone()
    return None if row is None else str(row[0])


def raw_membership_authority_complete(store: RawRevisionGovernanceHost, raw_id: str) -> bool:
    row = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT c.status = 'complete'
           AND NOT EXISTS (
               SELECT 1 FROM raw_session_memberships AS m
               WHERE m.raw_id = c.raw_id
                 AND (m.decision IS NULL OR m.decision IN ('ambiguous', 'deferred'))
           )
        FROM raw_membership_census AS c WHERE c.raw_id = ?
        """,
            (raw_id,),
        )
        .fetchone()
    )
    return row is not None and bool(row[0])


def raw_membership_decision_pending(store: RawRevisionGovernanceHost, raw_id: str) -> bool:
    """Return True only when this raw's own decision is genuinely undecided.

    ``raw_membership_authority_complete`` collapses three distinct
    non-complete states into one boolean: ``decision IS NULL`` (the
    raw-authority protocol's async classification -- the raw-materialization
    conveyor, see ``sources/revision_backfill.py`` -- has censused this raw
    but not yet arbitrated it), and ``decision IN ('ambiguous', 'deferred')``
    (arbitration already ran and concluded a genuine, byte-level conflict
    that requires new evidence to resolve, not the passage of time). Only
    the first is a legitimate hand-off; the second is a decided outcome
    that must surface as a failure (polylogue-emx2 vs. the fail-closed
    invariants pinned by #2684/#2716/#2718/#2837). This predicate isolates
    the NULL case so callers can distinguish "still pending" from "decided,
    unresolved" instead of treating both as non-failures.
    """
    row = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT c.status = 'complete'
           AND EXISTS (
               SELECT 1 FROM raw_session_memberships AS m
               WHERE m.raw_id = c.raw_id AND m.decision IS NULL
           )
        FROM raw_membership_census AS c WHERE c.raw_id = ?
        """,
            (raw_id,),
        )
        .fetchone()
    )
    return row is not None and bool(row[0])


def _carried_session_content_hash(session: ParsedSession) -> str:
    """The session's identity digest: the parse-bound one when carried.

    A prepared session's sink is lowered in place after parsing (active-path
    normalization, derived tool outcomes), the lowering a resident session
    receives only when its rows are built. Its bound digest names the parsed
    content, which is what every other route hashes; re-hashing the lowered
    sink gives a different digest for the same source bytes, and the writer
    then declines the worker's prepared rows as ``content_hash_mismatch``.
    """
    bound = bound_session_content_hash(session)
    return bound if bound is not None else session_content_hash(session)


def defer_raw_revision_adoption(
    store: RawRevisionGovernanceHost,
    prepared: PreparedRevisionAdoption,
) -> None:
    """Apply the original prepared deferral without rereading Source evidence."""
    from polylogue.storage.sqlite.reference_seal import current_index_mutation_scope

    scope = current_index_mutation_scope()
    if scope is None:
        raise RuntimeError("revision deferral requires its actual Index mutation scope")
    scope.require_connection(store._conn)
    if prepared.adoptable or prepared.session_id is None or not prepared.deferred_receipts:
        raise RuntimeError("revision deferral lacks its original prepared receipts")
    for receipt in prepared.deferred_receipts:
        if (
            receipt.session_id != prepared.session_id
            or receipt.decision is not ApplicationDecision.DEFERRED
            or receipt.accepted_raw_id is not None
            or receipt.accepted_source_revision is not None
            or receipt.accepted_content_hash is not None
        ):
            raise RuntimeError("revision deferral receipt does not preserve its original decision")
    decided_at_ms = int(time.time() * 1000)
    for receipt in prepared.deferred_receipts:
        check_compute_cancelled()
        record_revision_application_sync(store._conn, receipt, decided_at_ms=decided_at_ms)


def apply_raw_revision_replay(
    store: RawRevisionGovernanceHost,
    plan: RevisionReplayPlan,
    parsed_by_raw_id: dict[str, ParsedSession],
    *,
    prepared_outcome: PreparedRevisionReplayOutcome,
    acquired_at_ms: int,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "revision_replay",
    manage_transaction: bool = True,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    prepared_by_raw_id: dict[str, PreparedSessionRows] | None = None,
    prepared_required_raw_ids: frozenset[str] = frozenset(),
    preacquired_attachment_blobs_by_raw_id: Mapping[str, Mapping[object, tuple[bytes | None, int, str]]] | None = None,
    prepared_aggregate_session: ParsedSession | None = None,
    preacquired_aggregate_attachment_blobs: Mapping[object, tuple[bytes | None, int, str]] | None = None,
    prepared_aggregate_rows: PreparedSessionRows | None = None,
    prepared_write: PreparedSessionWrite | None = None,
    prepared_aggregate_content_hash: bytes | None = None,
    write_result: Callable[[ArchiveRawParsedWriteResult], None] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Apply a proven chain and atomically receipt its exact index state.

    ``prepared_by_raw_id`` supplies physically completed rows for a single
    accepted chunk. A composed chain uses its sealed aggregate rows and
    attachment carrier. Publication neither submits nor waits for preparation;
    changed preparation is a typed refusal.

    ``manage_transaction=False`` batches this cohort's index.db writes
    and terminal source.db parse-state markers into the caller's open
    transaction/pending-state instead of committing them immediately
    (polylogue-oikv). The caller must already own an
    ``IndexMutationScope`` and call
    ``commit()`` (or ``rollback()`` on failure) exactly once per batch, after
    every cohort in the batch has been applied. ``commit()`` always
    commits the index connection before flushing pending source markers
    (``_flush_pending_raw_parse_states``), so the "index commits, then
    source terminal markers commit" ordering invariant now holds at
    BATCH granularity instead of per-cohort: a crash anywhere before the
    shared ``commit()`` call discards the whole uncommitted batch (every
    batched cohort's index writes and terminal markers together), never
    a partial one, and a resume reprocesses every lost cohort from
    scratch with zero duplication.

    ``bulk_fts`` (polylogue-crd8, default ``False``) enables the
    guard-gated bulk FTS mode described on ``_bulk_fts_session_guard`` for
    whale prefix-sharing lineage cascades this replay triggers. Only the
    offline rebuild/backfill path passes ``True``; ordinary daemon replay
    stays on the unguarded per-row trigger path.

    ``bulk_build`` (polylogue-v6i3, default ``False``) is the broader
    bulk-generation-build lifecycle -- when ``True`` this skips the
    trailing ``repair_message_fts_index_sync`` per-session repopulate and
    relaxes ``assert_session_fts_exact_sync`` to its trigger-presence-only
    check, since the bulk-build caller repopulates ``messages_fts``
    archive-wide exactly once at readiness instead.
    """

    if not manage_transaction:
        from polylogue.storage.sqlite.reference_seal import current_index_mutation_scope

        scope = current_index_mutation_scope()
        if scope is None:
            raise RuntimeError("batched retained replay requires the caller's live Index mutation scope")
        scope.require_connection(store._conn)
    if not plan.accepted_raw_ids:
        raise ValueError("cannot apply a revision plan without an accepted chain")
    from polylogue.sources.dispatch import merge_parsed_session_chunks

    if prepared_outcome.plan != plan:
        raise PreparedSessionWriteRefusedError("byte replay outcome names another original plan")
    aggregate_sessions = (
        [prepared_aggregate_session]
        if prepared_aggregate_session is not None
        else (
            [parsed_by_raw_id[plan.accepted_raw_ids[0]]]
            if len(plan.accepted_raw_ids) == 1
            else merge_parsed_session_chunks(parsed_by_raw_id[raw_id] for raw_id in plan.accepted_raw_ids)
        )
    )
    if len(aggregate_sessions) != 1:
        raise RuntimeError("one logical revision chain did not compose to exactly one session")
    if prepared_aggregate_content_hash is None and prepared_aggregate_rows is not None:
        prepared_aggregate_content_hash = prepared_aggregate_rows.session_content_hash
    aggregate_content_hash = (
        prepared_aggregate_content_hash
        if prepared_aggregate_content_hash is not None
        else bytes.fromhex(_carried_session_content_hash(aggregate_sessions[0]))
    )
    if prepared_aggregate_content_hash is not None and len(prepared_aggregate_content_hash) != 32:
        raise PreparedSessionWriteRefusedError("prepared aggregate content hash is invalid")
    if preacquired_attachment_blobs_by_raw_id is None:
        raise PreparedSessionWriteRefusedError("revision replay requires sealed attachment preparation")
    if prepared_aggregate_session is not None and preacquired_aggregate_attachment_blobs is None:
        raise PreparedSessionWriteRefusedError("revision replay requires its original aggregate attachment view")
    for raw_id in plan.accepted_raw_ids:
        if raw_id not in preacquired_attachment_blobs_by_raw_id:
            raise PreparedSessionWriteRefusedError(f"revision replay lacks prepared attachments for {raw_id}")
    with store.index_mutation_scope() if manage_transaction else nullcontext():
        if prepared_outcome.suppressed:
            aggregate = aggregate_sessions[0]
            return str(make_session_id(aggregate.source_name, aggregate.provider_session_id)), ()
        retires_existing_head = prepared_outcome.retires_existing_head
        # Byte-governed evidence outranks a quarantined semantic head.
        # Retire that head only after the session writer actually writes;
        # suppression preserves both the prior session and its authority.
        pending_raw_ids = plan.accepted_raw_ids
        # One chain is one session, so the writer sees one session composed
        # by the same reduction that produced ``aggregate_content_hash``.
        # Writing each chunk separately made the persisted read model
        # disagree with that hash -- every chunk-local summary event
        # (Claude's ``claude_parse_coverage``) landed as its own row.
        composed_session = aggregate_sessions[0]
        # A composed aggregate consumes its exact carrier and enrolled
        # current-row view; otherwise the chain's own chunk claims compose.
        composed_attachment_blobs: Mapping[object, tuple[bytes | None, int, str]]
        if prepared_aggregate_session is not None:
            if preacquired_aggregate_attachment_blobs is None:
                raise PreparedSessionWriteRefusedError("prepared aggregate requires its original attachment claim view")
            composed_attachment_blobs = preacquired_aggregate_attachment_blobs
        else:
            composed_attachment_blobs = _PreparedAttachmentChain(
                preacquired_attachment_blobs_by_raw_id[raw_id] for raw_id in reversed(pending_raw_ids)
            )
        # The chain's newest accepted raw carries the composed write:
        # ``sessions.raw_id`` and the reparse receipt then name the tip
        # the head row is about to advertise.
        tip_raw_id = pending_raw_ids[-1]
        # ``prepared_aggregate_rows`` describes the exact composed session.
        # The raw-id map remains a single-chunk shortcut only.
        resolved_prepared: PreparedSessionRows | None = None
        if prepared_aggregate_rows is not None:
            resolved_prepared = prepared_aggregate_rows
        elif len(pending_raw_ids) == 1 and prepared_by_raw_id is not None:
            resolved_prepared = prepared_by_raw_id.get(tip_raw_id)
        index_started = time.perf_counter()
        result = _index_parsed_for_retained_raw(
            store,
            composed_session,
            raw_id=tip_raw_id,
            source_index=0,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=False,
            preacquired_attachment_blobs=composed_attachment_blobs,
            finalize_raw_parse=False,
            revision_authoritative=True,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            fresh_build=fresh_build,
            fresh_build_batch=fresh_build_batch,
            defer_fts_rebuild=not bulk_build,
            prepared=resolved_prepared,
            prepared_required=tip_raw_id in prepared_required_raw_ids or prepared_write is not None,
            prepared_write=prepared_write,
            content_hash=aggregate_content_hash.hex(),
        )
        if write_result is not None:
            write_result(result)
        if stage_timings_s is not None:
            key = f"{stage_timing_prefix}.index_parsed_write"
            stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - index_started)
        if result.publication_refused:
            return result.session_id, ()
        session_id = result.session_id
        if retires_existing_head:
            store._conn.execute(
                "DELETE FROM raw_revision_heads WHERE logical_source_key = ?",
                (plan.logical_source_key,),
            )
        store._conn.execute(
            "UPDATE sessions SET content_hash = ? WHERE session_id = ?",
            (aggregate_content_hash, session_id),
        )
        # The chain's chunks were each enriched; the composed aggregate is
        # bound only when every chunk read the same evidence.
        chain_keys = {
            parsed_by_raw_id[raw_id].enrichment_evidence_key
            for raw_id in plan.accepted_raw_ids
            if raw_id in parsed_by_raw_id
        }
        if len(chain_keys) == 1:
            if prepared_write is None:
                raise PreparedSessionWriteRefusedError("enrichment publication requires its original prepared write")
            _bind_retained_enrichment(
                store,
                aggregate_sessions[0].model_copy(update={"enrichment_evidence_key": next(iter(chain_keys))}),
                session_id=session_id,
                prepared_write=prepared_write,
            )
        if not bulk_build:
            repair_message_fts_index_sync(store._conn, [session_id])
        assert_session_fts_exact_sync(store._conn, session_id, bulk_build=bulk_build)
        stored = store._conn.execute("SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if stored is None or not isinstance(stored[0], bytes):
            raise RuntimeError("accepted revision did not produce a hashed session")
        decided_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        for receipt in prepared_outcome.application_receipts:
            if receipt.session_id != session_id or (
                receipt.accepted_content_hash is not None and receipt.accepted_content_hash != stored[0]
            ):
                raise PreparedSessionWriteRefusedError("byte replay wrote another prepared session revision")
            record_revision_application_sync(store._conn, receipt, decided_at_ms=decided_at_ms)
    return session_id, revision_replay_terminal_raw_ids(plan)


def _application_decision_for(decision: MembershipDecision) -> ApplicationDecision:
    """Translate a source-tier membership verdict into an index-tier replay decision.

    ``MembershipDecision`` (source.db, written by the membership-
    classification pipeline) and ``ApplicationDecision`` (index.db, written
    by replay) model the same underlying event -- a raw revision's fate
    within its logical-source cohort -- at two different tiers with two
    different, non-superset vocabularies; see both enums' docstrings and
    polylogue-z22ml for why they stay distinct rather than collapsing to
    one. This is the sole typed translation point between them, replacing a
    prior comment-only, string-``startswith``-based mapping with no
    compiler signal if either vocabulary changed.
    """
    if decision is MembershipDecision.AMBIGUOUS:
        return ApplicationDecision.AMBIGUOUS
    if decision in (
        MembershipDecision.SUPERSEDED_EQUIVALENT,
        MembershipDecision.SUPERSEDED_PREFIX,
        MembershipDecision.SUPERSEDED_BY_WINNER,
    ):
        return ApplicationDecision.SUPERSEDED
    return ApplicationDecision.SELECTED_BASELINE


def membership_decisions_for_classification(
    classification: MembershipClassification,
) -> dict[str, MembershipDecision]:
    """Return the source-tier decisions a fresh replay must persist."""
    decisions: dict[str, MembershipDecision] = dict.fromkeys(
        classification.ambiguous_raw_ids,
        MembershipDecision.AMBIGUOUS,
    )
    decisions.update(
        dict.fromkeys(
            classification.equivalent_raw_ids,
            MembershipDecision.SUPERSEDED_EQUIVALENT
            if classification.accepted_raw_ids
            else MembershipDecision.AMBIGUOUS,
        )
    )
    decisions.update(dict.fromkeys(classification.superseded_raw_ids, MembershipDecision.SUPERSEDED_BY_WINNER))
    for raw_id in classification.accepted_raw_ids[:-1]:
        decisions[raw_id] = MembershipDecision.SUPERSEDED_PREFIX
    if classification.accepted_raw_ids:
        decisions[classification.accepted_raw_ids[-1]] = MembershipDecision.APPLIED
    return decisions


def finalize_raw_parse_state(store: RawRevisionGovernanceHost, raw_id: str, *, state: RawSessionStateUpdate) -> None:
    """Commit one typed source parse state after its index outcome."""
    # A generic state update is also used by the retained-raw index route when
    # it has no typed worker disposition to persist.  Retire any prior
    # failure authority before recording that new untyped failure; otherwise
    # an old terminal/deferred carrier can continue to authorize replay after
    # the current attempt has failed for a different reason.
    if isinstance(state.parse_error, str) and state.parse_error:
        _retire_raw_failure_evidence(store, raw_id, manage_transaction=False)
    apply_source_raw_state_update(
        store._ensure_source_conn(),
        raw_id,
        state=state,
        manage_transaction=True,
    )


def mark_raw_parse_failed(
    store: RawRevisionGovernanceHost,
    raw_id: str,
    *,
    provider: Provider,
    error: BaseException,
    preserve_existing_failure_evidence: bool = False,
) -> None:
    """Persist a bounded parse/index failure for retained raw evidence."""
    conn = store._ensure_source_conn()
    with conn:
        if isinstance(error, (MissingProfileIdentityError, RetainedZipMembershipUnprovedError)):
            row = conn.execute(
                "SELECT source_path, source_index, acquired_at_ms FROM raw_sessions WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()
            if row is not None:
                _retire_raw_failure_evidence(store, raw_id, manage_transaction=False)
                record_raw_failure_evidence(
                    store,
                    raw_id,
                    provider=provider,
                    source_path=str(row[0] or raw_id),
                    source_index=int(row[1] or 0),
                    acquired_at_ms=int(row[2] or 0),
                    kind=(
                        RawFailureEvidenceKind.TERMINAL_RETAINED_ZIP_MEMBERSHIP_UNPROVED
                        if isinstance(error, RetainedZipMembershipUnprovedError)
                        else RawFailureEvidenceKind.TERMINAL_MISSING_PROFILE_IDENTITY
                    ),
                    manage_transaction=False,
                )
        elif isinstance(error, RawCASFrontierError) or _terminal_decode_kind(error, provider) is not None:
            # A CAS refusal is retryable authority evidence; a decode failure of
            # retained bytes is terminal (``terminal_decode_evidence``), the
            # same evidence the full route's census records.
            row = conn.execute(
                "SELECT source_path, source_index, acquired_at_ms FROM raw_sessions WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()
            if row is not None:
                _retire_raw_failure_evidence(store, raw_id, manage_transaction=False)
                record_raw_failure_evidence(
                    store,
                    raw_id,
                    provider=provider,
                    source_path=str(row[0] or raw_id),
                    source_index=int(row[1] or 0),
                    acquired_at_ms=int(row[2] or int(time.time() * 1000)),
                    kind=_terminal_decode_kind(error, provider) or RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER,
                    manage_transaction=False,
                )
        else:
            if not preserve_existing_failure_evidence:
                _supersede_deferred_cas_evidence(store, raw_id, provider=provider, manage_transaction=False)
                _retire_raw_failure_evidence(store, raw_id, manage_transaction=False)
        apply_source_raw_state_update(
            conn,
            raw_id,
            state=_raw_parse_failure_state(provider, error),
            manage_transaction=False,
        )


def _terminal_decode_kind(error: BaseException, provider: Provider) -> RawFailureEvidenceKind | None:
    from polylogue.sources.prepared_jsonl import terminal_decode_evidence

    return None if isinstance(error, RawCASFrontierError) else terminal_decode_evidence(error, provider=provider)


def record_raw_failure_evidence(
    store: RawRevisionGovernanceHost,
    raw_id: str,
    *,
    provider: Provider,
    source_path: str,
    source_index: int,
    acquired_at_ms: int,
    kind: RawFailureEvidenceKind,
    manage_transaction: bool = True,
) -> None:
    """Persist a closed parse-outcome classification beside retained bytes."""
    from polylogue.storage.sqlite.archive_tiers.source_write import _ConnectionSourceProducer

    _record_raw_failure_evidence(
        _ConnectionSourceProducer(store._ensure_source_conn()),
        raw_id,
        provider=provider,
        source_path=source_path,
        source_index=source_index,
        acquired_at_ms=acquired_at_ms,
        kind=kind,
        manage_transaction=manage_transaction,
    )


def _supersede_deferred_cas_evidence(
    store: RawRevisionGovernanceHost,
    raw_id: str,
    *,
    provider: Provider,
    manage_transaction: bool = True,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.source_write import _ConnectionSourceProducer

    _supersede_deferred_cas_with_producer(
        _ConnectionSourceProducer(store._ensure_source_conn()),
        raw_id,
        provider=provider,
        manage_transaction=manage_transaction,
    )


def _retire_raw_failure_evidence(
    store: RawRevisionGovernanceHost,
    raw_id: str,
    *,
    manage_transaction: bool = True,
) -> None:
    """Retire stale failure evidence before an untyped current failure."""
    from polylogue.storage.sqlite.archive_tiers.source_write import _ConnectionSourceProducer

    _retire_raw_failure_evidence_with_producer(
        _ConnectionSourceProducer(store._ensure_source_conn()),
        raw_id,
        manage_transaction=manage_transaction,
    )


def mark_raw_parse_succeeded(store: RawRevisionGovernanceHost, raw_id: str, *, provider: Provider) -> None:
    """Finalize one retained raw payload after every derived session commits."""
    conn = store._ensure_source_conn()
    with conn:
        _supersede_deferred_cas_evidence(store, raw_id, provider=provider, manage_transaction=False)
        apply_source_raw_state_update(
            conn,
            raw_id,
            state=_raw_parse_success_state(provider),
            manage_transaction=False,
        )


def _flush_pending_raw_parse_states(store: RawRevisionGovernanceHost) -> None:
    if not store._pending_raw_parse_states:
        return
    source_conn = store._ensure_source_conn()
    with source_conn:
        for raw_id, state in store._pending_raw_parse_states:
            provider = state.payload_provider
            if isinstance(provider, Provider) and isinstance(state.parsed_at, str) and state.parse_error is None:
                _supersede_deferred_cas_evidence(
                    store,
                    raw_id,
                    provider=provider,
                    manage_transaction=False,
                )
            apply_source_raw_state_update(
                source_conn,
                raw_id,
                state=state,
                manage_transaction=False,
            )
    store._pending_raw_parse_states.clear()


def _index_parsed_for_retained_raw(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    raw_id: str,
    source_index: int,
    stage_timings_s: dict[str, float] | None,
    stage_timing_prefix: str,
    manage_transaction: bool,
    preacquired_attachment_blobs: Mapping[object, tuple[bytes | None, int, str]],
    finalize_raw_parse: bool,
    revision_authoritative: bool = False,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    defer_fts_rebuild: bool = False,
    prepared: PreparedSessionRows | None = None,
    prepared_required: bool = False,
    prepared_write: PreparedSessionWrite | None = None,
    content_hash: str | None = None,
) -> ArchiveRawParsedWriteResult:
    provider = Provider.from_string(session.source_name)
    # Retained replay no longer has the parser's RawSessionData descriptor;
    # recover file_mtime from durable source evidence before the shared writer
    # computes freshness. This keeps replay equivalent to first acquisition.
    if prepared_write is None:
        raise PreparedSessionWriteRefusedError("retained publication requires its original prepared session write")
    session = normalize_session_timestamps(session, fallback_timestamp=prepared_write.fallback_timestamp)
    try:
        result = _write_parsed_precedence_result(
            store,
            session,
            raw_id=raw_id,
            source_index=source_index,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            preacquired_attachment_blobs=preacquired_attachment_blobs,
            revision_authoritative=revision_authoritative,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            fresh_build=fresh_build,
            fresh_build_batch=fresh_build_batch,
            defer_fts_rebuild=defer_fts_rebuild,
            prepared=prepared,
            prepared_required=prepared_required,
            prepared_write=prepared_write,
            content_hash=content_hash,
        )
    except PreparedSessionWriteRefusedError:
        # A required prepared carrier moving is a retryable admission refusal,
        # never parser-failure evidence for retained durable bytes.
        raise
    except Exception as exc:
        from polylogue.core.compute import DaemonBackpressureError
        from polylogue.core.sqlite_locking import is_transient_sqlite_lock
        from polylogue.core.storage_faults import storage_fault_kind

        if (
            isinstance(exc, DaemonBackpressureError)
            or is_transient_sqlite_lock(exc)
            or storage_fault_kind(exc) is not None
        ):
            # Compute admission pressure, writer contention, or a storage
            # fault says nothing about retained bytes: the raw stays pending
            # for retry instead of carrying a parse failure for valid input.
            raise
        if not _is_frozen_candidate(store):
            if isinstance(exc, RawCASFrontierError):
                # A retained-raw CAS refusal is retryable authority evidence.
                # Persist that carrier with the first source-tier failure
                # mutation so a crash cannot leave only the generic parse
                # diagnostic behind.
                mark_raw_parse_failed(store, raw_id, provider=provider, error=exc)
            else:
                finalize_raw_parse_state(store, raw_id, state=_raw_parse_failure_state(provider, exc))
        raise
    if finalize_raw_parse and not _is_frozen_candidate(store):
        success_state = _raw_parse_success_state(provider)
        if manage_transaction:
            finalize_raw_parse_state(store, raw_id, state=success_state)
        else:
            store._pending_raw_parse_states.append((raw_id, success_state))
    return result


def _raw_parse_success_state(provider: Provider) -> RawSessionStateUpdate:
    return RawSessionStateUpdate(
        parsed_at=datetime.now(UTC).isoformat(),
        parse_error=None,
        payload_provider=provider,
    )


def _raw_parse_failure_state(provider: Provider, exc: BaseException) -> RawSessionStateUpdate:
    error = f"{type(exc).__name__}: {exc}"
    return RawSessionStateUpdate(
        parse_error=error,
        payload_provider=provider,
        detection_warnings=error,
    )


def admit_work_event_raw(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    payload: bytes,
    raw_id: str,
    acquired_at_ms: int,
) -> str:
    """Retain one agent work event for its original prepared replay owner.

    The event raw is its own logical source: its raw id is both its logical
    key and its source path, and it is admitted as a singleton full baseline
    with the byte authority that frozen classification re-derives for it. It
    therefore never joins the byte-revision cohort, source-path cohort, or
    accepted head of the transcript it annotates. Its parser census inherits
    that key, because the parsed event names its session rather than itself.
    """
    if not is_work_event_raw_id(raw_id):
        raise ValueError(f"work event raw id must carry the work-event prefix: {raw_id!r}")
    if store._blob_publisher is None:
        raise RuntimeError("raw archive writes require a writable archive publisher")
    raw_hash, _raw_size = store._blob_publisher.write_from_bytes(payload)
    blob_publication_receipt_id = store._blob_publisher.receipt_id(raw_hash)
    store._blob_publisher.flush()
    source_conn = store._ensure_source_conn()
    admission = admit_raw_observation(
        source_conn,
        origin=origin_from_provider(session.source_name),
        capture_mode=session.source_name,
        source_path=raw_id,
        # A work event has no file origin: its declared ``work-event:``
        # coordinate is both its path and its canonical path, so it never
        # matches a file selection and never leaves the canonical path NULL.
        canonical_source_path=raw_id,
        source_index=-1,
        payload=payload,
        acquired_at_ms=acquired_at_ms,
        native_id=session.provider_session_id,
        raw_id=raw_id,
        logical_source_key=raw_id,
        baseline_revision=RawRevisionEnvelope(
            logical_source_key=raw_id,
            kind=RawRevisionKind.FULL,
            source_revision=raw_hash,
            acquisition_generation=0,
            baseline_raw_id=raw_id,
            authority=RawRevisionAuthority.BYTE_PROVEN,
        ),
        prior_head=None,
        blob_publication_receipt_id=blob_publication_receipt_id,
        manage_transaction=True,
        policy_snapshot=_policy_snapshot_for_store(store),
    )
    if admission.arm is not RawAdmissionArm.BASELINE or admission.raw_id != raw_id:
        raise RuntimeError(f"unexpected work event raw admission: {admission.arm!r} {admission.raw_id!r}")
    return raw_id


def prepare_accepted_head_reparse_receipt(
    index: sqlite3.Connection,
    source_read: SessionSourceRead,
    *,
    raw_id: str,
    session_id: str,
    content_hash: bytes,
    before_input: BeforeIndexInput | None,
) -> RevisionApplicationReceipt | None:
    """Capture the canonical correction of this raw's own accepted head."""
    columns = (
        "accepted_raw_id",
        "accepted_source_revision",
        "accepted_content_hash",
        "accepted_frontier_kind",
        "accepted_frontier",
        "acquisition_generation",
        "append_end_offset",
        "logical_source_key",
    )
    if before_input is not None:
        before_input(
            "raw_revision_heads",
            columns,
            "SELECT rowid FROM raw_revision_heads WHERE session_id=? AND accepted_raw_id=?",
            (session_id, raw_id),
        )
    with _governance_read_rows(
        index,
        f"SELECT {', '.join(columns)} FROM raw_revision_heads WHERE session_id=? AND accepted_raw_id=?",
        (session_id, raw_id),
    ) as rows:
        head = rows.fetchone()
    if head is None or head[2] is not None and bytes(head[2]) == content_hash:
        return None
    lineage = source_read.raw_revision_lineage(raw_id)
    if lineage is None:
        raise RuntimeError(f"accepted-head reparse source evidence is missing for {raw_id}")
    return RevisionApplicationReceipt(
        raw_id=raw_id,
        session_id=session_id,
        logical_source_key=str(head[7]),
        source_revision=str(head[1]),
        acquisition_generation=int(head[5]),
        decision=ApplicationDecision.REPARSE_REAFFIRMATION,
        accepted_raw_id=raw_id,
        accepted_source_revision=str(head[1]),
        accepted_content_hash=content_hash,
        accepted_frontier_kind=str(head[3]),
        accepted_frontier=int(head[4]),
        baseline_raw_id=lineage[0],
        predecessor_raw_id=lineage[1],
        append_end_offset=head[6],
        detail="reparse:accepted_head_content_correction",
    )


def record_prepared_accepted_head_reparse_receipt(
    index: sqlite3.Connection,
    prepared_write: PreparedSessionWrite,
    *,
    content_hash: str,
    decided_at_ms: int,
) -> None:
    """Consume the existing session carrier's exact selected correction."""
    receipt = prepared_write.reparse_receipt
    if receipt is None:
        return
    if receipt.accepted_content_hash != bytes.fromhex(content_hash) or receipt.session_id != prepared_write.session_id:
        raise PreparedSessionWriteRefusedError("session publication names another prepared reparse correction")
    record_revision_application_sync(index, receipt, decided_at_ms=decided_at_ms)


def prepared_raw_membership_retired_full_revision_siblings(
    seal: PreparedIndexMutation,
    logical_source_key: str,
) -> tuple[str, ...]:
    """Use the canonical sibling predicate on the merged selected Source state."""
    _load_membership_selector_inputs(seal, logical_source_key, None)
    with seal.source_rows(
        _RETIRED_MEMBERSHIP_SIBLINGS_SQL, _retired_membership_siblings_parameters(logical_source_key)
    ) as selected:
        rows = selected.fetchall()
    return tuple(str(row[0]) for row in rows)


_RAW_DIVERGENT_PATH_SQL = """
            SELECT 1
            FROM raw_sessions AS this
            WHERE this.logical_source_key = ? AND this.revision_kind = 'full'
              AND (
                  EXISTS (
                      SELECT 1 FROM raw_sessions AS other
                      WHERE other.source_path = this.source_path
                        AND other.raw_id != this.raw_id
                        AND other.revision_kind = 'full'
                        AND (other.logical_source_key IS NULL OR other.logical_source_key != this.logical_source_key)
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM raw_sessions AS other
                      JOIN raw_session_memberships AS m ON m.raw_id = other.raw_id
                      JOIN raw_membership_census AS c ON c.raw_id = other.raw_id
                      WHERE other.source_path = this.source_path
                        AND other.raw_id != this.raw_id
                        AND c.revision_authority = ?
                  )
              )
            LIMIT 1
            """


def _has_retained_full_byte_claim(row: sqlite3.Row | tuple[object, ...]) -> bool:
    """Only byte proof or an asserted root already owns its full authority."""
    return str(row[3]) == RawRevisionAuthority.BYTE_PROVEN.value or (
        str(row[3]) == RawRevisionAuthority.ASSERTED.value and row[4] is None
    )


def _classify_full_revision_byte_inputs(
    full_rows: Sequence[sqlite3.Row | tuple[object, ...]],
    open_input: Callable[[str, str], BinaryIO],
) -> tuple[tuple[str, str, str | None, str | None, int], ...]:
    """The one byte-chain/duplicate law, independent of Source publication."""
    historical: list[HistoricalRawRevisionStream] = []
    for row in full_rows:

        def open_payload(raw_id: str = str(row[0]), blob_hash: str = str(row[1])) -> BinaryIO:
            return open_input(raw_id, blob_hash)

        historical.append(
            HistoricalRawRevisionStream(
                raw_id=str(row[0]),
                payload_size=_source_integer(row[2]),
                open_payload=open_payload,
            )
        )
    # Original byte claims remain authority even when a later observation forks
    # or predates their root. Only undecided observations are classified anew.
    retained_claims = {
        str(row[0]): (
            str(row[3]),
            str(row[4]) if row[4] is not None else None,
            str(row[5]) if row[5] is not None else None,
            _source_integer(row[6]),
        )
        for row in full_rows
        if _has_retained_full_byte_claim(row)
    }
    # A Source-admitted asserted baseline is preserved, but does not prove
    # that another payload extends it. Only retained byte proof anchors that law.
    anchors = {
        raw_id: binding
        for raw_id, binding in retained_claims.items()
        if binding[0] == RawRevisionAuthority.BYTE_PROVEN.value
    }
    streams = {item.raw_id: item for item in historical}
    eligible = set(streams)
    if anchors:
        for raw_id in streams.keys() - anchors.keys():
            eligible.discard(raw_id)
            predates_anchor = False
            for anchor_id in anchors:
                pair = classify_historical_full_revision_streams([streams[anchor_id], streams[raw_id]])
                candidate = next(item for item in pair if item.raw_id == raw_id)
                if (
                    streams[raw_id].payload_size < streams[anchor_id].payload_size
                    and candidate.authority is RawRevisionAuthority.BYTE_PROVEN
                ):
                    predates_anchor = True
                if (
                    candidate.authority is RawRevisionAuthority.BYTE_PROVEN
                    and streams[raw_id].payload_size >= streams[anchor_id].payload_size
                ):
                    eligible.add(raw_id)
            if predates_anchor:
                eligible.discard(raw_id)
    decisions = classify_historical_full_revision_streams([item for item in historical if item.raw_id in eligible])
    by_raw_id = {decision.raw_id: decision for decision in decisions}
    baseline_ids = [decision.raw_id for decision in decisions if decision.relation == "baseline"]
    baseline_raw_id = baseline_ids[0] if len(baseline_ids) == 1 else None
    original_by_hash = {str(row[1]): str(row[0]) for row in full_rows if str(row[0]) in anchors}
    row_by_id = {str(row[0]): row for row in full_rows}
    resolved: dict[str, tuple[str, str | None, str | None, int]] = dict(retained_claims)

    # Parents have smaller payloads and duplicate representatives sort first;
    # this is an iterative pass even for arbitrarily long retained chains.
    for row in sorted(full_rows, key=lambda row: (_source_integer(row[2]), str(row[0]))):
        raw_id = str(row[0])
        if raw_id in resolved:
            continue
        result: tuple[str, str | None, str | None, int]
        duplicate_anchor = original_by_hash.get(str(row_by_id[raw_id][1]))
        decision = by_raw_id.get(raw_id)
        if duplicate_anchor is not None:
            authority, _predecessor, baseline, generation = anchors[duplicate_anchor]
            result = (authority, None, baseline, generation)
        elif decision is None or decision.authority is not RawRevisionAuthority.BYTE_PROVEN:
            result = (RawRevisionAuthority.QUARANTINED.value, None, None, 0)
        elif decision.duplicate_of_raw_id is not None:
            authority, _predecessor, baseline, generation = resolved[decision.duplicate_of_raw_id]
            result = (authority, None, baseline, generation)
        elif decision.predecessor_raw_id is not None:
            parent = decision.predecessor_raw_id
            authority, _predecessor, baseline, generation = resolved[parent]
            result = (
                (authority, parent, baseline or parent, generation + 1)
                if authority == RawRevisionAuthority.BYTE_PROVEN.value
                else (RawRevisionAuthority.QUARANTINED.value, None, None, 0)
            )
        elif anchors:
            result = (RawRevisionAuthority.QUARANTINED.value, None, None, 0)
        else:
            result = (decision.authority.value, None, baseline_raw_id, 0)
        resolved[raw_id] = result

    return tuple((str(row[0]), *resolved[str(row[0])]) for row in full_rows)


_RAW_BYTE_AUTHORITY_SQL = (
    "SELECT revision_authority, predecessor_raw_id, baseline_raw_id, acquisition_generation "
    "FROM raw_sessions WHERE raw_id=?"
)


def _write_full_revision_byte_updates(
    producer: RawRevisionBindingProducer,
    updates: Iterable[tuple[str, str, str | None, str | None, int]],
) -> None:
    for raw_id, authority, predecessor, baseline, generation in updates:
        check_compute_cancelled()
        operands = tuple(
            producer.binding_literal(value) for value in (authority, predecessor, baseline, generation, raw_id)
        )
        expressions = tuple(item[0] for item in operands)
        parameters = tuple(value for _expression, values in operands for value in values)
        producer.update_binding(
            raw_id,
            f"UPDATE raw_sessions SET revision_authority={expressions[0]}, predecessor_raw_id={expressions[1]}, "
            f"baseline_raw_id={expressions[2]}, acquisition_generation={expressions[3]} WHERE raw_id={expressions[4]}",
            parameters,
        )


_FULL_REVISION_BYTE_ROWS_SQL = """
        SELECT raw_id, lower(hex(blob_hash)) AS blob_hash, blob_size,
               revision_authority, predecessor_raw_id, baseline_raw_id, acquisition_generation
        FROM raw_sessions
        WHERE logical_source_key = ? AND revision_kind = 'full'
        """


def _load_byte_classification_inputs(seal: PreparedIndexMutation, logical_source_key: str) -> None:
    """Load exact key/path siblings before the canonical classification predicates.

    Hydration suppresses touched original versions; earlier selected mutations
    can also introduce new path matches. Every original same-path row is loaded
    before the negative/divergent predicate, even if its old governance is not
    currently eligible.
    """
    producer = _PreparedSourceProducer(seal)
    producer._load_artifact_inputs(
        "raw_sessions",
        "SELECT rowid FROM raw_sessions WHERE logical_source_key=? ORDER BY rowid",
        (logical_source_key,),
    )
    _load_membership_selector_inputs(seal, logical_source_key, None)
    after: str | None = None
    while True:
        with seal.source_rows(
            "SELECT DISTINCT source_path FROM raw_sessions WHERE logical_source_key=? AND revision_kind='full' "
            "AND (? IS NULL OR source_path>?) ORDER BY source_path LIMIT 256",
            (logical_source_key, after, after),
        ) as rows:
            page = tuple(str(row[0]) for row in rows)
        if not page:
            break
        for source_path in page:
            producer._load_artifact_inputs(
                "raw_sessions",
                "SELECT rowid FROM raw_sessions WHERE source_path=? ORDER BY rowid",
                (source_path,),
            )
        after = page[-1]
    # Path siblings can already be membership governed under a different key.
    # Load their exact census/member relations, not just the current key cohort.
    after_raw: str | None = None
    while True:
        with seal.source_rows(
            "SELECT raw_id FROM raw_sessions WHERE (? IS NULL OR raw_id>?) AND (logical_source_key=? "
            "OR source_path IN (SELECT source_path FROM raw_sessions WHERE logical_source_key=? AND revision_kind='full')) "
            "ORDER BY raw_id LIMIT 256",
            (after_raw, after_raw, logical_source_key, logical_source_key),
        ) as rows:
            page = tuple(str(row[0]) for row in rows)
        if not page:
            return
        for raw_id in page:
            _load_parser_census_source_inputs(seal, raw_id)
        after_raw = page[-1]


def prepare_raw_revision_byte_classification(
    seal: PreparedIndexMutation,
    logical_source_key: str,
    *,
    payload_store: BlobStore,
) -> tuple[bool, tuple[tuple[str, tuple[int, int, int, int, int]], ...]]:
    """Run the canonical byte law and its fixed point on the original Source tape.

    The parent owns the original read window and selected producer. This method
    never publishes, commits or advances observers. The returned file seals are
    revalidated before the parent publishes the captured canonical statements.
    """
    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.storage.blob_store import BlobVerificationCancelledError

    _load_byte_classification_inputs(seal, logical_source_key)
    with seal.source_rows(
        "SELECT 1 FROM raw_sessions WHERE logical_source_key=? AND source_revision IS NOT NULL LIMIT 1",
        (logical_source_key,),
    ) as rows:
        if rows.fetchone() is None:
            return False, ()
    with seal.source_rows(_FULL_REVISION_BYTE_ROWS_SQL, (logical_source_key,)) as rows:
        full_rows = rows.fetchall()
    if full_rows and prepared_raw_membership_retired_full_revision_siblings(seal, logical_source_key):
        full_rows = []
    if full_rows:
        with seal.source_rows(
            _RAW_DIVERGENT_PATH_SQL,
            (logical_source_key, RawRevisionAuthority.QUARANTINED.value),
        ) as rows:
            divergent = rows.fetchone() is not None
        if divergent:
            full_rows = []
    blob_stats: dict[str, tuple[int, int, int, int, int]] = {}
    for row in full_rows:
        check_compute_cancelled()
        raw_id, blob_hash, blob_size = str(row[0]), str(row[1]), int(row[2])
        actual_hash, actual_size = seal.retain_original_blob_input(raw_id)
        if actual_hash.hex() != blob_hash or actual_size != blob_size:
            raise PreparedRawClassificationStaleError("selected byte-classification input changed")
        path = payload_store.blob_path(blob_hash)
        try:
            before = _blob_stat_identity(path)
            verified = payload_store.verify(blob_hash, stop=compute_cancel_requested)
            after = _blob_stat_identity(path)
        except BlobVerificationCancelledError as failure:
            raise DaemonOperationCancelled("byte classification was cancelled") from failure
        except OSError as failure:
            raise PreparedRawClassificationStaleError("retained classification bytes disappeared") from failure
        if not verified or before != after:
            raise PreparedRawClassificationStaleError("retained classification bytes changed")
        blob_stats[blob_hash] = after
    # Retained authority cannot change without an undecided full observation.
    # Keep original blob capture and integrity checks above, but do not read
    # those bytes again for prefix comparison on a metadata-only append pass.
    updates = (
        ()
        if all(_has_retained_full_byte_claim(row) for row in full_rows)
        else _classify_full_revision_byte_inputs(full_rows, lambda _raw_id, blob_hash: payload_store.open(blob_hash))
    )
    for blob_hash, identity in blob_stats.items():
        check_compute_cancelled()
        try:
            current = _blob_stat_identity(payload_store.blob_path(blob_hash))
        except OSError as failure:
            raise PreparedRawClassificationStaleError("classification bytes disappeared during comparison") from failure
        if current != identity:
            raise PreparedRawClassificationStaleError("classification bytes changed during comparison")
    producer = _PreparedSourceProducer(seal)
    changed = any(
        producer.read_binding(raw_id, _RAW_BYTE_AUTHORITY_SQL, (raw_id,))
        != (authority, predecessor, baseline, generation)
        for raw_id, authority, predecessor, baseline, generation in updates
    )
    with seal.source_rows(_CONTIGUOUS_APPEND_CANDIDATES_SQL, (logical_source_key,)) as rows:
        promotable = _unique_contiguous_append_candidates(rows)
    if not changed and not promotable:
        return False, tuple(blob_stats.items())
    _write_full_revision_byte_updates(producer, updates)
    while True:
        check_compute_cancelled()
        with seal.source_rows(_CONTIGUOUS_APPEND_CANDIDATES_SQL, (logical_source_key,)) as rows:
            promotable = _unique_contiguous_append_candidates(rows)
        if not promotable:
            break
        _write_full_revision_byte_updates(
            producer,
            (
                (str(row[0]), RawRevisionAuthority.BYTE_PROVEN.value, str(row[1]), str(row[2]), int(row[3]))
                for row in promotable
            ),
        )
    return True, tuple(blob_stats.items())


_CONTIGUOUS_APPEND_CANDIDATES_SQL = """
            SELECT child.raw_id, parent.raw_id, CASE WHEN parent.revision_kind = 'full' THEN parent.raw_id ELSE parent.baseline_raw_id END,
                   parent.acquisition_generation + 1
            FROM raw_sessions AS child
            JOIN raw_sessions AS parent
              ON parent.logical_source_key = child.logical_source_key
             AND parent.source_revision = child.predecessor_source_revision
             AND parent.revision_authority = 'byte_proven'
             AND (
                 (parent.revision_kind = 'full' AND parent.blob_size = child.append_start_offset)
                 OR
                 (parent.revision_kind = 'append' AND parent.append_end_offset = child.append_start_offset)
             )
            WHERE child.logical_source_key = ?
              AND child.revision_kind = 'append'
              AND (
                  child.revision_authority = 'quarantined'
                  OR child.predecessor_raw_id != parent.raw_id
                  OR child.baseline_raw_id != CASE WHEN parent.revision_kind = 'full' THEN parent.raw_id ELSE parent.baseline_raw_id END
                  OR child.acquisition_generation != parent.acquisition_generation + 1
              )
            """


def _unique_contiguous_append_candidates(
    candidates: Iterable[sqlite3.Row | tuple[object, ...]],
) -> tuple[tuple[str, str, str, int], ...]:
    by_child: dict[str, list[tuple[str, str, str, int]]] = {}
    for row in candidates:
        generation = row[3]
        if not isinstance(generation, int):
            raise RuntimeError("canonical append authority returned a non-integer generation")
        by_child.setdefault(str(row[0]), []).append((str(row[0]), str(row[1]), str(row[2]), generation))
    return tuple(rows[0] for rows in by_child.values() if len(rows) == 1)


_RAW_REVISION_CANDIDATES_SQL = """
        SELECT raw_id, revision_kind, source_revision, acquisition_generation,
               revision_authority, blob_size, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, predecessor_source_revision
        FROM raw_sessions
        WHERE logical_source_key = ? AND source_revision IS NOT NULL
        """


def _revision_candidates_from_rows(
    logical_source_key: str,
    rows: Iterable[sqlite3.Row | tuple[Any, ...]],
) -> list[RevisionCandidate]:
    return [
        RevisionCandidate(
            raw_id=str(row[0]),
            logical_source_key=logical_source_key,
            kind=RawRevisionKind(str(row[1])),
            source_revision=str(row[2]),
            acquisition_generation=int(row[3]),
            authority=RawRevisionAuthority(str(row[4])),
            blob_size=int(row[5]),
            predecessor_source_revision=str(row[10]) if row[10] is not None else None,
            predecessor_raw_id=str(row[6]) if row[6] is not None else None,
            baseline_raw_id=str(row[7]) if row[7] is not None else None,
            append_start_offset=int(row[8]) if row[8] is not None else None,
            append_end_offset=int(row[9]) if row[9] is not None else None,
        )
        for row in rows
    ]


def prepared_raw_revision_candidates(
    seal: PreparedIndexMutation,
    logical_source_key: str,
) -> list[RevisionCandidate]:
    """Read the canonical candidate fields from their original selected Source."""
    _PreparedSourceProducer(seal)._load_artifact_inputs(
        "raw_sessions",
        "SELECT rowid FROM raw_sessions WHERE logical_source_key=? ORDER BY rowid",
        (logical_source_key,),
    )
    with seal.source_rows(_RAW_REVISION_CANDIDATES_SQL, (logical_source_key,)) as rows:
        return _revision_candidates_from_rows(logical_source_key, rows)


def prepared_raw_revision_replay_plan(
    seal: PreparedIndexMutation,
    logical_source_key: str,
) -> RevisionReplayPlan:
    return plan_revision_replay(prepared_raw_revision_candidates(seal, logical_source_key))


class RetainedRevisionBytesRead(Protocol):
    """Actual selected raw byte custody, independent of a writable archive."""

    def open_raw_revision_material(
        self,
        raw_id: str,
    ) -> AbstractContextManager[tuple[Provider, BinaryIO, str, RawRevisionKind]]: ...


@contextmanager
def _governance_read_rows(
    connection: sqlite3.Connection, sql: str, parameters: tuple[object, ...] = ()
) -> Iterator[sqlite3.Cursor]:
    """Retain the actual statement and settle it through the canonical cursor owner."""
    from polylogue.storage.io_phase_metrics import connection_cursor

    with connection_cursor(connection, sql, parameters) as cursor:
        yield cursor


_RAW_REVISION_DESCRIPTOR_SQL = """
    SELECT origin, detected_provider, capture_mode, lower(hex(blob_hash)), source_path, revision_kind, blob_size
    FROM raw_sessions WHERE raw_id = ?
"""


def _raw_revision_descriptor_from_row(
    row: sqlite3.Row | tuple[Any, ...] | None,
    raw_id: str,
) -> tuple[Provider, str, str, RawRevisionKind, int]:
    if row is None:
        raise KeyError(raw_id)
    return (
        (
            Provider.from_string(str(row[1]))
            if row[1] is not None
            else provider_from_origin(Origin.from_string(str(row[0])), family_hint=row[2])
        ),
        str(row[3]),
        str(row[4]),
        RawRevisionKind(str(row[5])),
        int(row[6]),
    )


def prepared_raw_revision_descriptor(
    seal: PreparedIndexMutation,
    raw_id: str,
) -> tuple[Provider, str, str, RawRevisionKind, int]:
    """Read selected metadata while accounting its original retained CAS."""
    _load_raw_session_input(seal, raw_id)
    blob_hash, byte_length = seal.retain_original_blob_input(raw_id)
    with seal.source_rows(_RAW_REVISION_DESCRIPTOR_SQL, (raw_id,)) as selected:
        descriptor = _raw_revision_descriptor_from_row(selected.fetchone(), raw_id)
    if descriptor[1] != blob_hash.hex() or descriptor[4] != byte_length:
        from polylogue.storage.sqlite.reference_seal import ReferenceSealError

        raise ReferenceSealError("selected parser descriptor changed its original acquisition bytes")
    return descriptor


_RAW_NATIVE_ID_SQL = "SELECT native_id FROM raw_sessions WHERE raw_id = ?"


def _raw_native_id_from_row(row: sqlite3.Row | tuple[Any, ...] | None) -> str | None:
    if row is None:
        return None
    value = row[0]
    return value if isinstance(value, str) and value.strip() else None


def prepared_raw_native_id(seal: PreparedIndexMutation, raw_id: str) -> str | None:
    _load_raw_session_input(seal, raw_id)
    with seal.source_rows(_RAW_NATIVE_ID_SQL, (raw_id,)) as rows:
        return _raw_native_id_from_row(rows.fetchone())


def prepared_raw_typed_logical_key(seal: PreparedIndexMutation, raw_id: str) -> str | None:
    """The logical key the raw is typed under on the original Source state."""
    _load_raw_session_input(seal, raw_id)
    with seal.source_rows("SELECT logical_source_key FROM raw_sessions WHERE raw_id=?", (raw_id,)) as rows:
        row = rows.fetchone()
    return None if row is None or row[0] is None else str(row[0])


def _raw_revision_rebuild_logical_keys(
    reader: RawMembershipSelectionRead,
    raw_ids: Sequence[str],
) -> tuple[str, ...]:
    paths = _selection_values(reader, "paths", set(raw_ids))
    return tuple(sorted(_selection_values(reader, "path_keys", paths)))


def _raw_replay_representative_query(keys: Sequence[str]) -> tuple[str, tuple[object, ...]]:
    """Rank each key's raws, whether the raw's revision key or its membership names it."""
    marks = ",".join("?" for _ in keys)
    order = raw_receipt_order_sql("raw_sessions")
    return (
        "SELECT logical_source_key, raw_id FROM ("
        f"SELECT raw_sessions.logical_source_key AS logical_source_key, raw_sessions.raw_id AS raw_id, {order} AS rank "
        f"FROM raw_sessions WHERE raw_sessions.logical_source_key IN ({marks}) "
        "UNION "
        f"SELECT m.logical_source_key, raw_sessions.raw_id, {order} "
        "FROM raw_session_memberships AS m JOIN raw_sessions ON raw_sessions.raw_id = m.raw_id "
        f"WHERE m.logical_source_key IN ({marks})"
        ") ORDER BY logical_source_key, rank DESC, raw_id ASC",
        (*keys, *keys),
    )


_RAW_MEMBERSHIP_CENSUS_COLUMNS = """
    r.raw_id,
    r.source_index,
    (
        EXISTS(SELECT 1 FROM raw_artifacts AS a WHERE a.raw_id = r.raw_id AND a.parse_as_session = 0)
        OR EXISTS(
            SELECT 1 FROM raw_membership_census AS c
            WHERE c.raw_id = r.raw_id
              AND c.parser_fingerprint = ?
              AND c.status = 'non_session'
              AND r.parsed_at_ms IS NOT NULL
              AND r.parse_error IS NULL
        )
    ),
    r.rowid
"""


def _raw_membership_census_query(raw_ids: Sequence[str] | None) -> tuple[str, tuple[object, ...]]:
    predicate = (
        "" if raw_ids is None else (f"WHERE r.raw_id IN ({','.join('?' for _ in raw_ids)})" if raw_ids else "WHERE 0")
    )
    return (
        f"SELECT {_RAW_MEMBERSHIP_CENSUS_COLUMNS} FROM raw_sessions AS r {predicate} ORDER BY r.raw_id",
        (raw_authority_parser_fingerprint(), *(raw_ids or ())),
    )


def _raw_membership_census_rows_from_rows(rows: Iterable[sqlite3.Row]) -> tuple[tuple[str, int, bool, int], ...]:
    result: list[tuple[str, int, bool, int]] = []
    for row in rows:
        check_compute_cancelled()
        result.append((str(row[0]), int(row[1]), bool(row[2]), int(row[3])))
    return tuple(result)


def _prepared_raw_has_byte_revision_dependents(seal: PreparedIndexMutation, raw_id: str) -> bool:
    """Merge staged matches with untouched original dependency authority."""
    check_compute_cancelled()
    parameters = (raw_id, raw_id, raw_id)
    with seal.source_rows(RAW_BYTE_REVISION_DEPENDENTS_SQL, parameters) as selected_rows:
        if selected_rows.fetchone() is not None:
            check_compute_cancelled()
            return True
    # LIMIT 1 applies only after suppressing touched original versions. An
    # earlier staged deletion may remove the first original matching row.
    with seal.original_rows(
        "source", "SELECT rowid FROM raw_sessions\n" + _RAW_BYTE_REVISION_DEPENDENTS_FILTER_SQL, parameters
    ) as original_rows:
        for row in original_rows:
            check_compute_cancelled()
            if not seal.source_row_is_touched("raw_sessions", int(row[0])):
                return True
    check_compute_cancelled()
    return False


def _load_parser_census_source_inputs(seal: PreparedIndexMutation, raw_id: str) -> None:
    """Hydrate this census predicate from its original pinned Source rows.

    The caller owns ``original_read_snapshot`` and ``source_producer``.
    Loading reads complete native cell descriptors, including large receipt
    cells, without returning their scalar payload to Python. The Native
    owner preserves an already loaded or touched coordinate, so this read
    cannot resurrect an earlier staged deletion or overwrite a staged update.
    Newly staged matches already belong to the selected Source state.
    """
    for table in (
        "raw_sessions",
        "raw_artifacts",
        "raw_session_memberships",
        "raw_membership_census",
        "raw_authority_parser_census",
    ):
        check_compute_cancelled()
        with seal.original_rows(
            "source", f"SELECT rowid FROM {table} WHERE raw_id = ? ORDER BY rowid", (raw_id,)
        ) as original_rows:
            for row in original_rows:
                check_compute_cancelled()
                with seal.verified_namespace():
                    if seal.source_row_is_loaded(table, int(row[0])):
                        continue
                    image = seal.retain_tier_row("source", table, int(row[0]))
                    if image is None:
                        raise RuntimeError("pinned parser census input disappeared")
                    seal.load_source_row(image)


def _prepared_membership_identity_keys(seal: PreparedIndexMutation, raw_id: str) -> Iterator[str]:
    after: str | None = None
    while True:
        check_compute_cancelled()
        with seal.source_rows(
            "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=? "
            "AND (? IS NULL OR logical_source_key>?) ORDER BY logical_source_key LIMIT 256",
            (raw_id, after, after),
        ) as rows:
            page = tuple(str(row[0]) for row in rows)
        if not page:
            return
        yield from page
        after = page[-1]


def _prepared_parser_receipt_keys(seal: PreparedIndexMutation, raw_id: str) -> Iterator[str]:
    from polylogue.archive.revision_authority import InvalidParserCensusKeysError
    from polylogue.storage.raw_authority import validated_parser_census_logical_keys

    def values() -> Generator[object, None, None]:
        with seal.source_rows(
            "SELECT item.value,item.type FROM raw_authority_parser_census c, "
            "json_each(c.logical_keys_json) item WHERE c.raw_id=? ORDER BY item.key",
            (raw_id,),
        ) as rows:
            for value, kind in rows:
                check_compute_cancelled()
                if kind != "text":
                    raise InvalidParserCensusKeysError("parser identity receipt has a non-string key")
                yield value

    yield from validated_parser_census_logical_keys(values())


def prepared_parser_census_is_current(seal: PreparedIndexMutation, raw_id: str) -> bool:
    """Validate the actual selected receipt with the existing disk measurement.

    Receipt JSON stays on the same native Source owner. The canonical key
    validator consumes its ordered rows after the membership stream settles;
    no receipt scalar or second identity inference enters Python.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    check_compute_cancelled()
    _load_parser_census_source_inputs(seal, raw_id)
    fingerprint = raw_authority_parser_fingerprint()
    with seal.source_rows(
        """
        SELECT r.logical_source_key,r.revision_kind,r.source_index,
               EXISTS(SELECT 1 FROM raw_artifacts a WHERE a.raw_id=r.raw_id AND a.parse_as_session=0),
               EXISTS(SELECT 1 FROM raw_membership_census mc WHERE mc.raw_id=r.raw_id
                      AND mc.parser_fingerprint=? AND mc.status='non_session'),
               EXISTS(SELECT 1 FROM raw_membership_census mc WHERE mc.raw_id=r.raw_id
                      AND mc.parser_fingerprint=? AND mc.status='failed' AND mc.revision_authority=?),
               COALESCE(c.parser_fingerprint=? AND c.status='complete' AND c.detail LIKE 'parser-observed:%',0),
               json_valid(c.logical_keys_json),
               CASE WHEN json_valid(c.logical_keys_json) THEN json_type(c.logical_keys_json) ELSE NULL END
        FROM raw_sessions r LEFT JOIN raw_authority_parser_census c ON c.raw_id=r.raw_id
        WHERE r.raw_id=?
        """,
        (fingerprint, fingerprint, RawRevisionAuthority.BYTE_PROVEN.value, fingerprint, raw_id),
    ) as rows:
        raw = rows.fetchone()
    if raw is None:
        raise ReferenceSealStaleError("selected parser census raw is absent")
    if not raw[6] or not raw[7] or raw[8] != "array":
        return False
    with parser_census_identity_measurement(
        raw_logical_key=raw[0],
        revision_kind=raw[1],
        membership_logical_keys=_prepared_membership_identity_keys(seal, raw_id),
        observed_logical_keys=_prepared_parser_receipt_keys(seal, raw_id),
        observed_are_receipt=True,
        check_stop=check_compute_cancelled,
    ) as measured:
        return measured.complete(
            typed_non_session=bool(raw[3]),
            parser_confirmed_non_session=bool(raw[4]),
            byte_governed_fragment=int(raw[2]) < 0 and bool(raw[5]),
        )


@dataclass(frozen=True, slots=True)
class _PreparedSourceProducer:
    seal: PreparedIndexMutation

    def material_publication_read(self, publisher: ArchiveBlobPublisher) -> BlobPublicationSourceRead:
        from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

        return PreparedSessionSourceRead(self.seal, blob_store=publisher)

    def material_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return self.binding_literal(value)

    def material_rows(self, material_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.storage.materials import _MATERIAL_ROWS_SQL

        self._load_artifact_inputs(
            "material_observations",
            "SELECT rowid FROM material_observations WHERE material_id=?",
            (material_id,),
        )
        return self.seal.source_rows(_MATERIAL_ROWS_SQL, (material_id,))

    def material_duplicate_rows(
        self,
        blob_hash: bytes,
        material_id: str,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.storage.materials import _MATERIAL_DUPLICATE_SQL

        # Hydrate every original match before selected-state LIMIT chooses a
        # winner, including later matches after a staged retarget/delete.
        self._load_artifact_inputs(
            "material_observations",
            "SELECT rowid FROM material_observations WHERE blob_hash=? AND material_id!=? ORDER BY rowid",
            (blob_hash, material_id),
        )
        return self.seal.source_rows(_MATERIAL_DUPLICATE_SQL, (blob_hash, material_id))

    def material_previous_rows(
        self,
        referrer_ref: str,
        source_uri: str,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.storage.materials import _MATERIAL_PREVIOUS_SQL

        self._load_artifact_inputs(
            "material_evidence_links",
            "SELECT rowid FROM material_evidence_links WHERE evidence_ref=? AND relation='refers_to' ORDER BY rowid",
            (referrer_ref,),
        )
        self._load_artifact_inputs(
            "material_observations",
            "SELECT rowid FROM material_observations WHERE source_uri=? ORDER BY rowid",
            (source_uri,),
        )
        return self.seal.source_rows(_MATERIAL_PREVIOUS_SQL, (referrer_ref, source_uri))

    def material_supersede_write(
        self,
        material_id: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        self._load_artifact_inputs(
            "material_observations",
            "SELECT rowid FROM material_observations WHERE material_id=?",
            (material_id,),
        )
        key = self.seal.retain_literal_scalar(material_id)
        return self.seal.source_statement(
            sql,
            parameters,
            table="material_observations",
            writable_targets=(("material_observations", (key,)),),
        )

    def material_write(
        self,
        material_id: str,
        supersedes_material_id: str | None,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        self._load_artifact_inputs(
            "material_observations",
            "SELECT rowid FROM material_observations WHERE material_id=?",
            (material_id,),
        )
        if supersedes_material_id is not None:
            self._load_artifact_inputs(
                "material_observations",
                "SELECT rowid FROM material_observations WHERE material_id=?",
                (supersedes_material_id,),
            )
        self.seal.source_allocation_dependencies("material_observations")
        key = self.seal.retain_literal_scalar(material_id)
        return self.seal.source_statement(
            sql,
            parameters,
            table="material_observations",
            writable_targets=(("material_observations", (key,)),),
            allocation_parameter=0,
        )

    def material_link_write(
        self,
        key: tuple[str, str, str],
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        self._load_artifact_inputs(
            "material_observations",
            "SELECT rowid FROM material_observations WHERE material_id=?",
            (key[0],),
        )
        self._load_artifact_inputs(
            "material_evidence_links",
            "SELECT rowid FROM material_evidence_links WHERE material_id=? AND evidence_ref=? AND relation=?",
            key,
        )
        self.seal.source_allocation_dependencies("material_evidence_links")
        cells = tuple(self.seal.retain_literal_scalar(_source_scalar(value)) for value in key)
        return self.seal.source_statement(
            sql,
            parameters,
            table="material_evidence_links",
            writable_targets=(("material_evidence_links", cells),),
            allocation_parameter=0,
        )

    def consume_material_receipt(self, claim: PreparedBlobPublicationClaim) -> None:
        from polylogue.storage.blob_publication import blob_publication_receipt_delete

        publication_id = claim.receipt.publication_id
        blob_hash = bytes.fromhex(claim.receipt.blob_hash)
        statement = blob_publication_receipt_delete(publication_id, blob_hash, literal=self.binding_literal)
        if statement is None:
            return
        self._load_artifact_inputs(
            "blob_publication_reservations",
            "SELECT rowid FROM blob_publication_reservations WHERE publication_id=? AND blob_hash=?",
            (publication_id, blob_hash),
        )
        key = self.seal.retain_literal_scalar(publication_id)
        with self.seal.source_statement(
            *statement,
            table="blob_publication_reservations",
            writable_targets=(("blob_publication_reservations", (key,)),),
        ):
            pass

    def binding_transaction(self, *, manage_transaction: bool) -> AbstractContextManager[object]:
        if manage_transaction:
            raise RuntimeError("prepared Source binding cannot commit its private preparation state")
        return nullcontext()

    def binding_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return self.seal.source_literal_expression(self.seal.retain_literal_scalar(_source_scalar(value)))

    def update_binding(
        self,
        raw_id: str,
        sql: str,
        parameters: tuple[object, ...],
        *,
        parser_singleton_witness: PreparedParserSingletonWitness | None = None,
    ) -> int:
        _load_parser_census_source_inputs(self.seal, raw_id)
        key = self.seal.retain_literal_scalar(raw_id)
        witness_options: dict[str, Any] = (
            {} if parser_singleton_witness is None else {"parser_singleton_witness": parser_singleton_witness}
        )
        with self.seal.source_statement(
            sql,
            parameters,
            table="raw_sessions",
            writable_targets=(("raw_sessions", (key,)),),
            **witness_options,
        ) as cursor:
            return cursor.rowcount

    def read_binding(self, raw_id: str, sql: str, parameters: tuple[object, ...]) -> tuple[object, ...] | None:
        _load_parser_census_source_inputs(self.seal, raw_id)
        with self.seal.source_rows(sql, parameters) as cursor:
            row = cursor.fetchone()
        return None if row is None else tuple(row)

    def membership_decision_write(
        self,
        raw_id: str,
        logical_source_key: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> None:
        _load_parser_census_source_inputs(self.seal, raw_id)
        key = tuple(self.seal.retain_literal_scalar(_source_scalar(value)) for value in (raw_id, logical_source_key))
        with self.seal.source_statement(
            sql,
            parameters,
            table="raw_session_memberships",
            writable_targets=(("raw_session_memberships", key),),
        ):
            pass

    def raw_insert_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return self.binding_literal(value)

    def raw_insert(
        self,
        raw_id: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        _load_parser_census_source_inputs(self.seal, raw_id)
        self.seal.source_allocation_dependencies("raw_sessions")
        key = self.seal.retain_literal_scalar(raw_id)
        return self.seal.source_statement(
            sql,
            parameters,
            table="raw_sessions",
            writable_targets=(("raw_sessions", (key,)),),
            prepared_cells={"raw_id": key},
            allocation_parameter=0,
        )

    def state_transaction(self, manage_transaction: bool) -> AbstractContextManager[object]:
        return self.binding_transaction(manage_transaction=manage_transaction)

    def state_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return self.binding_literal(value)

    def state_write(self, raw_id: str, sql: str, parameters: tuple[object, ...]) -> int:
        return self.update_binding(raw_id, sql, parameters)

    def artifact_transaction(self, manage_transaction: bool) -> AbstractContextManager[object]:
        return self.binding_transaction(manage_transaction=manage_transaction)

    def artifact_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return self.binding_literal(value)

    def _load_artifact_inputs(self, table: str, sql: str, parameters: tuple[object, ...]) -> None:
        # Every caller names one finite artifact predicate. Hydration supplies
        # readable inputs only; source_statement separately declares writes.
        with self.seal.original_rows("source", sql, parameters) as original_rows:
            for row in original_rows:
                check_compute_cancelled()
                rowid = int(row[0])
                with self.seal.verified_namespace():
                    if self.seal.source_row_is_loaded(table, rowid):
                        continue
                    image = self.seal.retain_tier_row("source", table, rowid)
                    if image is None:
                        raise RuntimeError("pinned artifact input disappeared")
                    self.seal.load_source_row(image)

    def artifact_coordinate_rows(
        self,
        raw_id: str,
        artifact: ArchiveSourceArtifact,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        # The incoming acquisition is a real FK input even when both sides
        # have blob receipts and the winner comparison never reads its row.
        self._load_artifact_inputs("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        original_sql, parameters = _artifact_coordinate_query(raw_id, artifact, columns="a.rowid")
        self._load_artifact_inputs("raw_artifacts", original_sql, parameters)
        selected_sql, parameters = _artifact_coordinate_query(raw_id, artifact, columns="a.artifact_id, a.raw_id")
        return self.seal.source_rows(selected_sql, parameters)

    def artifact_observation_rows(
        self,
        raw_id: str,
        *,
        receipt: bool,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        if receipt:
            self._load_artifact_inputs(
                "blob_refs",
                "SELECT rowid FROM blob_refs WHERE ref_id=? AND ref_type='raw_payload' ORDER BY rowid",
                (raw_id,),
            )
        else:
            self._load_artifact_inputs("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        sql, parameters = _artifact_observation_query(raw_id, receipt=receipt)
        return self.seal.source_rows(sql, parameters)

    def artifact_validation_failed(self, raw_id: str) -> bool:
        from polylogue.storage.sqlite.archive_tiers.source_write import _ARTIFACT_VALIDATION_STATUS_SQL

        self._load_artifact_inputs("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        with self.seal.source_rows(_ARTIFACT_VALIDATION_STATUS_SQL, (raw_id,)) as rows:
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
        # Artifact-ID conflicts are global even when no coordinate matched.
        # Their original row must reach the real ON CONFLICT/terminal guard.
        self._load_artifact_inputs(
            "raw_artifacts",
            "SELECT rowid FROM raw_artifacts WHERE artifact_id=?",
            (artifact_id,),
        )
        if allocation:
            self.seal.source_allocation_dependencies("raw_artifacts")
        key = self.seal.retain_literal_scalar(artifact_id)
        return self.seal.source_statement(
            sql,
            parameters,
            table="raw_artifacts",
            writable_targets=(("raw_artifacts", (key,)),),
            prepared_cells={"artifact_id": key},
            allocation_parameter=0 if allocation else None,
        )

    def blob_ref_is_excised(self, blob_hash: bytes) -> bool:
        self._load_artifact_inputs(
            "excised_content",
            "SELECT rowid FROM excised_content WHERE removed_hash=? AND hash_kind='blob_hash'",
            (blob_hash,),
        )
        with self.seal.source_rows(
            "SELECT 1 FROM excised_content WHERE removed_hash=? AND hash_kind='blob_hash' LIMIT 1",
            (blob_hash,),
        ) as rows:
            return rows.fetchone() is not None

    def blob_ref_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return self.binding_literal(value)

    def blob_ref_write(
        self,
        ref: ArchiveSourceBlobRef,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        self._load_artifact_inputs(
            "blob_refs",
            "SELECT rowid FROM blob_refs WHERE blob_hash=? AND ref_type=? AND ref_id=?",
            (ref.blob_hash, ref.ref_type, ref.raw_id),
        )
        self.seal.source_allocation_dependencies("blob_refs")
        key = tuple(
            self.seal.retain_literal_scalar(_source_scalar(value))
            for value in (
                ref.blob_hash,
                ref.ref_type,
                ref.raw_id,
                (ref.source_path or "") if ref.ref_type == "attachment" else None,
            )
        )
        return self.seal.source_statement(
            sql,
            parameters,
            table="blob_refs",
            writable_targets=(("blob_refs", key),),
            prepared_cells={
                "blob_hash": key[0],
                "ref_type": key[1],
                "ref_id": key[2],
                "source_path": self.seal.retain_literal_scalar(_source_scalar(ref.source_path)),
            },
            allocation_parameter=0,
        )

    def consume_reference_receipt(self, ref: ArchiveSourceBlobRef) -> None:
        from polylogue.storage.blob_publication import blob_publication_receipt_delete

        statement = blob_publication_receipt_delete(
            ref.publication_receipt_id, ref.blob_hash, literal=self.binding_literal
        )
        if statement is None:
            return
        self._load_artifact_inputs(
            "blob_publication_reservations",
            "SELECT rowid FROM blob_publication_reservations WHERE publication_id=? AND blob_hash=?",
            (ref.publication_receipt_id, ref.blob_hash),
        )
        key = self.seal.retain_literal_scalar(ref.publication_receipt_id)
        with self.seal.source_statement(
            *statement,
            table="blob_publication_reservations",
            writable_targets=(("blob_publication_reservations", (key,)),),
        ):
            pass


def prepare_raw_state_update(
    seal: PreparedIndexMutation,
    raw_id: str,
    *,
    state: RawSessionStateUpdate,
) -> None:
    """Retain the existing typed raw-state mutation in the parent's Source tape."""
    check_compute_cancelled()
    _apply_source_raw_state_update(_PreparedSourceProducer(seal), raw_id, state=state, manage_transaction=False)
    check_compute_cancelled()


def _prepared_membership_write_targets(
    seal: PreparedIndexMutation, raw_id: str
) -> Iterator[tuple[str, tuple[KnownTierCell, ...]]]:
    """Declare every actual membership deletion without retaining its payload."""
    after: int | None = None
    while True:
        check_compute_cancelled()
        predicate = "raw_id = ?" if after is None else "raw_id = ? AND rowid > ?"
        parameters = (raw_id,) if after is None else (raw_id, after)
        with seal.source_rows(
            f"SELECT rowid FROM raw_session_memberships WHERE {predicate} ORDER BY rowid LIMIT 256",
            parameters,
        ) as rows:
            page = tuple(int(row[0]) for row in rows)
        if not page:
            return
        for rowid in page:
            check_compute_cancelled()
            image = seal.retain_source_row("raw_session_memberships", rowid)
            if image is None:
                raise RuntimeError("selected membership deletion target disappeared")
            yield (
                "raw_session_memberships",
                tuple(image.cells[image.columns.index(column)] for column in ("raw_id", "logical_source_key")),
            )
        after = page[-1]


def _prepared_source_operands(
    seal: PreparedIndexMutation, *values: object
) -> tuple[tuple[str, ...], tuple[object, ...]]:
    """Render the finite canonical builder operands from original literal slots."""
    operands = tuple(
        seal.source_literal_expression(seal.retain_literal_scalar(_source_scalar(value))) for value in values
    )
    return (
        tuple(expression for expression, _parameters in operands),
        tuple(parameter for _expression, parameters in operands for parameter in parameters),
    )


def publish_prepared_revision_source(seal: PreparedIndexMutation, permit: KnownTierMutationPermit) -> None:
    """Publish the original prepared Source schedule on its dedicated writer.

    The admitted parent retains creator custody through this connection's
    physical close. Acceptance advances the same original Source observer
    before that close, while the dedicated writer can reserve its commit.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    if permit.terminal_parent is not seal or permit.tier != "source":
        raise ReferenceSealError("prepared Source publication requires its original Source permit")
    check_compute_cancelled()
    with permit.hold_authority(), permit.mutation_connection() as source:
        with source:
            from polylogue.storage.io_phase_metrics import close_connection_cursor

            admission_cursor = source.cursor()
            try:
                admission_cursor.execute("BEGIN IMMEDIATE")
            except BaseException as failure:
                try:
                    close_connection_cursor(source, admission_cursor)
                except BaseException as cleanup_failure:
                    raise BaseExceptionGroup(
                        "Source writer admission and cursor settlement failed", [failure, cleanup_failure]
                    ) from None
                raise
            else:
                close_connection_cursor(source, admission_cursor)
            permit.apply_source_statements(source)
            permit.allow_commit(source)
        seal.accept_known_tier_commit(permit.committed())


_CONVERTIBLE_FULL_REVISION_ROWS_SQL = """
    SELECT raw_id, revision_kind
    FROM raw_sessions
    WHERE logical_source_key = ?
    ORDER BY raw_id
"""


def _convertible_full_revision_raw_ids(rows: Sequence[sqlite3.Row | tuple[Any, ...]]) -> tuple[str, ...]:
    if not rows or any(str(row[1]) != RawRevisionKind.FULL.value for row in rows):
        return ()
    return tuple(str(row[0]) for row in rows)


def prepared_convertible_full_revision_raw_ids(
    seal: PreparedIndexMutation,
    logical_source_key: str,
) -> tuple[str, ...]:
    """Use the same full-only decision on merged original and staged rows."""
    with seal.original_rows(
        "source",
        "SELECT rowid FROM raw_sessions WHERE logical_source_key=? ORDER BY raw_id",
        (logical_source_key,),
    ) as original:
        for (rowid,) in original:
            check_compute_cancelled()
            if seal.source_row_is_loaded("raw_sessions", rowid):
                continue
            image = seal.retain_tier_row("source", "raw_sessions", rowid)
            if image is not None:
                seal.load_source_row(image)
    with seal.source_rows(_CONVERTIBLE_FULL_REVISION_ROWS_SQL, (logical_source_key,)) as selected:
        rows = selected.fetchall()
    return _convertible_full_revision_raw_ids(rows)


_PENDING_ENVELOPE_MEMBERSHIP_SQL = """
    SELECT 1 FROM raw_sessions AS r
    JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
    WHERE r.logical_source_key = ? LIMIT 1
"""


_RAW_PENDING_MEMBERSHIP_SQL = """
    SELECT 1 FROM raw_sessions AS r
    WHERE r.raw_id = ? AND substr(r.logical_source_key, 1, ?) = ?
      AND EXISTS (SELECT 1 FROM raw_session_memberships AS m WHERE m.raw_id = r.raw_id)
"""


_MEMBERSHIP_PENDING_ENVELOPE_SQL = """
    SELECT 1 FROM raw_session_memberships AS m
    JOIN raw_sessions AS r ON r.raw_id = m.raw_id
    WHERE m.logical_source_key = ? AND substr(r.logical_source_key, 1, ?) = ? LIMIT 1
"""


RawMembershipSelectionFamily = Literal["paths", "keys", "path_raws", "key_raws", "path_keys"]


class RawMembershipSelectionRead(Protocol):
    """Declared Source predicates for the membership fixed point and byte keys."""

    def raw_selection_values(
        self,
        family: RawMembershipSelectionFamily,
        operands: Sequence[str],
    ) -> set[str]: ...


def _raw_selection_queries(
    family: RawMembershipSelectionFamily,
    operands: tuple[str, ...],
) -> tuple[tuple[str, str, str, tuple[object, ...]], ...]:
    """Share exact selection SQL and its named hydration predicate family."""
    marks = ",".join("?" for _ in operands)
    shapes: tuple[tuple[str, str, str], ...]
    if family == "paths":
        shapes = (("raw_sessions", "source_path", f"raw_id IN ({marks})"),)
    elif family == "keys":
        shapes = (
            ("raw_session_memberships", "logical_source_key", f"raw_id IN ({marks})"),
            ("raw_sessions", "logical_source_key", f"raw_id IN ({marks}) AND logical_source_key IS NOT NULL"),
        )
    elif family == "path_raws":
        shapes = (("raw_sessions", "raw_id", f"source_path IN ({marks})"),)
    elif family == "path_keys":
        shapes = (
            ("raw_sessions", "logical_source_key", f"source_path IN ({marks}) AND logical_source_key IS NOT NULL"),
        )
    elif family == "key_raws":
        shapes = (
            ("raw_session_memberships", "raw_id", f"logical_source_key IN ({marks})"),
            ("raw_sessions", "raw_id", f"logical_source_key IN ({marks})"),
        )
    else:
        raise ValueError("unknown raw membership selection predicate")
    return tuple((table, column, predicate, operands) for table, column, predicate in shapes)


def _selection_values(
    reader: RawMembershipSelectionRead,
    family: RawMembershipSelectionFamily,
    values: set[str],
) -> set[str]:
    ordered = sorted(values)
    found: set[str] = set()
    for start in range(0, len(ordered), _MEMBERSHIP_EXPANSION_BATCH):
        check_compute_cancelled()
        found.update(reader.raw_selection_values(family, ordered[start : start + _MEMBERSHIP_EXPANSION_BATCH]))
    return found


def _expand_raw_membership_selection(
    reader: RawMembershipSelectionRead,
    raw_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep one fixed point over the ordinary or actual selected Source view."""
    selected = set(raw_ids)
    frontier = selected.copy()
    paths_seen: set[str] = set()
    keys_seen: set[str] = set()
    # These sets belong to this invocation's coherent Source view. A later
    # expansion after staged census changes must start with fresh frontiers.
    while frontier:
        check_compute_cancelled()
        paths = _selection_values(reader, "paths", frontier) - paths_seen
        keys = _selection_values(reader, "keys", frontier) - keys_seen
        paths_seen.update(paths)
        keys_seen.update(keys)
        discovered = _selection_values(reader, "path_raws", paths)
        discovered.update(_selection_values(reader, "key_raws", keys))
        frontier = discovered - selected
        selected.update(frontier)
    return tuple(sorted(selected)), tuple(sorted(keys_seen))


def _load_membership_selector_inputs(
    seal: PreparedIndexMutation,
    logical_source_key: str,
    source_generation_id: str | None,
) -> None:
    # Load membership matches without the mutable census/decision predicate.
    # Earlier staged changes may make an original incomplete row newly eligible.
    with seal.original_rows(
        "source",
        "SELECT rowid FROM raw_session_memberships WHERE logical_source_key=? ORDER BY raw_id",
        (logical_source_key,),
    ) as original:
        for (rowid,) in original:
            check_compute_cancelled()
            if seal.source_row_is_loaded("raw_session_memberships", rowid):
                continue
            image = seal.retain_tier_row("source", "raw_session_memberships", rowid)
            if image is not None:
                seal.load_source_row(image)
    after: str | None = None
    while True:
        with seal.source_rows(
            "SELECT raw_id FROM raw_session_memberships WHERE logical_source_key=? "
            "AND (? IS NULL OR raw_id>?) ORDER BY raw_id LIMIT 256",
            (logical_source_key, after, after),
        ) as selected:
            page = selected.fetchall()
        if not page:
            return
        for (raw_id,) in page:
            check_compute_cancelled()
            # The member's own raw row carries its typed retirement authority.
            predicates: list[tuple[str, str, tuple[object, ...]]] = [
                ("raw_membership_census", "SELECT rowid FROM raw_membership_census WHERE raw_id=?", (raw_id,)),
                ("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,)),
            ]
            if source_generation_id is not None:
                predicates.append(
                    (
                        "source_item_raw_members",
                        "SELECT rowid FROM source_item_raw_members WHERE source_generation_id=? AND raw_id=?",
                        (source_generation_id, raw_id),
                    )
                )
            for table, sql, parameters in predicates:
                with seal.original_rows("source", sql, parameters) as original:
                    for (rowid,) in original:
                        if seal.source_row_is_loaded(table, rowid):
                            continue
                        image = seal.retain_tier_row("source", table, rowid)
                        if image is not None:
                            seal.load_source_row(image)
        after = str(page[-1][0])


_RAW_FILE_MTIME_SQL = "SELECT file_mtime_ms FROM raw_sessions WHERE raw_id=?"


def _raw_revision_file_mtime(row: sqlite3.Row | tuple[Any, ...] | None, raw_id: str) -> str | None:
    if row is None:
        raise KeyError(f"unknown raw revision {raw_id}")
    if row[0] is None:
        return None
    return datetime.fromtimestamp(int(row[0]) / 1000, UTC).isoformat()


def prepared_raw_revision_file_mtime(seal: PreparedIndexMutation, raw_id: str) -> str | None:
    _load_raw_session_input(seal, raw_id)
    with seal.source_rows(_RAW_FILE_MTIME_SQL, (raw_id,)) as selected:
        row = selected.fetchone()
    return _raw_revision_file_mtime(row, raw_id)


_RAW_OBSERVATION_RECEIPT_SQL = """
    SELECT acquired_at_ms, rowid FROM blob_refs
    WHERE ref_id=? AND ref_type='raw_payload'
    ORDER BY rowid DESC LIMIT 1
"""


_RAW_ACQUISITION_TIME_SQL = "SELECT acquired_at_ms FROM raw_sessions WHERE raw_id=?"


def _raw_acquisition_observation(row: sqlite3.Row | tuple[Any, ...] | None, raw_id: str) -> tuple[int, int]:
    if row is None:
        raise KeyError(f"unknown raw revision {raw_id}")
    return int(row[0]), 0


def _load_raw_session_input(seal: PreparedIndexMutation, raw_id: str) -> None:
    with seal.original_rows("source", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,)) as original:
        for (rowid,) in original:
            check_compute_cancelled()
            if seal.source_row_is_loaded("raw_sessions", rowid):
                continue
            image = seal.retain_tier_row("source", "raw_sessions", rowid)
            if image is not None:
                seal.load_source_row(image)


def _load_raw_observation_inputs(seal: PreparedIndexMutation, raw_id: str) -> None:
    _load_raw_session_input(seal, raw_id)
    # No LIMIT before touched-coordinate suppression: an earlier staged
    # removal of the newest receipt must reveal the next surviving row.
    with seal.original_rows(
        "source",
        "SELECT rowid FROM blob_refs WHERE ref_id=? AND ref_type='raw_payload' ORDER BY rowid",
        (raw_id,),
    ) as original:
        for (rowid,) in original:
            check_compute_cancelled()
            if seal.source_row_is_loaded("blob_refs", rowid):
                continue
            image = seal.retain_tier_row("source", "blob_refs", rowid)
            if image is not None:
                seal.load_source_row(image)


_RAW_EXPORT_ORDER_SQL = """
    SELECT COALESCE((SELECT b.rowid FROM blob_refs b
                    WHERE b.ref_id=r.raw_id AND b.ref_type='raw_payload'
                    ORDER BY b.rowid DESC LIMIT 1), 0)
    FROM raw_sessions r WHERE r.raw_id=?
"""


def _raw_export_order_from_row(row: sqlite3.Row | tuple[object, ...] | None) -> int | None:
    return None if row is None else _source_integer(row[0])


def prepared_raw_export_order(seal: PreparedIndexMutation, raw_id: str) -> int | None:
    _load_raw_observation_inputs(seal, raw_id)
    with seal.source_rows(_RAW_EXPORT_ORDER_SQL, (raw_id,)) as rows:
        return _raw_export_order_from_row(rows.fetchone())


def prepared_raw_revision_observation_order(seal: PreparedIndexMutation, raw_id: str) -> tuple[int, int]:
    """Preserve actual receipt rowid ordering on merged selected evidence."""
    _load_raw_observation_inputs(seal, raw_id)
    with seal.source_rows(_RAW_OBSERVATION_RECEIPT_SQL, (raw_id,)) as selected:
        row = selected.fetchone()
    if row is not None:
        return int(row[0]), int(row[1])
    with seal.source_rows(_RAW_ACQUISITION_TIME_SQL, (raw_id,)) as selected:
        row = selected.fetchone()
    return _raw_acquisition_observation(row, raw_id)


_MEMBERSHIP_REBUILD_ROWS_SQL = """
                    SELECT m.raw_id
                    FROM raw_session_memberships AS m
                    JOIN raw_sessions AS r ON r.raw_id = m.raw_id
                    WHERE m.logical_source_key = ? AND r.revision_authority = 'byte_proven'
                    ORDER BY m.raw_id
                    """


def prepared_raw_membership_rebuild_raw_ids(
    seal: PreparedIndexMutation,
    logical_source_key: str,
) -> tuple[str, ...]:
    """Use the canonical sibling predicate on the merged selected Source state."""
    _load_membership_selector_inputs(seal, logical_source_key, None)
    with seal.source_rows(_MEMBERSHIP_REBUILD_ROWS_SQL, (logical_source_key,)) as selected:
        rows = selected.fetchall()
    return tuple(str(row[0]) for row in rows)


_RAW_REVISION_HEAD_SQL = "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?"


_MEMBERSHIP_LOGICAL_RAW_IDS_SQL = (
    "SELECT raw_id FROM raw_session_memberships WHERE logical_source_key = ? ORDER BY raw_id"
)


def _raw_revision_head_from_row(row: sqlite3.Row | tuple[object, ...] | None) -> str | None:
    return None if row is None else str(row[0])


_REPLAY_ADOPTION_SESSION_SQL = "SELECT content_hash FROM sessions WHERE session_id=?"


_REPLAY_ADOPTION_HEAD_SQL = "SELECT 1 FROM raw_revision_heads WHERE session_id=? LIMIT 1"


@dataclass(frozen=True, slots=True)
class PreparedRevisionAdoption:
    session_id: str | None
    adoptable: bool
    deferred_receipts: tuple[RevisionApplicationReceipt, ...] = ()


class RevisionReplayAdoptionRead(Protocol):
    """The original session/hash and head evidence used by one adoption law."""

    def adoption_session_hash(self, session_id: str) -> tuple[object, ...] | None: ...

    def adoption_session_is_governed(self, session_id: str) -> bool: ...

    def adoption_raw_revision(self, logical_source_key: str, raw_id: str) -> tuple[str, int]: ...


def prepare_raw_revision_replay_adoption(
    read: RevisionReplayAdoptionRead,
    sessions: Sequence[ParsedSession],
    *,
    logical_source_key: str,
    raw_ids: Sequence[str],
) -> PreparedRevisionAdoption:
    from polylogue.sources.dispatch import merge_parsed_session_chunks

    aggregate = [sessions[0]] if len(sessions) == 1 else merge_parsed_session_chunks(sessions)
    if len(aggregate) != 1:
        return PreparedRevisionAdoption(None, False)
    session = aggregate[0]
    session_id = str(make_session_id(session.source_name, session.provider_session_id))
    row = read.adoption_session_hash(session_id)
    if row is None:
        return PreparedRevisionAdoption(session_id, True)
    if read.adoption_session_is_governed(session_id):
        return PreparedRevisionAdoption(session_id, True)
    existing_hash = row[0]
    existing_hex = existing_hash.hex() if isinstance(existing_hash, bytes) else str(existing_hash or "")
    if existing_hex == _carried_session_content_hash(session):
        return PreparedRevisionAdoption(session_id, True)
    receipts: list[RevisionApplicationReceipt] = []
    for raw_id in raw_ids:
        check_compute_cancelled()
        revision, generation = read.adoption_raw_revision(logical_source_key, raw_id)
        receipts.append(
            RevisionApplicationReceipt(
                raw_id=raw_id,
                session_id=session_id,
                logical_source_key=logical_source_key,
                source_revision=revision,
                acquisition_generation=generation,
                decision=ApplicationDecision.DEFERRED,
                accepted_raw_id=None,
                accepted_source_revision=None,
                accepted_content_hash=None,
                detail="ordinary_replay:incomparable_existing_index_state",
            )
        )
    return PreparedRevisionAdoption(session_id, False, tuple(receipts))


_REPLAY_ADOPTION_RAW_REVISION_SQL = """
    SELECT COALESCE(r.source_revision, m.source_revision),
           COALESCE(r.acquisition_generation, m.acquisition_generation, 0)
    FROM raw_sessions AS r
    LEFT JOIN raw_session_memberships AS m
      ON m.raw_id = r.raw_id AND m.logical_source_key = ?
    WHERE r.raw_id = ?
"""


def revision_replay_frontier(
    existing_head: tuple[object, ...] | None,
    aggregate: ParsedSession,
) -> tuple[str, int | None]:
    """Use the publisher's frontier law and settle its semantic comparison."""
    if existing_head is None or str(existing_head[4]) != "semantic":
        return "byte", None
    projection = session_revision_projection(aggregate)
    try:
        frontier = len(projection.message_hashes) + len(projection.event_hashes) + len(projection.attachment_identities)
    except BaseException as primary:
        try:
            projection.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup("revision frontier and comparison cleanup failed", [primary, cleanup]) from None
        raise
    projection.close()
    return "semantic", frontier


def _authorize_selected_full_replacement(
    seal: PreparedIndexMutation,
    plan: RevisionReplayPlan,
    candidates: Mapping[str, RevisionCandidate],
    *,
    existing_head: tuple[object, ...] | None,
    session_id: str,
    content_hash: bytes,
) -> FullRevisionReplacementAuthorization | None:
    """Bind current selected Source FULL evidence to one exact original head."""
    if existing_head is None or str(existing_head[4]) != "byte" or not plan.accepted_raw_ids:
        return None
    baseline = candidates[plan.accepted_raw_ids[0]]
    accepted = candidates[plan.accepted_raw_ids[-1]]
    if (
        baseline.kind is not RawRevisionKind.FULL
        or baseline.authority is not RawRevisionAuthority.BYTE_PROVEN
        or baseline.logical_source_key != plan.logical_source_key
        or baseline.raw_id == str(existing_head[1])
        or not any(
            application.raw_id == baseline.raw_id and application.decision is ApplicationDecision.SELECTED_BASELINE
            for application in plan.applications
        )
    ):
        return None
    old = candidates.get(str(existing_head[1]))
    if old is None or old.source_revision != str(existing_head[2]):
        return None
    current_order = prepared_raw_export_order(seal, baseline.raw_id)
    previous_order = prepared_raw_export_order(seal, old.raw_id)
    if current_order is None or previous_order is None or current_order <= previous_order:
        return None
    # Source selected this unique proven baseline; receipt order supplies actual
    # acquisition freshness, independently of the byte extent of either revision.
    return FullRevisionReplacementAuthorization(
        logical_source_key=plan.logical_source_key,
        previous_head=existing_head,
        full_raw_id=baseline.raw_id,
        full_source_revision=baseline.source_revision,
        accepted_raw_id=accepted.raw_id,
        accepted_source_revision=accepted.source_revision,
        append_end_offset=accepted.append_end_offset,
        acquisition_generation=accepted.acquisition_generation,
        session_id=session_id,
        content_hash=content_hash,
        byte_length=accepted.append_end_offset or accepted.blob_size,
    )


def revision_replay_application_receipts(
    plan: RevisionReplayPlan,
    candidates: Mapping[str, RevisionCandidate],
    *,
    session_id: str,
    accepted_content_hash: bytes,
    accepted_frontier_kind: str,
    accepted_frontier: int | None,
    fold_authorization: FullSnapshotFoldAuthorization | None,
    full_replacement_authorization: FullRevisionReplacementAuthorization | None,
) -> Iterator[RevisionApplicationReceipt]:
    """Produce the exact ordered byte receipts for preparation and publication."""
    accepted_raw_id = plan.accepted_raw_ids[-1]
    accepted = candidates[accepted_raw_id]
    for application in plan.applications:
        check_compute_cancelled()
        candidate = candidates[application.raw_id]
        has_head = application.accepted_raw_id is not None
        yield RevisionApplicationReceipt(
            raw_id=candidate.raw_id,
            session_id=session_id,
            logical_source_key=plan.logical_source_key,
            source_revision=candidate.source_revision,
            acquisition_generation=accepted.acquisition_generation if has_head else candidate.acquisition_generation,
            decision=application.decision,
            accepted_raw_id=accepted_raw_id if has_head else None,
            accepted_source_revision=accepted.source_revision if has_head else None,
            accepted_content_hash=accepted_content_hash if has_head else None,
            accepted_frontier_kind=accepted_frontier_kind if has_head else None,
            accepted_frontier=(
                accepted_frontier
                if accepted_frontier_kind == "semantic"
                else accepted.append_end_offset or accepted.blob_size
            )
            if has_head
            else None,
            baseline_raw_id=candidate.baseline_raw_id,
            predecessor_raw_id=candidate.predecessor_raw_id,
            append_end_offset=accepted.append_end_offset,
            detail=application.detail,
            fold_authorization=(fold_authorization if candidate.raw_id == accepted_raw_id else None),
            full_replacement_authorization=(
                full_replacement_authorization
                if application.decision is ApplicationDecision.SELECTED_BASELINE
                else None
            ),
        )


class RevisionReplayOutcomeSourceRead(RetainedRevisionBytesRead, Protocol):
    def raw_revision_authority(self, raw_id: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class PreparedRevisionReplayOutcome:
    """The preceding byte publisher's exact selected Index outcome."""

    plan: RevisionReplayPlan
    candidates: Mapping[str, RevisionCandidate]
    existing_head: tuple[object, ...] | None
    retires_existing_head: bool
    suppressed: bool
    application_receipts: tuple[RevisionApplicationReceipt, ...]
    effective_head: tuple[object, ...] | None
    effective_session_revision: tuple[object, ...] | None
    fold_authorization: FullSnapshotFoldAuthorization | None
    accepted_frontier_kind: str | None
    accepted_frontier: int | None


def revision_replay_terminal_raw_ids(plan: RevisionReplayPlan) -> tuple[str, ...]:
    terminal = {
        application.raw_id
        for application in plan.applications
        if application.decision
        in {
            ApplicationDecision.SELECTED_BASELINE,
            ApplicationDecision.APPLIED_APPEND,
            ApplicationDecision.SUPERSEDED,
        }
    }
    # A superseded full is decided by this replay without being accepted; the
    # Index records its application, so its parse is acknowledged with the rest.
    ordered = (*plan.accepted_raw_ids, *(application.raw_id for application in plan.applications))
    return tuple(dict.fromkeys(raw_id for raw_id in ordered if raw_id in terminal))


def prepare_revision_replay_outcome(
    seal: PreparedIndexMutation,
    source_read: RevisionReplayOutcomeSourceRead,
    plan: RevisionReplayPlan,
    adoption: PreparedRevisionAdoption,
    *,
    aggregate_session: ParsedSession,
    aggregate_content_hash: bytes,
    prepared_write: PreparedSessionWrite,
) -> PreparedRevisionReplayOutcome:
    """Prepare the actual ordered byte outcome before writer admission.

    Original Source candidates and Index head inputs remain with this seal.
    The existing frontier, fold proof, receipt and CAS bodies decide the
    result; deferred adoption preserves the original head without fold work.
    """
    if not plan.accepted_raw_ids or len(aggregate_content_hash) != 32:
        raise ValueError("byte replay outcome requires its accepted chain and exact aggregate digest")
    candidates = {
        candidate.raw_id: candidate for candidate in prepared_raw_revision_candidates(seal, plan.logical_source_key)
    }
    seal.before_index_input(
        "raw_revision_heads",
        (
            "session_id",
            "accepted_raw_id",
            "accepted_source_revision",
            "accepted_content_hash",
            "accepted_frontier_kind",
            "accepted_frontier",
            "acquisition_generation",
            "append_end_offset",
        ),
        "SELECT rowid FROM raw_revision_heads WHERE logical_source_key=?",
        (plan.logical_source_key,),
    )
    with seal.original_rows(
        "index",
        "SELECT session_id,accepted_raw_id,accepted_source_revision,accepted_content_hash,"
        "accepted_frontier_kind,accepted_frontier,acquisition_generation,append_end_offset "
        "FROM raw_revision_heads WHERE logical_source_key=?",
        (plan.logical_source_key,),
    ) as rows:
        row = rows.fetchone()
    existing_head = None if row is None else tuple(row)
    original_session_revision = None
    if existing_head is not None:
        seal.before_index_input(
            "sessions",
            ("raw_id", "content_hash"),
            "SELECT rowid FROM sessions WHERE session_id=?",
            (existing_head[0],),
        )
        with seal.original_rows(
            "index", "SELECT raw_id,content_hash FROM sessions WHERE session_id=?", (existing_head[0],)
        ) as rows:
            session_row = rows.fetchone()
        original_session_revision = None if session_row is None else tuple(session_row)
    if not adoption.adoptable:
        effective = existing_head
        for receipt in adoption.deferred_receipts:
            effective = prepare_revision_application_head(effective, receipt)
        return PreparedRevisionReplayOutcome(
            plan,
            candidates,
            existing_head,
            False,
            False,
            adoption.deferred_receipts,
            effective,
            original_session_revision,
            None,
            None,
            None,
        )
    session_id = str(make_session_id(aggregate_session.source_name, aggregate_session.provider_session_id))
    if adoption.session_id != session_id:
        raise PreparedSessionWriteRefusedError("byte replay adoption names another prepared session")
    from polylogue.storage.sqlite.archive_tiers.session_suppression import _reader_suppresses

    # Use the existing canonical suppression body against this seal's actual
    # original User observer. The publisher consumes this no-write outcome;
    # adoption alone never proves that a tombstoned session will be written.
    suppressed = _reader_suppresses(seal.observer("user"), session_id)
    if suppressed:
        return PreparedRevisionReplayOutcome(
            plan,
            candidates,
            existing_head,
            False,
            True,
            (),
            existing_head,
            original_session_revision,
            None,
            None,
            None,
        )
    frontier_kind, frontier = revision_replay_frontier(existing_head, aggregate_session)
    retires_head = (
        existing_head is not None
        and frontier_kind == "semantic"
        and frontier is not None
        and str(existing_head[4]) == "semantic"
        and source_read.raw_revision_authority(str(existing_head[1])) == "quarantined"
    )
    effective = None if retires_head else existing_head
    reparse_receipt = prepared_write.reparse_receipt
    if reparse_receipt is not None:
        if reparse_receipt.accepted_content_hash != aggregate_content_hash or reparse_receipt.session_id != session_id:
            raise PreparedSessionWriteRefusedError("byte outcome names another prepared reparse correction")
        if not retires_head:
            effective = prepare_revision_application_head(effective, reparse_receipt)
    accepted = candidates[plan.accepted_raw_ids[-1]]
    fold = (
        _authorize_full_snapshot_fold(
            source_read,
            existing_head=effective,
            full_candidate=accepted,
            candidates=candidates,
        )
        if effective is not None and frontier_kind == "byte"
        else None
    )
    receipts = tuple(
        revision_replay_application_receipts(
            plan,
            candidates,
            session_id=session_id,
            accepted_content_hash=aggregate_content_hash,
            accepted_frontier_kind=frontier_kind,
            accepted_frontier=frontier,
            fold_authorization=fold,
            full_replacement_authorization=_authorize_selected_full_replacement(
                seal,
                plan,
                candidates,
                existing_head=effective,
                session_id=session_id,
                content_hash=aggregate_content_hash,
            ),
        )
    )
    for receipt in receipts:
        effective = prepare_revision_application_head(effective, receipt)
    session_values = session_revision_row_values(plan.accepted_raw_ids[-1], aggregate_content_hash)
    return PreparedRevisionReplayOutcome(
        plan,
        candidates,
        existing_head,
        retires_head,
        False,
        receipts,
        effective,
        (session_values["raw_id"], session_values["content_hash"]),
        fold,
        frontier_kind,
        frontier,
    )


class _MembershipDecisionProducer(SourceArtifactProducer, SourceRawStateProducer, Protocol):
    """The finite Source writes and reads used by cohort finalization."""

    def binding_transaction(self, *, manage_transaction: bool) -> AbstractContextManager[object]: ...

    def binding_literal(self, value: object) -> tuple[str, tuple[object, ...]]: ...

    def membership_decision_write(
        self,
        raw_id: str,
        logical_source_key: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> None: ...

    def read_binding(self, raw_id: str, sql: str, parameters: tuple[object, ...]) -> tuple[object, ...] | None: ...


#: Decisions that hand a raw's identity to membership governance.
_MEMBERSHIP_TERMINAL_DECISIONS = frozenset(
    {
        MembershipDecision.APPLIED,
        MembershipDecision.SUPERSEDED_PREFIX,
        MembershipDecision.SUPERSEDED_EQUIVALENT,
        MembershipDecision.SUPERSEDED_BY_WINNER,
    }
)

_MEMBERSHIP_COMPLETION_SQL = """
    SELECT c.status = 'complete'
       AND NOT EXISTS (
           SELECT 1 FROM raw_session_memberships AS m
           WHERE m.raw_id = c.raw_id
             AND (m.decision IS NULL OR m.decision IN ('ambiguous', 'deferred'))
       )
    FROM raw_membership_census AS c WHERE c.raw_id = ?
"""


def _record_membership_decisions(
    producer: _MembershipDecisionProducer,
    logical_source_key: str,
    classification: MembershipClassification,
    decisions: Mapping[str, MembershipDecision],
    *,
    decided_at_ms: int,
    manage_transaction: bool,
    projections: Mapping[str, SessionRevisionProjection] | None = None,
) -> Iterator[tuple[str, bool]]:
    """Use one canonical decision builder and completion read on both hosts.

    The caller supplies the actual winner/yield decisions. Completion is read
    only after every decision has been written, including earlier staged
    updates in a prepared Source producer.
    """
    updated_raw_ids: list[str] = []
    with producer.binding_transaction(manage_transaction=manage_transaction):
        for raw_id, decision in decisions.items():
            check_compute_cancelled()
            prior = producer.read_binding(
                raw_id,
                "SELECT revision_authority,decision FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?",
                (raw_id, logical_source_key),
            )
            if prior is None:
                # A byte-governed head compared with the cohort keeps its own
                # binding; it has no membership row for this decision.
                continue
            if (
                decision is MembershipDecision.DEFERRED
                and str(prior[0]) == "byte_proven"
                and str(prior[1]) in {settled.value for settled in _MEMBERSHIP_TERMINAL_DECISIONS}
            ):
                # A skipped derived attempt cannot retract settled Source
                # evidence or issue a new terminal acknowledgement for it.
                continue
            # The decision is about the projection classified now. Enrichment
            # evidence admitted after the census (a renamed index title) moves
            # that projection; record the revision this decision is about.
            projection = None if projections is None else projections.get(raw_id)
            assignments: dict[str, object] = {
                "decision": decision.value,
                "decided_at_ms": decided_at_ms,
                "revision_authority": "quarantined"
                if decision in {MembershipDecision.AMBIGUOUS, MembershipDecision.DEFERRED}
                else "byte_proven",
                "acquisition_generation": classification.accepted_raw_ids.index(raw_id)
                if raw_id in classification.accepted_raw_ids
                else 0,
            }
            if projection is not None:
                assignments["source_revision"] = projection.session_hash.hex()
                assignments["normalized_content_hash"] = projection.session_hash
                assignments["message_count"] = len(projection.message_hashes)
            # Placeholders bind in statement order: assignments, then the key.
            rendered = tuple(
                producer.binding_literal(value) for value in (*assignments.values(), raw_id, logical_source_key)
            )
            expressions = tuple(expression for expression, _ in rendered)
            parameters = tuple(value for _, operands in rendered for value in operands)
            set_clause = ",".join(
                f"{column}={expression}" for column, expression in zip(assignments, expressions, strict=False)
            )
            sql = (
                f"UPDATE raw_session_memberships SET {set_clause} "
                f"WHERE raw_id={expressions[-2]} AND logical_source_key={expressions[-1]}"
            )
            producer.membership_decision_write(raw_id, logical_source_key, sql, parameters)
            updated_raw_ids.append(raw_id)
    for raw_id in updated_raw_ids:
        check_compute_cancelled()
        complete = producer.read_binding(raw_id, _MEMBERSHIP_COMPLETION_SQL, (raw_id,))
        yield raw_id, complete is not None and bool(complete[0])


def prepare_membership_classification_source(
    seal: PreparedIndexMutation,
    logical_source_key: str,
    classification: MembershipClassification,
    *,
    decisions: Mapping[str, MembershipDecision],
    decided_at_ms: int,
    projections: Mapping[str, SessionRevisionProjection] | None = None,
    head_plan: MembershipHeadPlan | None = None,
) -> None:
    """Stage the canonical Source outcome after the parent resolves its head.

    The parent owns the original read and Source producer windows. Supplied
    decisions include the actual yield-to-existing-head outcome; this function
    neither chooses a winner nor authorizes Index publication. A head-plan
    ``conflict`` stages its retryable evidence on every deferred member and
    leaves the reaffirmed head's byte binding in place.
    """
    conflict = head_plan.conflict if head_plan is not None else None
    producer = _PreparedSourceProducer(seal)
    for raw_id, complete in _record_membership_decisions(
        producer,
        logical_source_key,
        classification,
        decisions,
        decided_at_ms=decided_at_ms,
        manage_transaction=False,
        projections=projections,
    ):
        check_compute_cancelled()
        if conflict is not None and head_plan is not None and raw_id in head_plan.deferred_raw_ids:
            _prepare_raw_parse_failure(
                producer,
                raw_id,
                error=conflict,
                kind=RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER,
                fails_observation=head_plan.conflict_fails_observation,
            )
            continue
        if conflict is None and decisions[raw_id] in _MEMBERSHIP_TERMINAL_DECISIONS:
            _release_membership_decided_byte_binding(seal, raw_id, logical_source_key)
        if complete:
            provider, _, _, _, _ = _raw_revision_descriptor_from_row(
                producer.read_binding(raw_id, _RAW_REVISION_DESCRIPTOR_SQL, (raw_id,)),
                raw_id,
            )
            _supersede_deferred_cas_with_producer(
                producer,
                raw_id,
                provider=provider,
                manage_transaction=False,
            )
            state = _raw_parse_success_state(provider)
        else:
            state = RawSessionStateUpdate(parsed_at=None, parse_error=None)
        _apply_source_raw_state_update(producer, raw_id, state=state, manage_transaction=False)
        check_compute_cancelled()


@dataclass(frozen=True, slots=True)
class MembershipHeadPlan:
    """The existing head decision, captured before publication begins.

    ``conflict`` is a retryable refusal to move the accepted head this pass.
    The cohort then publishes nothing; ``deferred_raw_ids`` are the members
    that carry the typed retry evidence instead of a decision. A refused
    replacement fails its observation (``conflict_fails_observation``); a
    member that forks from a byte-governed head completes its observation and
    waits, deferred, for a later one that can order the revisions. A head the
    cohort compared keeps its accepted content and byte binding; when it
    carries a membership row (a converted full), that row is decided
    ``applied`` (``reaffirmed_head_raw_id``).
    """

    existing_head: tuple[object, ...] | None
    yield_to_head_raw_id: str | None
    restored_head: tuple[object, ...] | None
    conflict: MembershipReplayConflictError | None = None
    deferred_raw_ids: tuple[str, ...] = ()
    conflict_fails_observation: bool = True
    reaffirmed_head_raw_id: str | None = None


def membership_decisions_for_head_plan(
    classification: MembershipClassification,
    head_plan: MembershipHeadPlan,
    *,
    suppressed: bool,
) -> dict[str, MembershipDecision]:
    """Use the actual winner, head yield, or no-write outcome on both tiers."""
    if head_plan.conflict is not None:
        conflict_decisions = dict.fromkeys(head_plan.deferred_raw_ids, MembershipDecision.DEFERRED)
        if head_plan.reaffirmed_head_raw_id is not None:
            conflict_decisions[head_plan.reaffirmed_head_raw_id] = MembershipDecision.APPLIED
        return conflict_decisions
    decisions = membership_decisions_for_classification(classification)
    if head_plan.yield_to_head_raw_id is not None:
        for raw_id in (
            *classification.accepted_raw_ids,
            *classification.equivalent_raw_ids,
            *classification.ambiguous_raw_ids,
            *classification.superseded_raw_ids,
        ):
            decisions[raw_id] = MembershipDecision.SUPERSEDED_EQUIVALENT
    elif suppressed:
        for raw_id in (*classification.accepted_raw_ids, *classification.equivalent_raw_ids):
            decisions[raw_id] = MembershipDecision.DEFERRED
    return decisions


_MEMBERSHIP_HEAD_INPUT_FIELDS = (
    "accepted_raw_id",
    "accepted_content_hash",
    "accepted_frontier_kind",
    "session_id",
    "accepted_frontier",
)


def membership_head_input_from_revision_head(head: tuple[object, ...] | None) -> tuple[object, ...] | None:
    """Project the publisher's exact row into the existing membership inputs."""
    if head is None:
        return None
    return tuple(head[REVISION_HEAD_ROW_FIELDS.index(field)] for field in _MEMBERSHIP_HEAD_INPUT_FIELDS)


def prepare_membership_head_plan(
    index: sqlite3.Connection,
    source_read: MembershipHeadSourceRead,
    logical_source_key: str,
    classification: MembershipClassification,
    *,
    before_input: BeforeIndexInput | None = None,
) -> MembershipHeadPlan:
    """Load original Index inputs, then use the one canonical head reduction."""
    if (
        not classification.accepted_raw_ids
        and not classification.ambiguous_raw_ids
        and not classification.superseded_raw_ids
    ):
        return MembershipHeadPlan(None, None, None)
    if before_input is not None:
        before_input(
            "raw_revision_heads",
            ("accepted_raw_id", "accepted_content_hash", "accepted_frontier_kind", "session_id", "accepted_frontier"),
            "SELECT rowid FROM raw_revision_heads WHERE logical_source_key=?",
            (logical_source_key,),
        )
    with _governance_read_rows(
        index,
        f"SELECT {', '.join(_MEMBERSHIP_HEAD_INPUT_FIELDS)} FROM raw_revision_heads WHERE logical_source_key=?",
        (logical_source_key,),
    ) as rows:
        head_row = rows.fetchone()
    if head_row is None:
        return MembershipHeadPlan(None, None, None)
    existing_head = tuple(head_row)
    session_id = str(existing_head[3])
    if before_input is not None:
        before_input(
            "sessions",
            ("raw_id", "content_hash"),
            "SELECT rowid FROM sessions WHERE session_id=?",
            (session_id,),
        )
    with _governance_read_rows(
        index, "SELECT raw_id, content_hash FROM sessions WHERE session_id=?", (session_id,)
    ) as rows:
        persisted_session = rows.fetchone()
    return prepare_membership_head_plan_from_inputs(
        source_read,
        logical_source_key,
        classification,
        existing_head=existing_head,
        persisted_session=None if persisted_session is None else tuple(persisted_session),
    )


def _persisted_owner_raw(persisted_session: Sequence[object] | None) -> str | None:
    """The raw that owns a persisted session row, if any.

    ``sessions.raw_id`` is nullable: a row without one makes no ownership
    claim, so it cannot be "owned by a raw outside the cohort". Coercing the
    NULL to text produced the raw ID ``'None'`` and refused a cohort's own
    replacement.
    """
    if persisted_session is None or persisted_session[0] is None:
        return None
    return str(persisted_session[0])


def prepare_membership_head_plan_from_inputs(
    source_read: MembershipHeadSourceRead,
    logical_source_key: str,
    classification: MembershipClassification,
    *,
    existing_head: tuple[object, ...] | None,
    persisted_session: tuple[object, ...] | None,
) -> MembershipHeadPlan:
    """Reduce selected head/session inputs using their actual Source authority.

    The original Index loader and the retained parent's ordered prepared
    publication outcomes share this decision body. The latter must supply
    the preceding publisher's exact row outcomes, including its refusal or
    no-op; a byte plan alone does not establish an effective head.
    """
    if existing_head is None:
        return MembershipHeadPlan(None, None, None)
    if not classification.accepted_raw_ids:
        head_raw_id = str(existing_head[0])
        if head_raw_id in classification.ambiguous_raw_ids and source_read.raw_revision_authority(head_raw_id) not in (
            None,
            "quarantined",
        ):
            # Membership cannot order these members against a byte-governed
            # head, and it never re-decides that head. The fork is a deferred
            # frontier on the members, not ambiguity debt on the head.
            return MembershipHeadPlan(
                existing_head,
                None,
                None,
                conflict=MembershipReplayConflictError(
                    "membership members fork from a byte-governed head: "
                    f"logical_source_key={logical_source_key!r} existing_head(raw_id={head_raw_id!r})"
                ),
                deferred_raw_ids=tuple(raw_id for raw_id in classification.ambiguous_raw_ids if raw_id != head_raw_id),
                conflict_fails_observation=False,
                reaffirmed_head_raw_id=head_raw_id,
            )
        return MembershipHeadPlan(None, None, None)
    accepted_raw_id = classification.accepted_raw_ids[-1]
    classified_raw_ids = frozenset(
        (*classification.accepted_raw_ids, *classification.equivalent_raw_ids, *classification.superseded_raw_ids)
    )
    existing_raw_id = str(existing_head[0])
    session_id = str(existing_head[3])
    chain_head_authority = (
        source_read.raw_revision_authority(existing_raw_id) if existing_raw_id not in classified_raw_ids else None
    )
    # Byte-governed output outside this membership cohort outranks its
    # quarantined captures. A scalar semantic frontier cannot prove dominance.
    if existing_raw_id not in classified_raw_ids and chain_head_authority not in (None, "quarantined"):
        return MembershipHeadPlan(existing_head, existing_raw_id, None)
    persisted_raw = _persisted_owner_raw(persisted_session)
    persisted_head_authority = (
        source_read.raw_revision_authority(persisted_raw)
        if persisted_raw is not None and persisted_raw not in classified_raw_ids
        else None
    )
    if persisted_head_authority not in (None, "quarantined"):
        assert persisted_raw is not None and persisted_session is not None
        revision = source_read.membership_head_revision(persisted_raw)
        if revision is None or revision[0] is None:
            raise RuntimeError("persisted byte-governed session lacks revision evidence")
        restored = (
            persisted_raw,
            str(revision[0]),
            persisted_session[1],
            int(cast(Any, revision[2] or revision[3])),
            int(cast(Any, revision[1] or 0)),
            revision[2],
            logical_source_key,
        )
        return MembershipHeadPlan(existing_head, persisted_raw, restored)
    # A refusal is a decided, retryable outcome for the cohort's own members.
    # The head and the persisted session owner keep their bindings.
    deferred_raw_ids = tuple(
        raw_id
        for raw_id in dict.fromkeys(
            (
                *classification.accepted_raw_ids,
                *classification.equivalent_raw_ids,
                *classification.ambiguous_raw_ids,
                *classification.superseded_raw_ids,
            )
        )
        if raw_id not in {existing_raw_id, persisted_raw}
    )
    if existing_raw_id not in classified_raw_ids or (
        persisted_raw is not None and persisted_raw != existing_raw_id and persisted_raw not in classified_raw_ids
    ):
        return MembershipHeadPlan(
            existing_head,
            None,
            None,
            conflict=MembershipReplayConflictError(
                "membership replay cannot retire an unrelated accepted head: "
                f"logical_source_key={logical_source_key!r} "
                f"existing_head(raw_id={existing_raw_id!r}, session_id={session_id!r}, "
                f"authority={chain_head_authority!r}) "
                f"cohort(accepted={classification.accepted_raw_ids!r}, "
                f"equivalent={classification.equivalent_raw_ids!r}, "
                f"ambiguous={classification.ambiguous_raw_ids!r}, "
                f"superseded={classification.superseded_raw_ids!r}) persisted_session_raw={persisted_raw!r}"
            ),
            deferred_raw_ids=deferred_raw_ids,
        )
    # Keeping the same head reaffirms it. Changing it requires proof that no
    # unclassified byte-append descendant still names its exact revision.
    if accepted_raw_id != existing_raw_id and source_read.membership_has_dangling_append(
        logical_source_key,
        existing_raw_id,
        classified_raw_ids,
    ):
        return MembershipHeadPlan(
            existing_head,
            None,
            None,
            conflict=MembershipReplayConflictError(
                "membership replay cannot replace a head with unresolved byte-append "
                f"evidence: logical_source_key={logical_source_key!r} existing_head(raw_id={existing_raw_id!r})"
            ),
            deferred_raw_ids=deferred_raw_ids,
            reaffirmed_head_raw_id=existing_raw_id,
        )
    return MembershipHeadPlan(existing_head, None, None)


def _apply_membership_head_plan(index: sqlite3.Connection, logical_source_key: str, plan: MembershipHeadPlan) -> None:
    if plan.restored_head is not None:
        with _governance_read_rows(
            index,
            "UPDATE raw_revision_heads SET accepted_raw_id=?, accepted_source_revision=?, accepted_content_hash=?, "
            "accepted_frontier_kind='byte', accepted_frontier=?, acquisition_generation=?, append_end_offset=? "
            "WHERE logical_source_key=?",
            plan.restored_head,
        ):
            pass
    elif plan.existing_head is not None and plan.yield_to_head_raw_id is None:
        with _governance_read_rows(
            index, "DELETE FROM raw_revision_heads WHERE logical_source_key=?", (logical_source_key,)
        ):
            pass


def _retire_superseded_membership_applications(
    conn: sqlite3.Connection, raw_id: str, logical_source_key: str, source_revision: str
) -> None:
    """Drop applications made for a membership projection that has moved.

    A membership revision is the semantic projection, which enrichment admitted
    after the census (a renamed index title) can change. Applications decided
    for the earlier projection describe no current evidence, and a rebuild from
    current evidence would not produce them.
    """
    conn.execute(
        "DELETE FROM raw_revision_applications WHERE raw_id=? AND logical_source_key=? AND source_revision!=?",
        (raw_id, logical_source_key, source_revision),
    )


def apply_prepared_membership_index(
    store: RawRevisionGovernanceHost,
    logical_source_key: str,
    classification: MembershipClassification,
    parsed_by_raw_id: Mapping[str, ParsedSession],
    projections_by_raw_id: Mapping[str, SessionRevisionProjection],
    head_plan: MembershipHeadPlan,
    *,
    decided_at_ms: int,
    preacquired_attachment_blobs: Mapping[object, tuple[bytes | None, int, str]],
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "membership_replay",
    bulk_fts: bool = False,
    bulk_build: bool = False,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    prepared_by_raw_id: Mapping[str, PreparedSessionRows] | None = None,
    prepared_required_raw_ids: frozenset[str] = frozenset(),
    prepared_write: PreparedSessionWrite | None = None,
    write_result: Callable[[ArchiveRawParsedWriteResult], None] | None = None,
) -> tuple[str | None, dict[str, MembershipDecision]]:
    """Apply the canonical Index outcome inside its original mutation scope."""
    from polylogue.storage.sqlite.reference_seal import current_index_mutation_scope

    scope = current_index_mutation_scope()
    if scope is None:
        raise RuntimeError("membership Index publication requires its actual mutation scope")
    scope.require_connection(store._conn)
    decisions = membership_decisions_for_head_plan(classification, head_plan, suppressed=False)
    if head_plan.conflict is not None or not classification.accepted_raw_ids:
        return None, decisions
    accepted_raw_id = classification.accepted_raw_ids[-1]
    accepted_session = parsed_by_raw_id[accepted_raw_id]
    attachments = preacquired_attachment_blobs
    existing_head = head_plan.existing_head
    yield_to_head_raw_id = head_plan.yield_to_head_raw_id
    if yield_to_head_raw_id is not None:
        _apply_membership_head_plan(store._conn, logical_source_key, head_plan)
        assert existing_head is not None
        session_id = str(existing_head[3])
        cohort_raw_ids = (
            *classification.accepted_raw_ids,
            *classification.equivalent_raw_ids,
            *classification.ambiguous_raw_ids,
            *classification.superseded_raw_ids,
        )
        for generation, raw_id in enumerate(cohort_raw_ids):
            projection = projections_by_raw_id[raw_id]
            _retire_superseded_membership_applications(
                store._conn, raw_id, logical_source_key, projection.session_hash.hex()
            )
            record_revision_application_sync(
                store._conn,
                RevisionApplicationReceipt(
                    raw_id=raw_id,
                    session_id=session_id,
                    logical_source_key=logical_source_key,
                    source_revision=projection.session_hash.hex(),
                    acquisition_generation=generation,
                    decision=ApplicationDecision.SUPERSEDED,
                    accepted_raw_id=None,
                    accepted_source_revision=None,
                    accepted_content_hash=None,
                    detail=f"membership:superseded_by_chain_governed_head:{yield_to_head_raw_id}",
                ),
                decided_at_ms=decided_at_ms,
            )
    else:
        index_started = time.perf_counter()
        result = _index_parsed_for_retained_raw(
            store,
            accepted_session,
            raw_id=accepted_raw_id,
            source_index=0,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=False,
            preacquired_attachment_blobs=attachments,
            finalize_raw_parse=False,
            revision_authoritative=True,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            fresh_build=fresh_build,
            fresh_build_batch=fresh_build_batch,
            defer_fts_rebuild=not bulk_build,
            prepared=(prepared_by_raw_id or {}).get(accepted_raw_id),
            prepared_required=accepted_raw_id in prepared_required_raw_ids or prepared_write is not None,
            prepared_write=prepared_write,
            content_hash=projections_by_raw_id[accepted_raw_id].session_hash.hex(),
        )
        if write_result is not None:
            write_result(result)
        if stage_timings_s is not None:
            key = f"{stage_timing_prefix}.index_parsed_write"
            stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - index_started)
        session_id = result.session_id
        if result.publication_refused:
            return session_id, membership_decisions_for_head_plan(classification, head_plan, suppressed=True)
        _apply_membership_head_plan(store._conn, logical_source_key, head_plan)
        if not bulk_build:
            repair_message_fts_index_sync(store._conn, [session_id])
        assert_session_fts_exact_sync(store._conn, session_id, bulk_build=bulk_build)
        with _governance_read_rows(
            store._conn, "SELECT content_hash FROM sessions WHERE session_id=?", (session_id,)
        ) as rows:
            stored = rows.fetchone()
        if stored is None or not isinstance(stored[0], bytes):
            raise RuntimeError("accepted membership did not produce a hashed session")
        accepted_projection = projections_by_raw_id[accepted_raw_id]
        semantic_frontier = (
            len(accepted_projection.message_hashes)
            + len(accepted_projection.event_hashes)
            + len(accepted_projection.attachment_identities)
        )
        cohort_raw_ids = (
            *classification.accepted_raw_ids,
            *classification.equivalent_raw_ids,
            *classification.ambiguous_raw_ids,
            *classification.superseded_raw_ids,
        )
        for generation, raw_id in enumerate(cohort_raw_ids):
            projection = projections_by_raw_id[raw_id]
            _retire_superseded_membership_applications(
                store._conn, raw_id, logical_source_key, projection.session_hash.hex()
            )
            decision = decisions.get(raw_id, MembershipDecision.APPLIED)
            is_ambiguous = decision is MembershipDecision.AMBIGUOUS
            record_revision_application_sync(
                store._conn,
                RevisionApplicationReceipt(
                    raw_id=raw_id,
                    session_id=session_id,
                    logical_source_key=logical_source_key,
                    source_revision=projection.session_hash.hex(),
                    acquisition_generation=generation,
                    decision=_application_decision_for(decision),
                    accepted_raw_id=accepted_raw_id if not is_ambiguous else None,
                    accepted_source_revision=(accepted_projection.session_hash.hex() if not is_ambiguous else None),
                    accepted_content_hash=stored[0] if not is_ambiguous else None,
                    accepted_frontier_kind="semantic" if not is_ambiguous else None,
                    accepted_frontier=semantic_frontier if not is_ambiguous else None,
                    detail=f"membership:{decision}",
                ),
                decided_at_ms=decided_at_ms,
            )
    if yield_to_head_raw_id is None:
        decisions[accepted_raw_id] = MembershipDecision.APPLIED
    return session_id, decisions


def _record_raw_failure_evidence(
    producer: SourceArtifactProducer,
    raw_id: str,
    *,
    provider: Provider,
    source_path: str,
    source_index: int,
    acquired_at_ms: int,
    kind: RawFailureEvidenceKind,
    manage_transaction: bool = True,
) -> None:
    """Share the canonical failure classification and artifact write law."""
    from hashlib import sha256

    from polylogue.core.sources import origin_from_provider
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact

    artifact_id = "raw-failure:" + sha256(f"{raw_id}:{kind.value}".encode()).hexdigest()
    validation_failed = producer.artifact_validation_failed(raw_id)
    outcome_code = (
        "corrupt_input"
        if kind
        in {
            RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
            RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE,
        }
        else kind.value
    )
    _upsert_raw_artifact(
        producer,
        raw_id,
        ArchiveSourceArtifact(
            artifact_id=artifact_id,
            origin=origin_from_provider(provider),
            source_path=source_path,
            source_index=source_index,
            artifact_kind=kind.value,
            classification_reason=raw_failure_classification_reason(
                diagnostic=None,
                evidence_ref=None,
                outcome_code=outcome_code,
                remediation=None,
                retryable=False,
                trusted_validation_failure=(
                    validation_failed
                    and kind
                    in {
                        RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
                        RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE,
                    }
                    and outcome_code == "corrupt_input"
                ),
            ),
            support_status=kind.support_status,
            parse_as_session=kind.lifecycle == "deferred",
            schema_eligible=kind.lifecycle == "deferred",
            first_observed_at_ms=acquired_at_ms,
            last_observed_at_ms=acquired_at_ms,
        ),
        manage_transaction=manage_transaction,
    )


def _supersede_deferred_cas_with_producer(
    producer: _MembershipDecisionProducer,
    raw_id: str,
    *,
    provider: Provider,
    manage_transaction: bool,
) -> None:
    """Terminalize deferred CAS evidence once its attempt has resolved.

    ``raw_artifacts`` stores the latest observation for a source coordinate,
    not an attempt history. Replace only an exact-coordinate deferred CAS
    observation, so a neighboring artifact cannot be consumed or cleared by
    this raw's outcome.
    """
    row = producer.read_binding(
        raw_id,
        """
        SELECT origin, source_path, source_index
        FROM raw_sessions
        WHERE raw_id = ?
        """,
        (raw_id,),
    )
    if row is None:
        return
    origin, source_path, source_index = row
    deferred = producer.read_binding(
        raw_id,
        """
        SELECT 1
        FROM raw_artifacts
        WHERE raw_id = ?
          AND origin IS ?
          AND source_path IS ?
          AND source_index IS ?
          AND artifact_kind = ?
          AND support_status = ?
        LIMIT 1
        """,
        (
            raw_id,
            origin,
            source_path,
            source_index,
            RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
            RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
        ),
    )
    if deferred is None:
        return
    _record_raw_failure_evidence(
        producer,
        raw_id,
        provider=provider,
        source_path=str(source_path or raw_id),
        source_index=_source_integer(source_index) if source_index is not None else 0,
        acquired_at_ms=int(time.time() * 1000),
        kind=RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER,
        manage_transaction=manage_transaction,
    )


def _retire_raw_failure_evidence_with_producer(
    producer: _MembershipDecisionProducer,
    raw_id: str,
    *,
    manage_transaction: bool,
) -> None:
    row = producer.read_binding(
        raw_id,
        "SELECT origin,source_path,source_index FROM raw_sessions WHERE raw_id=?",
        (raw_id,),
    )
    if row is None:
        return
    retired_kinds = sorted(
        kind.value
        for kind in RawFailureEvidenceKind
        if kind is not RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER
    )
    placeholders = ",".join("?" for _ in retired_kinds)
    after: str | None = None
    with producer.artifact_transaction(manage_transaction):
        while True:
            check_compute_cancelled()
            predicate = "" if after is None else " AND artifact_id > ?"
            selected = producer.read_binding(
                raw_id,
                "SELECT artifact_id FROM raw_artifacts WHERE raw_id=? AND origin IS ? "
                "AND source_path IS ? AND source_index IS ? "
                f"AND artifact_kind IN ({placeholders}){predicate} ORDER BY artifact_id LIMIT 1",
                (raw_id, *row, *retired_kinds, *((after,) if after is not None else ())),
            )
            if selected is None:
                break
            after = str(selected[0])
            values = (
                RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
                RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.support_status.value,
                raw_failure_classification_reason(
                    diagnostic=None,
                    evidence_ref=None,
                    outcome_code="failure_attempt_replaced",
                    remediation="inspect the current parser failure before retrying",
                    retryable=False,
                    trusted_validation_failure=False,
                ),
                after,
            )
            operands = tuple(producer.binding_literal(value) for value in values)
            expressions = tuple(expression for expression, _ in operands)
            parameters = tuple(value for _, params in operands for value in params)
            with producer.artifact_write(
                f"UPDATE raw_artifacts SET artifact_kind={expressions[0]},support_status={expressions[1]},"
                f"classification_reason={expressions[2]},parse_as_session=0,schema_eligible=0 "
                f"WHERE artifact_id={expressions[3]}",
                parameters,
                after,
                allocation=False,
            ):
                pass


def _prepare_raw_parse_success(
    producer: _MembershipDecisionProducer,
    raw_id: str,
    *,
    provider: Provider,
) -> None:
    check_compute_cancelled()
    _supersede_deferred_cas_with_producer(producer, raw_id, provider=provider, manage_transaction=False)
    _apply_source_raw_state_update(producer, raw_id, state=_raw_parse_success_state(provider), manage_transaction=False)
    check_compute_cancelled()


def _prepare_raw_parse_failure(
    producer: _MembershipDecisionProducer,
    raw_id: str,
    *,
    error: BaseException,
    kind: RawFailureEvidenceKind,
    fails_observation: bool = True,
) -> None:
    """Stage typed failure evidence and the parse failure for one raw.

    The prepared counterpart of ``mark_raw_parse_failed``: the latest typed
    evidence replaces an earlier attempt's, and ``parse_error`` stays a
    diagnostic beside it. Deferred evidence that does not fail its
    observation leaves the raw unparsed without a parse error.
    """
    check_compute_cancelled()
    provider, _, _, _, _ = _raw_revision_descriptor_from_row(
        producer.read_binding(raw_id, _RAW_REVISION_DESCRIPTOR_SQL, (raw_id,)), raw_id
    )
    row = producer.read_binding(
        raw_id, "SELECT source_path,source_index,acquired_at_ms FROM raw_sessions WHERE raw_id=?", (raw_id,)
    )
    if row is None:
        raise RuntimeError(f"failure evidence names an absent raw: {raw_id}")
    _retire_raw_failure_evidence_with_producer(producer, raw_id, manage_transaction=False)
    _record_raw_failure_evidence(
        producer,
        raw_id,
        provider=provider,
        source_path=str(row[0] or raw_id),
        source_index=_source_integer(row[1]) if row[1] is not None else 0,
        acquired_at_ms=int(cast(Any, row[2]) or int(time.time() * 1000)),
        kind=kind,
        manage_transaction=False,
    )
    _apply_source_raw_state_update(
        producer,
        raw_id,
        state=(
            _raw_parse_failure_state(provider, error)
            if fails_observation
            else RawSessionStateUpdate(parsed_at=None, parse_error=None)
        ),
        manage_transaction=False,
    )
    check_compute_cancelled()


def prepare_raw_parse_success(seal: PreparedIndexMutation, raw_id: str, *, provider: Provider) -> None:
    """Stage the canonical parse success after its selected Index outcome."""
    _prepare_raw_parse_success(_PreparedSourceProducer(seal), raw_id, provider=provider)


if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.storage.blob_publication import (
        ArchiveBlobPublisher,
        BlobPublicationSourceRead,
        PreparedBlobPublicationClaim,
    )
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.reference_seal import KnownTierCell, KnownTierMutationPermit, PreparedIndexMutation

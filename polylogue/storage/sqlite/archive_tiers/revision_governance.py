"""Raw revision & membership governance: the authority over which raw bytes win.

Writer module: index, source.
Twin-write contract: raw-membership-classification.

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
  (``classify_raw_revision_cohort``, ``raw_revision_replay_plan``,
  ``_raw_revision_candidates``, ``_authorize_full_snapshot_fold``);
- whether a parsed session is allowed to overwrite ``sessions`` at all, given
  the raw's recorded membership decision and revision-authority state
  (``_write_parsed_precedence_result``, ``apply_raw_revision_replay``,
  ``apply_raw_membership_classification``);
- how a raw's membership in a revision cohort is computed, persisted, and
  queried (``replace_raw_membership_census``, ``raw_membership_*``);
- bookkeeping for a raw's own parse lifecycle
  (``finalize_raw_parse_state``, ``mark_raw_parse_failed/succeeded``) and the
  narrow raw-write paths that hand a parsed session to this authority
  (``write_raw_and_parsed*``, ``write_parsed_for_retained_raw*``,
  ``write_raw_blob_and_parsed*``, ``_index_parsed_for_retained_raw``).

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

Every function here takes ``store: RawRevisionGovernanceHost`` as its first
argument instead of being a method on ``ArchiveStore``. ``ArchiveStore`` owns
a persistent lazy ``source.db`` connection plus in-flight write-batch state
(pending blob receipts, pending raw-parse-state flushes, the blob publisher)
as instance attributes; the governance surface needs a subset of that state
but must not gain silent access to the other ~9,000 lines of read-surface
internals that live alongside it. ``RawRevisionGovernanceHost`` is a
``Protocol`` naming exactly the seven members this module touches
(``_conn``, ``_ensure_source_conn``, ``_blob_publisher``,
``_pending_raw_parse_states``, ``_preacquire_attachment_blobs``,
``_write_counts``, ``_skipped_counts``). ``ArchiveStore`` already defines all
seven under those exact names, so it satisfies the protocol structurally —
no inheritance, no explicit adapter, and no import of ``ArchiveStore`` here
(which would create an import cycle: ``archive.py`` must import this module
to expose the governance surface as ``ArchiveStore`` methods again).

This was chosen over two alternatives: (a) passing the raw ``sqlite3.Connection``
alone — insufficient, because several functions need the lazily-opened
source.db connection, the blob publisher, and the pending-raw-parse-state
batch, not just one connection; (b) a mixin ``ArchiveStore`` inherits from —
rejected because inheritance gives every moved method unrestricted `self`
access to all of ``ArchiveStore``'s state, which is exactly the "reach back
into internals" this module is supposed to make impossible to do by
accident. The Protocol makes the dependency surface an explicit, readable
contract instead of "whatever `self` happens to have".

``ArchiveStore`` keeps one-line delegating methods
(``def classify_raw_revision_cohort(self, ...): return
classify_raw_revision_cohort(self, ...)``) so every existing external caller
(``polylogue/sources/live/batch.py``, ``polylogue/sources/live/append_ingest.py``,
``polylogue/sources/revision_backfill.py``, ``polylogue/storage/repair.py``,
``polylogue/pipeline/services/archive_ingest.py``, ``polylogue/api/archive.py``,
and every test that holds an ``ArchiveStore``/``archive`` instance) keeps
calling ``archive.classify_raw_revision_cohort(...)`` unchanged. That is not
a compatibility shim: there is exactly one implementation, it lives here, and
``ArchiveStore``'s method bodies are the call site — the same shape as any
other function-then-delegate extraction, not a parallel/duplicated
implementation kept alive for a deprecated audience.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast

from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_FLAGS
from polylogue.archive.revision_authority import (
    RETIRED_FULL_REVISION_GOVERNANCE_DETAILS,
    HistoricalRawRevisionStream,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
    classify_historical_full_revision_streams,
)
from polylogue.archive.revision_replay import (
    ApplicationDecision,
    RevisionCandidate,
    RevisionReplayPlan,
    plan_revision_replay,
)
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import origin_from_provider, provider_from_origin
from polylogue.pipeline.ids import SessionRevisionProjection, session_content_hash, session_revision_projection
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.sources.dispatch import merge_parsed_session_chunks
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync
from polylogue.storage.fts.session_repair import repair_session_fts_if_needed_sync
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.sqlite.archive_tiers.ingest_precedence import (
    BrowserCapturePrecedence,
    browser_capture_precedence,
    record_capture_gap_event,
    record_source_outage_events,
    session_has_parser_ingest_flag,
    should_skip_stale_replace,
    stored_message_count,
)
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    FullSnapshotFoldAuthorization,
    RevisionApplicationReceipt,
    assert_session_fts_exact_sync,
    record_revision_application_sync,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceBlobRef,
    apply_source_raw_state_update,
    bind_source_raw_revision,
    write_source_blob_refs,
    write_source_raw_session,
    write_source_raw_session_blob_ref,
)
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedSessionRows,
    _timestamp_ms,
    replace_parser_ingest_flag_tags,
    upsert_parser_ingest_flag_tags,
    write_parsed_session_to_archive,
)


class ActiveByteRevisionChainError(RuntimeError):
    """A byte-identical revision chain cannot admit a conflicting sibling."""


class RawRevisionGovernanceHost(Protocol):
    """The narrow slice of ``ArchiveStore`` this module is allowed to touch.

    ``ArchiveStore`` satisfies this structurally (duck typing) — it is never
    declared as implementing it explicitly, so this module never imports
    ``ArchiveStore`` and no import cycle can form.
    """

    _conn: sqlite3.Connection
    _blob_publisher: ArchiveBlobPublisher | None
    _pending_raw_parse_states: list[tuple[str, RawSessionStateUpdate]]

    def _ensure_source_conn(self) -> sqlite3.Connection: ...

    def _preacquire_attachment_blobs(
        self,
        session: ParsedSession,
        *,
        source_path: str,
        acquired_at_ms: int,
    ) -> tuple[
        dict[int, tuple[bytes | None, int, str]],
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


def _write_parsed_precedence_result(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    raw_id: str,
    source_index: int,
    stage_timings_s: dict[str, float] | None,
    stage_timing_prefix: str,
    manage_transaction: bool,
    preacquired_attachment_blobs: dict[int, tuple[bytes | None, int, str]] | None = None,
    revision_authoritative: bool = False,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    defer_fts_rebuild: bool = False,
    prepared: PreparedSessionRows | None = None,
) -> ArchiveRawParsedWriteResult:
    session_id = str(make_session_id(session.source_name, session.provider_session_id))
    content_hash = str(session_content_hash(session))
    existing_row = store._conn.execute(
        "SELECT content_hash, raw_id, updated_at_ms FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    existing_raw_id = str(existing_row["raw_id"] or "") if existing_row is not None else ""
    existing_hash = existing_row["content_hash"] if existing_row is not None else None
    existing_hash_hex = existing_hash.hex() if isinstance(existing_hash, bytes) else str(existing_hash or "")
    content_unchanged = existing_row is not None and existing_hash_hex == content_hash
    existing_is_dom_fallback = False
    incoming_is_dom_fallback = DOM_FALLBACK_INGEST_FLAG in session.ingest_flags
    existing_has_native_browser_payload = False
    incoming_has_native_browser_payload = any(flag in session.ingest_flags for flag in NATIVE_BROWSER_CAPTURE_FLAGS)
    current_stored_message_count = 0
    browser_precedence: BrowserCapturePrecedence = "default"

    if revision_authoritative:
        write_parsed_session_to_archive(
            store._conn,
            session,
            content_hash=content_hash,
            raw_id=raw_id,
            merge_append=source_index < 0,
            force_replace=source_index >= 0,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            preacquired_attachment_blobs=preacquired_attachment_blobs,
            manage_transaction=manage_transaction,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            defer_fts_rebuild=defer_fts_rebuild,
            prepared=prepared,
        )
        return ArchiveRawParsedWriteResult(
            raw_id=raw_id,
            session_id=session_id,
            content_changed=True,
            counts=store._write_counts(session),
        )
    governed = store._conn.execute(
        "SELECT 1 FROM raw_revision_heads WHERE session_id = ? LIMIT 1",
        (session_id,),
    ).fetchone()
    if governed is not None:
        return ArchiveRawParsedWriteResult(
            raw_id=raw_id,
            session_id=session_id,
            content_changed=False,
            counts=store._skipped_counts(session),
        )
    # polylogue-c737: ``governed`` above only catches a logical
    # identity with an ACCEPTED revision-authority head
    # (``raw_revision_heads``, populated by
    # ``apply_raw_membership_classification``/``apply_raw_revision_replay``
    # only when a cohort has a winner). A cohort that ``classify_
    # membership_revisions`` refused to arbitrate -- genuinely
    # ``raw_session_memberships.decision = 'ambiguous'`` -- never gets an
    # accepted head, so ``governed`` stays ``None`` here even though this
    # raw's own identity is recorded authority debt. Falling through to
    # the ordinary browser-capture-precedence/freshness logic below then
    # writes this raw's session unconditionally on its next parse --
    # last-writer-wins, exactly the "never silently choose between
    # branches" invariant this whole subsystem exists to enforce, and
    # the fidelity-losing side of an aistudio-drive ambiguous pair reaches
    # the index every time this reparses (measured live: 28 cohorts, 641
    # attachments reported unfetched despite the bytes existing in the
    # blob store). Refuse this raw explicitly instead of relying on an
    # absent head to imply "unclaimed, free to write".
    #
    # Scoped to the membership being written, not to the raw. One retained
    # raw routinely lowers to many sessions -- a Claude Code transcript and
    # its subagent sidechains, a bundle member set -- and those sessions are
    # arbitrated independently. Measured on the live archive: 295 raws carry
    # a mix of decisions, together holding 489 sessions whose own membership
    # is NOT ambiguous, and one raw carries 106 memberships. A raw-scoped
    # predicate suppresses every one of those sessions as soon as a single
    # sibling membership is ambiguous, which trades a fidelity downgrade for
    # outright absence -- a worse failure, and one that would have landed at
    # the next full rebuild.
    ambiguous_membership = (
        store._ensure_source_conn()
        .execute(
            """
            SELECT 1 FROM raw_session_memberships
            WHERE raw_id = ? AND provider_session_id = ? AND decision = 'ambiguous'
            LIMIT 1
            """,
            (raw_id, session.provider_session_id),
        )
        .fetchone()
    )
    if ambiguous_membership is not None:
        return ArchiveRawParsedWriteResult(
            raw_id=raw_id,
            session_id=session_id,
            content_changed=False,
            counts=store._skipped_counts(session),
        )

    if source_index >= 0 and existing_raw_id and raw_id and existing_raw_id != raw_id:
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
                counts=store._skipped_counts(session, session_events=session_event_count),
            )

    incoming_freshness_ms = _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at)
    if (
        source_index >= 0
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
            return ArchiveRawParsedWriteResult(
                raw_id=raw_id,
                session_id=session_id,
                content_changed=False,
                counts=store._skipped_counts(session),
            )

    if content_unchanged:
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
        fts_repaired = repair_session_fts_if_needed_sync(store._conn, session_id)
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
        )

    write_parsed_session_to_archive(
        store._conn,
        session,
        content_hash=content_hash,
        raw_id=raw_id,
        merge_append=source_index < 0,
        force_replace=browser_precedence == "replace",
        stage_timings_s=stage_timings_s,
        stage_timing_prefix=stage_timing_prefix,
        preacquired_attachment_blobs=preacquired_attachment_blobs,
        manage_transaction=manage_transaction,
        prepared=prepared,
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
    )


def write_raw_and_parsed(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    payload: bytes,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    raw_id: str | None = None,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    manage_transaction: bool = True,
    blob_publication_receipt_id: str | None = None,
    finalize_raw_parse: bool = True,
) -> tuple[str, str]:
    """Write raw acquisition bytes and the parsed session they produced."""
    result = write_raw_and_parsed_result(
        store,
        session,
        payload=payload,
        source_path=source_path,
        acquired_at_ms=acquired_at_ms,
        source_index=source_index,
        raw_id=raw_id,
        stage_timings_s=stage_timings_s,
        stage_timing_prefix=stage_timing_prefix,
        manage_transaction=manage_transaction,
        blob_publication_receipt_id=blob_publication_receipt_id,
        finalize_raw_parse=finalize_raw_parse,
    )
    return result.raw_id, result.session_id


def write_raw_payload(
    store: RawRevisionGovernanceHost,
    *,
    provider: Provider,
    capture_mode: Provider | None = None,
    payload: bytes,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    raw_id: str | None = None,
    blob_publication_receipt_id: str | None = None,
    revision: RawRevisionEnvelope | None = None,
) -> str:
    """Commit raw bytes before attempting to parse or index them."""
    if store._blob_publisher is None:
        raise RuntimeError("raw archive writes require a writable archive publisher")
    if blob_publication_receipt_id is None:
        raw_hash, _raw_size = store._blob_publisher.write_from_bytes(payload)
        blob_publication_receipt_id = store._blob_publisher.receipt_id(raw_hash)
    store._blob_publisher.flush()
    return write_source_raw_session(
        store._ensure_source_conn(),
        origin=origin_from_provider(provider),
        capture_mode=capture_mode or provider,
        source_path=source_path,
        source_index=source_index,
        payload=payload,
        acquired_at_ms=acquired_at_ms,
        raw_id=raw_id,
        blob_publication_receipt_id=blob_publication_receipt_id,
        revision=revision,
        manage_transaction=True,
    )


def write_raw_blob_ref(
    store: RawRevisionGovernanceHost,
    *,
    provider: Provider,
    capture_mode: Provider | None = None,
    blob_hash_hex: str,
    blob_size: int,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    raw_id: str | None = None,
    blob_publication_receipt_id: str | None = None,
    revision: RawRevisionEnvelope | None = None,
) -> str:
    """Commit a prepublished raw blob reference before parsing it."""
    if store._blob_publisher is not None:
        store._blob_publisher.flush()
    return write_source_raw_session_blob_ref(
        store._ensure_source_conn(),
        origin=origin_from_provider(provider),
        capture_mode=capture_mode or provider,
        source_path=source_path,
        source_index=source_index,
        blob_hash=bytes.fromhex(blob_hash_hex),
        blob_size=blob_size,
        acquired_at_ms=acquired_at_ms,
        raw_id=raw_id,
        blob_publication_receipt_id=blob_publication_receipt_id,
        revision=revision,
        manage_transaction=True,
    )


def write_parsed_for_retained_raw(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    raw_id: str,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    manage_transaction: bool = True,
    finalize_raw_parse: bool = True,
    revision_authoritative: bool = False,
) -> tuple[str, str]:
    """Index one session for raw evidence that is already durable."""
    result = write_parsed_for_retained_raw_result(
        store,
        session,
        raw_id=raw_id,
        source_path=source_path,
        acquired_at_ms=acquired_at_ms,
        source_index=source_index,
        stage_timings_s=stage_timings_s,
        stage_timing_prefix=stage_timing_prefix,
        manage_transaction=manage_transaction,
        finalize_raw_parse=finalize_raw_parse,
        revision_authoritative=revision_authoritative,
    )
    return result.raw_id, result.session_id


def write_parsed_for_retained_raw_result(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    raw_id: str,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    manage_transaction: bool = True,
    finalize_raw_parse: bool = True,
    revision_authoritative: bool = False,
) -> ArchiveRawParsedWriteResult:
    """Index one session for raw evidence that is already durable, with counts.

    Used both by append-chain replay and by any caller that must index
    several sessions parsed from ONE physical raw acquisition (e.g. a
    Claude Code/Codex grouped JSONL file whose content splits into
    multiple sessions) against the SAME raw_id, instead of writing a
    duplicate raw row per session.
    """
    preacquired_attachments, attachment_blob_refs = store._preacquire_attachment_blobs(
        session,
        source_path=source_path,
        acquired_at_ms=acquired_at_ms,
    )
    if store._blob_publisher is not None:
        store._blob_publisher.flush()
    write_source_blob_refs(store._ensure_source_conn(), raw_id, attachment_blob_refs)
    index_started = time.perf_counter()
    result = _index_parsed_for_retained_raw(
        store,
        session,
        raw_id=raw_id,
        source_index=source_index,
        stage_timings_s=stage_timings_s,
        stage_timing_prefix=stage_timing_prefix,
        manage_transaction=manage_transaction,
        preacquired_attachment_blobs=preacquired_attachments,
        finalize_raw_parse=finalize_raw_parse,
        revision_authoritative=revision_authoritative,
    )
    if stage_timings_s is not None:
        key = f"{stage_timing_prefix}.index_parsed_write"
        stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - index_started)
    return result


def bind_raw_revision(
    store: RawRevisionGovernanceHost, raw_id: str, revision: RawRevisionEnvelope, *, manage_transaction: bool = True
) -> None:
    """Bind acquisition evidence; ``manage_transaction=False`` batches (polylogue-amg1)."""
    bind_source_raw_revision(store._ensure_source_conn(), raw_id, revision, manage_transaction=manage_transaction)


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
        SELECT raw_id, COALESCE(baseline_raw_id, raw_id), acquisition_generation
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


def raw_membership_retired_full_revision_siblings(
    store: RawRevisionGovernanceHost, logical_source_key: str
) -> tuple[str, ...]:
    """Return raws previously retired from full-revision byte governance for this key.

    ``replace_raw_membership_census(..., retire_full_revision_governance=True)``
    nulls the retired raw's ``raw_sessions.logical_source_key`` and sets
    ``revision_authority='quarantined'`` -- it becomes invisible both to
    ``classify_raw_revision_cohort``'s own byte-row query and to
    ``raw_membership_rebuild_raw_ids``'s deliberately byte-proven-only
    filter (polylogue-lkrc/#2822 guards a different hazard: reopening a
    quarantined member against an already-established head). Its
    ``raw_session_memberships`` row, keyed by the raw's own parsed
    logical identity, survives retirement together with a
    ``raw_membership_census.detail`` marker naming this specific
    transition, so a later-arriving sibling for the same identity can
    still be told this identity has known, unresolved ambiguous
    evidence (polylogue-52l2) instead of being evaluated alone.

    Matches every literal in ``RETIRED_FULL_REVISION_GOVERNANCE_DETAILS``
    (polylogue-hm2f), not only the current
    ``HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL`` marker: durable
    ``raw_membership_census`` rows written before #3234 used a different,
    now-legacy literal at the live-watcher call site, and durable-tier
    detail strings are never silently rewritten in place.
    """
    detail_placeholders = ", ".join("?" for _ in RETIRED_FULL_REVISION_GOVERNANCE_DETAILS)
    where_clause = f"c.detail IN ({detail_placeholders})"
    rows = (
        store._ensure_source_conn()
        .execute(
            f"""
            SELECT m.raw_id
            FROM raw_session_memberships AS m
            JOIN raw_membership_census AS c ON c.raw_id = m.raw_id
            WHERE m.logical_source_key = ? AND {where_clause}
            ORDER BY m.raw_id
            """,
            (logical_source_key, *RETIRED_FULL_REVISION_GOVERNANCE_DETAILS),
        )
        .fetchall()
    )
    return tuple(str(row[0]) for row in rows)


def _raw_revision_source_path_has_divergent_evidence(store: RawRevisionGovernanceHost, logical_source_key: str) -> bool:
    """Detect a same-``source_path`` sibling under a DIFFERENT byte-revision key.

    Polylogue-eqnv: two raws of the identical physical document can end
    up with different ``logical_source_key`` values -- most concretely
    when one was censused by a parser version with an identity bug since
    fixed (the source_revision the other raw's key was assigned under
    never gets revisited, see ``uncensused_historical_revision_raw_ids``'s
    exact-fingerprint quiescence gate). Neither raw's own key surfaces
    the other in ``raw_membership_retired_full_revision_siblings``, so
    both would otherwise be accepted as independent one-member byte
    chains. ``source_path`` is the correct join key here (as elsewhere
    in this module, e.g. ``classify_untyped_full_revision_groups``): a
    real re-acquisition of the same document always keeps the same path.
    """
    detail_placeholders = ", ".join("?" for _ in RETIRED_FULL_REVISION_GOVERNANCE_DETAILS)
    row = (
        store._ensure_source_conn()
        .execute(
            f"""
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
                        AND c.detail IN ({detail_placeholders})
                  )
              )
            LIMIT 1
            """,
            (logical_source_key, *RETIRED_FULL_REVISION_GOVERNANCE_DETAILS),
        )
        .fetchone()
    )
    return row is not None


def classify_raw_revision_cohort(
    store: RawRevisionGovernanceHost, logical_source_key: str, *, check_source_path_identity_split: bool = False
) -> RevisionReplayPlan:
    """Promote only a unique byte-prefix full chain and contiguous appends.

    ``check_source_path_identity_split`` (polylogue-eqnv, default
    ``False`` -- opt in explicitly, see
    ``_raw_revision_source_path_has_divergent_evidence``) additionally
    refuses a singleton accept when another 'full' raw shares this raw's
    ``source_path`` under a DIFFERENT key. That heuristic is sound for a
    genuine re-acquisition of the same physical document (the offline
    backfill/rebuild path's use case, where a stale pre-fix parser
    identity can split one document across two keys) but is NOT sound in
    general: a watched source path can legitimately be atomically
    replaced with an entirely different session's content (same path,
    genuinely different identity, exercised by
    ``test_full_ingest_does_not_advance_cursor_across_same_size_replacement``
    et al.) -- the live incremental watcher (``sources/live/batch.py``)
    must not quarantine that as "divergent evidence", so it leaves this
    off.
    """
    if store._blob_publisher is None:
        raise RuntimeError("raw revision classification requires a writable blob publisher")
    source_conn = store._ensure_source_conn()
    full_rows = source_conn.execute(
        """
        SELECT raw_id, lower(hex(blob_hash)) AS blob_hash, blob_size
        FROM raw_sessions
        WHERE logical_source_key = ? AND revision_kind = 'full'
        """,
        (logical_source_key,),
    ).fetchall()
    # polylogue-52l2: a byte chain is classified against whichever full
    # rows the caller happens to have discovered/censused so far, not
    # against the complete sibling population for this logical identity
    # -- an earlier pass can have already retired ambiguous siblings to
    # membership governance (nulling their logical_source_key, see
    # raw_membership_retired_full_revision_siblings). If that leaves a
    # later-discovered raw as the ONLY remaining 'full' row here, it
    # would be evaluated as a trivial one-member "chain" and
    # unconditionally accepted as a byte-proven baseline by
    # classify_historical_full_revision_streams (no sibling to prove a
    # byte prefix against) -- permanently establishing session content
    # from whichever raw happened to be discovered last, independent of
    # which content is actually correct. Refuse the byte-chain path
    # entirely whenever this identity has retired sibling evidence: the
    # caller's existing "no accepted chain" fallback
    # (convertible_full_revision_raw_ids) folds these full rows into
    # membership governance instead, where the real prefix-based
    # classifier weighs every known sibling together.
    if full_rows and raw_membership_retired_full_revision_siblings(store, logical_source_key):
        full_rows = []
    # polylogue-eqnv: the guard above only catches a retired SIBLING
    # discoverable under the SAME logical_source_key. A raw whose
    # identity was assigned by a now-superseded parser (e.g. the
    # pre-#3179/z1c6 dispatch bug that appended a spurious "-0" to one
    # of two otherwise-identical Drive re-acquisitions) can carry a
    # logical_source_key that DIFFERS from a same-document sibling's --
    # neither raw's own key ever surfaces the other, so each gets
    # evaluated as a trivial one-member "chain" and unconditionally
    # accepted as a byte-proven singleton baseline, independent of
    # which content is actually correct. Two such raws then silently
    # materialize as two independent sessions (arbitrary last-write-
    # wins on the shared (origin, native_id) upsert) instead of ever
    # being compared. Detect this by ``source_path``: a real re-
    # acquisition of the same physical document always keeps the same
    # ``source_path``. Refuse the byte-chain path here too whenever
    # another raw at the same source_path is still an unretired 'full'
    # row under a different key, OR has already been retired to
    # membership governance under any key (a prior pass may have
    # re-derived a DIFFERENT, corrected key during its own retirement
    # reparse) -- the caller's existing "no accepted chain" fallback
    # folds this raw into membership governance too, where the real
    # content-based classifier weighs every known sibling together.
    if (
        full_rows
        and check_source_path_identity_split
        and _raw_revision_source_path_has_divergent_evidence(store, logical_source_key)
    ):
        full_rows = []
    historical: list[HistoricalRawRevisionStream] = []
    for row in full_rows:

        def open_payload(blob_hash: str = str(row[1])) -> BinaryIO:
            assert store._blob_publisher is not None
            return store._blob_publisher.open(blob_hash)

        historical.append(
            HistoricalRawRevisionStream(
                raw_id=str(row[0]),
                payload_size=int(row[2]),
                open_payload=open_payload,
            )
        )
    decisions = classify_historical_full_revision_streams(historical)
    by_raw_id = {decision.raw_id: decision for decision in decisions}
    baseline_ids = [decision.raw_id for decision in decisions if decision.relation == "baseline"]
    baseline_raw_id = baseline_ids[0] if len(baseline_ids) == 1 else None
    generation_by_raw_id: dict[str, int] = {}
    if baseline_raw_id is not None:
        current: str | None = baseline_raw_id
        generation = 0
        children = {
            decision.predecessor_raw_id: decision.raw_id
            for decision in decisions
            if decision.predecessor_raw_id is not None
        }
        while current is not None:
            generation_by_raw_id[current] = generation
            current = children.get(current)
            generation += 1
    with source_conn:
        for row in full_rows:
            raw_id = str(row[0])
            decision = by_raw_id.get(raw_id)
            authority = decision.authority if decision is not None else RawRevisionAuthority.QUARANTINED
            predecessor_raw_id = decision.predecessor_raw_id if decision is not None else None
            source_conn.execute(
                """
                UPDATE raw_sessions
                SET revision_authority = ?, predecessor_raw_id = ?, baseline_raw_id = ?,
                    acquisition_generation = ?
                WHERE raw_id = ?
                """,
                (
                    authority.value,
                    predecessor_raw_id,
                    baseline_raw_id if authority is RawRevisionAuthority.BYTE_PROVEN else None,
                    generation_by_raw_id.get(raw_id, 0),
                    raw_id,
                ),
            )
        _promote_contiguous_append_evidence(source_conn, logical_source_key)
    return raw_revision_replay_plan(store, logical_source_key)


def classify_untyped_full_revision_groups(
    store: RawRevisionGovernanceHost, raw_ids: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Group never-typed retained raws into proven byte-growth chains, no parse.

    A restore/backfill census over legacy raws otherwise has to fully parse
    every retained revision just to learn its ``logical_source_key`` --
    including revisions that a byte-only comparison already proves are a
    strict prefix of a later capture of the *same* file (polylogue-nh44:
    45GB/46% of a real restore's parse work was exactly this waste).
    ``source_path`` is an established cohort-equivalence edge elsewhere in
    this codebase (see ``raw_membership_selection_components_sync``), so
    grouping candidates by it before parsing is sound: two raws at the same
    path are always the same logical file at different capture times.

    Returns ``{head_raw_id: (older_raw_id, ...)}`` for every source_path
    cohort of >=2 raws whose bytes form a unique linear growth chain per
    ``classify_historical_full_revision_streams``. The caller still owns
    parsing the head (the only member whose content is ever actually
    indexed) and binding the proven-older members to its learned identity;
    this method performs no writes. Ambiguous, branching, or singleton
    groups are omitted so callers fall back to parsing every member.
    """
    if store._blob_publisher is None:
        raise RuntimeError("raw revision classification requires a writable blob publisher")
    if not raw_ids:
        return {}
    placeholders = ",".join("?" for _ in raw_ids)
    rows = (
        store._ensure_source_conn()
        .execute(
            f"""
        SELECT raw_id, source_path, lower(hex(blob_hash)), blob_size
        FROM raw_sessions
        WHERE raw_id IN ({placeholders}) AND revision_kind = 'unknown'
        """,
            tuple(raw_ids),
        )
        .fetchall()
    )
    by_path: dict[str, list[tuple[str, str, int]]] = {}
    for raw_id, source_path, blob_hash, blob_size in rows:
        by_path.setdefault(str(source_path), []).append((str(raw_id), str(blob_hash), int(blob_size)))
    groups: dict[str, tuple[str, ...]] = {}
    for members in by_path.values():
        if len(members) < 2:
            continue

        streams = []
        for raw_id, blob_hash, blob_size in members:

            def open_payload(blob_hash: str = blob_hash) -> BinaryIO:
                assert store._blob_publisher is not None
                return store._blob_publisher.open(blob_hash)

            streams.append(
                HistoricalRawRevisionStream(raw_id=raw_id, payload_size=blob_size, open_payload=open_payload)
            )
        decisions = classify_historical_full_revision_streams(streams)
        if not decisions or decisions[0].authority is not RawRevisionAuthority.BYTE_PROVEN:
            continue
        groups[decisions[-1].raw_id] = tuple(decision.raw_id for decision in decisions[:-1])
    return groups


def _promote_contiguous_append_evidence(conn: sqlite3.Connection, logical_source_key: str) -> None:
    while True:
        candidates = conn.execute(
            """
            SELECT child.raw_id, parent.raw_id, COALESCE(parent.baseline_raw_id, parent.raw_id),
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
                  OR child.baseline_raw_id != COALESCE(parent.baseline_raw_id, parent.raw_id)
                  OR child.acquisition_generation != parent.acquisition_generation + 1
              )
            """,
            (logical_source_key,),
        ).fetchall()
        by_child: dict[str, list[sqlite3.Row | tuple[object, ...]]] = {}
        for row in candidates:
            by_child.setdefault(str(row[0]), []).append(row)
        promotable = [rows[0] for rows in by_child.values() if len(rows) == 1]
        if not promotable:
            return
        changed = 0
        for row in promotable:
            cursor = conn.execute(
                """
                UPDATE raw_sessions
                SET revision_authority = 'byte_proven', predecessor_raw_id = ?,
                    baseline_raw_id = ?, acquisition_generation = ?
                WHERE raw_id = ?
                """,
                (str(row[1]), str(row[2]), int(cast(Any, row[3])), str(row[0])),
            )
            changed += int(cursor.rowcount)
        if not changed:
            return


def _raw_revision_authority(store: RawRevisionGovernanceHost, raw_id: str) -> str | None:
    row = (
        store._ensure_source_conn()
        .execute("SELECT revision_authority FROM raw_sessions WHERE raw_id = ?", (raw_id,))
        .fetchone()
    )
    return None if row is None or row[0] is None else str(row[0])


def raw_revision_replay_plan(store: RawRevisionGovernanceHost, logical_source_key: str) -> RevisionReplayPlan:
    return plan_revision_replay(_raw_revision_candidates(store, logical_source_key))


def _raw_revision_candidates(store: RawRevisionGovernanceHost, logical_source_key: str) -> list[RevisionCandidate]:
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
    store: RawRevisionGovernanceHost,
    *,
    existing_head: tuple[object, ...],
    full_candidate: RevisionCandidate,
    candidates: Mapping[str, RevisionCandidate],
) -> FullSnapshotFoldAuthorization | None:
    """Prove one full raw is exactly the accepted byte-append chain.

    The caller invokes this while holding the index replay transaction;
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
        tail_digest, tail_size = _raw_revision_payload_digest_and_size(store, current.raw_id)
        assert current.append_end_offset is not None
        assert current.append_start_offset is not None
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
        [current.raw_id, *reversed(tail_raw_ids)],
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
    if store._blob_publisher is None:
        raise RuntimeError("raw revision replay requires a writable blob publisher")
    row = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT origin, capture_mode, lower(hex(blob_hash)), source_path, revision_kind, blob_size
        FROM raw_sessions WHERE raw_id = ?
        """,
            (raw_id,),
        )
        .fetchone()
    )
    if row is None:
        raise KeyError(raw_id)
    return (
        provider_from_origin(Origin.from_string(str(row[0])), family_hint=row[1]),
        str(row[2]),
        str(row[3]),
        RawRevisionKind(str(row[4])),
        int(row[5]),
    )


@contextmanager
def open_raw_revision_material(
    store: RawRevisionGovernanceHost, raw_id: str
) -> Iterator[tuple[Provider, BinaryIO, str, RawRevisionKind]]:
    """Open a retained revision for bounded streaming consumption."""
    provider, blob_hash, source_path, kind, _blob_size = raw_revision_descriptor(store, raw_id)
    assert store._blob_publisher is not None
    with store._blob_publisher.open(blob_hash) as payload:
        yield provider, payload, source_path, kind


def raw_revision_material(
    store: RawRevisionGovernanceHost, raw_id: str
) -> tuple[Provider, bytes, str, RawRevisionKind]:
    """Read one retained revision with its parsing identity.

    Use ``open_raw_revision_material`` for potentially large blobs.
    """
    provider, blob_hash, source_path, kind, _blob_size = raw_revision_descriptor(store, raw_id)
    assert store._blob_publisher is not None
    return provider, store._blob_publisher.read_all(blob_hash), source_path, kind


def blob_path_for_hash(store: RawRevisionGovernanceHost, blob_hash: str) -> Path | None:
    """Return the real on-disk path for a content-addressed blob, if materialized.

    Some payload shapes (Hermes state.db/verification_evidence.db) need a
    real filesystem path -- ``sqlite3.connect`` cannot open in-memory
    bytes. Returns ``None`` when the blob is not (yet) materialized on
    disk so callers fall back to a bounded temp-file spill instead of
    trusting an unverified path.
    """
    assert store._blob_publisher is not None
    path = store._blob_publisher.blob_path(blob_hash)
    return path if path.exists() else None


def _raw_revision_payload_digest_and_size(store: RawRevisionGovernanceHost, raw_id: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open_raw_revision_material(store, raw_id) as (_provider, payload, _source_path, _kind):
        while chunk := payload.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _raw_revision_matches_segments(
    store: RawRevisionGovernanceHost, full_raw_id: str, segment_raw_ids: Sequence[str]
) -> bool:
    with open_raw_revision_material(store, full_raw_id) as (_provider, full, _source_path, _kind):
        for raw_id in segment_raw_ids:
            with open_raw_revision_material(store, raw_id) as (_provider, segment, _source_path, _kind):
                while chunk := segment.read(1024 * 1024):
                    if full.read(len(chunk)) != chunk:
                        return False
        return full.read(1) == b""


def unclassified_raw_revision_rows(store: RawRevisionGovernanceHost) -> tuple[tuple[str, int], ...]:
    """Return legacy rows that have no durable logical revision identity."""
    rows = (
        store._ensure_source_conn()
        .execute(
            """
        SELECT raw_id, source_index
        FROM raw_sessions
        WHERE logical_source_key IS NULL AND revision_authority = 'quarantined'
        ORDER BY raw_id
        """
        )
        .fetchall()
    )
    return tuple((str(row[0]), int(row[1])) for row in rows)


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


def raw_revision_rebuild_selection(
    store: RawRevisionGovernanceHost,
    raw_ids: list[str] | None,
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    """Expand requested raws only to complete same-source-path cohorts."""
    conn = store._ensure_source_conn()
    if raw_ids is None:
        return (
            unclassified_raw_revision_rows(store),
            tuple(
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT DISTINCT logical_source_key FROM raw_sessions
                    WHERE logical_source_key IS NOT NULL ORDER BY logical_source_key
                    """
                )
            ),
        )
    selected = tuple(dict.fromkeys(raw_ids))
    if not selected:
        return (), ()
    placeholders = ",".join("?" for _ in selected)
    source_paths = tuple(
        str(row[0])
        for row in conn.execute(
            f"SELECT DISTINCT source_path FROM raw_sessions WHERE raw_id IN ({placeholders})",
            selected,
        )
    )
    if not source_paths:
        return (), ()
    path_placeholders = ",".join("?" for _ in source_paths)
    unclassified = tuple(
        (str(row[0]), int(row[1]))
        for row in conn.execute(
            f"""
            SELECT raw_id, source_index FROM raw_sessions
            WHERE source_path IN ({path_placeholders})
              AND logical_source_key IS NULL
              AND revision_authority = 'quarantined'
            ORDER BY raw_id
            """,
            source_paths,
        )
    )
    logical_keys = tuple(
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
    return unclassified, logical_keys


def raw_membership_census_rows(
    store: RawRevisionGovernanceHost, raw_ids: Sequence[str] | None = None
) -> tuple[tuple[str, int], ...]:
    """Return every retained raw whose membership census may affect authority."""
    conn = store._ensure_source_conn()
    if raw_ids is None:
        rows = conn.execute("SELECT raw_id, source_index FROM raw_sessions ORDER BY raw_id").fetchall()
    elif raw_ids:
        placeholders = ",".join("?" for _ in raw_ids)
        rows = conn.execute(
            f"SELECT raw_id, source_index FROM raw_sessions WHERE raw_id IN ({placeholders}) ORDER BY raw_id",
            tuple(raw_ids),
        ).fetchall()
    else:
        rows = []
    return tuple((str(row[0]), int(row[1])) for row in rows)


def raw_payload_sizes(store: RawRevisionGovernanceHost, raw_ids: Sequence[str]) -> dict[str, int]:
    if not raw_ids:
        return {}
    placeholders = ",".join("?" for _ in raw_ids)
    rows = store._ensure_source_conn().execute(
        f"SELECT raw_id, blob_size FROM raw_sessions WHERE raw_id IN ({placeholders})",
        tuple(raw_ids),
    )
    return {str(row[0]): int(row[1] or 0) for row in rows}


def replace_raw_membership_census(
    store: RawRevisionGovernanceHost,
    raw_id: str,
    sessions: list[ParsedSession] | None,
    *,
    parser_fingerprint: str,
    censused_at_ms: int,
    detail: str = "",
    retire_full_revision_governance: bool = False,
    manage_transaction: bool = True,
) -> None:
    """Replace one raw's complete parser census and memberships.

    ``manage_transaction=False`` batches multiple raws' census writes
    into one caller-managed commit window (polylogue-amg1) -- the caller
    must call ``commit()`` (or ``rollback()`` on failure) itstore.
    """
    conn = store._ensure_source_conn()
    with conn if manage_transaction else nullcontext():
        if retire_full_revision_governance:
            revision = conn.execute(
                "SELECT logical_source_key, revision_kind FROM raw_sessions WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()
            if revision is None:
                raise RuntimeError(f"membership census raw is missing: {raw_id}")
            if revision[0] is not None and str(revision[1]) != RawRevisionKind.FULL.value:
                raise RuntimeError("only self-contained full raws can move to membership governance")
            dependent = conn.execute(
                """
                SELECT 1 FROM raw_sessions
                WHERE raw_id != ?
                  AND (predecessor_raw_id = ? OR baseline_raw_id = ?)
                LIMIT 1
                """,
                (raw_id, raw_id, raw_id),
            ).fetchone()
            if dependent is not None:
                raise ActiveByteRevisionChainError("an active byte-revision chain cannot move to membership governance")
            conn.execute(
                """
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
                WHERE raw_id = ?
                """,
                (raw_id,),
            )
        conn.execute("DELETE FROM raw_session_memberships WHERE raw_id = ?", (raw_id,))
        if sessions is not None:
            for session in sessions:
                projection = session_revision_projection(session)
                logical_key = f"{session.source_name.value}:{session.provider_session_id}"
                conn.execute(
                    """
                    INSERT INTO raw_session_memberships (
                        raw_id, logical_source_key, provider_session_id,
                        source_revision, normalized_content_hash, message_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw_id,
                        logical_key,
                        session.provider_session_id,
                        projection.session_hash.hex(),
                        projection.session_hash,
                        len(projection.message_hashes),
                    ),
                )
        status = "failed" if sessions is None else ("non_session" if not sessions else "complete")
        conn.execute(
            """
            INSERT INTO raw_membership_census (
                raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(raw_id) DO UPDATE SET
                parser_fingerprint=excluded.parser_fingerprint,
                status=excluded.status,
                member_count=excluded.member_count,
                censused_at_ms=excluded.censused_at_ms,
                detail=excluded.detail
            """,
            (raw_id, parser_fingerprint, status, len(sessions or []), censused_at_ms, detail),
        )


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


def expand_raw_membership_selection_sync(
    conn: sqlite3.Connection,
    raw_ids: list[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Expand raw scheduling hints using durable path/membership metadata."""
    if raw_ids is None:
        selected = {str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions")}
    else:
        selected = set(raw_ids)
    changed = True
    while changed and selected:
        changed = False
        placeholders = ",".join("?" for _ in selected)
        paths = {
            str(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT source_path FROM raw_sessions WHERE raw_id IN ({placeholders})",
                tuple(selected),
            )
        }
        if paths:
            path_marks = ",".join("?" for _ in paths)
            selected.update(
                str(row[0])
                for row in conn.execute(
                    f"SELECT raw_id FROM raw_sessions WHERE source_path IN ({path_marks})", tuple(paths)
                )
            )
        placeholders = ",".join("?" for _ in selected)
        keys = {
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT logical_source_key
                FROM raw_session_memberships
                WHERE raw_id IN ({placeholders})
                UNION
                SELECT logical_source_key
                FROM raw_sessions
                WHERE raw_id IN ({placeholders})
                  AND logical_source_key IS NOT NULL
                """,
                (*selected, *selected),
            )
        }
        before = len(selected)
        if keys:
            key_marks = ",".join("?" for _ in keys)
            selected.update(
                str(row[0])
                for row in conn.execute(
                    f"""
                    SELECT raw_id
                    FROM raw_session_memberships
                    WHERE logical_source_key IN ({key_marks})
                    UNION
                    SELECT raw_id
                    FROM raw_sessions
                    WHERE logical_source_key IN ({key_marks})
                    """,
                    (*keys, *keys),
                )
            )
        changed = len(selected) != before
    if not selected:
        return (), ()
    placeholders = ",".join("?" for _ in selected)
    logical_keys = tuple(
        sorted(
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT logical_source_key
                FROM raw_session_memberships
                WHERE raw_id IN ({placeholders})
                UNION
                SELECT logical_source_key
                FROM raw_sessions
                WHERE raw_id IN ({placeholders})
                  AND logical_source_key IS NOT NULL
                """,
                (*selected, *selected),
            )
        )
    )
    return tuple(sorted(selected)), logical_keys


def raw_membership_raw_ids(
    store: RawRevisionGovernanceHost,
    logical_source_key: str,
    *,
    include_complete_raw_id: str | None = None,
) -> tuple[str, ...]:
    """Return byte-proven candidates plus the complete raw being classified.

    A newly censused live snapshot has not received a membership decision
    yet, so it is deliberately quarantined until this classification
    completes. Admit only that caller-owned complete census alongside
    established byte-proven evidence; do not reopen unrelated quarantined
    members from a prior failed or ambiguous replay.
    """
    rows = (
        store._ensure_source_conn()
        .execute(
            """
            SELECT m.raw_id
            FROM raw_session_memberships AS m
            LEFT JOIN raw_membership_census AS c ON c.raw_id = m.raw_id
            WHERE m.logical_source_key = ?
              AND (
                m.revision_authority = 'byte_proven'
                OR (
                    m.raw_id = ?
                    AND c.status = 'complete'
                    AND m.decision IS NULL
                )
              )
            ORDER BY m.raw_id
            """,
            (logical_source_key, include_complete_raw_id),
        )
        .fetchall()
    )
    return tuple(str(row[0]) for row in rows)


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


def raw_revision_replay_adoptable(store: RawRevisionGovernanceHost, sessions: Sequence[ParsedSession]) -> bool:
    """Return whether replay may adopt an existing ungoverned session."""
    aggregate = merge_parsed_session_chunks(sessions)
    if len(aggregate) != 1:
        return False
    session = aggregate[0]
    session_id = str(make_session_id(session.source_name, session.provider_session_id))
    row = store._conn.execute(
        "SELECT content_hash FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return True
    governed = store._conn.execute(
        "SELECT 1 FROM raw_revision_heads WHERE session_id = ? LIMIT 1",
        (session_id,),
    ).fetchone()
    if governed is not None:
        return True
    existing_hash = row[0]
    existing_hex = existing_hash.hex() if isinstance(existing_hash, bytes) else str(existing_hash or "")
    return existing_hex == session_content_hash(session)


def defer_raw_revision_adoption(
    store: RawRevisionGovernanceHost,
    logical_source_key: str,
    raw_ids: Sequence[str],
    sessions: Sequence[ParsedSession],
) -> None:
    """Receipt a derived replay decision without rewriting source evidence."""
    if not raw_ids:
        return
    source_conn = store._ensure_source_conn()
    decided_at_ms = int(time.time() * 1000)
    aggregate = merge_parsed_session_chunks(sessions)
    if len(aggregate) != 1:
        raise RuntimeError("deferred revision cohort did not compose to one session")
    session = aggregate[0]
    session_id = str(make_session_id(session.source_name, session.provider_session_id))
    with store._conn:
        for raw_id in raw_ids:
            row = source_conn.execute(
                """
                SELECT COALESCE(r.source_revision, m.source_revision),
                       COALESCE(r.acquisition_generation, m.acquisition_generation, 0)
                FROM raw_sessions AS r
                LEFT JOIN raw_session_memberships AS m
                  ON m.raw_id = r.raw_id AND m.logical_source_key = ?
                WHERE r.raw_id = ?
                """,
                (logical_source_key, raw_id),
            ).fetchone()
            if row is None or row[0] is None:
                raise RuntimeError(f"deferred raw revision lacks source evidence: {raw_id}")
            record_revision_application_sync(
                store._conn,
                RevisionApplicationReceipt(
                    raw_id=raw_id,
                    session_id=session_id,
                    logical_source_key=logical_source_key,
                    source_revision=str(row[0]),
                    acquisition_generation=int(row[1]),
                    decision=ApplicationDecision.DEFERRED,
                    accepted_raw_id=None,
                    accepted_source_revision=None,
                    accepted_content_hash=None,
                    detail="ordinary_replay:incomparable_existing_index_state",
                ),
                decided_at_ms=decided_at_ms,
            )


def apply_raw_revision_replay(
    store: RawRevisionGovernanceHost,
    plan: RevisionReplayPlan,
    parsed_by_raw_id: dict[str, ParsedSession],
    *,
    acquired_at_ms: int,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "revision_replay",
    manage_transaction: bool = True,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    defer_fts: bool = False,
    skip_already_applied: bool = False,
    prepared_by_raw_id: dict[str, PreparedSessionRows | Future[PreparedSessionRows]] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Apply a proven chain and atomically receipt its exact index state.

    ``prepared_by_raw_id`` (polylogue-fpid) optionally supplies row tuples
    already built off the writer thread for one or more of ``plan.
    accepted_raw_ids`` -- keyed by ``raw_id``, value either the
    ``PreparedSessionRows`` itself or a ``Future`` resolved here right
    before use (so a caller can ``submit()`` the CPU-bound build on a
    background thread and let it run concurrently with this function's own
    attachment-preacquisition/blob-flush/head-lookup preamble, then pay only
    the (likely already-finished) ``Future.result()`` wait). Only ever
    consulted for ``position == 0`` (the chain's sole full-replace write --
    every other position is a ``merge_append``, which
    ``write_parsed_session_to_archive`` never accepts prepared rows for
    regardless). A missing key, ``None`` value, or a value that turns out
    stale (session content hash mismatch, or lineage tail-slicing changed
    ``messages`` after it was built) is always safe: ``write_parsed_session_
    to_archive`` falls back to building rows inline, byte-identical to
    ``prepared_by_raw_id=None``.

    ``manage_transaction=False`` batches this cohort's index.db writes
    and terminal source.db parse-state markers into the caller's open
    transaction/pending-state instead of committing them immediately
    (polylogue-oikv) -- the caller must call ``commit()`` (or
    ``rollback()`` on failure) itself, exactly once per batch, after
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

    ``skip_already_applied`` (polylogue-de2a, default ``False`` so every
    existing caller is byte-for-byte unchanged) skips the index write --
    not the parse, not the bookkeeping/receipt writes -- for every
    ``plan.accepted_raw_ids`` entry at or before the logical source's
    current ``raw_revision_heads.accepted_raw_id``. Without this, every
    single live append replays the WHOLE accumulated chain from its
    proven baseline through every previously accepted append: each of
    those historical positions is already durably indexed from an
    earlier call, so re-running ``_index_parsed_for_retained_raw`` for
    them is redundant ``INSERT OR REPLACE`` churn against ``messages``/
    ``blocks`` that re-triggers their ``messages_fts`` insert triggers
    for content that never changed. A long-lived session accumulating N
    small live appends over its lifetime pays O(N) redundant historical
    writes on its Nth append and O(N^2) cumulatively over its life --
    this is the confirmed root cause of the multi-minute/multi-hour
    writer-gate holds in polylogue-de2a (observed 860s and 9297s single
    holds), which in turn starved every other periodic write actor (FTS
    merge, WAL checkpoint) for the entire hold. Only the live append
    path opts in; the backfill/restore/membership replay paths keep the
    full self-healing re-apply (their content may predate a parser fix
    and legitimately need every historical position rewritten, and they
    run far less often than a live append).
    """
    if not plan.accepted_raw_ids:
        raise ValueError("cannot apply a revision plan without an accepted chain")
    candidates = {item.raw_id: item for item in _raw_revision_candidates(store, plan.logical_source_key)}
    aggregate_sessions = merge_parsed_session_chunks(parsed_by_raw_id[raw_id] for raw_id in plan.accepted_raw_ids)
    if len(aggregate_sessions) != 1:
        raise RuntimeError("one logical revision chain did not compose to exactly one session")
    aggregate_content_hash = bytes.fromhex(session_content_hash(aggregate_sessions[0]))
    attachments_by_raw_id: dict[str, dict[int, tuple[bytes | None, int, str]]] = {}
    attachment_refs_by_raw_id: dict[str, tuple[ArchiveSourceBlobRef, ...]] = {}
    for raw_id in plan.accepted_raw_ids:
        _provider, _blob_hash, source_path, _kind, _blob_size = raw_revision_descriptor(store, raw_id)
        acquired, refs = store._preacquire_attachment_blobs(
            parsed_by_raw_id[raw_id],
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
        )
        attachments_by_raw_id[raw_id] = acquired
        attachment_refs_by_raw_id[raw_id] = refs
    if store._blob_publisher is not None:
        store._blob_publisher.flush()
    for raw_id, refs in attachment_refs_by_raw_id.items():
        write_source_blob_refs(store._ensure_source_conn(), raw_id, refs)
    session_ids: set[str] = set()
    with store._conn if manage_transaction else nullcontext():
        existing_head = store._conn.execute(
            """SELECT session_id, accepted_raw_id, accepted_source_revision,
                      accepted_content_hash, accepted_frontier_kind, accepted_frontier
               FROM raw_revision_heads WHERE logical_source_key = ?""",
            (plan.logical_source_key,),
        ).fetchone()
        # Captured before any clearing below (the quarantined-membership
        # fold path nulls ``existing_head`` further down): the previous
        # run's accepted tip is exactly the boundary between "already
        # durably indexed" and "new tail" for THIS byte chain. If it
        # isn't present in the current chain at all (a fresh chain, a
        # cleared/superseded membership head, or a discontinuity), the
        # lookup below naturally falls back to "no skip" -- every
        # position gets indexed, identical to today's behavior.
        previously_accepted_raw_id = str(existing_head[1]) if existing_head is not None else None
        already_indexed_upto = -1
        if skip_already_applied and previously_accepted_raw_id is not None:
            try:
                already_indexed_upto = plan.accepted_raw_ids.index(previously_accepted_raw_id)
            except ValueError:
                already_indexed_upto = -1
        accepted_frontier_kind = (
            "semantic" if existing_head is not None and str(existing_head[4]) == "semantic" else "byte"
        )
        if accepted_frontier_kind == "semantic":
            accepted_projection = session_revision_projection(aggregate_sessions[0])
            accepted_frontier = (
                len(accepted_projection.message_hashes)
                + len(accepted_projection.event_hashes)
                + len(accepted_projection.attachment_identities)
            )
        else:
            accepted_frontier = None
        if (
            existing_head is not None
            and accepted_frontier_kind == "semantic"
            and accepted_frontier is not None
            and str(existing_head[4]) == "semantic"
            and _raw_revision_authority(store, str(existing_head[1])) == "quarantined"
        ):
            # The current head was written by membership replay for a
            # quarantined-authority raw (e.g. a browser-capture snapshot of
            # the same provider conversation). Chain evidence with
            # source-tier revision governance outranks quarantined capture
            # evidence unconditionally: a scalar semantic frontier cannot
            # prove the capture is a content-superset (a divergent capture
            # with more units is not "ahead"), so no count comparison is
            # attempted. The capture raw stays in the source tier and its
            # earlier receipts remain; re-adopting genuinely
            # content-ahead capture tails needs a real prefix-dominance
            # proof (follow-up bead), not a unit count.
            store._conn.execute(
                "DELETE FROM raw_revision_heads WHERE logical_source_key = ?",
                (plan.logical_source_key,),
            )
            existing_head = None
        for position, raw_id in enumerate(plan.accepted_raw_ids):
            if position <= already_indexed_upto:
                # Already durably written by an earlier accepted replay
                # of this exact byte chain (see ``skip_already_applied``
                # above) -- its session_id is deterministic from its own
                # parsed content, so recover it without re-running the
                # write.
                parsed = parsed_by_raw_id[raw_id]
                session_ids.add(str(make_session_id(parsed.source_name, parsed.provider_session_id)))
                continue
            # polylogue-fpid: only position 0 is ever a full-replace write
            # (every later position is merge_append, source_index=-1 above),
            # so a prepared entry is only ever resolved/consulted there --
            # see this function's docstring for the resolve-here-not-earlier
            # rationale and the always-safe stale-prepared fallback.
            resolved_prepared: PreparedSessionRows | None = None
            if position == 0 and prepared_by_raw_id is not None:
                prepared_candidate = prepared_by_raw_id.get(raw_id)
                if isinstance(prepared_candidate, Future):
                    resolved_prepared = prepared_candidate.result()
                else:
                    resolved_prepared = prepared_candidate
            index_started = time.perf_counter()
            result = _index_parsed_for_retained_raw(
                store,
                parsed_by_raw_id[raw_id],
                raw_id=raw_id,
                source_index=0 if position == 0 else -1,
                stage_timings_s=stage_timings_s,
                stage_timing_prefix=stage_timing_prefix,
                manage_transaction=False,
                preacquired_attachment_blobs=attachments_by_raw_id[raw_id],
                finalize_raw_parse=False,
                revision_authoritative=True,
                bulk_fts=bulk_fts,
                bulk_build=bulk_build,
                defer_fts_rebuild=not bulk_build,
                prepared=resolved_prepared,
            )
            if stage_timings_s is not None:
                key = f"{stage_timing_prefix}.index_parsed_write"
                stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - index_started)
            session_ids.add(result.session_id)
        if len(session_ids) != 1:
            raise RuntimeError("one logical revision chain produced multiple session ids")
        session_id = next(iter(session_ids))
        store._conn.execute(
            "UPDATE sessions SET content_hash = ? WHERE session_id = ?",
            (aggregate_content_hash, session_id),
        )
        if not bulk_build and not defer_fts:
            repair_message_fts_index_sync(store._conn, [session_id], record_exact_snapshot=False)
        if defer_fts:
            from polylogue.storage.fts.freshness import record_fts_surface_stale_preserving_counts_sync

            record_fts_surface_stale_preserving_counts_sync(
                store._conn,
                surface="messages_fts",
                detail="live authoritative replay deferred targeted session FTS repair",
            )
        assert_session_fts_exact_sync(
            store._conn,
            session_id,
            bulk_build=bulk_build,
            allow_pending=defer_fts,
        )
        stored = store._conn.execute("SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if stored is None or not isinstance(stored[0], bytes):
            raise RuntimeError("accepted revision did not produce a hashed session")
        accepted_raw_id = plan.accepted_raw_ids[-1]
        accepted = candidates[accepted_raw_id]
        fold_authorization = (
            _authorize_full_snapshot_fold(
                store, existing_head=tuple(existing_head), full_candidate=accepted, candidates=candidates
            )
            if existing_head is not None and accepted_frontier_kind == "byte"
            else None
        )
        decided_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        for application in plan.applications:
            candidate = candidates[application.raw_id]
            has_head = application.accepted_raw_id is not None
            record_revision_application_sync(
                store._conn,
                RevisionApplicationReceipt(
                    raw_id=candidate.raw_id,
                    session_id=session_id,
                    logical_source_key=plan.logical_source_key,
                    source_revision=candidate.source_revision,
                    acquisition_generation=accepted.acquisition_generation
                    if has_head
                    else candidate.acquisition_generation,
                    decision=application.decision,
                    accepted_raw_id=accepted_raw_id if has_head else None,
                    accepted_source_revision=accepted.source_revision if has_head else None,
                    accepted_content_hash=stored[0] if has_head else None,
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
                ),
                decided_at_ms=decided_at_ms,
            )
    terminal_raw_ids = {
        application.raw_id
        for application in plan.applications
        if application.decision
        in {
            ApplicationDecision.SELECTED_BASELINE,
            ApplicationDecision.APPLIED_APPEND,
            ApplicationDecision.SUPERSEDED,
        }
    }
    for raw_id in terminal_raw_ids:
        provider, _blob_hash, _source_path, _kind, _blob_size = raw_revision_descriptor(store, raw_id)
        if manage_transaction:
            mark_raw_parse_succeeded(store, raw_id, provider=provider)
        else:
            store._pending_raw_parse_states.append((raw_id, _raw_parse_success_state(provider)))
    return session_id, plan.accepted_raw_ids


def apply_raw_membership_classification(
    store: RawRevisionGovernanceHost,
    logical_source_key: str,
    classification: MembershipClassification,
    parsed_by_raw_id: dict[str, ParsedSession],
    projections_by_raw_id: dict[str, SessionRevisionProjection],
    *,
    acquired_at_ms: int,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "membership_replay",
    manage_transaction: bool = True,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    defer_fts: bool = False,
) -> str | None:
    """Apply one semantic member head and persist every membership decision.

    ``manage_transaction=False`` batches this cohort's index.db head
    write, its source.db membership-decision updates, and its terminal
    source.db parse-state marker into the caller's open
    transaction/pending-state instead of committing them immediately
    (polylogue-oikv) -- see ``apply_raw_revision_replay`` for the shared
    batch-commit invariant this mirrors.

    ``bulk_fts`` mirrors ``apply_raw_revision_replay``'s guard-gated bulk
    FTS mode (polylogue-crd8); default ``False``. ``bulk_build`` mirrors
    its broader bulk-generation-build lifecycle (polylogue-v6i3); default
    ``False``.
    """
    conn = store._ensure_source_conn()
    decided_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    decisions: dict[str, str] = dict.fromkeys(classification.ambiguous_raw_ids, "ambiguous")
    # "superseded_equivalent" asserts an accepted chain superseded the
    # member. With no accepted head, equivalence collapses back into the
    # unresolved cohort: labeling it superseded (and, downstream,
    # byte_proven) fabricates authority for a head that was never written
    # -- 914 headless-but-"byte_proven" logical sources on the 2026-07-20
    # rebuild walk came from exactly this mislabel.
    decisions.update(
        dict.fromkeys(
            classification.equivalent_raw_ids,
            "superseded_equivalent" if classification.accepted_raw_ids else "ambiguous",
        )
    )
    for raw_id in classification.accepted_raw_ids[:-1]:
        decisions[raw_id] = "superseded_prefix"
    session_id: str | None = None
    # Ambiguous evidence is debt, not deletion authority. A later branch
    # must not erase the last accepted session/head; a cold rebuild simply
    # has no accepted state to preserve.
    if classification.accepted_raw_ids:
        accepted_raw_id = classification.accepted_raw_ids[-1]
        accepted_session = parsed_by_raw_id[accepted_raw_id]
        _provider, _blob_hash, source_path, _kind, _blob_size = raw_revision_descriptor(store, accepted_raw_id)
        attachments, refs = store._preacquire_attachment_blobs(
            accepted_session,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
        )
        if store._blob_publisher is not None:
            store._blob_publisher.flush()
        write_source_blob_refs(conn, accepted_raw_id, refs)
        with store._conn if manage_transaction else nullcontext():
            existing_head = store._conn.execute(
                """
                SELECT accepted_raw_id, accepted_content_hash, accepted_frontier_kind, session_id,
                       accepted_frontier
                FROM raw_revision_heads WHERE logical_source_key = ?
                """,
                (logical_source_key,),
            ).fetchone()
            yield_to_head_raw_id: str | None = None
            if existing_head is not None:
                existing_raw_id = str(existing_head[0])
                classified_raw_ids = {
                    *classification.accepted_raw_ids,
                    *classification.equivalent_raw_ids,
                }
                chain_head_authority = (
                    _raw_revision_authority(store, existing_raw_id)
                    if existing_raw_id not in classified_raw_ids
                    else None
                )
                if existing_raw_id not in classified_raw_ids and chain_head_authority not in (
                    None,
                    "quarantined",
                ):
                    # The head is owned by chain-governed (non-quarantined)
                    # source evidence outside this quarantined membership
                    # cohort -- e.g. a byte-proven provider export of the
                    # same conversation this browser capture snapshotted.
                    # Provider-export evidence outranks quarantined capture
                    # evidence unconditionally: a scalar semantic frontier
                    # cannot prove the capture is a content-superset, so
                    # the cohort always yields (receipted below); the
                    # capture bytes stay in the source tier. Re-adopting a
                    # genuinely content-ahead capture tail needs a real
                    # prefix-dominance proof (follow-up bead).
                    yield_to_head_raw_id = existing_raw_id
                else:
                    persisted_session = store._conn.execute(
                        "SELECT raw_id, content_hash FROM sessions WHERE session_id = ?",
                        (str(existing_head[3]),),
                    ).fetchone()
                    # Retiring an IN-COHORT head only requires that the
                    # persisted session row was also written by this
                    # cohort (head raw or any classified member): both
                    # representative drift across resumed passes (cohort
                    # absorption can flip which equivalent member wrote
                    # the session row) and content-hash drift (parser
                    # fixes between resumed passes re-derive hashes; the
                    # same-raw CAS exemption already treats that as
                    # re-derivation, not conflict) are healed immediately
                    # by this very replay re-indexing the accepted
                    # member. Only a persisted session written by a raw
                    # FOREIGN to the cohort still refuses -- that is the
                    # genuine unrelated-head hazard this guard exists
                    # for.
                    persisted_raw = None if persisted_session is None else str(persisted_session[0])
                    persisted_head_authority = (
                        _raw_revision_authority(store, persisted_raw)
                        if persisted_raw is not None and persisted_raw not in classified_raw_ids
                        else None
                    )
                    if persisted_head_authority not in (None, "quarantined"):
                        # A resumed membership pass may have already
                        # installed a quarantined cohort representative
                        # into raw_revision_heads while the persisted
                        # session still belongs to an older byte-governed
                        # head. The persisted row is the authoritative
                        # claim in that mixed state; yield rather than
                        # treating its byte head as an unrelated hazard.
                        persisted_revision = conn.execute(
                            """
                            SELECT source_revision, acquisition_generation,
                                   append_end_offset, blob_size
                            FROM raw_sessions WHERE raw_id = ?
                            """,
                            (persisted_raw,),
                        ).fetchone()
                        if persisted_revision is None or persisted_revision[0] is None:
                            raise RuntimeError("persisted byte-governed session lacks revision evidence")
                        store._conn.execute(
                            """
                            UPDATE raw_revision_heads
                            SET accepted_raw_id = ?, accepted_source_revision = ?,
                                accepted_content_hash = ?, accepted_frontier_kind = 'byte',
                                accepted_frontier = ?, acquisition_generation = ?,
                                append_end_offset = ?
                            WHERE logical_source_key = ?
                            """,
                            (
                                persisted_raw,
                                str(persisted_revision[0]),
                                persisted_session[1],
                                int(persisted_revision[2] or persisted_revision[3]),
                                int(persisted_revision[1] or 0),
                                persisted_revision[2],
                                logical_source_key,
                            ),
                        )
                        yield_to_head_raw_id = persisted_raw
                    if yield_to_head_raw_id is None and (
                        existing_raw_id not in classified_raw_ids
                        or persisted_session is None
                        or (persisted_raw != existing_raw_id and persisted_raw not in classified_raw_ids)
                    ):
                        raise RuntimeError(
                            "membership replay cannot retire an unrelated accepted head: "
                            f"logical_source_key={logical_source_key!r} "
                            f"existing_head(raw_id={existing_raw_id!r}, session_id={str(existing_head[3])!r}, "
                            f"authority={chain_head_authority!r}) "
                            f"cohort(accepted={classification.accepted_raw_ids!r}, "
                            f"equivalent={classification.equivalent_raw_ids!r}, "
                            f"ambiguous={classification.ambiguous_raw_ids!r}) "
                            f"persisted_session_raw={None if persisted_session is None else str(persisted_session[0])!r}"
                        )
                    # polylogue-miwv: #3211 removed a byte-governance
                    # refusal here on the theory that this branch is only
                    # reachable after a real governance conversion (a
                    # still-set logical_source_key being interrupted-pass
                    # drift, not live evidence). That premise is false for
                    # the accepted head specifically: ``_apply_membership_
                    # sessions`` (sources/live/batch.py) unconditionally
                    # injects the current ``raw_revision_heads`` accepted
                    # raw into the comparison cohort even when it has
                    # NEVER been through a membership conversion -- the
                    # #2718 scenario this whole code path exists for
                    # (a byte-governed head being compared against
                    # membership-discovered content for the first time).
                    # Content-prefix growth alone cannot prove the older
                    # bundle raw supersedes a head that still has live,
                    # unresolved byte-append evidence hanging off it (a
                    # quarantined/pending append raw whose
                    # ``predecessor_source_revision`` chains to the head's
                    # own ``source_revision``) that this classification
                    # pass never saw. Only matters when replay is about to
                    # CHANGE the accepted raw (mirrors #2718's original
                    # ``accepted_raw_id != existing_raw_id`` guard
                    # condition) -- a classification that keeps the same
                    # existing_raw_id as the accepted member (e.g. the
                    # head's own content already dominates every other
                    # cohort member) is a no-op re-affirmation, not a
                    # replacement, regardless of dangling evidence.
                    # Narrower than #2718's original blanket
                    # ``logical_source_key IS NOT NULL`` check so #3211's
                    # own interrupted-pass-drift resumption (no dangling
                    # append descendant, just a stale un-nulled key) is
                    # unaffected.
                    if yield_to_head_raw_id is None and accepted_raw_id != existing_raw_id:
                        classified_placeholders = ", ".join("?" for _ in classified_raw_ids) or "NULL"
                        dangling_append = conn.execute(
                            f"""
                            SELECT 1
                            FROM raw_sessions AS child
                            WHERE child.logical_source_key = ?
                              AND child.raw_id != ?
                              AND child.raw_id NOT IN ({classified_placeholders})
                              AND child.predecessor_source_revision IS NOT NULL
                              AND child.predecessor_source_revision = (
                                  SELECT source_revision FROM raw_sessions WHERE raw_id = ?
                              )
                            LIMIT 1
                            """,
                            (logical_source_key, existing_raw_id, *classified_raw_ids, existing_raw_id),
                        ).fetchone()
                        if dangling_append is not None:
                            raise RuntimeError(
                                "membership replay cannot replace a head with unresolved byte-append "
                                f"evidence: logical_source_key={logical_source_key!r} "
                                f"existing_head(raw_id={existing_raw_id!r})"
                            )
                    if yield_to_head_raw_id is None:
                        store._conn.execute(
                            "DELETE FROM raw_revision_heads WHERE logical_source_key = ?",
                            (logical_source_key,),
                        )
            if yield_to_head_raw_id is not None:
                assert existing_head is not None
                session_id = str(existing_head[3])
                cohort_raw_ids = (
                    *classification.accepted_raw_ids,
                    *classification.equivalent_raw_ids,
                    *classification.ambiguous_raw_ids,
                )
                for generation, raw_id in enumerate(cohort_raw_ids):
                    projection = projections_by_raw_id[raw_id]
                    decisions[raw_id] = "superseded_equivalent"
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
                    defer_fts_rebuild=not bulk_build,
                )
                if stage_timings_s is not None:
                    key = f"{stage_timing_prefix}.index_parsed_write"
                    stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - index_started)
                session_id = result.session_id
                if not bulk_build and not defer_fts:
                    repair_message_fts_index_sync(store._conn, [session_id], record_exact_snapshot=False)
                if defer_fts:
                    from polylogue.storage.fts.freshness import record_fts_surface_stale_preserving_counts_sync

                    record_fts_surface_stale_preserving_counts_sync(
                        store._conn,
                        surface="messages_fts",
                        detail="live membership replay deferred targeted session FTS repair",
                    )
                assert_session_fts_exact_sync(
                    store._conn,
                    session_id,
                    bulk_build=bulk_build,
                    allow_pending=defer_fts,
                )
                stored = store._conn.execute(
                    "SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
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
                )
                for generation, raw_id in enumerate(cohort_raw_ids):
                    projection = projections_by_raw_id[raw_id]
                    decision = decisions.get(raw_id, "applied")
                    record_revision_application_sync(
                        store._conn,
                        RevisionApplicationReceipt(
                            raw_id=raw_id,
                            session_id=session_id,
                            logical_source_key=logical_source_key,
                            source_revision=projection.session_hash.hex(),
                            acquisition_generation=generation,
                            decision=(
                                ApplicationDecision.AMBIGUOUS
                                if decision == "ambiguous"
                                else ApplicationDecision.SUPERSEDED
                                if decision.startswith("superseded")
                                else ApplicationDecision.SELECTED_BASELINE
                            ),
                            accepted_raw_id=accepted_raw_id if decision != "ambiguous" else None,
                            accepted_source_revision=(
                                accepted_projection.session_hash.hex() if decision != "ambiguous" else None
                            ),
                            accepted_content_hash=stored[0] if decision != "ambiguous" else None,
                            accepted_frontier_kind="semantic" if decision != "ambiguous" else None,
                            accepted_frontier=semantic_frontier if decision != "ambiguous" else None,
                            detail=f"membership:{decision}",
                        ),
                        decided_at_ms=decided_at_ms,
                    )
        if yield_to_head_raw_id is None:
            decisions[accepted_raw_id] = "applied"

    with conn if manage_transaction else nullcontext():
        for raw_id, decision in decisions.items():
            conn.execute(
                """
                UPDATE raw_session_memberships
                SET decision = ?, decided_at_ms = ?,
                    revision_authority = ?,
                    acquisition_generation = ?
                WHERE raw_id = ? AND logical_source_key = ?
                """,
                (
                    decision,
                    decided_at_ms,
                    "quarantined" if decision in {"ambiguous", "deferred"} else "byte_proven",
                    classification.accepted_raw_ids.index(raw_id) if raw_id in classification.accepted_raw_ids else 0,
                    raw_id,
                    logical_source_key,
                ),
            )
    for raw_id in decisions:
        complete = conn.execute(
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
        ).fetchone()
        if complete is not None and bool(complete[0]):
            provider, _blob_hash, _source_path, _kind, _blob_size = raw_revision_descriptor(store, raw_id)
            if manage_transaction:
                mark_raw_parse_succeeded(store, raw_id, provider=provider)
            else:
                store._pending_raw_parse_states.append((raw_id, _raw_parse_success_state(provider)))
        else:
            # Incomplete cohort: this raw is not yet finalized (a sibling
            # member is still undecided), so this is a correction, not a
            # terminal marker -- always commits immediately regardless of
            # batching, matching prior behavior.
            with conn:
                conn.execute(
                    "UPDATE raw_sessions SET parsed_at_ms = NULL, parse_error = NULL WHERE raw_id = ?",
                    (raw_id,),
                )
    return session_id


def finalize_raw_parse_state(store: RawRevisionGovernanceHost, raw_id: str, *, state: RawSessionStateUpdate) -> None:
    """Commit one typed source parse state after its index outcome."""
    apply_source_raw_state_update(
        store._ensure_source_conn(),
        raw_id,
        state=state,
        manage_transaction=True,
    )


def mark_raw_parse_failed(
    store: RawRevisionGovernanceHost, raw_id: str, *, provider: Provider, error: BaseException
) -> None:
    """Persist a bounded parse/index failure for retained raw evidence."""
    finalize_raw_parse_state(store, raw_id, state=_raw_parse_failure_state(provider, error))


def mark_raw_parse_succeeded(store: RawRevisionGovernanceHost, raw_id: str, *, provider: Provider) -> None:
    """Finalize one retained raw payload after every derived session commits."""
    finalize_raw_parse_state(store, raw_id, state=_raw_parse_success_state(provider))


def _flush_pending_raw_parse_states(store: RawRevisionGovernanceHost) -> None:
    if not store._pending_raw_parse_states:
        return
    source_conn = store._ensure_source_conn()
    with source_conn:
        for raw_id, state in store._pending_raw_parse_states:
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
    preacquired_attachment_blobs: dict[int, tuple[bytes | None, int, str]],
    finalize_raw_parse: bool,
    revision_authoritative: bool = False,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    defer_fts_rebuild: bool = False,
    prepared: PreparedSessionRows | None = None,
) -> ArchiveRawParsedWriteResult:
    provider = Provider.from_string(session.source_name)
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
            defer_fts_rebuild=defer_fts_rebuild,
            prepared=prepared,
        )
    except Exception as exc:
        finalize_raw_parse_state(store, raw_id, state=_raw_parse_failure_state(provider, exc))
        raise
    if finalize_raw_parse:
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
    error = f"{type(exc).__name__}: {exc}"[:2000]
    return RawSessionStateUpdate(
        parse_error=error,
        payload_provider=provider,
        detection_warnings=error[:500],
    )


def write_raw_and_parsed_result(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    payload: bytes,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    raw_id: str | None = None,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    manage_transaction: bool = True,
    blob_publication_receipt_id: str | None = None,
    finalize_raw_parse: bool = True,
) -> ArchiveRawParsedWriteResult:
    """Write raw acquisition bytes and return write/skip counts.

    The durable source write always commits promptly so parallel publishers
    can establish their reservations. ``manage_transaction=False`` batches
    only the rebuildable index write; holding a source transaction across
    worker results would block the next pre-publication reservation.
    """

    def add_timing(name: str, started_at: float) -> None:
        if stage_timings_s is not None:
            key = f"{stage_timing_prefix}.{name}"
            stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - started_at)

    if store._blob_publisher is None:
        raise RuntimeError("raw archive writes require a writable archive publisher")
    if blob_publication_receipt_id is None:
        raw_hash, _raw_size = store._blob_publisher.write_from_bytes(payload)
        blob_publication_receipt_id = store._blob_publisher.receipt_id(raw_hash)
    preacquired_attachments, attachment_blob_refs = store._preacquire_attachment_blobs(
        session,
        source_path=source_path,
        acquired_at_ms=acquired_at_ms,
    )
    store._blob_publisher.flush()
    t0 = time.perf_counter()
    source_conn = store._ensure_source_conn()
    add_timing("source_connect", t0)
    t0 = time.perf_counter()
    raw_id = write_source_raw_session(
        source_conn,
        origin=origin_from_provider(session.source_name),
        capture_mode=session.source_name,
        source_path=source_path,
        source_index=source_index,
        native_id=session.provider_session_id,
        raw_id=raw_id,
        payload=payload,
        acquired_at_ms=acquired_at_ms,
        blob_publication_receipt_id=blob_publication_receipt_id,
        additional_blob_refs=attachment_blob_refs,
        manage_transaction=True,
    )
    add_timing("source_raw_write", t0)
    t0 = time.perf_counter()
    result = _index_parsed_for_retained_raw(
        store,
        session,
        raw_id=raw_id,
        source_index=source_index,
        stage_timings_s=stage_timings_s,
        stage_timing_prefix=stage_timing_prefix,
        manage_transaction=manage_transaction,
        preacquired_attachment_blobs=preacquired_attachments,
        finalize_raw_parse=finalize_raw_parse,
    )
    add_timing("index_parsed_write", t0)
    return result


def write_raw_blob_and_parsed(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    blob_hash_hex: str,
    blob_size: int,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    raw_id: str | None = None,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "full",
    manage_transaction: bool = True,
    blob_publication_receipt_id: str | None = None,
    finalize_raw_parse: bool = True,
) -> tuple[str, str]:
    """Write parsed session metadata for an already-materialized raw blob."""
    result = write_raw_blob_and_parsed_result(
        store,
        session,
        blob_hash_hex=blob_hash_hex,
        blob_size=blob_size,
        source_path=source_path,
        acquired_at_ms=acquired_at_ms,
        source_index=source_index,
        raw_id=raw_id,
        stage_timings_s=stage_timings_s,
        stage_timing_prefix=stage_timing_prefix,
        manage_transaction=manage_transaction,
        blob_publication_receipt_id=blob_publication_receipt_id,
        finalize_raw_parse=finalize_raw_parse,
    )
    return result.raw_id, result.session_id


def write_raw_blob_and_parsed_result(
    store: RawRevisionGovernanceHost,
    session: ParsedSession,
    *,
    blob_hash_hex: str,
    blob_size: int,
    source_path: str,
    acquired_at_ms: int,
    source_index: int = 0,
    raw_id: str | None = None,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "full",
    manage_transaction: bool = True,
    blob_publication_receipt_id: str | None = None,
    finalize_raw_parse: bool = True,
) -> ArchiveRawParsedWriteResult:
    """Write parsed metadata for a raw blob and return write/skip counts.

    See :meth:`write_raw_and_parsed_result` for the transaction contract.
    """

    def add_timing(name: str, started_at: float) -> None:
        if stage_timings_s is not None:
            key = f"{stage_timing_prefix}.{name}"
            stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - started_at)

    preacquired_attachments, attachment_blob_refs = store._preacquire_attachment_blobs(
        session,
        source_path=source_path,
        acquired_at_ms=acquired_at_ms,
    )
    if store._blob_publisher is not None:
        store._blob_publisher.flush()
    t0 = time.perf_counter()
    source_conn = store._ensure_source_conn()
    add_timing("source_connect", t0)
    t0 = time.perf_counter()
    raw_id = write_source_raw_session_blob_ref(
        source_conn,
        origin=origin_from_provider(session.source_name),
        capture_mode=session.source_name,
        source_path=source_path,
        source_index=source_index,
        native_id=session.provider_session_id,
        raw_id=raw_id,
        blob_hash=bytes.fromhex(blob_hash_hex),
        blob_size=blob_size,
        acquired_at_ms=acquired_at_ms,
        blob_publication_receipt_id=blob_publication_receipt_id,
        additional_blob_refs=attachment_blob_refs,
        manage_transaction=True,
    )
    add_timing("source_raw_blob_ref_write", t0)
    t0 = time.perf_counter()
    result = _index_parsed_for_retained_raw(
        store,
        session,
        raw_id=raw_id,
        source_index=source_index,
        stage_timings_s=stage_timings_s,
        stage_timing_prefix=stage_timing_prefix,
        manage_transaction=manage_transaction,
        preacquired_attachment_blobs=preacquired_attachments,
        finalize_raw_parse=finalize_raw_parse,
    )
    add_timing("index_parsed_write", t0)
    return result

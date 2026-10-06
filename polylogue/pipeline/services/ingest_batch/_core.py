"""Batch ingest orchestration with shared pure compute and caller-owned SQL.

Decode, validation, parsing, and transformation use the bounded compute
adapter. The caller applies completed results through its synchronous writer,
without retaining the whole parsed batch in memory.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_FLAGS
from polylogue.core.enums import Provider
from polylogue.core.raw_failure_evidence import CohortMembershipRefusalError, RetainedRawDecodeRefusalError
from polylogue.core.timestamp_authority import session_evidence_timestamps
from polylogue.logging import get_logger
from polylogue.pipeline.ids import (
    message_content_identity,
    session_content_hash,
)
from polylogue.pipeline.payload_types import ParseBatchObservation
from polylogue.pipeline.services.ingest_worker import (
    SessionWritePayload,
)
from polylogue.sinex.material_adapter import (
    PublicationEncodingError,
)
from polylogue.sinex.models import PublicationMode
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.attachment_reasons import AttachmentOwnerResolutionReason
from polylogue.storage.blob_publication import (
    ArchiveBlobPublisher,
)
from polylogue.storage.runtime import RawSessionRecord
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
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveWriteOutcome,
    ConnectionSessionSourceRead,
    LineageSignatureCache,
    PreparedSessionWrite,
    PreparedSessionWriteRefusedError,
    _composed_db_signatures,
    _message_content_hash,
    _normalized_message_native_id,
    _parsed_message_signature,
    _retain_stale_session_observations,
    prepare_session_write,
    raw_source_path,
    recorded_attachment_owner_gaps,
    replace_parser_ingest_flag_tags,
    upsert_parser_ingest_flag_tags,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.reference_seal import (
    current_index_mutation_scope,
)

if TYPE_CHECKING:
    from polylogue.core.protocols import ProgressCallback
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


logger = get_logger(__name__)


IngestHeartbeat = Callable[[], None]
# A heartbeat reports coordinator liveness; cancellation or a typed result
# settles the owned worker future, without an elapsed progress deadline.


# Sync DB writer
# ---------------------------------------------------------------------------


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _append_owner_resolutions(
    resolutions: list[dict[str, str]],
    payload: SessionWritePayload,
    gaps: Sequence[tuple[str, AttachmentOwnerResolutionReason]],
) -> None:
    for attachment_id, reason in gaps:
        resolutions.append(
            {
                "raw_id": payload.raw_id or "",
                "session_id": payload.session_id,
                "attachment_id": attachment_id,
                "reason": reason.value,
            }
        )


def _incoming_has_ingest_flag(payload: SessionWritePayload, flag: str | Sequence[str]) -> bool:
    flags = (flag,) if isinstance(flag, str) else flag
    return any(candidate in payload.parsed_session.ingest_flags for candidate in flags)


_FTS_REPAIR_COUNT_KEY = "_fts_repair"


def _needs_session_fts_repair(conn: sqlite3.Connection, session_id: str) -> bool:
    """Ask the FTS domain whether this unchanged session's partition drifted.

    Content-changed sessions are republished unconditionally; this covers the
    re-ingest of unchanged content, where the partition can still be stale from
    an interrupted earlier write. The rule is the FTS derivation's own
    inspection, so there is no second staleness definition to drift from it.
    """
    from polylogue.storage.fts.derivation import session_partition_is_valid_sync

    return not session_partition_is_valid_sync(conn, session_id)


def _composed_message_owners(conn: sqlite3.Connection, message_ids: Sequence[str]) -> dict[str, tuple[str, str | None]]:
    """Owning session and native id of each composed message, own and inherited alike."""
    found: dict[str, tuple[str, str | None]] = {}
    pending = list(dict.fromkeys(message_ids))
    for start in range(0, len(pending), 500):
        batch = pending[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        for message_id, session_id, native_id in conn.execute(
            f"SELECT message_id, session_id, native_id FROM messages WHERE message_id IN ({placeholders})",
            batch,
        ):
            found[str(message_id)] = (str(session_id), None if native_id is None else str(native_id).strip() or None)
    return found


def _append_delta_payload(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
) -> tuple[ParsedSession | None, int]:
    """Select the messages an append payload adds to the composed transcript.

    The composed transcript is the inherited lineage prefix plus the
    session's own rows. At the replay cursor an inherited message is matched
    by content signature: a child replays its parent's prefix under its own
    provider ids, and the writer stored none of those copies, so no id can
    match it. A message the session owns is matched by native identity when
    both sides carry one, and by content signature only when either lacks
    it. A tail-only append that repeats earlier content under a new native
    id is therefore new, not a replay.
    """
    existing_logical = _composed_db_signatures(conn, payload.session_id)
    owners = _composed_message_owners(conn, [message_id for message_id, _signature in existing_logical])
    composed: list[tuple[str | None, str, bool]] = []
    for message_id, signature in existing_logical:
        owner_session_id, composed_native_id = owners.get(message_id, (payload.session_id, None))
        composed.append((composed_native_id, signature, owner_session_id != payload.session_id))
    composed_native_ids = {native_id for native_id, _signature, _inherited in composed if native_id is not None}
    delta_messages: list[ParsedMessage] = []
    logical_prefix = 0
    for message in payload.parsed_session.messages:
        native_id = _normalized_message_native_id(message)
        signature = _parsed_message_signature(message)
        if logical_prefix < len(composed):
            composed_native_id, composed_signature, inherited = composed[logical_prefix]
            is_replayed_prefix = (
                native_id == composed_native_id
                if not inherited and native_id is not None and composed_native_id is not None
                else signature == composed_signature
            )
            if is_replayed_prefix:
                logical_prefix += 1
                continue
        if native_id is not None and native_id in composed_native_ids:
            continue
        delta_messages.append(message.model_copy(update={"position": None}))
    if not delta_messages and not payload.attachment_count:
        return None, len(payload.parsed_session.messages)
    # polylogue-3hfl7: the copy must NOT inherit the full session's bound
    # ``content_hash``. ``ParsedSession.content_hash`` is the parse-side
    # identity carrier every digest consumer prefers over recomputing
    # (``pipeline.ids.bound_session_content_hash``), so a delta that kept it
    # would report the MERGED session's digest while carrying only the new
    # messages -- ``prepare_session_rows(delta).session_content_hash`` would
    # name a row set it does not cover. Rebind it to the delta's own digest.
    delta = payload.parsed_session.model_copy(update={"messages": delta_messages, "content_hash": None})
    delta = delta.model_copy(update={"content_hash": str(session_content_hash(delta))})
    return delta, len(payload.parsed_session.messages) - len(delta_messages)


def _append_payload_changes_existing_message(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
) -> bool:
    """Return whether an incoming append revision changes an existing message.

    Native ids identify the same message across a growing provider revision,
    but the message's blocks can still be revised. A newer full revision must
    replace such overlap before appending its tail, otherwise the resulting
    session depends on arrival order.
    """
    existing_rows = {
        str(row[0]).strip(): (int(row[1]), int(row[2]), row[3])
        for row in conn.execute(
            """
            SELECT native_id, position, variant_index, content_hash
            FROM messages
            WHERE session_id = ? AND native_id IS NOT NULL
            """,
            (payload.session_id,),
        ).fetchall()
    }
    for message in payload.parsed_session.messages:
        native_id = _normalized_message_native_id(message)
        existing = existing_rows.get(native_id) if native_id is not None else None
        if existing is None:
            continue
        position, variant_index, existing_hash = existing
        incoming_hash = _message_content_hash(
            payload.session_id,
            message,
            position=position,
            variant_index=variant_index,
        )
        if incoming_hash != existing_hash:
            return True
    return False


def _retain_stale_revision_observations(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
) -> None:
    """Merge monotonic facts from a stale revision without replacing content."""
    _retain_stale_session_observations(
        conn,
        payload.session_id,
        payload.parsed_session,
        fallback_timestamp=payload.fallback_timestamp,
    )


def _incoming_write_regresses_attachment_coverage(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
    session_to_write: ParsedSession,
) -> bool:
    """Return whether writing ``session_to_write`` would lose acquired attachments.

    polylogue-ixry: two raw acquisitions of the same logical session can
    carry byte-identical message content -- and therefore an identical
    content-derived freshness timestamp -- while differing only in which
    attachments were actually fetched. Drive re-acquisition backfills
    attachment bytes into a *new* raw_id (`#3073`) without touching any
    message timestamp, and never establishes revision lineage
    (`predecessor_raw_id`/`logical_source_key`) the way the governed "live"
    batch path does for tailed origins, so the two raw rows never form a
    `raw_revision_heads` cohort either. Without this check, the freshness
    comparison above ties and falls through to "whichever raw this batch
    happens to (re)parse last wins" -- observed on the live archive to
    silently revert a completed attachment fetch back to `unfetched` with no
    signal (measured: 157/157 duplicate aistudio-drive source_paths landed
    on the pre-fetch revision). This only ever blocks a regression: an
    incoming write that ties or improves attachment coverage is unaffected.
    """
    incoming_acquired = sum(
        1
        for attachment in session_to_write.attachments
        if attachment.inline_bytes is not None or attachment.precomputed_blob is not None
    )
    existing_acquired_row = conn.execute(
        """
        SELECT COUNT(*) FROM attachment_refs r
        JOIN attachments a ON a.attachment_id = r.attachment_id
        WHERE r.session_id = ? AND a.acquisition_status = 'acquired'
        """,
        (payload.session_id,),
    ).fetchone()
    existing_acquired = int(existing_acquired_row[0]) if existing_acquired_row is not None else 0
    return incoming_acquired < existing_acquired


def _incoming_write_carries_distinct_messages(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
    session_to_write: ParsedSession,
) -> bool:
    """Return whether ``session_to_write`` holds message content the archive lacks.

    polylogue-5uoed: the attachment tie-break above decides the tie on
    attachment coverage ALONE. That is sound for the case it was measured on
    -- a Drive re-acquisition whose message content is byte-identical and
    whose only difference is which attachment bytes were fetched -- but it
    generalizes wrongly: a genuinely different revision whose content-derived
    freshness happens to tie can be skipped for carrying fewer attachments,
    and its distinct messages then never land. Skipping is counted, not
    silent, but the content is still lost on a fresh import.

    A message the session owns is compared by its complete semantic
    revision, ``message_content_identity``: the digest of every declared
    semantic field (``pipeline.ids``), stored as ``messages.content_identity``,
    so a revision that changes only model, stop reason, material origin or
    provider message id -- which the coarse lineage signature omits -- is new
    content. An inherited prefix message is compared by that lineage
    signature, as the append delta compares it (``_append_delta_payload``): a
    child replays its parent's prefix under its own provider ids, so no
    stored identity can match it. The composed transcript (inherited prefix +
    own tail, ``_composed_db_signatures``) is the same view the incoming full
    parse represents, so a revision that merely re-states what is stored is a
    multiset subset and stays skippable, while one that adds or revises a
    message is not. The measured aistudio-drive shape (identical messages,
    differing attachment coverage) is a subset by construction and remains
    blocked.

    Multiplicity matters: two byte-identical messages in the incoming parse
    against one stored occurrence is new content, so the comparison counts
    occurrences rather than testing set membership.
    """
    from polylogue.core.compute_cancel import check_compute_cancelled
    from polylogue.storage.sqlite.archive_tiers.write import _iter_composed_rows
    from polylogue.storage.sqlite.connection_profile import scratch_connection_context

    # This is a temporary comparison within the existing preparation owner;
    # only its boolean result crosses admission, alongside the predecessor.
    with scratch_connection_context(prefix="ingest-comparison-", filename="counts.db") as counts:
        with closing(counts.execute("PRAGMA temp_store=FILE")):
            pass
        with closing(
            counts.execute(
                "CREATE TEMP TABLE counts(kind TEXT NOT NULL, key TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY(kind,key)) WITHOUT ROWID"
            )
        ):
            pass
        with closing(_iter_composed_rows(conn, payload.session_id)) as composed:
            for message_id, signature, owner in composed:
                check_compute_cancelled()
                if owner == payload.session_id:
                    with closing(
                        conn.execute("SELECT content_identity FROM messages WHERE message_id=?", (message_id,))
                    ) as cursor:
                        row = cursor.fetchone()
                    if row is None or row[0] is None:
                        continue
                    kind, key = "own", str(row[0])
                else:
                    kind, key = "inherited", signature
                with closing(
                    counts.execute(
                        "INSERT INTO counts VALUES(?,?,1) ON CONFLICT(kind,key) DO UPDATE SET n=n+1", (kind, key)
                    )
                ):
                    pass
        for message in session_to_write.messages:
            check_compute_cancelled()
            identity = message_content_identity(message)
            with closing(
                counts.execute("UPDATE counts SET n=n-1 WHERE kind='own' AND key=? AND n>0", (identity,))
            ) as cursor:
                matched = cursor.rowcount
            if matched:
                continue
            signature = _parsed_message_signature(message)
            with closing(
                counts.execute("UPDATE counts SET n=n-1 WHERE kind='inherited' AND key=? AND n>0", (signature,))
            ) as cursor:
                matched = cursor.rowcount
            if matched:
                continue
            return True
        return False


def _write_session(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
    *,
    force_write: bool = False,
    signature_cache: LineageSignatureCache | dict[str, list[tuple[str, str]]] | None = None,
    stage_timings_s: dict[str, float] | None = None,
    blob_publisher: ArchiveBlobPublisher | None = None,
    source_conn: sqlite3.Connection | None = None,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    attachment_owner_resolutions: list[dict[str, str]] | None = None,
    manage_transaction: bool = True,
    prepared_writes: list[PreparedSessionWrite] | None = None,
) -> tuple[bool, dict[str, int]]:
    """Write one parsed session payload into the current archive index.

    ``manage_transaction=False`` is required whenever the caller already owns a
    transaction (the bulk ingest batch): the writer's own ``with conn:`` would
    otherwise COMMIT the caller's ``BEGIN IMMEDIATE`` at the first session, so
    the batch boundary -- and the FTS-trigger suspension that lives inside it --
    would not exist at runtime (polylogue-qoa75).

    Returns (content_changed, counts).
    """
    counts: dict[str, int] = {
        "sessions": 0,
        "messages": 0,
        "attachments": 0,
        "session_events": 0,
        "skipped_sessions": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
        "skipped_session_events": 0,
        "raw_links": 0,
        "sidecar_blob_bytes_new": 0,
        "sidecar_blob_bytes_dedup": 0,
        "sidecar_blobs_written": 0,
        "sidecar_blobs_refused_excised": 0,
    }

    existing_row = None
    if not fresh_build:
        existing_row = conn.execute(
            "SELECT content_hash, raw_id, updated_at_ms FROM sessions WHERE session_id = ?",
            (payload.session_id,),
        ).fetchone()
    elif conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (payload.session_id,)).fetchone() is not None:
        raise AssertionError(f"fresh_build requires an absent session_id: {payload.session_id}")
    if (
        fresh_build
        and (fresh_build_batch is None or not fresh_build_batch)
        and conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone() is not None
    ):
        raise AssertionError("fresh_build requires an empty archive generation")
    if fresh_build_batch is not None:
        fresh_build_batch.add(payload.session_id)
    existing_hash = existing_row["content_hash"] if existing_row is not None else None
    existing_hash_hex = existing_hash.hex() if isinstance(existing_hash, bytes) else str(existing_hash or "")
    content_unchanged = existing_row is not None and existing_hash_hex == payload.content_hash
    existing_raw_id = str(existing_row["raw_id"] or "") if existing_row is not None else ""
    if payload.prepared_distinct_messages is None:
        raise PreparedSessionWriteRefusedError("session decisions require canonical off-writer preparation")
    current_predecessor = tuple(existing_row) if existing_row is not None else None
    if current_predecessor != payload.prepared_predecessor:
        raise PreparedSessionWriteRefusedError("ingest predecessor changed after off-writer preparation")
    session_to_write = payload.parsed_session
    merge_append = False
    append_force_replace = False
    freshness_force_replace = False
    browser_precedence: BrowserCapturePrecedence = "default"

    if revision_authority_refuses_write(
        conn,
        source_conn,
        session_id=payload.session_id,
        raw_id=payload.raw_id or "",
        provider_session_id=payload.parsed_session.provider_session_id,
    ):
        _retain_stale_revision_observations(conn, payload)
        counts["skipped_sessions"] = 1
        counts["skipped_messages"] = payload.message_count
        counts["skipped_attachments"] = payload.attachment_count
        counts["skipped_session_events"] = len(payload.parsed_session.session_events)
        return False, counts

    if (
        not force_write
        and not payload.append_only
        and existing_raw_id
        and payload.raw_id
        and existing_raw_id != payload.raw_id
    ):
        existing_is_dom_fallback = session_has_parser_ingest_flag(conn, payload.session_id, DOM_FALLBACK_INGEST_FLAG)
        incoming_is_dom_fallback = _incoming_has_ingest_flag(payload, DOM_FALLBACK_INGEST_FLAG)
        existing_has_native_browser_payload = session_has_parser_ingest_flag(
            conn,
            payload.session_id,
            NATIVE_BROWSER_CAPTURE_FLAGS,
        )
        incoming_has_native_browser_payload = _incoming_has_ingest_flag(
            payload,
            NATIVE_BROWSER_CAPTURE_FLAGS,
        )
        current_stored_message_count = stored_message_count(conn, payload.session_id)
        lower_precedence_fallback = incoming_is_dom_fallback and not existing_is_dom_fallback
        browser_precedence = browser_capture_precedence(
            existing_is_dom_fallback=existing_is_dom_fallback,
            incoming_is_dom_fallback=incoming_is_dom_fallback,
            existing_has_native_payload=existing_has_native_browser_payload,
            incoming_has_native_payload=incoming_has_native_browser_payload,
            stored_message_count=current_stored_message_count,
            incoming_message_count=payload.message_count,
        )
        if browser_precedence == "skip":
            if lower_precedence_fallback:
                record_capture_gap_event(
                    conn,
                    session_id=payload.session_id,
                    existing_raw_id=existing_raw_id,
                    incoming_raw_id=payload.raw_id,
                    stored_message_count=current_stored_message_count,
                    incoming_message_count=payload.message_count,
                )
                counts["session_events"] = 1
            # Twin of the ArchiveStore skip path: a capture that loses the
            # content merge can still truthfully declare when it was NOT
            # observing the page — that outage telemetry survives the skip.
            outage_events = record_source_outage_events(
                conn,
                session_id=payload.session_id,
                events=payload.parsed_session.session_events,
            )
            _retain_stale_revision_observations(conn, payload)
            counts["session_events"] = counts.get("session_events", 0) + outage_events
            counts["skipped_sessions"] = 1
            counts["skipped_messages"] = payload.message_count
            counts["skipped_attachments"] = payload.attachment_count
            counts["skipped_session_events"] = len(payload.parsed_session.session_events) - outage_events
            return False, counts

    _incoming_created_at_ms, incoming_freshness_ms = session_evidence_timestamps(session_to_write)
    if incoming_freshness_ms is None:
        incoming_freshness_ms = _incoming_created_at_ms
    if (
        not force_write
        and browser_precedence != "replace"
        and not payload.append_only
        and existing_row is not None
        and incoming_freshness_ms is not None
    ):
        existing_updated_at_ms = existing_row["updated_at_ms"]
        existing_updated_at_int = int(existing_updated_at_ms) if existing_updated_at_ms is not None else None
        if should_skip_stale_replace(
            incoming_freshness_ms=incoming_freshness_ms,
            existing_updated_at_ms=existing_updated_at_int,
        ):
            _retain_stale_revision_observations(conn, payload)
            counts["skipped_sessions"] = 1
            counts["skipped_messages"] = payload.message_count
            counts["skipped_attachments"] = payload.attachment_count
            counts["skipped_session_events"] = len(payload.parsed_session.session_events)
            return False, counts
        freshness_force_replace = True
        if (
            existing_updated_at_int is not None
            and incoming_freshness_ms == existing_updated_at_int
            and existing_raw_id
            and payload.raw_id
            and existing_raw_id != payload.raw_id
            and _incoming_write_regresses_attachment_coverage(conn, payload, session_to_write)
            # Attachment coverage alone cannot tell a re-acquisition of the
            # same transcript from a genuinely different revision that happens
            # to tie on content-derived freshness (polylogue-5uoed). Skipping
            # the latter loses its distinct messages on a fresh import.
            and payload.prepared_distinct_messages is False
        ):
            counts["skipped_sessions"] = 1
            counts["skipped_messages"] = payload.message_count
            counts["skipped_attachments"] = payload.attachment_count
            counts["skipped_session_events"] = len(payload.parsed_session.session_events)
            return False, counts

    if payload.append_only and existing_row is not None:
        existing_updated_at_ms = existing_row["updated_at_ms"]
        existing_updated_at_int = int(existing_updated_at_ms) if existing_updated_at_ms is not None else None
        if (
            incoming_freshness_ms is not None
            and existing_updated_at_int is not None
            and incoming_freshness_ms < existing_updated_at_int
        ):
            # Append-only captures can arrive out of order after a restart or
            # replay. An older full revision may carry the same native
            # messages plus stale session metadata and attachment/event
            # projections. Once a newer revision is stored, accepting that
            # older envelope would create a hybrid session even though its
            # message delta is empty. Preserve the newer authority and leave
            # the raw row available for audit/replay.
            counts["skipped_sessions"] = 1
            counts["skipped_messages"] = payload.message_count
            counts["skipped_attachments"] = payload.attachment_count
            counts["skipped_session_events"] = len(payload.parsed_session.session_events)
            return False, counts
        prepared_append = payload.prepared_write
        if prepared_append is None:
            if not payload.prepared_append_noop:
                raise PreparedSessionWriteRefusedError("append publication lacks canonical preparation")
            if tuple(existing_row) != payload.prepared_predecessor:
                raise PreparedSessionWriteRefusedError("append predecessor changed after preparation")
            if payload.parsed_session.ingest_flags:
                upsert_parser_ingest_flag_tags(conn, payload.session_id, payload.parsed_session.ingest_flags)
            counts["raw_links"] = int(_refresh_session_raw_link(conn, payload.session_id, payload.raw_id))
            counts["skipped_sessions"] = 1
            counts["skipped_messages"] = payload.prepared_append_skipped_messages
            counts["skipped_attachments"] = payload.attachment_count
            counts["skipped_session_events"] = len(payload.parsed_session.session_events)
            if _needs_session_fts_repair(conn, payload.session_id):
                counts[_FTS_REPAIR_COUNT_KEY] = 1
            return False, counts
        if prepared_append.merge_append:
            counts["skipped_messages"] = payload.prepared_append_skipped_messages
            session_to_write = prepared_append.context.effective_session
            merge_append = True
        else:
            append_force_replace = True

    if not force_write and content_unchanged:
        if browser_precedence == "replace":
            replace_parser_ingest_flag_tags(conn, payload.session_id, payload.parsed_session.ingest_flags)
        elif payload.parsed_session.ingest_flags:
            upsert_parser_ingest_flag_tags(conn, payload.session_id, payload.parsed_session.ingest_flags)
        counts["raw_links"] = int(_refresh_session_raw_link(conn, payload.session_id, payload.raw_id))
        counts["skipped_sessions"] = 1
        counts["skipped_messages"] = payload.message_count
        counts["skipped_attachments"] = payload.attachment_count
        counts["skipped_session_events"] = len(payload.parsed_session.session_events)
        if _needs_session_fts_repair(conn, payload.session_id):
            counts[_FTS_REPAIR_COUNT_KEY] = 1
        _bind_session_enrichment(conn, source_conn, payload)
        # The unchanged content still leaves the same attachments unowned:
        # report the gaps the session's last write recorded, as a write would.
        if attachment_owner_resolutions is not None:
            _append_owner_resolutions(
                attachment_owner_resolutions, payload, recorded_attachment_owner_gaps(conn, payload.session_id)
            )
        return False, counts

    if (
        existing_row is None
        and not payload.parsed_session.messages
        and not force_write
        and not (
            payload.parsed_session.source_name is Provider.OTEL_GENAI
            and any(event.event_type == "otel_span_evidence" for event in payload.parsed_session.session_events)
        )
    ):
        counts["skipped_sessions"] = 1
        return False, counts

    preacquired_attachment_blobs: Mapping[object, tuple[bytes | None, int, str]] | None = None
    sidecar_blob_locators: Mapping[str, Mapping[str, str]] = {}
    if payload.prepared_artifact is not None:
        from polylogue.sources.prepared_jsonl import PreparedSidecarLocators
        from polylogue.storage.blob_publication import ConnectionBlobPublicationRead

        if source_conn is None:
            raise PreparedSessionWriteRefusedError("artifact publication requires its actual Source writer")
        publication_read = ConnectionBlobPublicationRead(source_conn)
        preacquired_attachment_blobs = payload.prepared_artifact.attachment_blobs(
            source_read=publication_read, session_id=payload.session_id
        )
        if payload.prepared_session_ordinal is None:
            raise PreparedSessionWriteRefusedError("sidecar publication lacks its captured artifact coordinate")
        locators = PreparedSidecarLocators(
            payload.prepared_artifact, payload.prepared_session_ordinal, publication_read
        )
        sidecar_blob_locators = locators
        counts.update(locators.publication_counts())
    elif any(
        item.inline_bytes is not None or item.precomputed_blob is not None for item in session_to_write.attachments
    ):
        raise PreparedSessionWriteRefusedError("attachment publication requires its sealed canonical artifact")

    prepared_write = payload.prepared_write
    if prepared_write is None:
        raise PreparedSessionWriteRefusedError("session has no canonical pre-admission prepared write")
    if source_conn is None:
        raise PreparedSessionWriteRefusedError("prepared session publication requires its actual Source reader")
    if prepared_writes is not None:
        # Register before the writer call so entry cleanup owns this carrier
        # even if publication raises before returning an outcome.
        prepared_writes.append(prepared_write)
    writer_outcomes: list[ArchiveWriteOutcome] = []
    write_parsed_session_to_archive(
        conn,
        session_to_write,
        # ``content_hash`` is the digest STORED on the sessions row, so it
        # stays the full session's even on an append: the next ingest of this
        # session compares its own full-session digest against that row to
        # decide the content is unchanged. ``pending_input_content_hash``
        # separately names what this call publishes -- the delta -- which is
        # the digest a prepared identity carrier must match (polylogue-3hfl7).
        content_hash=payload.content_hash,
        pending_input_content_hash=prepared_write.input_content_hash.hex(),
        prepared_write=prepared_write,
        raw_id=payload.raw_id,
        fallback_timestamp=payload.fallback_timestamp,
        source_read=ConnectionSessionSourceRead(source_conn),
        child_source_path=raw_source_path(ConnectionSessionSourceRead(source_conn), payload.raw_id),
        merge_append=merge_append,
        force_replace=(
            force_write or browser_precedence == "replace" or append_force_replace or freshness_force_replace
        ),
        signature_cache=signature_cache,
        stage_timings_s=stage_timings_s,
        preacquired_attachment_blobs=preacquired_attachment_blobs,
        sidecar_blob_locators=sidecar_blob_locators,
        mutation_scope=current_index_mutation_scope(),
        # Guard-gated bulk FTS for any prefix-tail re-extraction this write
        # cascades into (polylogue-crd8). Byte-identical to per-row trigger
        # mode (tests/unit/storage/test_bulk_fts_prefix_reextract.py) but
        # avoids the per-deleted-row action_pairs/FTS rebuild storm: a live
        # whale-session rewrite held the daemon writer >1h at 260GB of reads
        # with zero commits (2026-07-22) under per-row mode.
        bulk_fts=True,
        fresh_build=fresh_build,
        # The writer runs the same empty-generation guard as ``_write_session``
        # above and needs the same batch memory to satisfy it. Passing
        # ``fresh_build`` without the set left the writer with
        # ``fresh_build_batch=None``, so the second session of a fresh-build
        # batch found the first one's row and aborted the batch with
        # "fresh_build requires an empty archive generation".
        fresh_build_batch=fresh_build_batch,
        write_outcome=writer_outcomes,
        manage_transaction=manage_transaction,
    )
    if writer_outcomes and (writer_outcomes[0].stale_skipped or writer_outcomes[0].suppression_skipped):
        if prepared_writes is not None and prepared_write is not None:
            prepared_writes.remove(prepared_write)
            if prepared_write is not payload.prepared_write:
                prepared_write.close()
        if writer_outcomes[0].stale_skipped:
            _retain_stale_revision_observations(conn, payload)
        counts["skipped_sessions"] = 1
        counts["skipped_messages"] = payload.message_count
        counts["skipped_attachments"] = payload.attachment_count
        counts["skipped_session_events"] = len(payload.parsed_session.session_events)
        return False, counts
    if not (writer_outcomes and writer_outcomes[0].suppression_skipped):
        _bind_session_enrichment(conn, source_conn, payload)
    if attachment_owner_resolutions is not None and writer_outcomes:
        _append_owner_resolutions(
            attachment_owner_resolutions, payload, writer_outcomes[0].unresolved_attachment_owners
        )
    counts["sessions"] = 1
    counts["messages"] = len(session_to_write.messages)
    counts["attachments"] = len(session_to_write.attachments)
    counts["session_events"] = len(session_to_write.session_events)

    return True, counts


def _bind_session_enrichment(
    conn: sqlite3.Connection, source_conn: sqlite3.Connection | None, payload: SessionWritePayload
) -> None:
    """Bind this accepted session to its enrichment evidence, if still current.

    The parsed session carries the key of the evidence it was enriched from
    (stamped by the worker or retained enricher). Without a source handle, a
    carried key, or with evidence that moved since, nothing is bound and
    inspection re-derives the session on the retained route.
    """
    from polylogue.sources.revision_backfill import (
        provider_binds_enrichment,
        record_session_enrichment_binding,
        session_enrichment_evidence_key,
    )

    if source_conn is None or not payload.raw_id or not provider_binds_enrichment(payload.parsed_session.source_name):
        return

    row = source_conn.execute("SELECT source_path FROM raw_sessions WHERE raw_id = ?", (payload.raw_id,)).fetchone()
    native = conn.execute("SELECT native_id FROM sessions WHERE session_id = ?", (payload.session_id,)).fetchone()
    main = next((entry for entry in source_conn.execute("PRAGMA database_list") if entry[1] == "main"), None)
    if row is None or row[0] is None or native is None or main is None or not main[2]:
        return
    record_session_enrichment_binding(
        conn,
        session_id=payload.session_id,
        carried_key=payload.parsed_session.enrichment_evidence_key,
        current_key=session_enrichment_evidence_key(
            provider=payload.parsed_session.source_name,
            source_path=str(row[0]),
            native_id=str(native[0]),
            index_conn=conn,
            source_conn=source_conn,
            blob_root=Path(main[2]).parent / "blob",
        ),
    )


def _refresh_session_raw_link(conn: sqlite3.Connection, session_id: str, raw_id: str | None) -> bool:
    """Keep accepted unchanged parses linked to their latest acquired raw row."""
    if not raw_id:
        return False
    cursor = conn.execute(
        """
        UPDATE sessions
        SET raw_id = ?
        WHERE session_id = ?
          AND (raw_id IS NULL OR raw_id != ?)
        """,
        (raw_id, session_id, raw_id),
    )
    return cursor.rowcount > 0


class FtsTriggerRestorationError(RuntimeError):
    """Raised when the bulk-ingest FTS trigger suspension cannot be undone.

    A dropped-trigger window that survives the batch leaves every subsequent
    ordinary write out of ``messages_fts`` with no error anywhere, so this
    failure is escalated rather than suppressed (polylogue-qoa75).
    """


def _session_foreign_key_actions(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    actions: list[tuple[str, str, str]] = []
    table_rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    for table_row in table_rows:
        table_name = str(table_row[0])
        for fk in conn.execute(f"PRAGMA foreign_key_list({_quote_identifier(table_name)})").fetchall():
            if str(fk[2]) == "sessions":
                actions.append((table_name, str(fk[3]), str(fk[6] or "")))
    actions.sort(key=lambda item: item[0] == "sessions")
    return actions


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _ensure_ingest_index_incarnation(conn: sqlite3.Connection) -> None:
    """Commit a physical index identity before any retryable ingest writes."""
    filename = str(conn.execute("PRAGMA database_list").fetchone()[2])
    index_stat = Path(filename).stat()
    conn.execute("BEGIN IMMEDIATE")
    incarnation = conn.execute(
        "SELECT incarnation_id, device, inode FROM ingest_index_incarnation WHERE singleton = 1"
    ).fetchone()
    if incarnation is None:
        conn.execute(
            "INSERT INTO ingest_index_incarnation(singleton, incarnation_id, device, inode) VALUES (1, ?, ?, ?)",
            (str(uuid.uuid4()), index_stat.st_dev, index_stat.st_ino),
        )
    elif (int(incarnation[1]), int(incarnation[2])) != (index_stat.st_dev, index_stat.st_ino):
        conn.execute(
            "UPDATE ingest_index_incarnation SET incarnation_id = ?, device = ?, inode = ? WHERE singleton = 1",
            (str(uuid.uuid4()), index_stat.st_dev, index_stat.st_ino),
        )
    conn.commit()


def _resolve_codex_sidecar_snapshots(
    raw_artifacts: list[RawSessionRecord],
    *,
    archive_root: Path,
) -> None:
    """Require Codex evidence to have been carried by acquisition.

    The process-pool worker is subprocess-safe because all provider evidence
    must already be carried on the acquired record. A missing optional
    snapshot is represented as an empty bundle; it is never reconstructed
    from the recorded rollout path.
    """
    del archive_root
    for record in raw_artifacts:
        if record.payload_provider is Provider.CODEX and record.sidecar_snapshot is None:
            # Optional title artifacts may be absent.  This marker prevents
            # the worker from treating absence as permission for an ambient
            # read, while keeping the normal assembly fallback deterministic.
            record.sidecar_snapshot = {}


def _prepare_ingest_payloads(
    index: sqlite3.Connection,
    source: sqlite3.Connection,
    payloads: Sequence[SessionWritePayload],
) -> None:
    """Prepare the canonical session decisions before writer admission."""
    from polylogue.storage.sqlite.write_lease import current_write_lease

    if current_write_lease() is not None:
        raise RuntimeError("session preparation requires a lease-free caller")
    for payload in payloads:
        pending = payload.parsed_session
        merge_append = False
        existing = index.execute(
            "SELECT content_hash, raw_id, updated_at_ms FROM sessions WHERE session_id=?",
            (payload.session_id,),
        ).fetchone()
        payload.prepared_predecessor = tuple(existing) if existing is not None else None
        payload.prepared_distinct_messages = (
            _incoming_write_carries_distinct_messages(index, payload, pending) if existing is not None else True
        )
        if payload.append_only and existing is not None:
            created, updated = session_evidence_timestamps(pending, fallback_timestamp=payload.fallback_timestamp)
            incoming = updated or created
            newer = incoming is not None and existing[2] is not None and incoming > int(existing[2])
            replaces = newer and _append_payload_changes_existing_message(index, payload)
            if not replaces:
                delta, skipped = _append_delta_payload(index, payload)
                payload.prepared_append_skipped_messages = skipped
                if delta is None:
                    payload.prepared_append_noop = True
                    continue
                pending, merge_append = delta, True
        payload.prepared_write = prepare_session_write(
            index,
            pending,
            merge_append=merge_append,
            fallback_timestamp=payload.fallback_timestamp,
            source_read=ConnectionSessionSourceRead(source),
            raw_id=payload.raw_id,
            prepared_rows=payload.prepared_rows,
        )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


async def process_ingest_batch(
    service: ParsingService,
    backend: SQLiteBackend,
    batch_ids: list[str],
    result: ParseResult,
    progress_callback: ProgressCallback | None,
    *,
    force_write: bool = False,
    ingest_result_chunk_size: int = 0,
    suspend_fts_triggers: bool = False,
    fresh_build: bool = False,
) -> ParseBatchObservation | None:
    """Publish acquired Raw IDs through the caller's original retained owner.

    Acquisition has physically settled before this call. Canonical Raw replay
    owns Source census, Index publication and final acknowledgement; this host
    only projects its actual receipts into the parsing result.
    """
    if service.retained_runner is None:
        raise PermissionError("ingest publication requires its supplied retained Raw owner")
    from polylogue.config import load_polylogue_config

    publication_mode = PublicationMode.from_string(load_polylogue_config().sinex_mode)
    if publication_mode is not PublicationMode.OFF:
        raise PublicationEncodingError(
            "retained ingest requires its canonical accepted-marker and outbox producer before publication"
        )
    started = time.perf_counter()
    # A parse failure is a typed refusal the owner settled: a terminal decode
    # refusal or a refused cohort member. A membership-quarantined revision is
    # governance evidence, not a failed parse, so receipts' ``quarantined``
    # counts are not failures.
    refusals: list[object] = []

    def settle_terminal_refusal(_keys: tuple[str, ...], refusal: RetainedRawDecodeRefusalError) -> None:
        refusals.append(refusal)

    def settle_membership_refusal(refusal: CohortMembershipRefusalError) -> None:
        refusals.append(refusal)

    receipts = await service.retained_runner(
        tuple(batch_ids),
        on_terminal_refusal=settle_terminal_refusal,
        on_membership_refusal=settle_membership_refusal,
    )
    result.parse_failures += len(refusals)
    written: dict[str, None] = {}
    changed: dict[str, None] = {}
    for receipt in receipts:
        written.update(dict.fromkeys(receipt.written_session_ids))
        changed.update(dict.fromkeys(receipt.changed_session_ids))
        for key, count in receipt.written_counts.items():
            if key in result.counts:
                result.counts[key] += count
            if key in result.changed_counts and key != "sessions":
                result.changed_counts[key] += count
        for key, seconds in receipt.stage_timings_s.items():
            result.stage_timings_s[key] = result.stage_timings_s.get(key, 0.0) + seconds
    result.processed_ids.update(written)
    result._changed_session_ids.extend(key for key in changed if key not in result._changed_session_ids)
    result.changed_counts["sessions"] += len(changed)
    if progress_callback and receipts:
        progress_callback(len(batch_ids))
    return {
        "records": len(batch_ids),
        "sessions": len(written),
        "messages": sum(receipt.written_message_count for receipt in receipts),
        "changed_sessions": len(changed),
        "failed_raw_count": len(refusals),
        "converged": all(receipt.adoption_deferred == 0 for receipt in receipts),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


__all__ = ["process_ingest_batch"]

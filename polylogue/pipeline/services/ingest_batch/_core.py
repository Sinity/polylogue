"""Batch ingest orchestration: ProcessPool workers + sync sqlite3 writes.

Architecture:
- CPU-bound work (decode/validate/parse/transform) in ProcessPoolExecutor
- DB writes in main thread via sync sqlite3 (no aiosqlite async overhead)
- as_completed yields results as workers finish; writes drain completed worker
  results without retaining the whole parsed batch in memory

"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import pickle
import sqlite3
import time
import unicodedata
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import AsyncExitStack, closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, cast

from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_FLAGS
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.archive.revision_replay import RevisionReplayPlan
from polylogue.archive.write_gateway import ArchiveWriteGateway, WriteOperation
from polylogue.core.enums import BlockType, IngestOutcome, Origin, Provider
from polylogue.core.memory import release_process_memory
from polylogue.core.metrics import (
    read_current_rss_mb,
    read_peak_rss_children_mb,
    read_peak_rss_self_mb,
)
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamp_authority import session_evidence_timestamps
from polylogue.logging import get_logger
from polylogue.markers.preparation import marker_candidates_for_prepared_write, marker_recipe_fingerprint
from polylogue.pipeline.ids import bound_session_content_hash, session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.ingest_outcomes import (
    parser_defect_disposition,
    transient_error_disposition,
)
from polylogue.pipeline.payload_types import ParseBatchObservation
from polylogue.pipeline.services.ingest_worker import (
    IngestRecordResult,
    SessionWritePayload,
    ingest_record,
)
from polylogue.pipeline.services.process_pool import (
    process_pool_executor,
    select_ingest_worker_count,
    terminate_process_pool,
)
from polylogue.sinex.material_adapter import (
    PublicationBackpressureError,
    PublicationEncodingError,
    encode_parsed_session_publication,
)
from polylogue.sinex.models import PublicationMode, PublicationPayload
from polylogue.sinex.obligations import AsyncSqlConnection, stage_payload_async
from polylogue.sinex.service import PublicationService
from polylogue.sinex.transport import resolve_configured_transport
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInputRefusedError,
    PreparedAcceptedMarkerInput,
    finalize_pending_accepted_marker_input,
    prepare_accepted_marker_input,
)
from polylogue.storage.blob_publication import ArchiveBlobPublisher, consume_blob_publication_receipt
from polylogue.storage.raw.models import RawSessionStateUpdate
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
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    bind_raw_revision,
    classify_raw_revision_cohort_for_live_watch,
    raw_membership_raw_ids,
)
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveWriteOutcome,
    LineageSignatureCache,
    PreparedSessionWrite,
    _composed_db_signatures,
    _message_content_hash,
    _normalized_message_native_id,
    _parsed_message_signature,
    _repair_stale_session_observations,
    prepare_session_write,
    replace_parser_ingest_flag_tags,
    upsert_parser_ingest_flag_tags,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.connection import _load_sqlite_vec
from polylogue.storage.sqlite.connection_profile import (
    DB_TIMEOUT,
    WRITE_CONNECTION_PROFILE,
    open_isolated_write_connection,
    open_readonly_connection,
    write_connection_pragma_statements,
)
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

if TYPE_CHECKING:
    from polylogue.archive.write_effects import WriteEffect, WriteEffectContext
    from polylogue.core.protocols import ProgressCallback
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

from polylogue.pipeline.services.ingest_batch._memory import (
    INGEST_RELEASE_BLOB_MB_THRESHOLD,
    INGEST_RELEASE_MESSAGE_THRESHOLD,
    discard_ingest_result_payload,
    discard_session_data_payload,
    ingest_result_needs_memory_release,
)
from polylogue.pipeline.services.ingest_batch._models import (
    _DEFAULT_INGEST_WORKER_LIMIT,
    _SINEX_STAGED_PAYLOAD_LIMIT_BYTES,
    _BulkConnectionBackendLike,
    _ConnectionBackendLike,
    _IngestBatchSummary,
    _IngestWorkerRequest,
    _ParsingServiceRawStateLike,
    _PreparedIngestUnit,
    _RawIngestOutcome,
    _SessionEntry,
    _SourceSnapshot,
)
from polylogue.pipeline.services.ingest_batch._observations import _build_parse_batch_observation
from polylogue.pipeline.services.ingest_batch._summary import (
    apply_ingest_batch_summary,
    progressed_raw_count,
    successful_raw_ids,
)

logger = get_logger(__name__)


@dataclass
class _WorkerProgress:
    in_flight_raw_ids: list[str] = field(default_factory=list)
    completed_raw_count: int = 0
    total_raw_count: int = 0


class _BlobSized(Protocol):
    blob_size: int


IngestHeartbeat = Callable[[], None]
_INGEST_RESULT_WAIT_HEARTBEAT_S = 15.0
# A heartbeat only proves that the coordinator is alive.  It is deliberately
# not progress: a worker that completes no future must eventually be reported
# as unfinished so the ordinary retry/refusal path can settle its raw IDs.
# The deadline is per-progress window, not a batch wall-clock timeout, so a
# legitimately large source can run indefinitely while results keep arriving.
_INGEST_RESULT_PROGRESS_DEADLINE_S = 300.0
_INGEST_RESULT_CHUNK_SIZE = 100


# Sync DB writer
# ---------------------------------------------------------------------------


def _open_sync_connection(db_path: Path, *, archive_root: Path | None = None) -> sqlite3.Connection:
    """Open a sync sqlite3 connection with the same pragmas as the async backend."""
    bound_root = archive_root if archive_root is not None else db_path.parent
    # Bootstrap can create the archive before a SQLite connection exists, so
    # enforce the same lease boundary before that filesystem/database mutation.
    from polylogue.storage.sqlite.write_lease import require_write_lease

    require_write_lease("ingest archive bootstrap", archive_root=bound_root)
    if db_path.name == "index.db":
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        initialize_active_archive_root(bound_root)
    bound_root.mkdir(parents=True, exist_ok=True)
    # Index publication owns an index.db transaction only. Attaching source.db
    # here would make that transaction lock source.db too, then the attachment
    # publisher's durable reservation would wait behind its own writer. Source
    # precedence reads use the explicit read-only handle below, and receipt
    # consumption opens its own short, archive-bound source-tier transaction.
    conn = open_isolated_write_connection(
        db_path,
        purpose="ingest index publication",
        timeout=DB_TIMEOUT,
        archive_root=bound_root,
    )
    conn.row_factory = sqlite3.Row
    for statement in write_connection_pragma_statements(WRITE_CONNECTION_PROFILE):
        conn.execute(statement)
    if db_path.name == "index.db":
        ensure_runtime_indexes_sync(conn)
    _load_sqlite_vec(conn)
    return conn


def _format_foreign_key_violations(
    rows: Sequence[sqlite3.Row | tuple[object, ...] | Mapping[str, object | None]],
) -> str:
    """Render PRAGMA foreign_key_check rows without sqlite3.Row object reprs."""
    formatted: list[dict[str, object | None]] = []
    columns = ("table", "rowid", "parent", "fkid")
    for row in rows:
        if isinstance(row, Mapping):
            formatted.append(dict(row))
            continue
        if isinstance(row, sqlite3.Row):
            formatted.append({key: row[key] for key in row.keys()})  # noqa: SIM118 - sqlite3.Row iterates values
            continue
        formatted.append({key: row[idx] if idx < len(row) else None for idx, key in enumerate(columns)})
    return repr(formatted)


@dataclass(frozen=True)
class _ScopedForeignKey:
    """One foreign key of one table, with the session column that scopes it."""

    table: str
    fkid: int
    parent: str
    scope_column: str
    child_columns: tuple[str, ...]
    parent_columns: tuple[str, ...]


def _table_column_names(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    # ``table_xinfo`` and not ``table_info``: the latter omits VIRTUAL
    # generated columns, which is exactly what ``sessions.session_id`` is.
    return tuple(str(row[1]) for row in conn.execute(f"PRAGMA table_xinfo({_quote_identifier(table)})"))


def _primary_key_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = [row for row in conn.execute(f"PRAGMA table_xinfo({_quote_identifier(table)})") if int(row[5]) > 0]
    return tuple(str(row[1]) for row in sorted(rows, key=lambda row: int(row[5])))


def _session_scope_column(
    conn: sqlite3.Connection,
    table: str,
    groups: Mapping[int, Sequence[Sequence[object]]],
) -> str | None:
    """Return the column that ties one row of ``table`` to one session, if any."""
    if "session_id" in _table_column_names(conn, table):
        return "session_id"
    # No ``session_id``: a table that still cascades from exactly one
    # single-column reference to ``sessions`` is owned by that session
    # (``session_links.src_session_id``, ``delegation_facts.parent_session_id``).
    owners = {
        str(rows[0][3])
        for rows in groups.values()
        if len(rows) == 1 and str(rows[0][2]) == "sessions" and str(rows[0][6] or "").upper() == "CASCADE"
    }
    return owners.pop() if len(owners) == 1 else None


def _foreign_key_check_plan(
    conn: sqlite3.Connection,
) -> tuple[tuple[_ScopedForeignKey, ...], tuple[str, ...]]:
    """Partition every foreign key in the schema into scoped and unscoped checks.

    Derived from ``PRAGMA foreign_key_list``/``table_xinfo`` on the live
    connection rather than from a written-down table list, because a written
    list drifts: the one this replaced named three tables and the wrong parent
    for two of them, while the schema declares foreign keys on twenty-seven.

    The partition is total over the schema's foreign keys. Every table that
    names an owning session is probed with that session in scope; every table
    that does not is checked whole by SQLite itself. Nothing is skipped.
    """
    scoped: list[_ScopedForeignKey] = []
    unscoped: list[str] = []
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    for table_row in table_rows:
        table = str(table_row[0])
        foreign_keys = conn.execute(f"PRAGMA foreign_key_list({_quote_identifier(table)})").fetchall()
        if not foreign_keys:
            continue
        groups: dict[int, list[Sequence[object]]] = {}
        for fk in foreign_keys:
            groups.setdefault(int(fk[0]), []).append(fk)
        scope_column = _session_scope_column(conn, table, groups)
        if scope_column is None:
            unscoped.append(table)
            continue
        for fkid, rows in sorted(groups.items()):
            ordered = sorted(rows, key=lambda row: int(cast(int, row[1])))
            parent = str(ordered[0][2])
            child_columns = tuple(str(row[3]) for row in ordered)
            parent_columns = tuple(None if row[4] is None else str(row[4]) for row in ordered)
            if any(column is None for column in parent_columns):
                # ``REFERENCES parent`` without a column list means the
                # parent's primary key, in declaration order.
                resolved = _primary_key_columns(conn, parent)
                if len(resolved) != len(child_columns):
                    raise sqlite3.IntegrityError(
                        f"cannot resolve implicit parent key for {table}.fk{fkid} -> {parent}: "
                        f"{len(child_columns)} child columns against primary key {resolved}"
                    )
                parent_columns = resolved
            scoped.append(
                _ScopedForeignKey(
                    table=table,
                    fkid=fkid,
                    parent=parent,
                    scope_column=scope_column,
                    child_columns=child_columns,
                    parent_columns=cast(tuple[str, ...], parent_columns),
                )
            )
    return tuple(scoped), tuple(unscoped)


def _scoped_foreign_key_sql(check: _ScopedForeignKey, placeholders: str) -> str:
    child = _quote_identifier(check.table)
    parent = _quote_identifier(check.parent)
    scope = _quote_identifier(check.scope_column)
    selected = ", ".join(f"c.{_quote_identifier(column)}" for column in check.child_columns)
    # SQLite's default MATCH SIMPLE: a row with ANY NULL child column
    # satisfies the constraint, so those rows are not violations.
    not_null = " AND ".join(f"c.{_quote_identifier(column)} IS NOT NULL" for column in check.child_columns)
    joined = " AND ".join(
        f"p.{_quote_identifier(parent_column)} = c.{_quote_identifier(child_column)}"
        for child_column, parent_column in zip(check.child_columns, check.parent_columns, strict=True)
    )
    return f"""
        SELECT c.rowid AS violation_rowid, c.{scope} AS scope_value, {selected}
        FROM {child} AS c
        WHERE c.{scope} IN ({placeholders})
          AND {not_null}
          AND NOT EXISTS (SELECT 1 FROM {parent} AS p WHERE {joined})
        """


def _foreign_key_violations_for_sessions(
    conn: sqlite3.Connection,
    session_ids: Iterable[str],
    *,
    limit: int = 10,
) -> list[dict[str, object | None]]:
    """Return FK violations the bulk path's ``foreign_keys=OFF`` window admitted.

    ``PRAGMA foreign_key_check`` is the correct-by-construction check, but its
    only scoping granularity is a whole table -- SQLite offers no row-scoped
    form -- and this runs before every bulk commit, on an archive whose
    ``blocks`` table is not the batch. So the probes are generated from the
    live schema instead of written down, which is the property the previous
    hand-rolled list lacked: it probed each column of a compound key
    independently and therefore could not see two owners disagreeing.
    """
    scoped_session_ids = tuple(sorted({session_id for session_id in session_ids if session_id}))
    if not scoped_session_ids:
        return []
    placeholders = ",".join("?" for _ in scoped_session_ids)
    scoped_checks, unscoped_tables = _foreign_key_check_plan(conn)
    violations: list[dict[str, object | None]] = []
    for check in scoped_checks:
        sql = _scoped_foreign_key_sql(check, placeholders)
        for row in conn.execute(sql, scoped_session_ids).fetchall():
            violations.append(
                {
                    "table": check.table,
                    "rowid": row["violation_rowid"],
                    "parent": check.parent,
                    "fkid": check.fkid,
                    "session_id": row["scope_value"],
                    "child_key": {column: row[column] for column in check.child_columns},
                }
            )
            if len(violations) >= limit:
                return violations
    for table in unscoped_tables:
        # Reachable only through a scoped parent, so there is no session to
        # scope by. SQLite checks these whole rather than leaving them out.
        for row in conn.execute(f"PRAGMA foreign_key_check({_quote_identifier(table)})").fetchall():
            violations.append(
                {
                    "table": str(row[0]),
                    "rowid": row[1],
                    "parent": str(row[2]),
                    "fkid": row[3],
                    "session_id": None,
                    "child_key": None,
                }
            )
            if len(violations) >= limit:
                return violations
    return violations


def _incoming_has_ingest_flag(payload: SessionWritePayload, flag: str | Sequence[str]) -> bool:
    flags = (flag,) if isinstance(flag, str) else flag
    return any(candidate in payload.parsed_session.ingest_flags for candidate in flags)


def _session_parent_id(payload: SessionWritePayload) -> str | None:
    parent_native_id = payload.parsed_session.parent_session_provider_id
    if not parent_native_id:
        return None
    return str(make_session_id(payload.parsed_session.source_name, parent_native_id))


def _topo_sort_session_entries(
    entries: list[_SessionEntry],
) -> list[_SessionEntry]:
    """Sort session entries so parents in the same batch precede children."""
    ids_in_batch = {entry[1].session_id for entry in entries}
    no_parent: list[_SessionEntry] = []
    has_parent: list[_SessionEntry] = []

    for entry in entries:
        parent_id = _session_parent_id(entry[1])
        if parent_id and parent_id in ids_in_batch and parent_id != entry[1].session_id:
            has_parent.append(entry)
        else:
            no_parent.append(entry)

    if not has_parent:
        return entries

    ordered = list(no_parent)
    inserted_ids = {entry[1].session_id for entry in ordered}
    remaining = list(has_parent)
    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        next_remaining: list[_SessionEntry] = []
        for entry in remaining:
            parent_id = _session_parent_id(entry[1])
            if parent_id in inserted_ids:
                ordered.append(entry)
                inserted_ids.add(entry[1].session_id)
            else:
                next_remaining.append(entry)
        remaining = next_remaining
    ordered.extend(remaining)
    return ordered


_WRITE_SELECT_CHUNK_SIZE = 900
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


def _existing_native_message_ids(conn: sqlite3.Connection, session_id: str) -> set[str]:
    return {
        str(row[0]).strip()
        for row in conn.execute(
            "SELECT native_id FROM messages WHERE session_id = ? AND native_id IS NOT NULL",
            (session_id,),
        ).fetchall()
    }


def _append_delta_payload(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
) -> tuple[ParsedSession | None, int]:
    existing_native_ids = _existing_native_message_ids(conn, payload.session_id)
    existing_logical = _composed_db_signatures(conn, payload.session_id)
    delta_messages: list[ParsedMessage] = []
    logical_prefix = 0
    for message in payload.parsed_session.messages:
        native_id = _normalized_message_native_id(message)
        signature = _parsed_message_signature(message)
        is_replayed_prefix = logical_prefix < len(existing_logical) and existing_logical[logical_prefix][1] == signature
        if is_replayed_prefix:
            logical_prefix += 1
            continue
        if native_id is not None and native_id in existing_native_ids:
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


def _repair_stale_revision_observations(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
) -> None:
    """Merge monotonic facts from a stale revision without replacing content."""
    _repair_stale_session_observations(
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

    Comparing composed message signatures is the narrowest evidence that
    separates the two cases. ``_composed_db_signatures`` returns the stored
    session's composed transcript (inherited prefix + own tail), the same
    view the incoming full parse represents, and signatures carry role plus
    every block's type/text/tool name/tool input -- so a revision that merely
    re-states what is already stored is a multiset subset and stays skippable,
    while one that adds or revises any message is not. The measured
    aistudio-drive shape (identical messages, differing attachment coverage)
    is a subset by construction and remains blocked.

    Multiplicity matters: two byte-identical messages in the incoming parse
    against one stored occurrence is new content, so the comparison counts
    occurrences rather than testing set membership.
    """
    existing_signatures = Counter(
        signature for _message_id, signature in _composed_db_signatures(conn, payload.session_id)
    )
    incoming_signatures = Counter(_parsed_message_signature(message) for message in session_to_write.messages)
    return any(count > existing_signatures[signature] for signature, count in incoming_signatures.items())


#: Session-event families whose payload names a tool-result sidecar this
#: function must give a durable content-addressed home. Both carry the same
#: ``{acquisition_status, tool_use_id, content_replaced}`` payload shape.
_SIDECAR_EVENT_TYPES = ("claude_tool_result_sidecar", "gemini_cli_tool_output_sidecar")


def _preacquire_sidecar_blobs(
    session_to_write: ParsedSession,
    blob_publisher: ArchiveBlobPublisher,
    publication_receipts: list[tuple[str, bytes]],
) -> tuple[ParsedSession, dict[str, int]]:
    """Content-address + dedup acquired tool-result sidecar text (polylogue-rujy AC4).

    ``apply_tool_result_sidecars`` / ``apply_gemini_tool_output_sidecars``
    (parser-side, side-effect-free) already
    replaced a matched, truncated ``tool_result`` block's inline preview with
    the sidecar file's full text and recorded a bounded
    sidecar session event per file. That leaves the
    acquired bytes living only as an ordinary SQLite TEXT column: two
    sessions with byte-identical tool output (a repeated build log, the same
    lint run) each store their own full copy, and there is no record of how
    many genuinely new bytes an ingest run added to the archive versus how
    many were already present under the same hash.

    This mirrors the existing attachment-blob path (``_acquire_attachment_blob``
    / the ``preacquired_attachment_blobs`` loop above): it publishes the exact
    bytes the matched block now carries through the archive's content-addressed
    blob store, records the resulting ``blob_hash`` back onto the event so a
    reader can locate the durable copy, and returns per-run byte counts
    (new vs. deduplicated) for the ingest batch's own accounting. It never
    touches ``blocks.text`` -- that already carries the full text for FTS --
    this only adds a deduplicated, content-addressed second home for it.

    A no-op (returns ``session_to_write`` unchanged) unless the session
    actually carries a matched+replaced sidecar event, so a session from an
    origin without sidecars never pays this cost.
    """
    matched_tool_use_ids = {
        tool_use_id
        for event in session_to_write.session_events
        if event.event_type in _SIDECAR_EVENT_TYPES
        and event.payload.get("acquisition_status") == "matched"
        and event.payload.get("content_replaced")
        and isinstance(tool_use_id := event.payload.get("tool_use_id"), str)
    }
    if not matched_tool_use_ids:
        return session_to_write, {}

    text_by_tool_use_id: dict[str, str] = {
        block.tool_id: block.text
        for message in session_to_write.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT
        and block.tool_id is not None
        and block.tool_id in matched_tool_use_ids
        and block.text is not None
    }
    if not text_by_tool_use_id:
        return session_to_write, {}

    blob_hash_by_tool_use_id: dict[str, str] = {}
    bytes_new = 0
    bytes_dedup = 0
    for tool_use_id, text in text_by_tool_use_id.items():
        encoded = unicodedata.normalize("NFC", text).encode("utf-8")
        precomputed_hash = hashlib.sha256(encoded).hexdigest()
        already_present = blob_publisher.exists(precomputed_hash)
        hash_hex, size = blob_publisher.write_from_bytes(encoded)
        blob_hash_by_tool_use_id[tool_use_id] = hash_hex
        if already_present:
            bytes_dedup += size
        else:
            bytes_new += size
        receipt_id = blob_publisher.receipt_id(hash_hex)
        if receipt_id is not None:
            publication_receipts.append((receipt_id, bytes.fromhex(hash_hex)))

    updated_events = [
        event.model_copy(update={"payload": {**event.payload, "blob_hash": blob_hash_by_tool_use_id[tool_use_id]}})
        if event.event_type in _SIDECAR_EVENT_TYPES
        and isinstance(tool_use_id := event.payload.get("tool_use_id"), str)
        and tool_use_id in blob_hash_by_tool_use_id
        else event
        for event in session_to_write.session_events
    ]
    counts = {
        "sidecar_blob_bytes_new": bytes_new,
        "sidecar_blob_bytes_dedup": bytes_dedup,
        "sidecar_blobs_written": len(blob_hash_by_tool_use_id),
    }
    return session_to_write.model_copy(update={"session_events": updated_events}), counts


# polylogue-ojjet: the Drive revision-cohort classifier
# (``classify_historical_full_revision_streams``) deliberately re-derives
# every cohort member's true size and content hash from its bytes rather
# than trusting a caller-supplied value, and then streams pairwise prefix
# comparisons between the survivors. Both phases reach the blob store
# through ``store._blob_publisher.open``, so one classification re-opens the
# same handful of blobs once per hash and once per candidate pair, and a
# whole ingest pass repeats that for every raw that reaches
# ``_write_session``. Measured on the production ``_write_session`` route
# with one Gemini logical identity: 6 raws cost 87 opens of 6 distinct
# blobs, 12 raws cost 646 opens of 12, and 24 raws cost 4,896 opens of 24.
#
# Blobs are content-addressed and immutable, so the bytes behind a hash can
# never change underneath this cache. Serving those repeats from a
# pass-scoped, byte-bounded cache leaves every decision -- and therefore
# every lineage binding -- bit-identical while making the cohort's disk
# loads one per distinct blob instead of growing with the cohort. Nothing
# here caps a cohort or moves the classifier behind ``_write_session``'s
# skip check: which raws get lineage-recorded is unchanged.
_DRIVE_COHORT_BLOB_CACHE_MAX_BYTES = 64 * 1024 * 1024


class DriveRevisionCohortCache:
    """Pass-scoped, byte-bounded cache of Drive revision-cohort blob bytes.

    A blob larger than the remaining budget is never cached and is read
    straight from disk, so a single large cohort member cannot push the
    resident set past the budget.
    """

    def __init__(self, max_bytes: int = _DRIVE_COHORT_BLOB_CACHE_MAX_BYTES) -> None:
        self._blobs: dict[str, bytes] = {}
        self._remaining = max_bytes
        self.disk_loads = 0
        self.served_from_cache = 0

    def read(self, publisher: ArchiveBlobPublisher, hash_hex: str) -> bytes | None:
        cached = self._blobs.get(hash_hex)
        if cached is not None:
            self.served_from_cache += 1
            return cached
        path = publisher.blob_path(hash_hex)
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size > self._remaining:
            return None
        data = path.read_bytes()
        self._blobs[hash_hex] = data
        self._remaining -= len(data)
        self.disk_loads += 1
        return data


class _CohortCachingBlobPublisher(ArchiveBlobPublisher):
    """An :class:`ArchiveBlobPublisher` whose reads go through a cohort cache.

    Only ``open`` is intercepted. Every piece of publication bookkeeping is
    the inner publisher's own mutable state, shared by reference, so this
    wrapper can never fork a pending-blob queue or a receipt table.
    """

    def __init__(self, inner: ArchiveBlobPublisher, cache: DriveRevisionCohortCache) -> None:
        super().__init__(inner.source_db_path, inner.root, store=inner._store)
        self._inner = inner
        self._cohort_cache = cache
        self.publisher_id = inner.publisher_id
        self._pending = inner._pending
        self._latest_receipt_by_hash = inner._latest_receipt_by_hash
        self._pending_by_hash = inner._pending_by_hash

    def open(self, hash_hex: str) -> BinaryIO:
        data = self._cohort_cache.read(self._inner, hash_hex)
        if data is None:
            return self._inner.open(hash_hex)
        return io.BytesIO(data)


class _DriveRevisionGovernanceAdapter:
    """Minimal ``RawRevisionGovernanceHost`` for Drive lineage bookkeeping.

    ``bind_raw_revision``/``classify_raw_revision_cohort`` and their
    transitive call graph (``raw_membership_raw_ids``,
    ``_raw_revision_candidates``, ``_promote_contiguous_append_evidence``,
    ``raw_membership_retired_full_revision_siblings``,
    ``_raw_revision_source_path_has_divergent_evidence``) touch only
    ``store._ensure_source_conn()`` and, inside
    ``classify_raw_revision_cohort`` itself, ``store._blob_publisher``
    (verified by reading every line of that call graph, polylogue-sp72).
    Nothing in it touches ``_conn`` (index.db), ``_pending_raw_parse_states``,
    ``_preacquire_attachment_blobs``, ``_write_counts``, or
    ``_skipped_counts`` -- those Protocol members exist only because
    ``ArchiveStore`` (the Protocol's only other implementer) happens to
    carry them. This adapter implements them as typed stubs that raise if
    ever actually invoked, rather than reaching into ``ArchiveStore``'s
    ~9,000-line read surface just to satisfy an unused Protocol member.
    ``_conn`` is declared a plain settable attribute (not a read-only
    property) because the Protocol types it that way -- it is set to the
    same ``source_conn`` handle as a harmless placeholder that is never
    actually read by anything this adapter is used for.
    """

    def __init__(self, source_conn: sqlite3.Connection, blob_publisher: ArchiveBlobPublisher) -> None:
        self._source_conn = source_conn
        self._blob_publisher: ArchiveBlobPublisher | None = blob_publisher
        self.archive_root = blob_publisher.source_db_path.parent
        self._inactive_candidate_durable_read_only = False
        # Never read by bind_raw_revision/classify_raw_revision_cohort; see
        # class docstring for why this is a harmless placeholder value.
        self._conn = source_conn
        self._pending_raw_parse_states: list[tuple[str, RawSessionStateUpdate]] = []

    def _ensure_source_conn(self) -> sqlite3.Connection:
        return self._source_conn

    def commit(self) -> None:
        raise NotImplementedError("ingest_batch owns the index.db commit; this adapter never manages it")

    def _preacquire_attachment_blobs(
        self,
        session: ParsedSession,
        *,
        source_path: str,
        acquired_at_ms: int,
    ) -> tuple[dict[int, tuple[bytes | None, int, str]], tuple[ArchiveSourceBlobRef, ...]]:
        raise NotImplementedError(
            "_DriveRevisionGovernanceAdapter is used only for bind_raw_revision/"
            "classify_raw_revision_cohort, which never call this"
        )

    @staticmethod
    def _write_counts(session: ParsedSession) -> dict[str, int]:
        raise NotImplementedError(
            "_DriveRevisionGovernanceAdapter is used only for bind_raw_revision/"
            "classify_raw_revision_cohort, which never call this"
        )

    @staticmethod
    def _skipped_counts(session: ParsedSession, *, session_events: int = 0) -> dict[str, int]:
        raise NotImplementedError(
            "_DriveRevisionGovernanceAdapter is used only for bind_raw_revision/"
            "classify_raw_revision_cohort, which never call this"
        )


def _drive_structural_growth_predecessor(
    source_conn: sqlite3.Connection,
    blob_publisher: ArchiveBlobPublisher,
    *,
    raw_id: str,
    logical_source_key: str,
) -> tuple[str, int, str] | None:
    """Find a unique JSON-structural predecessor for ``raw_id``, if any.

    polylogue-1fijp AC (b): ``_bind_drive_revision_lineage``'s legacy path
    (below) only ever proves lineage via
    ``classify_raw_revision_cohort_for_live_watch``'s byte-prefix classifier
    (``archive/revision_authority.py``), which -- per PR #3656's finding --
    can never recognize the realistic ``_inject_live_drive_attachment_bytes``
    growth shape (a whole-document JSON re-serialization, not a byte-append).
    This looks for exactly one existing ``revision_kind='full'`` sibling for
    ``logical_source_key`` whose bytes are a JSON-structural predecessor of
    ``raw_id``'s own bytes (see ``sources.drive.structural_diff``) and, if
    found, returns ``(predecessor_raw_id, predecessor_generation,
    baseline_raw_id)`` for the caller to bind directly.

    Returns ``None`` on zero or more-than-one structural match (never guess
    between competing candidates), or when ``raw_id``'s own row cannot be
    found -- the caller falls through to the existing byte-prefix
    quarantine-then-classify path unchanged in every such case, so this is a
    strictly additive typed-lineage improvement, never a narrowing of what
    the legacy path already proves.
    """
    # Import lazily: this classifier is only needed for this Drive-specific
    # lineage branch, so it stays out of the batch module's import graph and
    # off the ingest-worker entry point's startup cost.  (The import cycle the
    # comment here used to name does not exist: sources/live/__init__.py is
    # fully lazy and sources/live/admission.py references neither
    # batch_support nor ingest_batch.)
    from polylogue.sources.drive.structural_diff import DriveStructuralRelation, classify_drive_structural_relation

    new_row = source_conn.execute(
        "SELECT lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id = ?",
        (raw_id,),
    ).fetchone()
    if new_row is None or new_row[0] is None:
        return None
    sibling_rows = source_conn.execute(
        """
        SELECT raw_id, lower(hex(blob_hash)), acquisition_generation, baseline_raw_id
        FROM raw_sessions
        WHERE logical_source_key = ? AND revision_kind = 'full' AND raw_id != ?
        """,
        (logical_source_key, raw_id),
    ).fetchall()
    if not sibling_rows:
        return None
    new_bytes = blob_publisher.read_all(str(new_row[0]))
    matches: list[tuple[str, int, str]] = []
    for row in sibling_rows:
        sibling_blob_hash = row[1]
        if sibling_blob_hash is None:
            continue
        sibling_raw_id = str(row[0])
        sibling_bytes = blob_publisher.read_all(str(sibling_blob_hash))
        relation = classify_drive_structural_relation(sibling_bytes, new_bytes)
        if relation is DriveStructuralRelation.STRUCTURAL_GROWTH:
            generation = int(row[2]) if row[2] is not None else 0
            baseline_raw_id = str(row[3]) if row[3] else sibling_raw_id
            matches.append((sibling_raw_id, generation, baseline_raw_id))
    if len(matches) != 1:
        return None
    return matches[0]


def _bind_drive_revision_lineage(
    session_to_write: ParsedSession,
    *,
    raw_id: str | None,
    source_conn: sqlite3.Connection | None,
    blob_publisher: ArchiveBlobPublisher | None,
    cohort_cache: DriveRevisionCohortCache | None = None,
) -> RevisionReplayPlan | None:
    """Best-effort revision-lineage bookkeeping for Drive re-acquisitions.

    polylogue-sp72: ``iter_drive_raw_data`` backfills live-fetched
    attachment bytes into cached Drive JSON on every ingest pass, minting a
    brand-new ``raw_id`` for the SAME logical session whenever the bytes
    change. This module's generic write path (``_write_session``, used by
    every non-tailed-watcher provider) only ever compares content-hash and
    freshness timestamps -- it never computes ``logical_source_key`` or
    calls the revision-governance cohort classifier the way the governed
    "live" batch path does for tailed origins (``sources/live/batch.py``,
    ``sources/live/append_ingest.py``). Confirmed live: every one of 157
    duplicate ``aistudio-drive`` raw pairs carries ``revision_kind='unknown'``,
    ``logical_source_key=NULL``, ``revision_authority='quarantined'`` -- no
    predecessor/baseline linkage at all.

    This mirrors ``live/batch.py``'s post-parse ``logical_source_key``
    computation and its ``bind_raw_revision``/``classify_raw_revision_cohort``
    call for a single-session raw with no pre-existing membership census,
    including its guard against double-governing an identity already owned
    by membership-census governance (``raw_membership_raw_ids``). Unlike
    ``live/batch.py``, this call is called unconditionally for every Drive
    raw that reaches ``_write_session`` -- including ones ``_write_session``
    ultimately skips as stale/duplicate -- so lineage metadata (and any
    future arbitration/rebuild consumer reading it, e.g. polylogue-x1gd) is
    recorded for every acquisition, not only the one that happened to win
    ``_write_session``'s own freshness/content-hash comparison. This
    function is non-fatal best-effort bookkeeping -- any failure (including
    ``classify_raw_revision_cohort`` requiring a writable blob publisher) is
    logged and swallowed (returning ``None``), never allowed to break the
    session write itself.

    polylogue-ojjet: ``cohort_cache``, when supplied, serves the cohort's
    repeated blob reads from a pass-scoped byte cache (see
    :class:`DriveRevisionCohortCache`). It bounds only repeated work -- the
    call above stays unconditional for every Drive raw, no cohort is capped,
    and the classifier still sees the identical bytes -- so the lineage this
    function records is unchanged.

    Returns the classifier's :class:`RevisionReplayPlan` on success (``None``
    on any early-return or swallowed failure) so the caller can tell whether
    ``raw_id`` is a governance-*proven* member of the accepted chain --
    see ``_write_session``'s use of this to bypass the #3453
    freshness-tie-break safety net once real lineage evidence has already
    settled the question that heuristic exists to guess at.

    Every real config and test fixture configures Drive acquisition as
    ``Source(name="gemini", folder=...)``, and ``iter_drive_raw_data`` sets
    ``provider_hint = Provider.from_string(source.name)`` -- so a parsed
    Drive session's ``source_name`` is ``Provider.GEMINI``, not
    ``Provider.DRIVE`` (both map to the same ``Origin.AISTUDIO_DRIVE``, see
    ``core/sources.py``). Gate on the value actually observed on the wire.

    polylogue-1fijp AC (b): before falling back to the legacy byte-prefix
    quarantine-then-classify dance, this first tries
    ``_drive_structural_growth_predecessor`` -- a JSON-structural-diff-aware
    check for the exact realistic re-acquisition shape (whole-document
    re-serialization, not a byte-append) that the byte-prefix classifier can
    never prove. A unique structural match is bound directly as a
    ``FULL``/``ASSERTED`` revision with a real ``predecessor_raw_id`` --
    ``ASSERTED``, not ``BYTE_PROVEN``, because the proof is JSON-structural,
    not byte-level (the existing byte-relation vocabulary in
    ``archive/revision_authority.py``/``storage/sqlite/archive_tiers/
    raw_admission.py`` deliberately reserves ``BYTE_PROVEN`` for the
    ``bytes.startswith()`` relation). When no unique structural match exists,
    behavior is byte-for-byte identical to before this change.
    """
    if source_conn is None or blob_publisher is None:
        return None
    if not raw_id:
        return None
    if session_to_write.source_name is not Provider.GEMINI:
        return None
    if source_conn.execute("PRAGMA query_only").fetchone()[0]:
        with closing(
            open_isolated_write_connection(
                blob_publisher.source_db_path,
                purpose="Drive revision lineage",
                archive_root=blob_publisher.source_db_path.parent,
            )
        ) as writer:
            return _bind_drive_revision_lineage(
                session_to_write,
                raw_id=raw_id,
                source_conn=writer,
                blob_publisher=blob_publisher,
                cohort_cache=cohort_cache,
            )
    logical_source_key = (
        f"{origin_from_provider(session_to_write.source_name).value}:{session_to_write.provider_session_id}"
    )
    if cohort_cache is not None:
        blob_publisher = _CohortCachingBlobPublisher(blob_publisher, cohort_cache)
    adapter = _DriveRevisionGovernanceAdapter(source_conn, blob_publisher)
    try:
        if raw_membership_raw_ids(adapter, logical_source_key):
            return None
        structural_match = _drive_structural_growth_predecessor(
            source_conn,
            blob_publisher,
            raw_id=raw_id,
            logical_source_key=logical_source_key,
        )
        if structural_match is not None:
            predecessor_raw_id, predecessor_generation, baseline_raw_id = structural_match
            bind_raw_revision(
                adapter,
                raw_id,
                RawRevisionEnvelope(
                    logical_source_key=logical_source_key,
                    kind=RawRevisionKind.FULL,
                    source_revision=raw_id,
                    predecessor_raw_id=predecessor_raw_id,
                    baseline_raw_id=baseline_raw_id,
                    acquisition_generation=predecessor_generation + 1,
                    authority=RawRevisionAuthority.ASSERTED,
                ),
            )
            return RevisionReplayPlan(
                logical_source_key=logical_source_key,
                applications=(),
                accepted_chain=(raw_id,),
            )
        bind_raw_revision(
            adapter,
            raw_id,
            RawRevisionEnvelope(
                logical_source_key=logical_source_key,
                kind=RawRevisionKind.FULL,
                source_revision=raw_id,
                acquisition_generation=0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        return classify_raw_revision_cohort_for_live_watch(adapter, logical_source_key)
    except Exception:
        logger.warning(
            "drive_revision_lineage_bind_failed",
            raw_id=raw_id,
            logical_source_key=logical_source_key,
            exc_info=True,
        )
        return None


def _write_session(
    conn: sqlite3.Connection,
    payload: SessionWritePayload,
    *,
    force_write: bool = False,
    signature_cache: LineageSignatureCache | dict[str, list[tuple[str, str]]] | None = None,
    stage_timings_s: dict[str, float] | None = None,
    blob_publisher: ArchiveBlobPublisher | None = None,
    pending_attachment_receipts: list[tuple[str, bytes]] | None = None,
    source_conn: sqlite3.Connection | None = None,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    attachment_owner_resolutions: list[dict[str, str]] | None = None,
    drive_plans: Mapping[str, RevisionReplayPlan | None] | None = None,
    drive_cohort_cache: DriveRevisionCohortCache | None = None,
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
    session_to_write = payload.parsed_session
    merge_append = False
    append_force_replace = False
    freshness_force_replace = False
    browser_precedence: BrowserCapturePrecedence = "default"

    drive_revision_plan = (
        drive_plans.get(payload.session_id)
        if drive_plans is not None
        else _bind_drive_revision_lineage(
            session_to_write,
            raw_id=payload.raw_id,
            source_conn=source_conn,
            blob_publisher=blob_publisher,
            cohort_cache=drive_cohort_cache,
        )
    )
    # polylogue-sp72 AC2: once real byte-prefix lineage evidence has proven
    # this raw is the classifier's accepted chain head for its logical
    # source key, the #3453 freshness-tie heuristic below no longer needs to
    # guess at a question governance has already answered -- it stays purely
    # a safety net for the ungoverned case (no source_conn/blob_publisher,
    # non-Gemini Drive identity, or a classification failure).
    drive_revision_proven_winner = (
        drive_revision_plan is not None
        and payload.raw_id is not None
        and payload.raw_id in drive_revision_plan.accepted_chain
    )

    if revision_authority_refuses_write(
        conn,
        source_conn,
        session_id=payload.session_id,
        raw_id=payload.raw_id or "",
        provider_session_id=payload.parsed_session.provider_session_id,
    ):
        _repair_stale_revision_observations(conn, payload)
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
            _repair_stale_revision_observations(conn, payload)
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
            _repair_stale_revision_observations(conn, payload)
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
            and not drive_revision_proven_winner
            and _incoming_write_regresses_attachment_coverage(conn, payload, session_to_write)
            # Attachment coverage alone cannot tell a re-acquisition of the
            # same transcript from a genuinely different revision that happens
            # to tie on content-derived freshness (polylogue-5uoed). Skipping
            # the latter loses its distinct messages on a fresh import.
            and not _incoming_write_carries_distinct_messages(conn, payload, session_to_write)
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
        newer_revision = (
            incoming_freshness_ms is not None
            and existing_updated_at_int is not None
            and incoming_freshness_ms > existing_updated_at_int
        )
        delta: ParsedSession | None = None
        if newer_revision and _append_payload_changes_existing_message(conn, payload):
            # A later full revision can revise an already-seen native message
            # as well as add a tail. Replace it authoritatively so message,
            # block, attachment, and event projections all come from the same
            # revision. Older/equal replays continue through the append delta
            # path and retain its unchanged-row contract.
            counts["skipped_messages"] = 0
            append_force_replace = True
        else:
            delta, skipped_messages = _append_delta_payload(conn, payload)
            counts["skipped_messages"] = skipped_messages
        if not append_force_replace:
            if delta is None:
                if payload.parsed_session.ingest_flags:
                    upsert_parser_ingest_flag_tags(conn, payload.session_id, payload.parsed_session.ingest_flags)
                counts["raw_links"] = int(_refresh_session_raw_link(conn, payload.session_id, payload.raw_id))
                counts["skipped_sessions"] = 1
                counts["skipped_attachments"] = payload.attachment_count
                counts["skipped_session_events"] = len(payload.parsed_session.session_events)
                if _needs_session_fts_repair(conn, payload.session_id):
                    counts[_FTS_REPAIR_COUNT_KEY] = 1
                return False, counts
            session_to_write = delta
            merge_append = True

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

    preacquired_attachment_blobs: dict[int, tuple[bytes | None, int, str]] | None = None
    publication_receipts: list[tuple[str, bytes]] = []
    if blob_publisher is not None:
        preacquired_attachment_blobs = {}
        for attachment in session_to_write.attachments:
            if attachment.inline_bytes is None:
                continue
            hash_hex, size = blob_publisher.write_from_bytes(attachment.inline_bytes)
            receipt_id = blob_publisher.receipt_id(hash_hex)
            blob_hash = bytes.fromhex(hash_hex)
            preacquired_attachment_blobs[id(attachment)] = (blob_hash, size, "acquired")
            if receipt_id is not None:
                publication_receipts.append((receipt_id, blob_hash))
        session_to_write, sidecar_blob_counts = _preacquire_sidecar_blobs(
            session_to_write, blob_publisher, publication_receipts
        )
        counts.update(sidecar_blob_counts)
        blob_publisher.flush()
    for attachment in session_to_write.attachments:
        # bd polylogue-8ac0: bytes for this attachment were already streamed
        # into the blob store during sidecar discovery (e.g. ChatGPT ``.dat``
        # asset acquisition) -- record the already-known hash/size directly
        # rather than re-hashing. Independent of ``blob_publisher`` (no new
        # write happens here) and skipped when ``inline_bytes`` already
        # claimed this attachment above.
        if attachment.inline_bytes is not None or attachment.precomputed_blob is None:
            continue
        if preacquired_attachment_blobs is None:
            preacquired_attachment_blobs = {}
        hash_hex, size = attachment.precomputed_blob
        preacquired_attachment_blobs[id(attachment)] = (bytes.fromhex(hash_hex), size, "acquired")

    prepared_write = (
        prepare_session_write(
            conn,
            session_to_write,
            merge_append=merge_append,
            fallback_timestamp=payload.fallback_timestamp,
            source_conn=source_conn,
            signature_cache=signature_cache,
        )
        if prepared_writes is not None
        else None
    )
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
        pending_input_content_hash=(
            prepared_write.input_content_hash.hex()
            if prepared_write is not None
            else (bound_session_content_hash(session_to_write) if merge_append else None)
        ),
        prepared_write=prepared_write,
        raw_id=payload.raw_id,
        fallback_timestamp=payload.fallback_timestamp,
        source_conn=source_conn,
        merge_append=merge_append,
        force_replace=(
            force_write or browser_precedence == "replace" or append_force_replace or freshness_force_replace
        ),
        signature_cache=signature_cache,
        stage_timings_s=stage_timings_s,
        preacquired_attachment_blobs=preacquired_attachment_blobs,
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
    if writer_outcomes and writer_outcomes[0].stale_skipped:
        _repair_stale_revision_observations(conn, payload)
        counts["skipped_sessions"] = 1
        counts["skipped_messages"] = payload.message_count
        counts["skipped_attachments"] = payload.attachment_count
        counts["skipped_session_events"] = len(payload.parsed_session.session_events)
        return False, counts
    if pending_attachment_receipts is not None:
        pending_attachment_receipts.extend(publication_receipts)
    if prepared_writes is not None and prepared_write is not None:
        prepared_writes.append(prepared_write)
    if attachment_owner_resolutions is not None and writer_outcomes:
        for attachment_id, reason in writer_outcomes[0].unresolved_attachment_owners:
            attachment_owner_resolutions.append(
                {
                    "raw_id": payload.raw_id or "",
                    "session_id": payload.session_id,
                    "attachment_id": attachment_id,
                    "reason": reason.value,
                }
            )
    counts["sessions"] = 1
    counts["messages"] = len(session_to_write.messages)
    counts["attachments"] = len(session_to_write.attachments)
    counts["session_events"] = len(session_to_write.session_events)

    return True, counts


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


def _record_outcome(summary: _IngestBatchSummary, ir: IngestRecordResult) -> None:
    summary.outcomes[ir.raw_id] = _RawIngestOutcome(
        raw_id=ir.raw_id,
        payload_provider=ir.payload_provider,
        validation_status=ir.validation_status,
        validation_error=ir.validation_error,
        parse_error=ir.parse_error,
        error=ir.error,
        had_sessions=bool(ir.sessions),
        outcome_code=ir.outcome_code,
        retryable=ir.retryable,
        evidence_ref=ir.evidence_ref,
        remediation=ir.remediation,
        diagnostic=ir.diagnostic,
    )
    summary.expected_marker_session_counts[ir.raw_id] = len(ir.sessions)
    if ir.sessions:
        summary.marker_request_sessions_by_raw_id[ir.raw_id] = [
            {
                "session_id": cdata.session_id,
                "input_content_hash": cdata.content_hash,
                "parser_fingerprint": parser_fingerprint_for_origin(
                    origin_from_provider(cdata.parsed_session.source_name)
                ),
                "lowering_fingerprint": lowering_fingerprint(),
                "marker_recipe_fingerprint": marker_recipe_fingerprint(),
            }
            for cdata in ir.sessions
        ]
    if ir.serialized_size_bytes is not None:
        summary.total_result_bytes += ir.serialized_size_bytes
        if ir.serialized_size_bytes > summary.max_result_bytes:
            summary.max_result_bytes = ir.serialized_size_bytes
            summary.max_result_raw_id = ir.raw_id
    if ir.schema_drift is not None:
        summary.schema_drift_observations.append(ir.schema_drift)


def _marker_request_facts(record: RawSessionRecord, *, validation_mode: str) -> dict[str, object]:
    """Bind marker identity to immutable acquisition facts, not parse projections.

    ``payload_provider`` is a read projection of ``detected_provider``. The
    ordinary acceptance transaction updates it, so using it here would make a
    replay's request key change after the first successful parse. Parser
    identity for the actual interpretation is separately carried per session.
    """
    acquisition_provider = Provider.from_string(record.source_name or "unknown")
    acquisition_origin = origin_from_provider(acquisition_provider)
    # Drive lineage governance refines the raw row's revision envelope after
    # parsing. That envelope is a mutable derived projection, not an
    # acquisition fact, so including it would give the same retained bytes a
    # new marker request key on retry. The immutable raw_id/blob/path/index
    # facts below distinguish Drive acquisitions; parsed session bindings add
    # the normalized session identity. Other providers retain their supplied
    # revision evidence in the request key.
    drive_revision_projection = acquisition_origin is Origin.AISTUDIO_DRIVE
    revision = None if drive_revision_projection else record.revision
    return {
        "blob_digest": record.blob_hash,
        "origin": acquisition_origin.value,
        "source_path": record.source_path,
        "source_index": record.source_index,
        "native_id": revision.logical_source_key if revision is not None else None,
        "revision": dataclasses.asdict(revision) if revision is not None else None,
        "source_name": record.source_name,
        "capture_mode": record.capture_mode.value if record.capture_mode is not None else None,
        "acquired_at": record.acquired_at,
        "file_mtime": record.file_mtime,
        "validation_mode": validation_mode,
        "marker_recipe_fingerprint": marker_recipe_fingerprint(),
        "lowering_fingerprint": lowering_fingerprint(),
        "parser_fingerprint": parser_fingerprint_for_origin(acquisition_origin),
    }


def _observe_current_rss(summary: _IngestBatchSummary) -> None:
    current_rss_mb = read_current_rss_mb()
    if current_rss_mb is None:
        return
    if summary.max_current_rss_mb is None or current_rss_mb > summary.max_current_rss_mb:
        summary.max_current_rss_mb = current_rss_mb


def _record_write_result(
    summary: _IngestBatchSummary,
    cdata: SessionWritePayload,
    *,
    content_changed: bool,
    counts: dict[str, int],
) -> None:
    summary.total_convos += 1
    summary.total_msgs += cdata.message_count

    ingest_changed = (
        counts["sessions"] + counts["messages"] + counts["attachments"] + counts["session_events"] + counts["raw_links"]
    ) > 0

    if ingest_changed or content_changed:
        summary.processed_ids.add(cdata.session_id)
    if content_changed:
        summary.changed_counts["sessions"] += 1
        summary.changed_session_ids.append(cdata.session_id)
        summary.fts_repair_session_ids.append(cdata.session_id)
    elif counts.get(_FTS_REPAIR_COUNT_KEY, 0):
        summary.fts_repair_session_ids.append(cdata.session_id)
    if counts["messages"]:
        summary.changed_counts["messages"] += counts["messages"]
    if counts["attachments"]:
        summary.changed_counts["attachments"] += counts["attachments"]
    if counts["session_events"]:
        summary.changed_counts["session_events"] += counts["session_events"]
    for key, value in counts.items():
        if key in summary.counts:
            summary.counts[key] += value


def _reuse_current_accepted_marker_carrier(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection | None,
    ir: IngestRecordResult,
    *,
    summary: _IngestBatchSummary,
) -> bool:
    """Reuse an accepted carrier before the ordinary session writer runs.

    A matching current-incarnation witness proves this exact accepted request
    already crossed the index commit boundary. A pending carrier without its
    own witness can make the same proof when every requested session is
    already materialized with the retained input hash in that incarnation.
    That occurs if a rollback lost the first raw's index transaction and a
    different raw then published the identical normalized session. In either
    case replay must keep the carrier and skip session preparation, whose
    no-op disposition could otherwise look like a different marker
    interpretation. Other missing witnesses (including a replacement index)
    still fall through to the normal writer and exact-byte re-witness checks.
    """
    if source_conn is None or not ir.sessions:
        return False
    facts = summary.marker_request_facts_by_raw_id.get(ir.raw_id)
    request_sessions = summary.marker_request_sessions_by_raw_id.get(ir.raw_id)
    if facts is None or request_sessions is None:
        return False

    from polylogue.storage.accepted_marker_inputs import prepare_accepted_marker_input, retained_marker_input_sync

    probe = prepare_accepted_marker_input(
        ir.raw_id,
        (),
        request_facts=facts,
        request_sessions=request_sessions,
    )
    retained = retained_marker_input_sync(source_conn, probe.identity)
    if retained is None:
        return False
    state, batch, retained_incarnation = retained
    if state not in ("accepted", "pending"):
        return False

    filename = str(index_conn.execute("PRAGMA database_list").fetchone()[2])
    index_stat = Path(filename).stat()
    incarnation = index_conn.execute(
        "SELECT incarnation_id, device, inode FROM ingest_index_incarnation WHERE singleton = 1"
    ).fetchone()
    if incarnation is None or (int(incarnation[1]), int(incarnation[2])) != (index_stat.st_dev, index_stat.st_ino):
        raise AcceptedMarkerInputRefusedError("index incarnation changed during marker retry classification")
    current_incarnation_id = str(incarnation[0])
    if state == "pending" and retained_incarnation != current_incarnation_id:
        return False
    value = json.loads(batch.payload)
    dispositions = [
        {
            "session_id": str(session.get("session_id", "")),
            "disposition": str(session.get("disposition", "no-op")),
        }
        for session in value["sessions"]
    ]
    encoded = json.dumps(dispositions, sort_keys=True, separators=(",", ":"))
    witness = index_conn.execute(
        "SELECT carrier_digest, dispositions_json, incarnation_id FROM ingest_marker_witnesses WHERE request_key = ?",
        (probe.identity,),
    ).fetchone()
    if witness is not None:
        if tuple(witness) != (batch.payload_sha256, encoded, current_incarnation_id):
            return False
        summary.marker_batches_by_raw_id[ir.raw_id] = batch
        return True

    # A pending carrier is bound to this physical index incarnation. If an
    # intervening raw has already materialized every exact request input, it
    # supplies the missing successful index publication without permitting a
    # new carrier to replace the retained bytes. Do not infer this from mere
    # session IDs: a stale or different interpretation can name the same ID.
    if state != "pending":
        return False
    for binding in request_sessions:
        session_id = binding.get("session_id")
        input_content_hash = binding.get("input_content_hash")
        if not isinstance(session_id, str) or not session_id:
            return False
        if not isinstance(input_content_hash, str) or len(input_content_hash) != 64:
            return False
        row = index_conn.execute(
            "SELECT lower(hex(content_hash)) FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None or str(row[0]) != input_content_hash.lower():
            return False
    summary.marker_batches_by_raw_id[ir.raw_id] = batch
    return True


class FtsTriggerRestorationError(RuntimeError):
    """Raised when the bulk-ingest FTS trigger suspension cannot be undone.

    A dropped-trigger window that survives the batch leaves every subsequent
    ordinary write out of ``messages_fts`` with no error anywhere, so this
    failure is escalated rather than suppressed (polylogue-qoa75).
    """


_SESSION_WRITE_SAVEPOINT = "ingest_session_write"


def _write_session_entry(
    conn: sqlite3.Connection,
    raw_id: str,
    cdata: SessionWritePayload,
    *,
    summary: _IngestBatchSummary,
    force_write: bool = False,
    signature_cache: LineageSignatureCache | dict[str, list[tuple[str, str]]] | None = None,
    blob_publisher: ArchiveBlobPublisher | None = None,
    pending_attachment_receipts: list[tuple[str, bytes]] | None = None,
    source_conn: sqlite3.Connection | None = None,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    drive_plans: Mapping[str, RevisionReplayPlan | None] | None = None,
    drive_cohort_cache: DriveRevisionCohortCache | None = None,
) -> bool:
    # polylogue-qoa75: when the batch owns the transaction the writer must not
    # manage one of its own, or its `with conn:` commits the batch's
    # BEGIN IMMEDIATE (and the FTS-trigger DROP inside it) at the first
    # session. Per-session failure isolation -- which the `except Exception`
    # below relies on to keep draining the batch -- is then provided by a
    # SAVEPOINT rather than by a per-session commit, so a failed session
    # contributes no partial rows and the batch transaction stays open.
    batch_owns_transaction = conn.in_transaction
    if batch_owns_transaction:
        conn.execute(f"SAVEPOINT {_SESSION_WRITE_SAVEPOINT}")
    try:
        t_write = time.perf_counter()
        write_stage_timings: dict[str, float] = {}
        prepared_writes: list[PreparedSessionWrite] = []
        content_changed, counts = _write_session(
            conn,
            cdata,
            manage_transaction=not batch_owns_transaction,
            force_write=force_write,
            signature_cache=signature_cache,
            stage_timings_s=write_stage_timings,
            blob_publisher=blob_publisher,
            pending_attachment_receipts=pending_attachment_receipts,
            source_conn=source_conn,
            fresh_build=fresh_build,
            fresh_build_batch=fresh_build_batch,
            attachment_owner_resolutions=summary.attachment_owner_resolutions,
            drive_plans=drive_plans,
            drive_cohort_cache=drive_cohort_cache,
            prepared_writes=prepared_writes,
        )
        marker_write = prepared_writes[0] if prepared_writes else None
        for stage, elapsed_s in write_stage_timings.items():
            summary.stage_timings_s[stage] = summary.stage_timings_s.get(stage, 0.0) + elapsed_s
        write_elapsed = time.perf_counter() - t_write
        summary.write_elapsed_s += write_elapsed
        if write_elapsed > summary.max_write_elapsed_s:
            summary.max_write_elapsed_s = write_elapsed
        _record_write_result(
            summary,
            cdata,
            content_changed=content_changed,
            counts=counts,
        )
        summary.marker_session_dispositions_by_raw_id.setdefault(raw_id, []).append(
            {
                "session_id": cdata.session_id,
                "disposition": (
                    "append"
                    if marker_write is not None and marker_write.merge_append
                    else "replace"
                    if marker_write is not None
                    else "no-op"
                ),
            }
        )
        if write_elapsed >= 1.0:
            logger.info(
                "slow_write",
                cid=cdata.session_id[:20],
                elapsed_s=round(write_elapsed, 2),
                msgs=cdata.message_count,
                changed_messages=counts["messages"],
                skipped_messages=counts["skipped_messages"],
                changed_session_events=counts["session_events"],
                attachments=cdata.attachment_count,
                stage_top=_top_stage_timings(write_stage_timings),
            )
        if batch_owns_transaction:
            conn.execute(f"RELEASE {_SESSION_WRITE_SAVEPOINT}")
        if marker_write is not None:
            marker_session: dict[str, object] = {
                "session_id": marker_write.session_id,
                "input_content_hash": marker_write.input_content_hash.hex(),
                "disposition": "append" if marker_write.merge_append else "replace",
                "parser_fingerprint": parser_fingerprint_for_origin(
                    origin_from_provider(cdata.parsed_session.source_name)
                ),
                "lowering_fingerprint": lowering_fingerprint(),
                "marker_recipe_fingerprint": marker_recipe_fingerprint(),
                "candidates": marker_candidates_for_prepared_write(marker_write),
            }
            summary.marker_sessions_by_raw_id.setdefault(raw_id, []).append(marker_session)
        return True
    except Exception as exc:
        if batch_owns_transaction:
            # Discard only this session's rows; the batch transaction (and its
            # suspended FTS triggers) survives so the drain can continue.
            conn.execute(f"ROLLBACK TO {_SESSION_WRITE_SAVEPOINT}")
            conn.execute(f"RELEASE {_SESSION_WRITE_SAVEPOINT}")
        logger.error("Error writing session: %s", exc)
        summary.parse_failures += 1
        summary.failed_raw_ids[raw_id] = str(exc)[:500]
        return False


def _top_stage_timings(stage_timings_s: dict[str, float], *, limit: int = 5) -> dict[str, float]:
    if not stage_timings_s:
        return {}
    return {
        stage: round(elapsed_s, 3)
        for stage, elapsed_s in sorted(stage_timings_s.items(), key=lambda item: item[1], reverse=True)[:limit]
    }


def _delete_stale_sessions_for_raw_entries(conn: sqlite3.Connection, ready_entries: list[_SessionEntry]) -> None:
    if not hasattr(conn, "execute"):
        return

    expected_by_raw_id: dict[str, set[str]] = {}
    for raw_id, cdata in ready_entries:
        if raw_id:
            expected_by_raw_id.setdefault(raw_id, set()).add(cdata.session_id)

    for raw_id, expected_session_ids in expected_by_raw_id.items():
        if not expected_session_ids:
            continue
        placeholders = ",".join("?" for _ in expected_session_ids)
        stale_session_ids = [
            str(row[0])
            for row in conn.execute(
                f"SELECT session_id FROM sessions WHERE raw_id = ? AND session_id NOT IN ({placeholders})",
                (raw_id, *sorted(expected_session_ids)),
            ).fetchall()
        ]
        if not stale_session_ids:
            continue
        _delete_sessions_without_fk_cascade(conn, stale_session_ids)
        conn.execute(
            f"DELETE FROM sessions WHERE raw_id = ? AND session_id NOT IN ({placeholders})",
            (raw_id, *sorted(expected_session_ids)),
        )


def _delete_sessions_without_fk_cascade(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Apply sessions FK actions manually while bulk ingest has FKs disabled."""
    if not session_ids:
        return
    placeholders = ",".join("?" for _ in session_ids)
    params = tuple(session_ids)
    for table_name, from_column, on_delete in _session_foreign_key_actions(conn):
        table = _quote_identifier(table_name)
        column = _quote_identifier(from_column)
        if table_name == "sessions" or on_delete.upper() == "SET NULL":
            conn.execute(f"UPDATE {table} SET {column} = NULL WHERE {column} IN ({placeholders})", params)
        elif on_delete.upper() == "CASCADE":
            conn.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", params)


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


def _drain_ready_session_entries(
    conn: sqlite3.Connection,
    ready_entries: list[_SessionEntry],
    *,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    force_write: bool = False,
    blob_publisher: ArchiveBlobPublisher | None = None,
    pending_attachment_receipts: list[tuple[str, bytes]] | None = None,
    source_conn: sqlite3.Connection | None = None,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    drive_plans: Mapping[str, RevisionReplayPlan | None] | None = None,
) -> int:
    if not fresh_build:
        _delete_stale_sessions_for_raw_entries(conn, ready_entries)
    written_count = 0
    # One signature cache per drained batch memoizes each session's own composed
    # signatures so a parent with K fork-children is computed once, not K times
    # (#2475, hotspot 1). Entries are invalidated when their own rows are
    # rewritten or re-extracted in this same batch.
    # The cache carries own and composed lineage signatures with a weighted
    # byte bound. A plain dict remains accepted by lower-level/test callers as
    # the explicit unbounded compatibility path.
    signature_cache = LineageSignatureCache()
    # polylogue-ojjet: one Drive revision-cohort blob cache per drained
    # batch, the same lifetime as the signature cache above.
    drive_cohort_cache = DriveRevisionCohortCache()
    if fresh_build and fresh_build_batch is None:
        fresh_build_batch = set()
    for raw_id, cdata in _topo_sort_session_entries(ready_entries):
        wrote = _write_session_entry(
            conn,
            raw_id,
            cdata,
            summary=summary,
            force_write=force_write,
            signature_cache=signature_cache,
            blob_publisher=blob_publisher,
            pending_attachment_receipts=pending_attachment_receipts,
            source_conn=source_conn,
            fresh_build=fresh_build,
            fresh_build_batch=fresh_build_batch,
            drive_plans=drive_plans,
            drive_cohort_cache=drive_cohort_cache,
        )
        discard_session_data_payload(cdata)
        if not wrote:
            continue
        written_count += 1
        materialized_ids.add(cdata.session_id)
    return written_count


def _run_ingest_record(
    raw_record: RawSessionRecord,
    request: _IngestWorkerRequest,
) -> IngestRecordResult:
    return ingest_record(
        raw_record,
        request.archive_root_str,
        request.validation_mode,
        request.measure_ingest_result_size,
        blob_root_str=request.blob_root_str,
    )


def _iter_ingest_results_sync(
    raw_artifacts: list[RawSessionRecord],
    *,
    request: _IngestWorkerRequest,
    worker_count: int,
    heartbeat: IngestHeartbeat | None = None,
    progress: _WorkerProgress | None = None,
    chunk_size: int = 0,
    force_process_pool: bool = False,
) -> Iterable[IngestRecordResult]:
    """Yield ingest results, optionally chunked to bound parsed-result memory."""
    total = len(raw_artifacts)
    if progress is not None:
        progress.total_raw_count = total
        progress.completed_raw_count = 0
        progress.in_flight_raw_ids.clear()

    if chunk_size <= 0 or total <= chunk_size:
        yield from _iter_ingest_results_chunk(
            raw_artifacts,
            request=request,
            worker_count=worker_count,
            heartbeat=heartbeat,
            progress=progress,
            force_process_pool=force_process_pool,
        )
        return

    for chunk_start in range(0, total, chunk_size):
        chunk = raw_artifacts[chunk_start : chunk_start + chunk_size]
        yield from _iter_ingest_results_chunk(
            chunk,
            request=request,
            worker_count=worker_count,
            heartbeat=heartbeat,
            progress=progress,
            force_process_pool=force_process_pool,
        )


def _iter_ingest_results_chunk(
    raw_artifacts: list[RawSessionRecord],
    *,
    request: _IngestWorkerRequest,
    worker_count: int,
    heartbeat: IngestHeartbeat | None = None,
    progress: _WorkerProgress | None = None,
    force_process_pool: bool = False,
) -> Iterable[IngestRecordResult]:
    """Process one chunk of raw_artifacts through the process pool."""
    if worker_count <= 1 and not force_process_pool:
        for raw_record in raw_artifacts:
            if progress is not None:
                progress.in_flight_raw_ids[:] = [raw_record.raw_id]
            if heartbeat is not None:
                heartbeat()
            yield _run_ingest_record(raw_record, request)
            if progress is not None:
                progress.completed_raw_count += 1
                progress.in_flight_raw_ids.clear()
        return
    executor = None
    stalled = False
    try:
        executor = process_pool_executor(max_workers=max(1, worker_count))
        raw_iter = iter(raw_artifacts)
        futures: dict[Future[IngestRecordResult], str] = {}
        max_in_flight = max(1, worker_count)

        def submit_next() -> bool:
            try:
                raw_record = next(raw_iter)
            except StopIteration:
                return False
            future = executor.submit(_run_ingest_record, raw_record, request)
            futures[future] = raw_record.raw_id
            if progress is not None:
                progress.in_flight_raw_ids[:] = list(futures.values())
            return True

        for _ in range(max_in_flight):
            if not submit_next():
                break
        last_progress_at = time.monotonic()
        while futures:
            remaining_deadline = _INGEST_RESULT_PROGRESS_DEADLINE_S - (time.monotonic() - last_progress_at)
            done, _pending = wait(
                tuple(futures),
                timeout=max(0.0, min(_INGEST_RESULT_WAIT_HEARTBEAT_S, remaining_deadline)),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if heartbeat is not None:
                    heartbeat()
                if time.monotonic() - last_progress_at >= _INGEST_RESULT_PROGRESS_DEADLINE_S:
                    # A completion may race the timed wait (and test doubles
                    # are allowed to report an empty ``done`` set). Re-check
                    # readiness before refusing anything so a result that was
                    # completed inside the deadline is never replaced by a
                    # retryable timeout outcome.
                    done = {future for future in futures if future.done()}
                    if not done:
                        stalled = True
                        unfinished = tuple(futures.items())
                        logger.warning(
                            "ingest worker progress deadline exceeded; refusing %d unfinished raw item(s)",
                            len(unfinished),
                        )
                        for future, raw_id in unfinished:
                            future.cancel()
                            stall_error = "worker progress deadline exceeded; retryable stalled/refused result"
                            stall_disposition = transient_error_disposition(
                                evidence_ref="worker:progress_deadline",
                                diagnostic=stall_error,
                            )
                            yield IngestRecordResult(
                                raw_id=raw_id,
                                error=stall_error,
                                # polylogue-u1ww0: without an explicit
                                # disposition this refusal inherited the
                                # dataclass default and was persisted as a
                                # non-retryable success.
                                outcome_code=stall_disposition.outcome_code,
                                retryable=stall_disposition.retryable,
                                evidence_ref=stall_disposition.evidence_ref,
                                remediation=stall_disposition.remediation,
                                diagnostic=stall_disposition.diagnostic,
                            )
                        futures.clear()
                        # ``Future.cancel()`` cannot stop a task that is already
                        # executing in a worker process, and neither can
                        # ``shutdown(cancel_futures=True)``. Without an explicit
                        # terminate, every stalled pass left its running workers
                        # alive holding CPU and memory, and repeated passes
                        # accumulated them for the life of the daemon.
                        if isinstance(executor, ProcessPoolExecutor):
                            terminate_process_pool(executor)
                        if progress is not None:
                            # Refused work is no longer owned by this
                            # coordinator. Leaving these ids in the progress
                            # snapshot would make a settled batch look as if
                            # its source cursor were still in flight.
                            progress.in_flight_raw_ids.clear()
                        continue
                else:
                    continue
            last_progress_at = time.monotonic()
            for future in done:
                raw_id = futures.pop(future)
                if progress is not None:
                    progress.in_flight_raw_ids[:] = list(futures.values())
                try:
                    result = future.result()
                except Exception as exc:
                    worker_disposition = parser_defect_disposition(
                        evidence_ref=f"worker:{type(exc).__name__}",
                        diagnostic=str(exc),
                    )
                    result = IngestRecordResult(
                        raw_id=raw_id,
                        error=f"worker: {exc}",
                        # polylogue-u1ww0: a crashed worker is a classified
                        # defect, not the dataclass default success.
                        outcome_code=worker_disposition.outcome_code,
                        retryable=worker_disposition.retryable,
                        evidence_ref=worker_disposition.evidence_ref,
                        remediation=worker_disposition.remediation,
                        diagnostic=worker_disposition.diagnostic,
                    )
                submit_next()
                if progress is not None:
                    progress.in_flight_raw_ids[:] = list(futures.values())
                    progress.completed_raw_count += 1
                yield result
    except (TypeError, pickle.PicklingError):
        for raw_record in raw_artifacts:
            if progress is not None:
                progress.in_flight_raw_ids[:] = [raw_record.raw_id]
            if heartbeat is not None:
                heartbeat()
            yield _run_ingest_record(raw_record, request)
            if progress is not None:
                progress.completed_raw_count += 1
                progress.in_flight_raw_ids.clear()
    finally:
        if executor is not None:
            shutdown = getattr(executor, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown(wait=not stalled, cancel_futures=True)
                except TypeError:
                    # Small test doubles and older executor implementations
                    # may not expose ``cancel_futures``.
                    shutdown(wait=not stalled)


def _select_ingest_worker_count(raw_artifacts: Sequence[_BlobSized], ingest_workers: int | None) -> int:
    return select_ingest_worker_count(
        raw_artifacts,
        ingest_workers,
        default_worker_limit=_DEFAULT_INGEST_WORKER_LIMIT,
    )


def _new_ingest_batch_summary(
    raw_artifacts: list[RawSessionRecord],
    *,
    ingest_workers: int | None,
) -> _IngestBatchSummary:
    summary = _IngestBatchSummary()
    summary.raw_record_count = len(raw_artifacts)
    summary.worker_count = _select_ingest_worker_count(raw_artifacts, ingest_workers)
    summary.total_blob_mb = sum(record.blob_size for record in raw_artifacts) / (1024 * 1024)
    return summary


def _make_ingest_worker_request(
    *,
    archive_root_str: str,
    blob_root_str: str,
    validation_mode: str,
    measure_ingest_result_size: bool,
) -> _IngestWorkerRequest:
    return _IngestWorkerRequest(
        archive_root_str=archive_root_str,
        blob_root_str=blob_root_str,
        validation_mode=validation_mode,
        measure_ingest_result_size=measure_ingest_result_size,
    )


def _record_failed_ingest_result(summary: _IngestBatchSummary, ir: IngestRecordResult) -> None:
    logger.error("Failed to ingest raw record", raw_id=ir.raw_id, error=ir.error)
    summary.parse_failures += 1
    summary.failed_raw_ids[ir.raw_id] = (ir.error or "unknown worker failure")[:500]


def _prepare_publication_payloads(
    ir: IngestRecordResult,
    *,
    summary: _IngestBatchSummary,
    publication_mode: PublicationMode = PublicationMode.OFF,
) -> tuple[PublicationPayload, ...]:
    """Encode one raw record before any of its index rows are written.

    Encoding first makes protocol reconciliation/backpressure an explicit raw
    failure rather than leaving an accepted source revision without an outbox
    payload.  The index transaction is still rebuildable pre-work; source-tier
    acceptance and these exact bytes are committed together later.
    """
    if publication_mode is PublicationMode.OFF:
        return ()
    payloads: list[PublicationPayload] = []
    payload_bytes = 0
    for cdata in ir.sessions:
        remaining_bytes = _SINEX_STAGED_PAYLOAD_LIMIT_BYTES - summary.publication_payload_bytes - payload_bytes
        payload = encode_parsed_session_publication(
            cdata.parsed_session,
            session_id=cdata.session_id,
            max_payload_bytes=remaining_bytes,
        )
        projected_bytes = summary.publication_payload_bytes + payload_bytes + payload.size_bytes
        if projected_bytes > _SINEX_STAGED_PAYLOAD_LIMIT_BYTES:
            raise PublicationBackpressureError(
                "Sinex exact-payload staging budget exceeded: "
                f"projected_bytes={projected_bytes} limit_bytes={_SINEX_STAGED_PAYLOAD_LIMIT_BYTES}"
            )
        payloads.append(payload)
        payload_bytes += payload.size_bytes
    return tuple(payloads)


def _drain_ingest_result(
    conn: sqlite3.Connection,
    ir: IngestRecordResult,
    *,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    publication_mode: PublicationMode = PublicationMode.OFF,
    primary_publication_service: PublicationService | None = None,
    ensure_index_transaction: Callable[[], None] | None = None,
    force_write: bool = False,
    blob_publisher: ArchiveBlobPublisher | None = None,
    pending_attachment_receipts: list[tuple[str, bytes]] | None = None,
    source_conn: sqlite3.Connection | None = None,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    drive_plans: Mapping[str, RevisionReplayPlan | None] | None = None,
    marker_acceptance_enabled: bool = False,
) -> None:
    _record_outcome(summary, ir)
    _observe_current_rss(summary)

    if ir.error:
        _record_failed_ingest_result(summary, ir)
        return

    if not ir.sessions:
        summary.skipped_raw_ids.add(ir.raw_id)
        return

    if marker_acceptance_enabled:
        session_ids = [cdata.session_id for cdata in ir.sessions]
        if len(session_ids) != len(set(session_ids)):
            raise AcceptedMarkerInputRefusedError(
                f"raw revision {ir.raw_id!r} contains duplicate normalized session IDs"
            )

    try:
        publication_payloads = _prepare_publication_payloads(
            ir,
            summary=summary,
            publication_mode=publication_mode,
        )
    except PublicationEncodingError as exc:
        logger.error(
            "Sinex publication payload rejected before accepted-revision write",
            raw_id=ir.raw_id,
            error_code=type(exc).__name__,
        )
        summary.parse_failures += 1
        summary.failed_raw_ids[ir.raw_id] = f"{type(exc).__name__}: {exc}"[:500]
        return

    if publication_mode is PublicationMode.PRIMARY:
        if primary_publication_service is None:
            raise PublicationEncodingError("primary ingest requires a pre-index publication service")
        object_ids = [payload.object_id for payload in publication_payloads]
        for payload in publication_payloads:
            primary_publication_service.stage_payload(payload)
        primary_publication_service.drain_once(object_ids=object_ids, limit=len(object_ids))
        if primary_publication_service.projection_blocked(object_ids):
            summary.publication_deferred_raw_ids.add(ir.raw_id)
            logger.info(
                "Sinex primary receipt deferred index projection",
                raw_id=ir.raw_id,
                object_count=len(object_ids),
            )
            return

    if ensure_index_transaction is not None:
        ensure_index_transaction()

    reuse_marker_carrier = (
        marker_acceptance_enabled
        and not force_write
        and _reuse_current_accepted_marker_carrier(conn, source_conn, ir, summary=summary)
    )

    drain_started = time.perf_counter()
    if reuse_marker_carrier:
        # Publication encoding/staging/admission above still runs in MIRROR
        # and PRIMARY. Only the state-dependent index session write is skipped;
        # the exact current-witness carrier continues through common
        # witness-check and source-finalization paths below.
        written_count = 0
    else:
        written_count = _drain_ready_session_entries(
            conn,
            [(ir.raw_id, cdata) for cdata in ir.sessions],
            summary=summary,
            materialized_ids=materialized_ids,
            force_write=force_write,
            blob_publisher=blob_publisher,
            pending_attachment_receipts=pending_attachment_receipts,
            source_conn=source_conn,
            fresh_build=fresh_build,
            fresh_build_batch=fresh_build_batch,
            drive_plans=drive_plans,
        )
    if written_count == 0:
        summary.skipped_raw_ids.add(ir.raw_id)
    # Keep the reconciled payload for both changed and duplicate revisions.
    # The source-tier raw acceptance transaction restages duplicates
    # idempotently, which also provides a safe backfill path when an operator
    # enables mirror/primary after an earlier off-mode ingest.
    if publication_payloads:
        summary.publication_payloads_by_raw_id[ir.raw_id] = list(publication_payloads)
        summary.publication_payload_bytes += sum(payload.size_bytes for payload in publication_payloads)
    summary.drain_elapsed_s += time.perf_counter() - drain_started
    _observe_current_rss(summary)


def _consume_ingest_results(
    conn: sqlite3.Connection,
    raw_artifacts: list[RawSessionRecord],
    *,
    worker_request: _IngestWorkerRequest,
    summary: _IngestBatchSummary,
    materialized_ids: set[str],
    publication_mode: PublicationMode,
    primary_publication_service: PublicationService | None = None,
    force_write: bool = False,
    heartbeat: IngestHeartbeat | None = None,
    progress: _WorkerProgress | None = None,
    ingest_result_chunk_size: int = 0,
    suspend_fts_triggers: bool = False,
    mark_fts_stale_on_suspend: bool = False,
    force_process_pool: bool = False,
    blob_publisher: ArchiveBlobPublisher | None = None,
    pending_attachment_receipts: list[tuple[str, bytes]] | None = None,
    source_conn: sqlite3.Connection | None = None,
    fresh_build: bool = False,
    marker_acceptance_enabled: bool = False,
) -> bool:
    result_iterator = iter(
        _iter_ingest_results_sync(
            raw_artifacts,
            request=worker_request,
            worker_count=summary.worker_count,
            heartbeat=heartbeat,
            progress=progress,
            chunk_size=ingest_result_chunk_size,
            force_process_pool=force_process_pool,
        )
    )
    transaction_started = False
    fresh_build_batch: set[str] | None = set() if fresh_build else None

    def ensure_index_transaction() -> None:
        nonlocal transaction_started
        if transaction_started:
            return
        if suspend_fts_triggers:
            conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN IMMEDIATE")
        if suspend_fts_triggers:
            from polylogue.storage.fts.fts_lifecycle import suspend_fts_triggers_sync

            suspend_fts_triggers_sync(conn, mark_stale=mark_fts_stale_on_suspend)
        transaction_started = True

    while True:
        wait_started = time.perf_counter()
        try:
            ir = next(result_iterator)
        except StopIteration:
            summary.teardown_elapsed_s = time.perf_counter() - wait_started
            break
        summary.result_wait_s += time.perf_counter() - wait_started
        release_after_drain = ingest_result_needs_memory_release(ir)
        try:
            _drain_ingest_result(
                conn,
                ir,
                summary=summary,
                materialized_ids=materialized_ids,
                publication_mode=publication_mode,
                primary_publication_service=primary_publication_service,
                ensure_index_transaction=ensure_index_transaction,
                force_write=force_write,
                blob_publisher=blob_publisher,
                pending_attachment_receipts=pending_attachment_receipts,
                source_conn=source_conn,
                fresh_build=fresh_build,
                fresh_build_batch=fresh_build_batch,
                marker_acceptance_enabled=marker_acceptance_enabled,
            )
        finally:
            discard_ingest_result_payload(ir)
            if release_after_drain:
                release_process_memory()
                _observe_current_rss(summary)
    return transaction_started


def _flush_ingest_results(
    conn: sqlite3.Connection,
    *,
    summary: _IngestBatchSummary,
) -> None:
    """Record final drain timing before committing.

    The commit + FTS trigger restore + FTS repair are deliberately
    deferred to ``_commit_sync_ingest_side_effects`` so they land as a
    single atomic write through the archive write gateway (#1242). This
    closes the gap where row commit and post-commit FTS repair were two
    separate transactions: if the repair failed, the data was already
    committed and the FTS index drifted silently.
    """
    flush_started = time.perf_counter()
    summary.flush_elapsed_s = time.perf_counter() - flush_started
    _observe_current_rss(summary)


def _commit_sync_ingest_side_effects(
    conn: sqlite3.Connection,
    *,
    db_path: Path,
    changed_session_ids: Sequence[str],
    repair_message_fts: bool = True,
    settle_deferred_effects: bool = False,
) -> None:
    """Run post-ingest side effects through the canonical write-effects path."""

    def settle_effect(effect: WriteEffect, context: WriteEffectContext) -> None:
        effect.run(context)

    ArchiveWriteGateway(db_path).commit_write_sync(
        WriteOperation.INGEST,
        {
            "_connection": conn,
            "changed_session_ids": tuple(changed_session_ids),
            "repair_message_fts": repair_message_fts,
            **({"deferred_scheduler": settle_effect} if settle_deferred_effects else {}),
        },
    )


def _publish_marker_witnesses_before_index_commit(
    index_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    summary: _IngestBatchSummary,
) -> None:
    """Persist pending source bytes, then witness them in the open index txn."""
    from polylogue.storage.accepted_marker_inputs import (
        persist_pending_marker_input_sync,
        prepare_accepted_marker_input,
        retained_marker_input_sync,
    )

    requests: dict[str, PreparedAcceptedMarkerInput] = {}
    for raw_id, facts in summary.marker_request_facts_by_raw_id.items():
        if raw_id in summary.failed_raw_ids or raw_id in summary.publication_deferred_raw_ids:
            continue
        reused = summary.marker_batches_by_raw_id.get(raw_id)
        if reused is not None:
            requests[raw_id] = reused
            continue
        selected = summary.marker_sessions_by_raw_id.get(raw_id, [])
        selected_by_id = {str(session.get("session_id", "")): session for session in selected}
        dispositions = {
            str(session.get("session_id", "")): str(session.get("disposition", "no-op"))
            for session in summary.marker_session_dispositions_by_raw_id.get(raw_id, [])
        }
        request_sessions = summary.marker_request_sessions_by_raw_id.get(raw_id, [])
        carrier_sessions: list[dict[str, object]] = []
        for binding in request_sessions:
            session_id = str(binding.get("session_id", ""))
            session = dict(selected_by_id.get(session_id, binding))
            session["disposition"] = dispositions.get(session_id, "no-op")
            session.setdefault("candidates", [])
            carrier_sessions.append(session)
        # Defensive fallback for adapters that produced a write entry without
        # its outcome frame. Preserve every prepared carrier in that case.
        request_ids = {str(session.get("session_id", "")) for session in carrier_sessions}
        carrier_sessions.extend(
            session for session in selected if str(session.get("session_id", "")) not in request_ids
        )
        requests[raw_id] = prepare_accepted_marker_input(
            raw_id,
            carrier_sessions,
            request_facts=facts,
            request_sessions=request_sessions,
        )
    if not requests:
        return
    index_filename = str(index_conn.execute("PRAGMA database_list").fetchone()[2])
    index_stat = Path(index_filename).stat()
    incarnation = index_conn.execute(
        "SELECT incarnation_id, device, inode FROM ingest_index_incarnation WHERE singleton = 1"
    ).fetchone()
    if incarnation is None or (int(incarnation[1]), int(incarnation[2])) != (index_stat.st_dev, index_stat.st_ino):
        raise AcceptedMarkerInputRefusedError("index incarnation changed during marker publication")
    incarnation_id = str(incarnation[0])

    source_path = archive_root / "source.db"
    source_states: dict[str, str] = {}
    retained_batches: dict[str, PreparedAcceptedMarkerInput] = {}
    retained_incarnations: dict[str, str | None] = {}
    with (
        closing(
            open_isolated_write_connection(
                source_path,
                purpose="pending accepted marker carrier",
                timeout=DB_TIMEOUT,
                archive_root=archive_root,
            )
        ) as source_conn,
        source_conn,
    ):
        source_conn.execute("BEGIN IMMEDIATE")
        for batch in requests.values():
            retained = retained_marker_input_sync(source_conn, batch.identity)
            if retained is None:
                source_states[batch.identity] = persist_pending_marker_input_sync(
                    source_conn, batch, expected_incarnation_id=incarnation_id
                )
                retained_batches[batch.identity] = batch
                retained_incarnations[batch.identity] = incarnation_id
            else:
                (
                    source_states[batch.identity],
                    retained_batches[batch.identity],
                    retained_incarnations[batch.identity],
                ) = retained
    for request in requests.values():
        batch = retained_batches[request.identity]
        retained_state = source_states[batch.identity]
        retained_incarnation = retained_incarnations[batch.identity]
        # Pending carriers belong to the index transaction they were prepared
        # for and cannot cross a physical index replacement. An accepted
        # carrier is different: its immutable source bytes and sequence remain
        # authoritative across rebuilds. Recompute the complete request and
        # require byte equality before publishing a witness for this
        # incarnation. A witness proves that the retained carrier was
        # published previously; it cannot authorize a newly prepared
        # interpretation after the session writer has changed index state.
        if retained_state != "accepted" and retained_incarnation != incarnation_id:
            raise AcceptedMarkerInputRefusedError("pending marker carrier belongs to a replaced index incarnation")
        if retained_state == "accepted" and retained_incarnation is None:
            raise AcceptedMarkerInputRefusedError("accepted marker carrier has no recorded index incarnation")
        if batch.payload != request.payload and retained_state != "pending-new":
            raise AcceptedMarkerInputRefusedError(
                "retry interpretation differs from the immutable retained marker carrier"
            )
        value = json.loads(batch.payload)
        carrier_sessions = value["sessions"]
        witness_dispositions = [
            {
                "session_id": str(session.get("session_id", "")),
                "disposition": str(session.get("disposition", "no-op")),
            }
            for session in carrier_sessions
        ]
        encoded = json.dumps(witness_dispositions, sort_keys=True, separators=(",", ":"))
        prior = index_conn.execute(
            "SELECT carrier_digest, dispositions_json, incarnation_id FROM ingest_marker_witnesses WHERE request_key = ?",
            (batch.identity,),
        ).fetchone()
        if prior is None and batch.payload != request.payload:
            raise AcceptedMarkerInputRefusedError("retained marker carrier has no matching index publication witness")
        if prior is None and retained_state not in ("pending-new", "pending", "pending-existing", "accepted"):
            raise AcceptedMarkerInputRefusedError(
                f"retained marker carrier state {retained_state!r} has no matching index publication witness"
            )
        expected = (batch.payload_sha256, encoded, incarnation_id)
        if prior is not None and tuple(prior) != expected:
            raise AcceptedMarkerInputRefusedError("index marker witness conflicts with pending carrier")
        index_conn.execute(
            "INSERT OR IGNORE INTO ingest_marker_witnesses(request_key, carrier_digest, dispositions_json, incarnation_id) "
            "VALUES (?, ?, ?, ?)",
            (batch.identity, *expected),
        )
        summary.marker_batches_by_raw_id[batch.raw_id] = batch


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


_DRIVE_REVISION_COLUMNS = (
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
_DRIVE_COHORT_MAX_ROWS = 1000
_DRIVE_COHORT_MAX_BYTES = 128 * 1024 * 1024


def _source_snapshot(
    conn: sqlite3.Connection, table: str, predicate: str, parameters: tuple[str, ...]
) -> _SourceSnapshot:
    cursor = conn.execute(f"SELECT * FROM {table} WHERE {predicate} LIMIT 0", parameters)
    columns = tuple(column[0] for column in cursor.description)
    order = ", ".join(str(index + 1) for index in range(len(columns)))
    rows = conn.execute(
        f"SELECT * FROM {table} WHERE {predicate} ORDER BY {order} LIMIT ?",
        (*parameters, _DRIVE_COHORT_MAX_ROWS + 1),
    ).fetchall()
    return _SourceSnapshot(table, predicate, parameters, columns, tuple(tuple(row) for row in rows))


def _ingest_index_binding(db_path: Path) -> tuple[str, int, int]:
    resolved = db_path.resolve(strict=True)
    stat = resolved.stat()
    return str(resolved), stat.st_dev, stat.st_ino


def _ingest_policy_binding(archive_root: Path) -> tuple[object, ...]:
    with closing(open_readonly_connection(archive_root / "user.db", validate_schema=False)) as user:
        epoch = tuple(user.execute("SELECT epoch FROM query_unit_frame_state WHERE singleton=1").fetchone())
    with closing(open_readonly_connection(archive_root / "audit.db", validate_schema=False)) as audit:
        head = audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head WHERE singleton=1").fetchone()
    return (*epoch, *(tuple(head) if head is not None else (None, None)))


def _ingest_revision_heads(db_path: Path, keys: tuple[str, ...]) -> tuple[tuple[object, ...], ...]:
    if not keys:
        return ()
    marks = ",".join("?" for _ in keys)
    with closing(open_readonly_connection(db_path, validate_schema=False)) as index:
        return tuple(
            tuple(row)
            for row in index.execute(
                f"SELECT * FROM raw_revision_heads WHERE logical_source_key IN ({marks}) "
                f"OR session_id IN ({marks}) ORDER BY logical_source_key",
                (*keys, *keys),
            )
        )


def _prepare_ingest_unit_sync(
    raw_id: str,
    *,
    db_path: Path,
    archive_root: Path,
    validation_mode: str,
    publication_mode: PublicationMode,
    measure_ingest_result_size: bool,
) -> _PreparedIngestUnit | None:
    """Finish one parser result and Drive comparison with all readers closed."""
    from polylogue.storage.sqlite.queries.mappers import _row_to_raw_session
    from polylogue.storage.sqlite.write_lease import current_write_lease

    if current_write_lease() is not None:
        raise RuntimeError("ingest preparation requires a lease-free caller")
    index_binding = _ingest_index_binding(db_path)
    policy_binding = _ingest_policy_binding(archive_root)
    with closing(open_readonly_connection(archive_root / "source.db", validate_schema=False)) as source:
        source.row_factory = sqlite3.Row
        row = source.execute("SELECT * FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
        if row is None:
            return None
        input_row = tuple(row)
        record = _row_to_raw_session(row)
    request = _make_ingest_worker_request(
        archive_root_str=str(archive_root),
        blob_root_str=str(archive_root / "blob"),
        validation_mode=validation_mode,
        measure_ingest_result_size=measure_ingest_result_size,
    )
    _resolve_codex_sidecar_snapshots([record], archive_root=archive_root)
    results = list(_iter_ingest_results_sync([record], request=request, worker_count=1))
    if len(results) != 1:
        raise RuntimeError("one raw input must produce exactly one completed ingest result")
    result = results[0]
    keys = tuple(sorted({payload.session_id for payload in result.sessions}))
    marks = ",".join("?" for _ in keys) or "NULL"
    with closing(open_readonly_connection(archive_root / "source.db", validate_schema=False)) as source:
        source.execute("BEGIN")
        raw = _source_snapshot(source, "raw_sessions", f"raw_id=? OR logical_source_key IN ({marks})", (raw_id, *keys))
        members = _source_snapshot(
            source, "raw_session_memberships", f"raw_id=? OR logical_source_key IN ({marks})", (raw_id, *keys)
        )
        cohort_ids = tuple(sorted({raw_id, *(str(row[members.columns.index("raw_id")]) for row in members.rows)}))
        cohort_marks = ",".join("?" for _ in cohort_ids)
        census = _source_snapshot(source, "raw_membership_census", f"raw_id IN ({cohort_marks})", cohort_ids)
        artifacts = _source_snapshot(source, "raw_artifacts", "raw_id=?", (raw_id,))
    snapshots = (raw, members, census, artifacts)
    raw_id_position = raw.columns.index("raw_id")
    stale = not any(row[raw_id_position] == raw_id and row == input_row for row in raw.rows)
    stale |= any(len(snapshot.rows) > _DRIVE_COHORT_MAX_ROWS for snapshot in snapshots)
    stale |= (
        sum(int(cast(int, row[raw.columns.index("blob_size")])) for row in raw.rows if row[raw_id_position] != raw_id)
        > _DRIVE_COHORT_MAX_BYTES
    )
    plans: dict[str, RevisionReplayPlan | None] = {}
    updates: tuple[tuple[object, ...], ...] = ()
    if not stale:
        # This private, bounded scratch relation lets the existing Drive
        # governance code calculate its exact updates without an archive writer.
        # Only the revision column delta survives; no SQL or connection escapes.
        with closing(sqlite3.connect(":memory:")) as scratch:
            for snapshot in snapshots:
                columns = ",".join(_quote_identifier(column) for column in snapshot.columns)
                scratch.execute(f"CREATE TABLE {snapshot.table} ({columns})")
                values = ",".join("?" for _ in snapshot.columns)
                scratch.executemany(f"INSERT INTO {snapshot.table} VALUES ({values})", snapshot.rows)
            scratch.commit()
            publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
            # polylogue-ojjet: every payload here classifies the same cohort
            # snapshot, so one cache spans the whole prepared unit.
            prepare_cohort_cache = DriveRevisionCohortCache()
            for payload in result.sessions:
                plans[payload.session_id] = _bind_drive_revision_lineage(
                    payload.parsed_session,
                    raw_id=raw_id,
                    source_conn=scratch,
                    blob_publisher=publisher,
                    cohort_cache=prepare_cohort_cache,
                )
            revision_positions = tuple(raw.columns.index(column) for column in _DRIVE_REVISION_COLUMNS)
            before = {row[raw_id_position]: tuple(row[pos] for pos in revision_positions) for row in raw.rows}
            revision_columns = ",".join(_DRIVE_REVISION_COLUMNS)
            updates = tuple(
                tuple(row)
                for row in scratch.execute(f"SELECT {revision_columns}, raw_id FROM raw_sessions")
                if tuple(row[:-1]) != before[row[-1]]
            )
    return _PreparedIngestUnit(
        result,
        snapshots,
        index_binding,
        policy_binding,
        _ingest_revision_heads(db_path, keys),
        keys,
        plans,
        updates,
        validation_mode,
        publication_mode.value,
        stale,
    )


def _prepared_ingest_is_current(
    prepared: _PreparedIngestUnit,
    *,
    db_path: Path,
    archive_root: Path,
    validation_mode: str,
    publication_mode: PublicationMode,
) -> bool:
    if (
        prepared.stale
        or prepared.validation_mode != validation_mode
        or prepared.publication_mode != publication_mode.value
    ):
        return False
    if (
        _ingest_index_binding(db_path) != prepared.index_binding
        or _ingest_policy_binding(archive_root) != prepared.policy_binding
    ):
        return False
    if _ingest_revision_heads(db_path, prepared.logical_keys) != prepared.revision_heads:
        return False
    with closing(open_readonly_connection(archive_root / "source.db", validate_schema=False)) as source:
        source.execute("BEGIN")
        return all(
            _source_snapshot(source, snapshot.table, snapshot.predicate, snapshot.parameters) == snapshot
            for snapshot in prepared.source_snapshots
        )


def _publish_drive_revision_updates(prepared: _PreparedIngestUnit, archive_root: Path) -> None:
    if not prepared.drive_revision_updates:
        return
    assignments = ",".join(f"{column}=?" for column in _DRIVE_REVISION_COLUMNS)
    with (
        closing(
            open_isolated_write_connection(
                archive_root / "source.db",
                purpose="prepared Drive lineage",
                archive_root=archive_root,
            )
        ) as source,
        source,
    ):
        source.execute("BEGIN IMMEDIATE")
        source.executemany(f"UPDATE raw_sessions SET {assignments} WHERE raw_id=?", prepared.drive_revision_updates)


def _process_ingest_batch_sync(
    raw_artifacts: list[RawSessionRecord],
    *,
    db_path: Path,
    archive_root_str: str,
    blob_root_str: str,
    validation_mode: str,
    ingest_workers: int | None,
    measure_ingest_result_size: bool,
    publication_mode: PublicationMode = PublicationMode.OFF,
    force_write: bool = False,
    repair_message_fts: bool = True,
    heartbeat: IngestHeartbeat | None = None,
    progress: _WorkerProgress | None = None,
    ingest_result_chunk_size: int = 0,
    suspend_fts_triggers: bool = False,
    force_process_pool: bool = False,
    fresh_build: bool = False,
    prepared_unit: _PreparedIngestUnit | None = None,
    marker_acceptance_enabled: bool = False,
) -> _IngestBatchSummary:
    if progress is None:
        progress = _WorkerProgress()
    summary = _new_ingest_batch_summary(raw_artifacts, ingest_workers=ingest_workers)
    if marker_acceptance_enabled:
        for record in raw_artifacts:
            summary.marker_request_facts_by_raw_id[record.raw_id] = _marker_request_facts(
                record, validation_mode=validation_mode
            )
    worker_request = _make_ingest_worker_request(
        archive_root_str=archive_root_str,
        blob_root_str=blob_root_str,
        validation_mode=validation_mode,
        measure_ingest_result_size=measure_ingest_result_size,
    )
    t_start = time.perf_counter()
    archive_root = Path(archive_root_str)
    if prepared_unit is not None:
        if not _prepared_ingest_is_current(
            prepared_unit,
            db_path=db_path,
            archive_root=archive_root,
            validation_mode=validation_mode,
            publication_mode=publication_mode,
        ):
            discard_ingest_result_payload(prepared_unit.result)
            logger.info("Drive preparation became stale; leaving raw state for a fresh pass")
            return summary
        _publish_drive_revision_updates(prepared_unit, archive_root)
    _resolve_codex_sidecar_snapshots(raw_artifacts, archive_root=archive_root)
    primary_publication_service = (
        PublicationService(
            archive_root / "source.db",
            PublicationMode.PRIMARY,
            resolve_configured_transport(),
        )
        if publication_mode is PublicationMode.PRIMARY
        else None
    )
    setup_started = time.perf_counter()
    conn = _open_sync_connection(db_path, archive_root=archive_root)
    summary.setup_elapsed_s = time.perf_counter() - setup_started
    materialized_ids: set[str] = set()
    blob_publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    pending_attachment_receipts: list[tuple[str, bytes]] = []
    # polylogue-c737: read-only source.db handle used solely to consult
    # ``raw_session_memberships`` decisions during the write-precedence
    # check in ``_write_session`` (mirrors ArchiveStore's own
    # ``_ensure_source_conn`` read path). Opened once per batch rather than
    # per session write.
    source_db_path = archive_root / "source.db"
    source_conn: sqlite3.Connection | None = None
    if source_db_path.exists():
        # Membership/precedence checks only read source.db.  A read-only
        # profile avoids taking a writer lock while the index publication is
        # admitted, and makes an accidental mutation fail at SQLite level.
        source_conn = open_readonly_connection(
            source_db_path,
            timeout_class="background-read",
            validate_schema=False,
        )
    _observe_current_rss(summary)
    transaction_started = False
    try:
        if marker_acceptance_enabled:
            _ensure_ingest_index_incarnation(conn)
        if prepared_unit is not None:

            def begin_prepared_transaction() -> None:
                nonlocal transaction_started
                if not transaction_started:
                    conn.execute("BEGIN IMMEDIATE")
                    transaction_started = True

            try:
                _drain_ingest_result(
                    conn,
                    prepared_unit.result,
                    summary=summary,
                    materialized_ids=materialized_ids,
                    publication_mode=publication_mode,
                    primary_publication_service=primary_publication_service,
                    ensure_index_transaction=begin_prepared_transaction,
                    force_write=force_write,
                    blob_publisher=blob_publisher,
                    pending_attachment_receipts=pending_attachment_receipts,
                    source_conn=source_conn,
                    fresh_build=fresh_build,
                    drive_plans=prepared_unit.drive_plans,
                    marker_acceptance_enabled=marker_acceptance_enabled,
                )
            finally:
                discard_ingest_result_payload(prepared_unit.result)
        else:
            transaction_started = _consume_ingest_results(
                conn,
                raw_artifacts,
                worker_request=worker_request,
                summary=summary,
                materialized_ids=materialized_ids,
                publication_mode=publication_mode,
                primary_publication_service=primary_publication_service,
                force_write=force_write,
                heartbeat=heartbeat,
                progress=progress,
                ingest_result_chunk_size=ingest_result_chunk_size,
                suspend_fts_triggers=suspend_fts_triggers,
                mark_fts_stale_on_suspend=suspend_fts_triggers and not repair_message_fts,
                force_process_pool=force_process_pool,
                blob_publisher=blob_publisher,
                pending_attachment_receipts=pending_attachment_receipts,
                source_conn=source_conn,
                fresh_build=fresh_build,
                marker_acceptance_enabled=marker_acceptance_enabled,
            )
        _flush_ingest_results(
            conn,
            summary=summary,
        )
        if marker_acceptance_enabled and not transaction_started:
            # Empty parse batches have no session write to start the index
            # transaction, but their empty disposition still needs a witness.
            conn.execute("BEGIN IMMEDIATE")
            transaction_started = True
        if transaction_started:
            if suspend_fts_triggers:
                fk_violations = _foreign_key_violations_for_sessions(conn, materialized_ids)
                if fk_violations:
                    detail = _format_foreign_key_violations(fk_violations)
                    raise sqlite3.IntegrityError(f"foreign key check failed during bulk ingest: {detail}")
            fts_repair_ids = set(summary.fts_repair_session_ids)
            if marker_acceptance_enabled:
                partial_raw_ids = sorted(
                    raw_id for raw_id in summary.failed_raw_ids if summary.marker_sessions_by_raw_id.get(raw_id)
                )
                if partial_raw_ids:
                    raise AcceptedMarkerInputRefusedError(
                        "ordinary batch has a partially written raw marker input; refusing the full index transaction: "
                        + ", ".join(partial_raw_ids)
                    )
                for raw_id, expected_count in summary.expected_marker_session_counts.items():
                    if raw_id in summary.failed_raw_ids or expected_count == 0:
                        continue
                    actual_count = len(summary.marker_request_sessions_by_raw_id.get(raw_id, ()))
                    if actual_count != expected_count:
                        raise AcceptedMarkerInputRefusedError(
                            f"accepted raw revision {raw_id!r} has {actual_count} prepared marker sessions; "
                            f"expected {expected_count}"
                        )
                _publish_marker_witnesses_before_index_commit(
                    conn,
                    archive_root=archive_root,
                    summary=summary,
                )
            # Side effects run before releasing the connection so data and post-
            # write effects share one transaction. The previous arrangement
            # ran side effects in a `finally` block — they fired even after
            # a rollback (silently restoring triggers on top of nothing) and
            # ran AFTER the row commit, so a failure between commit and FTS
            # repair would leave the index drifted. See #1242.
            commit_started = time.perf_counter()
            _commit_sync_ingest_side_effects(
                conn,
                db_path=db_path,
                changed_session_ids=tuple(fts_repair_ids),
                repair_message_fts=repair_message_fts,
                **({"settle_deferred_effects": True} if prepared_unit is not None else {}),
            )
            if pending_attachment_receipts:
                # Receipt consumption is a real source-tier mutation and must
                # use the same archive-bound lease as the index publication.
                with (
                    closing(
                        open_isolated_write_connection(
                            archive_root / "source.db",
                            purpose="ingest blob publication receipt",
                            timeout=DB_TIMEOUT,
                            archive_root=archive_root,
                        )
                    ) as source_conn,
                    source_conn,
                ):
                    source_conn.execute("BEGIN IMMEDIATE")
                    for publication_id, blob_hash in pending_attachment_receipts:
                        consume_blob_publication_receipt(source_conn, publication_id, blob_hash)
            summary.commit_elapsed_s = time.perf_counter() - commit_started
            from polylogue.storage.sqlite.maintenance import maybe_optimize_sqlite

            optimize_observation = maybe_optimize_sqlite(conn, reason="ingest_batch_commit")
            if optimize_observation.error is not None:
                logger.warning(
                    "sqlite_optimize_failed",
                    reason=optimize_observation.reason,
                    analysis_limit=optimize_observation.analysis_limit,
                    error=optimize_observation.error,
                )
            if summary.schema_drift_observations:
                from polylogue.schemas.drift_sentinel_sampling import (
                    record_schema_drift_observations_to_ops_sync,
                )

                # Best-effort ops.db telemetry (polylogue-da1). Runs after
                # the index.db commit above so a drift-shaped record is
                # never held back by, or made to depend on, this write --
                # the sentinel only augments ingest, it never gates it.
                record_schema_drift_observations_to_ops_sync(
                    db_path,
                    summary.schema_drift_observations,
                    archive_root=archive_root,
                )
    except BaseException:
        # polylogue-qoa75: BaseException, not Exception. The dropped-trigger
        # window is exactly the window an operator Ctrl-C (KeyboardInterrupt)
        # or a cancelled task (asyncio.CancelledError) lands in, and neither is
        # an Exception; catching only Exception left index.db with the FTS
        # triggers absent and every later write silently unindexed.
        #
        # Roll back the row writes.  If a caller explicitly opted into
        # dropped-trigger bulk mode, restore triggers before propagating
        # so an interrupted batch does not leave the database in a drift
        # state.  Daemon live ingest leaves triggers active and therefore
        # has no dropped-trigger window to recover from here.
        with contextlib.suppress(Exception):
            conn.rollback()
        if suspend_fts_triggers:
            from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync

            # Restoration is NOT suppressed: a failure here is the condition
            # that silently unindexes every subsequent write, so it must be
            # surfaced (chained to the failure that made restoration
            # necessary), never swallowed.
            try:
                restore_fts_triggers_sync(conn)
                conn.commit()
            except Exception as restore_exc:
                logger.error(
                    "fts_trigger_restore_failed",
                    error=str(restore_exc),
                )
                raise FtsTriggerRestorationError(
                    "FTS triggers could not be restored after an interrupted bulk ingest; "
                    "index.db is unindexed for search until an explicit FTS rebuild"
                ) from restore_exc
        raise
    finally:
        blob_publisher.discard_pending()
        if suspend_fts_triggers:
            with contextlib.suppress(Exception):
                conn.execute("PRAGMA foreign_keys = ON")
        conn.close()
        if source_conn is not None:
            source_conn.close()
    summary.worker_progress_in_flight = len(progress.in_flight_raw_ids)
    summary.worker_progress_completed = progress.completed_raw_count
    summary.worker_progress_total = progress.total_raw_count
    summary.elapsed_s = time.perf_counter() - t_start
    return summary


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
    repair_message_fts: bool = True,
    ingest_result_chunk_size: int = 0,
    suspend_fts_triggers: bool = False,
    fresh_build: bool = False,
    prepared_unit: _PreparedIngestUnit | None = None,
) -> ParseBatchObservation | None:
    """Process a batch of raw records through the unified ingest pipeline.

    1. Submit all records to ProcessPool (decode + validate + parse + transform)
    2. Consume results via as_completed — write to DB as each worker finishes
    3. Defer session insight refresh to caller (done once after all batches)

    When *ingest_result_chunk_size* > 0 and *batch_ids* exceeds it, raw
    records are split into sub-batches to bound the memory held by parsed
    results in the process pool and drain loop.
    """
    import asyncio

    if service.execution is not None and prepared_unit is None:
        from polylogue.config import load_polylogue_config

        settings = load_polylogue_config()
        last_observation = None
        for raw_id in batch_ids:
            unit = await service.execution.prepare(
                partial(
                    _prepare_ingest_unit_sync,
                    raw_id,
                    db_path=backend.db_path,
                    archive_root=service.archive_root,
                    validation_mode=settings.schema_validation,
                    publication_mode=PublicationMode.from_string(settings.sinex_mode),
                    measure_ingest_result_size=service.measure_ingest_result_size,
                )
            )
            if unit is None:
                continue

            async def publish(raw_id: str = raw_id, unit: _PreparedIngestUnit = unit) -> ParseBatchObservation | None:
                return await process_ingest_batch(
                    service,
                    backend,
                    [raw_id],
                    result,
                    progress_callback,
                    force_write=force_write,
                    repair_message_fts=repair_message_fts,
                    fresh_build=fresh_build,
                    prepared_unit=unit,
                )

            try:
                last_observation = await service.execution.publish("ingest", publish)
            finally:
                discard_ingest_result_payload(unit.result)
        return last_observation

    raw_artifacts = await service.repository.get_raw_sessions_batch(batch_ids)
    if not raw_artifacts:
        return None

    archive_root_str = str(service.archive_root)
    blob_root_str = str(service.archive_root / "blob")
    batch_started = time.perf_counter()
    rss_start_mb = read_current_rss_mb()
    peak_rss_self_start_mb = read_peak_rss_self_mb()

    # Get validation mode from environment and Sinex authority mode from the
    # canonical config layer.  Off mode is passed through so the sync writer
    # performs no protocol encoding or outbox work.
    from polylogue.config import load_polylogue_config

    _resolved_settings = load_polylogue_config()
    validation_mode = _resolved_settings.schema_validation
    publication_mode = PublicationMode.from_string(_resolved_settings.sinex_mode)
    configured_source_backend = getattr(service.repository, "source_backend", None)
    if publication_mode is not PublicationMode.OFF and configured_source_backend is None:
        raise PublicationEncodingError(
            "mirror/primary acceptance requires the durable source-tier backend; refusing before index publication"
        )

    sync_kwargs: dict[str, object] = {
        "db_path": backend.db_path,
        "archive_root_str": archive_root_str,
        "blob_root_str": blob_root_str,
        "validation_mode": validation_mode,
        "ingest_workers": service.ingest_workers,
        "measure_ingest_result_size": service.measure_ingest_result_size,
        "publication_mode": publication_mode,
        "force_write": force_write,
        "repair_message_fts": repair_message_fts,
        "ingest_result_chunk_size": ingest_result_chunk_size,
        "suspend_fts_triggers": suspend_fts_triggers,
    }
    # OFF without a source backend is a supported index-only mode. Marker
    # acceptance is enabled only when its durable owner is available.
    if configured_source_backend is not None:
        sync_kwargs["marker_acceptance_enabled"] = True
    if fresh_build:
        sync_kwargs["fresh_build"] = True
    if prepared_unit is not None:
        sync_kwargs["prepared_unit"] = prepared_unit
    if service.execution is None:
        batch_summary = await asyncio.to_thread(
            cast(Callable[..., _IngestBatchSummary], _process_ingest_batch_sync),
            raw_artifacts,
            **sync_kwargs,
        )
    else:
        batch_summary = await service.execution.publish_sync(
            "index",
            lambda: cast(Callable[..., _IngestBatchSummary], _process_ingest_batch_sync)(raw_artifacts, **sync_kwargs),
        )
    heavy_batch = (
        batch_summary.total_blob_mb >= INGEST_RELEASE_BLOB_MB_THRESHOLD
        or batch_summary.total_msgs >= INGEST_RELEASE_MESSAGE_THRESHOLD
    )
    raw_artifacts.clear()

    apply_ingest_batch_summary(result, batch_summary)
    progressed = progressed_raw_count(batch_summary)
    if progress_callback and progressed:
        progress_callback(progressed)

    if batch_summary.elapsed_s > 0.0:
        logger.info(
            "ingest_batch",
            elapsed_s=round(batch_summary.elapsed_s, 2),
            records=batch_summary.raw_record_count,
            blob_mb=round(batch_summary.total_blob_mb, 1),
            sessions=batch_summary.total_convos,
            messages=batch_summary.total_msgs,
            workers=batch_summary.worker_count,
            changed=len(batch_summary.changed_session_ids),
            result_mb=round(batch_summary.total_result_bytes / (1024 * 1024), 1),
            max_result_mb=round(batch_summary.max_result_bytes / (1024 * 1024), 1),
            max_result_raw_id=batch_summary.max_result_raw_id,
            max_current_rss_mb=batch_summary.max_current_rss_mb,
            write_s=round(batch_summary.write_elapsed_s, 2),
            max_write_s=round(batch_summary.max_write_elapsed_s, 2),
            commit_s=round(batch_summary.commit_elapsed_s, 2),
            drain_s=round(batch_summary.drain_elapsed_s, 2),
            flush_s=round(batch_summary.flush_elapsed_s, 2),
            wait_s=round(batch_summary.result_wait_s, 2),
            setup_s=round(batch_summary.setup_elapsed_s, 2),
            tear_s=round(batch_summary.teardown_elapsed_s, 2),
            # polylogue-rujy AC4: content-addressed tool-result sidecar blob
            # writes this batch actually added vs. deduplicated (0 unless the
            # batch touched Claude Code sessions with acquired sidecars).
            sidecar_blob_new_mb=round(batch_summary.counts["sidecar_blob_bytes_new"] / (1024 * 1024), 2),
            sidecar_blob_dedup_mb=round(batch_summary.counts["sidecar_blob_bytes_dedup"] / (1024 * 1024), 2),
            sidecar_blobs_written=batch_summary.counts["sidecar_blobs_written"],
        )

    succeeded_raw_ids = successful_raw_ids(batch_summary)
    raw_state_update_elapsed_s = await _persist_batch_raw_state_updates(
        service,
        backend,
        outcomes=batch_summary.outcomes,
        succeeded_raw_ids=succeeded_raw_ids,
        skipped_raw_ids=batch_summary.skipped_raw_ids,
        failed_raw_ids=batch_summary.failed_raw_ids,
        validation_mode=validation_mode,
        publication_mode=publication_mode,
        publication_payloads_by_raw_id=batch_summary.publication_payloads_by_raw_id,
        marker_sessions_by_raw_id=(
            batch_summary.marker_sessions_by_raw_id if configured_source_backend is not None else None
        ),
        marker_request_facts_by_raw_id=(
            batch_summary.marker_request_facts_by_raw_id if configured_source_backend is not None else None
        ),
        marker_request_sessions_by_raw_id=(
            batch_summary.marker_request_sessions_by_raw_id if configured_source_backend is not None else None
        ),
        marker_batches_by_raw_id=(
            batch_summary.marker_batches_by_raw_id if configured_source_backend is not None else None
        ),
    )
    batch_summary.publication_payloads_by_raw_id.clear()
    batch_summary.publication_payload_bytes = 0

    elapsed_s = time.perf_counter() - batch_started
    rss_end_mb = read_current_rss_mb()
    peak_rss_self_end_mb = read_peak_rss_self_mb()
    peak_rss_children_mb = read_peak_rss_children_mb()
    observation = _build_parse_batch_observation(
        batch_summary=batch_summary,
        elapsed_s=elapsed_s,
        raw_state_update_elapsed_s=raw_state_update_elapsed_s,
        rss_start_mb=rss_start_mb,
        rss_end_mb=rss_end_mb,
        peak_rss_self_start_mb=peak_rss_self_start_mb,
        peak_rss_self_end_mb=peak_rss_self_end_mb,
        peak_rss_children_mb=peak_rss_children_mb,
    )
    if heavy_batch:
        del batch_summary
    return observation


def _successful_raw_state_update(
    *,
    outcome: _RawIngestOutcome | None,
    parsed_at: str,
    validation_mode: str,
) -> RawSessionStateUpdate:
    if outcome is None:
        return RawSessionStateUpdate(
            parsed_at=parsed_at,
            parse_error=None,
        )
    return RawSessionStateUpdate(
        parsed_at=parsed_at,
        parse_error=None,
        payload_provider=outcome.payload_provider,
        validation_status=outcome.validation_status,
        validation_error=outcome.validation_error,
        validation_mode=validation_mode,
    )


def _skipped_raw_state_update(
    *,
    outcome: _RawIngestOutcome | None,
    parsed_at: str,
    validation_mode: str,
) -> RawSessionStateUpdate:
    return RawSessionStateUpdate(
        parsed_at=parsed_at,
        parse_error=None,
        payload_provider=outcome.payload_provider if outcome is not None else None,
        validation_status="skipped",
        validation_error="parsed raw payload produced no new materialized sessions",
        validation_mode=validation_mode,
    )


def _failed_raw_state_update(
    *,
    outcome: _RawIngestOutcome | None,
    error: str,
    validation_mode: str,
) -> RawSessionStateUpdate:
    if outcome is None:
        return RawSessionStateUpdate(
            parse_error=error,
            detection_warnings=error[:500] if error else None,
        )
    diagnostic = outcome.diagnostic or outcome.parse_error
    return RawSessionStateUpdate(
        parse_error=outcome.parse_error,
        detection_warnings=diagnostic[:500] if diagnostic else None,
        payload_provider=outcome.payload_provider,
        validation_status=outcome.validation_status,
        validation_error=outcome.validation_error or error,
        validation_mode=validation_mode,
    )


def _raw_failure_evidence_kind(outcome: _RawIngestOutcome | None) -> RawFailureEvidenceKind | None:
    """Map terminal worker input outcomes to closed source-tier carriers."""
    if outcome is None:
        return None
    try:
        outcome_code = IngestOutcome.from_string(outcome.outcome_code)
    except ValueError:
        return None
    return {
        IngestOutcome.CORRUPT_INPUT: RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        IngestOutcome.UNSUPPORTED_SHAPE: RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE,
    }.get(outcome_code)


async def _persist_batch_raw_state_updates(
    service: _ParsingServiceRawStateLike,
    backend: _BulkConnectionBackendLike,
    *,
    outcomes: dict[str, _RawIngestOutcome],
    succeeded_raw_ids: set[str],
    skipped_raw_ids: set[str],
    failed_raw_ids: dict[str, str],
    validation_mode: str,
    publication_mode: PublicationMode = PublicationMode.OFF,
    publication_payloads_by_raw_id: Mapping[str, Sequence[PublicationPayload]] | None = None,
    marker_sessions_by_raw_id: Mapping[str, Sequence[dict[str, object]]] | None = None,
    marker_request_facts_by_raw_id: Mapping[str, dict[str, object]] | None = None,
    marker_request_sessions_by_raw_id: Mapping[str, Sequence[dict[str, object]]] | None = None,
    marker_batches_by_raw_id: Mapping[str, PreparedAcceptedMarkerInput] | None = None,
) -> float:
    now_iso = datetime.now(timezone.utc).isoformat()
    raw_state_update_started = time.perf_counter()
    source_backend = service.repository.source_backend
    if publication_mode is not PublicationMode.OFF and source_backend is None:
        raise PublicationEncodingError(
            "mirror/primary acceptance requires the durable source-tier backend; "
            "refusing to stage a publication obligation through index.db"
        )

    now_ms = 0

    if marker_sessions_by_raw_id is not None and source_backend is None:
        raise AcceptedMarkerInputRefusedError("accepted marker inputs require the durable source-tier backend")

    async def retain_marker_input(raw_state_conn: AsyncSqlConnection, rid: str) -> None:
        if marker_sessions_by_raw_id is None:
            return
        sessions = marker_sessions_by_raw_id.get(rid, ())
        request_sessions = (marker_request_sessions_by_raw_id or {}).get(rid, ())
        if outcomes.get(rid) is not None and outcomes[rid].had_sessions and not request_sessions:
            raise AcceptedMarkerInputRefusedError(f"accepted raw revision {rid!r} has no prepared marker coverage")
        facts = (marker_request_facts_by_raw_id or {}).get(rid, {})
        batch = (marker_batches_by_raw_id or {}).get(rid)
        if batch is None:
            batch = prepare_accepted_marker_input(rid, sessions, request_facts=facts, request_sessions=request_sessions)
        assert isinstance(batch, PreparedAcceptedMarkerInput)
        await finalize_pending_accepted_marker_input(raw_state_conn, batch)

    async def stage_accepted_payloads(
        raw_state_conn: AsyncSqlConnection,
        rid: str,
        *,
        required: bool,
    ) -> None:
        if publication_mode is PublicationMode.OFF:
            return
        payloads = tuple((publication_payloads_by_raw_id or {}).get(rid, ()))
        if required and not payloads:
            raise PublicationEncodingError(f"accepted raw revision {rid!r} has no reconciled Sinex publication payload")
        for payload in payloads:
            await stage_payload_async(
                raw_state_conn,
                payload=payload,
                mode=publication_mode,
                now_ms=now_ms,
            )

    async with AsyncExitStack() as stack:
        raw_state_backend = source_backend if source_backend is not None else backend
        # bulk_connection() owns BEGIN IMMEDIATE but deliberately yields None.
        # connection() then reuses that backend-local active connection.
        await stack.enter_async_context(raw_state_backend.bulk_connection())
        now_ms = int(time.time() * 1000)
        raw_state_conn: AsyncSqlConnection | None = None
        if publication_mode is not PublicationMode.OFF or marker_sessions_by_raw_id is not None:
            assert source_backend is not None
            raw_state_conn = cast(
                AsyncSqlConnection,
                await stack.enter_async_context(source_backend.connection()),
            )
        for rid in succeeded_raw_ids:
            if rid in skipped_raw_ids:
                continue
            if source_backend is not None:
                await source_backend.supersede_deferred_cas_evidence(rid)
            await service.repository.update_raw_state(
                rid,
                state=_successful_raw_state_update(
                    outcome=outcomes.get(rid),
                    parsed_at=now_iso,
                    validation_mode=validation_mode,
                ),
            )
            # update_raw_state reuses source_backend's active bulk connection;
            # staging on the yielded connection therefore shares its BEGIN
            # IMMEDIATE/commit/rollback boundary.
            if raw_state_conn is not None:
                await retain_marker_input(raw_state_conn, rid)
                await stage_accepted_payloads(raw_state_conn, rid, required=True)
        for rid in skipped_raw_ids:
            if rid in failed_raw_ids:
                continue
            if source_backend is not None:
                await source_backend.supersede_deferred_cas_evidence(rid)
            await service.repository.update_raw_state(
                rid,
                state=_skipped_raw_state_update(
                    outcome=outcomes.get(rid),
                    parsed_at=now_iso,
                    validation_mode=validation_mode,
                ),
            )
            # Empty parse results have no payload.  Content-identical duplicate
            # revisions do, and are restaged idempotently in this transaction.
            if raw_state_conn is not None:
                await retain_marker_input(raw_state_conn, rid)
                await stage_accepted_payloads(raw_state_conn, rid, required=False)
        for rid, error in failed_raw_ids.items():
            await service.repository.update_raw_state(
                rid,
                state=_failed_raw_state_update(
                    outcome=outcomes.get(rid),
                    error=error,
                    validation_mode=validation_mode,
                ),
            )
            outcome = outcomes.get(rid)
            evidence_kind = _raw_failure_evidence_kind(outcome)
            if source_backend is not None:
                if outcome is not None and evidence_kind is not None:
                    await source_backend.save_raw_failure_evidence(
                        rid,
                        artifact_kind=evidence_kind.value,
                        support_status=evidence_kind.support_status.value,
                        outcome_code=outcome.outcome_code,
                        retryable=outcome.retryable,
                        evidence_ref=outcome.evidence_ref,
                        remediation=outcome.remediation,
                        diagnostic=outcome.diagnostic,
                    )
                elif outcome is not None:
                    await source_backend.retire_raw_failure_evidence(rid)
    return time.perf_counter() - raw_state_update_started


async def repair_message_fts_bulk(
    backend: _ConnectionBackendLike,
    changed_session_ids: Sequence[str],
) -> None:
    """Repair message FTS once after a multi-batch ingest pass."""
    session_ids = tuple(dict.fromkeys(changed_session_ids))
    if not session_ids:
        return

    from polylogue.storage.fts.fts_lifecycle import repair_fts_index_async

    async with backend.connection() as conn:
        await repair_fts_index_async(conn, session_ids)
        await conn.commit()


__all__ = [
    "_INGEST_RESULT_CHUNK_SIZE",
    "process_ingest_batch",
    "repair_message_fts_bulk",
]

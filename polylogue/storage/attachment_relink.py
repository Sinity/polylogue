"""Relink orphaned ``attachments`` rows by re-parsing raw source data.

polylogue-w06b: ``_write_attachments`` (``storage/sqlite/archive_tiers/write.py``)
was fixed (#3514) so a full-replace re-ingest that drops an attachment's owning
message now sweeps the ref-less ``attachments`` row at write time instead of
leaving it permanently unreachable. That fix is going-forward only: it cannot
help the 1,858 rows the live archive already had orphaned before it landed
(1,783 ``acquired``, 75 ``unfetched`` -- see the bead's forensics).

``attachments`` carries no ``session_id``/``message_id`` column -- the only
link back to a session is ``attachment_refs``, and ``attachment_id`` itself is
a pure content-identity hash of the attachment's own metadata (see
``write.py:_attachment_id``), independent of which session/message produced
it. So the *only* durable evidence that can reconstruct "which message owned
this orphan" is the original raw bytes still held in ``source.db``'s
``raw_sessions`` -- exactly the rebuildable-tier philosophy already documented
for ``index.db`` (see ``docs/architecture.md`` schema-regimes section): if the
raw is still there, re-parsing it can regenerate the same attachment_id and
message_id the write path would have produced, and confirm the owning message
is still present in the current index.

This module never guesses. A given orphan is only reported ``eligible`` for
relink when a raw session's re-parsed content reproduces the *exact* same
``attachment_id`` (same provider ids/path/name/mime/size -- see
``_attachment_id``) AND the resolved ``message_id`` is a real row in the
current ``messages`` table. Every other outcome -- no raw reproduces the
identity at all, or a raw reproduces it but the owning message no longer
exists in the current index -- is reported as ``ineligible`` with an exact
reason, mirroring the read-only plan/execute split used by
``raw_retention.plan_stale_supersession_reissue`` /
``reissue_stale_supersession_receipts``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload, ingest_record
from polylogue.storage.runtime.raw.records import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.write import (
    _attachment_caption,
    _attachment_id,
    _attachment_message_id_maps,
    _attachment_position,
    _attachment_source_url,
)
from polylogue.storage.sqlite.queries.mappers_archive import _row_to_raw_session

logger = get_logger(__name__)

# Raw parsing is real work (decode + provider dispatch + normalize); bound how
# many raw_sessions rows one plan pass inspects unless the caller overrides it.
DEFAULT_RAW_ROW_LIMIT = 5_000

_NO_RAW_MATCH_REASON = "no raw session in source.db reproduces this attachment's identity"
_MESSAGE_MISSING_REASON = (
    "matched raw content but the owning message is no longer present in the current index "
    "(session likely re-ingested without it since)"
)


@dataclass(frozen=True, slots=True)
class RelinkableAttachment:
    """One orphaned attachment a re-parsed raw proves is still recoverable."""

    attachment_id: str
    session_id: str
    message_id: str
    position: int
    upload_origin: str | None
    source_url: str | None
    caption: str | None
    raw_id: str


@dataclass(frozen=True, slots=True)
class UnrecoverableAttachment:
    attachment_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class OrphanedAttachmentRelinkPlan:
    """Read-only projection: what a relink pass would do against source.db.

    ``eligible`` is the *complete* (never sampled) set a relink must write a
    fresh ``attachment_refs`` row for -- callers writing refs must use every
    item here, not just ``unrecoverable_samples``. ``raw_rows_scanned`` records
    how much of ``raw_sessions`` was actually inspected (bounded by
    ``raw_row_limit``), so a caller can tell an exhaustive plan apart from a
    partial one that stopped early.
    """

    orphan_count: int
    eligible: tuple[RelinkableAttachment, ...]
    unrecoverable_reason_counts: Mapping[str, int]
    unrecoverable_samples: tuple[UnrecoverableAttachment, ...]
    raw_rows_scanned: int
    raw_rows_total: int


RawSessionParser = Callable[[RawSessionRecord], IngestRecordResult]


def _default_raw_session_parser(archive_root: Path, blob_root: Path | None) -> RawSessionParser:
    archive_root_str = str(archive_root)
    blob_root_str = str(blob_root) if blob_root is not None else None

    def _parse(raw_record: RawSessionRecord) -> IngestRecordResult:
        return ingest_record(raw_record, archive_root_str, "advisory", blob_root_str=blob_root_str)

    return _parse


def _read_orphaned_attachment_ids(index_conn: sqlite3.Connection) -> list[str]:
    rows = index_conn.execute(
        """
        SELECT attachment_id FROM attachments a
        WHERE NOT EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id = a.attachment_id)
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _message_exists(index_conn: sqlite3.Connection, message_id: str) -> bool:
    return index_conn.execute("SELECT 1 FROM messages WHERE message_id = ?", (message_id,)).fetchone() is not None


def _iter_raw_session_rows(source_conn: sqlite3.Connection, *, raw_row_limit: int | None) -> list[sqlite3.Row]:
    original_row_factory = source_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM raw_sessions ORDER BY acquired_at_ms DESC"
        if raw_row_limit is not None:
            query += f" LIMIT {int(raw_row_limit)}"
        return source_conn.execute(query).fetchall()
    finally:
        source_conn.row_factory = original_row_factory


def plan_orphaned_attachment_relink(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    blob_root: Path | None = None,
    sample_limit: int = 20,
    raw_row_limit: int | None = DEFAULT_RAW_ROW_LIMIT,
    raw_session_parser: RawSessionParser | None = None,
) -> OrphanedAttachmentRelinkPlan:
    """Read-only: find orphaned attachments a raw re-parse can prove recoverable.

    For each ``raw_sessions`` row (newest first, bounded by ``raw_row_limit``),
    re-parses it via ``ingest_record`` (the same decode+validate+parse+
    transform entry point the live ingest worker uses -- see module docstring)
    and checks every attachment it reproduces against the still-pending orphan
    set. A match is only accepted ``eligible`` when the resolved message_id is
    a real row in the CURRENT ``messages`` table; a content match whose message
    is gone is recorded ``ineligible`` (unless a later raw also reproduces the
    same attachment_id with a message that does still exist -- attachment_id
    is session-independent, so the same physical attachment can legitimately
    appear in more than one raw session).

    Never mutates ``index_conn``/``source_conn``. ``raw_session_parser`` is an
    injection point for tests; production callers should leave it ``None`` to
    use the real ``ingest_record`` parser.
    """
    orphan_ids = _read_orphaned_attachment_ids(index_conn)
    orphan_count = len(orphan_ids)
    if orphan_count == 0:
        return OrphanedAttachmentRelinkPlan(
            orphan_count=0,
            eligible=(),
            unrecoverable_reason_counts={},
            unrecoverable_samples=(),
            raw_rows_scanned=0,
            raw_rows_total=0,
        )

    pending: set[str] = set(orphan_ids)
    recovered: dict[str, RelinkableAttachment] = {}
    ineligible_reasons: dict[str, str] = {}

    parser = raw_session_parser or _default_raw_session_parser(archive_root, blob_root)

    total_rows_row = source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()
    raw_rows_total = int(total_rows_row[0]) if total_rows_row is not None else 0

    rows = _iter_raw_session_rows(source_conn, raw_row_limit=raw_row_limit)
    scanned = 0
    for row in rows:
        if not pending:
            break
        scanned += 1
        try:
            raw_record = _row_to_raw_session(row)
        except Exception as exc:  # pragma: no cover - defensive, malformed row shape
            logger.warning("attachment relink: could not build RawSessionRecord for raw_id=%s: %s", row["raw_id"], exc)
            continue
        try:
            result = parser(raw_record)
        except Exception as exc:
            logger.warning("attachment relink: reparse failed for raw_id=%s: %s", raw_record.raw_id, exc)
            continue
        if result.error is not None:
            continue
        for payload in result.sessions:
            _match_session_payload(payload, pending, recovered, ineligible_reasons, index_conn, raw_record.raw_id)

    reason_counts: dict[str, int] = {}
    samples: list[UnrecoverableAttachment] = []
    for attachment_id in orphan_ids:
        if attachment_id in recovered:
            continue
        reason = ineligible_reasons.get(attachment_id, _NO_RAW_MATCH_REASON)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if len(samples) < sample_limit:
            samples.append(UnrecoverableAttachment(attachment_id=attachment_id, reason=reason))

    return OrphanedAttachmentRelinkPlan(
        orphan_count=orphan_count,
        eligible=tuple(recovered[attachment_id] for attachment_id in orphan_ids if attachment_id in recovered),
        unrecoverable_reason_counts=reason_counts,
        unrecoverable_samples=tuple(samples),
        raw_rows_scanned=scanned,
        raw_rows_total=raw_rows_total,
    )


def _match_session_payload(
    payload: SessionWritePayload,
    pending: set[str],
    recovered: dict[str, RelinkableAttachment],
    ineligible_reasons: dict[str, str],
    index_conn: sqlite3.Connection,
    raw_id: str,
) -> None:
    if not pending:
        return
    session_id = payload.session_id
    messages = payload.parsed_session.messages
    by_native_message_id, by_message_position = _attachment_message_id_maps(session_id, messages)
    # Attachments are session-level (``ParsedSession.attachments``), each
    # linked to its owning message via ``message_provider_id`` -- mirroring
    # exactly how ``_write_attachments`` consumes them (write.py:_attachment_message_id_maps).
    for attachment in payload.parsed_session.attachments:
        attachment_id = _attachment_id(session_id, attachment)
        if attachment_id not in pending:
            continue
        message_id = (
            by_native_message_id.get(attachment.message_provider_id) if attachment.message_provider_id else None
        )
        if message_id is None and attachment.message_position is not None:
            message_id = by_message_position.get(attachment.message_position)
        if message_id is None:
            ineligible_reasons.setdefault(attachment_id, _NO_RAW_MATCH_REASON)
            continue
        if not _message_exists(index_conn, message_id):
            ineligible_reasons.setdefault(attachment_id, _MESSAGE_MISSING_REASON)
            continue
        recovered[attachment_id] = RelinkableAttachment(
            attachment_id=attachment_id,
            session_id=session_id,
            message_id=message_id,
            position=_attachment_position(attachment),
            upload_origin=attachment.upload_origin,
            source_url=_attachment_source_url(attachment),
            caption=_attachment_caption(attachment),
            raw_id=raw_id,
        )
        pending.discard(attachment_id)


@dataclass(frozen=True, slots=True)
class OrphanedAttachmentRelinkResult:
    orphan_count: int
    eligible_count: int
    relinked_count: int
    unrecoverable_reason_counts: Mapping[str, int]
    unrecoverable_samples: tuple[UnrecoverableAttachment, ...]
    errors: tuple[str, ...] = ()


def relink_orphaned_attachments(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    blob_root: Path | None = None,
    dry_run: bool = True,
    sample_limit: int = 20,
    raw_row_limit: int | None = DEFAULT_RAW_ROW_LIMIT,
    raw_session_parser: RawSessionParser | None = None,
) -> OrphanedAttachmentRelinkResult:
    """Relink orphaned attachments proven recoverable by :func:`plan_orphaned_attachment_relink`.

    ``dry_run=True`` (the default) performs no writes; it only runs the plan
    and reports what would happen. When ``dry_run=False``, writes ONE
    ``attachment_refs`` row per eligible orphan (mirroring the exact INSERT
    ``_write_attachments`` uses) and refreshes that attachment's ``ref_count``
    -- it never touches an attachment this plan did not mark eligible, and it
    is safe to call repeatedly (``INSERT OR REPLACE`` + a ref_count recompute,
    both idempotent). ``index_conn`` must be a writable index-tier connection
    when ``dry_run=False``; this function does not commit -- the caller owns
    transaction/commit scope.
    """
    plan = plan_orphaned_attachment_relink(
        index_conn,
        source_conn,
        archive_root=archive_root,
        blob_root=blob_root,
        sample_limit=sample_limit,
        raw_row_limit=raw_row_limit,
        raw_session_parser=raw_session_parser,
    )
    relinked = 0
    errors: list[str] = []
    if not dry_run:
        for item in plan.eligible:
            try:
                index_conn.execute(
                    """
                    INSERT OR REPLACE INTO attachment_refs (
                        attachment_id, session_id, message_id, position, upload_origin, source_url, caption
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.attachment_id,
                        item.session_id,
                        item.message_id,
                        item.position,
                        item.upload_origin,
                        item.source_url,
                        item.caption,
                    ),
                )
                index_conn.execute(
                    """
                    UPDATE attachments
                    SET ref_count = (
                        SELECT COUNT(*) FROM attachment_refs WHERE attachment_refs.attachment_id = attachments.attachment_id
                    )
                    WHERE attachment_id = ?
                    """,
                    (item.attachment_id,),
                )
                relinked += 1
            except sqlite3.Error as exc:
                logger.warning(
                    "attachment relink: failed to write ref for attachment_id=%s: %s", item.attachment_id, exc
                )
                errors.append(f"{item.attachment_id}: {exc}")

    return OrphanedAttachmentRelinkResult(
        orphan_count=plan.orphan_count,
        eligible_count=len(plan.eligible),
        relinked_count=relinked,
        unrecoverable_reason_counts=plan.unrecoverable_reason_counts,
        unrecoverable_samples=plan.unrecoverable_samples,
        errors=tuple(errors),
    )


__all__ = [
    "OrphanedAttachmentRelinkPlan",
    "OrphanedAttachmentRelinkResult",
    "RelinkableAttachment",
    "UnrecoverableAttachment",
    "plan_orphaned_attachment_relink",
    "relink_orphaned_attachments",
]

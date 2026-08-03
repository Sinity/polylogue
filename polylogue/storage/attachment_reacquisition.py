"""Backfill acquisition for historically-unfetched attachments (polylogue-pfdf).

79% of ``attachments`` rows (7,376 of 9,289 measured 2026-07-31) sit at
``acquisition_status='unfetched'`` -- bytes were never stored. #2469 fixed the
real producer (``_acquire_attachment_blob`` / ``ParsedAttachment.inline_bytes``
propagation) going forward, and a later fix (60d93b618, "acquire Claude
extracted payloads") taught the shared ``attachment_from_meta`` parser to pull
``meta["extracted_content"]`` into ``inline_bytes``. Neither fix is
retroactive: an attachment ingested *before* the relevant parser fix landed
still has no ``blob_hash``, even though its owning session's raw payload
(durable in ``source.db``'s ``raw_sessions``) may already contain everything
today's parser needs to extract it.

This module is the read-only classifier + operator-gated actuator pair the
bead's AC calls for: "backfill pass over unfetched attachments where the
source payload still contains the referenced bytes; unrecoverable ones marked
distinctly from 'unfetched'". It follows the same architecture as
:mod:`polylogue.storage.attachment_relink` (which recovers orphaned
*attachment_refs* linkage the same way) and the raw-side actuators in
:mod:`polylogue.storage.blob_integrity` / ``raw_live_source_reconciliation_apply``.

## Two static, zero-reparse-needed classes

Some unfetched attachments are provably unrecoverable *without* touching a
single raw byte, because the parser that created them documents it as
structural:

* ``sandbox_file`` (ChatGPT Code Interpreter deliverables): ``chatgpt.py``
  records these with ``source_url="sandbox:/mnt/data/..."`` and its own
  comment states plainly "the export/capture carries no bytes ... there is
  nothing local to fetch" -- the sandbox container is gone by the time the
  export exists. Detected here purely from ``attachment_refs.source_url``,
  no reparse needed.

Everything else is genuinely undetermined until checked against a re-parse of
its raw evidence.

## The reparse-based recoverable class

For every other unfetched attachment, the *only* sound way to know "does the
source still have the bytes" is to re-parse the raw payload still held in
``source.db`` with **today's** parser code (mirroring
``attachment_relink.plan_orphaned_attachment_relink``'s proven approach) and
check whether the reproduced ``ParsedAttachment`` now carries
``inline_bytes``. ``attachment_id`` is a pure content-identity hash of the
attachment's own metadata (``write.py:_attachment_id`` -- no session/message
component), so a match is accepted only when the identity hash is
byte-for-byte reproduced; nothing here fabricates or guesses a match.

## What is deliberately NOT attempted here

Drive-hosted (``upload_origin='drive'``) and OAuth-uploaded (ChatGPT/Claude
file-service) attachments whose bytes were never embedded in the export at
all need a *live, authenticated* fetch (Drive API / a ChatGPT downloader that
does not exist in this codebase) to ever succeed -- see polylogue-ck5v for the
live-Drive-fetch coupling problem, a separate, larger, network-touching
follow-on. This module never performs network I/O and never marks such an
attachment ``unavailable`` on that basis alone: an attachment this pass
cannot prove recoverable, and cannot prove structurally impossible either,
is reported ``undetermined`` and is left exactly as ``unfetched`` -- honest
about not-yet, not a false claim of never.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, ingest_record
from polylogue.storage.attachment_relink import DEFAULT_RAW_ROW_LIMIT, _iter_raw_session_rows
from polylogue.storage.blob_publication import ArchiveBlobPublisher, consume_blob_publication_receipt
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime.raw.records import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import _attachment_id
from polylogue.storage.sqlite.migration_runner import validate_backup_manifest_covers_derived_tier
from polylogue.storage.sqlite.queries.mappers_archive import _row_to_raw_session

logger = get_logger(__name__)

#: Bumped whenever comparison/promotion semantics change; recorded on every
#: manifest row so a later audit can tell which tool version acted.
TOOL_VERSION = "attachment-reacquisition-v1"

_SANDBOX_UNRECOVERABLE_REASON = (
    "assistant-generated Code Interpreter output (source_url startswith 'sandbox:'); "
    "chatgpt.py's own sandbox_file handling documents that the export never carries these bytes"
)
_NO_RECOVERY_EVIDENCE_REASON = (
    "no re-parsed raw_sessions row (within the scanned bound) reproduces this attachment's "
    "content-identity hash with inline bytes -- may need a live authenticated fetch (Drive/OAuth) "
    "or its raw evidence may be outside the scanned window; left unfetched, not marked unavailable"
)


@dataclass(frozen=True, slots=True)
class ReacquirableAttachment:
    """One unfetched attachment a re-parsed raw proves now carries bytes."""

    attachment_id: str
    raw_id: str
    session_id: str
    byte_count: int
    mime_type: str | None


@dataclass(frozen=True, slots=True)
class UnrecoverableAttachment:
    """One unfetched attachment proven structurally impossible to acquire."""

    attachment_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class AttachmentReacquisitionPlan:
    """Read-only projection: what a reacquisition pass would do against the archive.

    ``reacquirable``/``unrecoverable`` are the *complete* sets (never
    sampled) -- an actuator applying this plan must act on every item here,
    not a bounded preview. ``undetermined_count`` is whatever remains of the
    original unfetched set after both classifications; it is deliberately
    left as a count only (not a full list) because "no verdict yet" is not
    actionable state a caller would iterate.
    """

    unfetched_count: int
    reacquirable: tuple[ReacquirableAttachment, ...]
    unrecoverable: tuple[UnrecoverableAttachment, ...]
    undetermined_count: int
    raw_rows_scanned: int
    raw_rows_total: int

    @property
    def ok(self) -> bool:
        return self.unfetched_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "unfetched_count": self.unfetched_count,
            "reacquirable_count": len(self.reacquirable),
            "reacquirable_bytes": sum(item.byte_count for item in self.reacquirable),
            "unrecoverable_count": len(self.unrecoverable),
            "undetermined_count": self.undetermined_count,
            "raw_rows_scanned": self.raw_rows_scanned,
            "raw_rows_total": self.raw_rows_total,
        }


RawSessionParser = Callable[[RawSessionRecord], IngestRecordResult]


def _default_raw_session_parser(archive_root: Path, blob_root: Path | None) -> RawSessionParser:
    archive_root_str = str(archive_root)
    blob_root_str = str(blob_root) if blob_root is not None else None

    def _parse(raw_record: RawSessionRecord) -> IngestRecordResult:
        return ingest_record(raw_record, archive_root_str, "advisory", blob_root_str=blob_root_str)

    return _parse


def _read_unfetched_attachments(index_conn: sqlite3.Connection) -> dict[str, bool]:
    """Return ``{attachment_id: is_sandbox_only}`` for every unfetched row.

    ``is_sandbox_only`` is true when every non-null ``source_url`` recorded
    across this attachment's refs starts with ``sandbox:`` -- the marker
    ``chatgpt.py`` stamps exclusively on Code Interpreter deliverables.
    """
    original_row_factory = index_conn.row_factory
    index_conn.row_factory = sqlite3.Row
    try:
        rows = index_conn.execute(
            """
            SELECT a.attachment_id AS attachment_id, r.source_url AS source_url
            FROM attachments a
            LEFT JOIN attachment_refs r ON r.attachment_id = a.attachment_id
            WHERE a.acquisition_status = 'unfetched'
            """
        ).fetchall()
    finally:
        index_conn.row_factory = original_row_factory

    by_id: dict[str, list[str | None]] = {}
    for row in rows:
        by_id.setdefault(str(row["attachment_id"]), []).append(row["source_url"])

    return {
        attachment_id: bool(urls) and all(url is not None and url.startswith("sandbox:") for url in urls)
        for attachment_id, urls in by_id.items()
    }


def plan_attachment_reacquisition(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    blob_root: Path | None = None,
    raw_row_limit: int | None = DEFAULT_RAW_ROW_LIMIT,
    raw_session_parser: RawSessionParser | None = None,
) -> AttachmentReacquisitionPlan:
    """Classify unfetched attachments without mutating archive state.

    Never opens a write transaction and never touches the blob store. For
    each unfetched attachment, first checks the static sandbox-output
    signature (no reparse needed); everything else is checked by re-parsing
    ``raw_sessions`` rows (newest first, bounded by ``raw_row_limit`` --
    mirrors :func:`polylogue.storage.attachment_relink.plan_orphaned_attachment_relink`)
    via the real :func:`polylogue.pipeline.services.ingest_worker.ingest_record`
    entry point and recomputing each reproduced attachment's content-identity
    hash.
    """
    unfetched = _read_unfetched_attachments(index_conn)
    unfetched_count = len(unfetched)
    if unfetched_count == 0:
        return AttachmentReacquisitionPlan(
            unfetched_count=0,
            reacquirable=(),
            unrecoverable=(),
            undetermined_count=0,
            raw_rows_scanned=0,
            raw_rows_total=0,
        )

    unrecoverable: dict[str, str] = {
        attachment_id: _SANDBOX_UNRECOVERABLE_REASON for attachment_id, is_sandbox in unfetched.items() if is_sandbox
    }

    pending: set[str] = set(unfetched) - set(unrecoverable)
    reacquired: dict[str, ReacquirableAttachment] = {}

    parser = raw_session_parser or _default_raw_session_parser(archive_root, blob_root)

    total_rows_row = source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()
    raw_rows_total = int(total_rows_row[0]) if total_rows_row is not None else 0

    scanned = 0
    if pending:
        rows = _iter_raw_session_rows(source_conn, raw_row_limit=raw_row_limit)
        for row in rows:
            if not pending:
                break
            scanned += 1
            try:
                raw_record = _row_to_raw_session(row)
            except Exception as exc:  # pragma: no cover - defensive, malformed row shape
                logger.warning(
                    "attachment reacquisition: could not build RawSessionRecord for raw_id=%s: %s",
                    row["raw_id"],
                    exc,
                )
                continue
            try:
                result = parser(raw_record)
            except Exception as exc:
                logger.warning("attachment reacquisition: reparse failed for raw_id=%s: %s", raw_record.raw_id, exc)
                continue
            if result.error is not None:
                continue
            for payload in result.sessions:
                for attachment in payload.parsed_session.attachments:
                    attachment_id = _attachment_id(payload.session_id, attachment)
                    if attachment_id not in pending or attachment.inline_bytes is None:
                        continue
                    reacquired[attachment_id] = ReacquirableAttachment(
                        attachment_id=attachment_id,
                        raw_id=raw_record.raw_id,
                        session_id=payload.session_id,
                        byte_count=len(attachment.inline_bytes),
                        mime_type=attachment.mime_type,
                    )
                    pending.discard(attachment_id)

    return AttachmentReacquisitionPlan(
        unfetched_count=unfetched_count,
        reacquirable=tuple(reacquired.values()),
        unrecoverable=tuple(
            UnrecoverableAttachment(attachment_id=attachment_id, reason=reason)
            for attachment_id, reason in unrecoverable.items()
        ),
        undetermined_count=len(pending),
        raw_rows_scanned=scanned,
        raw_rows_total=raw_rows_total,
    )


@dataclass(frozen=True, slots=True)
class AttachmentReacquisitionResult:
    """Outcome of one classify-then-act pass (dry-run or applied)."""

    unfetched_count: int
    reacquirable_count: int
    reacquired_count: int
    reacquired_bytes: int
    unrecoverable_count: int
    marked_unavailable_count: int
    undetermined_count: int
    applied: bool
    manifest_path: str | None = None
    backup_manifest: str | None = None
    errors: tuple[str, ...] = ()

    @classmethod
    def from_plan(cls, plan: AttachmentReacquisitionPlan, *, applied: bool) -> AttachmentReacquisitionResult:
        return cls(
            unfetched_count=plan.unfetched_count,
            reacquirable_count=len(plan.reacquirable),
            reacquired_count=0,
            reacquired_bytes=0,
            unrecoverable_count=len(plan.unrecoverable),
            marked_unavailable_count=0,
            undetermined_count=plan.undetermined_count,
            applied=applied,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "unfetched_count": self.unfetched_count,
            "reacquirable_count": self.reacquirable_count,
            "reacquired_count": self.reacquired_count,
            "reacquired_bytes": self.reacquired_bytes,
            "unrecoverable_count": self.unrecoverable_count,
            "marked_unavailable_count": self.marked_unavailable_count,
            "undetermined_count": self.undetermined_count,
            "manifest_path": self.manifest_path,
            "backup_manifest": self.backup_manifest,
            "errors": list(self.errors),
        }


class AttachmentReacquisitionError(RuntimeError):
    """Raised when applying an attachment reacquisition pass is refused."""


def _write_jsonl_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    import json
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _checkpoint_index_tier(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise AttachmentReacquisitionError("could not checkpoint index.db before backup validation") from exc
    if row is None:
        raise AttachmentReacquisitionError("could not checkpoint index.db before backup validation")


def _reacquire_one(
    parser: RawSessionParser,
    source_conn: sqlite3.Connection,
    candidate: ReacquirableAttachment,
) -> bytes | None:
    """Re-run the reparse for exactly one candidate's raw_id, live, right before writing.

    Mirrors ``raw_live_source_reconciliation_apply``'s "never trust a
    previously computed report" discipline: the plan's verdict is what
    decided *which* attachments to act on, but the actual bytes published
    are re-derived fresh here rather than carried (unhashed, unpublished)
    across the classify/apply boundary.
    """
    original_row_factory = source_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    try:
        row = source_conn.execute("SELECT * FROM raw_sessions WHERE raw_id = ?", (candidate.raw_id,)).fetchone()
    finally:
        source_conn.row_factory = original_row_factory
    if row is None:
        return None
    raw_record = _row_to_raw_session(row)
    result = parser(raw_record)
    if result.error is not None:
        return None
    for payload in result.sessions:
        for attachment in payload.parsed_session.attachments:
            if _attachment_id(payload.session_id, attachment) == candidate.attachment_id and attachment.inline_bytes:
                return attachment.inline_bytes
    return None


def apply_attachment_reacquisition(
    archive_root: Path,
    *,
    manifest_path: Path | None = None,
    backup_manifest: Path | None = None,
    raw_row_limit: int | None = DEFAULT_RAW_ROW_LIMIT,
    max_count: int | None = None,
    dry_run: bool = True,
) -> AttachmentReacquisitionResult:
    """Classify unfetched attachments, then act on the safe verdicts.

    ``dry_run=True`` (the default) never opens a write transaction; it runs
    the same classifier a real apply would and reports what it would do.

    ``dry_run=False`` requires both ``manifest_path`` (an immutable,
    append-only JSONL receipt of every row acted on) and ``backup_manifest``
    (a verified backup manifest for the ``index`` tier, checked via
    :func:`polylogue.storage.sqlite.migration_runner.validate_backup_manifest_covers_derived_tier`
    -- index.db is rebuildable and was never wired for the cryptographic
    attestation durable-tier migrations require, so this checks manifest
    coverage and a byte-exact live fingerprint instead; this mutation is
    still gated behind a verified backup so an operator-authorized
    ``--apply`` can never be the first time backup coverage is checked).

    Only two actions are ever taken: promoting a ``reacquirable`` attachment's
    ``blob_hash``/``byte_count``/``acquisition_status`` to ``'acquired'``
    after publishing the freshly re-derived bytes through the same
    content-addressed blob store every other acquisition path uses, and
    marking an ``unrecoverable`` attachment's ``acquisition_status`` as
    ``'unavailable'``. ``undetermined`` attachments are never touched.
    """
    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    if not index_db.exists():
        raise FileNotFoundError(f"no index.db at {index_db}")
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")
    blob_store = BlobStore(archive_root / "blob")

    if dry_run:
        with (
            closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as index_conn,
            closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as source_conn,
        ):
            plan = plan_attachment_reacquisition(
                index_conn,
                source_conn,
                archive_root=archive_root,
                blob_root=blob_store.root,
                raw_row_limit=raw_row_limit,
            )
        return AttachmentReacquisitionResult.from_plan(plan, applied=False)

    if manifest_path is None:
        raise AttachmentReacquisitionError(
            "applying attachment reacquisition requires --manifest-path for the immutable receipt"
        )
    if backup_manifest is None:
        raise AttachmentReacquisitionError(
            "applying attachment reacquisition requires a verified backup manifest (--backup-manifest)"
        )

    index_conn = sqlite3.connect(index_db)
    try:
        _checkpoint_index_tier(index_conn)
        # Lock-free precheck: reject a missing/stale/wrong-tier manifest
        # before paying for classification + a write-lock acquisition.
        validate_backup_manifest_covers_derived_tier(backup_manifest, ArchiveTier.INDEX, connection=index_conn)

        with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as source_conn_ro:
            plan = plan_attachment_reacquisition(
                index_conn,
                source_conn_ro,
                archive_root=archive_root,
                blob_root=blob_store.root,
                raw_row_limit=raw_row_limit,
            )

        reacquirable = plan.reacquirable if max_count is None else plan.reacquirable[: max(0, max_count)]
        unrecoverable = plan.unrecoverable if max_count is None else plan.unrecoverable[: max(0, max_count)]

        parser = _default_raw_session_parser(archive_root, blob_store.root)
        publisher = ArchiveBlobPublisher(source_db, blob_store.root, store=blob_store)

        manifest_rows: list[dict[str, object]] = []
        errors: list[str] = []
        reacquired_writes: list[tuple[ReacquirableAttachment, str, int, str | None]] = []
        source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            for candidate in reacquirable:
                try:
                    inline_bytes = _reacquire_one(parser, source_conn, candidate)
                except Exception as exc:
                    logger.warning(
                        "attachment reacquisition: reparse-at-apply-time failed for attachment_id=%s raw_id=%s: %s",
                        candidate.attachment_id,
                        candidate.raw_id,
                        exc,
                    )
                    errors.append(f"{candidate.attachment_id}: {exc}")
                    continue
                if inline_bytes is None:
                    errors.append(f"{candidate.attachment_id}: reparse no longer reproduces inline bytes")
                    continue
                new_hash = hashlib.sha256(inline_bytes).hexdigest()
                existed_before = blob_store.exists(new_hash)
                published_hash, _size = publisher.write_from_bytes(inline_bytes)
                receipt_id = publisher.receipt_id(published_hash)
                reacquired_writes.append((candidate, new_hash, len(inline_bytes), receipt_id))
                manifest_rows.append(
                    {
                        "action": "reacquired",
                        "attachment_id": candidate.attachment_id,
                        "raw_id": candidate.raw_id,
                        "session_id": candidate.session_id,
                        "old_status": "unfetched",
                        "new_status": "acquired",
                        "blob_hash": new_hash,
                        "byte_count": len(inline_bytes),
                        "blob_already_existed": existed_before,
                        "compared_at_ms": int(time.time() * 1000),
                        "tool_version": TOOL_VERSION,
                    }
                )
            publisher.flush()
        finally:
            source_conn.close()

        for item in unrecoverable:
            manifest_rows.append(
                {
                    "action": "marked_unavailable",
                    "attachment_id": item.attachment_id,
                    "reason": item.reason,
                    "old_status": "unfetched",
                    "new_status": "unavailable",
                    "compared_at_ms": int(time.time() * 1000),
                    "tool_version": TOOL_VERSION,
                }
            )

        # Authoritative re-validation now that classification/publication is
        # done and we are about to take the write lock -- matches
        # migrate_archive_tier's / raw_live_source_reconciliation_apply's
        # pattern: a concurrent write between the precheck and this lock
        # acquisition would make the backup stale.
        validate_backup_manifest_covers_derived_tier(backup_manifest, ArchiveTier.INDEX, connection=index_conn)

        reacquired_count = 0
        reacquired_bytes = 0
        marked_unavailable_count = 0
        consumed_receipts: list[tuple[str | None, str]] = []
        index_conn.execute("BEGIN IMMEDIATE")
        try:
            for candidate, new_hash, byte_count, receipt_id in reacquired_writes:
                cursor = index_conn.execute(
                    """
                    UPDATE attachments
                    SET blob_hash = ?, byte_count = ?, acquisition_status = 'acquired'
                    WHERE attachment_id = ? AND acquisition_status = 'unfetched'
                    """,
                    (bytes.fromhex(new_hash), byte_count, candidate.attachment_id),
                )
                if cursor.rowcount != 1:
                    # Already changed under us since classification -- skip
                    # rather than assert, matching the conservative posture
                    # raw_live_source_reconciliation_apply uses.
                    continue
                consumed_receipts.append((receipt_id, new_hash))
                reacquired_count += 1
                reacquired_bytes += byte_count

            for item in unrecoverable:
                cursor = index_conn.execute(
                    """
                    UPDATE attachments
                    SET acquisition_status = 'unavailable'
                    WHERE attachment_id = ? AND acquisition_status = 'unfetched'
                    """,
                    (item.attachment_id,),
                )
                if cursor.rowcount == 1:
                    marked_unavailable_count += 1

            quick_check = index_conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise AttachmentReacquisitionError(f"index.db quick_check failed after apply: {quick_check!r}")
        except Exception:
            if index_conn.in_transaction:
                index_conn.rollback()
            raise
        else:
            index_conn.commit()

        # blob_publication_reservations lives in source.db, a different tier
        # connection from the one that just committed the attachments UPDATE
        # above -- consuming it here (after the index-tier commit succeeded)
        # is safe: the reservation is a crash-window safety net for blob GC,
        # not a cross-tier atomicity requirement, and an unconsumed
        # reservation is merely reconciled later, never data loss.
        if consumed_receipts:
            source_write_conn = sqlite3.connect(source_db)
            try:
                for receipt_id, new_hash in consumed_receipts:
                    consume_blob_publication_receipt(source_write_conn, receipt_id, bytes.fromhex(new_hash))
                source_write_conn.commit()
            finally:
                source_write_conn.close()

        _write_jsonl_manifest(manifest_path, manifest_rows)
    finally:
        index_conn.close()

    return AttachmentReacquisitionResult(
        unfetched_count=plan.unfetched_count,
        reacquirable_count=len(plan.reacquirable),
        reacquired_count=reacquired_count,
        reacquired_bytes=reacquired_bytes,
        unrecoverable_count=len(plan.unrecoverable),
        marked_unavailable_count=marked_unavailable_count,
        undetermined_count=plan.undetermined_count,
        applied=True,
        manifest_path=str(manifest_path),
        backup_manifest=str(backup_manifest),
        errors=tuple(errors),
    )


__all__ = [
    "TOOL_VERSION",
    "AttachmentReacquisitionError",
    "AttachmentReacquisitionPlan",
    "AttachmentReacquisitionResult",
    "ReacquirableAttachment",
    "UnrecoverableAttachment",
    "apply_attachment_reacquisition",
    "plan_attachment_reacquisition",
]

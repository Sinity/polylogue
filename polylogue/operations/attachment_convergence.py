"""Daemon-owned convergence for provider-hosted attachment bytes.

Attachment acquisition used to happen only while the Drive folder iterator
was reading a payload.  This owner works from the canonical attachment rows
instead, so a payload restored from a ZIP (or any other route) is still
eligible for the same live Drive fetch.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from polylogue.core.identity_law import attachment_acquisition_coordinate
from polylogue.core.stage_admission import admit_stage_write
from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.logging import WARNING, emit, get_logger
from polylogue.storage.blob_publication import ArchiveBlobPublisher, publication_refused
from polylogue.storage.blob_store import PreparedBlob
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceBlobRef,
    is_blob_hash_excised,
    write_source_blob_refs,
)
from polylogue.storage.sqlite.connection_profile import attach_database, open_readonly_connection
from polylogue.storage.sqlite.queries.attachment_records import (
    UNFETCHED_DRIVE_REFERENCE_SQL,
    contested_native_id_predicate,
    unambiguous_native_id_sql,
    unresolved_attachment_identity_count,
)

logger = get_logger(__name__)

DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT = 25


@dataclass(frozen=True, slots=True)
class AttachmentConvergenceResult:
    inspected: int = 0
    acquired: int = 0
    terminal: int = 0
    deferred: int = 0
    #: Candidates refused because the downloaded bytes hash to content the
    #: operator durably excised. Counted separately from ``terminal`` so a
    #: privacy refusal is never read as an ordinary transport failure.
    excised: int = 0
    #: Unfetched Drive references whose provider identity is contested -- two
    #: observations of one id kind for one reference. They are neither
    #: candidates nor terminal: downloading under a lexically chosen id binds
    #: bytes to the wrong attachment, and marking them ``unavailable`` would
    #: assert a permanent absence nobody measured. They stay ``unfetched`` and
    #: out of the bounded window so one contested reference cannot starve it.
    unresolved_identity: int = 0
    #: Candidates whose canonical object exists under its SHA-256 path but no
    #: longer hashes to that name. A *sub-count* of ``deferred``, not a
    #: separate outcome: the row stays ``unfetched`` and retryable, because
    #: neither of the two states this pass can otherwise produce is true --
    #: the bytes are not acquired, and their absence is not permanent.
    contradicted: int = 0
    #: Unfetched Drive references whose supplying raw -- the acquisition whose
    #: parse produced the reference -- is not retained in ``source.db``. The
    #: durable blob ref names that raw, and binding the bytes to any other raw
    #: (the session's current one included) attributes them to an acquisition
    #: that never held them. Like contested identity they are neither
    #: transport work nor terminal: they stay ``unfetched`` until the derived
    #: index converges on the retained raws. Counted by the same SQL predicate
    #: that excludes them from the transport window.
    unattributed: int = 0

    @property
    def transport_pending(self) -> bool:
        """Whether executable transport work remains for a later window.

        Contested identity is not transport work: no retry can resolve it, so
        it never keeps the stage pending (which would re-execute it forever).
        """
        return self.deferred > 0

    @property
    def complete(self) -> bool:
        """Whether the whole owed obligation is discharged.

        A contested or unattributed reference is neither acquired nor
        terminally absent, so it keeps the obligation incomplete even when no
        transport work remains.
        """
        return self.deferred == 0 and self.unresolved_identity == 0 and self.unattributed == 0


@dataclass(frozen=True, slots=True)
class _CandidateWindow:
    rows: tuple[sqlite3.Row, ...]
    #: Owed references whose supplying raw is not retained.
    unattributed: int


@dataclass(frozen=True, slots=True)
class _Acquired:
    """One reference whose bytes this pass holds, with its durable source ref."""

    attachment_id: str
    blob_hash: bytes
    byte_count: int
    ref: ArchiveSourceBlobRef


#: Placeholder budget for one ``raw_sessions`` membership probe.
_RAW_PROBE_CHUNK = 500


def _retained_raw_ids(source_conn: sqlite3.Connection, raw_ids: set[str]) -> set[str]:
    """The subset of ``raw_ids`` that ``source.db`` durably retains."""
    ordered = sorted(raw_ids)
    retained: set[str] = set()
    for start in range(0, len(ordered), _RAW_PROBE_CHUNK):
        chunk = ordered[start : start + _RAW_PROBE_CHUNK]
        marks = ",".join("?" for _ in chunk)
        retained.update(
            str(row[0])
            for row in source_conn.execute(f"SELECT raw_id FROM raw_sessions WHERE raw_id IN ({marks})", chunk)
        )
    return retained


def _candidate_rows(conn: sqlite3.Connection, source_conn: sqlite3.Connection, *, limit: int) -> _CandidateWindow:
    """Select a bounded transport window using the durable supplier relation.

    Join the retained source tier before LIMIT so missing suppliers cannot
    starve valid candidates. Do not page the whole archive through Python to
    find one eligible reference: the stage's one-row work probe uses this same
    query. The supplier is checked again under the publication writer lease.
    """
    wanted = max(0, int(limit))
    if wanted == 0:
        return _CandidateWindow(rows=(), unattributed=0)
    source_path = next(str(row[2]) for row in source_conn.execute("PRAGMA database_list") if row[1] == "main")
    if not source_path:
        raise ValueError("attachment convergence requires a retained source database")
    attached = {str(row[1]): str(row[2]) for row in conn.execute("PRAGMA database_list")}
    alias = "attachment_source"
    if alias not in attached:
        attach_database(conn, source_path, alias=alias)
    elif Path(attached[alias]).resolve() != Path(source_path).resolve():
        raise ValueError("attachment convergence source attachment changed")
    conn.row_factory = sqlite3.Row
    retained = "EXISTS (SELECT 1 FROM attachment_source.raw_sessions AS raw WHERE raw.raw_id = r.supplying_raw_id)"
    predicate = f"{UNFETCHED_DRIVE_REFERENCE_SQL} AND NOT {contested_native_id_predicate()}"
    unattributed = int(conn.execute(f"SELECT COUNT(*) {predicate} AND NOT {retained}").fetchone()[0])
    rows = conn.execute(
        f"""
        SELECT a.attachment_id, r.ref_id, r.session_id, r.upload_origin,
               r.source_url, r.supplying_raw_id,
               COALESCE(
                   ({unambiguous_native_id_sql("file")}),
                   ({unambiguous_native_id_sql("drive")}),
                   ({unambiguous_native_id_sql("attachment")})
               ) AS provider_file_id
        {predicate} AND {retained}
        ORDER BY a.attachment_id, r.ref_id
        LIMIT ?
        """,
        (wanted,),
    ).fetchall()
    return _CandidateWindow(rows=tuple(rows), unattributed=unattributed)


def inspect_attachment_readiness(index: sqlite3.Connection, source: sqlite3.Connection | None) -> dict[str, int]:
    """Measure owed references against the caller's exact Source and Index.

    Transport terminal answers and excision refusals share the stored
    ``unavailable`` disposition. Their per-pass typed events retain the reason;
    this snapshot reports the measured terminal denominator without guessing it.
    """
    if source is None:
        raise sqlite3.OperationalError("attachment supplier authority unavailable")
    unresolved = unresolved_attachment_identity_count(index)
    unattributed = allowed = 0
    for raw_id, identity_blocked in index.execute(
        f"SELECT r.supplying_raw_id, {contested_native_id_predicate()} {UNFETCHED_DRIVE_REFERENCE_SQL}"
    ):
        if identity_blocked:
            continue
        if raw_id is None or source.execute("SELECT 1 FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone() is None:
            unattributed += 1
        else:
            allowed += 1
    terminal = int(
        index.execute(
            "SELECT COUNT(*) FROM attachments a WHERE a.acquisition_status='unavailable' AND EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id=a.attachment_id AND r.upload_origin='drive')"
        ).fetchone()[0]
    )
    acquired = int(
        index.execute(
            "SELECT COUNT(*) FROM attachments a WHERE a.acquisition_status='acquired' AND EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id=a.attachment_id AND r.upload_origin='drive')"
        ).fetchone()[0]
    )
    return {
        "allowed_unfetched": allowed,
        "unresolved_identity": unresolved,
        "unattributed": unattributed,
        "terminal_unavailable": terminal,
        "acquired": acquired,
    }


def _report_unattributed(unattributed: int) -> None:
    """Name owed references whose supplying acquisition is not retained."""
    if unattributed:
        emit(
            "operations.attachment_convergence.supplier_unretained",
            level=WARNING,
            outcome="degraded",
            unattributed=unattributed,
            reason="attachment reference names no retained raw acquisition",
        )


def _report_unresolved_identity(conn: sqlite3.Connection) -> int:
    """Count contested owed references and name them as a degraded outcome.

    Not a transport failure and not a terminal absence: the archive holds two
    provider identities for one reference and no evidence that ranks them.
    Named on every pass that looks, rather than letting a silent lexical
    choice make the ambiguity invisible.
    """
    unresolved_identity = unresolved_attachment_identity_count(conn)
    if unresolved_identity:
        emit(
            "operations.attachment_convergence.identity_unresolved",
            level=WARNING,
            outcome="degraded",
            unresolved_identity=unresolved_identity,
            reason="two native ids of one kind for one attachment reference",
        )
    return unresolved_identity


def _permanent_failure(exc: BaseException) -> bool:
    """Classify only the provider's explicit answer about the file as terminal.

    The Drive gateway owns reading a provider HTTP error: it raises
    ``DriveNotFoundError`` for 404 and ``DriveAccessDeniedError`` for a 403
    that denies the file, and leaves a throttled 403 (``userRateLimitExceeded``
    and its siblings) as the provider's own, retryable exception. Everything
    else -- a rate limit, a transport or auth failure, or a local
    ``PermissionError`` while staging the download -- keeps the row
    ``unfetched`` for a later pass.
    """
    from polylogue.sources.drive.types import DriveAccessDeniedError, DriveNotFoundError

    return isinstance(exc, (DriveNotFoundError, DriveAccessDeniedError))


def _acquisition_coordinate(row: sqlite3.Row) -> str:
    """Keep the provider coordinate distinct even when payload bytes agree."""
    return attachment_acquisition_coordinate(row["provider_file_id"], str(row["attachment_id"]))


def _surviving_blob_ref(
    source_conn: sqlite3.Connection,
    *,
    attachment_id: str,
    raw_id: str,
    source_path: str,
    blob_store: ArchiveBlobPublisher,
    verified_hashes: dict[bytes, bool],
) -> tuple[bytes, int] | None:
    """Return the blob a retained source ref still names, when it survives.

    The durable source tier keeps an ``attachment`` blob ref per raw session
    and acquisition coordinate.  After a rebuild the derived attachment row is
    ``unfetched`` again while those bytes are still on disk, so a re-bind is
    the correct recovery and a re-download is waste that a dead provider file
    would turn into permanent loss.

    Ambiguity is refused rather than guessed: binding the wrong blob to an
    attachment is worse than fetching it again.  ``_acquisition_coordinate``
    keeps the coordinate per attachment rather than per raw session, so the
    refusal is reserved for genuinely indistinguishable evidence instead of
    firing on every raw that carries more than one attachment.

    Survival is decided by re-hashing the object, not by ``exists``
    (polylogue-o0uw5).  ``exists`` answers "a file sits at that path", while
    the re-bind it gates writes ``acquisition_status = 'acquired'`` -- a
    positive claim that these exact bytes were fetched and stored.  A path
    that survived with content the recorded hash no longer names is not a
    survivor.  The fresh-acquisition path in
    :func:`converge_drive_attachments` hashes the payload it actually read,
    so this keeps both routes to ``acquired`` backed by the same evidence
    instead of leaving the cheaper one trusted.

    A contradicted object falls through to the provider, which either
    republishes real bytes under a *different* hash or terminates the row
    ``unavailable``.  It does not follow that the republished bytes may be
    written over the contradiction: when the provider returns the original
    payload, its destination is the contradicted path, and
    :func:`converge_drive_attachments` refuses the publication there rather
    than deduping a good payload onto a bad file.
    """
    rows = source_conn.execute(
        """
        SELECT blob_hash, size_bytes
        FROM blob_refs
        WHERE ref_type = 'attachment' AND ref_id = ? AND source_path = ?
        """,
        (raw_id, source_path),
    ).fetchall()
    if len(rows) != 1:
        return None
    blob_hash = bytes(rows[0][0])
    size_bytes = int(rows[0][1])
    if is_blob_hash_excised(source_conn, blob_hash):
        return None
    verified = verified_hashes.get(blob_hash)
    if verified is None:
        verified = blob_store.verify(blob_hash.hex())
        verified_hashes[blob_hash] = verified
    if not verified:
        if blob_store.exists(blob_hash.hex()):
            # Durable-ledger evidence contradicted by the object it names.
            # Silently re-downloading would repair the row and erase the only
            # signal that a published blob decayed.
            emit(
                "operations.attachment_convergence.survivor_contradicted",
                level=WARNING,
                outcome="degraded",
                attachment_id=attachment_id,
                raw_id=raw_id,
                blob_hash=blob_hash.hex(),
                reason="stored object does not hash to the recorded blob ref",
            )
        return None
    return blob_hash, size_bytes


def _download_prepared(
    publisher: ArchiveBlobPublisher,
    provider_file_id: str,
    download_into: Callable[[str, IO[bytes]], None],
) -> PreparedBlob:
    """Stream one provider file into this archive's blob staging area.

    The download lands directly in the private staging file that becomes the
    prepared blob, so an attachment of any size costs bounded memory and one
    staged copy on disk.
    """
    return publisher.prepare_from_writer(lambda handle: download_into(provider_file_id, handle))


def converge_drive_attachments(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    download_into: Callable[[str, IO[bytes]], None],
    limit: int = DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT,
    now_ms: Callable[[], int] | None = None,
    open_write_connections: Callable[[], tuple[sqlite3.Connection, sqlite3.Connection]] | None = None,
) -> AttachmentConvergenceResult:
    """Fetch one bounded window and publish durable attachment state.

    The query is intentionally route-neutral: it starts at indexed attachment
    references, not at ``iter_drive_raw_data`` or a source path.  Each
    download streams into a staging file, so an attachment of any size costs
    bounded memory; its bytes are hashed as staged and published before the
    index row is marked acquired.  Explicit not-found results are terminal
    ``unavailable`` rows; all other failures remain retryable.

    ``open_write_connections`` separates the scan from the publication: the
    passed connections then serve the candidate scan and survival probe, and
    the writable pair is opened inside the admitted write section. Opening a
    daemon write connection itself requires the lease, so a caller that runs
    this pass off the writer -- the daemon stage engine does -- must supply the
    opener rather than hand in connections it could not have opened yet.
    """
    unresolved_identity = _report_unresolved_identity(index_conn)
    window = _candidate_rows(index_conn, source_conn, limit=limit)
    unattributed = window.unattributed
    rows = window.rows
    if not rows:
        _report_unattributed(unattributed)
        return AttachmentConvergenceResult(unresolved_identity=unresolved_identity, unattributed=unattributed)
    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    acquired: list[_Acquired] = []
    #: Rows bound to a blob whose stored object still re-hashes to the
    #: recorded identity; no provider request and no new source blob ref, but
    #: the same durable index outcome -- and the same content evidence -- as a
    #: fresh acquisition.
    rebound_rows: list[tuple[str, bytes, int]] = []
    terminal_ids: list[str] = []
    excised_ids: list[str] = []
    #: Rows this pass refuses to acquire because the canonical destination
    #: for their recorded hash holds different bytes. They stay ``unfetched``.
    contradicted_ids: list[str] = []
    deferred = 0
    # One content-addressed attachment can have refs in several sessions.  A
    # bounded pass must not spend one Drive request per ref; retain only the
    # fetch outcome (never the payload) and still emit one source ref per raw.
    fetch_outcomes: dict[str, tuple[str, bytes | None, int]] = {}
    verified_survivors: dict[bytes, bool] = {}
    observed_at_ms = now_ms() if now_ms is not None else int(time.time() * 1000)

    try:
        for row in rows:
            attachment_id = str(row["attachment_id"])
            provider_file_id = row["provider_file_id"]
            if not isinstance(provider_file_id, str) or not provider_file_id:
                terminal_ids.append(attachment_id)
                continue
            # The acquisition whose parse produced this reference, checked
            # against ``raw_sessions`` when the window was drawn.
            raw_id = str(row["supplying_raw_id"])
            source_path = _acquisition_coordinate(row)
            surviving = _surviving_blob_ref(
                source_conn,
                attachment_id=attachment_id,
                raw_id=raw_id,
                source_path=source_path,
                blob_store=publisher,
                verified_hashes=verified_survivors,
            )
            if surviving is not None:
                # The bytes are already in the blob store and the durable
                # source ledger still points at them.  Re-bind the rebuilt
                # index row instead of spending a provider request on content
                # the archive never lost.
                rebound_rows.append((attachment_id, surviving[0], surviving[1]))
                continue
            cached = fetch_outcomes.get(provider_file_id)
            if cached is not None:
                outcome, cached_hash, cached_size = cached
                if outcome == "terminal":
                    terminal_ids.append(attachment_id)
                    continue
                if outcome == "excised":
                    excised_ids.append(attachment_id)
                    continue
                if outcome == "deferred":
                    deferred += 1
                    continue
                if outcome == "contradicted":
                    # One provider file, several attachment rows: the blocked
                    # publication is a property of the destination, so every
                    # row sharing it is blocked too rather than re-downloading.
                    contradicted_ids.append(attachment_id)
                    deferred += 1
                    continue
                assert outcome == "acquired" and cached_hash is not None
                acquired.append(
                    _Acquired(
                        attachment_id,
                        cached_hash,
                        cached_size,
                        ArchiveSourceBlobRef(
                            blob_hash=cached_hash,
                            raw_id=raw_id,
                            ref_type="attachment",
                            source_path=source_path,
                            size_bytes=cached_size,
                            acquired_at_ms=observed_at_ms,
                            publication_receipt_id=publisher.receipt_id(cached_hash.hex()),
                        ),
                    )
                )
                continue
            prepared: PreparedBlob | None = None
            try:
                prepared = _download_prepared(publisher, provider_file_id, download_into)
                candidate_hash = bytes.fromhex(prepared.hash_hex)
                if is_blob_hash_excised(source_conn, candidate_hash):
                    # Refuse BEFORE publishing. write_from_bytes would stage
                    # the payload and queue a receipt that publisher.flush()
                    # then reserves and moves into the blob store, and a
                    # reservation makes the hash permanently GC-immune. The
                    # operator excised exactly these bytes; re-downloading
                    # them from Drive must not put them back.
                    fetch_outcomes[provider_file_id] = ("excised", None, 0)
                    excised_ids.append(attachment_id)
                    emit(
                        "operations.attachment_convergence.excised_refused",
                        level=WARNING,
                        outcome="degraded",
                        attachment_id=attachment_id,
                        blob_hash=candidate_hash.hex(),
                        reason="durable excision ledger",
                    )
                    continue
                if publisher.exists(candidate_hash.hex()) and not publisher.verify(candidate_hash.hex()):
                    # The destination these bytes would publish to already
                    # holds *different* bytes. ``ArchiveBlobPublisher.flush()``
                    # reaches ``BlobStore.publish_prepared``, which discards a
                    # staged payload whenever its destination exists -- so
                    # publishing here would throw the good payload away, leave
                    # the contradicted object in place, and still write the
                    # index row ``acquired`` under that hash. Convergence would
                    # report a recovery it had not performed.
                    #
                    # Refuse before staging anything. The row stays
                    # ``unfetched`` and retryable: the bytes are not acquired,
                    # and their absence is not permanent -- what is broken is a
                    # durable object, which this pass has no authority to
                    # replace.
                    fetch_outcomes[provider_file_id] = ("contradicted", None, 0)
                    contradicted_ids.append(attachment_id)
                    deferred += 1
                    emit(
                        "operations.attachment_convergence.publication_blocked",
                        level=WARNING,
                        outcome="degraded",
                        attachment_id=attachment_id,
                        raw_id=raw_id,
                        blob_hash=candidate_hash.hex(),
                        reason="canonical blob destination does not hash to its own name",
                    )
                    continue
                blob_hash_hex, byte_count = publisher.queue_prepared(prepared)
                prepared = None
            except Exception as exc:
                if _permanent_failure(exc):
                    fetch_outcomes[provider_file_id] = ("terminal", None, 0)
                    terminal_ids.append(attachment_id)
                else:
                    fetch_outcomes[provider_file_id] = ("deferred", None, 0)
                    deferred += 1
                    logger.info("attachment convergence deferred %s: %s", attachment_id, exc)
                continue
            finally:
                # A download refused before publication (excised, contradicted,
                # failed) leaves its staged file behind; it is never queued.
                if prepared is not None:
                    publisher.discard_prepared(prepared)

            blob_hash = bytes.fromhex(blob_hash_hex)
            fetch_outcomes[provider_file_id] = ("acquired", blob_hash, byte_count)
            acquired.append(
                _Acquired(
                    attachment_id,
                    blob_hash,
                    byte_count,
                    ArchiveSourceBlobRef(
                        blob_hash=blob_hash,
                        raw_id=raw_id,
                        ref_type="attachment",
                        source_path=source_path,
                        size_bytes=byte_count,
                        acquired_at_ms=observed_at_ms,
                        publication_receipt_id=publisher.receipt_id(blob_hash_hex),
                    ),
                )
            )

        def publish_attachment_outcomes() -> None:
            """The one section of this pass that writes; every download is behind us."""
            if open_write_connections is None:
                _publish(index_conn, source_conn)
                return
            write_index, write_source = open_write_connections()
            try:
                _publish(write_index, write_source)
            finally:
                write_source.close()
                write_index.close()

        def _publish(index_conn: sqlite3.Connection, source_conn: sqlite3.Connection) -> None:
            nonlocal unattributed
            # The window checked each supplier against ``raw_sessions`` before
            # downloading; a raw can still be retired while the downloads ran.
            # Re-check under the writer, so no durable ref names a raw that is
            # gone, and drop publications only those refs would have used.
            retained = _retained_raw_ids(source_conn, {str(item.ref.raw_id) for item in acquired})
            kept = [item for item in acquired if str(item.ref.raw_id) in retained]
            unattributed += len(acquired) - len(kept)
            kept_receipts = {item.ref.publication_receipt_id for item in kept}
            for receipt_id in {item.ref.publication_receipt_id for item in acquired} - kept_receipts:
                if receipt_id is not None:
                    publisher.discard_pending_receipt(receipt_id)
            acquired[:] = kept
            publisher.flush()
            # Preserve the publication-boundary excision check after filtering
            # by supplying acquisition. Never publish a source reference for
            # bytes refused during flush or excised since their download.
            from polylogue.storage.sqlite.archive_tiers.source_write import is_blob_hash_excised

            refused_hashes = {
                item.blob_hash
                for item in acquired
                if publication_refused(publisher, item.blob_hash.hex())
                or is_blob_hash_excised(source_conn, item.blob_hash)
            }
            if refused_hashes:
                excised_ids.extend(item.attachment_id for item in acquired if item.blob_hash in refused_hashes)
                acquired[:] = [item for item in acquired if item.blob_hash not in refused_hashes]
            if acquired:
                by_raw_id: dict[str, list[ArchiveSourceBlobRef]] = {}
                for item in acquired:
                    by_raw_id.setdefault(str(item.ref.raw_id), []).append(item.ref)
                for raw_id, refs in by_raw_id.items():
                    write_source_blob_refs(source_conn, raw_id, tuple(refs).__iter__)
            if acquired:
                source_conn.commit()
            acquired_rows = [(item.attachment_id, item.blob_hash, item.byte_count) for item in acquired]
            if acquired_rows or rebound_rows:
                with index_conn:
                    for attachment_id, blob_hash, byte_count in (*acquired_rows, *rebound_rows):
                        index_conn.execute(
                            """
                            UPDATE attachments
                            SET blob_hash = ?, byte_count = ?, acquisition_status = 'acquired'
                            WHERE attachment_id = ? AND acquisition_status = 'unfetched'
                            """,
                            (blob_hash, byte_count, attachment_id),
                        )
            if excised_ids:
                # Same terminal index state as an unavailable payload -- the
                # bytes will never be acquired -- but reached by refusal, which
                # the result counts and the log names.
                with index_conn:
                    index_conn.executemany(
                        "UPDATE attachments SET acquisition_status = 'unavailable' "
                        "WHERE attachment_id = ? AND acquisition_status = 'unfetched'",
                        ((attachment_id,) for attachment_id in excised_ids),
                    )
            if terminal_ids:
                with index_conn:
                    index_conn.executemany(
                        "UPDATE attachments SET acquisition_status = 'unavailable' WHERE attachment_id = ? AND acquisition_status = 'unfetched'",
                        ((attachment_id,) for attachment_id in terminal_ids),
                    )

        admit_stage_write("convergence.stage.attachment_bytes.publish", publish_attachment_outcomes)
        if _candidate_rows(index_conn, source_conn, limit=1).rows:
            # The scheduler records a false result as convergence debt.  Count
            # the remaining canonical rows as deferred even when this window
            # itself had no transport failure.
            deferred += 1
    finally:
        publisher.discard_pending()

    _report_unattributed(unattributed)
    return AttachmentConvergenceResult(
        inspected=len(rows),
        acquired=len(acquired) + len(rebound_rows),
        terminal=len(terminal_ids),
        deferred=deferred,
        excised=len(excised_ids),
        unresolved_identity=unresolved_identity,
        contradicted=len(contradicted_ids),
        unattributed=unattributed,
    )


def make_attachment_convergence_stage(
    db_path: Path,
    *,
    archive_root: Path,
    client_factory: Callable[[], object],
    limit: int = DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT,
) -> ConvergenceStage:
    """Build the bounded whole-archive stage used by daemon convergence."""

    def _open_write() -> tuple[sqlite3.Connection, sqlite3.Connection]:
        """Open the writable pair. Only legal inside the admitted write section."""
        from polylogue.storage.sqlite.connection_profile import open_daemon_connection

        index = open_daemon_connection(db_path, archive_root=archive_root)
        source = open_daemon_connection(archive_root / "source.db", archive_root=archive_root)
        return index, source

    def _open_read() -> tuple[sqlite3.Connection, sqlite3.Connection]:
        return (
            open_readonly_connection(db_path),
            open_readonly_connection(archive_root / "source.db"),
        )

    def _has_work() -> bool:
        """Whether a resolvable owed reference exists for a transport window.

        The same predicate as the fetch window: a contested reference is not
        work any execution can do, so counting it here re-executed the stage on
        every pass for as long as the ambiguity stood. It is reported as
        degraded instead, and leaves the obligation incomplete.
        """
        source_db = archive_root / "source.db"
        if not db_path.exists() or not source_db.exists():
            return False
        conn = open_readonly_connection(db_path)
        source = open_readonly_connection(source_db)
        try:
            _report_unresolved_identity(conn)
            window = _candidate_rows(conn, source, limit=1)
            if not window.rows:
                _report_unattributed(window.unattributed)
            return bool(window.rows)
        finally:
            source.close()
            conn.close()

    def check(_path: Path) -> bool:
        return _has_work()

    def execute(_path: Path) -> StageExecuteReturn:
        # Construction of the Drive client and every download it performs
        # happen here, outside the writer: only ``open_write_connections`` and
        # the publication it feeds run under the admitted lease.
        index, source = _open_read()
        try:
            client = client_factory()
            result = converge_drive_attachments(
                index,
                source,
                archive_root=archive_root,
                download_into=client.download_into,  # type: ignore[attr-defined]
                limit=limit,
                open_write_connections=_open_write,
            )
            return not result.transport_pending
        finally:
            source.close()
            index.close()

    def check_many(paths: Sequence[Path]) -> set[Path]:
        active = tuple(paths)
        return set(active) if active and _has_work() else set()

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        active = tuple(paths)
        if not active:
            return True
        return execute(active[0])

    return ConvergenceStage(
        name="attachment_bytes",
        description="Backfill provider-hosted attachment bytes from canonical references",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        false_means_pending=True,
        whole_archive=True,
        writer_admission="bridged",
    )


def make_configured_attachment_convergence_stage(db_path: Path, *, archive_root: Path) -> ConvergenceStage:
    """Construct the daemon stage with the runtime's configured Drive client."""
    from polylogue.config import resolve_runtime_config
    from polylogue.sources.drive.source_factory import build_drive_source_client

    def client_factory() -> object:
        return build_drive_source_client(config=resolve_runtime_config().drive_config)

    return make_attachment_convergence_stage(
        db_path,
        archive_root=archive_root,
        client_factory=client_factory,
    )


__all__ = [
    "AttachmentConvergenceResult",
    "DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT",
    "converge_drive_attachments",
    "make_configured_attachment_convergence_stage",
    "make_attachment_convergence_stage",
]

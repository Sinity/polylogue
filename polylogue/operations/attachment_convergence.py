"""Daemon-owned convergence for provider-hosted attachment bytes.

Attachment acquisition used to happen only while the Drive folder iterator
was reading a payload.  This owner works from the canonical attachment rows
instead, so a payload restored from a ZIP (or any other route) is still
eligible for the same live Drive fetch.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.stage_admission import admit_stage_write
from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.logging import WARNING, emit, get_logger
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceBlobRef,
    is_blob_hash_excised,
    write_source_blob_refs,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.queries.attachment_records import (
    contested_native_id_predicate,
    unambiguous_native_id_sql,
)

logger = get_logger(__name__)

DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT = 25
DEFAULT_MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024


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

    @property
    def complete(self) -> bool:
        return self.deferred == 0


#: Drive-hosted references still owed bytes. ``upload_origin`` and
#: ``acquisition_status`` decide eligibility; identity decides resolvability.
_UNFETCHED_DRIVE_REFERENCE_SQL = """
    FROM attachments AS a
    JOIN attachment_refs AS r ON r.attachment_id = a.attachment_id
    JOIN sessions AS s ON s.session_id = r.session_id
    WHERE a.acquisition_status = 'unfetched'
      AND r.upload_origin = 'drive'
"""


def _candidate_rows(conn: sqlite3.Connection, *, limit: int) -> list[sqlite3.Row]:
    """One bounded window of references with a resolvable provider identity.

    The download coordinate comes from :func:`unambiguous_native_id_sql`, the
    same selection the session reads use, so what the operator is shown and
    what the downloader asks for cannot diverge. A reference whose identity is
    contested is excluded here rather than downloaded under a lexical winner;
    :func:`_unresolved_identity_count` reports it.
    """
    conn.row_factory = sqlite3.Row
    return list(
        conn.execute(
            f"""
            SELECT a.attachment_id, r.ref_id, r.session_id, r.upload_origin,
                   r.source_url, s.raw_id,
                   COALESCE(
                       ({unambiguous_native_id_sql("file")}),
                       ({unambiguous_native_id_sql("drive")}),
                       ({unambiguous_native_id_sql("attachment")})
                   ) AS provider_file_id
            {_UNFETCHED_DRIVE_REFERENCE_SQL}
              AND NOT {contested_native_id_predicate()}
            ORDER BY a.attachment_id, r.ref_id
            LIMIT ?
            """,
            (max(0, int(limit)),),
        ).fetchall()
    )


def _unresolved_identity_count(conn: sqlite3.Connection) -> int:
    """How many owed Drive references this pass refused to resolve at all."""
    row = conn.execute(
        f"""
        SELECT COUNT(*)
        {_UNFETCHED_DRIVE_REFERENCE_SQL}
          AND {contested_native_id_predicate()}
        """
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _permanent_failure(exc: BaseException) -> bool:
    """Classify only explicit absence/access failures as terminal.

    Transport/auth failures remain retryable.  The Drive client deliberately
    exposes ``DriveNotFoundError`` for the permanent 404 case; importing the
    type lazily keeps this module usable with small test doubles.
    """
    from polylogue.sources.drive.types import DriveNotFoundError

    if isinstance(exc, (DriveNotFoundError, PermissionError)):
        return True
    # The gateway preserves provider HTTP errors for callers that need the
    # response metadata.  Treat only explicit absence/access responses as
    # terminal; rate limits and transport failures must retain convergence
    # debt for a later pass.
    status = getattr(getattr(exc, "resp", None), "status", None)
    return status in {403, 404}


def _acquisition_coordinate(row: sqlite3.Row) -> str:
    """Name the attachment whose bytes a durable source ref carries.

    ``blob_refs`` is keyed on ``(blob_hash, ref_type, ref_id)``; the raw
    session id alone therefore does not distinguish two attachments of the
    same raw, and ``source_url`` is ``None`` for every Drive-hosted document
    (``sources/parsers/drive_support_attachments.py`` never sets it).  Writing
    one raw-wide coordinate made every multi-attachment raw ambiguous on the
    way back: after a derived-tier rebuild the survival probe refused, both
    rows fell through to the provider, and a deleted file turned retained
    bytes into a terminal ``unavailable`` row.  The provider file id is the
    stable per-attachment name available at both ends of that round trip.
    """
    source_url = row["source_url"]
    if isinstance(source_url, str) and source_url:
        return source_url
    return f"attachment:{row['provider_file_id']}"


def _surviving_blob_ref(
    source_conn: sqlite3.Connection,
    *,
    attachment_id: str,
    raw_id: str,
    source_path: str,
    blob_store: ArchiveBlobPublisher,
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
    survivor: it falls through to the provider, which either republishes real
    bytes or terminates the row ``unavailable``.  The fresh-acquisition path
    in :func:`converge_drive_attachments` hashes the payload it actually read,
    so this keeps both routes to ``acquired`` backed by the same evidence
    instead of leaving the cheaper one trusted.
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
    if not blob_store.verify(blob_hash.hex()):
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


def converge_drive_attachments(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    download_bytes: Callable[[str], bytes],
    limit: int = DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT,
    max_attachment_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
    now_ms: Callable[[], int] | None = None,
    open_write_connections: Callable[[], tuple[sqlite3.Connection, sqlite3.Connection]] | None = None,
) -> AttachmentConvergenceResult:
    """Fetch one bounded window and publish durable attachment state.

    The query is intentionally route-neutral: it starts at indexed attachment
    references, not at ``iter_drive_raw_data`` or a source path.  Successful
    bytes are hashed from the bytes actually read and published before the
    index row is marked acquired.  Oversize and explicit not-found results are
    terminal ``unavailable`` rows; all other failures remain retryable.

    ``open_write_connections`` separates the scan from the publication: the
    passed connections then serve the candidate scan and survival probe, and
    the writable pair is opened inside the admitted write section. Opening a
    daemon write connection itself requires the lease, so a caller that runs
    this pass off the writer -- the daemon stage engine does -- must supply the
    opener rather than hand in connections it could not have opened yet.
    """
    unresolved_identity = _unresolved_identity_count(index_conn)
    if unresolved_identity:
        # Not a transport failure and not a terminal absence: the archive
        # holds two provider identities for one reference and no evidence
        # that ranks them. Name it every pass rather than letting a silent
        # lexical choice make the ambiguity invisible.
        emit(
            "operations.attachment_convergence.identity_unresolved",
            level=WARNING,
            outcome="degraded",
            unresolved_identity=unresolved_identity,
            reason="two native ids of one kind for one attachment reference",
        )
    rows = _candidate_rows(index_conn, limit=limit)
    if not rows:
        return AttachmentConvergenceResult(unresolved_identity=unresolved_identity)
    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    acquired_refs: list[ArchiveSourceBlobRef] = []
    acquired_rows: list[tuple[str, bytes, int]] = []
    #: Rows bound to a blob whose stored object still re-hashes to the
    #: recorded identity; no provider request and no new source blob ref, but
    #: the same durable index outcome -- and the same content evidence -- as a
    #: fresh acquisition.
    rebound_rows: list[tuple[str, bytes, int]] = []
    terminal_ids: list[str] = []
    excised_ids: list[str] = []
    deferred = 0
    # One content-addressed attachment can have refs in several sessions.  A
    # bounded pass must not spend one Drive request per ref; retain only the
    # fetch outcome (never the payload) and still emit one source ref per raw.
    fetch_outcomes: dict[str, tuple[str, bytes | None, int]] = {}
    observed_at_ms = now_ms() if now_ms is not None else int(time.time() * 1000)

    try:
        for row in rows:
            attachment_id = str(row["attachment_id"])
            provider_file_id = row["provider_file_id"]
            if not isinstance(provider_file_id, str) or not provider_file_id:
                terminal_ids.append(attachment_id)
                continue
            raw_id = str(row["raw_id"])
            source_path = _acquisition_coordinate(row)
            surviving = _surviving_blob_ref(
                source_conn,
                attachment_id=attachment_id,
                raw_id=raw_id,
                source_path=source_path,
                blob_store=publisher,
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
                assert outcome == "acquired" and cached_hash is not None
                blob_hash = cached_hash
                byte_count = cached_size
                acquired_rows.append((attachment_id, blob_hash, byte_count))
                acquired_refs.append(
                    ArchiveSourceBlobRef(
                        blob_hash=blob_hash,
                        raw_id=raw_id,
                        ref_type="attachment",
                        source_path=source_path,
                        size_bytes=byte_count,
                        acquired_at_ms=observed_at_ms,
                        publication_receipt_id=publisher.receipt_id(blob_hash.hex()),
                    )
                )
                continue
            try:
                payload = download_bytes(provider_file_id)
                if len(payload) > max_attachment_bytes:
                    fetch_outcomes[provider_file_id] = ("terminal", None, 0)
                    terminal_ids.append(attachment_id)
                    continue
                candidate_hash = hashlib.sha256(payload).digest()
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
                blob_hash_hex, byte_count = publisher.write_from_bytes(payload)
            except Exception as exc:
                if _permanent_failure(exc):
                    fetch_outcomes[provider_file_id] = ("terminal", None, 0)
                    terminal_ids.append(attachment_id)
                else:
                    fetch_outcomes[provider_file_id] = ("deferred", None, 0)
                    deferred += 1
                    logger.info("attachment convergence deferred %s: %s", attachment_id, exc)
                continue

            blob_hash = bytes.fromhex(blob_hash_hex)
            assert hashlib.sha256(payload).digest() == blob_hash
            fetch_outcomes[provider_file_id] = ("acquired", blob_hash, byte_count)
            acquired_rows.append((attachment_id, blob_hash, byte_count))
            acquired_refs.append(
                ArchiveSourceBlobRef(
                    blob_hash=blob_hash,
                    raw_id=str(row["raw_id"]),
                    ref_type="attachment",
                    source_path=source_path,
                    size_bytes=byte_count,
                    acquired_at_ms=observed_at_ms,
                    publication_receipt_id=publisher.receipt_id(blob_hash_hex),
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
            publisher.flush()
            if acquired_refs:
                by_raw_id: dict[str, list[ArchiveSourceBlobRef]] = {}
                for ref in acquired_refs:
                    by_raw_id.setdefault(str(ref.raw_id), []).append(ref)
                for raw_id, refs in by_raw_id.items():
                    write_source_blob_refs(source_conn, raw_id, tuple(refs))
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
        if _candidate_rows(index_conn, limit=1):
            # The scheduler records a false result as convergence debt.  Count
            # the remaining canonical rows as deferred even when this window
            # itself had no transport failure.
            deferred += 1
    finally:
        publisher.discard_pending()

    return AttachmentConvergenceResult(
        inspected=len(rows),
        acquired=len(acquired_rows) + len(rebound_rows),
        terminal=len(terminal_ids),
        deferred=deferred,
        excised=len(excised_ids),
        unresolved_identity=unresolved_identity,
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
        if not db_path.exists():
            return False
        conn = open_readonly_connection(db_path)
        try:
            return bool(
                conn.execute(
                    """
                    SELECT 1 FROM attachments a JOIN attachment_refs r ON r.attachment_id = a.attachment_id
                    WHERE a.acquisition_status = 'unfetched' AND r.upload_origin = 'drive' LIMIT 1
                    """
                ).fetchone()
            )
        finally:
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
                download_bytes=client.download_bytes,  # type: ignore[attr-defined]
                limit=limit,
                open_write_connections=_open_write,
            )
            return result.complete
        finally:
            source.close()
            index.close()

    return ConvergenceStage(
        name="attachment_bytes",
        description="Backfill provider-hosted attachment bytes from canonical references",
        check=check,
        execute=execute,
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
    "DEFAULT_MAX_ATTACHMENT_BYTES",
    "converge_drive_attachments",
    "make_configured_attachment_convergence_stage",
    "make_attachment_convergence_stage",
]

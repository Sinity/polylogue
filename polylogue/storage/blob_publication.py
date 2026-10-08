"""Archive-owned protection for content-addressed blob publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import time
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, BinaryIO, Protocol, cast
from uuid import uuid4

from polylogue.core.prepared_file import PreparedFileSeal
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.core.storage_faults import ArchiveStorageFaultError, StorageFaultKind
from polylogue.storage.blob_liveness import BlobLiveness, LivenessState, inspect_blob_liveness
from polylogue.storage.blob_store import BlobStore, Heartbeat, PreparedBlob
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.connection_profile import (
    open_readonly_connection,
    open_source_tier_write_connection,
    readonly_connection_context,
)
from polylogue.storage.sqlite.population_admission import assert_population_admitted
from polylogue.storage.sqlite.write_lease import require_write_lease

if TYPE_CHECKING:
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner
    from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit, PreparedIndexMutation


@dataclass(frozen=True, slots=True)
class BlobPublicationReceipt:
    """Identity of one publication attempt, independent of content identity."""

    publication_id: str
    blob_hash: str
    size_bytes: int
    publisher_id: str


@dataclass(frozen=True, slots=True, init=False)
class PreparedBlobPublicationClaim:
    """An exact publication claim allocated by its owning publisher."""

    receipt: BlobPublicationReceipt
    seal: PreparedFileSeal
    prepared_path: Path
    publisher: ArchiveBlobPublisher


def derived_publication_id(coordinate: str, blob_hash: str) -> str:
    """The stable publication identity of one blob for one retained coordinate."""
    digest = hashlib.sha256(
        b"polylogue-publication-claim-v1\0" + coordinate.encode("utf-8") + b"\0" + blob_hash.encode("ascii")
    ).hexdigest()
    return f"claim-{digest}"


def _prepared_publication_claim(
    publisher: ArchiveBlobPublisher,
    receipt: BlobPublicationReceipt,
    seal: PreparedFileSeal,
    prepared_path: Path,
) -> PreparedBlobPublicationClaim:
    claim = object.__new__(PreparedBlobPublicationClaim)
    object.__setattr__(claim, "receipt", receipt)
    object.__setattr__(claim, "seal", seal)
    object.__setattr__(claim, "prepared_path", Path(os.path.abspath(prepared_path)))
    object.__setattr__(claim, "publisher", publisher)
    return claim


def _prepared_claim_record(claim: PreparedBlobPublicationClaim) -> str:
    """Persist an owning publisher's claim inside a sealed preparation row."""
    return json.dumps(
        {
            "receipt": asdict(claim.receipt),
            "seal": asdict(claim.seal),
            "prepared_path": str(claim.prepared_path),
        },
        sort_keys=True,
    )


def _prepared_claim_from_record(encoded: str, publisher: ArchiveBlobPublisher) -> PreparedBlobPublicationClaim:
    """Restore a claim from its verified carrier, retaining the same publisher."""
    record = json.loads(encoded)
    receipt = BlobPublicationReceipt(**record["receipt"])
    seal = PreparedFileSeal(**record["seal"])
    if receipt.publisher_id != publisher.publisher_id:
        raise ValueError("sealed publication belongs to another publisher")
    if receipt.blob_hash != seal.sha256 or receipt.size_bytes != seal.size:
        raise ValueError("sealed publication claim disagrees with its file proof")
    path = Path(record["prepared_path"])
    if path != Path(os.path.abspath(path)):
        raise ValueError("sealed publication path is not its captured absolute path")
    # Restoring after publication need not reopen a private file already moved
    # into the blob namespace. Queue admission validates the actual path/file.
    path.relative_to(Path(os.path.abspath(publisher.root / ".staging")))
    return _prepared_publication_claim(publisher, receipt, seal, path)


@dataclass(frozen=True, slots=True)
class ArchiveWriterExclusion:
    """Proof that every archive-owned publisher is excluded for one archive."""

    source_db_path: Path
    _lock_file: IO[bytes]


@dataclass(frozen=True, slots=True)
class BlobPublicationInspection:
    publication_id: str
    blob_hash: str
    size_bytes: int
    publisher_id: str
    reserved_at_ms: int
    blob_present: bool
    liveness: BlobLiveness


@dataclass(frozen=True, slots=True)
class BlobPublicationReconciliation:
    cleared_referenced: int = 0
    cleared_missing: int = 0
    retained_referenced: int = 0
    retained_missing: int = 0
    unresolved: int = 0
    retained_blocked: int = 0
    blockers: tuple[str, ...] = ()
    scanned: int = 0
    last_scanned_publication_id: str | None = None


@dataclass(frozen=True, slots=True)
class BlobPublicationAbandonment:
    abandoned: int
    skipped_referenced: int
    missing_receipts: int


def _writer_lock_path(source_db_path: Path) -> Path:
    return source_db_path.with_name(".blob-publication-writers.lock")


@contextmanager
def exclude_archive_blob_publishers(source_db_path: Path) -> Iterator[ArchiveWriterExclusion]:
    """Acquire archive-wide exclusion against instrumented blob publishers."""
    lock_path = _writer_lock_path(source_db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield ArchiveWriterExclusion(source_db_path.resolve(), lock_file)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _archive_blob_publisher_slot(source_db_path: Path) -> Iterator[BinaryIO]:
    lock_path = _writer_lock_path(source_db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        # Actual file close releases flock. An explicit unlock before close
        # would surrender exclusion while a failed close retained the handle.
        yield lock_file


@dataclass(frozen=True, slots=True)
class BlobPublicationReservationStore:
    """Persist receipt rows in one source-tier transaction per batch."""

    source_db_path: Path

    def _open_connection(self) -> sqlite3.Connection:
        """Open the reservation's dedicated source writer under shared policy.

        Reservations intentionally keep their own short transaction: publisher
        flush runs after any caller-owned source transaction has committed, so
        joining an arbitrary archive handle would widen the publication
        boundary. The common source factory applies local pragmas only;
        source.db's WAL mode was established by fresh bootstrap.
        """
        return open_source_tier_write_connection(self.source_db_path, archive_root=self.source_db_path.parent)

    def prepare_many(
        self,
        receipts: Sequence[BlobPublicationReceipt],
        *,
        reference_seal: PreparedIndexMutation,
    ) -> tuple[KnownTierMutationPermit, frozenset[str]]:
        """Stage the first exact reservation unit on the original witness."""
        reference_seal.require_source_target(self.source_db_path)
        now_ms = int(time.time() * 1000)
        with reference_seal.original_read_snapshot(), reference_seal.source_producer(require_empty_schedule=True):
            observer = reference_seal.observer("source")
            excised = _excised_hashes(observer, {receipt.blob_hash for receipt in receipts})
            for receipt in receipts:
                if receipt.blob_hash in excised:
                    continue
                with reference_seal.original_rows(
                    "source",
                    "SELECT rowid FROM blob_publication_reservations WHERE publication_id=?",
                    (receipt.publication_id,),
                ) as rows:
                    found = rows.fetchone()
                if found is not None:
                    image = reference_seal.retain_tier_row("source", "blob_publication_reservations", found[0])
                    assert image is not None
                    reference_seal.load_source_row(image)
                    existing = dict(zip(image.columns, image.cells, strict=True))
                    # A derived claim identity re-adopts the reservation an
                    # earlier attempt left: the same publication of the same
                    # bytes, whichever publisher instance reserved it.
                    expected: dict[str, None | int | float | str | bytes] = {
                        "blob_hash": bytes.fromhex(receipt.blob_hash),
                        "size_bytes": receipt.size_bytes,
                    }
                    if any(
                        not reference_seal._literal_scalar_equal(existing[column], value)
                        for column, value in expected.items()
                    ):
                        raise ValueError("publication claim collides with another reservation")
                # Even an existing identical receipt has a canonical DO
                # NOTHING statement and must consume its no-effect schedule.
                cells = tuple(
                    reference_seal.retain_literal_scalar(value)
                    for value in (
                        receipt.publication_id,
                        bytes.fromhex(receipt.blob_hash),
                        receipt.publisher_id,
                    )
                )
                expressions = []
                operands: tuple[object, ...] = (None,)
                for cell in cells:
                    expression, bindings = reference_seal.source_literal_expression(cell)
                    expressions.append(expression)
                    operands += bindings
                with reference_seal.source_statement(
                    "INSERT INTO blob_publication_reservations("
                    "rowid,publication_id,blob_hash,publisher_id,size_bytes,reserved_at_ms) "
                    f"VALUES(?,{','.join(expressions)},?,?) ON CONFLICT(publication_id) DO NOTHING",
                    (*operands, receipt.size_bytes, now_ms),
                    table="blob_publication_reservations",
                    writable_targets=(("blob_publication_reservations", (cells[0],)),),
                    allocation_parameter=0,
                ):
                    pass
        return reference_seal.prepare_source_mutation(), excised

    def reserve_many(self, receipts: Sequence[BlobPublicationReceipt]) -> frozenset[str]:
        """Reserve ``receipts`` and return the blob hashes refused as excised.

        The excision ledger is read in the same write transaction, so no
        route can publish bytes the operator excised: a refused hash gets no
        reservation, and its caller discards the staged file instead of
        exposing it (polylogue-u6jyu).
        """
        if not receipts:
            return frozenset()
        now_ms = int(time.time() * 1000)
        require_write_lease(f"blob publication({self.source_db_path})", archive_root=self.source_db_path.parent)
        conn = self._open_connection()
        try:
            with closing(conn.execute("BEGIN IMMEDIATE")):
                pass
            excised = _excised_hashes(conn, {receipt.blob_hash for receipt in receipts})
            receipts = [receipt for receipt in receipts if receipt.blob_hash not in excised]
            new_receipts = []
            for receipt in receipts:
                with closing(
                    conn.execute(
                        "SELECT blob_hash, size_bytes, publisher_id FROM blob_publication_reservations WHERE publication_id = ?",
                        (receipt.publication_id,),
                    )
                ) as cursor:
                    existing = cursor.fetchone()
                if existing is None:
                    new_receipts.append(receipt)
                elif tuple(existing)[:2] != (bytes.fromhex(receipt.blob_hash), receipt.size_bytes):
                    raise ValueError("publication claim collides with another reservation")
            with closing(
                conn.executemany(
                    """
                INSERT INTO blob_publication_reservations (
                    publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        (
                            receipt.publication_id,
                            bytes.fromhex(receipt.blob_hash),
                            receipt.size_bytes,
                            receipt.publisher_id,
                            now_ms,
                        )
                        for receipt in new_receipts
                    ),
                )
            ):
                pass
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return excised


def _excised_hashes(conn: sqlite3.Connection, blob_hashes: set[str]) -> frozenset[str]:
    """Return which of ``blob_hashes`` (hex) the durable excision ledger names."""
    if not blob_hashes or not _table_exists(conn, "excised_content"):
        return frozenset()
    excised: set[str] = set()
    ordered = sorted(blob_hashes)
    for start in range(0, len(ordered), 500):
        chunk = ordered[start : start + 500]
        placeholders = ", ".join("?" for _ in chunk)
        with closing(
            conn.execute(
                f"SELECT removed_hash FROM excised_content WHERE hash_kind = 'blob_hash' AND removed_hash IN ({placeholders})",
                [bytes.fromhex(blob_hash) for blob_hash in chunk],
            )
        ) as cursor:
            rows = cursor.fetchall()
        excised.update(bytes(row[0]).hex() for row in rows)
    return frozenset(excised)


class AdoptedBlobEvictedError(ArchiveStorageFaultError):
    """A blob published outside the archive protocol was reclaimed before its reservation.

    The bytes still live in the retained source, so the input is not at fault:
    it is a storage fault, retried with the input's cursor untouched, and the
    retry publishes the bytes again.
    """

    def __init__(self, blob_hashes: Sequence[str]) -> None:
        self.blob_hashes = tuple(blob_hashes)
        super().__init__(
            StorageFaultKind.EVICTED,
            FileNotFoundError(f"adopted blob(s) missing before reservation: {', '.join(self.blob_hashes)}"),
        )


class ArchiveBlobPublisher(BlobStore):
    """Batch prepare, reserve, then publish blobs for one archive."""

    def __init__(self, source_db_path: Path, blob_root: Path, *, store: BlobStore | None = None) -> None:
        super().__init__(blob_root)
        self.source_db_path = source_db_path
        self.publisher_id = str(uuid4())
        self._store = store or BlobStore(blob_root)
        if self._store.root != blob_root:
            raise ValueError("publisher store root must match blob_root")
        self._pending: list[tuple[BlobPublicationReceipt, PreparedBlob]] = []
        self._adoptions: list[BlobPublicationReceipt] = []
        self._latest_receipt_by_hash: dict[str, str] = {}
        self._pending_by_hash: dict[str, PreparedBlob] = {}
        self._refused_as_excised: set[str] = set()
        self._reservation_seal: PreparedIndexMutation | None = None
        self._reservation_permit: KnownTierMutationPermit | None = None
        self._reservation_receipts: tuple[BlobPublicationReceipt, ...] = ()
        self._reservation_excised: frozenset[str] = frozenset()
        self._reservation_accepted = False
        self._reservation_native_owner: NativeSQLCustodyOwner | None = None

    def _queue(self, prepared: PreparedBlob, claim: PreparedBlobPublicationClaim | None = None) -> tuple[str, int]:
        receipt = (
            claim.receipt
            if claim is not None
            else BlobPublicationReceipt(
                publication_id=str(uuid4()),
                blob_hash=prepared.hash_hex,
                size_bytes=prepared.size_bytes,
                publisher_id=self.publisher_id,
            )
        )
        self._pending.append((receipt, prepared))
        if claim is None:
            self._latest_receipt_by_hash[prepared.hash_hex] = receipt.publication_id
        self._pending_by_hash[prepared.hash_hex] = prepared
        return prepared.hash_hex, prepared.size_bytes

    def _validate_claim_path(self, path: Path) -> Path:
        staging_root = Path(os.path.abspath(self._store.root / ".staging"))
        path = Path(os.path.abspath(path))
        path.relative_to(staging_root)
        cursor = path.parent
        while True:
            info = cursor.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("prepared claim must remain in owned private staging")
            if cursor == staging_root:
                return path
            cursor = cursor.parent

    def prepare_claim(self, prepared: PreparedBlob, *, coordinate: str | None = None) -> PreparedBlobPublicationClaim:
        """Allocate an exact claim and seal its bytes during off-writer preparation.

        ``coordinate`` names what the bytes are for (the retained raw and its
        position within it). Its claim identity is derived from that coordinate
        and the blob, so a retry of a crashed attempt re-adopts the reservation
        the attempt left and its reference transaction consumes it, instead of
        reserving again beside a receipt nothing will ever consume.
        """
        self._validate_claim_path(prepared.temporary_path)
        seal = PreparedFileSeal.capture(prepared.temporary_path)
        if seal.sha256 != prepared.hash_hex or seal.size != prepared.size_bytes:
            raise ValueError("prepared publication claim disagrees with its file")
        publication_id = str(uuid4()) if coordinate is None else derived_publication_id(coordinate, prepared.hash_hex)
        receipt = BlobPublicationReceipt(publication_id, prepared.hash_hex, prepared.size_bytes, self.publisher_id)
        return _prepared_publication_claim(self, receipt, seal, prepared.temporary_path)

    def queue_prepared(
        self,
        prepared: PreparedBlob,
        *,
        claim: PreparedBlobPublicationClaim | None = None,
    ) -> tuple[str, int]:
        """Queue bytes prepared by shared compute for writer-owned publication.

        This performs no source-tier mutation.  The admitted archive writer
        still owns ``flush()``, which reserves the receipt and publishes the
        staged file together under the publisher exclusion protocol.
        """
        staging_root = Path(os.path.abspath(self._store.root / ".staging"))
        try:
            if claim is None:
                prepared.temporary_path.resolve().relative_to(staging_root.resolve())
            else:
                Path(os.path.abspath(prepared.temporary_path)).relative_to(staging_root)
        except ValueError as exc:
            raise ValueError("prepared blob must belong to this archive's private staging root") from exc
        if claim is not None:
            if claim.publisher is not self or claim.receipt.publisher_id != self.publisher_id:
                raise ValueError("prepared claim belongs to another publisher")
            if claim.receipt.blob_hash != prepared.hash_hex or claim.receipt.size_bytes != prepared.size_bytes:
                raise ValueError("prepared claim does not name these bytes")
            if Path(os.path.abspath(prepared.temporary_path)) != claim.prepared_path:
                raise ValueError("prepared claim names another private path")
            if not claim.prepared_path.exists():
                # A reused sealed carrier names the same exact reservation,
                # never a new hash-only adoption or a fabricated receipt.
                with readonly_connection_context(self.source_db_path, validate_schema=False) as connection:
                    from polylogue.storage.sqlite.archive_tiers.source_write import is_blob_hash_excised

                    if is_blob_hash_excised(connection, bytes.fromhex(claim.receipt.blob_hash)):
                        return prepared.hash_hex, prepared.size_bytes
                    self.validate_published_claim(ConnectionBlobPublicationRead(connection), claim, source_path="")
                return prepared.hash_hex, prepared.size_bytes
            try:
                self._validate_claim_path(claim.prepared_path)
                claim.seal.verify(prepared.temporary_path, full=False)
            except (OSError, ValueError) as failure:
                raise ArchiveStorageFaultError(StorageFaultKind.EVICTED, failure) from failure
        return self._queue(prepared, claim)

    def validate_published_claim(
        self, source: BlobPublicationSourceRead, claim: PreparedBlobPublicationClaim, *, source_path: str
    ) -> tuple[str, int]:
        """Verify the exact reservation and final bytes inside the owning Source transaction."""
        from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

        if source.publication_source_path() != self.source_db_path.resolve():
            raise ValueError("publication belongs to another Source database")
        if claim.publisher is not self or claim.receipt.publisher_id != self.publisher_id:
            raise ValueError("publication belongs to another captured publisher")
        receipt = claim.receipt
        blob_hash = bytes.fromhex(receipt.blob_hash)
        if source.publication_blob_is_excised(blob_hash):
            raise ContentExcisedError(blob_hash=blob_hash, source_path=source_path)
        row = source.publication_reservation(receipt.publication_id)
        if row is None or tuple(row)[:2] != (blob_hash, receipt.size_bytes):
            raise ArchiveStorageFaultError(
                StorageFaultKind.EVICTED, FileNotFoundError("publication reservation is absent or changed")
            )
        try:
            info = self._store.blob_path(receipt.blob_hash).lstat()
        except OSError as failure:
            raise ArchiveStorageFaultError(StorageFaultKind.EVICTED, failure) from failure
        if not stat.S_ISREG(info.st_mode) or info.st_size != receipt.size_bytes:
            raise ArchiveStorageFaultError(
                StorageFaultKind.EVICTED, FileNotFoundError("publication bytes are absent or changed")
            )
        return receipt.blob_hash, receipt.size_bytes

    def write_from_path(self, source: Path, *, heartbeat: Heartbeat | None = None) -> tuple[str, int]:
        return self._queue(self._store.prepare_from_path(source, heartbeat=heartbeat))

    def write_from_fileobj(self, source: IO[bytes], *, heartbeat: Heartbeat | None = None) -> tuple[str, int]:
        return self._queue(self._store.prepare_from_fileobj(source, heartbeat=heartbeat))

    def write_from_writer(
        self, write: Callable[[IO[bytes]], None], *, heartbeat: Heartbeat | None = None
    ) -> tuple[str, int]:
        return self._queue(self._store.prepare_from_writer(write, heartbeat=heartbeat))

    def write_from_bytes(self, data: bytes) -> tuple[str, int]:
        return self._queue(self._store.prepare_from_bytes(data))

    def adopt_published(self, blob_hash: str, size_bytes: int) -> tuple[str, int]:
        """Queue a reservation for bytes another process already published.

        The streamed browser-capture decode runs in a compute worker that has
        no archive write lease, so it publishes attachment bytes straight into
        the store. Until a durable row references them those bytes are
        GC-eligible. ``flush()`` reserves the hash like any publication and,
        under the same exclusion GC unlinks under, proves the file is still
        present with the declared size; a reclaimed file raises
        :class:`AdoptedBlobEvictedError` instead of letting a missing blob be
        recorded as acquired.
        """
        self._store.blob_path(blob_hash)  # validates the hash before it is queued
        receipt = BlobPublicationReceipt(
            publication_id=str(uuid4()),
            blob_hash=blob_hash,
            size_bytes=size_bytes,
            publisher_id=self.publisher_id,
        )
        self._adoptions.append(receipt)
        self._latest_receipt_by_hash[blob_hash] = receipt.publication_id
        return blob_hash, size_bytes

    def _adopted_present(self, receipt: BlobPublicationReceipt) -> bool:
        try:
            status = self._store.blob_path(receipt.blob_hash).stat()
        except FileNotFoundError:
            return False
        return stat.S_ISREG(status.st_mode) and status.st_size == receipt.size_bytes

    def receipt_id(self, blob_hash: str) -> str | None:
        """Return the receipt for the most recent write of *blob_hash*."""
        return self._latest_receipt_by_hash.get(blob_hash)

    @property
    def has_pending(self) -> bool:
        """Whether ``flush()`` would do real work (take the source write lock).

        A batched replay caller commits its open archive transaction before a
        non-empty flush so this publisher's separate ``BEGIN IMMEDIATE``
        connection never waits behind the batch's held source.db write lock
        (the deadlock that previously forced per-cohort replay commits).
        """
        return bool(self._pending or self._adoptions)

    def prepare_flush(self, *, reference_seal: PreparedIndexMutation) -> None:
        """Prepare this exact reservation unit before later Source preparation.

        The existing publisher keeps its canonical receipt batch. Publication
        applies only this first same-witness unit, accepts its receipt, settles
        the dedicated connection, then exposes paths under publisher exclusion.
        """
        if self._reservation_permit is not None:
            raise ValueError("publisher already retains an original prepared reservation batch")
        reference_seal.require_source_target(self.source_db_path)
        receipts = (*(receipt for receipt, _prepared in self._pending), *self._adoptions)
        if not receipts:
            return
        permit, excised = BlobPublicationReservationStore(self.source_db_path).prepare_many(
            receipts,
            reference_seal=reference_seal,
        )
        self._reservation_seal = reference_seal
        self._reservation_permit = permit
        self._reservation_receipts = receipts
        self._reservation_excised = excised

    def settle_prepared_flush(self, *, reference_seal: PreparedIndexMutation) -> None:
        """Settle the original accepted child's custody before another gate."""
        if (
            self._reservation_seal is not reference_seal
            or self._reservation_permit is None
            or not self._reservation_accepted
        ):
            raise ValueError("prepared flush settlement requires its original authoritative accepted reservation")
        reference_seal._require_live_owner()
        owner = self._reservation_native_owner
        if owner is None:
            raise ValueError("accepted reservation lost its original native settlement owner")
        owner.close()

    def _publish_prepared_reservations(
        self,
        receipts: tuple[BlobPublicationReceipt, ...],
        reference_seal: PreparedIndexMutation,
    ) -> frozenset[str]:
        permit = self._reservation_permit
        original_receipts = iter(self._reservation_receipts)
        remaining_matches = all(any(candidate == receipt for candidate in original_receipts) for receipt in receipts)
        if (
            permit is None
            or self._reservation_seal is not reference_seal
            or receipts != self._reservation_receipts
            and not (self._reservation_accepted and remaining_matches)
        ):
            raise ValueError("publisher reservation batch differs from its original off-writer preparation")
        if self._reservation_accepted:
            # Only the original authoritative receipt reaches this branch.
            # Retry physical settlement of its exact retained child, never
            # replay a consumed schedule or invent a second reservation.
            self.settle_prepared_flush(reference_seal=reference_seal)
        else:
            from polylogue.storage.sqlite.connection_profile import native_sql_children

            with permit.hold_authority(), permit.mutation_connection() as connection:
                self._reservation_native_owner = next(
                    child for child in native_sql_children(reference_seal) if child.connection is connection
                )
                with reference_seal._owned_cursor(connection, "BEGIN IMMEDIATE"):
                    pass
                permit.apply_source_statements(connection)
                permit.allow_commit(connection)
                connection.commit()
                reference_seal.accept_known_tier_commit(permit.committed())
                self._reservation_accepted = True
        # Publisher exclusion is still held. A changed reservation/excision
        # generation after acceptance cannot be absorbed by resumed exposure.
        reference_seal.validate_observers_current()
        # Placement can fail after exposing an earlier member. Keep this exact
        # accepted batch until all placements settle; a retry validates currency
        # and deduplicates those same bytes without replaying Source statements.
        return self._reservation_excised

    def _retire_reservation_batch(self) -> None:
        self._reservation_permit = None
        self._reservation_seal = None
        self._reservation_receipts = ()
        self._reservation_accepted = False
        self._reservation_native_owner = None
        self._reservation_excised = frozenset()

    def flush(self, *, reference_seal: PreparedIndexMutation | None = None) -> tuple[BlobPublicationReceipt, ...]:
        """Commit all receipts once, then expose all corresponding final paths.

        Adopted blobs are checked before anything is reserved, under the
        publisher slot GC's unlink excludes, so a blob present at the check
        stays present once its reservation commits. When one is missing
        nothing of this batch is reserved or published: every queued write is
        discarded and :class:`AdoptedBlobEvictedError` is raised.
        """
        if not self._pending and not self._adoptions:
            return ()
        if reference_seal is not None:
            if self._reservation_accepted and self._reservation_seal is reference_seal:
                # Original physical cleanup precedes currency refusal. A
                # foreign commit blocks exposure, never the child's close.
                self.settle_prepared_flush(reference_seal=reference_seal)
            reference_seal.require_source_target(self.source_db_path)
        pending = tuple(self._pending)
        adoptions = tuple(self._adoptions)
        receipts = (*(receipt for receipt, _prepared in pending), *adoptions)
        with _archive_blob_publisher_slot(self.source_db_path):
            missing = [receipt.blob_hash for receipt in adoptions if not self._adopted_present(receipt)]
            if missing:
                # The excision ledger is consulted before absence is called a
                # storage fault: excision removes the bytes on purpose, so an
                # adopted blob that is gone and excised is refused as excised
                # by the reservation below (no reservation, recorded like any
                # flush refusal), never raised as an eviction.
                excised_missing = self._ledger_excised(missing)
                missing = [blob_hash for blob_hash in missing if blob_hash not in excised_missing]
                if missing:
                    self.discard_pending()
                    raise AdoptedBlobEvictedError(missing)
            excised = (
                BlobPublicationReservationStore(self.source_db_path).reserve_many(receipts)
                if reference_seal is None
                else self._publish_prepared_reservations(receipts, reference_seal)
            )
            for receipt, prepared in pending:
                if receipt.blob_hash in excised:
                    self._store.discard_prepared(prepared)
                    self._latest_receipt_by_hash.pop(receipt.blob_hash, None)
            for receipt in adoptions:
                if receipt.blob_hash in excised:
                    self._latest_receipt_by_hash.pop(receipt.blob_hash, None)
            self._store.publish_many(prepared for receipt, prepared in pending if receipt.blob_hash not in excised)
            if reference_seal is not None:
                self._retire_reservation_batch()
        self._refused_as_excised.update(excised)
        self._pending.clear()
        self._adoptions.clear()
        self._pending_by_hash.clear()
        return tuple(receipt for receipt in receipts if receipt.blob_hash not in excised)

    def refused_as_excised(self, blob_hash: str) -> bool:
        """Whether a flush() refused *blob_hash* because it is excised."""
        return blob_hash in self._refused_as_excised

    def _ledger_excised(self, blob_hashes: Sequence[str]) -> frozenset[str]:
        """Read the durable excision ledger for *blob_hashes* (caller holds the slot)."""
        conn = open_readonly_connection(self.source_db_path, timeout_class="background-read", validate_schema=False)
        try:
            return _excised_hashes(conn, set(blob_hashes))
        finally:
            conn.close()

    def excised_now(self, blob_hash: str) -> bool:
        """Whether the durable ledger names *blob_hash*, read under publisher exclusion.

        A flush's refusals cover only excisions committed before it; one that
        committed after the flush released its slot is visible only in the
        ledger. The shared slot orders this read against any excision that
        is still running, and a hit is remembered like a flush refusal.
        """
        if blob_hash in self._refused_as_excised:
            return True
        with _archive_blob_publisher_slot(self.source_db_path):
            conn = open_readonly_connection(self.source_db_path, timeout_class="background-read", validate_schema=False)
            try:
                excised = bool(_excised_hashes(conn, {blob_hash}))
            finally:
                conn.close()
        if excised:
            self._refused_as_excised.add(blob_hash)
        return excised

    def forget_refusals(self) -> None:
        """Drop the refusals a caller has already reconciled.

        A long-lived publisher (one accepted source's page walk) otherwise
        keeps one hash per refused file for the whole operation.
        """
        self._refused_as_excised.clear()

    def forget_completed_claim(self, claim: PreparedBlobPublicationClaim) -> None:
        """Retire local tracking after this exact claim is sealed and flushed.

        The carrier and Source reservation retain the receipt. Other queued
        captures of identical bytes retain their own publication ownership.
        """
        if claim.publisher is not self or claim.receipt.publisher_id != self.publisher_id:
            raise ValueError("prepared claim belongs to another publisher")
        publication_id = claim.receipt.publication_id
        if any(receipt.publication_id == publication_id for receipt, _ in self._pending) or any(
            receipt.publication_id == publication_id for receipt in self._adoptions
        ):
            raise RuntimeError("queued publication claim has not completed")
        blob_hash = claim.receipt.blob_hash
        if self._latest_receipt_by_hash.get(blob_hash) == publication_id:
            self._latest_receipt_by_hash.pop(blob_hash)
        if not any(receipt.blob_hash == blob_hash for receipt, _ in self._pending) and not any(
            receipt.blob_hash == blob_hash for receipt in self._adoptions
        ):
            self._refused_as_excised.discard(blob_hash)

    def discard_pending_receipt(self, publication_id: str) -> bool:
        """Drop one queued publication or adoption by its receipt, before any flush.

        Receipts, not hashes, identify a capture: two identical captures share
        a hash, and dropping one must not strand or drop the other. Returns
        whether the receipt was still queued.
        """
        if not any(receipt.publication_id == publication_id for receipt, _ in self._pending) and not any(
            receipt.publication_id == publication_id for receipt in self._adoptions
        ):
            return False
        if self._reservation_permit is not None and not self._reservation_accepted:
            raise ValueError("an unaccepted prepared reservation requires complete batch abandonment")
        if self._reservation_native_owner is not None:
            self._reservation_native_owner.close()
        discarded = self._discard_receipt_from_queue(publication_id)
        if not self._pending and not self._adoptions:
            self._retire_reservation_batch()
        return discarded

    def _discard_receipt_from_queue(self, publication_id: str) -> bool:
        blob_hash: str | None = None
        for index, (receipt, prepared) in enumerate(self._pending):
            if receipt.publication_id == publication_id:
                self._store.discard_prepared(prepared)
                del self._pending[index]
                blob_hash = receipt.blob_hash
                break
        else:
            for index, receipt in enumerate(self._adoptions):
                if receipt.publication_id == publication_id:
                    # An adopted blob was published by its worker; dropping the
                    # adoption only leaves those bytes to ordinary GC.
                    del self._adoptions[index]
                    blob_hash = receipt.blob_hash
                    break
        if blob_hash is None:
            return False
        earlier = [(receipt, prepared) for receipt, prepared in self._pending if receipt.blob_hash == blob_hash]
        earlier_adoptions = [receipt for receipt in self._adoptions if receipt.blob_hash == blob_hash]
        if earlier:
            self._latest_receipt_by_hash[blob_hash] = earlier[-1][0].publication_id
            self._pending_by_hash[blob_hash] = earlier[-1][1]
        else:
            self._pending_by_hash.pop(blob_hash, None)
            if earlier_adoptions:
                self._latest_receipt_by_hash[blob_hash] = earlier_adoptions[-1].publication_id
            else:
                self._latest_receipt_by_hash.pop(blob_hash, None)
        return True

    def discard_pending(self) -> None:
        # Terminal abandonment cannot remove prepared bytes while this batch's
        # actual native child still owns unsettled SQL/attachment custody.
        owner = self._reservation_native_owner
        if owner is not None:
            owner.close()
        failures: list[BaseException] = []
        for receipt, _prepared in tuple(self._pending):
            try:
                self._discard_receipt_from_queue(receipt.publication_id)
            except BaseException as exc:
                failures.append(exc)
        for receipt in tuple(self._adoptions):
            self._discard_receipt_from_queue(receipt.publication_id)
        if failures:
            raise BaseExceptionGroup("pending blob cleanup failed", failures)
        self._retire_reservation_batch()

    def blob_path(self, hash_hex: str) -> Path:
        final_path = self._store.blob_path(hash_hex)
        if final_path.exists():
            # Still retained for another session: a refused publication of
            # the same hash does not make its existing bytes unreadable.
            return final_path
        if hash_hex in self._refused_as_excised:
            # The staged bytes were discarded at flush. A reader gets the typed
            # excision instead of a path that does not exist.
            from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

            raise ContentExcisedError(blob_hash=bytes.fromhex(hash_hex), source_path=f"blob:{hash_hex}")
        prepared = self._pending_by_hash.get(hash_hex)
        return prepared.temporary_path if prepared is not None else final_path

    def exists(self, hash_hex: str) -> bool:
        if self._store.blob_path(hash_hex).exists():
            return True
        if hash_hex in self._refused_as_excised:
            return False
        return self.blob_path(hash_hex).exists()

    def open(self, hash_hex: str) -> BinaryIO:
        return self.blob_path(hash_hex).open("rb")

    def read_prefix(self, hash_hex: str, n: int = 65536) -> bytes:
        with self.open(hash_hex) as handle:
            return handle.read(n)

    def read_all(self, hash_hex: str) -> bytes:
        return self.blob_path(hash_hex).read_bytes()


def publication_receipt_id(blob_store: BlobStore, blob_hash: str) -> str | None:
    """Read an optional receipt without coupling pure source APIs to archives."""
    receipt_getter = getattr(blob_store, "receipt_id", None)
    if not callable(receipt_getter):
        return None
    receipt_id = receipt_getter(blob_hash)
    return str(receipt_id) if receipt_id is not None else None


def publication_refused(blob_store: BlobStore, blob_hash: str) -> bool:
    """Whether a flush of *blob_store* refused *blob_hash* as excised.

    A plain ``BlobStore`` publishes immediately and never refuses.
    """
    refused = getattr(blob_store, "refused_as_excised", None)
    return bool(callable(refused) and refused(blob_hash))


def refuse_excised_attachment_blobs(
    preacquired: dict[Any, tuple[bytes | None, int, str]],
    *,
    publisher: BlobStore | None = None,
    source_conn: sqlite3.Connection | None = None,
) -> dict[Any, tuple[bytes | None, int, str]]:
    """Downgrade acquired attachments whose bytes are excised to ``unavailable``.

    Excision is decided at two places: a flush that refused the staged bytes
    (``publisher``), and the durable ledger for bytes published earlier and
    excised since (``source_conn``). Either way the blob is not on disk, so the
    attachment row must not claim it: it keeps its identity and size with no
    blob hash, in the declared terminal ``unavailable`` state.
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import is_blob_hash_excised

    result: dict[Any, tuple[bytes | None, int, str]] = {}
    for key, (blob_hash, size, status) in preacquired.items():
        if (
            status == "acquired"
            and blob_hash is not None
            and (
                (publisher is not None and publication_refused(publisher, blob_hash.hex()))
                or (source_conn is not None and is_blob_hash_excised(source_conn, blob_hash))
            )
        ):
            result[key] = (None, size, "unavailable")
        else:
            result[key] = (blob_hash, size, status)
    return result


def reconcile_refused_attachments(
    acquired: dict[Any, tuple[bytes | None, int, str]],
    refs: tuple[Any, ...],
    publisher: BlobStore | None,
    *,
    source_conn: sqlite3.Connection | None = None,
) -> tuple[dict[Any, tuple[bytes | None, int, str]], tuple[Any, ...]]:
    """Drop what a flush refused, or the ledger now excises, from queued attachments and their refs.

    An excision can commit between the caller's ledger check and its flush
    (the flush then refuses and discards the bytes), or after a successful
    flush and before the caller writes its references. Either way the
    attachment is recorded ``unavailable`` and its reference is not written,
    exactly as when the ledger check itself saw the excision, instead of the
    reference write failing the whole replay.
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import is_blob_hash_excised

    if publisher is None and source_conn is None:
        return acquired, refs

    def excised(blob_hash: bytes) -> bool:
        return (publisher is not None and publication_refused(publisher, blob_hash.hex())) or (
            source_conn is not None and is_blob_hash_excised(source_conn, blob_hash)
        )

    return (
        refuse_excised_attachment_blobs(acquired, publisher=publisher, source_conn=source_conn),
        tuple(ref for ref in refs if not excised(bytes(ref.blob_hash))),
    )


def require_published(blob_store: BlobStore, blob_hash: str, *, source_path: str) -> None:
    """Raise ContentExcisedError when *blob_hash* is excised, by the flush or since.

    A caller that reads its snapshot back from the store after flushing must
    stop here: the refused bytes were discarded, so the path it would open
    does not exist, and the outcome is the typed excision, not a parse
    failure. An archive publisher also rechecks the durable ledger under
    publisher exclusion, so an excision committed after the flush is refused
    here too.
    """
    excised_now = getattr(blob_store, "excised_now", None)
    if publication_refused(blob_store, blob_hash) or (callable(excised_now) and excised_now(blob_hash)):
        from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

        raise ContentExcisedError(blob_hash=bytes.fromhex(blob_hash), source_path=source_path)


def flush_blob_publications(blob_store: BlobStore) -> tuple[BlobPublicationReceipt, ...]:
    """Flush an injected archive publisher; plain BlobStore is already final."""
    flush = getattr(blob_store, "flush", None)
    if not callable(flush):
        return ()
    result = flush()
    return tuple(result)


def consume_blob_publication_receipt(
    conn: sqlite3.Connection,
    publication_id: str | None,
    blob_hash: bytes,
) -> None:
    """Consume exactly one publication receipt in its durable-ref transaction."""
    if publication_id is None:
        return
    conn.execute(
        "DELETE FROM blob_publication_reservations WHERE publication_id = ? AND blob_hash = ?",
        (publication_id, blob_hash),
    )


def release_refused_publication_receipt(
    source_db_path: Path,
    publication_id: str | None,
    blob_hash: str | None,
) -> bool:
    """Drop the reservation for a publication whose referent was refused.

    A reservation protects staged bytes only until the durable row that
    references them exists. When the archive *refuses* that row --
    ``ContentExcisedError`` on durably excised content -- the referent will
    never exist, and the success path's ``consume_blob_publication_receipt``
    never runs. The orphaned reservation then makes the hash permanently
    GC-immune (``inspect_blob_reservation`` reports ``LIVE``), so every
    repeat pass over the same unchanged source file accrues another receipt
    and the excised bytes stay on disk forever.

    Releasing it hands the bytes back to ordinary blob GC, which reclaims
    them precisely because nothing references them. Returns whether a row
    was removed, so the caller can report the disposition rather than
    assuming it.
    """
    if publication_id is None or blob_hash is None:
        return False
    require_write_lease(f"blob publication refusal({source_db_path})", archive_root=source_db_path.parent)
    conn = open_source_tier_write_connection(source_db_path, archive_root=source_db_path.parent)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM blob_publication_reservations WHERE publication_id = ? AND blob_hash = ?",
            (publication_id, bytes.fromhex(blob_hash)),
        )
        removed = cursor.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return removed > 0


def consume_restored_raw_blob_receipts(
    source_db_path: Path,
    restored: Sequence[tuple[BlobPublicationReceipt, str]],
) -> int:
    """Consume receipts for bytes restored under an already-committed raw row.

    Restoring an absent retained blob publishes bytes whose referencing
    ``raw_sessions`` row committed long before, so no later reference
    transaction exists to consume the reservation. This is that transaction:
    each ``(receipt, raw_id)`` is consumed only while the raw row still names
    the receipt's hash, so a reservation never outlives its protection
    silently and never clears for a referent that is gone. Returns the number
    consumed.
    """
    if not restored:
        return 0
    require_write_lease(f"blob restoration receipts({source_db_path})", archive_root=source_db_path.parent)
    conn = open_source_tier_write_connection(source_db_path, archive_root=source_db_path.parent)
    try:
        conn.execute("BEGIN IMMEDIATE")
        consumed = 0
        for receipt, raw_id in restored:
            blob_hash = bytes.fromhex(receipt.blob_hash)
            cursor = conn.execute(
                "DELETE FROM blob_publication_reservations WHERE publication_id = ? AND blob_hash = ? "
                "AND EXISTS (SELECT 1 FROM raw_sessions WHERE raw_id = ? AND blob_hash = ?)",
                (receipt.publication_id, blob_hash, raw_id, blob_hash),
            )
            consumed += cursor.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return consumed


def _liveness_decision(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection | None,
    blob_hash: bytes,
) -> BlobLiveness:
    """Delegate to the canonical blob-liveness relation.

    Uses the same union GC and integrity consume: direct row-level
    ``blob_hash`` referents (``raw_sessions``/``attachments``/
    ``raw_hook_events``) and a ``blob_refs`` row only when its ``ref_type``
    still joins to a live referent -- a dangling ``blob_refs`` row alone
    (the prior behavior here) is not evidence of liveness.
    """
    return inspect_blob_liveness(
        source_conn,
        blob_hash.hex(),
        index_conn=index_conn,
        require_index=True,
    )


def inspect_blob_publication_receipts(
    source_db_path: Path,
    blob_root: Path,
    *,
    index_db_path: Path | None = None,
    max_count: int | None = None,
    after_publication_id: str | None = None,
    publication_ids: tuple[str, ...] | None = None,
) -> tuple[BlobPublicationInspection, ...]:
    """Return receipt evidence, optionally bounded by a stable ID cursor."""
    from polylogue.storage.archive_identity import ArchiveLocation

    if max_count is not None and max_count <= 0:
        raise ValueError("max_count must be positive when provided")
    source_conn = open_readonly_connection(source_db_path, timeout_class="background-read", validate_schema=False)
    index_conn: sqlite3.Connection | None = None
    try:
        source_conn.row_factory = sqlite3.Row
        if index_db_path is not None:
            resolved_index: Path = index_db_path
        else:
            resolved_index = ArchiveLocation.resolve(source_db_path.parent).active_index_path
        index_conn = (
            open_readonly_connection(resolved_index, timeout_class="background-read", validate_schema=False)
            if resolved_index.exists()
            else None
        )
        store = BlobStore(blob_root)
        if not _table_exists(source_conn, "blob_publication_reservations"):
            return ()
        if publication_ids is not None:
            if not publication_ids:
                return ()
            rows = source_conn.execute(
                f"""
                SELECT publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
                FROM blob_publication_reservations
                WHERE publication_id IN ({",".join("?" for _ in publication_ids)})
                ORDER BY publication_id
                """,
                publication_ids,
            ).fetchall()
        elif max_count is None and after_publication_id is None:
            rows = source_conn.execute(
                """
                SELECT publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
                FROM blob_publication_reservations
                ORDER BY reserved_at_ms, publication_id
                """
            ).fetchall()
        else:
            predicates = ""
            parameters: list[object] = []
            if after_publication_id is not None:
                predicates = "WHERE publication_id > ?"
                parameters.append(after_publication_id)
            limit = ""
            if max_count is not None:
                limit = "LIMIT ?"
                parameters.append(max_count)
            rows = source_conn.execute(
                f"""
                SELECT publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
                FROM blob_publication_reservations
                {predicates}
                ORDER BY publication_id
                {limit}
                """,
                parameters,
            ).fetchall()
        return tuple(
            BlobPublicationInspection(
                publication_id=str(row["publication_id"]),
                blob_hash=bytes(row["blob_hash"]).hex(),
                size_bytes=int(row["size_bytes"]),
                publisher_id=str(row["publisher_id"]),
                reserved_at_ms=int(row["reserved_at_ms"]),
                blob_present=store.exists(bytes(row["blob_hash"]).hex()),
                liveness=_liveness_decision(source_conn, index_conn, bytes(row["blob_hash"])),
            )
            for row in rows
        )
    finally:
        if index_conn is not None:
            index_conn.close()
        source_conn.close()


def reconcile_blob_publication_reservations(
    source_db_path: Path,
    blob_root: Path,
    *,
    index_db_path: Path | None = None,
    writer_exclusion: ArchiveWriterExclusion | None = None,
    max_count: int | None = None,
    after_publication_id: str | None = None,
) -> BlobPublicationReconciliation:
    """Classify receipts; clear safe rows only with archive-wide exclusion."""
    inspections = inspect_blob_publication_receipts(
        source_db_path,
        blob_root,
        index_db_path=index_db_path,
        max_count=max_count,
        after_publication_id=after_publication_id,
    )
    may_clear = (
        writer_exclusion is not None
        and writer_exclusion.source_db_path == source_db_path.resolve()
        and not writer_exclusion._lock_file.closed
    )
    cleared_referenced = 0
    cleared_missing = 0
    retained_referenced = 0
    retained_missing = 0
    unresolved = 0
    retained_blocked = 0
    blockers: list[str] = []
    clear_ids: list[str] = []
    for item in inspections:
        if item.liveness.state is LivenessState.BLOCKED:
            retained_blocked += 1
            blockers.extend(item.liveness.blockers)
        elif item.liveness.state is LivenessState.LIVE:
            # A hash-level owner proves retention, not which publication made
            # it durable. Only the matching reference transaction consumes a
            # reservation, so two same-hash receipts cannot clear each other.
            retained_referenced += 1
        elif not item.blob_present:
            if may_clear:
                clear_ids.append(item.publication_id)
                cleared_missing += 1
            else:
                retained_missing += 1
        else:
            unresolved += 1
    if clear_ids:
        require_write_lease(f"blob publication reconciliation({source_db_path})", archive_root=source_db_path.parent)
        conn = open_source_tier_write_connection(source_db_path, archive_root=source_db_path.parent)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                "DELETE FROM blob_publication_reservations WHERE publication_id = ?",
                ((publication_id,) for publication_id in clear_ids),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return BlobPublicationReconciliation(
        cleared_referenced=cleared_referenced,
        cleared_missing=cleared_missing,
        retained_referenced=retained_referenced,
        retained_missing=retained_missing,
        unresolved=unresolved,
        retained_blocked=retained_blocked,
        blockers=tuple(dict.fromkeys(blockers)),
        scanned=len(inspections),
        last_scanned_publication_id=inspections[-1].publication_id if inspections else None,
    )


def reconcile_blob_publication_reservations_under_exclusion(
    source_db_path: Path,
    blob_root: Path,
    *,
    index_db_path: Path | None = None,
    max_count: int | None = None,
    after_publication_id: str | None = None,
) -> BlobPublicationReconciliation:
    """Reconcile receipts while holding archive-wide publisher exclusion.

    ``reconcile_blob_publication_reservations`` only clears rows when handed
    a live ``ArchiveWriterExclusion`` -- without one, ``may_clear`` is always
    false and every classified row is merely retained forever, a durable
    reservation leak. This entry point acquires the exclusion itself so a
    caller (daemon startup, a maintenance command) cannot silently make
    deletion unreachable by forgetting to pass one (polylogue-qs0a).
    """
    with exclude_archive_blob_publishers(source_db_path) as exclusion:
        return reconcile_blob_publication_reservations(
            source_db_path,
            blob_root,
            index_db_path=index_db_path,
            writer_exclusion=exclusion,
            max_count=max_count,
            after_publication_id=after_publication_id,
        )


def abandon_blob_publication_receipts(
    source_db_path: Path,
    blob_root: Path,
    publication_ids: Sequence[str],
    *,
    confirmed: bool,
    index_db_path: Path | None = None,
) -> BlobPublicationAbandonment:
    """Explicitly abandon selected unreferenced receipts under exclusion."""
    if not confirmed:
        raise ValueError("confirmed=True is required to abandon publication receipts")
    requested = tuple(dict.fromkeys(publication_ids))
    with exclude_archive_blob_publishers(source_db_path):
        # An abandonment is an explicit terminal disposition, not a hash-level
        # reconciliation shortcut. Recheck each requested receipt while source
        # and index writers are excluded in the same order GC uses before its
        # final unlink window.
        from polylogue.storage.archive_identity import ArchiveLocation

        resolved_index = index_db_path or ArchiveLocation.resolve(source_db_path.parent).active_index_path
        require_write_lease(f"blob publication abandonment({source_db_path})", archive_root=source_db_path.parent)
        source_conn = open_source_tier_write_connection(source_db_path, archive_root=source_db_path.parent)
        index_conn: sqlite3.Connection | None = None
        abandoned: list[str] = []
        skipped_referenced = 0
        found = 0
        try:
            source_conn.execute("BEGIN IMMEDIATE")
            if not resolved_index.exists():
                raise RuntimeError("index tier is unavailable")
            assert_population_admitted(resolved_index)
            from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection

            index_conn = open_isolated_write_connection(
                resolved_index, purpose="blob publication liveness fence", archive_root=source_db_path.parent
            )
            index_conn.execute("BEGIN IMMEDIATE")
            rows = source_conn.execute(
                f"SELECT publication_id, blob_hash FROM blob_publication_reservations "
                f"WHERE publication_id IN ({', '.join('?' for _ in requested)})",
                requested,
            ).fetchall()
            found = len(rows)
            for publication_id, blob_hash in rows:
                decision = _liveness_decision(source_conn, index_conn, bytes(blob_hash))
                if decision.state is LivenessState.UNREFERENCED:
                    abandoned.append(str(publication_id))
                else:
                    skipped_referenced += 1
            if abandoned:
                source_conn.executemany(
                    "DELETE FROM blob_publication_reservations WHERE publication_id = ?",
                    ((publication_id,) for publication_id in abandoned),
                )
            source_conn.commit()
        except Exception:
            source_conn.rollback()
            raise
        finally:
            if index_conn is not None:
                if index_conn.in_transaction:
                    index_conn.rollback()
                index_conn.close()
            source_conn.close()
    return BlobPublicationAbandonment(
        abandoned=len(abandoned),
        skipped_referenced=skipped_referenced,
        missing_receipts=len(requested) - found,
    )


__all__ = [
    "AdoptedBlobEvictedError",
    "ArchiveBlobPublisher",
    "ArchiveWriterExclusion",
    "BlobPublicationAbandonment",
    "BlobPublicationInspection",
    "BlobPublicationReceipt",
    "BlobPublicationReconciliation",
    "BlobPublicationReservationStore",
    "abandon_blob_publication_receipts",
    "consume_blob_publication_receipt",
    "exclude_archive_blob_publishers",
    "flush_blob_publications",
    "inspect_blob_publication_receipts",
    "publication_receipt_id",
    "consume_restored_raw_blob_receipts",
    "reconcile_blob_publication_reservations",
    "reconcile_blob_publication_reservations_under_exclusion",
]


class BlobPublicationSourceRead(Protocol):
    """Actual Source inputs used by the existing published-claim proof."""

    def publication_source_path(self) -> Path: ...

    def publication_blob_is_excised(self, blob_hash: bytes) -> bool: ...

    def publication_reservation(self, publication_id: str) -> tuple[bytes, int, str] | None: ...


class RetainedAttachmentSourceRead(BlobPublicationSourceRead, Protocol):
    """Current original Raw and durable acquisition reference proof."""

    def retained_attachment_reference(
        self, raw_id: str, raw_blob_hash: bytes, coordinate: str, blob_hash: bytes, size_bytes: int
    ) -> None: ...


_RETAINED_ATTACHMENT_REFERENCE_SQL = (
    "SELECT 1 FROM raw_sessions r JOIN blob_refs b ON b.ref_id=r.raw_id "
    "WHERE r.raw_id=? AND r.blob_hash=? AND b.ref_type='attachment' "
    "AND coalesce(b.source_path,'')=? AND b.blob_hash=? AND b.size_bytes=? LIMIT 1"
)


def _require_retained_attachment_reference(row: object) -> None:
    if row is None:
        raise ValueError("retained attachment has no exact original Raw acquisition reference")


_PUBLICATION_RESERVATION_SQL = (
    "SELECT blob_hash,size_bytes,publisher_id FROM blob_publication_reservations WHERE publication_id=?"
)


def _publication_reservation_from_row(row: sqlite3.Row | tuple[object, ...] | None) -> tuple[bytes, int, str] | None:
    if row is None:
        return None
    return cast(tuple[bytes, int, str], tuple(row))


class ConnectionBlobPublicationRead:
    """Borrow the ordinary actual Source owner and settle every proof cursor."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def publication_source_path(self) -> Path:
        with connection_cursor(self._connection, "PRAGMA database_list") as rows:
            database_path = next((str(row[2]) for row in rows if row[1] == "main"), "")
        if not database_path:
            raise ValueError("publication requires an actual Source database")
        return Path(database_path).resolve()

    def publication_blob_is_excised(self, blob_hash: bytes) -> bool:
        from polylogue.storage.sqlite.archive_tiers.source_write import is_blob_hash_excised

        return is_blob_hash_excised(self._connection, blob_hash)

    def publication_reservation(self, publication_id: str) -> tuple[bytes, int, str] | None:
        with connection_cursor(self._connection, _PUBLICATION_RESERVATION_SQL, (publication_id,)) as rows:
            return _publication_reservation_from_row(rows.fetchone())

    def retained_attachment_reference(
        self, raw_id: str, raw_blob_hash: bytes, coordinate: str, blob_hash: bytes, size_bytes: int
    ) -> None:
        with connection_cursor(
            self._connection,
            _RETAINED_ATTACHMENT_REFERENCE_SQL,
            (raw_id, raw_blob_hash, coordinate, blob_hash, size_bytes),
        ) as rows:
            _require_retained_attachment_reference(rows.fetchone())


def blob_publication_receipt_delete(
    publication_id: str | None,
    blob_hash: bytes,
    *,
    literal: Callable[[object], tuple[str, tuple[object, ...]]],
) -> tuple[str, tuple[object, ...]] | None:
    """Build the same exact receipt-consumption predicate for either host."""
    if publication_id is None:
        return None
    publication_expression, publication_parameters = literal(publication_id)
    blob_expression, blob_parameters = literal(blob_hash)
    return (
        "DELETE FROM blob_publication_reservations "
        f"WHERE publication_id = {publication_expression} AND blob_hash = {blob_expression}",
        (*publication_parameters, *blob_parameters),
    )


def _publication_receipt_operand(value: object) -> tuple[str, tuple[object, ...]]:
    return "?", (value,)


if TYPE_CHECKING:
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner
    from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit, PreparedIndexMutation

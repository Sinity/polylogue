"""The daemon's cold build as an owned inactive index generation (polylogue-b7dkb).

A cold build is not a second ingest route. It is the ordinary dispatcher-fed
ingest pass writing its index rows into a generation that no reader can open,
so the pass can take the levers an active generation must refuse:

* ``journal_mode=MEMORY``, ``synchronous=OFF``, ``locking_mode=EXCLUSIVE``
  (the bulk-build connection profile) -- a corrupt candidate is discarded,
  never promoted, never read;
* the deferred reader indexes dropped for the whole build and created once at
  the readiness boundary -- a reader-visible schema change that
  ``schema_manifest`` would refuse on the active generation;
* the index page size chosen at creation, which SQLite freezes on the first
  allocated page.

The lifecycle is the existing one (``storage/index_generation.py``): the store
creates the generation, the daemon opens it once per intake page, and one
``promote()`` swaps the ``index.db`` symlink under the lifecycle inode lock.
Readers keep resolving the previous active generation for the whole build; a
crash mid-build leaves an inactive generation that is discarded rather than
recovered, because ``source.db`` and the blob store are durable and the
re-ingest is idempotent by content hash.

The cold build is licensed only while the *active* index generation holds no
sessions. That is also what makes the daemon's other derived-tier owners
(FTS, embeddings, session profiles) safe to leave running: they converge the
active generation, which has nothing in it, and converge the promoted one
afterwards.
"""

from __future__ import annotations

import errno
import json
import os
import sqlite3
import threading
import time
import types
import uuid
import zlib
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import ERROR, WARNING, emit
from polylogue.maintenance.candidate_capacity import (
    InsufficientCapacityError,
    evidence_allocation_block_bytes,
    require_candidate_capacity,
)
from polylogue.storage.archive_identity import MAINTENANCE_STATE_DIRNAME
from polylogue.storage.index_generation import (
    IndexGeneration,
    IndexGenerationStore,
    rebuild_source_evidence_snapshot,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

if TYPE_CHECKING:
    from polylogue.sources.live.production_baseline import BaselineProgress, ProductionSourceBaseline
    from polylogue.sources.live.watcher import WatchSource
    from polylogue.storage.index_generation import PreparedIndexPromotion


_ACCEPTED_PROGRESS_STALL_AFTER_S = 60.0


def is_transient_cold_storage_errno(error_no: int | None) -> bool:
    """I/O errors eligible for a same-process cold-settlement retry."""
    return error_no in {errno.EAGAIN, errno.EBUSY, errno.EIO, errno.ESTALE, errno.ETIMEDOUT}


def _typed_promotion_io_failure(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, sqlite3.Error):
            code = getattr(current, "sqlite_errorcode", None)
            if isinstance(code, int) and code & 0xFF in {
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
                sqlite3.SQLITE_FULL,
                sqlite3.SQLITE_IOERR,
                sqlite3.SQLITE_CORRUPT,
                sqlite3.SQLITE_NOTADB,
            }:
                return True
        # The pointer is already published and reconcile_promoted() succeeded:
        # capacity and access faults can be swallowed here as well as the
        # transient pre-swap retry errors.
        if isinstance(current, OSError) and (
            is_transient_cold_storage_errno(current.errno)
            or current.errno in {errno.ENOSPC, errno.EDQUOT, errno.ENOENT, errno.EACCES}
        ):
            return True
        current = current.__cause__
    return False


__all__ = [
    "ColdBuildGeneration",
    "active_cold_build_generation",
    "active_index_generation_is_empty",
    "clear_cold_build_generation",
    "is_transient_cold_storage_errno",
    "register_cold_build_generation",
]


def _cold_build_owner_id() -> str:
    """One owner per daemon process: ownership is what promotion checks."""
    return f"cold-build:{os.getpid()}"


def _reclaim_abandoned_cold_generations(store: IndexGenerationStore) -> None:
    """Remove inactive cold builds left by a stopped daemon before capacity admission."""
    live = active_cold_build_generation(store.archive_root)
    if live is not None and not live.settled:
        raise RuntimeError("cannot reclaim cold builds while a generation is registered")
    ColdBuildGeneration.reconcile_interrupted_promotions(store.archive_root)
    if not store.generations_root.exists():
        return
    active_target = store.active_pointer.resolve(strict=False)
    for root in sorted(store.generations_root.iterdir()):
        if not root.name.startswith("gen-"):
            continue
        generation = store.load(root.name)
        if not generation.owner_id.startswith("cold-build:"):
            continue
        if generation.state == "promoting":
            if store.discard_unpublished_promotion(generation):
                emit(
                    "daemon.cold_build.abandoned_reclaimed",
                    outcome="ok",
                    reason="interrupted_pre_swap",
                    generation_id=generation.generation_id,
                )
            continue
        if generation.state != "inactive":
            continue
        if Path(generation.index_path).resolve(strict=False) == active_target:
            raise RuntimeError("inactive cold-build generation is the active index")
        if store.discard_if_inactive(generation):
            emit(
                "daemon.cold_build.abandoned_reclaimed",
                outcome="ok",
                reason="inactive_previous_build",
                generation_id=generation.generation_id,
            )


def _hold_ops_checkpoints(archive_root: Path) -> sqlite3.Connection | None:
    """Hold one ``ops.db`` handle open for a cold-build pass.

    Measured defect (polylogue-rk0it, synthetic 16-file cold-build page):
    ``CursorStore._connect_ops`` opens a one-shot ``ops.db`` connection for
    every cursor, convergence-debt and stage-event write that falls outside
    the pass' ``ops_write_scope`` -- 94 open/commit/close cycles for 16 files.
    Every one of those closes is the *last* connection to the file, so SQLite
    runs its close-time checkpoint: fsync the WAL, copy it into ``ops.db``,
    fsync that, delete the WAL, fsync the directory. ``strace`` counted 111
    ``fdatasync`` calls on ``ops.db``/``ops.db-wal`` plus 39 on the archive
    directory for eight ingested files, against six on ``source.db``; and
    ``sqlite3.Connection.commit`` alone was 0.586 s of a 2.195 s ingest
    window, all of it on ``ops.db``. A profiled open/commit/close cycle costs
    ~14-20 ms; the same commit with any second handle attached costs ~0.3 ms.

    Holding a handle makes those closes stop being the last one, so the
    checkpoint never fires there. Nothing about the *writes* changes: same
    statements, same per-operation commits, same ``WRITE_CONNECTION_PROFILE``
    (WAL, ``synchronous=NORMAL``, 30 s busy timeout) that ``_connect_ops``
    already opens with -- which is why the holder is opened through exactly
    that factory rather than a read handle. A read-only handle was tried and
    is *not* equivalent: ``ops.db`` is still in rollback-journal mode when a
    cold build begins, and a reader attached before the first writer's
    ``journal_mode=WAL`` transition holds nothing afterwards.

    The holder never opens a transaction, so it takes no lock and blocks no
    checkpoint, no reader and no other writer.

    Durability, stated exactly (rk0it AC3):

    * ``ops`` is the disposable tier. It carries ingest cursors, convergence
      debt and stage telemetry -- all reconstructible, all re-derived by
      re-ingesting the file, which is idempotent by content hash.
    * Process crash: unchanged. The commits are in ``ops.db-wal``, a durable
      file the next open replays. Deferring the *checkpoint* does not defer
      the commit.
    * Power loss: this is the window that widens, and it is the only one.
      At ``synchronous=NORMAL`` a WAL commit is not fsynced, so today's
      per-write close-time checkpoint is what happened to make each ops write
      power-loss durable. With the holder they become power-loss durable at
      the next checkpoint: the daemon's recurring coordinator, which runs
      every 300 s (``daemon/cli.py::_WAL_CHECKPOINT_INTERVAL_SECONDS``) over
      every archive tier including ``ops.db``, or the holder's own release
      when the generation settles.
    * What a loss costs: up to one checkpoint interval of cursor advances and
      convergence debt. The recovery is to re-ingest those files, which is
      what an un-advanced cursor already means.
    * Why this direction is the safe one: the hazard an ingest cursor can
      actually cause is surviving a source write that did not -- a cursor
      more durable than the durable tier it certifies. ``source.db`` is
      WAL/``NORMAL`` too and is checkpointed at that same recurring boundary,
      so the shape this replaces (ops flushed per write, source flushed per
      coordinator tick) is the inverted one. Moving ops' flush boundary onto
      source's tick removes an inversion rather than adding one.
    * Boundary: cold build only, released the moment the generation settles.
      It is not a change to the live archive's ops policy, and it never
      touches ``source``, ``user`` or ``audit`` -- the tiers a rebuild cannot
      reconstruct.

    Failures are not swallowed. Opening ``ops.db`` for writing under the lease
    the caller already holds is the same thing every cursor write in the pass
    is about to do, so an error here is the pass failing early rather than a
    performance property quietly not applying.
    """
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import open_connection

    ops_db = archive_root / "ops.db"
    if not ops_db.exists():
        # Not a degraded open: an archive root with no ops tier has no
        # close-time churn to suppress, and creating one here would leave an
        # uninitialized tier behind whatever the build does next.
        return None
    # ``validate_schema=False`` deliberately: this handle reads no rows.
    # Coupling a checkpoint-suppression holder to tier-schema validation would
    # invent a new way for a build to refuse.
    conn = open_connection(
        ops_db,
        tier=ArchiveTier.OPS,
        validate_schema=False,
        archive_root=archive_root,
        # The holder outlives the thread that opened it: passes run on the
        # write coordinator's worker threads and the settle that releases it
        # runs on another. The handle is only ever touched under the
        # single-writer lease, so this is a lifetime statement, not concurrent
        # use. Leaving the thread check on made every cross-thread pass close
        # the holder and open a new one -- a leak per thread, and no
        # suppression while the replacement was being built.
        check_same_thread=False,
    )
    if _ops_holder_is_attached(conn, ops_db):
        return conn
    # The handle is open but attached to nothing, so it suppresses nothing.
    # Keeping it would be a connection with no purpose.
    conn.close()
    return None


def _ops_holder_is_attached(conn: sqlite3.Connection, ops_db: Path) -> bool:
    """Whether this handle actually suppresses the close-time checkpoint.

    Measured, not assumed, and the obvious proxies are all wrong. Holding a
    connection object is not enough, and neither is reading ``journal_mode``
    back as ``wal``: SQLite opens the database file and attaches the WAL index
    lazily, so a handle that has only run PRAGMAs leaves ``ops.db-shm`` absent
    and the one-shot writers keep checkpointing at ~20 ms/cycle. One real read
    attaches it and the same cycle costs ~0.3 ms.

    So this runs the read and then checks the only direct evidence there is:
    the WAL index exists. The read is fully fetched, leaving no open
    transaction -- a held read mark would block the recurring coordinator's
    PASSIVE checkpoint, which is the boundary the durability argument leans on.

    ``False`` means "this handle is attached to nothing", which is an ordinary
    answer on a tier still in rollback-journal mode. A read that *fails* is
    not that answer and is not caught here.
    """
    conn.execute("SELECT count(*) FROM sqlite_schema").fetchall()
    physical_ops_db = ops_db.resolve()
    return physical_ops_db.with_name(f"{physical_ops_db.name}-shm").exists()


def active_index_generation_is_empty(archive_root: Path) -> bool:
    """Whether the archive's currently active index generation holds no sessions.

    Bootstraps the durable tiers as a side effect, exactly as any first
    writable open does. That bootstrap is a precondition for the cold build,
    not an accident: a generation created before ``source.db`` exists gets no
    read-through symlink for it and would quietly grow a second durable tier
    inside the candidate directory.
    """
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        return archive.index_generation_empty_at_open


@dataclass(slots=True)
class ColdBuildGeneration:
    """One owned inactive generation being filled by ordinary ingest passes."""

    archive_root: Path
    generation: IndexGeneration
    reason: str
    operation_id: str
    _store: IndexGenerationStore
    source_baseline: ProductionSourceBaseline
    _promoted: bool = False
    _discarded: bool = False
    _receipt_cleared: bool = False
    _promotion_candidate_ready: bool = field(default=False, init=False, repr=False)
    settlement_state: str = "building"
    settlement_reason: str | None = None
    settlement_last_error: str | None = None
    settlement_attempts: int = 0
    settlement_next_retry_at: float | None = None
    #: Open for the build's lifetime so one-shot ``ops.db`` writers stop
    #: checkpointing on every close. See :func:`_hold_ops_checkpoints`.
    _ops_checkpoint_holder: sqlite3.Connection | None = None
    #: Open intake pages (:meth:`begin_ops_page`). While one is open, closing
    #: an archive pass keeps the holder, because the page's cursor,
    #: convergence and attempt writes still follow it.
    _ops_page_depth: int = 0
    # Disposable, generation-scoped projection over the candidate application
    # receipts. The candidate index and durable source rows remain authority.
    _accepted_progress_weights: dict[tuple[str, int, str], int] = field(init=False, repr=False)
    _accepted_progress_total: int = field(init=False, repr=False)
    _accepted_progress_seen: set[tuple[str, int, str]] = field(default_factory=set, init=False, repr=False)
    _accepted_progress_count: int = field(default=0, init=False, repr=False)
    _accepted_progress_rowid: int = field(default=0, init=False, repr=False)
    _accepted_progress_index_identity: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _accepted_progress_started_at: float = field(default_factory=time.monotonic, init=False, repr=False)
    _accepted_progress_last_observed_at: float | None = field(default=None, init=False, repr=False)
    _accepted_progress_last_advanced_at: float | None = field(default=None, init=False, repr=False)
    _accepted_progress_denominator_sealed: bool = field(default=False, init=False, repr=False)
    _accepted_progress_valid: bool = field(default=False, init=False, repr=False)
    _accepted_progress_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reset_accepted_progress(self.source_baseline)

    def _reset_accepted_progress(self, baseline: ProductionSourceBaseline) -> int:
        accepted = baseline.accepted
        weights: dict[tuple[str, int, str], int] = {}
        for row in accepted:
            if row.revision is None:
                continue
            key = (row.path, row.source_index or 0, row.revision)
            weights[key] = weights.get(key, 0) + 1
        with self._accepted_progress_lock:
            previous_count = self._accepted_progress_count
            self._accepted_progress_total = len(accepted)
            self._accepted_progress_denominator_sealed = bool(baseline.digest)
            self._accepted_progress_weights = weights
            self._accepted_progress_seen.clear()
            self._accepted_progress_count = 0
            self._accepted_progress_rowid = 0
            self._accepted_progress_index_identity = None
            self._accepted_progress_valid = False
        return previous_count

    @property
    def accepted_progress(self) -> tuple[int | None, int, float | None, float | None]:
        """Matching applied revisions, sealed denominator, lifetime rate, and ETA.

        The rate is cumulative from build start. A sealed complete count has
        zero ETA. Otherwise ETA requires recent count advancement; an
        observation alone never renews its clock.
        A warm status call only reads this cached projection. An intake pass
        advances it from candidate receipts after its writer has closed.
        """
        with self._accepted_progress_lock:
            denominator = self._accepted_progress_total
            observed_at = time.monotonic()
            self._accepted_progress_last_observed_at = observed_at
            if not self._accepted_progress_valid:
                return None, denominator, None, None
            count = self._accepted_progress_count
            elapsed = observed_at - self._accepted_progress_started_at
            rate = count / elapsed if count > 0 and elapsed > 0 else None
            last_advanced_at = self._accepted_progress_last_advanced_at
            advancing = (
                last_advanced_at is not None and observed_at - last_advanced_at <= _ACCEPTED_PROGRESS_STALL_AFTER_S
            )
            eta = (
                0.0
                if self._accepted_progress_denominator_sealed and count >= denominator
                else (
                    (denominator - count) / rate
                    if self._accepted_progress_denominator_sealed and advancing and rate is not None
                    else None
                )
            )
            return count, denominator, rate, eta

    def invalidate_accepted_progress(self) -> None:
        """Refuse a stale ETA after a failed projection refresh."""
        with self._accepted_progress_lock:
            self._accepted_progress_valid = False

    def refresh_accepted_progress(self) -> None:
        """Reconcile newly applied candidate rows against accepted source revisions.

        Run after a completed intake pass, never on the status read path. The
        rowid cursor bounds work to new applications; the seen set prevents
        multiple session decisions for one raw revision from inflating progress.
        """
        if self.settled:
            return
        try:
            index_path = Path(self.generation.index_path)
            index_stat = index_path.stat()
            identity = (index_stat.st_dev, index_stat.st_ino)
            # The candidate deliberately has deferred indexes while under
            # construction, so its final schema identity is not yet valid.
            with closing(
                open_readonly_connection(
                    index_path,
                    tier=ArchiveTier.INDEX,
                    validate_schema=False,
                    timeout_class="background-read",
                )
            ) as candidate:
                candidate.execute("BEGIN")
                head_row = candidate.execute("SELECT MAX(rowid) FROM raw_revision_applications").fetchone()
                head = int(head_row[0]) if head_row is not None and head_row[0] is not None else 0
                with self._accepted_progress_lock:
                    cursor = self._accepted_progress_rowid
                    if head < cursor or self._accepted_progress_index_identity != identity:
                        self._accepted_progress_rowid = cursor = 0
                        self._accepted_progress_seen.clear()
                        self._accepted_progress_count = 0
                rows = candidate.execute(
                    "SELECT rowid, raw_id FROM raw_revision_applications "
                    "WHERE rowid > ? AND rowid <= ? AND decision IN "
                    "('selected_baseline', 'reparse_reaffirmation', 'applied_append', 'superseded') "
                    "ORDER BY rowid",
                    (cursor, head),
                ).fetchall()
            # source.db is durable and must pass its normal schema check.
            with closing(
                open_readonly_connection(
                    self.archive_root / "source.db",
                    tier=ArchiveTier.SOURCE,
                    timeout_class="background-read",
                )
            ) as source:
                resolved: dict[str, tuple[str, int, str] | None] = {}
                matched: set[tuple[str, int, str]] = set()
                for _, raw_id_value in rows:
                    raw_id = str(raw_id_value)
                    key = resolved.get(raw_id)
                    if raw_id not in resolved:
                        raw_row = source.execute(
                            "SELECT source_path, source_index, lower(hex(blob_hash)) "
                            "FROM raw_sessions WHERE raw_id = ?",
                            (raw_id,),
                        ).fetchone()
                        key = (str(raw_row[0]), int(raw_row[1]), str(raw_row[2])) if raw_row else None
                        resolved[raw_id] = key
                    if key in self._accepted_progress_weights:
                        matched.add(key)
            with self._accepted_progress_lock:
                new_keys = matched.difference(self._accepted_progress_seen)
                self._accepted_progress_seen.update(new_keys)
                count_advance = sum(self._accepted_progress_weights[key] for key in new_keys)
                self._accepted_progress_count += count_advance
                self._accepted_progress_rowid = head
                self._accepted_progress_index_identity = identity
                observed_at = time.monotonic()
                self._accepted_progress_last_observed_at = observed_at
                if count_advance:
                    self._accepted_progress_last_advanced_at = observed_at
                self._accepted_progress_valid = True
        except (OSError, sqlite3.Error) as exc:
            self.invalidate_accepted_progress()
            emit(
                "daemon.cold_build.accepted_progress_unreadable",
                level=WARNING,
                outcome="degraded",
                reason="accepted_progress_unreadable",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )

    @staticmethod
    def observe_source_baseline(
        sources: tuple[WatchSource, ...],
        *,
        progress: BaselineProgress | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ProductionSourceBaseline:
        """Capture the production discovery denominator a cold build will bind.

        The walk, pre-acquisition classification and revision hashing read
        source files only and write nothing, so they run before, and outside,
        the writer call :meth:`begin` makes. A slow or large source then
        never holds the archive's single writer.
        """
        from polylogue.sources.live.production_baseline import capture_production_source_baseline

        if progress is not None:
            progress("baseline_walk")
        return capture_production_source_baseline(
            sources,
            operation_id=f"cold-build-{uuid.uuid4().hex}",
            cancelled=cancelled,
            progress=progress,
        )

    @classmethod
    def begin(
        cls,
        archive_root: Path,
        *,
        reason: str,
        observed: ProductionSourceBaseline,
        owner_id: str | None = None,
        progress: BaselineProgress | None = None,
    ) -> ColdBuildGeneration:
        """Create the inactive generation this build will fill.

        ``observed`` is the baseline :meth:`observe_source_baseline` captured
        off the writer; this writer call merges it with a pending receipt,
        publishes it, and binds it. ``progress`` hears each remaining phase
        (capacity projection, source snapshot, generation creation) as it
        starts, so the caller can report the time before the first intake
        page instead of an idle status.

        Checks free space before the generation directory exists. A cold
        build is the whole index again on disk beside the one still serving
        reads, and it is engaged automatically whenever the active generation
        is empty -- so this preflight cannot be conditioned on the operator
        having asked for it, or the unattended 40 GB case would be the one
        left unguarded. The refusal is fatal on purpose: there is no smaller
        build to fall back to, and letting ingest fill the active generation
        instead would allocate the same bytes with readers attached.
        """
        archive_root = Path(archive_root)
        # Every durable member must already exist: ``create`` links exactly
        # the members it finds, and the candidate open then refuses a member
        # that appeared afterwards ("invalid read-through target"). The blob
        # directory in particular is created lazily by the first publication,
        # which on a fresh root happens during the build.
        with ArchiveStore.open_existing(archive_root, read_only=False):
            pass
        (archive_root / "blob").mkdir(mode=0o700, exist_ok=True)
        store = IndexGenerationStore.for_archive_root(archive_root)
        _reclaim_abandoned_cold_generations(store)
        operation_id = observed.operation_id
        from polylogue.sources.live.production_baseline import (
            MATERIAL_BYTE_DEFINITION,
            load_pending_production_baseline,
            merge_pending_production_baseline,
            publish_pending_production_baseline,
            unretained_source_material,
        )

        def phase(name: str) -> None:
            if progress is not None:
                progress(name)

        baseline = merge_pending_production_baseline(observed, load_pending_production_baseline(archive_root))
        publish_pending_production_baseline(archive_root, baseline)
        phase("capacity_projection")
        blob_block_bytes, source_db_block_bytes = evidence_allocation_block_bytes(archive_root)
        prospective_material_bytes, prospective_retained_allocation_bytes, prospective_source_db_allocation_bytes = (
            unretained_source_material(baseline, archive_root / "source.db", blob_block_bytes, source_db_block_bytes)
        )
        try:
            require_candidate_capacity(
                archive_root,
                operation_id=operation_id,
                prospective_material_bytes=prospective_material_bytes,
                prospective_retained_allocation_bytes=prospective_retained_allocation_bytes,
                prospective_source_db_allocation_bytes=prospective_source_db_allocation_bytes,
                prospective_generation_baseline_bytes=(
                    (
                        len(
                            json.dumps(
                                {"generation_id": "gen-placeholder", "baseline": baseline.as_dict()},
                                indent=2,
                                sort_keys=True,
                            ).encode()
                        )
                        + max(blob_block_bytes, source_db_block_bytes)
                        - 1
                    )
                    // max(blob_block_bytes, source_db_block_bytes)
                    * max(blob_block_bytes, source_db_block_bytes)
                ),
                baseline_digest=baseline.digest,
                material_byte_definition=MATERIAL_BYTE_DEFINITION,
            )
        except InsufficientCapacityError as refusal:
            # The projection's numbers ride in ``error_detail`` rather than in
            # named fields: ``logging_fields`` registers no byte-sized capacity
            # field, and an unregistered field is dropped at the emit boundary
            # rather than recorded. The persisted receipt under
            # ``.maintenance-state`` is not written on a refusal either, so
            # this event is the only place the shortfall is stated.
            emit(
                "daemon.cold_build.capacity_refused",
                level=ERROR,
                outcome="error",
                reason=reason,
                operation_id=operation_id,
                error_type=type(refusal).__name__,
                error_detail=str(refusal),
            )
            raise
        snapshot = ""
        if (archive_root / "source.db").exists():
            phase("source_snapshot")
            snapshot = rebuild_source_evidence_snapshot(archive_root)
        phase("generation_create")
        generation = store.create(owner_id=owner_id or _cold_build_owner_id(), source_snapshot=snapshot)
        baseline_path = Path(generation.index_path).parent / "source-baseline.json"
        with baseline_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {"generation_id": generation.generation_id, "baseline": baseline.as_dict()},
                stream,
                indent=2,
                sort_keys=True,
            )
        emit(
            "daemon.cold_build.generation_created",
            outcome="ok",
            generation_id=generation.generation_id,
            owner_id=generation.owner_id,
            # ``limit`` is the declared count field for a page bound.
            limit=generation.page_size,
            reason=reason,
            operation_id=operation_id,
        )
        return cls(
            archive_root=archive_root,
            generation=generation,
            reason=reason,
            operation_id=operation_id,
            _store=store,
            source_baseline=baseline,
        )

    @classmethod
    def reconcile_interrupted_promotions(cls, archive_root: Path) -> None:
        """Finish a cold pointer swap or matching pending receipt left by process exit."""
        from polylogue.sources.live.production_baseline import (
            ProductionBaselineError,
            ProductionSourceBaseline,
            load_pending_production_baseline,
        )

        store = IndexGenerationStore.for_archive_root(archive_root)
        if not store.generations_root.exists():
            return
        active_target = store.active_pointer.resolve(strict=False)
        for root in sorted(store.generations_root.iterdir()):
            if not root.name.startswith("gen-"):
                continue
            generation = store.load(root.name)
            if generation.state not in {"promoting", "active"} or not generation.owner_id.startswith("cold-build:"):
                continue
            if Path(generation.index_path).resolve(strict=False) != active_target:
                continue
            pending = load_pending_production_baseline(Path(archive_root)) if generation.state == "active" else None
            if generation.state == "active" and pending is None:
                continue
            binding = root / "source-baseline.json"
            try:
                fd = os.open(binding, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, encoding="utf-8") as stream:
                    payload = json.load(stream)
                if not isinstance(payload, dict) or payload.get("generation_id") != generation.generation_id:
                    raise ProductionBaselineError("interrupted cold-build baseline binding mismatch")
                baseline_payload = payload.get("baseline")
                if not isinstance(baseline_payload, dict):
                    raise ProductionBaselineError("interrupted cold-build baseline binding is unavailable")
                baseline = ProductionSourceBaseline.from_dict(baseline_payload)
                if payload != {"generation_id": generation.generation_id, "baseline": baseline.as_dict()}:
                    raise ProductionBaselineError("interrupted cold-build baseline binding changed")
            except (OSError, ValueError, TypeError) as exc:
                raise ProductionBaselineError("interrupted cold-build baseline binding is unavailable") from exc
            if generation.state == "active" and pending is not None and pending.digest != baseline.digest:
                continue  # a newer inactive build owns the pending receipt
            candidate = cls(
                archive_root=Path(archive_root),
                generation=generation,
                reason="interrupted_promotion",
                operation_id=baseline.operation_id,
                _store=store,
                source_baseline=baseline,
                _promoted=True,
            )
            candidate.reconcile_promoted()
            emit(
                "daemon.cold_build.interrupted_promotion_reconciled",
                outcome="ok",
                generation_id=generation.generation_id,
            )

    @property
    def generation_root(self) -> Path:
        return Path(self.generation.index_path).parent

    @property
    def generation_id(self) -> str:
        return self.generation.generation_id

    @property
    def settled(self) -> bool:
        """Whether this generation has already been promoted or discarded."""
        return self._promoted or self._discarded

    @property
    def publication_complete(self) -> bool:
        return self._promoted and self._receipt_cleared

    @property
    def promoted(self) -> bool:
        return self._promoted

    @property
    def discarded(self) -> bool:
        return self._discarded

    def record_settlement(
        self,
        state: str,
        *,
        reason: str | None = None,
        last_error: str | None = None,
        next_retry_at: float | None = None,
        attempted: bool = True,
    ) -> None:
        if attempted:
            self.settlement_attempts += 1
        self.settlement_state = state
        self.settlement_reason = reason
        self.settlement_last_error = last_error
        self.settlement_next_retry_at = next_retry_at

    def settlement_external_revision(self, sources: tuple[WatchSource, ...] = ()) -> tuple[int, ...]:
        """Evidence a settlement callback cannot change itself."""

        def unavailable(error: OSError) -> tuple[int, int, int]:
            error_type = f"{type(error).__module__}.{type(error).__qualname__}"
            return (-2, error.errno if error.errno is not None else -1, zlib.crc32(error_type.encode()))

        paths = (
            self.archive_root / "source.db",
            self.archive_root / "source.db-wal",
        )
        revision: list[int] = []
        for path in paths:
            try:
                metadata = path.stat()
            except FileNotFoundError:
                revision.extend((-1, -1, -1))
            except OSError as exc:
                revision.extend(unavailable(exc))
            else:
                revision.extend((metadata.st_ino, metadata.st_size, metadata.st_mtime_ns))
        for source in sources:
            try:
                metadata = source.root.stat()
            except FileNotFoundError:
                revision.extend((-1, -1, -1, -1))
            except OSError as exc:
                revision.extend((*unavailable(exc), -1))
            else:
                revision.extend((metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_mode))
        # Pointer publication can fail on archive-root permissions. Its ctime
        # also changes during ordinary daemon writes, so only watch identity
        # and mode here to avoid retrying an unchanged blocked settlement.
        try:
            root_metadata = self.archive_root.stat()
        except FileNotFoundError:
            revision.extend((-1, -1, -1))
        except OSError as exc:
            revision.extend(unavailable(exc))
        else:
            revision.extend((0, root_metadata.st_ino, root_metadata.st_mode))
        # Receipt publication/unlink can fail on parent permissions even when
        # the child file's own metadata does not move. Watch identity and mode;
        # settlement's own file writes also move directory ctime and mtime.
        for directory in (
            self.archive_root / MAINTENANCE_STATE_DIRNAME / "production-source-baseline",
            self.generation_root,
        ):
            try:
                metadata = directory.stat()
            except FileNotFoundError:
                revision.extend((-1, -1, -1))
            except OSError as exc:
                revision.extend(unavailable(exc))
            else:
                revision.extend((0, metadata.st_ino, metadata.st_mode))
        # Baseline rebinding may replace this file itself. Only its access mode
        # is external permission evidence; watching its inode would mistake
        # our own publication for an external repair during the callback.
        try:
            binding_metadata = (self.generation_root / "source-baseline.json").stat()
        except FileNotFoundError:
            revision.extend((-1, -1, -1))
        except OSError as exc:
            revision.extend(unavailable(exc))
        else:
            revision.extend((0, binding_metadata.st_mode, 0))
        if self.settlement_reason == "capacity_unavailable":
            # Capacity admission accounts for candidate, blob, source DB and
            # receipt destinations, which can live on different filesystems.
            for destination in (
                self.archive_root,
                self.generation_root,
                self.archive_root / "blob",
                self.archive_root / "source.db",
                self.archive_root / MAINTENANCE_STATE_DIRNAME,
            ):
                try:
                    space = os.statvfs(destination)
                except OSError as exc:
                    revision.extend(unavailable(exc))
                else:
                    revision.extend((0, space.f_bavail * space.f_frsize, 0))
        return tuple(revision)

    def settlement_evidence_revision(self, sources: tuple[WatchSource, ...] = ()) -> tuple[int, ...]:
        """Cheap external and candidate hints for retrying a blocked verdict."""
        revision = list(self.settlement_external_revision(sources))
        for path in (
            self.archive_root / MAINTENANCE_STATE_DIRNAME / "production-source-baseline" / "pending.json",
            Path(self.generation.index_path),
            self.generation_root / "generation.json",
            self.generation_root / "source-baseline.json",
        ):
            try:
                metadata = path.stat()
            except FileNotFoundError:
                revision.extend((-1, -1, -1, -1))
            except OSError as exc:
                error_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
                revision.extend((-2, exc.errno if exc.errno is not None else -1, zlib.crc32(error_type.encode()), -1))
            else:
                revision.extend((metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_mode))
        return tuple(revision)

    def observe_faulted_baseline(
        self, sources: tuple[WatchSource, ...], *, cancel: threading.Event | None = None
    ) -> ProductionSourceBaseline | None:
        """Recapture source bytes off the writer worker before binding them."""
        if not any(row.disposition == "fault" for row in self.source_baseline.decisions):
            return None
        from polylogue.sources.live.production_baseline import capture_production_source_baseline

        return capture_production_source_baseline(
            sources, operation_id=self.operation_id, cancelled=cancel.is_set if cancel is not None else None
        )

    def refresh_faulted_baseline(self, observed: ProductionSourceBaseline) -> bool:
        """Replace a faulted observation while retaining its accepted revisions.

        The old observation is immutable. A new one can resolve a discovery
        fault, but the pending receipt, capacity admission and generation
        binding must all name the new merged denominator before promotion.
        """
        if not any(row.disposition == "fault" for row in self.source_baseline.decisions):
            return False
        import json

        from polylogue.core.durable_fs import atomic_replace
        from polylogue.sources.live.production_baseline import (
            MATERIAL_BYTE_DEFINITION,
            ProductionBaselineError,
            load_pending_production_baseline,
            merge_pending_production_baseline,
            publish_pending_production_baseline,
            unretained_source_material,
        )

        if observed.source_signature != self.source_baseline.source_signature:
            raise ProductionBaselineError("cold-build source declaration changed during settlement")
        merged = merge_pending_production_baseline(observed, self.source_baseline)
        merged = merge_pending_production_baseline(merged, load_pending_production_baseline(self.archive_root))
        if merged.digest == self.source_baseline.digest:
            return False
        blob_block_bytes, source_db_block_bytes = evidence_allocation_block_bytes(self.archive_root)
        prospective_material_bytes, prospective_retained_allocation_bytes, prospective_source_db_allocation_bytes = (
            unretained_source_material(merged, self.archive_root / "source.db", blob_block_bytes, source_db_block_bytes)
        )
        require_candidate_capacity(
            self.archive_root,
            operation_id=self.operation_id,
            existing_candidate_generation_id=self.generation_id,
            prospective_material_bytes=prospective_material_bytes,
            prospective_retained_allocation_bytes=prospective_retained_allocation_bytes,
            prospective_source_db_allocation_bytes=prospective_source_db_allocation_bytes,
            prospective_generation_baseline_bytes=(
                (
                    len(
                        json.dumps(
                            {"generation_id": self.generation_id, "baseline": merged.as_dict()},
                            indent=2,
                            sort_keys=True,
                        ).encode()
                    )
                    + max(blob_block_bytes, source_db_block_bytes)
                    - 1
                )
                // max(blob_block_bytes, source_db_block_bytes)
                * max(blob_block_bytes, source_db_block_bytes)
            ),
            baseline_digest=merged.digest,
            material_byte_definition=MATERIAL_BYTE_DEFINITION,
        )
        publish_pending_production_baseline(self.archive_root, merged)
        atomic_replace(
            self.generation_root / "source-baseline.json",
            json.dumps(
                {"generation_id": self.generation_id, "baseline": merged.as_dict()}, indent=2, sort_keys=True
            ).encode(),
        )
        self.source_baseline = merged
        previous_count = self._reset_accepted_progress(merged)
        with self._accepted_progress_lock:
            previous_advance_at = self._accepted_progress_last_advanced_at
        self.refresh_accepted_progress()
        with self._accepted_progress_lock:
            if self._accepted_progress_valid and self._accepted_progress_count <= previous_count:
                self._accepted_progress_last_advanced_at = previous_advance_at
        return True

    def open_writer(self) -> ArchiveStore:
        """Open one ingest pass against the candidate.

        Every pass re-opens: the dispatcher hands the daemon one intake page
        at a time, and the connection-scoped bulk pragmas cost nothing to
        re-apply. The generation itself is what persists across passes.
        """
        if self.settled:
            raise RuntimeError(f"cold-build generation {self.generation_id} is no longer writable")
        self._retain_ops_checkpoints()
        try:
            self._store.restore_unpublished_promotion(self.generation_id)
            archive = ArchiveStore.open_cold_build_generation(
                self.generation_root,
                generation_id=self.generation_id,
                owner_id=self.generation.owner_id,
                defer_secondary_indexes=True,
            )
        except BaseException:
            # The holder is acquired before the candidate open so that every
            # pass gets the same checkpoint policy.  If opening the candidate
            # fails, however, there is no pass left to own that handle; keep a
            # failed open from pinning ops WAL frames until the whole build is
            # settled (or leaking across a retry).
            self._release_ops_checkpoint_holder()
            raise

        # Tie the checkpoint holder to the intake page: retaining it across
        # pages would silently widen the ops power-loss window to the whole
        # build. Inside an open page (:meth:`begin_ops_page`) the archive pass
        # ends before the page's cursor, convergence and attempt writes, so
        # the page end releases it; a pass outside any page releases it on
        # close. ``ArchiveStore`` is intentionally not changed for this
        # cold-build concern; binding the existing close method preserves its
        # public type and all normal close/rollback behavior.
        close = archive.close

        def close_page(_archive: ArchiveStore) -> None:
            try:
                close()
            finally:
                if self._ops_page_depth == 0:
                    self._release_ops_checkpoint_holder()

        # Rebinding close on the instance (not the class) so the ops checkpoint
        # holder is released on whichever path closes this pass.
        archive.close = types.MethodType(close_page, archive)  # type: ignore[method-assign]
        return archive

    def begin_ops_page(self) -> None:
        """Keep the ops checkpoint holder until :meth:`end_ops_page`.

        The holder is still acquired by the page's first :meth:`open_writer`;
        this only moves its release from that pass's close to the page end.
        """
        self._ops_page_depth += 1

    def end_ops_page(self) -> None:
        """Close one intake page; the outermost one releases the holder."""
        if self._ops_page_depth <= 0:
            raise RuntimeError("cold-build ops page ended without a matching begin")
        self._ops_page_depth -= 1
        if self._ops_page_depth == 0:
            self._release_ops_checkpoint_holder()

    def session_count(self) -> int:
        """How many sessions the build has materialized so far."""
        with self.open_writer() as archive:
            row = archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0]) if row is not None else 0

    def _retain_ops_checkpoints(self) -> None:
        """Hold ``ops.db`` open for this pass so its writers stop checkpointing.

        Established here rather than in :meth:`begin` because the suppression
        is an *attachment*, not an open: it is asserted per pass, which is the
        dispatcher page the batching is bounded by, and re-asserted if a
        journal-mode transition ever dropped it.
        """
        holder = self._ops_checkpoint_holder
        if holder is not None and _ops_holder_is_attached(holder, self.archive_root / "ops.db"):
            return
        self._release_ops_checkpoint_holder()
        self._ops_checkpoint_holder = _hold_ops_checkpoints(self.archive_root)

    def _release_ops_checkpoint_holder(self) -> None:
        """Give ``ops.db`` its close-time checkpoint back at the build boundary.

        Closing the holder makes the next one-shot ops writer the last
        connection again, so the deferred WAL is checkpointed on its close --
        the build does not hand a settled archive a WAL it never drains.

        A pass and settlement can use different write-coordinator worker
        threads, so the handle is opened without the same-thread check.
        """
        holder = self._ops_checkpoint_holder
        if holder is None:
            return
        self._ops_checkpoint_holder = None
        holder.close()

    def prepare_promotion_candidate(self) -> None:
        """Run the final candidate writes before off-gate promotion proof."""
        if self._promoted:
            return
        if self._discarded:
            raise RuntimeError(f"cold-build generation {self.generation_id} is already settled")
        import json

        from polylogue.sources.live.production_baseline import (
            ProductionBaselineError,
        )

        receipt = json.loads((self.generation_root / "source-baseline.json").read_text(encoding="utf-8"))
        if receipt != {"generation_id": self.generation_id, "baseline": self.source_baseline.as_dict()}:
            raise ProductionBaselineError("production source baseline is not bound to this generation")
        with self.open_writer() as archive:
            archive.run_generation_readiness_pass()
        self.source_baseline.verify(self.archive_root / "source.db")
        # Measured here and nowhere else: after the readiness pass the
        # candidate carries its rows, its deferred indexes and its FTS.
        # This samples final candidate allocation before pointer promotion;
        # it is not a measured all-build peak. Without this the receipt keeps
        # ``final_candidate_allocated_bytes == 0`` and ``calibrated_index_ratio``
        # returns its unmeasured default forever.
        self._store.observe_candidate_capacity(operation_id=self.operation_id, generation_id=self.generation_id)
        self._promotion_candidate_ready = True

    def prepare_promotion_proof(self) -> PreparedIndexPromotion:
        """Build the full retained reference/coverage proof outside writer custody."""
        if not self._promotion_candidate_ready:
            raise RuntimeError("cold-build candidate must finish readiness before promotion proof")
        return self._store.prepare_promotion(self.generation)

    def promote_prepared(self, prepared: PreparedIndexPromotion) -> IndexGeneration:
        """Settle a previously prepared promotion while holding writer custody."""
        if self._promoted:
            return self.reconcile_promoted()
        if self._discarded:
            raise RuntimeError(f"cold-build generation {self.generation_id} is already settled")
        if not self._promotion_candidate_ready:
            raise RuntimeError("cold-build candidate readiness was not published")
        try:
            promoted = self._store.promote(self.generation, prepared)
        except Exception as exc:
            current = self._store.load(self.generation_id)
            if self._store.active_pointer.resolve() == Path(current.index_path).resolve():
                # The store can fail after its pointer swap but before it
                # writes active metadata. The pointer already publishes this
                # candidate, so stop routing cold writes and finish recovery.
                self._promoted = True
                recovered = self.reconcile_promoted()
                if _typed_promotion_io_failure(exc):
                    return recovered
                raise
            if current.state == "promoting":
                if self._store.unpublished_rollback_pending(self.generation_id):
                    # A full filesystem can reject even a prepared metadata
                    # replace. Keep the typed storage fault and restore the
                    # inactive record at the next writer pass.
                    raise
                recovered = self._store.recover_promotion(self.generation_id)
                if recovered.state != "inactive":
                    raise RuntimeError("cold-build promotion could not restore inactive routing") from exc
            raise
        self._promoted = True
        try:
            from polylogue.sources.live.production_baseline import clear_pending_production_baseline

            clear_pending_production_baseline(self.archive_root, self.source_baseline)
            self._receipt_cleared = True
        finally:
            self._release_ops_checkpoint_holder()
        emit(
            "daemon.cold_build.generation_promoted",
            outcome="ok",
            generation_id=promoted.generation_id,
            predecessor=promoted.predecessor_generation_id,
            reason=self.reason,
        )
        return promoted

    def promote(self) -> IndexGeneration:
        """Run readiness, prepare references off-gate, then swap the pointer."""
        from polylogue.core.write_lease import current_write_lease

        if self._promoted:
            # The remaining tail retires its baseline receipt under writer
            # custody. Source acknowledgement belongs to ordinary retained replay.
            if current_write_lease() is not None:
                return self.reconcile_promoted()
            from polylogue.storage.sqlite.write_lease import write_lease

            with write_lease("storage.cold_build.reconcile_promoted", archive_root=self.archive_root):
                return self.reconcile_promoted()
        if current_write_lease() is not None:
            raise RuntimeError("cold-build promotion must prepare references before writer admission")
        self.prepare_promotion_candidate()
        with self.prepare_promotion_proof() as prepared:
            from polylogue.storage.sqlite.write_lease import require_write_lease, write_lease

            require_write_lease("cold-build promotion", archive_root=self.archive_root)
            with write_lease("storage.cold_build.promote", archive_root=self.archive_root):
                return self.promote_prepared(prepared)

    def reconcile_promoted(self) -> IndexGeneration:
        """Finish a failed receipt tail only after confirming the active pointer."""
        if not self._promoted:
            raise RuntimeError("cold-build candidate has not been promoted")
        try:
            promoted = self._store.load(self.generation_id)
            if self._store.active_pointer.resolve() != Path(promoted.index_path).resolve():
                raise RuntimeError("promoted cold-build candidate is not the active index")
            if promoted.state == "promoting":
                self.source_baseline.verify(self.archive_root / "source.db")
                with closing(
                    open_readonly_connection(
                        Path(promoted.index_path), tier=ArchiveTier.INDEX, timeout_class="background-read"
                    )
                ) as candidate:
                    candidate.execute("SELECT COUNT(*) FROM sessions").fetchone()
                promoted = self._store.complete_promotion_recovery(self.generation_id)
            if promoted.state != "active":
                raise RuntimeError("promoted cold-build candidate has incomplete metadata")
            if not self._receipt_cleared:
                from polylogue.sources.live.production_baseline import clear_pending_production_baseline

                clear_pending_production_baseline(self.archive_root, self.source_baseline, allow_missing=True)
                self._receipt_cleared = True
            return promoted
        finally:
            self._release_ops_checkpoint_holder()

    def discard(self) -> bool:
        """Drop a never-promoted generation; the previous active one is untouched."""
        if self.settled:
            return False
        discarded = self._store.discard_if_inactive(self.generation)
        self._discarded = True
        self._release_ops_checkpoint_holder()
        emit(
            "daemon.cold_build.generation_discarded",
            outcome="ok" if discarded else "degraded",
            generation_id=self.generation_id,
            reason=self.reason,
        )
        return discarded


_LOCK = threading.Lock()
_ACTIVE: ColdBuildGeneration | None = None


def register_cold_build_generation(generation: ColdBuildGeneration) -> None:
    """Make one cold build the process's live-write destination.

    Process-scoped for the same reason ``core.degraded.degraded_reason()`` is:
    the fact is true of the whole daemon process, and the live write open
    (``sources/live/archive_open.py``) is the one place that has to consult
    it. Threading it through the watcher, the batch processor and the intake
    adapters would make every one of them carry a parameter whose only job is
    to arrive here unchanged.
    """
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None and not _ACTIVE.settled:
            raise RuntimeError(f"a cold build is already registered: {_ACTIVE.generation_id}")
        _ACTIVE = generation


def active_cold_build_generation(archive_root: Path | None = None) -> ColdBuildGeneration | None:
    """Return the registered cold build for ``archive_root``, if any."""
    with _LOCK:
        generation = _ACTIVE
    if generation is None or generation.settled:
        return None
    if archive_root is not None and Path(archive_root).resolve() != generation.archive_root.resolve():
        return None
    return generation


def clear_cold_build_generation() -> None:
    global _ACTIVE
    with _LOCK:
        _ACTIVE = None

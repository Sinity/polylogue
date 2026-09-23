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

import os
import sqlite3
import threading
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from polylogue.logging import ERROR, emit
from polylogue.maintenance.candidate_capacity import (
    InsufficientCapacityError,
    require_candidate_capacity,
)
from polylogue.storage.index_generation import (
    IndexGeneration,
    IndexGenerationStore,
    rebuild_source_evidence_snapshot,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

__all__ = [
    "WANTED_SOURCE_FREEZE_COMMAND",
    "ColdBuildGeneration",
    "active_cold_build_generation",
    "active_index_generation_is_empty",
    "clear_cold_build_generation",
    "register_cold_build_generation",
]


#: The only route that publishes a frozen wanted-source receipt.
WANTED_SOURCE_FREEZE_COMMAND: Final = "polylogue ops maintenance wanted-sources --freeze"


def _cold_build_owner_id() -> str:
    """One owner per daemon process: ownership is what promotion checks."""
    return f"cold-build:{os.getpid()}"


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
    return ops_db.with_name(f"{ops_db.name}-shm").exists()


def _require_frozen_wanted_sources(archive_root: Path, *, reason: str, operation_id: str) -> None:
    """Refuse a cold build that no complete, valid wanted-source receipt authorizes.

    The rebuild's conservation proof needs a denominator fixed *before* the
    build (polylogue-co2iz). ``require_rebuild_preflight`` has owned that
    check since #5304, but its only caller was the operator CLI, so the build
    driver -- the thing whose behaviour the denominator constrains -- started
    without ever consulting it.

    Like the capacity preflight three lines below, this is not conditioned on
    the operator having asked for a cold build: the auto-engaged build on an
    empty active generation is precisely the unattended whole-archive case
    that must not run unauthorized. It *is* conditioned on a denominator
    existing at all, and that is a different question from operator intent:

    * the receipt enumerates ``config.source_declarations``, i.e. explicitly
      configured standalone roots only (discovery roots are ambient provider
      state the campaign policy excludes by design);
    * with no declared root, ``build_wanted_source_receipt`` itself refuses
      ("no source is declared: the rebuild denominator would be empty"), so a
      receipt requirement there is unsatisfiable, not strict -- it would make
      every live-capture-only archive permanently unbuildable;
    * once a receipt *is* published, it is validated unconditionally, even if
      the roots were later undeclared. A frozen denominator that the current
      configuration no longer matches is a refusal, never a silent downgrade.

    The refusal is typed (``WantedSourceReceiptError``) and names both the
    defect and the single command that produces a receipt. There is no branch
    here that falls back to walking source roots fresh.
    """
    from polylogue.config import configured_source_declarations, resolve_runtime_config
    from polylogue.maintenance.source_manifest_continuity import (
        WantedSourceReceiptError,
        campaign_default_wanted_source_policy,
        require_rebuild_preflight,
        wanted_source_receipt_is_published,
    )

    declarations = configured_source_declarations(resolve_runtime_config())
    if not declarations and not wanted_source_receipt_is_published(archive_root):
        # The build proceeds -- with no declared root
        # ``build_wanted_source_receipt`` itself refuses, so requiring a
        # receipt here would make every live-capture-only archive permanently
        # unbuildable. But proceeding without a frozen denominator is a named
        # gap, not a clean run, and this docstring's own promise is "never a
        # silent downgrade". Reporting ``ok`` made an unauthorized-denominator
        # build indistinguishable from an authorized one in the event stream,
        # which is exactly how a whole-archive rebuild runs unnoticed without
        # the conservation proof polylogue-co2iz requires. The event kind is
        # the named reason -- ``logging_fields`` registers no
        # ``degraded_reason`` and an unregistered field is dropped at the emit
        # boundary rather than recorded.
        emit(
            "daemon.cold_build.wanted_sources_undeclared",
            outcome="degraded",
            reason=reason,
            operation_id=operation_id,
            sources=0,
        )
        return
    try:
        preflight = require_rebuild_preflight(
            archive_root,
            policy=campaign_default_wanted_source_policy(),
            declarations=declarations,
        )
    except WantedSourceReceiptError as refusal:
        # Same reporting constraint as the capacity refusal below: the digests
        # and counts have no registered logging field, and a refused build
        # writes no receipt, so ``error_detail`` is the whole record.
        detail = (
            f"cold build refused: {refusal}. {len(declarations)} declared source root(s) form the "
            f"rebuild denominator and no complete, valid frozen receipt authorizes this build; "
            f"produce one with `{WANTED_SOURCE_FREEZE_COMMAND}`."
        )
        emit(
            "daemon.cold_build.wanted_sources_refused",
            level=ERROR,
            outcome="error",
            reason=reason,
            operation_id=operation_id,
            error_type=type(refusal).__name__,
            error_detail=detail,
            sources=len(declarations),
        )
        raise WantedSourceReceiptError(detail) from refusal
    emit(
        "daemon.cold_build.wanted_sources_authorized",
        outcome="ok",
        reason=reason,
        operation_id=operation_id,
        content_hash=preflight.receipt_sha256,
        sources=len(declarations),
        files=preflight.item_count,
        bytes=preflight.byte_count,
    )


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
    _promoted: bool = False
    _discarded: bool = False
    #: Open for the build's lifetime so one-shot ``ops.db`` writers stop
    #: checkpointing on every close. See :func:`_hold_ops_checkpoints`.
    _ops_checkpoint_holder: sqlite3.Connection | None = None

    @classmethod
    def begin(cls, archive_root: Path, *, reason: str, owner_id: str | None = None) -> ColdBuildGeneration:
        """Create the inactive generation this build will fill.

        Refuses on an unauthorized denominator (``_require_frozen_wanted_sources``)
        and then on insufficient free space, both *before* the generation
        directory exists. A cold build is the whole index again on disk beside the one
        still serving reads, and it is engaged automatically whenever the
        active generation is empty -- so this preflight cannot be conditioned
        on the operator having asked for it, or the unattended 40 GB case
        would be the one left unguarded. The refusal is fatal on purpose:
        there is no smaller build to fall back to, and letting ingest fill
        the active generation instead would allocate the same bytes with
        readers attached.
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
        # The whole-tree walk this costs is measured in tens of seconds on a
        # real archive (77s over ~790k inodes), against a build measured in
        # hours -- and it runs once per build, not once per pass, because
        # ``begin`` is only reached when a cold build is actually starting.
        # That is the only cost gate this needs; intent is not a gate.
        operation_id = f"cold-build-{uuid.uuid4().hex}"
        # Authorization before allocation, and before the capacity walk: this
        # is the cheaper of the two preflights and the one whose refusal means
        # "this build must not happen at all" rather than "not here, not now".
        _require_frozen_wanted_sources(archive_root, reason=reason, operation_id=operation_id)
        try:
            require_candidate_capacity(archive_root, operation_id=operation_id)
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
        store = IndexGenerationStore.for_archive_root(archive_root)
        snapshot = ""
        if (archive_root / "source.db").exists():
            snapshot = rebuild_source_evidence_snapshot(archive_root)
        generation = store.create(owner_id=owner_id or _cold_build_owner_id(), source_snapshot=snapshot)
        emit(
            "daemon.cold_build.generation_created",
            outcome="ok",
            generation_id=generation.generation_id,
            owner_id=generation.owner_id,
            page_size=generation.page_size,
            reason=reason,
            operation_id=operation_id,
        )
        return cls(
            archive_root=archive_root,
            generation=generation,
            reason=reason,
            operation_id=operation_id,
            _store=store,
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

        # The dispatcher owns one ArchiveStore for one intake page.  Tie the
        # checkpoint holder to that same lifetime: retaining it across pages
        # would silently widen the ops power-loss window to the whole build.
        # ``ArchiveStore`` is intentionally not changed for this cold-build
        # concern; binding the existing close method preserves its public
        # type and all normal close/rollback behavior.
        close = archive.close

        def close_page(_archive: ArchiveStore) -> None:
            try:
                close()
            finally:
                self._release_ops_checkpoint_holder()

        # Rebinding close on the instance (not the class) so the ops checkpoint
        # holder is released on whichever path closes this page.
        archive.close = types.MethodType(close_page, archive)  # type: ignore[method-assign]
        return archive

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

        The daemon's shutdown path calls :meth:`discard` from the event-loop
        thread while the holder was opened on a write-coordinator thread, which
        is exactly why the handle is opened without the same-thread check.
        """
        holder = self._ops_checkpoint_holder
        if holder is None:
            return
        self._ops_checkpoint_holder = None
        holder.close()

    def promote(self) -> IndexGeneration:
        """Run the readiness pass and swap the active-index pointer."""
        if self.settled:
            raise RuntimeError(f"cold-build generation {self.generation_id} is already settled")
        with self.open_writer() as archive:
            archive.run_generation_readiness_pass()
        # Measured here and nowhere else: after the readiness pass the
        # candidate carries its rows, its deferred indexes and its FTS, which
        # is this build's real peak, and ``promote`` is about to start moving
        # the tree around. Without this the recorded receipt keeps
        # ``actual_peak_index_bytes == 0`` and ``calibrated_index_ratio``
        # returns its unmeasured default forever.
        self._store.observe_candidate_capacity(operation_id=self.operation_id, generation_id=self.generation_id)
        promoted = self._store.promote(self.generation)
        self._promoted = True
        self._release_ops_checkpoint_holder()
        emit(
            "daemon.cold_build.generation_promoted",
            outcome="ok",
            generation_id=promoted.generation_id,
            predecessor=promoted.predecessor_generation_id,
            reason=self.reason,
        )
        return promoted

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

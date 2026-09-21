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
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

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
    "ColdBuildGeneration",
    "active_cold_build_generation",
    "active_index_generation_is_empty",
    "clear_cold_build_generation",
    "register_cold_build_generation",
]


def _cold_build_owner_id() -> str:
    """One owner per daemon process: ownership is what promotion checks."""
    return f"cold-build:{os.getpid()}"


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

    @classmethod
    def begin(cls, archive_root: Path, *, reason: str, owner_id: str | None = None) -> ColdBuildGeneration:
        """Create the inactive generation this build will fill.

        Refuses on insufficient free space *before* the generation directory
        exists. A cold build is the whole index again on disk beside the one
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
        return ArchiveStore.open_cold_build_generation(
            self.generation_root,
            generation_id=self.generation_id,
            owner_id=self.generation.owner_id,
            defer_secondary_indexes=True,
        )

    def session_count(self) -> int:
        """How many sessions the build has materialized so far."""
        with self.open_writer() as archive:
            row = archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0]) if row is not None else 0

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

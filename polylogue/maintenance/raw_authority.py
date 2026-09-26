"""Product boundary for durable raw-authority maintenance.

The storage implementation deliberately owns the durable receipts and replay
algorithms.  CLI and daemon surfaces use this module so that they share one
typed product operation rather than importing storage internals directly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config, active_archive_root
from polylogue.core.json import JSONDocument

if TYPE_CHECKING:
    from polylogue.storage.raw_reconciler import RawAuthorityFrontierCensus


def inspect_frontier(config: Config) -> RawAuthorityFrontierCensus:
    from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier

    return inspect_raw_authority_frontier(config)


def finalize_codex_state_snapshots(config: Config) -> int:
    """Finalize admitted Codex state snapshots that carry no terminal receipt.

    Must run before the raw-materialization source-selection gate: such a
    raw is an incomparable cursor row to the gate, and every route that could
    finalize it sits behind the gate.
    """
    from polylogue.sources.codex_state_evidence import resolve_retained_codex_state_receipts

    return resolve_retained_codex_state_receipts(config.archive_root)


@contextlib.contextmanager
def materialization_generation_lease(config: Config) -> Iterator[Path]:
    """Pin one active index generation through a replay-adjacent closure."""
    from polylogue.storage.index_generation import ActiveWriterLease

    lease = ActiveWriterLease(active_archive_root(config))
    lease.acquire()
    try:
        yield config.current_db_path()
    finally:
        lease.close()


class ArchiveWriterRebuildExclusion:
    """Product authority preventing an offline rebuild from overlapping a writer."""

    def __init__(self, archive_root: Path) -> None:
        from polylogue.storage.index_generation import ActiveWriterLease

        self._lease = ActiveWriterLease(archive_root)
        self._retained_until_process_exit = False
        self._lease.acquire()

    def retain_until_process_exit(self) -> None:
        """Keep exclusion when a writer cannot be proven drained.

        The raw file descriptor deliberately remains open and is reclaimed by
        the OS at process exit. Releasing it after a bounded shutdown timeout
        would let an offline rebuild overlap the admitted writer that caused
        that timeout.
        """
        self._retained_until_process_exit = True

    def release(self) -> None:
        """Release exclusion after every admitted writer is proven drained."""
        self._lease.close()
        self._retained_until_process_exit = False

    def release_if_safe(self) -> None:
        """Release unless shutdown transferred authority to process lifetime."""
        if not self._retained_until_process_exit:
            self.release()


@contextlib.contextmanager
def archive_writer_rebuild_exclusion(archive_root: Path) -> Iterator[ArchiveWriterRebuildExclusion]:
    """Acquire process-lifetime-capable rebuild exclusion for an archive writer."""
    exclusion = ArchiveWriterRebuildExclusion(archive_root)
    try:
        yield exclusion
    finally:
        exclusion.release_if_safe()


def list_blockers(archive_root: Path, *, limit: int = 100, offset: int = 0) -> JSONDocument:
    """Read-only, paginated inventory of unresolved raw-authority blockers (operator discovery surface).

    Returns an envelope (``blockers``, ``offset``, ``limit``, ``returned_count``,
    ``total_count``, ``truncated``, ``next_offset``) rather than a bare list so
    a caller with more than ``limit`` unresolved blockers can tell rows were
    dropped and page to the next batch instead of only ever seeing page one.
    """
    from polylogue.storage.raw_authority import list_unresolved_raw_authority_blockers

    return list_unresolved_raw_authority_blockers(archive_root, limit=limit, offset=offset)


__all__ = [
    "ArchiveWriterRebuildExclusion",
    "archive_writer_rebuild_exclusion",
    "inspect_frontier",
    "list_blockers",
    "materialization_generation_lease",
]

"""Product boundary for durable raw-authority maintenance.

The storage implementation deliberately owns the durable receipts and replay
algorithms.  CLI and daemon surfaces use this module so that they share one
typed product operation rather than importing storage internals directly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from polylogue.config import Config, active_archive_root
from polylogue.core.json import JSONDocument

if TYPE_CHECKING:
    from polylogue.storage.raw_authority import RawAuthorityCensusReceipt
    from polylogue.storage.raw_reconciler import RawAuthorityFrontierApplyReport, RawAuthorityFrontierCensus


RAW_MATERIALIZATION_ORDINARY_BLOB_LIMIT_BYTES: Final = 64 * 1024 * 1024
RAW_MATERIALIZATION_WHALE_BLOB_LIMIT_BYTES: Final = 8 * 1024 * 1024 * 1024


def inspect_frontier(config: Config) -> RawAuthorityFrontierCensus:
    from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier

    return inspect_raw_authority_frontier(config)


def _validate_frontier_apply_report(
    report: object,
    *,
    selected_plan_ids: tuple[str, ...],
    preview_census_id: str,
) -> RawAuthorityFrontierApplyReport:
    """Reject an actuator response that cannot conserve selected plan outcomes."""
    from polylogue.storage.raw_reconciler import validate_raw_authority_frontier_apply_report

    return validate_raw_authority_frontier_apply_report(
        report,
        selected_plan_ids=selected_plan_ids,
        preview_census_id=preview_census_id,
    )


def apply_frontier(
    config: Config,
    *,
    preview_census_id: str,
    selected_plan_ids: tuple[str, ...],
) -> RawAuthorityFrontierApplyReport:
    from polylogue.daemon.write_coordinator import daemon_write_lease_active

    if not daemon_write_lease_active():
        raise RuntimeError("raw authority frontier apply requires the daemon writer lease")

    from polylogue.storage.raw_reconciler import apply_raw_authority_frontier

    report = apply_raw_authority_frontier(
        config,
        preview_census_id=preview_census_id,
        selected_plan_ids=selected_plan_ids,
    )
    return _validate_frontier_apply_report(
        report,
        selected_plan_ids=selected_plan_ids,
        preview_census_id=preview_census_id,
    )


def finalize_codex_state_snapshots(config: Config) -> int:
    """Finalize admitted Codex state snapshots that carry no terminal receipt.

    Must run before the raw-materialization source-selection gate: such a
    raw is an incomparable cursor row to the gate, and every route that could
    finalize it sits behind the gate.
    """
    from polylogue.sources.codex_state_evidence import resolve_retained_codex_state_receipts

    return resolve_retained_codex_state_receipts(config.archive_root)


def recover_interrupted_frontier(config: Config) -> tuple[str, ...]:
    from polylogue.storage.raw_reconciler import recover_interrupted_raw_authority_frontier

    return recover_interrupted_raw_authority_frontier(config)


def auto_resolve_stale_plan_blockers(config: Config) -> int:
    """Clear every unresolved stale-plan blocker automatically (polylogue-d7im).

    See ``storage.raw_authority.auto_resolve_stale_plan_blockers`` for why
    this is safe to run unattended: a stale-plan blocker requires no
    judgment content. This explicit maintenance operation does not run
    during ordinary address-scoped publication.
    """
    from polylogue.storage.raw_authority import auto_resolve_stale_plan_blockers as _auto_resolve

    return _auto_resolve(config.archive_root)


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


def unfinished_materialization_census_ids(
    archive_root: Path, *, after_sequence: int = 0, limit: int = 128
) -> tuple[tuple[str, int], ...]:
    """Page named durable startup obligations, not the observation backlog."""
    from polylogue.storage.raw_authority import unfinished_raw_authority_census_ids

    return unfinished_raw_authority_census_ids(archive_root, after_sequence=after_sequence, limit=limit)


def recover_materialization_censuses(
    config: Config, *, census_ids: Sequence[str]
) -> tuple[RawAuthorityCensusReceipt, ...]:
    """Finish named crash-left ledger obligations without replaying source bytes.

    Startup supplies durable census IDs. The canonical adapter checks their
    exact component outputs; unrelated archive rows never enter this recovery.
    Failed source/application evidence remains an explicit durable blocker.
    """
    from polylogue.operations.raw_observation_derivation import raw_observation_frame
    from polylogue.storage.derived.raw import RawObservationDerivation
    from polylogue.storage.raw_authority import (
        build_raw_replay_plans,
        finalize_raw_authority_census,
        raw_authority_census_replay_plans,
        recover_interrupted_raw_authority_censuses,
    )

    root = active_archive_root(config)
    completed = []
    with materialization_generation_lease(config) as index_db:
        scopes = recover_interrupted_raw_authority_censuses(root, index_db_path=index_db, census_ids=census_ids)
        adapter = RawObservationDerivation(root)
        frame = raw_observation_frame(root)
        for census_id, _scope in scopes:
            plans = raw_authority_census_replay_plans(root, census_id)
            components = []
            for plan in plans:
                states = adapter.inspect(frame, plan.input_raw_ids)
                if any(state != "valid" for state in states.values()):
                    components.append(plan.input_raw_ids)
            post_plans = build_raw_replay_plans(root, components, index_db_path=index_db) if components else ()
            completed.append(
                finalize_raw_authority_census(
                    root,
                    census_id,
                    post_plans=post_plans,
                    post_residual={},
                    interrupted=True,
                )
            )
    return tuple(completed)


def read_census(archive_root: Path, query_handle: str, *, limit: int, offset: int | None) -> JSONDocument:
    from polylogue.storage.raw_authority import read_raw_authority_census

    return read_raw_authority_census(archive_root, query_handle, limit=limit, offset=offset)


def read_detail(archive_root: Path, query_handle: str, *, chunk_chars: int, offset: int | None) -> JSONDocument:
    from polylogue.storage.raw_authority import read_raw_authority_detail

    return read_raw_authority_detail(archive_root, query_handle, chunk_chars=chunk_chars, offset=offset)


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
    "apply_frontier",
    "archive_writer_rebuild_exclusion",
    "inspect_frontier",
    "list_blockers",
    "materialization_generation_lease",
    "read_census",
    "read_detail",
    "recover_interrupted_frontier",
    "recover_materialization_censuses",
    "unfinished_materialization_census_ids",
]

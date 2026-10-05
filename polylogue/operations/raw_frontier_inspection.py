"""Inspect accepted Raw frontiers under the supplied preparation owner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.storage.frontier_inspection import (
    inspect_prepared_raw_authority_frontier,
)

if TYPE_CHECKING:
    from polylogue.core.compute import BoundedComputeAdapter


def frontier_coverage_for_archive(archive_root: Path) -> dict[str, object]:
    """Borrow the canonical pinned coverage read."""
    from polylogue.storage.frontier_inspection import read_frontier_coverage_for_archive

    return read_frontier_coverage_for_archive(archive_root)


def make_raw_frontier_inspection_stage(db_path: Path, *, compute_adapter: BoundedComputeAdapter) -> ConvergenceStage:
    """Consume journal changes before any obsolete journal rows are pruned."""
    root = db_path.parent

    def check(_path: Path) -> bool:
        coverage = frontier_coverage_for_archive(root)
        return not (coverage.get("current") and coverage.get("healthy"))

    def check_many(paths: Sequence[Path]) -> set[Path]:
        return set(paths) if paths and check(paths[0]) else set()

    def execute(_path: Path) -> StageExecuteReturn:
        # The caller's run_convergence_sync supplies the real exclusive worker
        # and original stage bridge. This factory supplies no new kernel.
        return inspect_prepared_raw_authority_frontier(
            root, input_demand=compute_adapter.amend_current_input_demand
        ).healthy

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        return True if not paths else execute(paths[0])

    return ConvergenceStage(
        name="raw_frontier_inspection",
        description="Inspect changed accepted Raw frontier inputs",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        whole_archive=True,
        subject_independent=True,
        false_means_pending=True,
        writer_admission="bridged",
    )

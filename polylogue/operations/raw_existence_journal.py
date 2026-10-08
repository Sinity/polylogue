"""Daemon owner of the live-admission raw-existence journals' consumed rows.

The source/index triggers append one changed key per raw deletion,
re-pointing or head advance, and live admission reads only the rows past its
certificate's watermark. Rows at or below that watermark have served their
purpose; without an owner deleting them the journals grow with every ingest
for the life of the archive.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.logging import span
from polylogue.storage.frontier_existence import has_consumed_journal_rows, prune_consumed_journal_rows

if TYPE_CHECKING:
    from polylogue.core.compute import BoundedComputeAdapter


def make_raw_existence_journal_prune_stage(
    db_path: Path, *, compute_adapter: BoundedComputeAdapter
) -> ConvergenceStage:
    """Prune consumed journal rows once per pass; ``check`` is two indexed reads."""
    archive_root = db_path.parent

    def check(_path: Path) -> bool:
        return has_consumed_journal_rows(archive_root)

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        return set(paths) if check(paths[0]) else set()

    def execute(_path: Path) -> StageExecuteReturn:
        return execute_many((_path,))

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        with span("daemon.stage.execute", stage="raw_existence_journal_prune", files=len(paths)) as work:
            work.ok(
                pruned=prune_consumed_journal_rows(
                    archive_root, input_demand=compute_adapter.amend_current_input_demand
                )
            )
            return True

    return ConvergenceStage(
        name="raw_existence_journal_prune",
        description="Delete raw-existence journal rows the live admission certificate has consumed",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        writer_admission="bridged",
    )


__all__ = ["make_raw_existence_journal_prune_stage"]

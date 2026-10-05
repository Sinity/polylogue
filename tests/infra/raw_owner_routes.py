"""Drive the daemon's raw-observation owner from fixtures.

Append acquisition and retained replay publish only through
``RawObservationConvergenceOwner``. These helpers borrow that real owner (its
compute adapter, write coordinator and settlement) for one call; they add no
alternate preparation or publication path.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner

if TYPE_CHECKING:
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
    from polylogue.sources.live.metrics import LiveBatchMetrics
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage
    from polylogue.sources.revision_backfill import PreparedRevisionReplayResult


def _owner_archive_root(owner: Any) -> Path:
    return Path(getattr(owner._polylogue, "archive_root", owner._cursor._db_path.parent))


async def ingest_append_with_owner_async(owner: Any, plans: list[_AppendPlan]) -> _AppendResult:
    """Acquire, prepare and publish append plans on the actual raw owner."""
    async with prepared_live_convergence_owner(_owner_archive_root(owner)) as raw_owner:
        return await raw_owner.ingest_append_plans(owner, plans)


def ingest_append_with_owner(owner: Any, plans: list[_AppendPlan]) -> _AppendResult:
    """Synchronous form of :func:`ingest_append_with_owner_async`."""
    return asyncio.run(ingest_append_with_owner_async(owner, plans))


def retained_raw_ids(archive_root: Path) -> tuple[str, ...]:
    """Every retained raw, in acquisition order."""
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        return tuple(str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))


async def replay_retained_raws_async(
    archive_root: Path, raw_ids: Sequence[str] | None = None
) -> tuple[PreparedRevisionReplayResult, ...]:
    """Replay retained raws into the active Index through the owner's replay route."""
    selected = retained_raw_ids(archive_root) if raw_ids is None else tuple(raw_ids)
    async with prepared_live_convergence_owner(archive_root) as raw_owner:
        return await raw_owner.replay_retained_raw_ids(selected)


def replay_retained_raws(
    archive_root: Path, raw_ids: Sequence[str] | None = None
) -> tuple[PreparedRevisionReplayResult, ...]:
    """Synchronous form of :func:`replay_retained_raws_async`."""
    return asyncio.run(replay_retained_raws_async(archive_root, raw_ids))


@asynccontextmanager
async def cold_rebuilt_index(archive_root: Path) -> AsyncIterator[Path]:
    """Build a fresh Index generation from retained Source evidence alone.

    The owned generation receives every retained raw through the cold-build
    replay route; the caller reads its Index while it exists. The generation is
    discarded on exit and the active Index is never touched.
    """
    raws = retained_raw_ids(archive_root)

    def begin() -> ColdBuildGeneration:
        return ColdBuildGeneration.begin(
            archive_root, reason="test-retained-rebuild", sources=(WatchSource("fixture", archive_root / "absent"),)
        )

    generation = await run_archive_fixture_write(archive_root, begin)
    register_cold_build_generation(generation)
    try:
        async with prepared_live_convergence_owner(archive_root) as raw_owner:
            await raw_owner.replay_retained_raw_ids(raws)
        await run_archive_fixture_write(archive_root, generation.prepare_promotion_candidate)
        yield Path(generation.generation.index_path)
    finally:
        clear_cold_build_generation()
        await run_archive_fixture_write(archive_root, generation.discard)


@dataclass(frozen=True, slots=True)
class LiveOwnerSet:
    """The daemon's live intake owners, bound to the running event loop."""

    compute: BoundedComputeAdapter
    coordinator: DaemonWriteCoordinator
    stage: LiveSQLiteCaptureStage
    raw_owner: RawObservationConvergenceOwner

    def watcher_kwargs(self) -> dict[str, Any]:
        """``LiveWatcher`` owner arguments, exactly as the daemon passes them."""
        return {
            "write_coordinator": self.coordinator,
            "sqlite_capture_stage": self.stage,
            "append_runner": self.raw_owner.ingest_append_plans,
            "convergence_runner": self.raw_owner.run_convergence_sync,
            "retained_runner": self.raw_owner.ingest_retained_raw_ids,
        }

    def processor_slots(self) -> dict[str, Any]:
        """``LiveBatchProcessor`` owner slots, matching its constructor arguments."""
        return {
            "_sqlite_capture_stage": self.stage,
            "_sync_runner": self.coordinator.run_sync,
            "_append_runner": self.raw_owner.ingest_append_plans,
            "_retained_runner": self.raw_owner.ingest_retained_raw_ids,
            "_convergence_runner": self.raw_owner.run_convergence_sync,
        }


@asynccontextmanager
async def live_owner_set(root: Path) -> AsyncIterator[LiveOwnerSet]:
    """Build the daemon's live owners for ``root`` and settle them physically."""
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage

    root.mkdir(parents=True, exist_ok=True)
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    stage = LiveSQLiteCaptureStage(compute_adapter=compute)
    try:
        async with prepared_live_convergence_owner(
            root, compute_adapter=compute, write_coordinator=coordinator
        ) as raw_owner:
            yield LiveOwnerSet(compute=compute, coordinator=coordinator, stage=stage, raw_owner=raw_owner)
    finally:
        primary = sys.exception()
        failures: list[BaseException] = []
        settled = False
        try:
            # Idempotent: a watcher that owned the stage may already have closed it.
            stage.shutdown()
        except BaseException as failure:
            failures.append(failure)
        try:
            settled = await coordinator.shutdown(timeout=float("inf"))
            if not settled:
                raise RuntimeError("live owner coordinator did not physically settle")
        except BaseException as failure:
            failures.append(failure)
        try:
            await asyncio.to_thread(compute.shutdown, wait=settled)
        except BaseException as failure:
            failures.append(failure)
        if failures:
            if primary is not None:
                failures.insert(0, primary)
            raise BaseExceptionGroup("live owner settlement failed", failures) from primary


@asynccontextmanager
async def supplied_live_owners(processor: LiveBatchProcessor) -> AsyncIterator[LiveBatchProcessor]:
    """Supply one pass's canonical owners to a directly constructed processor.

    The daemon constructs ``LiveBatchProcessor`` with its capture stage, writer
    coordinator and raw owner. A fixture processor built outside an event loop
    receives the same owners here, bound to the running loop, for one pass;
    the slots are restored before the owners physically settle.
    """
    async with live_owner_set(_owner_archive_root(processor)) as owners:
        slots = owners.processor_slots()
        previous = {slot: getattr(processor, slot) for slot in slots}
        for slot, value in slots.items():
            setattr(processor, slot, value)
        try:
            yield processor
        finally:
            for slot, value in previous.items():
                setattr(processor, slot, value)


async def ingest_files_with_owners(
    processor: LiveBatchProcessor, paths: Sequence[Path], **kwargs: Any
) -> LiveBatchMetrics:
    """Run one ``ingest_files`` pass with the canonical owners supplied."""
    async with supplied_live_owners(processor):
        return await processor.ingest_files(list(paths), **kwargs)


def run_ingest_files(processor: LiveBatchProcessor, paths: Sequence[Path], **kwargs: Any) -> LiveBatchMetrics:
    """Synchronous form of :func:`ingest_files_with_owners`."""
    return asyncio.run(ingest_files_with_owners(processor, paths, **kwargs))


__all__ = [
    "LiveOwnerSet",
    "cold_rebuilt_index",
    "ingest_files_with_owners",
    "ingest_append_with_owner",
    "ingest_append_with_owner_async",
    "live_owner_set",
    "replay_retained_raws",
    "replay_retained_raws_async",
    "retained_raw_ids",
    "run_ingest_files",
    "supplied_live_owners",
]

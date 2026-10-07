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
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.core.enums import ValidationMode
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner

if TYPE_CHECKING:
    from polylogue.archive.revision_authority import RawRevisionAuthority
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.derivation import DerivationReport
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.operations.intake_adapters import RawMaterializationDiscovery
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
    from polylogue.sources.live.metrics import LiveBatchMetrics
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.revision_backfill import PreparedRevisionReplayResult
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation


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
        return (await raw_owner.replay_retained_raw_ids(selected)).require_complete()


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
            archive_root,
            reason="test-retained-rebuild",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", archive_root / "absent"),)),
        )

    generation = await run_archive_fixture_write(archive_root, begin)
    register_cold_build_generation(generation)
    try:
        async with prepared_live_convergence_owner(archive_root) as raw_owner:
            (await raw_owner.replay_retained_raw_ids(raws)).require_complete()
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
        # A slot a test already filled (a recording or failing runner) is its
        # declared seam and stays in place; only empty slots receive owners.
        slots = {slot: value for slot, value in owners.processor_slots().items() if getattr(processor, slot) is None}
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


async def seed_membership_census_async(
    archive_root: Path,
    entries: Sequence[tuple[str, Sequence[ParsedSession]]],
    *,
    parser_fingerprint: str,
    censused_at_ms: int = 1,
    revision_authority: RawRevisionAuthority | None = None,
    detail: str = "",
    retire_full_revision_governance: bool = False,
) -> None:
    """Record membership census receipts through the canonical prepared Source route.

    Each census is prepared on one original source-only seal and published by
    its own Source permit under the raw owner's admitted worker, the same
    sequence production preparation uses. No eager census wrapper is involved.
    """
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.archive_tiers.revision_governance import replace_raw_membership_census
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    async with prepared_live_convergence_owner(archive_root) as raw_owner:
        retained: list[PreparedIndexMutation] = []

        def seed() -> None:
            seal = PreparedIndexMutation.source_only(archive_root=archive_root)
            retained.append(seal)
            with seal:
                with seal.original_read_snapshot(), seal.source_producer():
                    for raw_id, sessions in entries:
                        replace_raw_membership_census(
                            seal,
                            raw_id,
                            list(sessions),
                            parser_fingerprint=parser_fingerprint,
                            censused_at_ms=censused_at_ms,
                            detail=detail,
                            revision_authority=revision_authority,
                            retire_full_revision_governance=retire_full_revision_governance,
                        )
                permit = seal.prepare_source_mutation()

                def publish() -> None:
                    with permit.hold_authority(), permit.mutation_connection() as source:
                        with closing(source.execute("BEGIN IMMEDIATE")):
                            pass
                        permit.apply_source_statements(source)
                        permit.allow_commit(source)
                        source.commit()
                        seal.accept_known_tier_commit(permit.committed())

                admit_stage_write("test.membership-census.seed", publish)
            retained.remove(seal)

        await raw_owner.run_prepared_sync(
            "test.membership-census.prepare", seed, settlement_owners=lambda: tuple(retained), estimated_bytes=0
        )


def seed_membership_census(
    archive_root: Path,
    entries: Sequence[tuple[str, Sequence[ParsedSession]]],
    *,
    parser_fingerprint: str,
    censused_at_ms: int = 1,
    revision_authority: RawRevisionAuthority | None = None,
    detail: str = "",
    retire_full_revision_governance: bool = False,
) -> None:
    """Synchronous form of :func:`seed_membership_census_async`."""
    asyncio.run(
        seed_membership_census_async(
            archive_root,
            entries,
            parser_fingerprint=parser_fingerprint,
            censused_at_ms=censused_at_ms,
            revision_authority=revision_authority,
            detail=detail,
            retire_full_revision_governance=retire_full_revision_governance,
        )
    )


def _lease_writer(root: Path) -> Callable[[str, Callable[[], bool]], bool]:
    def writer(actor: str, work: Callable[[], bool]) -> bool:
        with write_lease(actor, archive_root=root):
            return work()

    return writer


async def converge_pending_raws_async(
    raw_owner: RawObservationConvergenceOwner,
    archive_root: Path,
    *,
    limit: int,
    discovery: RawMaterializationDiscovery | None = None,
) -> DerivationReport:
    """One fair-intake pass, as the daemon runs it, folded into one report.

    The daemon's ``RawMaterializationDiscovery`` offers a bounded page of
    pending raws and each is converged by ``converge_raw_id``. A raw whose
    convergence raises is the intake's retryable admission: it is reported
    pending with its error. Pass the same ``discovery`` to continue its
    traversal across passes, as the daemon's intake lifetime does.
    """
    from polylogue.daemon.derivation import (
        DerivationKey,
        DerivationReport,
        KeyOutcome,
        Outcome,
        PendingReason,
        WorkCounters,
    )
    from polylogue.operations.intake_adapters import RawMaterializationDiscovery
    from polylogue.operations.raw_observation_derivation import RAW_OBSERVATION_DOMAIN, raw_observation_frame

    discovery = RawMaterializationDiscovery(archive_root) if discovery is None else discovery
    page = await asyncio.to_thread(discovery.discover_pending_raw_ids, limit)
    outcomes: list[KeyOutcome] = []
    counts = dict.fromkeys(Outcome, 0)
    work: dict[str, int] = {}
    for raw_id, _cost in page:
        try:
            report = await raw_owner.converge_raw_id(raw_id)
        except Exception as failure:
            outcomes.append(
                KeyOutcome(
                    DerivationKey(RAW_OBSERVATION_DOMAIN, raw_id),
                    Outcome.PENDING,
                    reason=PendingReason.BLOCKED,
                    error=f"{type(failure).__name__}: {failure}",
                )
            )
            counts[Outcome.PENDING] += 1
            continue
        outcomes.extend(report.outcomes)
        for outcome in Outcome:
            counts[outcome] += report.count(outcome)
        for counter in fields(WorkCounters):
            work[counter.name] = work.get(counter.name, 0) + int(getattr(report.work, counter.name))
    return DerivationReport(raw_observation_frame(archive_root), tuple(outcomes), counts, WorkCounters(**work))


def converge_pending_raws_with_owner(
    archive_root: Path,
    *,
    limit: int = 128,
    passes: int = 1,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
) -> DerivationReport:
    """Run ``passes`` fair-intake passes on one raw owner and discovery; return the last report."""
    from polylogue.operations.intake_adapters import RawMaterializationDiscovery

    async def run() -> DerivationReport:
        report: DerivationReport | None = None
        discovery = RawMaterializationDiscovery(archive_root)
        async with prepared_live_convergence_owner(archive_root, validation_mode=validation_mode) as raw_owner:
            for _ in range(passes):
                report = await converge_pending_raws_async(raw_owner, archive_root, limit=limit, discovery=discovery)
        assert report is not None
        return report

    return asyncio.run(run())


def inspect_raw_observations(archive_root: Path, raw_ids: Sequence[str]) -> Mapping[str, str]:
    """Inspect retained raws through the canonical adapter on the raw owner's creator."""
    from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame

    def inspect(compute_adapter: BoundedComputeAdapter) -> Mapping[str, str]:
        return make_raw_observation_derivation(archive_root, compute_adapter=compute_adapter).inspect(
            raw_observation_frame(archive_root), list(raw_ids)
        )

    async def run() -> Mapping[str, str]:
        async with prepared_live_convergence_owner(archive_root) as raw_owner:
            return await raw_owner.run_convergence_sync(
                "test.raw-observation.inspect", inspect, raw_owner._compute_adapter
            )

    return asyncio.run(run())


def seed_parser_census(archive_root: Path, raw_ids: Sequence[str]) -> None:
    """Record current-parser census receipts through the canonical prepared Source route."""
    from polylogue.storage.sqlite.archive_tiers.revision_governance import record_current_parser_source_census

    def prepare(seal: PreparedIndexMutation) -> None:
        for raw_id in raw_ids:
            record_current_parser_source_census(seal, raw_id)

    _publish_source_preparation(archive_root, prepare, actor="test.parser-census.seed")


def _publish_source_preparation(
    archive_root: Path, prepare: Callable[[PreparedIndexMutation], None], *, actor: str
) -> None:
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    async def run() -> None:
        async with prepared_live_convergence_owner(archive_root) as raw_owner:
            retained: list[PreparedIndexMutation] = []

            def seed() -> None:
                seal = PreparedIndexMutation.source_only(archive_root=archive_root)
                retained.append(seal)
                with seal:
                    with seal.original_read_snapshot(), seal.source_producer():
                        prepare(seal)
                    permit = seal.prepare_source_mutation()

                    def publish() -> None:
                        with permit.hold_authority(), permit.mutation_connection() as source:
                            with closing(source.execute("BEGIN IMMEDIATE")):
                                pass
                            permit.apply_source_statements(source)
                            permit.allow_commit(source)
                            source.commit()
                            seal.accept_known_tier_commit(permit.committed())

                    admit_stage_write(actor, publish)
                retained.remove(seal)

            await raw_owner.run_prepared_sync(
                f"{actor}.prepare", seed, settlement_owners=lambda: tuple(retained), estimated_bytes=0
            )

    asyncio.run(run())


__all__ = [
    "converge_pending_raws_async",
    "converge_pending_raws_with_owner",
    "inspect_raw_observations",
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
    "seed_parser_census",
    "seed_membership_census",
    "seed_membership_census_async",
    "supplied_live_owners",
]

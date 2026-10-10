"""Daemon composition for one exact raw-observation derivation.

Raw preparation is deliberately independent of the daemon writer.  The only
writer admission is the adapter's one-observation publication, forwarded from
the bounded compute worker through :class:`DaemonWriteThreadBridge`. Archive
reads and preparation live in :mod:`polylogue.operations.raw_observation_owner`;
this owner supplies admission, settlement and convergence around them.
"""

from __future__ import annotations

import asyncio
import pickle
import threading
from builtins import BaseExceptionGroup
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from polylogue.core import compute
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.compute_cancel import compute_cancel
from polylogue.core.enums import ValidationMode
from polylogue.core.raw_failure_evidence import (
    CohortMembershipRefusalError,
    RetainedRawDecodeRefusalError,
    RetainedRawDependencyRefusalError,
)
from polylogue.core.stage_admission import stage_write_admission
from polylogue.core.write_lease import adopt_write_lease
from polylogue.daemon.convergence import DaemonConverger, DerivationConvergenceOwner
from polylogue.daemon.derivation import Budget, DerivationReport
from polylogue.daemon.drive_catchup import DriveCatchupExecution
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.logging import propagate
from polylogue.operations.raw_observation_derivation import (
    RAW_OBSERVATION_DOMAIN,
    raw_observation_frame,
    raw_observation_payload_bytes,
)
from polylogue.operations.raw_observation_owner import RawObservationArchiveWork, retained_settlement_owners

if TYPE_CHECKING:
    from polylogue.core.sql_settlement import SQLCustodyOwner
    from polylogue.operations.raw_observation_owner import (
        AppendIngestOwner,
        AppendPlan,
        AppendResult,
        PreparedSessionSourceRead,
        RawObservationReplacement,
        RetainedMaterializationResult,
        RetainedReplayOutcome,
    )

T = TypeVar("T")
P = ParamSpec("P")


class RawObservationConvergenceOwner:
    """Converge exact admitted raw IDs through the canonical raw adapter.

    This is intentionally a small owner rather than another raw pipeline.  It
    owns only process-local serialization and borrows the daemon-wide compute
    and write owners supplied by composition.
    """

    def __init__(
        self,
        archive_root: Path,
        *,
        compute_adapter: BoundedComputeAdapter,
        write_bridge: DaemonWriteThreadBridge,
        write_coordinator: DaemonWriteCoordinator,
        validation_mode: ValidationMode = ValidationMode.ADVISORY,
    ) -> None:
        self._archive_root = archive_root
        self._compute_adapter = compute_adapter
        self._write_bridge = write_bridge
        self._write_coordinator = write_coordinator
        self._validation_mode = validation_mode
        self._archive = RawObservationArchiveWork(
            archive_root, compute_adapter=compute_adapter, validation_mode=validation_mode
        )
        self._converge_lock = asyncio.Lock()

    async def run_prepared_sync(
        self,
        actor: str,
        operation: Callable[[], T],
        *,
        settlement_owners: Callable[[], tuple[SQLCustodyOwner, ...]],
        estimated_bytes: int,
    ) -> T:
        """Retain dependency-discovering Raw work under exclusive byte admission.

        ``estimated_bytes`` counts every retained plan buffer operand, including
        duplicate bytes. Additional selected input reads amend that counter
        before hydration on each witness/accepted epoch. It is neither a unique
        archive-CAS forecast nor resident-memory measurement.
        """
        cancelled = threading.Event()

        def admit_write(write_actor: str, work: Callable[[], T]) -> T:
            # This body remains on the admitted compute creator. Nested SQL
            # drain would block the coordinator's original terminal delivery.
            with (
                self._write_bridge.hold(write_actor) as delegation,
                adopt_write_lease(delegation),
            ):
                return work()

        def run() -> T:
            with stage_write_admission(admit_write):
                return operation()

        cancellation = compute.CancellationHandle()

        def request_cancel() -> None:
            cancelled.set()
            cancellation.cancel()

        physical: compute.SubmittedOperation[None] | None = None

        def submit_worker(worker: Callable[[], None]) -> Future[None]:
            nonlocal physical
            # _run_writer_worker submits once and retains that original creator
            # for cleanup retries; it never dispatches replacement creators.
            submitted = self._compute_adapter.submit(
                propagate(worker),
                admission_class="incremental-background",
                estimated_bytes=estimated_bytes,
                exclusive_bytes=True,
                cancellation=cancellation,
            )
            physical = submitted
            return submitted.future

        cancellation_token = compute_cancel.set(cancelled)
        try:
            pending = asyncio.create_task(
                self._write_coordinator.run_prepared_sync(
                    actor, run, submit_worker=submit_worker, settlement_owners=settlement_owners
                ),
                name=f"polylogue-raw-prepared:{actor}",
            )
        finally:
            compute_cancel.reset(cancellation_token)
        primary: BaseException | None = None
        try:
            return await DriveCatchupExecution(self._write_coordinator, compute_adapter=self._compute_adapter).settle(
                pending, label=actor, cancel_requested=request_cancel
            )
        except BaseException as failure:
            primary = failure
            raise
        finally:
            if physical is not None:
                try:
                    await physical.wait()
                except compute.DaemonOperationCancelled:
                    if primary is None:
                        raise
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup(
                            "prepared result and physical settlement failed", [primary, cleanup]
                        ) from primary
                    raise

    async def run_convergence_sync(
        self, actor: str, operation: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Run the stage producer on the original admitted preparation worker."""
        # Failed original seals and native children remain in the coordinator's
        # existing retained sync-owner registry until their creator closes them.
        return await self.run_prepared_sync(
            actor,
            partial(operation, *args, **kwargs),
            settlement_owners=lambda: (),
            estimated_bytes=0,
        )

    async def ingest_append_plans(self, owner: AppendIngestOwner, plans: list[AppendPlan]) -> AppendResult:
        """Acquire, prepare and publish append raws under the existing physical worker owner."""
        retained: list[RawObservationReplacement] = []
        return await self.run_prepared_sync(
            "watcher.live_ingest.append",
            self._archive.append_plans_operation(
                owner, plans, retained=retained, require_authority=self._require_source_frontier_authority
            ),
            settlement_owners=partial(retained_settlement_owners, retained),
            estimated_bytes=sum(len(plan.payload) for plan in plans),
        )

    async def converge_raw_id(self, raw_id: str) -> DerivationReport:
        """Prepare, revalidate, and publish exactly ``raw_id`` if still pending."""
        if not raw_id:
            raise ValueError("raw observation id must be non-empty")
        async with self._converge_lock:
            from polylogue.core.write_lease import coordinator_write_lease_active

            if coordinator_write_lease_active():
                raise RuntimeError("raw observation convergence must start after the daemon writer lease is released")
            self._require_source_frontier_authority(raw_id)
            await self._prepare_cold_destination()
            adapter, index_path, _index_destination = self._archive.destination_adapter()
            owner = DerivationConvergenceOwner(
                DaemonConverger((), derivations=(adapter,)),
                compute_adapter=self._compute_adapter,
                write_bridge=self._write_bridge,
            )
            frame = raw_observation_frame(
                self._archive_root,
                raw_ids=(raw_id,),
                index_db_path=index_path,
                validation_mode=self._validation_mode,
            )
            # The parse holds the retained payload: reserve its size up front.
            # A read fault propagates as retryable rather than admitting the
            # parse as a zero-byte task beside a full reservation.
            payload_bytes = await asyncio.to_thread(raw_observation_payload_bytes, self._archive_root, raw_id)
            return await owner.converge(
                frame,
                budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
                domains=(RAW_OBSERVATION_DOMAIN,),
                resume=False,
                estimated_bytes=payload_bytes,
                exclusive_bytes=True,
            )

    async def _prepare_cold_destination(self, before_publication: Callable[[], None] | None = None) -> None:
        """Establish the original cold writer profile before recording read identity."""
        establish = self._archive.cold_destination(before_publication)
        if establish is not None:
            await self._write_coordinator.run_sync("retained.cold.destination", establish)

    def _require_source_frontier_authority(self, raw_id: str) -> None:
        self._archive.require_source_frontier_authority(raw_id)

    async def ingest_retained_raw_ids(
        self,
        raw_ids: Sequence[str],
        *,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
        on_dependency_refusal: Callable[[RetainedRawDependencyRefusalError], None] | None = None,
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None] | None = None,
        before_publication: Callable[[], None] | None = None,
    ) -> RetainedReplayOutcome:
        acquired = tuple(raw_ids)
        return (
            await self.materialize_retained_raw_ids(
                acquired,
                on_terminal_refusal=on_terminal_refusal,
                on_dependency_refusal=on_dependency_refusal,
                on_membership_refusal=on_membership_refusal,
                before_publication=before_publication,
            )
        ).outcome

    async def materialize_retained_raw_ids(
        self,
        raw_ids: Sequence[str],
        *,
        select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
        on_dependency_refusal: Callable[[RetainedRawDependencyRefusalError], None] | None = None,
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None] | None = None,
        before_publication: Callable[[], None] | None = None,
    ) -> RetainedMaterializationResult:
        acquired = tuple(raw_ids)
        await self._write_coordinator.run_sync(
            "live.retained.destination", self._archive.retained_destination(before_publication)
        )
        return await self._materialize_retained_raw_ids(
            acquired,
            select_retained_raw_ids=select_retained_raw_ids,
            on_terminal_refusal=on_terminal_refusal,
            on_dependency_refusal=on_dependency_refusal,
            on_membership_refusal=on_membership_refusal,
            before_publication=before_publication,
        )

    async def replay_retained_raw_ids(
        self,
        raw_ids: Sequence[str],
        *,
        select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
        on_dependency_refusal: Callable[[RetainedRawDependencyRefusalError], None] | None = None,
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None] | None = None,
        before_publication: Callable[[], None] | None = None,
    ) -> RetainedReplayOutcome:
        return (
            await self._materialize_retained_raw_ids(
                raw_ids,
                select_retained_raw_ids=select_retained_raw_ids,
                on_terminal_refusal=on_terminal_refusal,
                on_dependency_refusal=on_dependency_refusal,
                on_membership_refusal=on_membership_refusal,
                before_publication=before_publication,
            )
        ).outcome

    async def _materialize_retained_raw_ids(
        self,
        raw_ids: Sequence[str],
        *,
        select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
        on_dependency_refusal: Callable[[RetainedRawDependencyRefusalError], None] | None = None,
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None] | None = None,
        before_publication: Callable[[], None] | None = None,
    ) -> RetainedMaterializationResult:
        """Settle real selected replay receipts, including preparatory Source phases."""
        selected = tuple(dict.fromkeys(raw_ids))
        if any(not raw_id for raw_id in selected):
            raise ValueError("retained observation IDs must be non-empty")
        await self._prepare_cold_destination(before_publication)
        width = self._compute_adapter.snapshot().by_class("incremental-background").ceiling_slots
        results: list[RetainedMaterializationResult] = []
        remaining = list(selected)
        del selected, raw_ids
        async with self._converge_lock:
            while remaining:
                # This is an admission window, not an input cap. Component
                # expansion can legitimately exceed it; every suffix is kept.
                offered = tuple(remaining[:width])
                scope_operand = pickle.dumps(offered, protocol=pickle.HIGHEST_PROTOCOL)
                # Default sidecar ownership starts from this window. A caller's
                # explicit dependency selection is preserved before expansion.
                selection = select_retained_raw_ids or self._archive.sidecar_owner_selector(offered)
                capture = self._archive.neutral_capture_operation(
                    offered,
                    destination=self._archive.destination_adapter,
                    require_authority=self._require_source_frontier_authority,
                    selection=selection,
                )
                # Capture has no publication. Await the compute future itself:
                # coordinator receipt delivery precedes physical slot release.
                captured = self._compute_adapter.submit(
                    propagate(capture),
                    admission_class="incremental-background",
                    estimated_bytes=len(scope_operand),
                    exclusive_bytes=True,
                )
                try:
                    page = await captured.wait()
                except BaseException as capture_failure:
                    if captured.future.done() and captured.future.exception() is None:
                        abandoned = captured.future.result()
                        if abandoned is not None:
                            try:
                                abandoned.close()
                            except BaseException as cleanup:
                                raise BaseExceptionGroup(
                                    "capture cancellation and cleanup failed", [capture_failure, cleanup]
                                ) from capture_failure
                    raise
                primary: BaseException | None = None
                try:
                    if page is not None:
                        await self._archive.parse_neutral_page(page)
                    retained: list[RawObservationReplacement] = []
                    replay = self._archive.retained_replay_operation(
                        scope_operand,
                        retained=retained,
                        destination=self._archive.destination_adapter,
                        require_authority=self._require_source_frontier_authority,
                        select_retained_raw_ids=selection,
                        on_terminal_refusal=on_terminal_refusal,
                        on_dependency_refusal=on_dependency_refusal,
                        on_membership_refusal=on_membership_refusal,
                        before_publication=before_publication,
                        neutral_page=page,
                    )
                    result = await self.run_prepared_sync(
                        "watcher.live_ingest.retained",
                        replay,
                        settlement_owners=partial(retained_settlement_owners, retained),
                        estimated_bytes=len(scope_operand),
                    )
                    results.append(result)
                    considered = {*offered, *result.considered_raw_ids}
                    remaining = [raw_id for raw_id in remaining if raw_id not in considered]
                except BaseException as failure:
                    primary = failure
                    raise
                finally:
                    if page is not None:
                        try:
                            page.close()
                        except BaseException as cleanup:
                            if primary is not None:
                                raise BaseExceptionGroup(
                                    "retained page and cleanup failed", [primary, cleanup]
                                ) from primary
                            raise
        return self._archive.combine_retained_results(results)

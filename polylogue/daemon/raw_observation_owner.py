"""Daemon composition for one exact raw-observation derivation.

Raw preparation is deliberately independent of the daemon writer.  The only
writer admission is the adapter's one-observation publication, forwarded from
the bounded compute worker through :class:`DaemonWriteThreadBridge`.
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
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeVar

from polylogue.core import compute
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.compute_cancel import compute_cancel
from polylogue.core.raw_failure_evidence import (
    CohortMembershipRefusalError,
    RetainedRawDecodeRefusalError,
    RetainedRawDependencyRefusalError,
)
from polylogue.core.stage_admission import stage_write_admission
from polylogue.core.write_lease import adopt_write_lease
from polylogue.daemon.convergence import DaemonConverger, DerivationConvergenceOwner
from polylogue.daemon.derivation import Budget, DerivationFrame, DerivationReport
from polylogue.daemon.drive_catchup import DriveCatchupExecution
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.logging import propagate
from polylogue.operations.raw_observation_derivation import (
    RAW_OBSERVATION_DOMAIN,
    make_raw_observation_derivation,
    publish_raw_observation_once,
    raw_observation_frame,
)

if TYPE_CHECKING:
    from polylogue.sources.revision_backfill import PreparedRevisionReplayResult, RevisionCensusResult
    from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead


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
    ) -> None:
        self._archive_root = archive_root
        self._compute_adapter = compute_adapter
        self._write_bridge = write_bridge
        self._write_coordinator = write_coordinator
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

        def submit_worker(worker: Callable[[], None]) -> Future[None]:
            return self._compute_adapter.submit(
                propagate(worker),
                admission_class="incremental-background",
                estimated_bytes=estimated_bytes,
                exclusive_bytes=True,
                cancellation=cancellation,
            ).future

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
        return await DriveCatchupExecution(self._write_coordinator, compute_adapter=self._compute_adapter).settle(
            pending, label=actor, cancel_requested=request_cancel
        )

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

    async def ingest_append_plans(self, owner: _AppendIngestOwner, plans: list[_AppendPlan]) -> _AppendResult:
        """Acquire, prepare and publish append raws under the existing physical worker owner."""
        from polylogue.sources.live.append_ingest import ingest_append_plans

        retained: list[RawObservationReplacement] = []
        acquired_inputs: list[tuple[str, bytes, int]] = []

        def converge_raw(root: Path, raw_id: str, plan: _AppendPlan) -> None:
            if root.resolve() != self._archive_root.resolve():
                raise ValueError("append raw belongs to a different archive owner")
            self._require_source_frontier_authority(raw_id)
            # These operands belong to this still-admitted task. A future plan
            # is not prepaid for a late original CAS read until it is acquired.
            acquired_inputs.append((raw_id, bytes.fromhex(plan.payload_hash), len(plan.payload)))
            publish_raw_observation_once(
                root,
                raw_id,
                retained_replacements=retained,
                compute_adapter=self._compute_adapter,
                prepaid_blob_inputs=tuple(acquired_inputs),
            )

        return await self.run_prepared_sync(
            "watcher.live_ingest.append",
            lambda: ingest_append_plans(owner, plans, converge_raw=converge_raw),
            settlement_owners=lambda: tuple(
                replacement
                for replacement in retained
                if replacement.reference_seal is None or not replacement.reference_seal._closed
            ),
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
            adapter, index_path = self._destination_adapter()
            owner = DerivationConvergenceOwner(
                DaemonConverger((), derivations=(adapter,)),
                compute_adapter=self._compute_adapter,
                write_bridge=self._write_bridge,
            )
            frame = raw_observation_frame(
                self._archive_root,
                raw_ids=(raw_id,),
                index_db_path=index_path,
            )
            return await owner.converge(
                frame,
                budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
                domains=(RAW_OBSERVATION_DOMAIN,),
                resume=False,
                estimated_bytes=0,
                exclusive_bytes=True,
            )

    async def _prepare_cold_destination(self, before_publication: Callable[[], None] | None = None) -> None:
        """Establish the original cold writer profile before recording read identity."""
        from polylogue.sources.live.cold_build import active_cold_build_generation

        cold_build = active_cold_build_generation(self._archive_root)
        if cold_build is None:
            return

        def establish() -> None:
            if before_publication is not None:
                before_publication()
            with cold_build.open_writer():
                pass

        await self._write_coordinator.run_sync("retained.cold.destination", establish)

    def _destination_adapter(self) -> tuple[RawObservationDerivation, Path | None]:
        """Bind preparation and its frame to the actual registered destination."""
        from polylogue.sources.live.cold_build import active_cold_build_generation

        cold_build = active_cold_build_generation(self._archive_root)
        generation = None if cold_build is None else cold_build.generation
        index_path = None if generation is None else Path(generation.index_path)
        return (
            make_raw_observation_derivation(
                self._archive_root,
                compute_adapter=self._compute_adapter,
                index_db_path=index_path,
                owned_generation=generation,
            ),
            index_path,
        )

    def _require_source_frontier_authority(self, raw_id: str) -> None:
        """Refuse exactly the raw paths the durable frontier cannot authorize.

        Fair intake owns retry/isolation of this typed refusal.  The check is
        deliberately at the canonical owner boundary so periodic and whale
        callers cannot select a raw observation through a legacy scanner and
        then publish it without the source-frontier proof.
        """
        from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation
        from polylogue.readiness.capability import raw_frontier_source_selection_refusal

        refusal = raw_frontier_source_selection_refusal(self._archive_root, raw_ids=(raw_id,))
        if refusal.unattributed_reason is not None:
            raise RuntimeError(f"raw observation source-selection gate blocked: {refusal.unattributed_reason}")
        if not refusal.source_paths:
            return
        source_path = (
            make_raw_observation_derivation(self._archive_root, compute_adapter=self._compute_adapter)
            .source_paths((raw_id,))
            .get(raw_id)
        )
        if source_path in refusal.source_paths:
            raise RuntimeError(
                f"raw observation source-selection gate blocked: raw {raw_id} is on refused source path {source_path}"
            )

    async def ingest_retained_raw_ids(
        self,
        raw_ids: Sequence[str],
        *,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
        on_dependency_refusal: Callable[[RetainedRawDependencyRefusalError], None] | None = None,
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None] | None = None,
        before_publication: Callable[[], None] | None = None,
    ) -> tuple[PreparedRevisionReplayResult, ...]:
        from polylogue.sources.live.sidecar_resolution import select_retained_claude_sidecar_owner_raw_ids

        acquired = tuple(raw_ids)

        def prepare_destination() -> None:
            # Establish the selected writer profile and runtime indexes
            # before the retained preparation records its original file identity.
            # Source-only acquisition deliberately never opened this destination.
            from polylogue.sources.live.cold_build import active_cold_build_generation
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

            if before_publication is not None:
                before_publication()
            cold_build = active_cold_build_generation(self._archive_root)
            with (
                ArchiveStore.open_existing(self._archive_root, read_only=False)
                if cold_build is None
                else cold_build.open_writer()
            ):
                pass

        await self._write_coordinator.run_sync("live.retained.destination", prepare_destination)
        return await self.replay_retained_raw_ids(
            acquired,
            on_terminal_refusal=on_terminal_refusal,
            on_dependency_refusal=on_dependency_refusal,
            on_membership_refusal=on_membership_refusal,
            before_publication=before_publication,
            select_retained_raw_ids=lambda reader: tuple(
                dict.fromkeys((*acquired, *select_retained_claude_sidecar_owner_raw_ids(reader, acquired)))
            ),
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
    ) -> tuple[PreparedRevisionReplayResult, ...]:
        """Settle real selected replay receipts, including preparatory Source phases.

        The selector borrows each canonical Raw preparation's original reader;
        it never returns an archive handle or substitutes publication authority.
        The original semantic inputs detect an unchanged preparatory phase. Actual
        Source receipts and the new phase's original witnesses own every effect.
        """
        from polylogue.core.compute_cancel import check_compute_cancelled
        from polylogue.core.stage_admission import admit_stage_write
        from polylogue.sources.revision_backfill import PreparedRevisionReplayResult, RetainedPreparationRetryableError

        selected = tuple(dict.fromkeys(raw_ids))
        if any(not raw_id for raw_id in selected):
            raise ValueError("retained observation IDs must be non-empty")
        await self._prepare_cold_destination(before_publication)
        scope_operand = pickle.dumps(selected, protocol=pickle.HIGHEST_PROTOCOL)
        del selected, raw_ids
        retained: list[RawObservationReplacement] = []

        def replay() -> tuple[PreparedRevisionReplayResult, ...]:
            adapter, index_path = self._destination_adapter()

            def retire_original(current: RawObservationReplacement, primary: BaseException | None = None) -> None:
                try:
                    current.close()
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup(
                            "retained operation and physical retirement failed", [primary, cleanup]
                        ) from primary
                    raise
                retained.remove(current)

            results: list[PreparedRevisionReplayResult] = []
            visited: set[str] = set()
            refused_ids: set[str] = set()
            dependency_blocked_ids: set[str] = set()
            selected = pickle.loads(scope_operand)
            for raw_id in selected:
                if raw_id in visited:
                    continue
                self._require_source_frontier_authority(raw_id)
                previous_progress: tuple[str, tuple[object, ...]] | None = None
                while True:
                    check_compute_cancelled()
                    original_inputs: list[tuple[object, ...]] = []
                    original_keys: dict[str, tuple[str, ...]] = {}

                    def select_original(
                        reader: PreparedSessionSourceRead,
                        *,
                        selected_raw_id: str = raw_id,
                        captured_inputs: list[tuple[object, ...]] = original_inputs,
                        captured_keys: dict[str, tuple[str, ...]] = original_keys,
                    ) -> Sequence[str]:
                        # Exclude only explicit selections whose actual original
                        # Source receipt still refuses. Canonical dependencies are
                        # expanded afterward and retain their refusal obligation.
                        for refused_id in tuple(refused_ids):
                            if reader.raw_terminal_decode_refusal(refused_id) is None:
                                refused_ids.remove(refused_id)
                        selected_ids = tuple(
                            item
                            for item in (
                                selected_raw_id,
                                *(select_retained_raw_ids(reader) if select_retained_raw_ids else ()),
                            )
                            if item not in refused_ids and item not in dependency_blocked_ids
                        )
                        expanded, _member_keys = reader.expand_raw_membership_selection(selected_ids)
                        for dependency_id in expanded:
                            if dependency_id in refused_ids:
                                dependency = reader.raw_terminal_decode_refusal(dependency_id)
                                if dependency is not None:
                                    raise RetainedRawDependencyRefusalError(
                                        selected_raw_id,
                                        tuple(sorted(reader.raw_selection_values("keys", (selected_raw_id,)))),
                                        dependency,
                                    )
                        logical_keys = reader.raw_revision_rebuild_logical_keys(expanded)
                        captured_keys.update(
                            (item, tuple(sorted(reader.raw_selection_values("keys", (item,))))) for item in expanded
                        )
                        current = tuple((item, reader.raw_parser_census_is_current(item)) for item in expanded)
                        captured_inputs.append(
                            (
                                tuple((item, reader.raw_revision_descriptor(item)) for item in expanded),
                                current,
                                reader.raw_membership_census_rows(expanded),
                                tuple((key, reader.raw_revision_replay_plan(key)) for key in logical_keys)
                                if all(valid for _item, valid in current)
                                else (),
                            )
                        )
                        return expanded

                    frame = raw_observation_frame(self._archive_root, raw_ids=(raw_id,), index_db_path=index_path)
                    try:
                        replacement = adapter.compute(
                            frame, raw_id, replay_current=True, select_retained_raw_ids=select_original
                        )
                    except RetainedRawDecodeRefusalError as refusal:
                        # The compute wrapper has settled the original parent.
                        # Its reader captured the exact subject keys before the
                        # current Source receipt refused this selected input.
                        if on_terminal_refusal is None or refusal.raw_id not in original_keys:
                            raise
                        on_terminal_refusal(original_keys[refusal.raw_id], refusal)
                        visited.add(refusal.raw_id)
                        refused_ids.add(refusal.raw_id)
                        if refusal.raw_id == raw_id:
                            break
                        continue
                    except RetainedRawDependencyRefusalError as refusal:
                        # compute settles its original parent before propagating
                        # this exact error; failed settlement remains a group.
                        if on_dependency_refusal is None:
                            raise refusal.dependency from refusal
                        on_dependency_refusal(refusal)
                        dependency_blocked_ids.add(raw_id)
                        visited.add(raw_id)
                        break
                    retained.append(replacement)
                    if replacement.prepared_key_refusals and on_membership_refusal is None:
                        # Strict callers receive the original per-key outcome
                        # only after its original preparation physically closes.
                        membership_error = replacement.prepared_key_refusals[0]
                        retire_original(replacement, membership_error)
                        raise membership_error
                    phase = (
                        "census"
                        if replacement.needs_source_census
                        else ("classification" if replacement.needs_source_classification else "replay")
                    )
                    if len(original_inputs) != 1:
                        progress_error = RetainedPreparationRetryableError(
                            "retained preparation omitted its original phase inputs"
                        )
                        retire_original(replacement, progress_error)
                        raise progress_error
                    progress_operand = (phase, original_inputs[0])
                    if previous_progress == progress_operand:
                        progress_error = RetainedPreparationRetryableError(
                            "accepted preparatory Source phase left identical original inputs"
                        )
                        retire_original(replacement, progress_error)
                        raise progress_error
                    phases: list[
                        tuple[
                            Literal["census", "classification", "replay"],
                            RevisionCensusResult | PreparedRevisionReplayResult,
                        ]
                    ] = []

                    def receive(
                        phase: Literal["census", "classification", "replay"],
                        receipt: RevisionCensusResult | PreparedRevisionReplayResult,
                        received: list[
                            tuple[
                                Literal["census", "classification", "replay"],
                                RevisionCensusResult | PreparedRevisionReplayResult,
                            ]
                        ] = phases,
                    ) -> None:
                        received.append((phase, receipt))

                    publication_failures: list[BaseException] = []

                    def publish_original(
                        current_frame: DerivationFrame = frame,
                        current_replacement: RawObservationReplacement = replacement,
                        current_failures: list[BaseException] = publication_failures,
                        current_receive: Callable[
                            [
                                Literal["census", "classification", "replay"],
                                RevisionCensusResult | PreparedRevisionReplayResult,
                            ],
                            None,
                        ] = receive,
                    ) -> bool:
                        if before_publication is not None:
                            before_publication()
                        return adapter.publish(
                            current_frame,
                            current_replacement,
                            phase_receipt=current_receive,
                            publication_failure=current_failures.append,
                        )

                    try:
                        try:
                            published = admit_stage_write(
                                "watcher.live_ingest.retained.publish",
                                publish_original,
                            )
                        except BaseException as primary:
                            # Admission and caller validation can fail before
                            # the publisher receives its original carrier.
                            retire_original(replacement, primary)
                            raise
                    except RetainedRawDecodeRefusalError as refusal:
                        if on_terminal_refusal is None:
                            raise
                        if replacement.reference_seal is not None and not replacement.reference_seal._closed:
                            raise
                        on_terminal_refusal(original_keys[refusal.raw_id], refusal)
                        visited.add(refusal.raw_id)
                        refused_ids.add(refusal.raw_id)
                        if refusal.raw_id == raw_id:
                            break
                        # A selected independent refusal does not settle the
                        # healthy subject. Its next original selection excludes
                        # that offered input but still expands dependencies.
                        continue
                    # publish physically settles its same original parent.
                    # A failed close raises before this removal/result.
                    retire_original(replacement)
                    if published:
                        terminal = [receipt for phase, receipt in phases if phase == "replay"]
                        if len(terminal) != 1 or not isinstance(terminal[0], PreparedRevisionReplayResult):
                            raise RetainedPreparationRetryableError("retained replay lacks its actual terminal receipt")
                        if replacement.prepared_key_refusals:
                            assert on_membership_refusal is not None
                            for membership_refusal in replacement.prepared_key_refusals:
                                on_membership_refusal(membership_refusal)
                        results.append(terminal[0])
                        visited.update(replacement.raw_ids)
                        break
                    if publication_failures:
                        raise publication_failures[0]
                    if len(phases) != 1 or phases[0][0] not in ("census", "classification"):
                        raise RetainedPreparationRetryableError(
                            "retained publication deferred without accepted Source progress"
                        )
                    previous_progress = progress_operand
            return tuple(results)

        async with self._converge_lock:
            return await self.run_prepared_sync(
                "watcher.live_ingest.retained",
                replay,
                settlement_owners=lambda: tuple(
                    replacement
                    for replacement in retained
                    if replacement.reference_seal is None or not replacement.reference_seal._closed
                ),
                estimated_bytes=len(scope_operand),
            )


T = TypeVar("T")


P = ParamSpec("P")

if TYPE_CHECKING:
    from polylogue.core.sql_settlement import SQLCustodyOwner
    from polylogue.sources.live.append_ingest import _AppendIngestOwner
    from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
    from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement

"""Archive-facing work of the resident raw-observation owner.

The daemon owner (``polylogue.daemon.raw_observation_owner``) keeps write
admission, prepared-worker settlement and convergence orchestration. This
module holds everything that reads or prepares Source, Index and cold-build
state, and returns the exact operations the daemon runs on its admitted
workers. It never imports the daemon: the daemon passes its collaborators in.
"""

from __future__ import annotations

import pickle
import sqlite3
from builtins import BaseExceptionGroup
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.core.compute import (
    BoundedComputeAdapter,
    DaemonBackpressureError,
    DaemonOperationCancelled,
    SubmittedOperation,
)
from polylogue.core.enums import ValidationMode
from polylogue.core.raw_failure_evidence import (
    CohortMembershipRefusalError,
    RetainedRawDecodeRefusalError,
    RetainedRawDependencyRefusalError,
)
from polylogue.logging import WARNING, emit
from polylogue.operations.raw_observation_derivation import (
    make_raw_observation_derivation,
    publish_raw_observation_once,
    raw_observation_frame,
)

if TYPE_CHECKING:
    from polylogue.core.sql_settlement import SQLCustodyOwner
    from polylogue.sources.live.append_ingest import _AppendIngestOwner
    from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.sources.revision_backfill import (
        PreparedRevisionReplayResult,
        RetainedReplayOutcome,
        RevisionCensusResult,
    )
    from polylogue.storage.derived.raw import NeutralRawPreparation, RawObservationDerivation, RawObservationReplacement
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
    from polylogue.storage.sqlite.reference_seal import IndexMutationDestination

    #: Public annotation names for the daemon owner, which may not reach the
    #: source or storage packages directly.
    AppendIngestOwner: TypeAlias = _AppendIngestOwner
    AppendPlan: TypeAlias = _AppendPlan
    AppendResult: TypeAlias = _AppendResult

ReplayPhase = Literal["census", "classification", "replay"]
DestinationAdapter = Callable[[], "tuple[RawObservationDerivation, Path | None, IndexMutationDestination | None]"]


@dataclass(frozen=True, slots=True)
class RetainedMaterializationResult:
    """Replay receipts plus the exact Index destination used by the Raw owner.

    ``None`` names the operation's already-pinned active Index. An inactive
    candidate is carried by its existing ``IndexMutationDestination`` owner.
    """

    outcome: RetainedReplayOutcome
    index_destination: IndexMutationDestination | None
    considered_raw_ids: tuple[str, ...] = ()


def retained_settlement_owners(retained: Sequence[RawObservationReplacement]) -> tuple[SQLCustodyOwner, ...]:
    """Replacements whose original preparation has not physically settled."""
    return tuple(
        replacement
        for replacement in retained
        if replacement.reference_seal is None or not replacement.reference_seal._closed
    )


def _publish_after(before_publication: Callable[[], None] | None, publish: Callable[[], bool]) -> bool:
    if before_publication is not None:
        before_publication()
    return publish()


def _receive_phase(
    received: list[tuple[ReplayPhase, RevisionCensusResult | PreparedRevisionReplayResult]],
    phase: ReplayPhase,
    receipt: RevisionCensusResult | PreparedRevisionReplayResult,
) -> None:
    received.append((phase, receipt))


def _isolates_as_raw_failure(failure: Exception) -> bool:
    """Whether a preparation failure belongs to its raw rather than to the whole page.

    Cancellation, compute backpressure, a missing writer, archive storage
    faults, SQLite state and failed physical cleanup hold for every raw of the
    page alike, so they stop it. Any other failure is that raw's own retryable
    outcome.
    """
    from polylogue.core.compute import DaemonBackpressureError, DaemonOperationCancelled
    from polylogue.core.storage_faults import storage_fault_kind
    from polylogue.storage.sqlite.write_lease import UnleasedWriteError

    if isinstance(
        failure,
        (BaseExceptionGroup, DaemonOperationCancelled, DaemonBackpressureError, UnleasedWriteError, sqlite3.Error),
    ):
        return False
    return storage_fault_kind(failure) is None


def _preparation_failure_operation(failure: Exception) -> str:
    """Public producer code identity, without traceback paths or operand text."""
    operation = "raw_preparation"
    traceback = failure.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__")
        if isinstance(module, str) and module.startswith("polylogue."):
            operation = traceback.tb_frame.f_code.co_qualname
        traceback = traceback.tb_next
    return operation


class RawObservationArchiveWork:
    """Source/Index/cold-build work for one archive's raw-observation owner."""

    def __init__(
        self,
        archive_root: Path,
        *,
        compute_adapter: BoundedComputeAdapter,
        validation_mode: ValidationMode = ValidationMode.ADVISORY,
    ) -> None:
        self._archive_root = archive_root
        self._compute_adapter = compute_adapter
        self._validation_mode = validation_mode

    def require_source_frontier_authority(self, raw_id: str) -> None:
        """Refuse exactly the raw paths the durable frontier cannot authorize.

        Fair intake owns retry/isolation of this typed refusal.  The check is
        deliberately at the canonical owner boundary so periodic and whale
        callers cannot select a raw observation through a legacy scanner and
        then publish it without the source-frontier proof.
        """
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

    def cold_destination(self, before_publication: Callable[[], None] | None = None) -> Callable[[], None] | None:
        """The writer operation establishing the original cold profile, if a cold build is active."""
        from polylogue.sources.live.cold_build import active_cold_build_generation

        cold_build = active_cold_build_generation(self._archive_root)
        if cold_build is None:
            return None

        def establish() -> None:
            if before_publication is not None:
                before_publication()
            with cold_build.open_writer():
                pass

        return establish

    def retained_destination(self, before_publication: Callable[[], None] | None = None) -> Callable[[], None]:
        """The writer operation that opens the selected destination before retained preparation.

        It establishes the writer profile and runtime indexes before the
        retained preparation records its original file identity; source-only
        acquisition deliberately never opened this destination.
        """

        def prepare() -> None:
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

        return prepare

    def destination_adapter(
        self,
    ) -> tuple[RawObservationDerivation, Path | None, IndexMutationDestination | None]:
        """Bind preparation, frame and receipt to the actual registered destination."""
        from polylogue.sources.live.cold_build import active_cold_build_generation
        from polylogue.storage.sqlite.reference_seal import IndexMutationDestination

        cold_build = active_cold_build_generation(self._archive_root)
        generation = None if cold_build is None else cold_build.generation
        index_path = None if generation is None else Path(generation.index_path)
        index_destination = None if generation is None else IndexMutationDestination.owned_inactive(generation)
        return (
            make_raw_observation_derivation(
                self._archive_root,
                compute_adapter=self._compute_adapter,
                index_db_path=index_path,
                owned_generation=generation,
                validation_mode=self._validation_mode,
            ),
            index_path,
            index_destination,
        )

    def append_plans_operation(
        self,
        owner: AppendIngestOwner,
        plans: list[AppendPlan],
        *,
        retained: list[RawObservationReplacement],
        require_authority: Callable[[str], None],
    ) -> Callable[[], AppendResult]:
        """Acquire, prepare and publish append raws; run on the admitted worker."""
        from polylogue.sources.live.append_ingest import ingest_append_plans

        acquired_inputs: list[tuple[str, bytes, int]] = []

        def converge_raw(root: Path, raw_id: str, plan: _AppendPlan) -> None:
            if root.resolve() != self._archive_root.resolve():
                raise ValueError("append raw belongs to a different archive owner")
            require_authority(raw_id)
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

        return lambda: ingest_append_plans(owner, plans, converge_raw=converge_raw)

    @staticmethod
    def sidecar_owner_selector(acquired: tuple[str, ...]) -> Callable[[PreparedSessionSourceRead], Sequence[str]]:
        """Select the acquired raws plus the Claude sidecar owners they imply."""
        from polylogue.sources.live.sidecar_resolution import select_retained_claude_sidecar_owner_raw_ids

        return lambda reader: tuple(
            dict.fromkeys((*acquired, *select_retained_claude_sidecar_owner_raw_ids(reader, acquired)))
        )

    def neutral_capture_operation(
        self,
        raw_ids: Sequence[str],
        *,
        destination: DestinationAdapter,
        require_authority: Callable[[str], None],
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
    ) -> Callable[[], NeutralRawPreparation | None]:
        """Capture only; the admitted creator retires all Source observers."""

        def capture() -> NeutralRawPreparation | None:
            for raw_id in raw_ids:
                require_authority(raw_id)
            adapter, _path, _destination = destination()
            try:
                return adapter.capture_neutral_raws(raw_ids, selection=selection)
            except Exception as failure:
                # Failed optional neutral work is still owed to the canonical
                # per-raw owner, which attributes its typed refusal/failure.
                if isinstance(
                    failure, (RetainedRawDecodeRefusalError, RetainedRawDependencyRefusalError)
                ) or _isolates_as_raw_failure(failure):
                    return None
                raise

        return capture

    async def parse_neutral_page(self, page: NeutralRawPreparation) -> None:
        """Submit closed-file parsers outside a parent reservation, then settle all.

        The existing adapter owns slots, complements and the byte envelope.
        Backpressure keeps the next job pending; completed artifacts remain
        disk-backed until the ordered canonical consumer takes their exact key.
        """
        pending: deque[tuple[tuple[object, ...], SubmittedOperation[PreparedJsonl]]] = deque()
        width = self._compute_adapter.snapshot().by_class("incremental-background").ceiling_slots
        primary: BaseException | None = None

        async def receive() -> None:
            key, submitted = pending[0]
            try:
                artifact = await submitted.wait()
            except Exception as failure:
                if not _isolates_as_raw_failure(failure):
                    raise
                # Canonical preparation retries and attributes this raw.
            else:
                page.artifacts[key] = artifact
            pending.popleft()

        try:
            for key, estimated_bytes, operation in page.parser_jobs(self._validation_mode):
                while True:
                    try:
                        submitted = self._compute_adapter.submit(
                            operation, admission_class="incremental-background", estimated_bytes=estimated_bytes
                        )
                    except DaemonBackpressureError:
                        if not pending:
                            raise
                        await receive()
                    else:
                        pending.append((key, submitted))
                        break
                if len(pending) >= width:
                    await receive()
            while pending:
                await receive()
        except BaseException as failure:
            primary = failure
            raise
        finally:
            # Request every cancellation before joining any creator. A cancelled
            # wait can still have produced files: retain those for page cleanup.
            failures: list[BaseException] = []
            for _key, submitted in pending:
                submitted.cancellation.cancel()
                submitted.retry_sql_settlement()
            for key, submitted in pending:
                try:
                    artifact = await submitted.wait()
                except DaemonOperationCancelled:
                    pass
                except BaseException as cleanup:
                    if cleanup is not primary:
                        failures.append(cleanup)
                    if submitted.future.done() and submitted.future.exception() is None:
                        page.artifacts[key] = submitted.future.result()
                else:
                    page.artifacts[key] = artifact
            if failures:
                raise BaseExceptionGroup(
                    "neutral parsing and creator settlement failed",
                    [*(() if primary is None else (primary,)), *failures],
                ) from primary

    @staticmethod
    def combine_retained_results(results: Sequence[RetainedMaterializationResult]) -> RetainedMaterializationResult:
        from polylogue.sources.revision_backfill import RetainedReplayOutcome

        destination = results[-1].index_destination if results else None
        return RetainedMaterializationResult(
            RetainedReplayOutcome(
                tuple(receipt for result in results for receipt in result.outcome.receipts),
                tuple(failure for result in results for failure in result.outcome.failures),
            ),
            destination,
            tuple(raw_id for result in results for raw_id in result.considered_raw_ids),
        )

    def retained_replay_operation(
        self,
        scope_operand: bytes,
        *,
        retained: list[RawObservationReplacement],
        destination: DestinationAdapter,
        require_authority: Callable[[str], None],
        select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None,
        on_dependency_refusal: Callable[[RetainedRawDependencyRefusalError], None] | None,
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None] | None,
        before_publication: Callable[[], None] | None,
        neutral_page: NeutralRawPreparation | None = None,
    ) -> Callable[[], RetainedMaterializationResult]:
        """Settle real selected replay receipts, including preparatory Source phases.

        The selector borrows each canonical Raw preparation's original reader;
        it never returns an archive handle or substitutes publication authority.
        The original semantic inputs detect an unchanged preparatory phase. Actual
        Source receipts and the new phase's original witnesses own every effect.

        One raw's failed preparation does not stop its page. A preparation that
        fails while censusing its page's identity-opaque siblings is prepared
        again with a frame scope of that raw alone, so a sibling's fault cannot
        block it; a raw that fails on its own (with the cohort its selection
        needs) is that raw's retryable failure, returned beside the receipts
        its siblings published.
        """
        archive_root = self._archive_root

        def replay() -> RetainedMaterializationResult:
            from polylogue.core.compute_cancel import check_compute_cancelled
            from polylogue.core.stage_admission import admit_stage_write
            from polylogue.sources.revision_backfill import (
                PreparedRevisionReplayResult,
                RetainedPreparationRetryableError,
                RetainedRawRetryableFailure,
                RetainedReplayOutcome,
            )

            adapter, index_path, index_destination = destination()

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
            failures: list[RetainedRawRetryableFailure] = []
            visited: set[str] = set()
            refused_ids: set[str] = set()
            dependency_blocked_ids: set[str] = set()
            failed_ids: set[str] = set()
            selected = pickle.loads(scope_operand)
            for raw_id in selected:
                if raw_id in visited:
                    continue
                require_authority(raw_id)
                # Set once this raw's page-scoped preparation has failed: it
                # prepares again with a frame scope of itself alone.
                isolated = False
                previous_progress: tuple[str, tuple[object, ...]] | None = None
                # After a lineage-deferral pass only its deferred children are
                # re-prepared; the keys that pass published are not replayed again.
                deferred_selection: tuple[str, ...] | None = None
                while True:
                    check_compute_cancelled()
                    original_inputs: list[tuple[object, ...]] = []
                    original_keys: dict[str, tuple[str, ...]] = {}

                    def select_original(
                        reader: PreparedSessionSourceRead,
                        *,
                        selected_raw_id: str = raw_id,
                        reselected: tuple[str, ...] | None = deferred_selection,
                        captured_inputs: list[tuple[object, ...]] = original_inputs,
                        captured_keys: dict[str, tuple[str, ...]] = original_keys,
                        isolated_scope: bool = isolated,
                    ) -> Sequence[str]:
                        # A preparation that widens to a lineage parent selects
                        # again on its new reader; the final attempt's capture
                        # is the original input.
                        captured_inputs.clear()
                        captured_keys.clear()
                        # Settled independent page offers need no second parse.
                        # Canonical dependencies expand afterward, so a settled
                        # member required by this unit retains its obligation.
                        for refused_id in tuple(refused_ids):
                            if reader.raw_terminal_decode_refusal(refused_id) is None:
                                refused_ids.remove(refused_id)
                        selected_ids = tuple(
                            item
                            for item in (
                                reselected
                                if reselected is not None
                                else (
                                    (selected_raw_id,)
                                    if isolated_scope
                                    else (
                                        selected_raw_id,
                                        *(select_retained_raw_ids(reader) if select_retained_raw_ids else ()),
                                    )
                                )
                            )
                            if item not in visited
                            and item not in refused_ids
                            and item not in dependency_blocked_ids
                            and item not in failed_ids
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

                    # The frame scope is the whole selection, so an opaque
                    # envelope's census can cover its unreplayed siblings --
                    # except inputs this operation already settled, which an
                    # independent sibling's census must not offer again.
                    frame_scope = (
                        (raw_id,)
                        if isolated
                        else tuple(
                            item
                            for item in selected
                            if item not in visited
                            and item not in refused_ids
                            and item not in dependency_blocked_ids
                            and item not in failed_ids
                        )
                    )
                    frame = raw_observation_frame(
                        archive_root,
                        raw_ids=frame_scope,
                        index_db_path=index_path,
                        validation_mode=self._validation_mode,
                    )
                    try:
                        replacement = adapter.compute(
                            frame,
                            raw_id,
                            replay_current=True,
                            select_retained_raw_ids=select_original,
                            neutral_page=neutral_page,
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
                    except Exception as failure:
                        # compute settles its original parent before
                        # propagating; nothing of this attempt is retained.
                        if not _isolates_as_raw_failure(failure):
                            raise
                        retry_single = not isolated and frame_scope != (raw_id,)
                        emit(
                            "storage.raw_observation.preparation_isolated",
                            level=WARNING,
                            outcome="degraded",
                            reason="page_scope_failed" if retry_single else "raw_preparation_failed",
                            error_type=type(failure).__name__,
                            operation=_preparation_failure_operation(failure),
                            phase="source_preparation",
                            productive_id=raw_id,
                            raws=len(frame_scope),
                        )
                        if retry_single:
                            # A census of the page's opaque siblings may have
                            # failed on one of them: prepare this raw alone.
                            isolated = True
                            continue
                        failures.append(RetainedRawRetryableFailure(raw_id, failure))
                        failed_ids.add(raw_id)
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
                    progress_operand = (phase, (original_inputs[0], replacement.raw_ids))
                    if previous_progress == progress_operand:
                        progress_error = RetainedPreparationRetryableError(
                            "accepted preparatory Source phase left identical original inputs"
                        )
                        retire_original(replacement, progress_error)
                        raise progress_error
                    phases: list[tuple[ReplayPhase, RevisionCensusResult | PreparedRevisionReplayResult]] = []
                    publication_failures: list[BaseException] = []
                    try:
                        try:
                            published = admit_stage_write(
                                "watcher.live_ingest.retained.publish",
                                partial(
                                    _publish_after,
                                    before_publication,
                                    partial(
                                        adapter.publish,
                                        frame,
                                        replacement,
                                        phase_receipt=partial(_receive_phase, phases),
                                        publication_failure=publication_failures.append,
                                    ),
                                ),
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
                    # Census and classification a preparation committed in
                    # place precede its own replay receipt.
                    replay_phases = [receipt for phase, receipt in phases if phase == "replay"]
                    source_phases = [phase for phase, _receipt in phases if phase != "replay"]
                    if (
                        len(replay_phases) == 1
                        and replacement.prepared_lineage_deferrals
                        and isinstance(replay_phases[0], PreparedRevisionReplayResult)
                    ):
                        # Parents published; their deferred children are
                        # re-prepared against them next. The deferral itself
                        # is the progress (a key is deferred at most once), so
                        # the identical-Source-inputs guard does not apply.
                        if replacement.prepared_key_refusals:
                            assert on_membership_refusal is not None
                            for membership_refusal in replacement.prepared_key_refusals:
                                on_membership_refusal(membership_refusal)
                        results.append(replay_phases[0])
                        visited.update(set(replacement.raw_ids).difference(replacement.lineage_deferred_raw_ids))
                        deferred_selection = replacement.lineage_deferred_raw_ids
                        previous_progress = None
                        continue
                    if replay_phases or not source_phases:
                        raise RetainedPreparationRetryableError(
                            "retained publication deferred without accepted Source progress"
                        )
                    previous_progress = progress_operand
            return RetainedMaterializationResult(
                RetainedReplayOutcome(tuple(results), tuple(failures)), index_destination, tuple(visited | failed_ids)
            )

        return replay


__all__ = [
    "AppendIngestOwner",
    "AppendPlan",
    "AppendResult",
    "PreparedRevisionReplayResult",
    "PreparedSessionSourceRead",
    "RawObservationArchiveWork",
    "RawObservationDerivation",
    "RawObservationReplacement",
    "RetainedMaterializationResult",
    "RetainedReplayOutcome",
    "retained_settlement_owners",
]

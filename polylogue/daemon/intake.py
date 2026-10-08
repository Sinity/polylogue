"""Fair, resumable intake across every class of acquirable work.

One dispatcher services the browser spool, the hook spool, configured
source acquisition, and admitted raw observations through domain adapters.
No class holds absolute priority over another: a permanently failing first
item, or a class with hundreds of thousands of pending files, cannot stop
bounded progress on its siblings.

Queue authority stays with the domain -- files on disk, source evidence
rows. Everything this module keeps is a disposable hint: deficits,
discovery position, attempt counts, the ready set. Delete all of it and a
restart resumes through bounded discovery, because correctness never
depended on it and no pass ever enumerates a whole spool.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, cast, runtime_checkable

from polylogue.core.compute_cancel import raise_if_operation_cancelled
from polylogue.core.raw_failure_evidence import PartialAdmission
from polylogue.daemon.observation import Observation, ObservationBoard, ObservationState
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
from polylogue.logging import ERROR, WARNING, emit

__all__ = [
    "AdmissionOutcome",
    "AdmissionResult",
    "FairIntakeDispatcher",
    "IntakeAdapter",
    "IntakeClassSpec",
    "IntakeItem",
    "IntakePass",
    "IntakeClassReport",
    "is_transient_admission_error",
]


@dataclass(frozen=True, slots=True)
class IntakeItem:
    """One unit of acquirable work, identified stably by its domain.

    ``item_id`` must be derivable again from the domain's own authority
    after a restart, so a duplicate delivery is recognizable rather than
    merely improbable.
    """

    item_id: str
    class_name: str
    payload: object = None
    estimated_cost: int = 1
    revision: str | None = None
    """The observed content revision behind ``item_id``, when the domain can
    see one (a file's size, mtime and inode). Isolation and failure streaks
    bind to it, so new content under the same identity is admitted afresh."""


class AdmissionOutcome(str, Enum):
    """What happened when an adapter tried to admit an item."""

    ADMITTED = "admitted"
    DUPLICATE = "duplicate"
    """Already admitted under this identity; acknowledging again is safe."""

    EXCLUDED = "excluded"
    """The domain durably refused this item; acknowledging is safe, but it is
    not progress.

    Distinct from :attr:`DUPLICATE` (polylogue-onbz3): a refused item was never
    admitted under any identity, so reporting it as a duplicate turns a pass
    that admitted nothing into one that claims to have re-seen prior work. The
    exclusion is durable (``live_cursor.excluded``), so the queue entry is
    released and the item is not retried, but it does not count toward
    :attr:`IntakePass.progressed`."""

    DEFERRED = "deferred"
    """The domain deliberately admitted nothing for this item this pass.

    Bounded backpressure, not a failure and not a refusal: the file carried no
    new authority-relevant content, so its queue entry is released and the
    domain's own retry evidence owns the follow-up. Counting it as
    :attr:`DUPLICATE` claimed the item had already been admitted under this
    identity, which is exactly the false-progress shape polylogue-onbz3
    removed for :attr:`EXCLUDED`."""

    RETRYABLE = "retryable"
    """This attempt failed; the item may succeed later."""

    TERMINAL = "terminal"
    """This item can never be admitted by this build."""

    CLASS_TERMINAL = "class_terminal"
    """The class itself cannot continue; it halts."""


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    outcome: AdmissionOutcome
    reason: str | None = None
    actual_cost: int | None = None
    transient: bool = True
    """For ``RETRYABLE``: whether the cause can clear on its own (lock,
    storage fault, backlog). A non-transient failure repeating with the same
    signature on an unchanged item is isolated instead of retried forever."""
    #: A RETRYABLE item the adapter never attempted (e.g. the daemon is
    #: degraded): it counts toward no attempt budget or cooldown.
    unattempted: bool = False
    #: For ``ADMITTED``: what the admission left out, when it took in only
    #: part of the item (a truncated final record). Counted as admitted and
    #: as partial, so a partial admission is never a plain success.
    partial: PartialAdmission | None = None

    @property
    def acknowledgeable(self) -> bool:
        """Whether the item's queue entry may be released."""
        return self.outcome in (
            AdmissionOutcome.ADMITTED,
            AdmissionOutcome.DUPLICATE,
            AdmissionOutcome.EXCLUDED,
            AdmissionOutcome.DEFERRED,
        )


@runtime_checkable
class IntakeAdapter(Protocol):
    """A domain's view of its own queue.

    ``discover`` must be bounded: it returns at most *limit* items from
    wherever it left off and never costs a full scan.
    """

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]: ...

    async def admit(self, item: IntakeItem) -> AdmissionResult: ...

    # Optional: ``admit_page(items) -> Mapping[item_id, AdmissionResult]``.
    # An adapter that defines it is handed the whole budget-bounded page at
    # once and pays its fixed per-batch cost once instead of once per item.
    # It must still report one outcome per item so deficit, retry and
    # isolation accounting stay per item; the dispatcher falls back to
    # ``admit`` per item when it is absent.

    async def acknowledge(self, item: IntakeItem) -> None:
        """Release the item's queue entry. Atomic and idempotent."""


DEFAULT_INTAKE_BYTE_BUDGET = 64 * 1024 * 1024
"""Payload bytes a single dispatcher pass may charge across all classes.

Every adapter denominates ``IntakeItem.estimated_cost`` in payload bytes and
reconciles against ``source_payload_read_bytes``, so the cycle budget shares
that unit. Declared once here; ``DaemonIntakeService`` reads it rather than
keeping a second copy.
"""

UNMEASURABLE_INTAKE_COST_BYTES = DEFAULT_INTAKE_BYTE_BUDGET
"""Byte cost charged by a class whose payload size cannot be measured.

A remote sync reports a changed-row count, never bytes, so it has no honest
estimate in the budget's unit. Charging the literal ``1`` it used to charge
(polylogue-swicx) let an arbitrarily large Drive sync consume one byte of a
64 MiB deficit and run again on every pass while its siblings waited. An
unmeasurable-size sync therefore reserves a full budget share rather than
under-reporting: it still runs, but takes its class's whole share of each
pass it runs in.
"""


def _reconciled_charge(actual_cost: int, estimated_cost: int, charge: int) -> int:
    """The deficit an admitted item finally costs, given what was charged up front.

    An item charged its whole estimate pays its measured cost. An oversized
    item, charged only the deficit that was left, pays at most that charge:
    a smaller measured cost is still refunded, a larger one is not carried
    into later passes.
    """
    if charge >= estimated_cost:
        return actual_cost
    return min(actual_cost, charge)


@dataclass(frozen=True, slots=True)
class IntakeClassSpec:
    """One scheduled class of intake work."""

    name: str
    adapter: IntakeAdapter
    weight: int = 1
    """Share of each cycle's budget, relative to sibling classes."""

    page_size: int = 32
    """Upper bound on one discovery call."""

    max_attempts: int = 3
    """Retryable attempts on one item identity before a process-local cooldown."""

    max_deterministic_cooldowns: int = 3
    """Cooldowns an unchanged item may spend on the same non-transient failure
    before it is isolated as a terminal refusal (polylogue-wyi9p)."""

    retry_cooldown_s: float = 5.0
    """Delay before retrying an item that exhausted ``max_attempts``."""


@dataclass(frozen=True, slots=True)
class IntakeClassReport:
    """What one class achieved in one pass."""

    name: str
    admitted: int = 0
    planned_count: int = 0
    discovery_duration_ms: float = 0.0
    admission_duration_ms: float = 0.0
    budget_blocked: bool = False
    page_full: bool = False
    partial_plan: bool = False
    duplicates: int = 0
    excluded: int = 0
    """Items the domain durably refused. Acknowledged, never counted as progress."""

    deferred: int = 0
    """Items the domain deliberately admitted nothing for. Not progress."""

    partially_admitted: int = 0
    """Admitted items that took in only part of their source. Also counted in ``admitted``."""

    retried: int = 0
    isolated: int = 0
    """Terminal items set aside for the remainder of this process."""

    discovered: int = 0
    discovery_failed: bool = False
    """Discovery raised, so every count here is an absence of measurement.

    A halted report already publishes unmeasured. This flag separates the
    third case -- a class that was scheduled, tried, and could not look --
    from a genuine zero. ``reason`` alone cannot: a halted report carries one
    too (polylogue-swicx).
    """

    estimated_cost: int = 0
    actual_cost: int = 0
    reconciled_cost: int = 0
    halted: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class IntakePass:
    """The result of one bounded dispatcher pass."""

    classes: tuple[IntakeClassReport, ...]
    skipped_halted: tuple[str, ...] = ()
    duration_s: float = 0.0

    @property
    def admitted(self) -> int:
        return sum(report.admitted for report in self.classes)

    @property
    def progressed(self) -> bool:
        """Whether this pass admitted new work.

        Re-recognising a duplicate is not progress: the caller shortens its
        sleep when a pass progressed, and counting duplicates here made a
        static source re-run discovery about twenty times a second forever
        (polylogue-swicx). An idle class backs off to the idle delay.
        """
        return any(report.admitted for report in self.classes)

    @property
    def quiescent(self) -> bool:
        """True only when every class saw a short, fully handled idle page."""
        if self.skipped_halted:
            return False
        return all(
            not (
                report.admitted
                or report.retried
                or report.deferred
                or report.isolated
                or report.discovery_failed
                or report.halted
                or report.budget_blocked
                or report.page_full
                or report.partial_plan
            )
            for report in self.classes
        )

    def report_for(self, name: str) -> IntakeClassReport | None:
        for report in self.classes:
            if report.name == name:
                return report
        return None

    def require_report(self, name: str) -> IntakeClassReport:
        """Return *name*'s report, or raise naming the class that is missing."""
        report = self.report_for(name)
        if report is None:
            raise KeyError(f"no intake class named {name!r} in this pass")
        return report


@dataclass
class _ClassRuntime:
    """Disposable scheduling hints for one class."""

    deficit: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
    retry_after: dict[str, float] = field(default_factory=dict)
    #: item_id -> the revision it was isolated at. A different revision of
    #: the same item is new content and releases the isolation.
    isolated: dict[str, str | None] = field(default_factory=dict)
    #: item_id -> ((revision, failure signature), consecutive cooldowns)
    exhaustions: dict[str, tuple[tuple[str, str], int]] = field(default_factory=dict)
    #: item_id -> the one deterministic failure signature seen in every
    #: result of its current attempt window, or ``None`` once any result in
    #: the window was transient or had another signature. Only a clean
    #: window extends the exhaustion streak.
    windows: dict[str, tuple[str, str] | None] = field(default_factory=dict)


def _item_revision(item: IntakeItem) -> str:
    """The revision an item's failures bind to; its cost when none is observed."""

    return item.revision if item.revision is not None else f"cost:{int(item.estimated_cost)}"


def _failure_signature(reason: str | None) -> str:
    """The stable part of a failure reason: its exception type when named."""

    text = str(reason or "")
    head, sep, _detail = text.partition(":")
    return head if sep else text


def is_transient_admission_error(exc: BaseException) -> bool:
    """Whether an escaped admission error can clear without any change.

    Lock contention, timeouts and storage faults are conditions of the host or
    archive. A KeyError, a TypeError or an SQLite error such as ``no such
    table`` is a defect that repeats identically forever, whatever its class.
    """

    from polylogue.core.sqlite_locking import is_transient_sqlite_lock
    from polylogue.core.storage_faults import ArchiveStorageFaultError, storage_fault_kind

    return (
        isinstance(exc, (OSError, TimeoutError, ArchiveStorageFaultError))
        or is_transient_sqlite_lock(exc)
        or storage_fault_kind(exc) is not None
    )


class FairIntakeDispatcher:
    """Deficit round-robin over intake classes.

    Each pass grants every schedulable class its weight, then spends the
    accumulated deficit on items that class discovered. A class that cannot
    make progress keeps its deficit but never consumes another class's, so
    the pass's cost stays bounded whatever any single class is doing.
    """

    def __init__(
        self,
        classes: Iterable[IntakeClassSpec],
        *,
        halts: HaltRegistry | None = None,
        board: ObservationBoard | None = None,
        frame: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._classes = tuple(classes)
        by_name = {spec.name: spec for spec in self._classes}
        if len(by_name) != len(self._classes):
            raise ValueError("duplicate intake class name")
        self._halts = halts
        self._board = board
        self._frame = frame
        self._clock = clock
        self._runtime = {spec.name: _ClassRuntime() for spec in self._classes}

    @property
    def classes(self) -> tuple[IntakeClassSpec, ...]:
        return self._classes

    def schedulable_classes(self) -> tuple[IntakeClassSpec, ...]:
        """Classes this pass may plan work for.

        A halted class is excluded here, at selection. Nothing downstream
        discovers its items, forms a batch from them, or takes the writer
        lease on its behalf.
        """
        return tuple(spec for spec in self._classes if not self._is_halted(spec.name))

    def isolated_items(self, class_name: str) -> frozenset[str]:
        """Item identities this process has set aside in *class_name*."""
        return frozenset(self._runtime[class_name].isolated)

    async def run_once(self, *, budget: int = DEFAULT_INTAKE_BYTE_BUDGET) -> IntakePass:
        """Run one bounded pass and return what each class achieved."""
        started = self._clock()
        schedulable = self.schedulable_classes()
        schedulable_names = {spec.name for spec in schedulable}
        skipped = tuple(spec.name for spec in self._classes if spec.name not in schedulable_names)
        reports: list[IntakeClassReport] = []

        total_weight = sum(spec.weight for spec in schedulable) or 1
        from polylogue.core.degraded import is_fully_degraded

        for spec in schedulable:
            if is_fully_degraded():
                # A class that just degraded the daemon (a structural database
                # error) ends the pass: later classes would still open archive
                # tiers. The service parks from the next tick.
                break
            runtime = self._runtime[spec.name]
            share = max(1, budget * spec.weight // total_weight)
            runtime.deficit += share
            class_started = self._clock()
            report = await self._service_class(spec, runtime)
            reports.append(report)
            if report.planned_count:
                duration_ms = max(0.0, (self._clock() - class_started) * 1000)
                emit(
                    "daemon.intake.page",
                    outcome="degraded"
                    if report.retried
                    or report.isolated
                    or report.halted
                    or report.excluded
                    or report.deferred
                    or report.partially_admitted
                    else "ok",
                    component=spec.name,
                    files=report.planned_count,
                    bytes=report.estimated_cost,
                    duration_ms=duration_ms,
                    succeeded=report.admitted,
                    failed=report.isolated,
                    retried=report.retried,
                    refused=report.excluded,
                    deferred=report.deferred,
                    partially_admitted=report.partially_admitted,
                    stage_timings_ms={
                        "discovery": report.discovery_duration_ms,
                        "admission": report.admission_duration_ms,
                    },
                )

        for name in skipped:
            record = self._halts.record_for(unit_id(UnitKind.INTAKE_CLASS, name)) if self._halts else None
            reports.append(
                IntakeClassReport(
                    name=name,
                    halted=True,
                    reason=f"{record.reason.value}: {record.message}" if record else "halted",
                )
            )

        result = IntakePass(
            classes=tuple(reports),
            skipped_halted=skipped,
            duration_s=self._clock() - started,
        )
        self._publish(result)
        return result

    # -- internals ----------------------------------------------------

    async def _service_class(self, spec: IntakeClassSpec, runtime: _ClassRuntime) -> IntakeClassReport:
        if runtime.deficit <= 0:
            return IntakeClassReport(name=spec.name, budget_blocked=True)

        # ``page_size`` bounds the discovery call in rows; ``deficit`` is
        # denominated in payload bytes, so it cannot bound a row count. The
        # page plan below is what spends the deficit.
        limit = spec.page_size
        discovery_started = self._clock()
        try:
            page: list[IntakeItem] = list(await spec.adapter.discover(limit=limit))
        except Exception as exc:
            raise_if_operation_cancelled(exc)
            emit(
                "daemon.intake.discovery_failed",
                level=WARNING,
                outcome="error",
                reason="discovery_raised",
                component=spec.name,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            return IntakeClassReport(name=spec.name, discovery_failed=True, reason=f"discovery failed: {exc}")
        discovery_duration_ms = max(0.0, (self._clock() - discovery_started) * 1000)

        admitted = duplicates = excluded = deferred = partially_admitted = retried = isolated = 0
        estimated_cost = actual_cost = 0
        # Plan the page against the class deficit first, then admit the whole
        # plan in one adapter call. The deficit is denominated in payload
        # bytes, so the plan is bytes-aware by construction: it fills up to
        # the class share and stops. Every per-batch fixed cost the domain
        # pays -- the writer hold, the tier bootstrap, one convergence pass --
        # is then paid once per page rather than once per file, while the
        # per-item accounting below is unchanged.
        planned: list[IntakeItem] = []
        charges: dict[str, int] = {}
        for item in page:
            if runtime.deficit <= 0:
                break
            if item.item_id in runtime.isolated:
                if runtime.isolated[item.item_id] == _item_revision(item):
                    continue
                # The item changed since it was set aside: new content gets a
                # fresh admission and a fresh failure streak.
                del runtime.isolated[item.item_id]
                runtime.exhaustions.pop(item.item_id, None)
                runtime.attempts.pop(item.item_id, None)
                runtime.windows.pop(item.item_id, None)
                runtime.retry_after.pop(item.item_id, None)
            retry_after = runtime.retry_after.get(item.item_id)
            if retry_after is not None:
                if self._clock() < retry_after:
                    continue
                runtime.retry_after.pop(item.item_id, None)
            item_cost = max(1, int(item.estimated_cost))
            # A single item may be larger than the per-class byte budget. It
            # still gets one bounded admission attempt; otherwise a large
            # but valid source would wait forever while its siblings consume
            # the deficit in later passes. This is also why a page is never
            # split below one item.
            if runtime.deficit < item_cost and planned:
                break
            # Charge the estimate before admission. An adapter cannot hide a
            # large item behind a cheap synthetic page identity. An item
            # larger than the whole remaining deficit is admitted alone and
            # charged that deficit: it takes this pass's share, not the
            # shares of the next thousand passes. Charging its full size
            # left the class in debt for as many passes as the item is
            # larger than one share, so the retry of a failed whale -- or
            # the next whale -- waited on unrelated passes for an hour.
            charge = min(item_cost, runtime.deficit)
            runtime.deficit -= charge
            charges[item.item_id] = charge
            estimated_cost += item_cost
            planned.append(item)

        admission_started = self._clock()
        for item, result in zip(planned, await self._admit_page(spec, planned), strict=True):
            item_cost = max(1, int(item.estimated_cost))
            charge = charges[item.item_id]
            if result.outcome is AdmissionOutcome.CLASS_TERMINAL:
                self._halt_class(spec.name, result.reason or "class reported terminal failure")
                return IntakeClassReport(
                    name=spec.name,
                    admitted=admitted,
                    planned_count=len(planned),
                    discovery_duration_ms=discovery_duration_ms,
                    admission_duration_ms=max(0.0, (self._clock() - admission_started) * 1000),
                    page_full=len(page) >= limit,
                    partial_plan=len(planned) < len(page),
                    duplicates=duplicates,
                    excluded=excluded,
                    deferred=deferred,
                    partially_admitted=partially_admitted,
                    retried=retried,
                    isolated=isolated,
                    discovered=len(page),
                    estimated_cost=estimated_cost,
                    actual_cost=actual_cost,
                    reconciled_cost=actual_cost - estimated_cost,
                    halted=True,
                    reason=result.reason,
                )
            if result.acknowledgeable:
                await spec.adapter.acknowledge(item)
                runtime.attempts.pop(item.item_id, None)
                runtime.windows.pop(item.item_id, None)
                runtime.retry_after.pop(item.item_id, None)
                # Only an uninterrupted failure streak may isolate an item.
                runtime.exhaustions.pop(item.item_id, None)
                # `or` would treat an explicit 0 (a deferred retry
                # placeholder reporting no work attempted) as absent and
                # substitute the full estimate, permanently consuming
                # deficit for work that never ran.
                item_actual_cost = item_cost if result.actual_cost is None else max(0, int(result.actual_cost))
                actual_cost += item_actual_cost
                # Reconcile the estimate after preparation. A larger actual
                # cost consumes future deficit; a smaller one is returned.
                runtime.deficit -= _reconciled_charge(item_actual_cost, item_cost, charge) - charge
                if result.outcome is AdmissionOutcome.ADMITTED:
                    admitted += 1
                    if result.partial is not None:
                        partially_admitted += 1
                        emit(
                            "daemon.intake.item_partial",
                            level=WARNING,
                            outcome="degraded",
                            reason=result.partial.reason,
                            component=spec.name,
                            complete_record_count=result.partial.complete_record_count,
                            complete_prefix_bytes=result.partial.complete_prefix_bytes,
                            source_bytes=result.partial.source_bytes,
                        )
                elif result.outcome is AdmissionOutcome.EXCLUDED:
                    excluded += 1
                elif result.outcome is AdmissionOutcome.DEFERRED:
                    deferred += 1
                else:
                    duplicates += 1
                continue
            if result.outcome is AdmissionOutcome.TERMINAL:
                runtime.attempts.pop(item.item_id, None)
                runtime.windows.pop(item.item_id, None)
                runtime.retry_after.pop(item.item_id, None)
                runtime.exhaustions.pop(item.item_id, None)
                runtime.isolated[item.item_id] = _item_revision(item)
                isolated += 1
                emit(
                    "daemon.intake.item_isolated",
                    level=WARNING,
                    outcome="refused",
                    reason="terminal_refusal",
                    component=spec.name,
                    source_id=item.item_id,
                    error_detail=str(result.reason),
                )
                continue
            # A retry still costs its estimate unless the adapter can report
            # a measured cost, including zero for a confirmed unattempted item.
            item_actual_cost = item_cost if result.actual_cost is None else max(0, int(result.actual_cost))
            actual_cost += item_actual_cost
            runtime.deficit -= _reconciled_charge(item_actual_cost, item_cost, charge) - charge
            if result.unattempted:
                retried += 1
                continue
            attempts = runtime.attempts.get(item.item_id, 0) + 1
            runtime.attempts[item.item_id] = attempts
            signature = None if result.transient else (_item_revision(item), _failure_signature(result.reason))
            if item.item_id not in runtime.windows:
                runtime.windows[item.item_id] = signature
            elif runtime.windows[item.item_id] != signature:
                runtime.windows[item.item_id] = None
            if runtime.windows[item.item_id] is None:
                # Any interruption breaks the consecutive streak at once,
                # not only one that lands on the cooldown boundary.
                runtime.exhaustions.pop(item.item_id, None)
            retried += 1
            emit(
                "daemon.intake.item_retryable",
                level=WARNING,
                outcome="degraded",
                reason="admission_failed",
                component=spec.name,
                source_id=item.item_id,
                attempts=attempts,
                error_detail=str(result.reason),
            )
            if attempts >= spec.max_attempts:
                runtime.attempts.pop(item.item_id, None)
                window = runtime.windows.pop(item.item_id, None)
                if window is not None:
                    previous = runtime.exhaustions.get(item.item_id)
                    rounds = previous[1] + 1 if previous is not None and previous[0] == window else 1
                    runtime.exhaustions[item.item_id] = (window, rounds)
                    if rounds >= spec.max_deterministic_cooldowns:
                        # The same defect on the same unchanged item, again
                        # and again: a typed terminal refusal, visible as an
                        # isolated item, instead of a retry every cooldown.
                        runtime.exhaustions.pop(item.item_id, None)
                        runtime.retry_after.pop(item.item_id, None)
                        runtime.isolated[item.item_id] = _item_revision(item)
                        isolated += 1
                        emit(
                            "daemon.intake.item_isolated",
                            level=WARNING,
                            outcome="refused",
                            reason="deterministic_failure",
                            component=spec.name,
                            source_id=item.item_id,
                            attempts=attempts * rounds,
                            error_detail=str(result.reason),
                        )
                        continue
                else:
                    runtime.exhaustions.pop(item.item_id, None)
                runtime.retry_after[item.item_id] = self._clock() + max(0.0, spec.retry_cooldown_s)
                emit(
                    "daemon.intake.item_cooling_down",
                    level=WARNING,
                    outcome="degraded",
                    reason="max_attempts_reached",
                    component=spec.name,
                    source_id=item.item_id,
                    attempts=attempts,
                    error_detail=str(result.reason),
                )

        return IntakeClassReport(
            name=spec.name,
            admitted=admitted,
            planned_count=len(planned),
            discovery_duration_ms=discovery_duration_ms,
            admission_duration_ms=max(0.0, (self._clock() - admission_started) * 1000),
            page_full=len(page) >= limit,
            partial_plan=len(planned) < len(page),
            duplicates=duplicates,
            excluded=excluded,
            deferred=deferred,
            partially_admitted=partially_admitted,
            retried=retried,
            isolated=isolated,
            discovered=len(page),
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            reconciled_cost=actual_cost - estimated_cost,
        )

    async def _admit(self, spec: IntakeClassSpec, item: IntakeItem) -> AdmissionResult:
        try:
            return await spec.adapter.admit(item)
        except Exception as exc:
            raise_if_operation_cancelled(exc)
            return AdmissionResult(
                AdmissionOutcome.RETRYABLE,
                reason=f"{type(exc).__name__}: {exc}",
                transient=is_transient_admission_error(exc),
            )

    async def _admit_page(self, spec: IntakeClassSpec, items: Sequence[IntakeItem]) -> list[AdmissionResult]:
        """Admit a planned page, preferring the adapter's page-shaped entry.

        The returned list is positional over *items*: every planned item gets
        exactly one outcome, so an adapter that batches its writes still
        cannot collapse the scheduler's per-item deficit, retry and isolation
        accounting into one batch-level verdict. An adapter that omits an item
        from its mapping has not reported it, which is retryable -- never a
        silent acknowledgement.
        """
        if not items:
            return []
        admit_page = getattr(spec.adapter, "admit_page", None)
        if admit_page is None:
            return [await self._admit(spec, item) for item in items]
        try:
            results = cast(
                Mapping[str, AdmissionResult],
                await admit_page(tuple(items)),
            )
        except Exception as exc:
            raise_if_operation_cancelled(exc)
            # The adapter handled none of it, so say so once per page: an
            # escaped error otherwise appears only in per-item reasons.
            emit(
                "daemon.intake.page_refused",
                level=WARNING,
                outcome="degraded",
                reason="page_admission_escaped",
                component=spec.name,
                files=len(items),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            reason = f"{type(exc).__name__}: {exc}"
            transient = is_transient_admission_error(exc)
            return [AdmissionResult(AdmissionOutcome.RETRYABLE, reason=reason, transient=transient) for _ in items]
        return [
            results.get(item.item_id)
            or AdmissionResult(AdmissionOutcome.RETRYABLE, reason="adapter reported no outcome for this item")
            for item in items
        ]

    def _is_halted(self, class_name: str) -> bool:
        if self._halts is None:
            return False
        return self._halts.is_halted(unit_id(UnitKind.INTAKE_CLASS, class_name))

    def _halt_class(self, class_name: str, message: str) -> None:
        if self._halts is None:
            emit(
                "daemon.intake.halt_unrecorded",
                level=ERROR,
                outcome="error",
                reason="no_halt_registry",
                component=class_name,
                error_detail=message,
            )
            return
        self._halts.halt(
            unit_id(UnitKind.INTAKE_CLASS, class_name),
            reason=HaltReason.TERMINAL_REFUSAL,
            message=message,
            frame=self._frame,
        )

    def _publish(self, result: IntakePass) -> None:
        if self._board is None:
            return
        for report in result.classes:
            component = f"intake.{report.name}"
            if report.halted or report.discovery_failed:
                self._board.publish(
                    Observation.unmeasured(
                        component,
                        ObservationState.FAILED,
                        reason=report.reason or ("halted" if report.halted else "discovery failed"),
                        frame=self._frame or None,
                    )
                )
                continue
            self._board.publish(
                Observation.measured(
                    component,
                    {
                        "admitted": report.admitted,
                        "duplicates": report.duplicates,
                        "excluded": report.excluded,
                        "deferred": report.deferred,
                        "partially_admitted": report.partially_admitted,
                        "retried": report.retried,
                        "isolated": report.isolated,
                        "discovered": report.discovered,
                        "estimated_cost": report.estimated_cost,
                        "actual_cost": report.actual_cost,
                        "reconciled_cost": report.reconciled_cost,
                    },
                    frame=self._frame or None,
                )
            )

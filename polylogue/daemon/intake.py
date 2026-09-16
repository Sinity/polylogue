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

import inspect
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypeVar, cast, overload, runtime_checkable

from polylogue.daemon.observation import Observation, ObservationBoard, ObservationState
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
from polylogue.logging import ERROR, WARNING, emit

_T = TypeVar("_T")

__all__ = [
    "AdmissionOutcome",
    "AdmissionResult",
    "FairIntakeDispatcher",
    "IntakeAdapter",
    "IntakeClassSpec",
    "IntakeItem",
    "IntakePass",
    "IntakeClassReport",
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
under-reporting: it still runs, but it pays for the passes it occupies.
"""


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

    retry_cooldown_s: float = 5.0
    """Delay before retrying an item that exhausted ``max_attempts``."""


@dataclass(frozen=True, slots=True)
class IntakeClassReport:
    """What one class achieved in one pass."""

    name: str
    admitted: int = 0
    duplicates: int = 0
    excluded: int = 0
    """Items the domain durably refused. Acknowledged, never counted as progress."""

    deferred: int = 0
    """Items the domain deliberately admitted nothing for. Not progress."""

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
    isolated: set[str] = field(default_factory=set)


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
        for spec in schedulable:
            runtime = self._runtime[spec.name]
            share = max(1, budget * spec.weight // total_weight)
            runtime.deficit += share
            reports.append(await self._service_class(spec, runtime))

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
            return IntakeClassReport(name=spec.name)

        # ``page_size`` bounds the discovery call in rows; ``deficit`` is
        # denominated in payload bytes, so it cannot bound a row count. The
        # page plan below is what spends the deficit.
        limit = spec.page_size
        try:
            page: list[IntakeItem] = list(await _maybe_await(spec.adapter.discover(limit=limit)))
        except Exception as exc:
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

        admitted = duplicates = excluded = deferred = retried = isolated = 0
        estimated_cost = actual_cost = 0
        # Plan the page against the class deficit first, then admit the whole
        # plan in one adapter call. The deficit is denominated in payload
        # bytes, so the plan is bytes-aware by construction: it fills up to
        # the class share and stops. Every per-batch fixed cost the domain
        # pays -- the writer hold, the tier bootstrap, one convergence pass --
        # is then paid once per page rather than once per file, while the
        # per-item accounting below is unchanged.
        planned: list[IntakeItem] = []
        for item in page:
            if runtime.deficit <= 0:
                break
            if item.item_id in runtime.isolated:
                continue
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
            # large item behind a cheap synthetic page identity.
            runtime.deficit -= item_cost
            estimated_cost += item_cost
            planned.append(item)

        for item, result in zip(planned, await self._admit_page(spec, planned), strict=True):
            item_cost = max(1, int(item.estimated_cost))
            if result.outcome is AdmissionOutcome.CLASS_TERMINAL:
                self._halt_class(spec.name, result.reason or "class reported terminal failure")
                return IntakeClassReport(
                    name=spec.name,
                    admitted=admitted,
                    duplicates=duplicates,
                    excluded=excluded,
                    deferred=deferred,
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
                await _maybe_await(spec.adapter.acknowledge(item))
                runtime.attempts.pop(item.item_id, None)
                runtime.retry_after.pop(item.item_id, None)
                item_actual_cost = max(1, int(result.actual_cost or item_cost))
                actual_cost += item_actual_cost
                # Reconcile the estimate after preparation. A larger actual
                # cost consumes future deficit; a smaller one is returned.
                runtime.deficit -= item_actual_cost - item_cost
                if result.outcome is AdmissionOutcome.ADMITTED:
                    admitted += 1
                elif result.outcome is AdmissionOutcome.EXCLUDED:
                    excluded += 1
                elif result.outcome is AdmissionOutcome.DEFERRED:
                    deferred += 1
                else:
                    duplicates += 1
                continue
            if result.outcome is AdmissionOutcome.TERMINAL:
                runtime.attempts.pop(item.item_id, None)
                runtime.retry_after.pop(item.item_id, None)
                runtime.isolated.add(item.item_id)
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
            attempts = runtime.attempts.get(item.item_id, 0) + 1
            runtime.attempts[item.item_id] = attempts
            retried += 1
            if attempts >= spec.max_attempts:
                runtime.attempts.pop(item.item_id, None)
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
            duplicates=duplicates,
            excluded=excluded,
            deferred=deferred,
            retried=retried,
            isolated=isolated,
            discovered=len(page),
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            reconciled_cost=actual_cost - estimated_cost,
        )

    async def _admit(self, spec: IntakeClassSpec, item: IntakeItem) -> AdmissionResult:
        try:
            return await _maybe_await(spec.adapter.admit(item))
        except Exception as exc:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"{type(exc).__name__}: {exc}")

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
                await _maybe_await(admit_page(tuple(items))),
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return [AdmissionResult(AdmissionOutcome.RETRYABLE, reason=reason) for _ in items]
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


@overload
async def _maybe_await(value: Awaitable[_T]) -> _T: ...


@overload
async def _maybe_await(value: _T) -> _T: ...


async def _maybe_await(value: object) -> object:
    """Await an adapter result while keeping tiny synchronous test doubles useful."""
    if inspect.isawaitable(value):
        return await cast(Awaitable[object], value)
    return value

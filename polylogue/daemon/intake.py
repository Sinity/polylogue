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

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from polylogue.daemon.observation import Observation, ObservationBoard, ObservationState
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id

logger = logging.getLogger(__name__)

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


class AdmissionOutcome(str, Enum):
    """What happened when an adapter tried to admit an item."""

    ADMITTED = "admitted"
    DUPLICATE = "duplicate"
    """Already admitted under this identity; acknowledging again is safe."""

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

    @property
    def acknowledgeable(self) -> bool:
        """Whether the item's queue entry may be released."""
        return self.outcome in (AdmissionOutcome.ADMITTED, AdmissionOutcome.DUPLICATE)


@runtime_checkable
class IntakeAdapter(Protocol):
    """A domain's view of its own queue.

    ``discover`` must be bounded: it returns at most *limit* items from
    wherever it left off and never costs a full scan.
    """

    def discover(self, *, limit: int) -> Sequence[IntakeItem]: ...

    def admit(self, item: IntakeItem) -> AdmissionResult: ...

    def acknowledge(self, item: IntakeItem) -> None:
        """Release the item's queue entry. Atomic and idempotent."""


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
    """Retryable attempts on one item identity before it is isolated."""


@dataclass(frozen=True, slots=True)
class IntakeClassReport:
    """What one class achieved in one pass."""

    name: str
    admitted: int = 0
    duplicates: int = 0
    retried: int = 0
    isolated: int = 0
    """Items that exhausted their attempts and were set aside."""

    discovered: int = 0
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
        return any(report.admitted or report.duplicates for report in self.classes)

    def report_for(self, name: str) -> IntakeClassReport | None:
        for report in self.classes:
            if report.name == name:
                return report
        return None


@dataclass
class _ClassRuntime:
    """Disposable scheduling hints for one class."""

    deficit: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
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

    def run_once(self, *, budget: int = 64) -> IntakePass:
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
            reports.append(self._service_class(spec, runtime))

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

    def _service_class(self, spec: IntakeClassSpec, runtime: _ClassRuntime) -> IntakeClassReport:
        if runtime.deficit <= 0:
            return IntakeClassReport(name=spec.name)

        limit = min(spec.page_size, runtime.deficit)
        try:
            page = list(spec.adapter.discover(limit=limit))
        except Exception as exc:
            logger.warning("intake: class %s discovery failed: %s", spec.name, exc, exc_info=True)
            return IntakeClassReport(name=spec.name, reason=f"discovery failed: {exc}")

        admitted = duplicates = retried = isolated = 0
        for item in page:
            if runtime.deficit <= 0:
                break
            if item.item_id in runtime.isolated:
                continue
            runtime.deficit -= 1
            result = self._admit(spec, item)
            if result.outcome is AdmissionOutcome.CLASS_TERMINAL:
                self._halt_class(spec.name, result.reason or "class reported terminal failure")
                return IntakeClassReport(
                    name=spec.name,
                    admitted=admitted,
                    duplicates=duplicates,
                    retried=retried,
                    isolated=isolated,
                    discovered=len(page),
                    halted=True,
                    reason=result.reason,
                )
            if result.acknowledgeable:
                spec.adapter.acknowledge(item)
                runtime.attempts.pop(item.item_id, None)
                if result.outcome is AdmissionOutcome.ADMITTED:
                    admitted += 1
                else:
                    duplicates += 1
                continue
            if result.outcome is AdmissionOutcome.TERMINAL:
                runtime.isolated.add(item.item_id)
                isolated += 1
                logger.warning("intake: %s/%s terminal: %s", spec.name, item.item_id, result.reason)
                continue
            attempts = runtime.attempts.get(item.item_id, 0) + 1
            runtime.attempts[item.item_id] = attempts
            retried += 1
            if attempts >= spec.max_attempts:
                runtime.isolated.add(item.item_id)
                isolated += 1
                logger.warning(
                    "intake: %s/%s isolated after %d attempts: %s",
                    spec.name,
                    item.item_id,
                    attempts,
                    result.reason,
                )

        return IntakeClassReport(
            name=spec.name,
            admitted=admitted,
            duplicates=duplicates,
            retried=retried,
            isolated=isolated,
            discovered=len(page),
        )

    def _admit(self, spec: IntakeClassSpec, item: IntakeItem) -> AdmissionResult:
        try:
            return spec.adapter.admit(item)
        except Exception as exc:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"{type(exc).__name__}: {exc}")

    def _is_halted(self, class_name: str) -> bool:
        if self._halts is None:
            return False
        return self._halts.is_halted(unit_id(UnitKind.INTAKE_CLASS, class_name))

    def _halt_class(self, class_name: str, message: str) -> None:
        if self._halts is None:
            logger.error("intake: class %s reported terminal failure with no halt registry: %s", class_name, message)
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
            if report.halted:
                self._board.publish(
                    Observation.unmeasured(
                        component,
                        ObservationState.FAILED,
                        reason=report.reason or "halted",
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
                        "retried": report.retried,
                        "isolated": report.isolated,
                        "discovered": report.discovered,
                    },
                    frame=self._frame or None,
                )
            )

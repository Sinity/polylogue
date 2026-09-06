"""Fair, resumable intake: no class starves, no item wedges its siblings.

The regression this replaces gave one spool class absolute priority: while
any browser-capture file existed, raw materialization was skipped entirely.
Four undrainable files stopped unrelated work for as long as they sat
there. These tests are written so that restoring class-global preemption,
disabling weighted fairness, or losing the isolation of a poison item makes
one of them red.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.daemon.intake import (
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeClassSpec,
    IntakeItem,
)
from polylogue.daemon.observation import ObservationBoard, ObservationState
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id


class FakeAdapter:
    """A spool whose queue authority is its own pending list."""

    def __init__(self, class_name: str, pending: Sequence[str], *, outcome_for=None) -> None:
        self.class_name = class_name
        self.pending = list(pending)
        self.acknowledged: list[str] = []
        self.admitted: list[str] = []
        self.discover_calls: list[int] = []
        self._outcome_for = outcome_for or (lambda _item: AdmissionResult(AdmissionOutcome.ADMITTED))

    def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        self.discover_calls.append(limit)
        return [IntakeItem(item_id=name, class_name=self.class_name) for name in self.pending[:limit]]

    def admit(self, item: IntakeItem) -> AdmissionResult:
        result = self._outcome_for(item)
        if result.outcome is AdmissionOutcome.ADMITTED:
            self.admitted.append(item.item_id)
        return result

    def acknowledge(self, item: IntakeItem) -> None:
        self.acknowledged.append(item.item_id)
        # Atomic and idempotent: acknowledging twice releases one entry.
        if item.item_id in self.pending:
            self.pending.remove(item.item_id)


def test_a_huge_class_cannot_starve_its_siblings() -> None:
    """Mutation: give ``huge`` absolute priority and ``small`` admits nothing."""
    huge = FakeAdapter("huge", [f"h{index}" for index in range(10_000)])
    small = FakeAdapter("small", ["s0", "s1"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="huge", adapter=huge, page_size=8),
            IntakeClassSpec(name="small", adapter=small, page_size=8),
        ]
    )

    result = dispatcher.run_once(budget=16)

    assert result.report_for("small").admitted == 2
    assert result.report_for("huge").admitted > 0
    assert small.pending == []


def test_no_pass_enumerates_a_whole_spool() -> None:
    """Discovery is paged; a per-tick full scan cannot converge at spool scale."""
    huge = FakeAdapter("huge", [f"h{index}" for index in range(830_789)])
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="huge", adapter=huge, page_size=32)])

    dispatcher.run_once(budget=64)

    assert huge.discover_calls == [32]


def test_weight_decides_the_share_of_one_pass() -> None:
    """Mutation: ignore ``weight`` and both classes admit the same count."""
    heavy = FakeAdapter("heavy", [f"a{index}" for index in range(100)])
    light = FakeAdapter("light", [f"b{index}" for index in range(100)])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="heavy", adapter=heavy, weight=3, page_size=64),
            IntakeClassSpec(name="light", adapter=light, weight=1, page_size=64),
        ]
    )

    result = dispatcher.run_once(budget=40)

    assert result.report_for("heavy").admitted > result.report_for("light").admitted


def test_a_poison_item_is_isolated_and_its_siblings_continue() -> None:
    """Mutation: drop the attempt bound and the head item blocks the class."""

    def outcome(item: IntakeItem) -> AdmissionResult:
        if item.item_id == "poison":
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="cannot parse")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("hooks", ["poison", "good0", "good1"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, max_attempts=2, page_size=8)])

    dispatcher.run_once(budget=8)
    dispatcher.run_once(budget=8)

    assert dispatcher.isolated_items("hooks") == frozenset({"poison"})
    assert adapter.admitted == ["good0", "good1", "good0", "good1"] or set(adapter.admitted) == {"good0", "good1"}
    assert adapter.pending == ["poison"]


def test_an_adapter_that_raises_is_one_item_retried_not_a_dead_class() -> None:
    def outcome(item: IntakeItem) -> AdmissionResult:
        if item.item_id == "boom":
            raise RuntimeError("adapter exploded")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("hooks", ["boom", "fine"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, max_attempts=1, page_size=8)])

    result = dispatcher.run_once(budget=8)

    assert result.report_for("hooks").admitted == 1
    assert dispatcher.isolated_items("hooks") == frozenset({"boom"})


def test_a_terminal_item_is_set_aside_without_further_attempts() -> None:
    def outcome(item: IntakeItem) -> AdmissionResult:
        if item.item_id == "unparseable":
            return AdmissionResult(AdmissionOutcome.TERMINAL, reason="unknown envelope version")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("browser", ["unparseable", "ok"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="browser", adapter=adapter, page_size=8)])

    dispatcher.run_once(budget=8)

    assert dispatcher.isolated_items("browser") == frozenset({"unparseable"})
    assert adapter.acknowledged == ["ok"]


def test_duplicate_delivery_is_acknowledged_without_double_admission() -> None:
    """Crash after admission but before acknowledgement replays the item."""
    adapter = FakeAdapter(
        "hooks",
        ["already"],
        outcome_for=lambda _item: AdmissionResult(AdmissionOutcome.DUPLICATE, reason="content hash present"),
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=8)])

    result = dispatcher.run_once(budget=8)

    assert result.report_for("hooks").duplicates == 1
    assert result.report_for("hooks").admitted == 0
    assert adapter.acknowledged == ["already"]
    assert adapter.admitted == []


def test_losing_every_scheduling_hint_preserves_correctness() -> None:
    """Deficits, attempts and the ready set are hints; the spool is authority."""
    adapter = FakeAdapter("hooks", [f"h{index}" for index in range(6)])
    first = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=2)])
    first.run_once(budget=2)

    # A restart constructs a new dispatcher with no hints at all.
    second = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=2)])
    while adapter.pending:
        second.run_once(budget=2)

    assert sorted(adapter.acknowledged) == [f"h{index}" for index in range(6)]
    assert len(adapter.acknowledged) == len(set(adapter.acknowledged))


def test_a_class_in_terminal_refusal_is_excluded_at_selection(tmp_path: Path) -> None:
    """The claude-code wedge: refusing late is the defect.

    Mutation: consult the halt inside ``admit`` instead of
    ``schedulable_classes`` and this reddens -- discovery runs again.
    """
    halts = HaltRegistry(tmp_path)
    dead = FakeAdapter(
        "claude-code",
        ["c0", "c1"],
        outcome_for=lambda _item: AdmissionResult(
            AdmissionOutcome.CLASS_TERMINAL, reason="refusing further ingest until restart"
        ),
    )
    live = FakeAdapter("codex", ["x0", "x1"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="claude-code", adapter=dead, page_size=8),
            IntakeClassSpec(name="codex", adapter=live, page_size=8),
        ],
        halts=halts,
        frame="daemon:1",
    )

    dispatcher.run_once(budget=8)
    discover_calls_at_halt = len(dead.discover_calls)

    for _ in range(5):
        dispatcher.run_once(budget=8)

    assert len(dead.discover_calls) == discover_calls_at_halt, "a halted class was planned again"
    assert live.acknowledged == ["x0", "x1"]
    assert halts.is_halted(unit_id(UnitKind.INTAKE_CLASS, "claude-code"))


def test_a_halted_class_is_named_in_its_status_observation(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.INTAKE_CLASS, "claude-code"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further ingest until restart",
        frame="daemon:1",
    )
    board = ObservationBoard()
    adapter = FakeAdapter("claude-code", ["c0"])
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="claude-code", adapter=adapter, page_size=8)],
        halts=halts,
        board=board,
    )

    result = dispatcher.run_once(budget=8)

    assert result.skipped_halted == ("claude-code",)
    observation = board.get_or_unavailable("intake.claude-code")
    assert observation.state is ObservationState.FAILED
    assert "refusing further ingest until restart" in (observation.reason or "")
    assert observation.value is None


def test_a_halted_class_survives_a_restart(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.INTAKE_CLASS, "claude-code"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further ingest until restart",
        frame="daemon:1",
    )

    adapter = FakeAdapter("claude-code", ["c0"])
    restarted = FairIntakeDispatcher(
        [IntakeClassSpec(name="claude-code", adapter=adapter, page_size=8)],
        halts=HaltRegistry(tmp_path),
    )

    restarted.run_once(budget=8)

    assert adapter.discover_calls == []
    assert adapter.acknowledged == []


def test_a_discovery_failure_reports_rather_than_raising() -> None:
    class BrokenAdapter(FakeAdapter):
        def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            raise OSError("spool directory vanished")

    live = FakeAdapter("codex", ["x0"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="broken", adapter=BrokenAdapter("broken", []), page_size=8),
            IntakeClassSpec(name="codex", adapter=live, page_size=8),
        ]
    )

    result = dispatcher.run_once(budget=8)

    assert "spool directory vanished" in (result.report_for("broken").reason or "")
    assert result.report_for("codex").admitted == 1


def test_duplicate_class_names_are_refused() -> None:
    adapter = FakeAdapter("hooks", [])
    with pytest.raises(ValueError, match="duplicate intake class name"):
        FairIntakeDispatcher(
            [
                IntakeClassSpec(name="hooks", adapter=adapter),
                IntakeClassSpec(name="hooks", adapter=adapter),
            ]
        )

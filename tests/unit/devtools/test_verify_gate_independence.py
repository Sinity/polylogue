"""A red static gate must not hide pytest state.

The nightly corpus once aborted at a flaky browser-extension vitest case and
never ran pytest, so a run recorded as red and a run where pytest was never
measured were indistinguishable for days.

Anti-vacuity: in ``devtools.verify._main``, widen the step loop's short circuit
from ``if args.quick: break`` to ``break`` and
``test_a_red_static_gate_does_not_skip_pytest`` goes red -- pytest never runs
and no pytest verdict reaches the receipt. These cases drive the production
``_main`` loop, not a copy of it.
"""

from __future__ import annotations

from typing import Any

import pytest

from devtools import verify as verify_module
from devtools.testmon_provision import TestmonGraphState, TestmonGraphStatus

STEPS: list[tuple[str, list[str]]] = [
    ("gate js-tests", ["js"]),
    ("gate layering", ["lay"]),
    ("pytest (all)", ["pytest"]),
]


class _Artifacts:
    step_id = "step"


class _Run:
    """A receipt sink: enough surface for ``_main`` without touching the cache."""

    def __init__(self, **_kwargs: Any) -> None:
        self.steps: list[dict[str, Any]] = []

    def record_selection(self, **_kwargs: Any) -> None:
        return None

    def start_step(self, *, label: str, cmd: list[str]) -> _Artifacts:
        del label, cmd
        return _Artifacts()

    def finish_step(self, *, step_id: str, result: Any) -> None:
        del step_id, result
        return None


@pytest.fixture
def driven(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], dict[str, Any]]:
    """Run ``_main`` for real, with only its step execution and IO replaced."""
    executed: list[str] = []
    payload: dict[str, Any] = {}

    def _run(label: str, command: list[str], *, run: Any) -> tuple[int, float, dict[str, Any]]:
        del command, run
        executed.append(label)
        failed = label == "gate js-tests"
        return (1 if failed else 0), 0.0, {"diagnosis": "gate_failed" if failed else "gate_passed"}

    def _finish(**kwargs: Any) -> dict[str, Any]:
        payload.update(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(verify_module, "_run", _run)
    monkeypatch.setattr(verify_module, "build_verify_steps", lambda **_kwargs: list(STEPS))
    monkeypatch.setattr(verify_module, "VerifyRun", _Run)
    monkeypatch.setattr(verify_module, "sync_testmon_graph", lambda _root: False)
    monkeypatch.setattr(
        verify_module,
        "inspect_testmon_graph",
        lambda _root: TestmonGraphState(TestmonGraphStatus.USABLE, "stub"),
    )
    monkeypatch.setattr(verify_module, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(verify_module, "_finish_and_record_verification", _finish)
    monkeypatch.setattr(verify_module, "_emit", lambda *_a, **_k: None)
    return executed, payload


def test_a_red_static_gate_does_not_skip_pytest(driven: tuple[list[str], dict[str, Any]]) -> None:
    executed, payload = driven
    assert verify_module._main(["--all"]) == 1
    assert executed == ["gate js-tests", "gate layering", "pytest (all)"]
    # Every gate's outcome reaches the receipt, not only the first red one.
    receipt = payload["workload_receipt"]
    assert receipt is not None
    assert payload["exit_code"] == 1


def test_pytest_state_is_distinguishable_from_unmeasured(driven: tuple[list[str], dict[str, Any]]) -> None:
    """A red static gate still leaves a pytest verdict; an abort would leave none."""
    executed, payload = driven
    verify_module._main(["--all"])
    assert "pytest (all)" in executed
    aggregate = payload["pytest_aggregate"]
    assert aggregate["selection_mode"] == "all"
    # The run is red overall while the corpus itself was measured, so "pytest
    # red" and "pytest unmeasured" are not the same receipt.
    assert aggregate["terminal_green"] is False
    assert aggregate["complete_corpus_covered"] is False


def test_quick_still_stops_at_the_first_red_gate(driven: tuple[list[str], dict[str, Any]]) -> None:
    """The fast static gate keeps its short circuit; only the test tiers changed."""
    executed, _payload = driven
    assert verify_module._main(["--quick"]) == 1
    assert executed == ["gate js-tests"]

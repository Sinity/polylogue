"""Static gate failures are all visible without hiding pytest state.

The nightly corpus once aborted at a flaky browser-extension vitest case and
never ran pytest, so a run recorded as red and a run where pytest was never
measured were indistinguishable for days.

Anti-vacuity: stop the ``devtools.verify._main`` step loop after the first red
gate and one of these tests loses a gate or a pytest verdict. These cases drive
the production ``_main`` loop, not a copy of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from devtools import verify as verify_module
from devtools.testmon_provision import TestmonGraphState, TestmonGraphStatus
from devtools.verify_runs import VerifyRun

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

    def _run(label: str, command: list[str], *, run: Any, runner: Any = None) -> tuple[int, float, dict[str, Any]]:
        del command, run, runner
        executed.append(label)
        outcomes = {"gate js-tests": (1, "js_failed"), "gate layering": (2, "layering_failed")}
        rc, diagnosis = outcomes.get(label, (0, "gate_passed"))
        return rc, 0.0, {"diagnosis": diagnosis}

    def _finish(**kwargs: Any) -> dict[str, Any]:
        payload.update(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(verify_module, "_run", _run)
    monkeypatch.setattr(
        verify_module,
        "build_verify_steps",
        lambda **kwargs: list(STEPS[:-1] if kwargs["quick"] else STEPS),
    )
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


def test_quick_records_every_static_gate_after_a_failure(driven: tuple[list[str], dict[str, Any]]) -> None:
    executed, payload = driven
    assert verify_module._main(["--quick"]) == 1
    assert executed == ["gate js-tests", "gate layering"]
    assert payload["exit_code"] == 1
    assert payload["diagnosis"] == "js_failed"
    assert payload["pytest_aggregate"]["selection_mode"] == "quick"
    assert payload["pytest_aggregate"]["selected_union_count"] == 0
    assert payload["workload_receipt"]["status"] == "failed"
    assert [phase["name"] for phase in payload["workload_receipt"]["phases"]] == executed


def test_quick_receipt_keeps_all_blocking_gate_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history: dict[str, Any] = {}
    gates = [("gate first", ["first"]), ("gate second", ["second"]), ("gate last", ["last"])]

    def run_gate(label: str, command: list[str], *, run: VerifyRun, runner: str) -> tuple[int, float, dict[str, Any]]:
        del runner
        artifacts = run.start_step(label=label, cmd=command)
        rc = {"gate first": 4, "gate second": 2}.get(label, 0)
        diagnosis = label.replace(" ", "_") + ("_failed" if rc else "_passed")
        run.finish_step(step_id=artifacts.step_id, result={"exit": rc, "diagnosis": diagnosis})
        return rc, 0.0, {"diagnosis": diagnosis}

    monkeypatch.setattr(verify_module, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify_module, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(verify_module, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify_module, "build_verify_steps", lambda **_kwargs: gates)
    monkeypatch.setattr(verify_module, "_run", run_gate)
    monkeypatch.setattr(verify_module, "append_verify_history", lambda payload: history.update(payload))

    assert verify_module._main(["--quick"]) == 4
    receipt = json.loads((tmp_path / history["artifact_dir"] / "run.json").read_text())
    assert receipt["exit_code"] == 4
    assert receipt["diagnosis"] == "gate_first_failed"
    assert [(step["name"], step["exit"]) for step in receipt["steps"]] == [
        ("gate first", 4),
        ("gate second", 2),
        ("gate last", 0),
    ]
    assert receipt["pytest_aggregate"]["selected_union_count"] == 0

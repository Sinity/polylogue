"""The quick gate must be able to see a test module that cannot be collected.

``devtools verify --quick`` is the hosted merge gate and runs no test bodies,
so a module that raises on import used to merge green: the author's focused
selection described a pre-rebase tree, and CI never collected anything.

Anti-vacuity: drop ``test-collection`` from ``devtools.gate.GATES`` (or mark it
``in_quick=False``) and ``test_the_collection_gate_is_in_the_quick_path`` goes
red. Make ``main`` ignore pytest's exit code -- return 0 unconditionally -- and
both failure cases go red. The gate's reach is asserted as narrowly as it is
real: collection, not execution.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from devtools import verify_test_collection
from devtools.agent_env import HARNESS_RUN_ENV
from devtools.gate import GATES_BY_NAME, quick_gates


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _drive(monkeypatch: pytest.MonkeyPatch, completed: _Completed) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _run(command: list[str], **kwargs: Any) -> _Completed:
        calls.append({"command": command, **kwargs})
        return completed

    monkeypatch.setattr(subprocess, "run", _run)
    return calls


def test_the_collection_gate_is_in_the_quick_path() -> None:
    """The merge gate is `verify --quick`; a gate outside it catches nothing."""
    assert "test-collection" in {gate.name for gate in quick_gates()}
    gate = GATES_BY_NAME["test-collection"]
    assert gate.blocking is True


def test_the_command_collects_and_never_executes() -> None:
    command = verify_test_collection.collection_command()
    assert "--collect-only" in command
    # No testmon: a collection that wrote fingerprints would describe a corpus
    # no later run selected from.
    assert "--testmon" not in " ".join(command)
    assert command[1:3] == ["-m", "pytest"]


def test_the_run_names_itself_to_the_bare_pytest_refusal() -> None:
    """Inside an agent job the guard would otherwise make the gate always red."""
    assert verify_test_collection.collection_env()[HARNESS_RUN_ENV] == verify_test_collection.COLLECTION_RUN_ID


def test_an_uncollectable_module_fails_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    _drive(
        monkeypatch,
        _Completed(2, stdout="ERROR tests/unit/test_x.py\n", stderr="ImportError: cannot import name 'Nope'\n"),
    )
    assert verify_test_collection.main([]) == 1


def test_a_corpus_that_collects_nothing_fails_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit 5 is 'collection succeeded, selected nothing' -- never true of the corpus."""
    _drive(monkeypatch, _Completed(5, stdout="no tests ran\n"))
    assert verify_test_collection.main([]) == 1


def test_a_clean_collection_passes_and_reports_its_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _drive(monkeypatch, _Completed(0, stdout="1234 tests collected in 12.00s\n"))
    assert verify_test_collection.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["collected"] == 1234
    assert payload["required_gate"]["gate_passed"] is True


def test_a_zero_count_is_not_a_pass(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A green exit with no collected count means collection never reached the corpus."""
    _drive(monkeypatch, _Completed(0, stdout="\n"))
    assert verify_test_collection.main(["--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False

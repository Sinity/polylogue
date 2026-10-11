"""Readiness follows reservation admission and never precedes execution."""

import os
import signal
import socket
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

from devtools import pytest_slot
from devtools.pytest_memory_admission import EX_TEMPFAIL, RESOURCE_NOT_READY


@pytest.mark.parametrize("abstract", [False, True])
def test_native_readiness_datagram(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, abstract: bool) -> None:
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # Managed scratch paths can exceed sockaddr_un's pathname capacity.
        # A directory-fd address keeps the socket in this fixture's directory.
        target = "@pytest-ready-" + uuid.uuid4().hex if abstract else f"/proc/self/fd/{directory}/notify.socket"
        monkeypatch.setenv("AGENTCTL_EXECUTION_READINESS", "notify")
        monkeypatch.setenv("NOTIFY_SOCKET", target)
        monkeypatch.setattr(time, "monotonic", lambda: 17.5)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
            receiver.bind("\0" + target[1:] if abstract else target)
            receiver.settimeout(1)
            assert pytest_slot._notify_admission_complete() == 17.5
            assert b"READY=1" in receiver.recv(1024).splitlines()
    finally:
        os.close(directory)


def test_notify_mode_requires_a_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTCTL_EXECUTION_READINESS", "notify")
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    with pytest.raises(RuntimeError):
        pytest_slot._notify_admission_complete()


def test_ready_follows_admission_and_reservation_is_released(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = []
    ledger = SimpleNamespace(release=lambda: trace.append("release"))
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda env: ledger)

    def admit(argv: Sequence[str], **kwargs: object) -> tuple[list[str], None]:
        trace.append("admit")
        return list(argv), None

    def ready() -> float:
        trace.append("ready")
        return 100.0

    def execute(command: Sequence[str], **kwargs: object) -> tuple[int, dict[str, str]]:
        trace.append("execute")
        assert kwargs["execution_started"] == 100.0
        return 0, {"status": "success"}

    monkeypatch.setattr(pytest_slot, "admit_width", admit)
    monkeypatch.setattr(pytest_slot, "_notify_admission_complete", ready)
    monkeypatch.setattr(pytest_slot, "_run_held_admitted", execute)
    assert pytest_slot._run_held(["true"], cwd="/", env={}, stdout=None, on_exit=lambda: None)[0] == 0
    assert trace == ["admit", "ready", "execute", "release"]


def test_receipt_separates_admission_and_execution() -> None:
    receipt = pytest_slot._slot_receipt(
        status="success",
        elapsed_s=12.0,
        execution_elapsed_s=2.0,
        sizing={"admission_ledger": {"waited_s": 10.0}},
        memory=None,
    )
    assert receipt["elapsed_s"] == 12.0
    assert receipt["admission_wait_s"] == 10.0
    assert receipt["execution_elapsed_s"] == 2.0
    deferred = pytest_slot._slot_receipt(status="deferred", elapsed_s=10.0, sizing=None, memory=None)
    assert deferred["execution_elapsed_s"] is None
    assert deferred["admission_wait_s"] is None


def test_readiness_failure_releases_admitted_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    released = []
    ledger = SimpleNamespace(release=lambda: released.append(True))
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda env: ledger)
    monkeypatch.setattr(pytest_slot, "admit_width", lambda argv, **kwargs: (list(argv), None))
    monkeypatch.setenv("AGENTCTL_EXECUTION_READINESS", "notify")
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    with pytest.raises(RuntimeError):
        pytest_slot._run_held(["must-not-run"], cwd="/", env={}, stdout=None, on_exit=lambda: None)
    assert released == [True]


def test_deferred_admission_neither_signals_ready_nor_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    released = []
    ledger = SimpleNamespace(release=lambda: released.append(True))
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda env: ledger)
    monkeypatch.setattr(
        pytest_slot,
        "admit_width",
        lambda argv, **kwargs: (list(argv), {"admission": RESOURCE_NOT_READY}),
    )

    def unexpected(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("deferred admission entered execution")

    monkeypatch.setattr(pytest_slot, "_notify_admission_complete", unexpected)
    monkeypatch.setattr(pytest_slot, "_run_held_admitted", unexpected)
    code, receipt = pytest_slot._run_held(["must-not-run"], cwd="/", env={}, stdout=None, on_exit=lambda: None)
    assert code == EX_TEMPFAIL
    assert receipt["status"] == "deferred"
    assert receipt["execution_elapsed_s"] is None
    assert released == [True]


@pytest.mark.parametrize("execution_started,expected", [(None, None), (18.0, 2.0)])
def test_interrupted_receipt_preserves_execution_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execution_started: float | None, expected: float | None
) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 20.0)
    receipt = pytest_slot._write_interrupted_result(
        tmp_path / "run.log",
        environment={},
        started=10.0,
        execution_started=execution_started,
        signal_number=signal.SIGTERM,
        worktree_provenance=None,
    )
    assert receipt["status"] == "interrupted"
    assert receipt["elapsed_s"] == 10.0
    assert receipt["execution_elapsed_s"] == expected

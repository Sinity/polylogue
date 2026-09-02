"""The type daemon must not outlive the gates that start it."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from devtools import gate, verify


def test_starting_the_daemon_bounds_its_idle_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """A started daemon carries an idle timeout, so an unused checkout releases it.

    Anti-vacuity: drop the flag and this is red. Without it every gate leaves a
    multi-gigabyte daemon resident for as long as its checkout exists, and one
    accumulates per checkout until the host runs out of memory.
    """
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(list(argv))
        # Report no running daemon so the start path is taken.
        return SimpleNamespace(returncode=0 if "start" in argv else 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    dmypy = str(verify.ROOT / ".venv/bin/dmypy")
    assert gate.mypy_command() == [
        dmypy,
        "run",
        f"--timeout={gate.DMYPY_IDLE_TIMEOUT_SECONDS}",
        "--",
        "--no-error-summary",
    ]

    start = next(argv for argv in calls if "start" in argv)
    assert f"--timeout={gate.DMYPY_IDLE_TIMEOUT_SECONDS}" in start
    assert gate.DMYPY_IDLE_TIMEOUT_SECONDS > 0

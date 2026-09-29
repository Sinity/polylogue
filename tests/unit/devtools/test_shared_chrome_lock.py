from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from devtools.shared_chrome_lock import SharedChromeLockTimeoutError, shared_chrome_extension_lock


def test_anti_vacuity_shared_extension_workflows_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    active = 0
    maximum = 0
    gate = threading.Lock()

    def operation() -> None:
        nonlocal active, maximum
        with shared_chrome_extension_lock(timeout_s=5):
            with gate:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with gate:
                active -= 1

    threads = [threading.Thread(target=operation) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum == 1


def _contend(timeout_s: float) -> list[BaseException]:
    errors: list[BaseException] = []

    def contender() -> None:
        try:
            with shared_chrome_extension_lock(timeout_s=timeout_s):
                pass
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=contender)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "contender hung"
    return errors


def test_anti_vacuity_lock_acquisition_is_bounded_past_a_wedged_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocking flock would wait forever here instead of raising once the holder's budget passed."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with shared_chrome_extension_lock(timeout_s=0.2):
        errors = _contend(0.05)
    assert len(errors) == 1
    assert isinstance(errors[0], SharedChromeLockTimeoutError)


def test_waiter_outlasts_its_own_budget_while_the_holder_is_within_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: a waiter bounded only by its own 0.05s budget fails behind a valid 0.3s hold."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    def holder() -> None:
        with shared_chrome_extension_lock(timeout_s=5):
            time.sleep(0.3)

    thread = threading.Thread(target=holder)
    thread.start()
    time.sleep(0.05)
    errors = _contend(0.05)
    thread.join()
    assert errors == []

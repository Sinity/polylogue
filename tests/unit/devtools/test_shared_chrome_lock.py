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


def test_anti_vacuity_lock_acquisition_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocking flock would wait forever here instead of raising at the deadline."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with shared_chrome_extension_lock(timeout_s=5):
        waiter_error: list[BaseException] = []

        def contender() -> None:
            try:
                with shared_chrome_extension_lock(timeout_s=0.2):
                    pass
            except BaseException as exc:
                waiter_error.append(exc)

        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(waiter_error) == 1
    assert isinstance(waiter_error[0], SharedChromeLockTimeoutError)

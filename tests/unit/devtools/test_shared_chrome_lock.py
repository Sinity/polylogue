from __future__ import annotations

import threading
import time

from devtools.shared_chrome_lock import shared_chrome_extension_lock


def test_anti_vacuity_shared_extension_workflows_are_serialized(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    active = 0
    maximum = 0
    gate = threading.Lock()

    def operation() -> None:
        nonlocal active, maximum
        with shared_chrome_extension_lock():
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

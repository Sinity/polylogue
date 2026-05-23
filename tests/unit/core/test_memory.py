from __future__ import annotations

import pytest

from polylogue.core import memory


def test_release_process_memory_collects_and_trims_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class FakeMallocTrim:
        argtypes: object = None
        restype: object = None

        def __call__(self, value: int) -> int:
            calls.append(value)
            return 1

    class FakeLibc:
        malloc_trim = FakeMallocTrim()

    monkeypatch.setattr("polylogue.core.memory.gc.collect", lambda: 7)
    monkeypatch.setattr("polylogue.core.memory.sys.platform", "linux")
    monkeypatch.setattr("polylogue.core.memory.ctypes.CDLL", lambda _name: FakeLibc())

    result = memory.release_process_memory()

    assert result.collected_objects == 7
    assert result.malloc_trim_called is True
    assert calls == [0]

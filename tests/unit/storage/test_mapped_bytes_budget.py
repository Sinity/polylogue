"""polylogue-e98k: reconcile the SQLite mmap/cache budget with the cgroup memory limit.

Production dependencies exercised here: ``mapped_bytes_budget`` (the real
arithmetic over the real ``BULK_BUILD_*``/``DAEMON_WRITE_*``/``READ_*``
constants -- not a reimplemented copy), ``check_mapped_bytes_budget_against_cgroup_limit``
(the real cgroup-reading composition), and ``log_mapped_bytes_budget_check``
(the real branch selecting debug/info/warning). Reverting the budget formula
to the single largest constant (``BULK_BUILD_MMAP_SIZE_BYTES`` alone) would
make ``test_mapped_bytes_budget_reflects_documented_worst_case`` fail, not
merely report a smaller number; reverting ``at_risk_limits`` to always return
empty would make ``test_at_risk_when_limit_at_or_below_budget`` fail to warn.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from polylogue.storage.sqlite.connection_profile import (
    BULK_BUILD_CACHE_SIZE_KIB,
    BULK_BUILD_MMAP_SIZE_BYTES,
    DAEMON_WRITE_CACHE_SIZE_KIB,
    DAEMON_WRITE_MMAP_SIZE_BYTES,
    DEFAULT_MEMORY_BUDGET_BYTES,
    MEMORY_BUDGET_BYTES,
    READ_CACHE_SIZE_KIB,
    READ_MMAP_SIZE_BYTES,
    MappedBytesBudgetCheck,
    check_mapped_bytes_budget_against_cgroup_limit,
    log_mapped_bytes_budget_check,
    mapped_bytes_budget,
)


def _import_profile_values(*, budget_bytes: int | None) -> dict[str, int]:
    env = os.environ.copy()
    if budget_bytes is None:
        env.pop("POLYLOGUE_MEMORY_BUDGET_BYTES", None)
    else:
        env["POLYLOGUE_MEMORY_BUDGET_BYTES"] = str(budget_bytes)
    code = """
from polylogue.storage.sqlite.connection_profile import (
    BULK_BUILD_CACHE_SIZE_KIB,
    BULK_BUILD_MMAP_SIZE_BYTES,
    DAEMON_WRITE_CACHE_SIZE_KIB,
    DAEMON_WRITE_MMAP_SIZE_BYTES,
    MEMORY_BUDGET_BYTES,
    READ_CACHE_SIZE_KIB,
    READ_MMAP_SIZE_BYTES,
    WRITE_CACHE_SIZE_KIB,
    WRITE_MMAP_SIZE_BYTES,
)
print(" ".join(str(value) for value in (
    MEMORY_BUDGET_BYTES,
    WRITE_CACHE_SIZE_KIB,
    DAEMON_WRITE_CACHE_SIZE_KIB,
    READ_CACHE_SIZE_KIB,
    BULK_BUILD_CACHE_SIZE_KIB,
    WRITE_MMAP_SIZE_BYTES,
    DAEMON_WRITE_MMAP_SIZE_BYTES,
    READ_MMAP_SIZE_BYTES,
    BULK_BUILD_MMAP_SIZE_BYTES,
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    values = [int(value) for value in completed.stdout.split()]
    keys = (
        "memory_budget_bytes",
        "write_cache_size_kib",
        "daemon_write_cache_size_kib",
        "read_cache_size_kib",
        "bulk_build_cache_size_kib",
        "write_mmap_size_bytes",
        "daemon_write_mmap_size_bytes",
        "read_mmap_size_bytes",
        "bulk_build_mmap_size_bytes",
    )
    return dict(zip(keys, values, strict=True))


def test_declared_budget_preserves_defaults_when_absent() -> None:
    values = _import_profile_values(budget_bytes=None)
    assert values == {
        "memory_budget_bytes": DEFAULT_MEMORY_BUDGET_BYTES,
        "write_cache_size_kib": 131072,
        "daemon_write_cache_size_kib": 16384,
        "read_cache_size_kib": 32768,
        "bulk_build_cache_size_kib": 524288,
        "write_mmap_size_bytes": 1073741824,
        "daemon_write_mmap_size_bytes": 67108864,
        "read_mmap_size_bytes": 134217728,
        "bulk_build_mmap_size_bytes": 4294967296,
    }


def test_declared_budget_scales_all_profile_mmap_and_cache_sizes() -> None:
    values = _import_profile_values(budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES // 2)
    assert values == {
        "memory_budget_bytes": DEFAULT_MEMORY_BUDGET_BYTES // 2,
        "write_cache_size_kib": 65536,
        "daemon_write_cache_size_kib": 8192,
        "read_cache_size_kib": 16384,
        "bulk_build_cache_size_kib": 262144,
        "write_mmap_size_bytes": 536870912,
        "daemon_write_mmap_size_bytes": 33554432,
        "read_mmap_size_bytes": 67108864,
        "bulk_build_mmap_size_bytes": 2147483648,
    }


def test_scaled_budget_drives_startup_warning_threshold() -> None:
    env = os.environ.copy()
    env["POLYLOGUE_MEMORY_BUDGET_BYTES"] = str(DEFAULT_MEMORY_BUDGET_BYTES // 2)
    code = """
import polylogue.core.metrics as metrics
from polylogue.storage.sqlite.connection_profile import (
    MEMORY_BUDGET_BYTES,
    check_mapped_bytes_budget_against_cgroup_limit,
    mapped_bytes_budget,
)
budget = mapped_bytes_budget()
metrics.read_cgroup_memory_max_bytes = lambda: budget
metrics.read_cgroup_memory_high_bytes = lambda: budget
check = check_mapped_bytes_budget_against_cgroup_limit()
print(check.memory_budget_bytes, check.budget_bytes, check.at_risk_limits)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    budget_value, mapped_value, thresholds = completed.stdout.strip().split(maxsplit=2)
    assert int(budget_value) == DEFAULT_MEMORY_BUDGET_BYTES // 2
    assert int(mapped_value) > 0
    assert thresholds == "('memory.max', 'memory.high')"


@pytest.mark.parametrize("raw_budget", ["", "0", "-1", "not-a-number"])
def test_invalid_declared_budget_fails_loudly(raw_budget: str) -> None:
    env = os.environ.copy()
    env["POLYLOGUE_MEMORY_BUDGET_BYTES"] = raw_budget
    code = "from polylogue.storage.sqlite.connection_profile import MEMORY_BUDGET_BYTES; print(MEMORY_BUDGET_BYTES)"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode != 0
    assert "POLYLOGUE_MEMORY_BUDGET_BYTES" in completed.stderr


def test_mapped_bytes_budget_reflects_documented_worst_case() -> None:
    """Budget = bulk-build + daemon-write + N concurrent read connections.

    Not just the single largest constant (``BULK_BUILD_MMAP_SIZE_BYTES``) --
    that was exactly the gap the 2026-07-31 incident exposed: one bulk-build
    connection alone was assumed representative of the worst case, when the
    daemon's own write connection and concurrent reads add on top of it.
    """
    expected = (
        BULK_BUILD_MMAP_SIZE_BYTES
        + BULK_BUILD_CACHE_SIZE_KIB * 1024
        + DAEMON_WRITE_MMAP_SIZE_BYTES
        + DAEMON_WRITE_CACHE_SIZE_KIB * 1024
        + 4 * (READ_MMAP_SIZE_BYTES + READ_CACHE_SIZE_KIB * 1024)
    )
    assert mapped_bytes_budget() == expected
    # Strictly greater than the single largest constant alone -- proves this
    # isn't a no-op reformulation of the old single-constant framing.
    assert mapped_bytes_budget() > BULK_BUILD_MMAP_SIZE_BYTES

    # concurrent_read_connections is a real parameter, not vestigial.
    assert mapped_bytes_budget(concurrent_read_connections=0) < mapped_bytes_budget(concurrent_read_connections=4)


def test_check_reads_cgroup_limits_via_core_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.core.metrics.read_cgroup_memory_max_bytes",
        lambda: 20 * 1024 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "polylogue.core.metrics.read_cgroup_memory_high_bytes",
        lambda: 16 * 1024 * 1024 * 1024,
    )
    check = check_mapped_bytes_budget_against_cgroup_limit()
    assert check.memory_max_bytes == 20 * 1024 * 1024 * 1024
    assert check.memory_high_bytes == 16 * 1024 * 1024 * 1024
    assert check.budget_bytes == mapped_bytes_budget()
    assert check.memory_budget_bytes == MEMORY_BUDGET_BYTES


def test_at_risk_when_limit_at_or_below_budget() -> None:
    budget = mapped_bytes_budget()
    check = MappedBytesBudgetCheck(
        budget_bytes=budget,
        memory_max_bytes=budget + 1,
        memory_high_bytes=budget,  # exactly at the budget: still at-risk
        concurrent_read_connections=4,
    )
    assert check.at_risk_limits == ("memory.high",)

    check_both_ok = MappedBytesBudgetCheck(
        budget_bytes=budget,
        memory_max_bytes=budget * 2,
        memory_high_bytes=budget * 2,
        concurrent_read_connections=4,
    )
    assert check_both_ok.at_risk_limits == ()

    check_both_risky = MappedBytesBudgetCheck(
        budget_bytes=budget,
        memory_max_bytes=budget - 1,
        memory_high_bytes=budget - 1,
        concurrent_read_connections=4,
    )
    assert set(check_both_risky.at_risk_limits) == {"memory.max", "memory.high"}


def test_no_cgroup_limit_detected_degrades_to_debug_log() -> None:
    """No cgroup files present (dev machine, no controller) must never raise --
    it's the ordinary case, not an error."""
    check = MappedBytesBudgetCheck(
        budget_bytes=mapped_bytes_budget(),
        memory_max_bytes=None,
        memory_high_bytes=None,
        concurrent_read_connections=4,
    )

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def debug(self, message: str, **kw: object) -> None:
            self.calls.append(("debug", message, kw))

        def info(self, message: str, **kw: object) -> None:
            self.calls.append(("info", message, kw))

        def warning(self, message: str, **kw: object) -> None:
            self.calls.append(("warning", message, kw))

    recorder = _Recorder()
    log_mapped_bytes_budget_check(recorder, check)  # type: ignore[arg-type]

    assert len(recorder.calls) == 1
    level, message, _kw = recorder.calls[0]
    assert level == "debug"
    assert message == "mmap_budget_no_cgroup_limit_detected"


def test_warns_when_at_risk() -> None:
    budget = mapped_bytes_budget()
    check = MappedBytesBudgetCheck(
        budget_bytes=budget,
        memory_max_bytes=budget * 2,
        memory_high_bytes=budget - 1,
        concurrent_read_connections=4,
    )

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def debug(self, message: str, **kw: object) -> None:
            self.calls.append(("debug", message, kw))

        def info(self, message: str, **kw: object) -> None:
            self.calls.append(("info", message, kw))

        def warning(self, message: str, **kw: object) -> None:
            self.calls.append(("warning", message, kw))

    recorder = _Recorder()
    log_mapped_bytes_budget_check(recorder, check)  # type: ignore[arg-type]

    assert len(recorder.calls) == 1
    level, message, kw = recorder.calls[0]
    assert level == "warning"
    assert message == "mmap_budget_at_or_above_cgroup_limit"
    assert kw["at_risk_limits"] == ["memory.high"]
    assert kw["memory_budget_bytes"] == MEMORY_BUDGET_BYTES


def test_warning_threshold_uses_effective_budget() -> None:
    budget = mapped_bytes_budget()
    check = MappedBytesBudgetCheck(
        budget_bytes=budget,
        memory_max_bytes=budget,
        memory_high_bytes=budget,
        concurrent_read_connections=4,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
    )

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def debug(self, message: str, **kw: object) -> None:
            self.calls.append(("debug", message, kw))

        def info(self, message: str, **kw: object) -> None:
            self.calls.append(("info", message, kw))

        def warning(self, message: str, **kw: object) -> None:
            self.calls.append(("warning", message, kw))

    recorder = _Recorder()
    log_mapped_bytes_budget_check(recorder, check)  # type: ignore[arg-type]

    assert recorder.calls[0][0] == "warning"
    assert recorder.calls[0][2]["memory_budget_bytes"] == MEMORY_BUDGET_BYTES
    assert recorder.calls[0][2]["budget_mb"] == round(budget / (1024 * 1024), 1)


def test_logs_info_when_within_limits() -> None:
    budget = mapped_bytes_budget()
    check = MappedBytesBudgetCheck(
        budget_bytes=budget,
        memory_max_bytes=budget * 3,
        memory_high_bytes=budget * 2,
        concurrent_read_connections=4,
    )

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def debug(self, message: str, **kw: object) -> None:
            self.calls.append(("debug", message, kw))

        def info(self, message: str, **kw: object) -> None:
            self.calls.append(("info", message, kw))

        def warning(self, message: str, **kw: object) -> None:
            self.calls.append(("warning", message, kw))

    recorder = _Recorder()
    log_mapped_bytes_budget_check(recorder, check)  # type: ignore[arg-type]

    assert len(recorder.calls) == 1
    level, message, _kw = recorder.calls[0]
    assert level == "info"
    assert message == "mmap_budget_within_cgroup_limit"

"""The corpus runs at a width the host can actually hold.

Two full corpus runs were killed by the out-of-memory daemon at about 6.2 GB
peak while the operator's desktop and other lanes were resident. A killed run
measures nothing; a narrower run that finishes measures everything.

Anti-vacuity: make ``memory_bounded_worker_cap`` return ``CORPUS_MAX_WORKERS``
unconditionally and ``test_a_loaded_host_runs_narrower`` goes red -- the width
returns to the fixed one that was killed at the memory this case describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.verify import (
    CONTROLLER_PEAK_MIB,
    CORPUS_MAX_WORKERS,
    WORKER_PEAK_MIB,
    available_memory_mib,
    memory_bounded_worker_cap,
)


def _meminfo(tmp_path: Path, available_mib: int) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       32689696 kB\nMemFree:         1000000 kB\nMemAvailable:   {available_mib * 1024} kB\n",
        encoding="utf-8",
    )
    return path


def test_an_idle_host_runs_the_full_width(tmp_path: Path) -> None:
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 28000))
    assert workers == CORPUS_MAX_WORKERS
    assert basis["narrowed"] is False


def test_a_loaded_host_runs_narrower(tmp_path: Path) -> None:
    """The measured condition of the killed runs: about 4.7 GiB available."""
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 4707))
    assert workers < CORPUS_MAX_WORKERS
    # The chosen width fits in the headroom-adjusted budget; the fixed one does not.
    assert workers * WORKER_PEAK_MIB + CONTROLLER_PEAK_MIB <= 4707 * 0.8
    assert CORPUS_MAX_WORKERS * WORKER_PEAK_MIB + CONTROLLER_PEAK_MIB > 4707 * 0.8
    assert basis["narrowed"] is True
    assert basis["available_mib"] == 4707


def test_a_starved_host_still_runs_one_worker(tmp_path: Path) -> None:
    """A slow run beats a killed run; it never resolves to zero workers."""
    workers, _basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 200))
    assert workers == 1


def test_an_unreadable_meminfo_does_not_narrow_silently(tmp_path: Path) -> None:
    workers, basis = memory_bounded_worker_cap(meminfo=tmp_path / "absent")
    assert workers == CORPUS_MAX_WORKERS
    assert basis["basis"] == "unmeasured"


@pytest.mark.parametrize("content", ["", "MemTotal: 100 kB\n", "MemAvailable: not-a-number kB\n"])
def test_a_malformed_meminfo_is_unmeasured(tmp_path: Path, content: str) -> None:
    path = tmp_path / "meminfo"
    path.write_text(content, encoding="utf-8")
    assert available_memory_mib(meminfo=path) is None


def test_the_basis_is_recorded_for_the_receipt(tmp_path: Path) -> None:
    """The width is evidence, not a silent policy."""
    _workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 8000))
    assert basis.keys() >= {"available_mib", "worker_peak_mib", "controller_peak_mib", "workers", "basis"}

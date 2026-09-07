"""A managed pytest run records what it took, per process and in aggregate.

A run that systemd-oomd kills reports only that it died: the width it ran at,
the peak it reached and which process class reached it are exactly what the
next width decision needs and exactly what a killed run leaves behind. The
sampler reads the process group the run owns, so the attribution covers the
xdist workers as well as their controller.

Anti-vacuity:
- sum the last sample instead of keeping the maximum and
  ``test_the_peak_survives_a_later_quieter_sample`` goes red -- a peak read
  after the run settles is the plateau, not what got it killed;
- attribute by pid without filtering the process group and
  ``test_only_the_run_s_own_process_group_is_attributed`` goes red -- the
  host's other work would be charged to the run;
- drop a process from the record once it exits and
  ``test_a_process_that_exits_keeps_its_attribution`` goes red, which is the
  case that matters: a killed worker is gone by the time anyone reads the
  receipt;
- report a peak of zero when nothing was readable and
  ``test_an_unreadable_procfs_is_unmeasured_not_zero`` goes red -- an absent
  measurement would otherwise pass for a run that took nothing;
- keep every process seen and ``test_the_attribution_is_bounded`` goes red: a
  corpus run forks thousands of children and the receipt is durable.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from devtools.pytest_memory import MAX_ATTRIBUTED_PROCESSES, ProcessGroupMemorySampler

KIB = 1024


def _proc(tmp_path: Path, name: str = "proc") -> Path:
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    return directory


def _process(proc: Path, pid: int, *, pgid: int, command: str = "pytest", pss_kib: int = 0, rss_kib: int = 0) -> Path:
    """One process in a stub procfs, as the sampler reads it."""
    directory = proc / str(pid)
    directory.mkdir(exist_ok=True)
    # The comm field is parenthesised and may contain spaces; the fields the
    # sampler reads come after it.
    (directory / "stat").write_text(f"{pid} ({command} worker) S 1 {pgid} 0 0 -1 0\n", encoding="utf-8")
    (directory / "comm").write_text(f"{command}\n", encoding="utf-8")
    (directory / "smaps_rollup").write_text(
        f"00400000-7fff ---p 00000000 00:00 0 [rollup]\n"
        f"Rss:            {rss_kib or pss_kib} kB\n"
        f"Pss:            {pss_kib} kB\n"
        f"Private_Clean:  {pss_kib // 4} kB\n"
        f"Private_Dirty:  {pss_kib // 4} kB\n"
        f"Swap:           0 kB\n",
        encoding="utf-8",
    )
    return directory


def _meminfo(tmp_path: Path, available_mib: int, name: str = "meminfo") -> Path:
    path = tmp_path / name
    path.write_text(f"MemTotal: 32689696 kB\nMemAvailable: {available_mib * 1024} kB\n", encoding="utf-8")
    return path


def _sampler(tmp_path: Path, proc: Path, *, pgid: int = 100, available_mib: int = 8000) -> ProcessGroupMemorySampler:
    return ProcessGroupMemorySampler(pgid, proc=proc, meminfo=_meminfo(tmp_path, available_mib))


def test_the_group_peak_is_the_sum_of_its_processes(tmp_path: Path) -> None:
    """The controller and its workers are one budget, and named individually."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, command="pytest", pss_kib=1075 * KIB)
    for pid in (101, 102, 103):
        _process(proc, pid, pgid=100, command="pytest-xdist", pss_kib=600 * KIB)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    document = sampler.snapshot()

    assert document["peak"]["pss_kib"] == (1075 + 3 * 600) * KIB
    assert document["peak"]["processes"] == 4
    assert {entry["pid"] for entry in document["processes"]} == {100, 101, 102, 103}
    # Worst first: the controller costs more than any single worker.
    assert document["processes"][0]["pid"] == 100
    assert document["processes"][0]["peak_pss_kib"] == 1075 * KIB
    assert document["processes"][0]["command"] == "pytest"


def test_only_the_run_s_own_process_group_is_attributed(tmp_path: Path) -> None:
    """The workstation runs a desktop; none of it belongs to this run's budget."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=500 * KIB)
    _process(proc, 555, pgid=555, command="chrome", pss_kib=4000 * KIB)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    document = sampler.snapshot()

    assert document["peak"]["pss_kib"] == 500 * KIB
    assert [entry["pid"] for entry in document["processes"]] == [100]


def test_the_peak_survives_a_later_quieter_sample(tmp_path: Path) -> None:
    """What matters is the worst moment, not the state the run finished in."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=3000 * KIB)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    _process(proc, 100, pgid=100, pss_kib=200 * KIB)
    sampler.sample()
    document = sampler.snapshot()

    assert document["peak"]["pss_kib"] == 3000 * KIB
    assert document["processes"][0]["peak_pss_kib"] == 3000 * KIB
    assert document["samples"] == 2


def test_a_process_that_exits_keeps_its_attribution(tmp_path: Path) -> None:
    """A worker that was killed is the one the receipt most needs to name."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=700 * KIB)
    worker = _process(proc, 101, pgid=100, command="pytest-xdist", pss_kib=2500 * KIB)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    for path in sorted(worker.iterdir()):
        path.unlink()
    worker.rmdir()
    sampler.sample()
    document = sampler.snapshot()

    named = {entry["pid"]: entry for entry in document["processes"]}
    assert named[101]["peak_pss_kib"] == 2500 * KIB
    assert document["peak"]["pss_kib"] == (700 + 2500) * KIB


def test_the_lowest_host_memory_seen_is_recorded(tmp_path: Path) -> None:
    """The width was chosen against MemAvailable; the run says what it became."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    meminfo = _meminfo(tmp_path, 6000)
    sampler = ProcessGroupMemorySampler(100, proc=proc, meminfo=meminfo)
    sampler.sample()
    meminfo.write_text("MemAvailable: 1228800 kB\n", encoding="utf-8")
    sampler.sample()
    document = sampler.snapshot()

    assert document["host_mem_available_mib"] == {"at_start": 6000, "minimum": 1200}


def test_an_unreadable_procfs_is_unmeasured_not_zero(tmp_path: Path) -> None:
    """No reading is not a reading of nothing."""
    sampler = _sampler(tmp_path, _proc(tmp_path))
    sampler.sample()
    document = sampler.snapshot()

    assert document["unmeasured"] == "no sample observed the process group"
    assert document["observed_samples"] == 0
    assert document["peak"]["pss_kib"] == 0


def test_the_attribution_is_bounded(tmp_path: Path) -> None:
    """The worst processes are named; the rest are counted."""
    proc = _proc(tmp_path)
    total = MAX_ATTRIBUTED_PROCESSES + 10
    for index in range(total):
        _process(proc, 100 + index, pgid=100, pss_kib=(index + 1) * KIB)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    document = sampler.snapshot()

    assert len(document["processes"]) == MAX_ATTRIBUTED_PROCESSES
    assert document["processes_seen"] == total
    assert document["processes"][0]["peak_pss_kib"] == total * KIB
    assert document["peak"]["processes"] == total


@pytest.mark.uses_real_clock("waits on a real sampling thread")
def test_sampling_in_its_own_thread_measures_a_live_group(tmp_path: Path) -> None:
    """The production path: start, let the thread sample, stop and read."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=900 * KIB)
    sampler = ProcessGroupMemorySampler(100, interval_s=0.01, proc=proc, meminfo=_meminfo(tmp_path, 4000))
    sampler.start()
    for _attempt in range(200):
        if sampler.snapshot()["observed_samples"]:
            break
        time.sleep(0.01)
    document = sampler.stop()

    assert document["observed_samples"] >= 1
    assert document["peak"]["pss_kib"] == 900 * KIB

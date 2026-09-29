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

import json
import time
from pathlib import Path

import pytest

from devtools.pytest_memory import MAX_ATTRIBUTED_PROCESSES, ProcessGroupMemorySampler

KIB = 1024


def _proc(tmp_path: Path, name: str = "proc") -> Path:
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    return directory


def _process(
    proc: Path,
    pid: int,
    *,
    pgid: int,
    command: str = "pytest",
    pss_kib: int = 0,
    rss_kib: int = 0,
    ppid: int = 1,
    start_time: int = 1,
) -> Path:
    """One process in a stub procfs, as the sampler reads it."""
    directory = proc / str(pid)
    directory.mkdir(exist_ok=True)
    # The comm field is parenthesised and may contain spaces; the fields the
    # sampler reads come after it.
    fields = ["S", str(ppid), str(pgid), *(["0"] * 16), str(start_time)]
    (directory / "stat").write_text(f"{pid} ({command} worker) {' '.join(fields)}\n", encoding="utf-8")
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


@pytest.mark.parametrize("cgroup_record", ["0::/shared/pytest\n", "2:memory:/shared/pytest\n"])
def test_ambient_cgroup_does_not_authorize_unrelated_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cgroup_record: str
) -> None:
    """The ambient cgroup includes another run; only owned descendants belong here."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    _process(proc, 101, pgid=900, ppid=100, command="detached", pss_kib=700 * KIB)
    _process(proc, 555, pgid=555, command="unrelated", pss_kib=4000 * KIB)
    (proc / "self").mkdir()
    (proc / "self" / "cgroup").write_text(cgroup_record, encoding="utf-8")
    read_text = Path.read_text

    def read_fixture_members(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path in {
            Path("/sys/fs/cgroup/shared/pytest/cgroup.procs"),
            Path("/sys/fs/cgroup/memory/shared/pytest/cgroup.procs"),
        }:
            return "100\n101\n555\n"
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_fixture_members)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    result = sampler.snapshot()
    assert result["peak"]["pss_kib"] == 800 * KIB
    assert {row["pid"] for row in result["processes"]} == {100, 101}


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


def test_each_memory_metric_keeps_its_own_peak(tmp_path: Path) -> None:
    """RSS can spike at a different moment than PSS and is still evidence."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=3000 * KIB, rss_kib=3200 * KIB)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    _process(proc, 100, pgid=100, pss_kib=2000 * KIB, rss_kib=5000 * KIB)
    sampler.sample()
    document = sampler.snapshot()

    assert document["peak"]["pss_kib"] == 3000 * KIB
    assert document["peak"]["rss_kib"] == 5000 * KIB


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


def test_a_periodic_snapshot_persists_memory_and_context(tmp_path: Path) -> None:
    """A kill can leave the last complete sidecar even without a final receipt."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=900 * KIB)
    sidecar = tmp_path / "pytest.telemetry.json"
    sampler = ProcessGroupMemorySampler(
        100,
        proc=proc,
        meminfo=_meminfo(tmp_path, 4000),
        snapshot_path=sidecar,
        snapshot_context=lambda: {
            "sizing": {"workers": 3, "requested_workers": 8},
            "progress": {"selected_count": 23449, "terminal_count": 1200},
        },
    )

    sampler.sample()

    telemetry = json.loads(sidecar.read_text(encoding="utf-8"))
    assert telemetry["kind"] == "polylogue.pytest-slot-telemetry"
    assert telemetry["sizing"]["workers"] == 3
    assert telemetry["progress"]["terminal_count"] == 1200
    assert telemetry["memory"]["peak"]["pss_kib"] == 900 * KIB
    assert telemetry["memory"]["processes"][0]["pid"] == 100


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


def test_a_followed_rerun_group_is_part_of_the_run_peak(tmp_path: Path) -> None:
    """A slot job's in-slot rerun runs in a new session; its memory still counts.

    Anti-vacuity: make ``follow`` a no-op and the second sample reads the
    finished first group again, so the rerun's larger peak never appears.
    """
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=200 * KIB)
    sampler = _sampler(tmp_path, proc, pgid=100)
    sampler.sample()

    _process(proc, 300, pgid=300, pss_kib=900 * KIB)
    # No explicit sample: ``follow`` observes the new group at once, so a
    # rerun that ends within one interval is still measured.
    sampler.follow(300)

    document = sampler.stop()
    assert document["peak"]["pss_kib"] == 900 * KIB


def test_detached_descendant_remains_owned_until_its_pid_is_reused(tmp_path: Path) -> None:
    """Start-time identity retains reparented children without charging a reused PID."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    _process(proc, 101, pgid=900, ppid=100, start_time=10, pss_kib=200 * KIB)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    _process(proc, 101, pgid=900, ppid=1, start_time=10, pss_kib=700 * KIB)
    sampler.sample()
    assert sampler.snapshot()["peak"]["pss_kib"] == 800 * KIB

    _process(proc, 101, pgid=900, ppid=1, start_time=20, command="unrelated", pss_kib=4000 * KIB)
    sampler.sample()
    result = sampler.snapshot()
    assert result["peak"]["pss_kib"] == 800 * KIB
    assert next(row for row in result["processes"] if row["pid"] == 101)["peak_pss_kib"] == 700 * KIB


def test_pid_reused_between_discovery_and_memory_read_is_not_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery-time ownership must still describe the process whose memory was read."""
    proc = _proc(tmp_path)
    directory = _process(proc, 100, pgid=100, start_time=10, pss_kib=100 * KIB)
    read_text = Path.read_text

    def replace_on_rollup_read(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path == directory / "smaps_rollup":
            _process(proc, 100, pgid=900, start_time=20, pss_kib=4000 * KIB)
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", replace_on_rollup_read)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    document = sampler.snapshot()
    assert document["peak"]["pss_kib"] == 0
    assert document["processes"] == []

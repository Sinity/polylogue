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
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

from devtools.pytest_memory import CUSTODY_ENV, MAX_ATTRIBUTED_PROCESSES, ProcessGroupMemorySampler

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
    fields = ["S", "1", str(pgid), *(["0"] * 16), str(pid * 10)]
    (directory / "stat").write_text(f"{pid} ({command} worker) " + " ".join(fields) + "\n", encoding="utf-8")
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


def test_marker_owns_detached_child_among_shared_cgroup_peers(tmp_path: Path) -> None:
    """A setsid child remains charged to the managed cgroup after leaving its process group."""
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    detached = _process(proc, 101, pgid=900, command="detached", pss_kib=700 * KIB)
    (detached / "environ").write_bytes((CUSTODY_ENV + "=owned").encode() + b"\0")
    unrelated = _process(proc, 555, pgid=555, command="peer", pss_kib=4000 * KIB)
    (unrelated / "environ").write_bytes((CUSTODY_ENV + "=other").encode() + b"\0")
    (proc / "self").mkdir()
    (proc / "self" / "cgroup").write_text("0::/managed/pytest\n")
    cgroup_root = tmp_path / "cgroup"
    member_dir = cgroup_root / "managed" / "pytest"
    member_dir.mkdir(parents=True)
    (member_dir / "cgroup.procs").write_text("100\n101\n555\n")
    sampler = ProcessGroupMemorySampler(100, proc=proc, custody_marker="owned", meminfo=_meminfo(tmp_path, 8000))
    sampler.sample()
    assert sampler.snapshot()["peak"]["pss_kib"] == 800 * KIB


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


def test_reused_numeric_group_never_acquires_custody(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    leader = _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    sampler = _sampler(tmp_path, proc)
    sampler.sample()
    (leader / "stat").write_text("100 (peer) S 1 100 " + "0 " * 16 + "99999\n")
    _process(proc, 101, pgid=100, pss_kib=8000 * KIB)
    sampler.sample()
    document = sampler.snapshot()
    assert document["peak"]["pss_kib"] == 100 * KIB
    assert document["processes_seen"] == 1


def test_proven_detached_identity_survives_unreadable_marker(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    child = _process(proc, 101, pgid=900, pss_kib=700 * KIB)
    (child / "environ").write_bytes((CUSTODY_ENV + "=owned").encode() + b"\0")
    sampler = ProcessGroupMemorySampler(100, proc=proc, custody_marker="owned", meminfo=_meminfo(tmp_path, 8000))
    sampler.sample()
    (child / "environ").unlink()
    _process(proc, 101, pgid=900, pss_kib=1100 * KIB)
    sampler.sample()
    assert sampler.snapshot()["peak"]["pss_kib"] == 1200 * KIB
    assert {entry["pid"] for entry in sampler.snapshot()["processes"]} == {100, 101}


def test_unknown_unreadable_marker_leaves_visible_incomplete_sampling(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    _process(proc, 555, pgid=555, pss_kib=9000 * KIB)
    sampler = ProcessGroupMemorySampler(100, proc=proc, custody_marker="owned", meminfo=_meminfo(tmp_path, 8000))
    sampler.sample()
    document = sampler.snapshot()
    assert document["peak"]["pss_kib"] == 100 * KIB
    assert document["incomplete"]
    assert {entry["pid"] for entry in document["processes"]} == {100}


def test_reused_pid_during_payload_read_never_contributes_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools import pytest_memory

    proc = _proc(tmp_path)
    leader = _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    sampler = _sampler(tmp_path, proc)
    original = pytest_memory._rollup

    def raced(pid: int, *, proc: Path) -> dict[str, int] | None:
        result = original(pid, proc=proc)
        (leader / "stat").write_text("100 (peer) S 1 100 " + "0 " * 16 + "99999\n")
        return result

    monkeypatch.setattr(pytest_memory, "_rollup", raced)
    sampler.sample()
    document = sampler.snapshot()
    assert document["observed_samples"] == 0
    assert document["unmeasured"] and document["incomplete"]
    assert document["processes"] == []


def test_custody_marker_comparison_streams_large_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.pytest_memory import _marker_matches
    from tests.infra.memory_attribution_fixture import EnvironmentReader

    sizes: list[int] = []
    original_open = Path.open

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original_open(path, *args, **kwargs)
        return EnvironmentReader(cast(BinaryIO, handle), sizes) if path.name == "environ" else handle

    proc = _proc(tmp_path)
    process = _process(proc, 100, pgid=100)
    (process / "environ").write_bytes(b"UNRELATED=" + b"x" * 300000 + b"\0" + (CUSTODY_ENV + "=owned").encode() + b"\0")
    monkeypatch.setattr(Path, "open", tracked_open)
    assert _marker_matches(100, "owned", proc=proc) is True
    assert _marker_matches(100, "other", proc=proc) is False
    assert sizes and all(size == 65536 for size in sizes), sizes


@pytest.mark.uses_real_clock("observes actual admitted controller and detached child processes")
@pytest.mark.parametrize("route", ["held", "queued"])
@pytest.mark.parametrize("reparent_before_sample", [False, True])
def test_actual_launch_owns_detached_child_and_excludes_shared_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    reparent_before_sample: bool,
) -> None:
    from devtools import pytest_slot
    from devtools.worker_memory import FOCUSED_CHARGE
    from tests.infra.memory_attribution_fixture import DETACHED_PROGRAM, wait_until_detached_child_is_reparented

    if not Path("/proc/self/stat").exists():
        pytest.skip("actual procfs custody observation is unavailable")
    observed_reparenting: list[tuple[int, int]] = []
    if reparent_before_sample:
        original_sampler = ProcessGroupMemorySampler

        def after_reparenting(pgid: int, **kwargs: Any) -> ProcessGroupMemorySampler:
            observed_reparenting.append(wait_until_detached_child_is_reparented(tmp_path))
            return original_sampler(pgid, **kwargs)

        monkeypatch.setattr(pytest_slot, "ProcessGroupMemorySampler", after_reparenting)
    monkeypatch.setenv(CUSTODY_ENV, "ambient-parent")
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    environment = {**os.environ, "POLYLOGUE_PYTEST_RUN_ID": "neutral-receipt", CUSTODY_ENV: "ambient-parent"}
    command = [sys.executable, "-c", DETACHED_PROGRAM, str(tmp_path)]
    try:
        if route == "held":
            code, receipt = pytest_slot._run_held_admitted(
                command,
                cwd=str(tmp_path),
                env=environment,
                stdout=None,
                on_exit=lambda: None,
                started=time.monotonic(),
                worktree_provenance=None,
                profile=FOCUSED_CHARGE,
                sizing=None,
                telemetry_path=None,
                result_path=None,
                on_interrupt=None,
            )
        else:
            monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: None)
            monkeypatch.setattr(pytest_slot, "admit_width", lambda argv, **_kwargs: (list(argv), None))
            launch, log = tmp_path / "launch.json", tmp_path / "slot.log"
            pytest_slot._write_launch(launch, argv=command, cwd=str(tmp_path), env=environment, log_path=log)
            code = pytest_slot._run_launch(launch)
            receipt = json.loads(log.with_suffix(".result.json").read_text())
        identities = json.loads((tmp_path / "identities.json").read_text())
        assert code == 0
        pids = {entry["pid"] for entry in receipt["memory"]["processes"]}
        assert identities["detached"] in pids, pids
        if reparent_before_sample:
            assert observed_reparenting == [(identities["controller"], identities["detached"])]
            assert identities["controller"] not in pids, pids
        else:
            assert identities["controller"] in pids, pids
        assert peer.pid not in pids and os.getpid() not in pids, pids
        assert identities["custody"] != "ambient-parent" and identities["receipt"] == "neutral-receipt"
        assert os.environ[CUSTODY_ENV] == environment[CUSTODY_ENV] == "ambient-parent"
    finally:
        (tmp_path / "stop").touch()
        peer.terminate()
        peer.wait()
        path = tmp_path / "identities.json"
        if path.exists():
            detached = json.loads(path.read_text())["detached"]
            pytest_slot._group_reaped(detached)


def test_reused_leader_between_group_check_and_payload_is_not_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools import pytest_memory

    proc = _proc(tmp_path)
    leader = _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    sampler = _sampler(tmp_path, proc)
    original = pytest_memory._identity
    reads = 0

    def raced(pid: int, *, proc: Path) -> object:
        nonlocal reads
        if pid == 100:
            reads += 1
            if reads == 2:
                (leader / "stat").write_text("100 (peer) S 1 100 " + "0 " * 16 + "99999\n")
        return original(pid, proc=proc)

    monkeypatch.setattr(pytest_memory, "_identity", raced)
    sampler.sample()
    document = sampler.snapshot()
    assert document["observed_samples"] == 0 and document["incomplete"]
    assert document["processes"] == []


def test_reused_owned_pid_keeps_separate_birth_identity(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    leader = _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    (leader / "environ").write_bytes((CUSTODY_ENV + "=owned").encode() + b"\0")
    sampler = ProcessGroupMemorySampler(100, proc=proc, custody_marker="owned", meminfo=_meminfo(tmp_path, 8000))
    sampler.sample()
    (leader / "stat").write_text("100 (successor) S 1 100 " + "0 " * 16 + "99999\n")
    sampler.sample()
    document = sampler.snapshot()
    assert document["processes_seen"] == 2
    assert {(entry["pid"], entry["start_ticks"]) for entry in document["processes"]} == {(100, 1000), (100, 99999)}


def test_enumeration_fault_preserves_proven_birth_for_resumed_marker_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _proc(tmp_path)
    _process(proc, 100, pgid=100, pss_kib=100 * KIB)
    child = _process(proc, 101, pgid=900, pss_kib=700 * KIB)
    (child / "environ").write_bytes((CUSTODY_ENV + "=owned").encode() + b"\0")
    sampler = ProcessGroupMemorySampler(100, proc=proc, custody_marker="owned", meminfo=_meminfo(tmp_path, 8000))
    sampler.sample()
    original = Path.iterdir

    def denied(path: Path) -> Any:
        if path == proc:
            raise PermissionError("neutral enumeration fault")
        return original(path)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "iterdir", denied)
        sampler.sample()
    (child / "environ").unlink()
    _process(proc, 101, pgid=900, pss_kib=1400 * KIB)
    sampler.sample()
    document = sampler.snapshot()
    assert document["peak"]["pss_kib"] == 1500 * KIB
    assert document["incomplete"]
    assert document["processes_seen"] == 2
    assert {entry["pid"] for entry in document["processes"]} == {100, 101}

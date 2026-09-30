"""Concurrent pytest slots reserve pool charge before launching workers."""

from __future__ import annotations

from pathlib import Path

from devtools.pytest_memory_admission import AdmissionLedger, admit_width
from devtools.worker_memory import ChargeProfile


def test_a_waiting_slot_retries_after_the_admitted_job_releases(tmp_path: Path, monkeypatch) -> None:
    pool = tmp_path / "pool.slice"
    pool.mkdir()
    job = pool / "job.slice"
    job.mkdir()
    (job / "memory.current").write_text("0", encoding="ascii")
    ledger_dir = tmp_path / "admission"
    first = AdmissionLedger(ledger_dir, proc=tmp_path / "proc", pid=100)
    second = AdmissionLedger(ledger_dir, proc=tmp_path / "proc", pid=101)
    monkeypatch.setattr("devtools.pytest_memory_admission._process_start_ticks", lambda pid, *, proc: pid + 1000)
    profile = ChargeProfile(worker_anon_mib=800, worker_cache_mib=200, controller_mib=0)

    def size(argv, *, outstanding_mib=None, **_kwargs):
        available = 1000 - (outstanding_mib(pool) if outstanding_mib is not None else 0)
        admitted = available >= 1000
        return list(argv), {
            "admission": "admitted" if admitted else "resource_not_ready",
            "cgroup_directory": str(job),
            "predicted_charge_mib": 1000.0,
            "workers": 1 if admitted else 0,
            "narrowed": False,
        }

    first_command, first_basis = admit_width(
        ["pytest"],
        size=size,
        profile=profile,
        max_workers=1,
        ledger=first,
        report=lambda _message: None,
    )
    assert first_command == ["pytest"]
    assert first_basis is not None and first_basis["workers"] == 1

    waits: list[float] = []

    def release_first(delay: float) -> None:
        waits.append(delay)
        first._path(100).unlink(missing_ok=True)

    command, basis = admit_width(
        ["pytest"],
        size=size,
        profile=profile,
        max_workers=1,
        ledger=second,
        report=lambda _message: None,
        sleep=release_first,
    )

    assert command == ["pytest"]
    assert basis is not None and basis["workers"] == 1
    assert waits
    assert basis["admission_ledger"]["waited_s"] == 0.0
    second.release()


def test_a_shortfall_with_no_reclaimable_job_is_reported_without_reserving(tmp_path: Path, monkeypatch) -> None:
    ledger = AdmissionLedger(tmp_path / "admission", proc=tmp_path / "proc", pid=101)
    monkeypatch.setattr("devtools.pytest_memory_admission._process_start_ticks", lambda pid, *, proc: pid + 1000)
    profile = ChargeProfile(worker_anon_mib=800, worker_cache_mib=200, controller_mib=0)

    command, basis = admit_width(
        ["pytest"],
        size=lambda argv, **_kwargs: (
            list(argv),
            {
                "admission": "resource_not_ready",
                "cgroup_directory": str(tmp_path),
                "predicted_charge_mib": 1000.0,
                "workers": 0,
            },
        ),
        profile=profile,
        max_workers=1,
        ledger=ledger,
        report=lambda _message: None,
    )

    assert command == ["pytest"]
    assert basis is not None and basis["admission"] == "resource_not_ready"
    assert not ledger._path(101).exists()


def test_a_holder_in_an_unrelated_pool_does_not_make_this_slot_wait(tmp_path: Path, monkeypatch) -> None:
    pools = tmp_path / "pools"
    pool_a = pools / "pool-a.slice"
    pool_b = pools / "pool-b.slice"
    job_a = pool_a / "job.slice"
    job_b = pool_b / "job.slice"
    for directory in (pool_a, pool_b, job_a, job_b):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "memory.current").write_text("0", encoding="ascii")
    ledger_dir = tmp_path / "admission"
    first = AdmissionLedger(ledger_dir, proc=tmp_path / "proc", pid=100)
    second = AdmissionLedger(ledger_dir, proc=tmp_path / "proc", pid=101)
    monkeypatch.setattr("devtools.pytest_memory_admission._process_start_ticks", lambda pid, *, proc: pid + 1000)
    profile = ChargeProfile(worker_anon_mib=800, worker_cache_mib=200, controller_mib=0)

    admit_width(
        ["pytest"],
        size=lambda argv, **_kwargs: (
            list(argv),
            {
                "admission": "admitted",
                "cgroup_directory": str(job_b),
                "limiting_cgroups": [str(pool_b)],
                "predicted_charge_mib": 1000.0,
                "workers": 1,
            },
        ),
        profile=profile,
        max_workers=1,
        ledger=first,
        report=lambda _message: None,
    )

    _command, basis = admit_width(
        ["pytest"],
        size=lambda argv, *, outstanding_mib=None, **_kwargs: (
            list(argv),
            {
                "admission": "resource_not_ready",
                "cgroup_directory": str(job_a),
                "limiting_cgroups": [str(pool_a)],
                "predicted_charge_mib": 1000.0,
                "workers": 0,
            },
        ),
        profile=profile,
        max_workers=1,
        ledger=second,
        report=lambda _message: None,
    )

    assert basis is not None and basis["admission"] == "resource_not_ready"
    assert basis["admission_ledger"]["holders"] == 0
    assert not second._path(101).exists()
    first.release()

"""Next-start termination reconciliation (polylogue-peo).

Every test runs the production route: a real ``DaemonLifecycle`` row in a real
ops tier, a later run's reconciliation over it, and the persisted receipt.
Only the host is synthetic: a fake evidence adapter stands in for the journal,
the kernel log and the cgroup filesystem, and the adapter tests at the end run
the production ``HostTerminationEvidence`` against a fake ``journalctl`` and
fake ``/proc`` and ``/sys/fs/cgroup`` trees.

Anti-vacuity, per mutation the bead names:

- drop the run id join (receipts keyed by anything but the lifecycle row's
  run id, or the workload receipt's ``daemon_run_id``) and
  ``test_receipt_joins_the_run_and_its_last_workload_by_run_id`` goes red;
- drop the kernel pid match or the cgroup group-kill delta and the OOM and
  cgroup tests classify ``unknown``;
- drop the ``unknown`` fallthrough (defaulting to crash or external stop) and
  ``test_temporal_adjacency_alone_stays_unknown`` goes red.
"""

from __future__ import annotations

import json
import sqlite3
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from polylogue.daemon import lifecycle as lifecycle_module
from polylogue.daemon.lifecycle import DaemonLifecycle, lifecycle_status
from polylogue.operations.daemon_termination import (
    SOURCE_BOOT,
    SOURCE_CGROUP,
    SOURCE_KERNEL_OOM,
    SOURCE_SERVICE_MANAGER,
    SOURCE_WORKLOAD,
    EvidenceSource,
    HostRunIdentity,
    HostTerminationEvidence,
    TerminationClassification,
    TerminationReceipt,
    capture_host_run_identity,
)
from tests.infra.frozen_clock import FrozenClock

PRIOR_PID = 4242
BOOT = "11111111-2222-3333-4444-555555555555"
INVOCATION = "0123456789abcdef0123456789abcdef"
CGROUP = "/user.slice/user-1000.slice/user@1000.service/app.slice/polylogued.service"


def _host(**overrides: object) -> HostRunIdentity:
    values: dict[str, object] = {
        "pid": PRIOR_PID,
        "boot_id": BOOT,
        "invocation_id": INVOCATION,
        "unit": "polylogued.service",
        "cgroup_path": CGROUP,
        "cgroup_inode": 77,
        "memory_events": {"oom": 0, "oom_kill": 0, "oom_group_kill": 0},
    }
    values.update(overrides)
    return HostRunIdentity(**values)  # type: ignore[arg-type]


@dataclass
class FakeEvidence:
    """Synthetic host: each source answers with records or a missing reason."""

    manager: Sequence[Mapping[str, object]] | str = "no_manager_records_for_invocation"
    kernel: Sequence[Mapping[str, object]] | str = ()
    oomd: Sequence[Mapping[str, object]] | str = ()
    cgroup: Mapping[str, object] | str = field(
        default_factory=lambda: {"inode": 77, "oom": 0, "oom_kill": 0, "oom_group_kill": 0}
    )
    boot_id: str | None = BOOT
    calls: list[str] = field(default_factory=list)

    @staticmethod
    def _source(name: str, value: Sequence[Mapping[str, object]] | str) -> EvidenceSource:
        if isinstance(value, str):
            return EvidenceSource.missing(name, value)
        return EvidenceSource(name=name, records=tuple(value), refs=(f"fake:{name}",))

    def service_manager(self, host: HostRunIdentity) -> EvidenceSource:
        self.calls.append("service_manager")
        return self._source(SOURCE_SERVICE_MANAGER, self.manager)

    def kernel_oom(self, host: HostRunIdentity, *, since_ms: int, until_ms: int) -> EvidenceSource:
        self.calls.append("kernel_oom")
        return self._source(SOURCE_KERNEL_OOM, self.kernel)

    def oomd_kills(self, host: HostRunIdentity, *, since_ms: int, until_ms: int) -> EvidenceSource:
        return self._source("systemd_oomd", self.oomd)

    def cgroup_memory_events(self, host: HostRunIdentity) -> EvidenceSource:
        if isinstance(self.cgroup, str):
            return EvidenceSource.missing(SOURCE_CGROUP, self.cgroup)
        return EvidenceSource(name=SOURCE_CGROUP, records=(self.cgroup,), refs=("fake:cgroup",))

    def current_boot_id(self) -> str | None:
        return self.boot_id


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(lifecycle_module, "archive_root", lambda: tmp_path)
    return tmp_path


def _prior_run(
    archive: Path,
    clock: FrozenClock,
    *,
    host: HostRunIdentity | None = None,
    stop: str | None = None,
    signal_name: str | None = None,
) -> DaemonLifecycle:
    run = DaemonLifecycle.start(run_id="prior-run", archive_root_path=archive, host=host or _host())
    clock.advance(60)
    run.heartbeat()
    clock.advance(30)
    if signal_name is not None:
        with sqlite3.connect(archive / "ops.db") as conn:
            conn.execute("UPDATE daemon_lifecycle SET signal = ? WHERE run_id = ?", (signal_name, run.run_id))
    if stop is not None:
        run.stop(exit_kind=stop)
    clock.advance(30)
    return run


def _reconcile(archive: Path, evidence: FakeEvidence) -> tuple[DaemonLifecycle, tuple[TerminationReceipt, ...]]:
    current = DaemonLifecycle.start(run_id="current-run", archive_root_path=archive, host=_host(pid=5151))
    receipts = current.reconcile_ended_runs(evidence)
    current.record_termination_receipts(receipts)
    return current, receipts


def _only(receipts: tuple[TerminationReceipt, ...]) -> TerminationReceipt:
    assert [receipt.run_id for receipt in receipts] == ["prior-run"]
    return receipts[0]


# ---------------------------------------------------------------------------
# The run's own stop marker
# ---------------------------------------------------------------------------


def test_a_clean_stop_classifies_from_the_stop_marker_without_host_evidence(
    archive: Path, frozen_clock: FrozenClock
) -> None:
    prior = _prior_run(archive, frozen_clock, stop="clean")
    evidence = FakeEvidence()
    _current, receipts = _reconcile(archive, evidence)

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.CLEAN
    assert receipt.observed["lifecycle.exit_kind"] == "clean"
    assert f"ops:daemon_lifecycle:{prior.run_id}" in receipt.evidence_refs
    assert evidence.calls == []


def test_a_handled_sigterm_classifies_as_handled_signal(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock, stop="signal", signal_name="SIGTERM")
    _current, receipts = _reconcile(archive, FakeEvidence())

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.HANDLED_SIGNAL
    assert receipt.observed["lifecycle.signal"] == "SIGTERM"


@pytest.mark.parametrize("exit_kind", ["error", "atexit"])
def test_an_abnormal_in_process_stop_is_a_crash(archive: Path, frozen_clock: FrozenClock, exit_kind: str) -> None:
    _prior_run(archive, frozen_clock, stop=exit_kind)
    _current, receipts = _reconcile(archive, FakeEvidence())
    assert _only(receipts).classification is TerminationClassification.CRASH


# ---------------------------------------------------------------------------
# A vanished run: host evidence tied to it by identity
# ---------------------------------------------------------------------------


def test_sigkill_from_outside_is_an_external_stop(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    evidence = FakeEvidence(
        manager=[{"MESSAGE_ID": "98e322203f7a4ed290d09fe03c09fe15", "EXIT_CODE": "killed", "EXIT_STATUS": "KILL"}]
    )
    _current, receipts = _reconcile(archive, evidence)

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.EXTERNAL_STOP
    assert receipt.observed[f"{SOURCE_SERVICE_MANAGER}.exit_code"] == "killed"
    assert "cause" in receipt.inferred
    assert "fake:service_manager" in receipt.evidence_refs


def test_a_stop_timeout_escalation_is_an_external_stop(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock, signal_name="SIGTERM")
    evidence = FakeEvidence(manager=[{"UNIT_RESULT": "timeout"}])
    _current, receipts = _reconcile(archive, evidence)
    assert _only(receipts).classification is TerminationClassification.EXTERNAL_STOP


def test_a_signal_handler_exit_without_a_stop_marker_is_a_handled_signal(
    archive: Path, frozen_clock: FrozenClock
) -> None:
    """Exit 143 after a recorded SIGTERM is the handler's own exit; any other status is a crash."""
    _prior_run(archive, frozen_clock, signal_name="SIGTERM")
    _current, receipts = _reconcile(archive, FakeEvidence(manager=[{"EXIT_CODE": "exited", "EXIT_STATUS": "143"}]))
    assert _only(receipts).classification is TerminationClassification.HANDLED_SIGNAL


def test_an_abnormal_exit_status_is_a_crash(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    _current, receipts = _reconcile(archive, FakeEvidence(manager=[{"EXIT_CODE": "exited", "EXIT_STATUS": "144"}]))
    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.CRASH
    assert receipt.observed[f"{SOURCE_SERVICE_MANAGER}.exit_status"] == "144"


def test_the_service_manager_watchdog_is_classified(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    evidence = FakeEvidence(manager=[{"UNIT_RESULT": "watchdog", "EXIT_CODE": "dumped", "EXIT_STATUS": "ABRT"}])
    _current, receipts = _reconcile(archive, evidence)
    assert _only(receipts).classification is TerminationClassification.WATCHDOG


def test_a_kernel_oom_kill_naming_the_runs_pid_is_an_oom_kill(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    evidence = FakeEvidence(
        manager=[{"EXIT_CODE": "killed", "EXIT_STATUS": "KILL"}],
        kernel=[{"MESSAGE": f"Out of memory: Killed process {PRIOR_PID} (python3) total-vm:9000000kB"}],
    )
    _current, receipts = _reconcile(archive, evidence)

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.OOM_KILL
    assert receipt.observed[f"{SOURCE_KERNEL_OOM}.killed_pid"] == PRIOR_PID


def test_the_manager_oom_result_is_an_oom_kill(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    _current, receipts = _reconcile(archive, FakeEvidence(manager=[{"UNIT_RESULT": "oom-kill"}]))
    assert _only(receipts).classification is TerminationClassification.OOM_KILL


def test_a_group_kill_of_the_same_cgroup_instance_is_a_cgroup_kill(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    evidence = FakeEvidence(cgroup={"inode": 77, "oom": 3, "oom_kill": 2, "oom_group_kill": 1})
    _current, receipts = _reconcile(archive, evidence)

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.CGROUP_KILL
    assert receipt.observed[f"{SOURCE_CGROUP}.oom_group_kill_delta"] == 1


def test_a_recreated_cgroup_cannot_be_compared(archive: Path, frozen_clock: FrozenClock) -> None:
    """A new cgroup instance restarted its counters: nothing to compare, and it says so."""
    _prior_run(archive, frozen_clock)
    evidence = FakeEvidence(cgroup={"inode": 99, "oom": 0, "oom_kill": 0, "oom_group_kill": 5})
    _current, receipts = _reconcile(archive, evidence)

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.UNKNOWN
    assert receipt.missing_sources[SOURCE_CGROUP] == "cgroup_recreated"


def test_a_changed_boot_id_is_a_host_gap(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    _current, receipts = _reconcile(archive, FakeEvidence(boot_id="99999999-0000-0000-0000-000000000000"))

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.HOST_GAP
    assert receipt.observed[f"{SOURCE_BOOT}.changed"] is True


def test_temporal_adjacency_alone_stays_unknown(archive: Path, frozen_clock: FrozenClock) -> None:
    """Another process's OOM kill inside the window is reported, never used as the cause."""
    _prior_run(archive, frozen_clock)
    evidence = FakeEvidence(kernel=[{"MESSAGE": "Out of memory: Killed process 777 (chrome)"}])
    _current, receipts = _reconcile(archive, evidence)

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.UNKNOWN
    assert receipt.observed[f"{SOURCE_KERNEL_OOM}.other_killed_pids"] == [777]
    assert receipt.missing_sources[SOURCE_SERVICE_MANAGER] == "no_manager_records_for_invocation"


def test_a_vanished_run_with_no_evidence_names_what_is_missing(archive: Path, frozen_clock: FrozenClock) -> None:
    """The historical exit-144 shape: no retained source, so no postmortem is manufactured."""
    prior = _prior_run(archive, frozen_clock)
    evidence = FakeEvidence(
        manager="journal_query_failed",
        kernel="journal_query_failed",
        oomd="journal_query_failed",
        cgroup="cgroup_removed",
        boot_id=None,
    )
    current, receipts = _reconcile(archive, evidence)

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.UNKNOWN
    assert receipt.missing_sources == {
        SOURCE_SERVICE_MANAGER: "journal_query_failed",
        SOURCE_KERNEL_OOM: "journal_query_failed",
        "systemd_oomd": "journal_query_failed",
        SOURCE_CGROUP: "cgroup_removed",
        SOURCE_BOOT: "current_boot_id_unavailable",
        SOURCE_WORKLOAD: "resident_workload_receipt_unavailable",
    }
    with sqlite3.connect(archive / "ops.db") as conn:
        last_heartbeat = conn.execute(
            "SELECT last_heartbeat_at_ms FROM daemon_lifecycle WHERE run_id = ?", (prior.run_id,)
        ).fetchone()[0]
        current_start = conn.execute(
            "SELECT started_at_ms FROM daemon_lifecycle WHERE run_id = ?", (current.run_id,)
        ).fetchone()[0]
    assert receipt.last_good_at_ms == last_heartbeat
    assert (receipt.ended_after_ms, receipt.ended_before_ms) == (last_heartbeat, current_start)
    assert receipt.to_payload()["remediation"]


def test_a_run_that_recorded_no_host_identity_is_unknown_not_guessed(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    with sqlite3.connect(archive / "ops.db") as conn:
        conn.execute("UPDATE daemon_lifecycle SET details_json = '{}' WHERE run_id = 'prior-run'")
    evidence = FakeEvidence(manager=[{"UNIT_RESULT": "oom-kill"}])
    _current, receipts = _reconcile(archive, evidence)

    receipt = _only(receipts)
    assert receipt.classification is TerminationClassification.UNKNOWN
    assert receipt.missing_sources[SOURCE_SERVICE_MANAGER] == "run_host_identity_unrecorded"
    assert evidence.calls == []


# ---------------------------------------------------------------------------
# Persistence, idempotence, status and the run-id join
# ---------------------------------------------------------------------------


def test_reconciliation_is_idempotent_per_run(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    current, receipts = _reconcile(archive, FakeEvidence(manager=[{"UNIT_RESULT": "watchdog"}]))

    assert current.reconcile_ended_runs(FakeEvidence()) == ()
    assert current.record_termination_receipts(receipts) == 0
    with sqlite3.connect(archive / "ops.db") as conn:
        rows = conn.execute("SELECT run_id, classification FROM daemon_termination_receipts").fetchall()
    assert rows == [("prior-run", "watchdog")]


def test_every_unreconciled_run_is_classified_against_its_successors_start(
    archive: Path, frozen_clock: FrozenClock
) -> None:
    """A run that died before its successor reconciled it is still accounted for."""
    DaemonLifecycle.start(run_id="oldest-run", archive_root_path=archive, host=_host(pid=1))
    frozen_clock.advance(100)
    middle = DaemonLifecycle.start(run_id="middle-run", archive_root_path=archive, host=_host(pid=2))
    frozen_clock.advance(100)
    current = DaemonLifecycle.start(run_id="current-run", archive_root_path=archive, host=_host(pid=3))

    receipts = current.reconcile_ended_runs(FakeEvidence())
    assert [receipt.run_id for receipt in receipts] == ["oldest-run", "middle-run"]
    with sqlite3.connect(archive / "ops.db") as conn:
        middle_start = conn.execute(
            "SELECT started_at_ms FROM daemon_lifecycle WHERE run_id = ?", (middle.run_id,)
        ).fetchone()[0]
    assert receipts[0].ended_before_ms == middle_start


def test_status_reports_the_last_termination_and_pending_runs(archive: Path, frozen_clock: FrozenClock) -> None:
    _prior_run(archive, frozen_clock)
    current = DaemonLifecycle.start(run_id="current-run", archive_root_path=archive, host=_host(pid=5151))

    pending = lifecycle_status()
    assert pending["run_id"] == current.run_id
    assert pending["last_termination"] is None
    assert pending["pending_reconciliation_runs"] == ["prior-run"]

    current.record_termination_receipts(current.reconcile_ended_runs(FakeEvidence()))
    reconciled = lifecycle_status()
    last = cast("dict[str, object]", reconciled["last_termination"])
    assert last["run_id"] == "prior-run"
    assert last["classification"] == "unknown"
    assert last["remediation"]
    assert reconciled["pending_reconciliation_runs"] == []


def test_receipt_joins_the_run_and_its_last_workload_by_run_id(
    archive: Path, frozen_clock: FrozenClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route observed inside the prior daemon run carries its run id to the receipt."""
    from polylogue.logging import set_run_context
    from polylogue.operations.route_observation import measure_route

    prior = _prior_run(archive, frozen_clock)
    set_run_context(run_id=prior.run_id, component="daemon")
    try:
        with measure_route(surface="cli", route="cli.status") as observation:
            pass
    finally:
        set_run_context()
    assert observation.receipt is not None
    workload = observation.receipt.to_workload_receipt()
    assert workload.daemon_run_id == prior.run_id
    assert workload.to_payload()["daemon_run_id"] == prior.run_id

    _current, receipts = _reconcile(archive, FakeEvidence())
    receipt = _only(receipts)
    assert receipt.last_workload is None
    assert SOURCE_WORKLOAD in receipt.missing_sources


def test_a_route_outside_a_daemon_carries_no_daemon_run_id(archive: Path) -> None:
    from polylogue.operations.route_observation import measure_route

    with measure_route(surface="cli", route="cli.status") as observation:
        pass
    assert observation.receipt is not None
    workload = observation.receipt.to_workload_receipt()
    assert workload.daemon_run_id is None
    assert "daemon_run_id" not in workload.to_payload()


# ---------------------------------------------------------------------------
# The production host adapter against fake host sources
# ---------------------------------------------------------------------------


def _fake_proc(root: Path, *, pid: int, cgroup: str) -> Path:
    proc = root / "proc"
    (proc / str(pid)).mkdir(parents=True)
    (proc / str(pid) / "cgroup").write_text(f"0::{cgroup}\n")
    (proc / "sys" / "kernel" / "random").mkdir(parents=True)
    (proc / "sys" / "kernel" / "random" / "boot_id").write_text(BOOT + "\n")
    return proc


def test_capture_and_read_back_the_cgroup_baseline(tmp_path: Path) -> None:
    proc = _fake_proc(tmp_path, pid=PRIOR_PID, cgroup=CGROUP)
    cgroup_root = tmp_path / "cgroup"
    unit_dir = cgroup_root / CGROUP.lstrip("/")
    unit_dir.mkdir(parents=True)
    (unit_dir / "memory.events").write_text("low 0\nhigh 0\nmax 4\noom 1\noom_kill 1\noom_group_kill 0\n")

    host = capture_host_run_identity(
        pid=PRIOR_PID,
        environ={"INVOCATION_ID": INVOCATION, "SYSTEMD_EXEC_PID": str(PRIOR_PID)},
        proc=proc,
        cgroup_root=cgroup_root,
    )
    assert host.boot_id == BOOT
    assert host.invocation_id == INVOCATION
    # An id inherited from an ancestor unit is not this run's invocation.
    inherited = capture_host_run_identity(
        pid=PRIOR_PID,
        environ={"INVOCATION_ID": INVOCATION, "SYSTEMD_EXEC_PID": "1"},
        proc=proc,
        cgroup_root=cgroup_root,
    )
    assert inherited.invocation_id is None
    assert host.unit == "polylogued.service"
    assert host.memory_events == {"oom": 1, "oom_kill": 1, "oom_group_kill": 0}
    assert HostRunIdentity.from_details({"host": host.to_details()}) == host

    (unit_dir / "memory.events").write_text("oom 2\noom_kill 1\noom_group_kill 1\n")
    adapter = HostTerminationEvidence(journalctl="/nonexistent/journalctl", proc=proc, cgroup_root=cgroup_root)
    current = adapter.cgroup_memory_events(host)
    assert current.missing_reason is None
    assert current.records[0] == {"inode": host.cgroup_inode, "oom": 2, "oom_kill": 1, "oom_group_kill": 1}
    assert adapter.current_boot_id() == BOOT


def _fake_journalctl(tmp_path: Path, records: Sequence[Mapping[str, object]]) -> str:
    script = tmp_path / "journalctl"
    lines = "\n".join(json.dumps(record) for record in records)
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{lines}\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_the_journal_adapter_reads_unit_results_and_kernel_pids(tmp_path: Path) -> None:
    journal = _fake_journalctl(
        tmp_path,
        [
            {
                "MESSAGE": "polylogued.service: Main process exited, code=killed, status=9/KILL",
                "MESSAGE_ID": "98e322203f7a4ed290d09fe03c09fe15",
                "EXIT_CODE": "killed",
                "EXIT_STATUS": "KILL",
            },
            {"MESSAGE": "an unrelated line"},
            {"MESSAGE": f"Out of memory: Killed process {PRIOR_PID} (python3)"},
        ],
    )
    adapter = HostTerminationEvidence(journalctl=journal)
    host = _host()

    manager = adapter.service_manager(host)
    assert manager.missing_reason is None
    assert [record.get("EXIT_STATUS") for record in manager.records] == ["KILL"]
    kernel = adapter.kernel_oom(host, since_ms=0, until_ms=1_000)
    assert [record["MESSAGE"] for record in kernel.records] == [f"Out of memory: Killed process {PRIOR_PID} (python3)"]


def test_the_journal_adapter_names_why_a_source_is_missing(tmp_path: Path) -> None:
    adapter = HostTerminationEvidence(journalctl="/nonexistent/journalctl")
    assert adapter.service_manager(_host()).missing_reason == "journalctl_unavailable"
    assert adapter.service_manager(_host(invocation_id=None)).missing_reason == "run_not_under_service_manager"
    empty = HostTerminationEvidence(journalctl=_fake_journalctl(tmp_path, []))
    assert empty.service_manager(_host()).missing_reason == "no_manager_records_for_invocation"

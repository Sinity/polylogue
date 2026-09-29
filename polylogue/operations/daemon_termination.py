"""Next-start reconciliation of how each daemon run ended (polylogue-peo).

A process cannot explain its own SIGKILL, OOM kill, cgroup kill, watchdog
abort or host loss: nothing in it runs afterwards. The in-process forensics in
:mod:`polylogue.daemon.lifecycle` (signal row, stop marker, atexit sentinel,
heartbeats) record what the process *could* see. Everything else is read by
the next run, from outside: the service manager's record of the prior
invocation, kernel and ``systemd-oomd`` kill records naming the prior process
or cgroup, the cgroup's ``memory.events`` counters against the baseline the
prior run recorded at start, and the boot identity.

The result is one :class:`TerminationReceipt` per ended run, persisted once in
the ops tier. It separates what a source directly observed from what the
classification infers, cites the evidence it used, bounds when the run ended,
and names every source it could not consult and why. Classification needs
direct evidence tied to the run by identity -- its pid, systemd invocation id,
cgroup instance or boot id. Proximity in time never upgrades ``unknown``: a
kernel OOM record for another pid inside the window is reported, not used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from polylogue.core.types import DaemonTerminationClass, require_literal
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    ArchiveDaemonLifecycle,
    UnreconciledDaemonRun,
    latest_daemon_termination_receipt,
    record_daemon_termination_receipt,
    unreconciled_daemon_runs,
)


class TerminationClassification(str, Enum):
    """How a daemon run ended. Values are :data:`DaemonTerminationClass`."""

    CLEAN = "clean"
    HANDLED_SIGNAL = "handled_signal"
    EXTERNAL_STOP = "external_stop"
    WATCHDOG = "watchdog"
    OOM_KILL = "oom_kill"
    CGROUP_KILL = "cgroup_kill"
    CRASH = "crash"
    HOST_GAP = "host_gap"
    UNKNOWN = "unknown"


# The enum and the storage vocabulary are one set; a member added to one alone
# would be refused at the write boundary.
for _member in TerminationClassification:
    require_literal(_member.value, DaemonTerminationClass, name="daemon termination classification")

_REMEDIATION: Mapping[TerminationClassification, str] = {
    TerminationClassification.CLEAN: "none: the run stopped on request",
    TerminationClassification.HANDLED_SIGNAL: "none: the run handled a terminating signal and shut down",
    TerminationClassification.EXTERNAL_STOP: (
        "find what stopped or killed the unit at the recorded bounds; a stop that escalated to SIGKILL "
        "means shutdown outlasted the service's stop timeout"
    ),
    TerminationClassification.WATCHDOG: (
        "the service manager's watchdog aborted the run: inspect the last workload and the thread dump"
    ),
    TerminationClassification.OOM_KILL: (
        "the run was killed for memory: compare the unit's memory limit with the last workload's peak"
    ),
    TerminationClassification.CGROUP_KILL: (
        "the unit's cgroup was killed as a group (OOM group kill or systemd-oomd): check memory pressure policy"
    ),
    TerminationClassification.CRASH: (
        "the run exited abnormally: read the journal and the last workload at the recorded bounds"
    ),
    TerminationClassification.HOST_GAP: (
        "the host rebooted while the run was live; no in-host record of the run's own end survives"
    ),
    TerminationClassification.UNKNOWN: (
        "no retained source records how the run ended; the missing sources name what to retain"
    ),
}

#: Source names, stable tokens in ``missing_sources`` and ``evidence_refs``.
SOURCE_LIFECYCLE = "lifecycle"
SOURCE_SERVICE_MANAGER = "service_manager"
SOURCE_KERNEL_OOM = "kernel_oom"
SOURCE_OOMD = "systemd_oomd"
SOURCE_CGROUP = "cgroup_memory_events"
SOURCE_BOOT = "boot_identity"
SOURCE_WORKLOAD = "workload"

_MEMORY_EVENT_COUNTERS = ("oom", "oom_kill", "oom_group_kill")


# ---------------------------------------------------------------------------
# What a run records about its host at start
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HostRunIdentity:
    """The host-side identity one daemon run records when it starts.

    Each field is what a later reconciliation joins host evidence on: the pid
    for kernel kill records, the service manager's invocation id for the unit
    result, the cgroup path and inode (the cgroup *instance*) plus its
    ``memory.events`` counters for group kills, and the boot id for host loss.
    ``None`` means the start could not observe it; the receipt then names that
    source as missing rather than guessing.
    """

    pid: int
    boot_id: str | None = None
    invocation_id: str | None = None
    unit: str | None = None
    cgroup_path: str | None = None
    cgroup_inode: int | None = None
    memory_events: Mapping[str, int] | None = None

    def to_details(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "boot_id": self.boot_id,
            "invocation_id": self.invocation_id,
            "unit": self.unit,
            "cgroup_path": self.cgroup_path,
            "cgroup_inode": self.cgroup_inode,
            "memory_events": None if self.memory_events is None else dict(self.memory_events),
        }

    @classmethod
    def from_details(cls, details: Mapping[str, object]) -> HostRunIdentity | None:
        """Read the identity a lifecycle row's ``details['host']`` recorded, if any."""
        host = details.get("host")
        if not isinstance(host, Mapping):
            return None
        pid = host.get("pid")
        if not isinstance(pid, int):
            return None
        raw_events = host.get("memory_events")
        memory_events = (
            {str(key): int(value) for key, value in raw_events.items() if isinstance(value, int)}
            if isinstance(raw_events, Mapping)
            else None
        )
        inode = host.get("cgroup_inode")
        return cls(
            pid=pid,
            boot_id=_optional_text(host.get("boot_id")),
            invocation_id=_optional_text(host.get("invocation_id")),
            unit=_optional_text(host.get("unit")),
            cgroup_path=_optional_text(host.get("cgroup_path")),
            cgroup_inode=inode if isinstance(inode, int) else None,
            memory_events=memory_events,
        )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def read_memory_events(path: Path) -> dict[str, int] | None:
    """Parse a cgroup v2 ``memory.events`` file, or ``None`` when unreadable."""
    try:
        text = path.read_text()
    except OSError:
        return None
    counters: dict[str, int] = {}
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        if name in _MEMORY_EVENT_COUNTERS and value.strip().isdigit():
            counters[name] = int(value)
    return counters


def _own_cgroup_path(proc: Path, pid: int) -> str | None:
    try:
        text = (proc / str(pid) / "cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        # cgroup v2 unified hierarchy: ``0::/user.slice/.../polylogued.service``
        if line.startswith("0::"):
            path = line[3:].strip()
            return path or None
    return None


def capture_host_run_identity(
    *,
    pid: int | None = None,
    environ: Mapping[str, str] | None = None,
    proc: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> HostRunIdentity:
    """Observe this process's host identity for its lifecycle row."""
    resolved_pid = os.getpid() if pid is None else pid
    env = os.environ if environ is None else environ
    boot_id: str | None
    try:
        boot_id = (proc / "sys" / "kernel" / "random" / "boot_id").read_text().strip() or None
    except OSError:
        boot_id = None
    cgroup_path = _own_cgroup_path(proc, resolved_pid)
    cgroup_inode: int | None = None
    memory_events: dict[str, int] | None = None
    unit: str | None = None
    if cgroup_path is not None:
        cgroup_dir = cgroup_root / cgroup_path.lstrip("/")
        try:
            cgroup_inode = cgroup_dir.stat().st_ino
        except OSError:
            cgroup_inode = None
        memory_events = read_memory_events(cgroup_dir / "memory.events")
        leaf = cgroup_path.rsplit("/", 1)[-1]
        unit = leaf if leaf.endswith(".service") else None
    return HostRunIdentity(
        pid=resolved_pid,
        boot_id=boot_id,
        invocation_id=_optional_text(env.get("INVOCATION_ID")),
        unit=unit,
        cgroup_path=cgroup_path,
        cgroup_inode=cgroup_inode,
        memory_events=memory_events,
    )


# ---------------------------------------------------------------------------
# Evidence sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """What one source returned for one run, or why it returned nothing.

    ``records`` are the source's own structured records (journal entries,
    counters). ``missing_reason`` is a stable token; a source with a reason
    contributed nothing to the classification.
    """

    name: str
    records: tuple[Mapping[str, object], ...] = ()
    refs: tuple[str, ...] = ()
    missing_reason: str | None = None

    @classmethod
    def missing(cls, name: str, reason: str) -> EvidenceSource:
        return cls(name=name, missing_reason=reason)


class TerminationEvidence(Protocol):
    """Host evidence about one ended run. Production reads the host; tests inject records."""

    def service_manager(self, host: HostRunIdentity) -> EvidenceSource:
        """The service manager's records about the run's invocation (unit result, main-process exit)."""
        ...

    def kernel_oom(self, host: HostRunIdentity, *, since_ms: int, until_ms: int) -> EvidenceSource:
        """Kernel OOM-kill records in the run's boot between the two instants."""
        ...

    def oomd_kills(self, host: HostRunIdentity, *, since_ms: int, until_ms: int) -> EvidenceSource:
        """``systemd-oomd`` kill records between the two instants."""
        ...

    def cgroup_memory_events(self, host: HostRunIdentity) -> EvidenceSource:
        """The recorded cgroup's current ``memory.events`` with its instance inode."""
        ...

    def current_boot_id(self) -> str | None:
        """This boot's id, or ``None`` when the host does not expose one."""
        ...


_KERNEL_KILLED_PID = re.compile(r"Killed process (\d+)")
_KERNEL_OOM_KILL_PID = re.compile(r"oom-kill:.*?[,\s]pid=(\d+)")


def kernel_oom_pid(message: str) -> int | None:
    """Return the pid a kernel OOM-kill record names, or ``None``."""
    match = _KERNEL_KILLED_PID.search(message) or _KERNEL_OOM_KILL_PID.search(message)
    return int(match.group(1)) if match else None


class HostTerminationEvidence:
    """Production evidence adapter over ``journalctl``, ``/proc`` and ``/sys/fs/cgroup``.

    Reads only; it never waits on a deadline of its own. The reconciliation
    that calls it runs as its own daemon service, off the writer and outside
    startup, so a slow journal delays the receipt, not the daemon.
    """

    def __init__(
        self,
        *,
        journalctl: str | None = None,
        proc: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
    ) -> None:
        self._journalctl = journalctl if journalctl is not None else shutil.which("journalctl")
        self._proc = proc
        self._cgroup_root = cgroup_root

    def _journal(self, *args: str) -> tuple[list[dict[str, object]], str | None]:
        if self._journalctl is None:
            return [], "journalctl_unavailable"
        try:
            completed = subprocess.run(
                [self._journalctl, "--no-pager", "--output=json", *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return [], "journalctl_unavailable"
        records: list[dict[str, object]] = []
        for line in completed.stdout.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append(record)
        if completed.returncode != 0 and not records:
            return [], "journal_query_failed"
        return records, None

    def service_manager(self, host: HostRunIdentity) -> EvidenceSource:
        if host.invocation_id is None:
            return EvidenceSource.missing(SOURCE_SERVICE_MANAGER, "run_not_under_service_manager")
        # The manager tags its own messages about a unit with the invocation
        # id: INVOCATION_ID for the system manager, USER_INVOCATION_ID for a
        # user manager. ``+`` is journalctl's disjunction.
        records, reason = self._journal(
            f"INVOCATION_ID={host.invocation_id}", "+", f"USER_INVOCATION_ID={host.invocation_id}"
        )
        if reason is not None:
            return EvidenceSource.missing(SOURCE_SERVICE_MANAGER, reason)
        unit_records = tuple(record for record in records if _is_unit_result_record(record))
        if not unit_records:
            return EvidenceSource.missing(SOURCE_SERVICE_MANAGER, "no_manager_records_for_invocation")
        return EvidenceSource(
            name=SOURCE_SERVICE_MANAGER,
            records=unit_records,
            refs=(f"journal:invocation:{host.invocation_id}",),
        )

    def kernel_oom(self, host: HostRunIdentity, *, since_ms: int, until_ms: int) -> EvidenceSource:
        if host.boot_id is None:
            return EvidenceSource.missing(SOURCE_KERNEL_OOM, "run_boot_id_unrecorded")
        records, reason = self._journal(
            "_TRANSPORT=kernel",
            f"_BOOT_ID={host.boot_id.replace('-', '')}",
            f"--since=@{since_ms // 1000}",
            f"--until=@{until_ms // 1000 + 1}",
        )
        if reason is not None:
            return EvidenceSource.missing(SOURCE_KERNEL_OOM, reason)
        oom_records = tuple(record for record in records if kernel_oom_pid(str(record.get("MESSAGE", ""))) is not None)
        return EvidenceSource(name=SOURCE_KERNEL_OOM, records=oom_records, refs=(f"journal:kernel:{host.boot_id}",))

    def oomd_kills(self, host: HostRunIdentity, *, since_ms: int, until_ms: int) -> EvidenceSource:
        if host.cgroup_path is None:
            return EvidenceSource.missing(SOURCE_OOMD, "run_cgroup_unrecorded")
        records, reason = self._journal(
            "_COMM=systemd-oomd", f"--since=@{since_ms // 1000}", f"--until=@{until_ms // 1000 + 1}"
        )
        if reason is not None:
            return EvidenceSource.missing(SOURCE_OOMD, reason)
        return EvidenceSource(name=SOURCE_OOMD, records=tuple(records), refs=("journal:systemd-oomd",))

    def cgroup_memory_events(self, host: HostRunIdentity) -> EvidenceSource:
        if host.cgroup_path is None or host.cgroup_inode is None or host.memory_events is None:
            return EvidenceSource.missing(SOURCE_CGROUP, "run_cgroup_baseline_unrecorded")
        cgroup_dir = self._cgroup_root / host.cgroup_path.lstrip("/")
        try:
            inode = cgroup_dir.stat().st_ino
        except OSError:
            return EvidenceSource.missing(SOURCE_CGROUP, "cgroup_removed")
        counters = read_memory_events(cgroup_dir / "memory.events")
        if counters is None:
            return EvidenceSource.missing(SOURCE_CGROUP, "memory_events_unreadable")
        return EvidenceSource(
            name=SOURCE_CGROUP,
            records=({"inode": inode, **counters},),
            refs=(f"cgroup:{host.cgroup_path}/memory.events",),
        )

    def current_boot_id(self) -> str | None:
        try:
            return (self._proc / "sys" / "kernel" / "random" / "boot_id").read_text().strip() or None
        except OSError:
            return None


# systemd's catalogued message ids for a unit's main process exiting and for a
# unit entering the failed state; both carry the structured result fields.
_MESSAGE_ID_PROCESS_EXIT = "98e322203f7a4ed290d09fe03c09fe15"
_MESSAGE_ID_UNIT_FAILED = "d9b373ed55a64feb8242e02dbe79a49c"
_MESSAGE_ID_UNIT_RESULT_FIELDS = ("UNIT_RESULT", "EXIT_CODE", "EXIT_STATUS")


def _is_unit_result_record(record: Mapping[str, object]) -> bool:
    message_id = str(record.get("MESSAGE_ID", ""))
    if message_id in (_MESSAGE_ID_PROCESS_EXIT, _MESSAGE_ID_UNIT_FAILED):
        return True
    return any(name in record for name in _MESSAGE_ID_UNIT_RESULT_FIELDS)


# ---------------------------------------------------------------------------
# The receipt and its classifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TerminationReceipt:
    """How one daemon run ended, with the evidence that says so."""

    run_id: str
    classification: TerminationClassification
    reconciled_at_ms: int
    reconciled_by_run_id: str
    started_at_ms: int
    last_good_at_ms: int
    """The last instant the run is known alive: its final heartbeat or stop marker."""
    ended_after_ms: int
    ended_before_ms: int
    """Clock bounds on the end. They come from the run's own rows and the next
    run's start, never from a source record's timestamp."""
    observed: Mapping[str, object] = field(default_factory=dict)
    """Facts a source recorded directly, keyed by source-qualified name."""
    inferred: Mapping[str, object] = field(default_factory=dict)
    """Conclusions drawn from the observed facts, each named as such."""
    evidence_refs: tuple[str, ...] = ()
    missing_sources: Mapping[str, str] = field(default_factory=dict)
    """Source name -> stable reason it contributed nothing."""
    last_workload: Mapping[str, object] | None = None

    @property
    def remediation(self) -> str:
        return _REMEDIATION[self.classification]

    def to_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "classification": self.classification.value,
            "reconciled_at_ms": self.reconciled_at_ms,
            "reconciled_by_run_id": self.reconciled_by_run_id,
            "started_at_ms": self.started_at_ms,
            "last_good_at_ms": self.last_good_at_ms,
            "ended_after_ms": self.ended_after_ms,
            "ended_before_ms": self.ended_before_ms,
            "observed": dict(self.observed),
            "inferred": dict(self.inferred),
            "evidence_refs": list(self.evidence_refs),
            "missing_sources": dict(sorted(self.missing_sources.items())),
            "last_workload": None if self.last_workload is None else dict(self.last_workload),
            "remediation": self.remediation,
        }


def _signal_number(signal_name: str) -> int:
    try:
        return int(signal.Signals[signal_name])
    except KeyError:
        return -1


def _signal_name(exit_status: str) -> str:
    return exit_status if exit_status.startswith("SIG") else f"SIG{exit_status}"


def classify_termination(
    run: ArchiveDaemonLifecycle,
    *,
    next_started_at_ms: int | None,
    reconciled_at_ms: int,
    reconciled_by_run_id: str,
    evidence: TerminationEvidence,
    last_workload: EvidenceSource,
) -> TerminationReceipt:
    """Classify how ``run`` ended from direct evidence only.

    A stop marker is the run's own direct record and decides the class
    without host evidence. A run without one is classified from host records
    tied to it by identity, in this order: the service manager's watchdog or
    OOM result, a kernel OOM kill naming its pid, a group kill of its cgroup
    instance, the manager's record of how its main process exited, the signal
    row it wrote itself, then a changed boot id. With none of these it is
    ``unknown``, whatever else happened near the time.
    """
    ended_before_ms = next_started_at_ms if next_started_at_ms is not None else reconciled_at_ms
    observed: dict[str, object] = {"lifecycle.last_heartbeat_at_ms": run.last_heartbeat_at_ms}
    inferred: dict[str, object] = {}
    refs: list[str] = [f"ops:daemon_lifecycle:{run.run_id}"]
    missing: dict[str, str] = {}
    workload_payload: dict[str, object] | None = None
    if last_workload.missing_reason is not None:
        missing[SOURCE_WORKLOAD] = last_workload.missing_reason
    elif last_workload.records:
        workload_payload = dict(last_workload.records[-1])
        refs.extend(last_workload.refs)

    def receipt(classification: TerminationClassification, *, ended_after_ms: int) -> TerminationReceipt:
        return TerminationReceipt(
            run_id=run.run_id,
            classification=classification,
            reconciled_at_ms=reconciled_at_ms,
            reconciled_by_run_id=reconciled_by_run_id,
            started_at_ms=run.started_at_ms,
            last_good_at_ms=run.last_heartbeat_at_ms,
            ended_after_ms=ended_after_ms,
            ended_before_ms=max(ended_after_ms, ended_before_ms),
            observed=observed,
            inferred=inferred,
            evidence_refs=tuple(dict.fromkeys(refs)),
            missing_sources=missing,
            last_workload=workload_payload,
        )

    if run.signal is not None:
        observed["lifecycle.signal"] = run.signal
    if run.stopped_at_ms is not None:
        observed["lifecycle.stopped_at_ms"] = run.stopped_at_ms
        observed["lifecycle.exit_kind"] = run.exit_kind
        if run.exit_kind == "signal" or run.signal is not None:
            classification = TerminationClassification.HANDLED_SIGNAL
        elif run.exit_kind == "clean":
            classification = TerminationClassification.CLEAN
        else:
            # ``error`` (a fatal exception reached shutdown) and ``atexit`` (the
            # interpreter exited without the orderly stop) are both the run's
            # own record of an abnormal end.
            classification = TerminationClassification.CRASH
        return receipt(classification, ended_after_ms=run.stopped_at_ms)

    host = HostRunIdentity.from_details(run.details)
    if host is None:
        for name in (SOURCE_SERVICE_MANAGER, SOURCE_KERNEL_OOM, SOURCE_OOMD, SOURCE_CGROUP, SOURCE_BOOT):
            missing[name] = "run_host_identity_unrecorded"
        classification = (
            TerminationClassification.HANDLED_SIGNAL if run.signal is not None else TerminationClassification.UNKNOWN
        )
        if run.signal is not None:
            inferred["shutdown"] = "incomplete: the signal row exists but no stop marker followed"
        return receipt(classification, ended_after_ms=run.last_heartbeat_at_ms)

    observed["lifecycle.pid"] = host.pid
    window = {"since_ms": run.started_at_ms, "until_ms": ended_before_ms}

    manager = evidence.service_manager(host)
    unit_result: str | None = None
    exit_code: str | None = None
    exit_status: str | None = None
    if manager.missing_reason is not None:
        missing[SOURCE_SERVICE_MANAGER] = manager.missing_reason
    else:
        refs.extend(manager.refs)
        for record in manager.records:
            unit_result = _optional_text(record.get("UNIT_RESULT")) or unit_result
            exit_code = _optional_text(record.get("EXIT_CODE")) or exit_code
            exit_status = _optional_text(record.get("EXIT_STATUS")) or exit_status
        for name, value in (("unit_result", unit_result), ("exit_code", exit_code), ("exit_status", exit_status)):
            if value is not None:
                observed[f"{SOURCE_SERVICE_MANAGER}.{name}"] = value

    kernel = evidence.kernel_oom(host, **window)
    kernel_hit = False
    if kernel.missing_reason is not None:
        missing[SOURCE_KERNEL_OOM] = kernel.missing_reason
    else:
        named = [kernel_oom_pid(str(record.get("MESSAGE", ""))) for record in kernel.records]
        other_pids = sorted({pid for pid in named if pid is not None and pid != host.pid})
        kernel_hit = host.pid in named
        if kernel_hit:
            observed[f"{SOURCE_KERNEL_OOM}.killed_pid"] = host.pid
            refs.extend(kernel.refs)
        if other_pids:
            # Reported so a reader sees memory pressure existed, never used:
            # another process's OOM kill says nothing about this run.
            observed[f"{SOURCE_KERNEL_OOM}.other_killed_pids"] = other_pids

    oomd = evidence.oomd_kills(host, **window)
    oomd_hit = False
    if oomd.missing_reason is not None:
        missing[SOURCE_OOMD] = oomd.missing_reason
    elif host.cgroup_path is not None:
        oomd_hit = any(host.cgroup_path in str(record.get("MESSAGE", "")) for record in oomd.records)
        if oomd_hit:
            observed[f"{SOURCE_OOMD}.killed_cgroup"] = host.cgroup_path
            refs.extend(oomd.refs)

    cgroup = evidence.cgroup_memory_events(host)
    group_kill_delta = 0
    if cgroup.missing_reason is not None:
        missing[SOURCE_CGROUP] = cgroup.missing_reason
    elif cgroup.records:
        current = cgroup.records[-1]
        if current.get("inode") != host.cgroup_inode:
            # A new cgroup instance restarted its counters; the prior run's
            # final values are gone, so there is nothing to compare.
            missing[SOURCE_CGROUP] = "cgroup_recreated"
        else:
            baseline = host.memory_events or {}
            deltas = {
                counter: int(reading) - int(baseline.get(counter, 0))
                for counter in _MEMORY_EVENT_COUNTERS
                if isinstance((reading := current.get(counter)), int)
            }
            for counter, delta in deltas.items():
                observed[f"{SOURCE_CGROUP}.{counter}_delta"] = delta
            group_kill_delta = deltas.get("oom_group_kill", 0)
            refs.extend(cgroup.refs)

    current_boot = evidence.current_boot_id()
    boot_changed = False
    if host.boot_id is None:
        missing[SOURCE_BOOT] = "run_boot_id_unrecorded"
    elif current_boot is None:
        missing[SOURCE_BOOT] = "current_boot_id_unavailable"
    else:
        boot_changed = current_boot != host.boot_id
        observed[f"{SOURCE_BOOT}.changed"] = boot_changed

    ended_after = run.last_heartbeat_at_ms
    if unit_result == "watchdog":
        return receipt(TerminationClassification.WATCHDOG, ended_after_ms=ended_after)
    if unit_result == "oom-kill" or kernel_hit:
        return receipt(TerminationClassification.OOM_KILL, ended_after_ms=ended_after)
    if oomd_hit or group_kill_delta > 0:
        return receipt(TerminationClassification.CGROUP_KILL, ended_after_ms=ended_after)
    if exit_code in ("killed", "dumped") and exit_status is not None:
        if exit_code == "dumped":
            inferred["cause"] = f"the main process dumped core on {_signal_name(exit_status)}"
            return receipt(TerminationClassification.CRASH, ended_after_ms=ended_after)
        inferred["cause"] = f"the main process was killed by {_signal_name(exit_status)} it did not handle"
        return receipt(TerminationClassification.EXTERNAL_STOP, ended_after_ms=ended_after)
    if unit_result == "timeout":
        inferred["cause"] = "the unit's stop timed out and the manager killed it"
        return receipt(TerminationClassification.EXTERNAL_STOP, ended_after_ms=ended_after)
    if exit_code == "exited" and exit_status is not None:
        if exit_status == "0":
            inferred["stop_marker"] = "missing: the process exited 0 without recording its stop"
            return receipt(TerminationClassification.CLEAN, ended_after_ms=ended_after)
        if run.signal is not None and exit_status == str(128 + _signal_number(run.signal)):
            # The signal handler exits 128+signum: the run handled the signal
            # it recorded and only its stop marker was lost.
            inferred["stop_marker"] = "missing: the run exited through its signal handler"
            return receipt(TerminationClassification.HANDLED_SIGNAL, ended_after_ms=ended_after)
        return receipt(TerminationClassification.CRASH, ended_after_ms=ended_after)
    if run.signal is not None:
        inferred["shutdown"] = "incomplete: the signal row exists but no stop marker followed"
        return receipt(TerminationClassification.HANDLED_SIGNAL, ended_after_ms=ended_after)
    if boot_changed:
        inferred["cause"] = "the host booted again while the run was live"
        return receipt(TerminationClassification.HOST_GAP, ended_after_ms=ended_after)
    return receipt(TerminationClassification.UNKNOWN, ended_after_ms=ended_after)


# ---------------------------------------------------------------------------
# Reconciliation over the ops tier
# ---------------------------------------------------------------------------


def last_workload_evidence(conn: sqlite3.Connection, run_id: str) -> EvidenceSource:
    """The run's last route-observation workload receipt, joined by its run id."""
    row = conn.execute(
        """
        SELECT observation_id, surface, route, phase, started_at_ms, duration_ms, status
        FROM route_observations
        WHERE json_extract(attributes_json, '$.route_receipt.daemon_run_id') = ?
        ORDER BY started_at_ms DESC, observation_id DESC
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return EvidenceSource.missing(SOURCE_WORKLOAD, "no_workload_receipt_carries_run_id")
    return EvidenceSource(
        name=SOURCE_WORKLOAD,
        records=(
            {
                "observation_id": str(row[0]),
                "surface": str(row[1]),
                "route": str(row[2]),
                "phase": str(row[3]),
                "started_at_ms": int(row[4]),
                "duration_ms": int(row[5]),
                "status": str(row[6]),
            },
        ),
        refs=(f"ops:route_observation:{row[0]}",),
    )


def reconcile_ended_runs(
    conn: sqlite3.Connection,
    *,
    current_run_id: str,
    evidence: TerminationEvidence,
    now_ms: int,
) -> tuple[TerminationReceipt, ...]:
    """Classify every earlier run of this archive that has no receipt yet.

    Reads only; :func:`record_termination_receipts` persists the result.
    Every unreconciled run is classified, not just the newest, so a run that
    died before its successor could reconcile it is still accounted for.
    """
    pending: Sequence[UnreconciledDaemonRun] = unreconciled_daemon_runs(conn, current_run_id=current_run_id)
    return tuple(
        classify_termination(
            item.lifecycle,
            next_started_at_ms=item.next_started_at_ms,
            reconciled_at_ms=now_ms,
            reconciled_by_run_id=current_run_id,
            evidence=evidence,
            last_workload=last_workload_evidence(conn, item.lifecycle.run_id),
        )
        for item in pending
    )


def record_termination_receipts(conn: sqlite3.Connection, receipts: Sequence[TerminationReceipt]) -> int:
    """Persist ``receipts`` once each and return how many this call wrote."""
    written = 0
    for item in receipts:
        written += int(
            record_daemon_termination_receipt(
                conn,
                run_id=item.run_id,
                classification=item.classification.value,
                reconciled_at_ms=item.reconciled_at_ms,
                reconciled_by_run_id=item.reconciled_by_run_id,
                receipt=item.to_payload(),
            )
        )
    return written


def termination_status(conn: sqlite3.Connection, *, current_run_id: str) -> dict[str, object]:
    """The status projection: the newest receipt and how many runs await one."""
    latest = latest_daemon_termination_receipt(conn)
    pending = unreconciled_daemon_runs(conn, current_run_id=current_run_id)
    return {
        "last_termination": None if latest is None else latest.receipt,
        "pending_reconciliation_runs": [item.lifecycle.run_id for item in pending],
    }


__all__ = [
    "EvidenceSource",
    "HostRunIdentity",
    "HostTerminationEvidence",
    "TerminationClassification",
    "TerminationEvidence",
    "TerminationReceipt",
    "capture_host_run_identity",
    "classify_termination",
    "kernel_oom_pid",
    "last_workload_evidence",
    "read_memory_events",
    "reconcile_ended_runs",
    "record_termination_receipts",
    "termination_status",
]

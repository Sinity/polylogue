"""Share one pytest pool's memory between the jobs that start in it.

Every pytest slot sizes its xdist width from what its cgroup has left when its
workers start. A cgroup reading only counts memory already charged, so two
jobs that start within a second of each other both read the whole headroom
and both take the full width: that is how the quick pool (eight slots under
one 9 GiB ceiling) reached sustained pressure and systemd-oomd killed its
units (polylogue-h9da0). Dividing the ceiling by the slot count instead would
give each job 1,152 MiB, less than one focused xdist worker, so the pool
would do less work than it can hold.

This module keeps a reservation ledger instead: one small file per admitted
slot under ``$XDG_RUNTIME_DIR/polylogue-pytest-admission``, written under an
exclusive ``flock``. Sizing subtracts, at every limited cgroup level, what
other admitted jobs under that level have reserved and not yet charged
(``max(0, reserved - their cgroup's current use)``). A job's reservation is
the peak charge its chosen width was predicted at, so it stops counting once
the job has actually allocated it.

When the budget left holds no worker at all the slot waits, first come first
served, while another admitted job holds memory it will release; the wait is
bounded by those jobs' progress, not by a timeout. When nothing else in the
ledger holds memory the shortfall cannot be waited out, and the slot returns
the typed, retryable ``resource_not_ready`` outcome without launching pytest
(polylogue-o5rjp).

A reservation belongs to a process, identified by pid and kernel start time,
so a slot that died without releasing is dropped by the next reader rather
than holding memory forever.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

from devtools.worker_memory import ChargeProfile, OutstandingReservations, cgroup_usage_mib

__all__ = [
    "ADMISSION_DIR_ENV",
    "EX_TEMPFAIL",
    "RESOURCE_NOT_READY",
    "AdmissionLedger",
    "Reservation",
    "admission_ledger",
    "admit_width",
    "admission_not_ready",
]

#: Where the ledger lives when set; otherwise under ``XDG_RUNTIME_DIR``.
ADMISSION_DIR_ENV: Final = "POLYLOGUE_PYTEST_ADMISSION_DIR"
LEDGER_DIR_NAME: Final = "polylogue-pytest-admission"
#: The typed deferral when the pool cannot hold one worker and nothing will free memory.
RESOURCE_NOT_READY: Final = "resource_not_ready"
#: sysexits.h EX_TEMPFAIL: retryable, not a pytest failure.
EX_TEMPFAIL: Final = 75
#: How often a waiting slot looks at the ledger again.
WAIT_POLL_S: Final = 2.0
#: How often a waiting slot says it is still waiting.
WAIT_REPORT_INTERVAL_S: Final = 300.0

State = Literal["waiting", "admitted"]
Sizer = Callable[..., tuple[list[str], dict[str, Any] | None]]


@dataclass(frozen=True)
class Reservation:
    """One slot's claim on its pool, as the ledger file records it."""

    pid: int
    start_ticks: int
    cgroup: str
    reserved_mib: float
    state: State
    ticket: int


def _process_start_ticks(pid: int, *, proc: Path) -> int | None:
    """The kernel start time of ``pid`` (``/proc/<pid>/stat`` field 22), or None if it is gone."""
    try:
        text = (proc / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # The command name may contain spaces and parentheses; fields resume after the last ')'.
    fields = text.rpartition(")")[2].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def _parent_pid(pid: int, *, proc: Path) -> int | None:
    """The parent of ``pid`` (``/proc/<pid>/stat`` field 4), or None if it is gone."""
    try:
        text = (proc / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields = text.rpartition(")")[2].split()
    try:
        return int(fields[1])
    except (IndexError, ValueError):
        return None


class AdmissionLedger:
    """The reservation files of one pool, read and written under one lock."""

    def __init__(self, directory: Path, *, proc: Path = Path("/proc"), pid: int | None = None) -> None:
        self.directory = directory
        self.proc = proc
        self.pid = os.getpid() if pid is None else pid

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(self.directory / "lock", "a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _path(self, pid: int) -> Path:
        return self.directory / f"{pid}.json"

    def enclosing_pids(self) -> frozenset[int]:
        """Every ancestor of this process: the runs it executes inside."""
        ancestors: set[int] = set()
        current = _parent_pid(self.pid, proc=self.proc)
        while current is not None and current > 1 and current not in ancestors:
            ancestors.add(current)
            current = _parent_pid(current, proc=self.proc)
        return frozenset(ancestors)

    def live_reservations(self) -> list[Reservation]:
        """Every reservation whose process still runs; the rest are removed. Caller holds the lock."""
        live: list[Reservation] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                reservation = Reservation(**json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                path.unlink(missing_ok=True)
                continue
            if _process_start_ticks(reservation.pid, proc=self.proc) != reservation.start_ticks:
                path.unlink(missing_ok=True)
                continue
            live.append(reservation)
        return live

    def record(self, *, cgroup: str, reserved_mib: float, state: State, ticket: int) -> None:
        """Write this process's reservation. Caller holds the lock."""
        start = _process_start_ticks(self.pid, proc=self.proc)
        if start is None:
            return
        reservation = Reservation(self.pid, start, cgroup, round(float(reserved_mib), 1), state, ticket)
        temporary = self._path(self.pid).with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(reservation)), encoding="utf-8")
        temporary.replace(self._path(self.pid))

    def discard(self) -> None:
        """Remove this process's reservation. Caller holds the lock."""
        self._path(self.pid).unlink(missing_ok=True)

    def release(self) -> None:
        """Give this process's reservation back to the pool."""
        with contextlib.suppress(OSError), self.locked():
            self.discard()


def admission_ledger(env: Mapping[str, str]) -> AdmissionLedger | None:
    """The ledger this slot shares with its pool, or None when no location is known."""
    configured = env.get(ADMISSION_DIR_ENV)
    if configured:
        return AdmissionLedger(Path(configured))
    runtime = env.get("XDG_RUNTIME_DIR")
    if not runtime:
        return None
    return AdmissionLedger(Path(runtime) / LEDGER_DIR_NAME)


def _outstanding(reservations: Sequence[Reservation]) -> OutstandingReservations:
    """What admitted reservations under one level have not yet charged to it."""

    def under(level: Path) -> float:
        total = 0.0
        for reservation in reservations:
            if reservation.state != "admitted":
                continue
            cgroup = Path(reservation.cgroup)
            if not cgroup.is_relative_to(level):
                continue
            total += max(0.0, reservation.reserved_mib - cgroup_usage_mib(cgroup))
        return total

    return under


def admission_not_ready(sizing: Mapping[str, Any] | None) -> bool:
    """True when sizing forbids launching even one worker."""
    return sizing is not None and sizing.get("admission") == RESOURCE_NOT_READY


def admit_width(
    argv: Sequence[str],
    *,
    size: Sizer,
    profile: ChargeProfile,
    max_workers: int | None,
    ledger: AdmissionLedger | None,
    report: Callable[[str], None],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[str], dict[str, Any] | None]:
    """Size ``argv`` against the pool's memory net of other jobs' reservations.

    Returns the command and its sizing basis. The basis carries
    ``admission_ledger`` -- how long this slot waited, how many jobs held
    reservations, and what they had not yet charged -- and ``admission``,
    which is ``resource_not_ready`` when the slot must not launch. An admitted
    slot holds its reservation until :meth:`AdmissionLedger.release`.
    """
    if ledger is None:
        return size(list(argv), profile=profile, max_workers=max_workers)
    started = clock()
    ticket: int | None = None
    reported_at: float | None = None
    # A run nested inside an admitted slot (a test that runs ``devtools test``)
    # charges the enclosing run's cgroup, which that run's reservation already
    # covers. It runs within that reservation: waiting on the pool -- the
    # enclosing reservation included -- would wait for memory only the
    # enclosing run's completion can release.
    enclosing = ledger.enclosing_pids()
    with ledger.locked():
        within_enclosing = any(
            item.pid in enclosing and item.state == "admitted" for item in ledger.live_reservations()
        )
    if within_enclosing:
        command, sizing = size(list(argv), profile=profile, max_workers=max_workers)
        if sizing is not None:
            sizing["admission_ledger"] = {
                "waited_s": 0.0,
                "holders": 0,
                "waiting_ahead": 0,
                "reserved_by_other_jobs_mib": 0.0,
                "within_enclosing_reservation": True,
            }
        return command, sizing
    while True:
        with ledger.locked():
            others = [
                item for item in ledger.live_reservations() if item.pid != ledger.pid and item.pid not in enclosing
            ]
            if ticket is None:
                ticket = 1 + max((item.ticket for item in others), default=0)
            command, sizing = size(
                list(argv), profile=profile, max_workers=max_workers, outstanding_mib=_outstanding(others)
            )
            cgroup = sizing.get("cgroup_directory") if sizing is not None else None
            if sizing is None or not isinstance(cgroup, str):
                # Nothing a reservation could be charged against: no cgroup,
                # or an argv the sizer does not understand.
                ledger.discard()
                return command, sizing
            limiting = tuple(Path(path) for path in sizing.get("limiting_cgroups", ["/"]))

            def relevant(item: Reservation, *, limiting: tuple[Path, ...] = limiting) -> bool:
                cgroup = Path(item.cgroup)
                return any(cgroup.is_relative_to(level) for level in limiting)

            holders = [item for item in others if item.state == "admitted" and relevant(item)]
            earlier = [item for item in others if item.state == "waiting" and item.ticket < ticket and relevant(item)]
            unreleased = sum(max(0.0, item.reserved_mib - cgroup_usage_mib(Path(item.cgroup))) for item in holders)
            sizing["admission_ledger"] = {
                "waited_s": round(clock() - started, 1),
                "holders": len(holders),
                "waiting_ahead": len(earlier),
                "reserved_by_other_jobs_mib": round(unreleased, 1),
            }
            if not admission_not_ready(sizing) and not earlier:
                ledger.record(
                    cgroup=cgroup, reserved_mib=float(sizing["predicted_charge_mib"]), state="admitted", ticket=ticket
                )
                return command, sizing
            if not holders and not earlier:
                # Nothing in the pool will free memory this slot can count on.
                ledger.discard()
                return command, sizing
            ledger.record(cgroup=cgroup, reserved_mib=0.0, state="waiting", ticket=ticket)
        now = clock()
        if reported_at is None or now - reported_at >= WAIT_REPORT_INTERVAL_S:
            reported_at = now
            report(
                f"pytest slot: waiting for pool memory: {len(holders)} admitted job(s) hold "
                f"{round(unreleased)} MiB not yet charged, {len(earlier)} slot(s) waiting ahead; "
                f"{round(now - started)} s so far."
            )
        sleep(WAIT_POLL_S)
